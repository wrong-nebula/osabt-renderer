#!/usr/bin/env python3
"""OS-ABT Seeing Space — Local Development Viewer & Optical Instrumentation Server.

Serves the interactive Seeing Space workbench (See Inside, See Across Time,
See Across Possibilities) backed directly by live .f32img renders, G-buffers,
epipolar x-t slices, oracle_report.json, and .agent/state.json on disk.

Zero GPU contention: read-only disk inspection with mtime-aware caching so the
user can follow along live as builds and oracle runs progress.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import os
import struct
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WB_DIR = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
RUN_DIR = ART_DIR / "oracle_run"
REPORT_PATH = ART_DIR / "oracle_report.json"
STATE_PATH = ROOT / ".agent" / "state.json"
NOTES_PATH = ART_DIR / "workbench_notes.jsonl"
SCENES_DIR = ROOT / "oracle_scenes"

SCENES_META = {
    "a": {
        "id": "a",
        "code": "SCENE_A",
        "title": "Scene A — Orthogonal Bleed & Thin Parapet",
        "sensor": "S4_orthogonal_bleed",
        "subtitle": "Curved pillar (sid=2) + 8cm backlit thin parapet (sid=3) vs sunlit courtyard floor",
    },
    "b": {
        "id": "b",
        "code": "SCENE_B",
        "title": "Scene B — Grazing Micro-Geometry Aliasing",
        "sensor": "S3_grazing_aliasing",
        "subtitle": "Irrational micro-grooves at 82°–87° grazing across 16-frame sub-pixel drift",
    },
    "c": {
        "id": "c",
        "code": "SCENE_C",
        "title": "Scene C — Dirac Impulse & Whip-Pan Disocclusion",
        "sensor": "S5_temporal_shock",
        "subtitle": "120-frame timeline: f40 6.5x Dirac flash, f60–90 34° whip-pan, f75 chromatic step",
    },
    "d": {
        "id": "d",
        "code": "SCENE_D",
        "title": "Scene D — White-Furnace & Thin-Shell Reciprocity",
        "sensor": "S1_white_furnace",
        "subtitle": "Unit spherical furnace: exact Stokes thin glass + Kulla-Conty rough conductor",
    },
}


class LRUCache:
    def __init__(self, max_items: int = 96):
        self.max_items = max_items
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: Any, val: Any) -> None:
        with self._lock:
            self._data[key] = val
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


F32_CACHE = LRUCache(max_items=64)
PNG_CACHE = LRUCache(max_items=160)
NOTES_LOCK = threading.Lock()


def read_f32img_cached(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    arr = F32_CACHE.get(key)
    if arr is not None:
        return arr
    with open(path, "rb") as f:
        hdr = f.read(16)
        if len(hdr) < 16:
            return None
        magic, w, h, c = struct.unpack("<4sIII", hdr)
        if magic != b"OSAB":
            return None
        raw = np.frombuffer(f.read(), dtype=np.float32)
    expected = int(w) * int(h) * int(c)
    if raw.size != expected:
        return None
    arr = raw.reshape((h, w, c)).copy()
    F32_CACHE.put(key, arr)
    return arr


def aces_tonemap(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def linear_to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 1e-8), 1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def inferno_lut(t: np.ndarray) -> np.ndarray:
    """Perceptually uniform dark-to-incandescent scientific error ramp (0..1 -> RGB u8)."""
    t = np.clip(t, 0.0, 1.0)
    stops = np.array([
        [0.00, 0.03, 0.04, 0.07],
        [0.20, 0.16, 0.06, 0.32],
        [0.45, 0.58, 0.14, 0.34],
        [0.70, 0.90, 0.38, 0.12],
        [0.88, 0.98, 0.72, 0.18],
        [1.00, 0.99, 0.98, 0.76],
    ], dtype=np.float32)
    xs = stops[:, 0]
    r = np.interp(t, xs, stops[:, 1])
    g = np.interp(t, xs, stops[:, 2])
    b = np.interp(t, xs, stops[:, 3])
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def diverging_lut(s: np.ndarray) -> np.ndarray:
    """Signed error map (-1..+1): cyan (-) <- obsidian (0) -> coral/amber (+)."""
    s = np.clip(s, -1.0, 1.0)
    pos = np.maximum(s, 0.0)
    neg = np.maximum(-s, 0.0)
    r = 0.05 + 0.95 * np.power(pos, 0.75) + 0.08 * neg
    g = 0.06 + 0.35 * np.power(pos, 1.1) + 0.78 * np.power(neg, 0.8)
    b = 0.08 + 0.18 * pos + 0.92 * np.power(neg, 0.75)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def sid_palette(sid: np.ndarray) -> np.ndarray:
    s = np.round(sid).astype(np.int32)
    lut = np.array([
        [14, 18, 26],     # 0 sky
        [68, 128, 148],   # 1 floor
        [242, 166, 48],   # 2 pillar / displaced
        [56, 217, 192],   # 3 parapet / thin glass
        [232, 96, 108],   # 4 alcove / back wall
        [142, 118, 186],  # 5 ceiling
        [120, 200, 110],  # 6+
    ], dtype=np.uint8)
    idx = np.clip(s, 0, len(lut) - 1)
    return lut[idx]


def frame_path(scene: str, mode: str, frame: int) -> Path:
    return RUN_DIR / f"scene_{scene}" / mode / f"frame_{frame:04d}.f32img"


def gbuf_path(scene: str, kind: str, frame: int = 0) -> Path:
    return RUN_DIR / f"scene_{scene}" / "gt" / f"gbuf_{kind}_{frame:04d}.f32img"


def epipolar_path(scene: str, mode: str) -> Path:
    return RUN_DIR / f"scene_{scene}" / mode / "epipolar.f32img"


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def encode_png(rgb_u8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def scan_artifacts() -> dict[str, Any]:
    scenes: dict[str, Any] = {}
    latest_mtime = 0
    for sc, meta in SCENES_META.items():
        sc_dir = RUN_DIR / f"scene_{sc}"
        modes_info: dict[str, Any] = {}
        all_frames: set[int] = set()
        if sc_dir.exists():
            for mode in ("osabt", "baseline", "ablated", "gt"):
                mdir = sc_dir / mode
                if not mdir.exists():
                    continue
                frames = []
                for p in sorted(mdir.glob("frame_*.f32img")):
                    try:
                        fid = int(p.stem.split("_")[1])
                        frames.append(fid)
                        all_frames.add(fid)
                        latest_mtime = max(latest_mtime, p.stat().st_mtime_ns)
                    except (IndexError, ValueError):
                        pass
                tel = None
                tpath = mdir / "telemetry.json"
                if tpath.exists():
                    try:
                        tel = json.loads(tpath.read_text())
                        latest_mtime = max(latest_mtime, tpath.stat().st_mtime_ns)
                    except Exception:
                        tel = None
                modes_info[mode] = {
                    "frames": frames,
                    "has_epipolar": (mdir / "epipolar.f32img").exists(),
                    "has_gbuf": (mdir / "gbuf_norm_0000.f32img").exists(),
                    "telemetry": tel,
                }
        scenes[sc] = {
            **meta,
            "frames": sorted(all_frames),
            "modes": modes_info,
            "has_gbuf": (sc_dir / "gt" / "gbuf_norm_0000.f32img").exists(),
            "has_epipolar": (sc_dir / "osabt" / "epipolar.f32img").exists(),
        }

    report = None
    report_mtime = 0
    if REPORT_PATH.exists():
        try:
            report_mtime = REPORT_PATH.stat().st_mtime_ns
            latest_mtime = max(latest_mtime, report_mtime)
            report = json.loads(REPORT_PATH.read_text())
        except Exception:
            report = None

    state = None
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            latest_mtime = max(latest_mtime, STATE_PATH.stat().st_mtime_ns)
        except Exception:
            state = None

    # Check binary build timestamps so the viewer reflects fresh compiles immediately
    builds = {}
    for name in ("osabt_cpu", "osabt_metal"):
        bp = ROOT / "build" / name
        if bp.exists():
            st = bp.stat()
            builds[name] = {"mtime": st.st_mtime, "size": st.st_size}
            latest_mtime = max(latest_mtime, st.st_mtime_ns)
        else:
            builds[name] = None

    # Check live/smoke previews if present
    extras = []
    for sub in ("metal_smoke", "cpu_smoke", "live"):
        p = ART_DIR / sub / "frame_0000.f32img"
        if p.exists():
            st = p.stat()
            latest_mtime = max(latest_mtime, st.st_mtime_ns)
            extras.append({"name": sub, "mtime": st.st_mtime})

    return {
        "server_time": time.time(),
        "version_ns": latest_mtime,
        "report_mtime": report_mtime / 1e9 if report_mtime else 0,
        "oracle_report": report,
        "agent_state": state,
        "builds": builds,
        "scenes": scenes,
        "extras": extras,
        "notes_count": count_notes(),
    }


def count_notes() -> int:
    if not NOTES_PATH.exists():
        return 0
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def read_notes() -> list[dict[str, Any]]:
    if not NOTES_PATH.exists():
        return []
    out = []
    with NOTES_LOCK:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    pass
    return out


def append_note(entry: dict[str, Any]) -> dict[str, Any]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": str(entry.get("verdict", "N")).upper()[:1],
        "scene": str(entry.get("scene", "a")),
        "mode": str(entry.get("mode", "osabt")),
        "frame": int(entry.get("frame", 0)),
        "x": entry.get("x"),
        "y": entry.get("y"),
        "note": str(entry.get("note", ""))[:400],
        "metrics": entry.get("metrics"),
    }
    with NOTES_LOCK:
        with open(NOTES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return record


def render_image_png(params: dict[str, str]) -> bytes | None:
    scene = params.get("scene", "a")
    mode = params.get("mode", "osabt")
    frame = int(params.get("frame", "0"))
    view = params.get("view", "rgb")
    ref_mode = params.get("ref", "gt")
    exposure = float(params.get("exposure", "0.0"))
    amp = float(params.get("amp", "8.0"))
    tonemap = params.get("tonemap", "aces")
    scale = math.pow(2.0, exposure)

    if view == "epipolar":
        p = epipolar_path(scene, mode)
        if not p.exists():
            return None
        st = p.stat()
        ckey = ("epi", str(p), st.st_mtime_ns, exposure, tonemap)
        cached = PNG_CACHE.get(ckey)
        if cached is not None:
            return cached
        arr = read_f32img_cached(p)
        if arr is None:
            return None
        rgb = arr[..., :3] * scale
        mapped = aces_tonemap(rgb) if tonemap == "aces" else np.clip(rgb, 0.0, 1.0)
        # Upscale vertically by 2x for crisp readability (120 rows -> 240 rows)
        u8 = linear_to_srgb_u8(mapped)
        u8 = np.repeat(u8, 2, axis=0)
        png = encode_png(u8)
        PNG_CACHE.put(ckey, png)
        return png

    if view in ("norm", "sid", "cos"):
        kind = "aux" if view == "cos" else "norm"
        p = gbuf_path(scene, kind, 0)
        if not p.exists():
            return None
        st = p.stat()
        ckey = (view, str(p), st.st_mtime_ns)
        cached = PNG_CACHE.get(ckey)
        if cached is not None:
            return cached
        arr = read_f32img_cached(p)
        if arr is None:
            return None
        if view == "norm":
            u8 = np.clip(np.round((arr[..., :3] * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        elif view == "sid":
            u8 = sid_palette(arr[..., 3])
        else:
            cos_t = np.clip(arr[..., 0], 0.0, 1.0)
            u8 = inferno_lut(1.0 - cos_t)
            # Highlight exact 82-87 deg grazing ROI in spectral cyan
            c_lo, c_hi = math.cos(math.radians(87.0)), math.cos(math.radians(82.0))
            roi = (cos_t >= c_lo) & (cos_t <= c_hi)
            u8[roi] = np.array([56, 217, 192], dtype=np.uint8)
        png = encode_png(u8)
        PNG_CACHE.put(ckey, png)
        return png

    p = frame_path(scene, mode, frame)
    if not p.exists():
        return None
    st = p.stat()
    rp = frame_path(scene, ref_mode, frame) if view in ("err", "serr") else None
    r_mtime = rp.stat().st_mtime_ns if (rp and rp.exists()) else 0
    ckey = (view, str(p), st.st_mtime_ns, str(rp), r_mtime, round(exposure, 2), round(amp, 2), tonemap)
    cached = PNG_CACHE.get(ckey)
    if cached is not None:
        return cached

    arr = read_f32img_cached(p)
    if arr is None:
        return None
    rgb = arr[..., :3]

    if view == "rgb":
        img = rgb * scale
        mapped = aces_tonemap(img) if tonemap == "aces" else np.clip(img, 0.0, 1.0)
        u8 = linear_to_srgb_u8(mapped)
    elif view == "luma":
        lv = np.log1p(np.maximum(luma(rgb) * scale, 0.0)) / math.log(1.0 + 4.0)
        u8 = inferno_lut(lv)
    elif view == "err":
        if rp is None or not rp.exists():
            return None
        ref = read_f32img_cached(rp)
        if ref is None:
            return None
        diff = np.mean(np.abs(rgb - ref[..., :3]), axis=-1)
        u8 = inferno_lut(np.clip(diff * amp, 0.0, 1.0))
    elif view == "serr":
        if rp is None or not rp.exists():
            return None
        ref = read_f32img_cached(rp)
        if ref is None:
            return None
        dl = luma(rgb) - luma(ref[..., :3])
        u8 = diverging_lut(dl * amp)
    else:
        return None

    png = encode_png(u8)
    PNG_CACHE.put(ckey, png)
    return png


def compute_probe(scene: str, frame: int, x: int, y: int) -> dict[str, Any]:
    out: dict[str, Any] = {"scene": scene, "frame": frame, "x": x, "y": y, "modes": {}}
    gt_arr = read_f32img_cached(frame_path(scene, "gt", frame))
    gt_rgb = None
    H = W = 0
    if gt_arr is not None:
        H, W = gt_arr.shape[:2]
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        out["x"], out["y"], out["width"], out["height"] = x, y, W, H
        gt_rgb = gt_arr[y, x, :3].astype(float)

    for mode in ("gt", "osabt", "baseline", "ablated"):
        arr = read_f32img_cached(frame_path(scene, mode, frame))
        if arr is None:
            continue
        if W == 0:
            H, W = arr.shape[:2]
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            out["x"], out["y"], out["width"], out["height"] = x, y, W, H
        px = arr[y, x, :3].astype(float)
        y0, y1 = max(0, y - 4), min(H, y + 5)
        x0, x1 = max(0, x - 4), min(W, x + 5)
        patch = arr[y0:y1, x0:x1, :3]
        pl = luma(patch)
        rec: dict[str, Any] = {
            "rgb": [round(float(v), 5) for v in px],
            "luma": round(float(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]), 5),
            "patch9_mean": round(float(np.mean(pl)), 5),
            "patch9_std": round(float(np.std(pl)), 5),
        }
        if gt_rgb is not None and mode != "gt":
            d = np.abs(px - gt_rgb)
            rec["abs_err"] = round(float(np.mean(d)), 6)
            rec["err_rgb"] = [round(float(px[c] - gt_rgb[c]), 5) for c in range(3)]
        out["modes"][mode] = rec

    # Attach G-buffer optical & Stiefel neighborhood diagnostics if available
    gn = read_f32img_cached(gbuf_path(scene, "norm", 0))
    if gn is not None and 0 <= y < gn.shape[0] and 0 <= x < gn.shape[1]:
        np_v = gn[y, x, :3].astype(float)
        sid_p = float(gn[y, x, 3])
        neighbors = {}
        for label, (dx, dy) in (("left", (-1, 0)), ("right", (1, 0)), ("up", (0, -1)), ("down", (0, 1))):
            qx, qy = x + dx, y + dy
            if 0 <= qx < gn.shape[1] and 0 <= qy < gn.shape[0]:
                nq = gn[qy, qx, :3].astype(float)
                sid_q = float(gn[qy, qx, 3])
                ndot = float(np.dot(np_v, nq))
                t = max(0.0, min(1.0, (ndot - 0.20) * 1.25))
                w_stiefel = t * t * (3.0 - 2.0 * t) if abs(sid_p - sid_q) <= 0.25 else 0.0
                neighbors[label] = {
                    "ndot": round(ndot, 4),
                    "sid": int(round(sid_q)),
                    "stiefel_w": round(w_stiefel, 4),
                }
        out["gbuf"] = {
            "normal": [round(float(v), 4) for v in np_v],
            "sid": int(round(sid_p)),
            "neighbors": neighbors,
        }
    ga = read_f32img_cached(gbuf_path(scene, "aux", 0))
    if ga is not None and 0 <= y < ga.shape[0] and 0 <= x < ga.shape[1]:
        cos_t = float(ga[y, x, 0])
        out.setdefault("gbuf", {})["cos_theta"] = round(cos_t, 5)
        out["gbuf"]["angle_deg"] = round(math.degrees(math.acos(max(-1.0, min(1.0, cos_t)))), 2)
    return out


def compute_scanline(scene: str, frame: int, y: int, step: int = 2) -> dict[str, Any]:
    res: dict[str, Any] = {"scene": scene, "frame": frame, "y": y, "step": step, "series": {}}
    gt = read_f32img_cached(frame_path(scene, "gt", frame))
    W = 0
    gt_l = None
    if gt is not None:
        H, W = gt.shape[:2]
        y = max(0, min(H - 1, y))
        res["y"], res["width"] = y, W
        gt_l = luma(gt[y, ::step, :3])
        res["series"]["gt"] = [round(float(v), 4) for v in gt_l]
    for mode in ("osabt", "baseline", "ablated"):
        arr = read_f32img_cached(frame_path(scene, mode, frame))
        if arr is None:
            continue
        if W == 0:
            H, W = arr.shape[:2]
            y = max(0, min(H - 1, y))
            res["y"], res["width"] = y, W
        lv = luma(arr[y, ::step, :3])
        res["series"][mode] = [round(float(v), 4) for v in lv]
        if gt is not None:
            err = np.mean(np.abs(arr[y, ::step, :3] - gt[y, ::step, :3]), axis=-1)
            res["series"][f"{mode}_err"] = [round(float(v), 5) for v in err]
    gn = read_f32img_cached(gbuf_path(scene, "norm", 0))
    if gn is not None and y < gn.shape[0]:
        res["sid_row"] = [int(round(float(v))) for v in gn[y, ::step, 3]]
    return res


def compute_timeline(scene: str) -> dict[str, Any]:
    sc_dir = RUN_DIR / f"scene_{scene}"
    gt_dir = sc_dir / "gt"
    frames = []
    if gt_dir.exists():
        for p in sorted(gt_dir.glob("frame_*.f32img")):
            try:
                frames.append(int(p.stem.split("_")[1]))
            except Exception:
                pass
    series: dict[str, list[dict[str, Any]]] = {"osabt": [], "baseline": [], "ablated": []}
    for f in frames:
        gt = read_f32img_cached(frame_path(scene, "gt", f))
        if gt is None:
            continue
        gt_rgb = gt[..., :3]
        for mode in ("osabt", "baseline", "ablated"):
            arr = read_f32img_cached(frame_path(scene, mode, f))
            if arr is None:
                continue
            err = arr[..., :3] - gt_rgb
            rmse = float(np.sqrt(np.mean(err * err)))
            max_e = float(np.max(np.abs(err)))
            mean_l = float(np.mean(luma(arr[..., :3])))
            series[mode].append({
                "frame": f,
                "rmse": round(rmse, 5),
                "max_err": round(max_e, 5),
                "mean_luma": round(mean_l, 4),
            })
    events = []
    if scene == "c":
        events = [
            {"frame": 40, "label": "6.5× Dirac Impulse", "color": "#f87171"},
            {"frame": 60, "label": "Whip-Pan Start (34°)", "color": "#f5a623"},
            {"frame": 75, "label": "Chromatic Step", "color": "#38d9c0"},
            {"frame": 90, "label": "Whip-Pan End", "color": "#f5a623"},
        ]
    elif scene == "b":
        events = [{"frame": 0, "label": "Irrational Sub-Pixel Drift (0..15)", "color": "#38d9c0"}]
    return {"scene": scene, "frames": frames, "series": series, "events": events}


class WorkbenchHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep stdout clean unless debugging
        pass

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Any, status: int = 200) -> None:
        self._send_bytes(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        if path == "/api/status":
            self._send_json(scan_artifacts())
            return

        if path == "/api/image":
            png = render_image_png(qs)
            if png is None:
                self._send_json({"error": "Image not available"}, status=404)
                return
            self._send_bytes(png, "image/png")
            return

        if path == "/api/probe":
            scene = qs.get("scene", "a")
            frame = int(qs.get("frame", "0"))
            x = int(qs.get("x", "640"))
            y = int(qs.get("y", "360"))
            self._send_json(compute_probe(scene, frame, x, y))
            return

        if path == "/api/profile":
            scene = qs.get("scene", "a")
            frame = int(qs.get("frame", "0"))
            y = int(qs.get("y", "360"))
            step = max(1, min(8, int(qs.get("step", "2"))))
            self._send_json(compute_scanline(scene, frame, y, step))
            return

        if path == "/api/timeline":
            scene = qs.get("scene", "c")
            self._send_json(compute_timeline(scene))
            return

        if path == "/api/notes":
            self._send_json({"notes": read_notes()})
            return

        if path == "/api/notes/export":
            raw = NOTES_PATH.read_bytes() if NOTES_PATH.exists() else b""
            self._send_bytes(
                raw,
                "application/x-ndjson; charset=utf-8",
                extra_headers={"Content-Disposition": 'attachment; filename="osabt_triage_notes.jsonl"'},
            )
            return

        # Static assets from workbench/
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        if ".." in rel or rel.startswith("/"):
            self._send_json({"error": "Forbidden"}, status=403)
            return
        fp = (WB_DIR / rel).resolve()
        if not str(fp).startswith(str(WB_DIR.resolve())) or not fp.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return
        ctype, _ = mimetypes.guess_type(str(fp))
        if fp.suffix == ".wgsl":
            ctype = "text/plain; charset=utf-8"
        self._send_bytes(fp.read_bytes(), ctype or "application/octet-stream")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/notes":
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n) if n > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=400)
                return
            rec = append_note(payload)
            self._send_json({"ok": True, "record": rec}, status=201)
            return
        self._send_json({"error": "Not found"}, status=404)


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    ap = argparse.ArgumentParser(description="OS-ABT Seeing Space Workbench Server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()

    srv = ReusableHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"[OS-ABT Workbench] Listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
