#!/usr/bin/env python3
"""Frozen Adversarial Verification Oracle (SHA-256 locked in ORACLE.sha256).
Executes deterministic Tier-1 computational sensors across Scenes A-D:
  S1: White-Furnace Energy Conservation & Thin-Shell Reciprocity (Scene D)
  S2: Deterministic CPU-Reference <-> GPU-Shader Parity (Scenes A-D)
  S3: Non-Divisor Grazing Aliasing & Shimmer Suppression >= 18 dB (Scene B)
  S4: Orthogonal Normal-Bleed Rejection across |n_a.n_b| < 0.15 (Scene A)
  S5: Dirac Impulse & Disocclusion Whip-Pan Temporal Stability (Scene C, 120f)
  S6: Equal-Time Ablation & Apple M1 <= 4.0 ms/frame 720p Budget Gate
"""
import argparse, json, math, os, struct, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CPU_BIN = ROOT / "build" / "osabt_cpu"
GPU_BIN = ROOT / "build" / "osabt_metal"
OUT_ROOT = ROOT / "artifacts" / "oracle_run"
SCENES_DIR = ROOT / "oracle_scenes"

KEY_FRAMES_C = [30, 40, 41, 42, 44, 68, 75, 82, 115]

def read_f32img(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        hdr = f.read(16)
        magic, w, h, c = struct.unpack("<4sIII", hdr)
        if magic != b"OSAB":
            raise ValueError(f"Bad magic in {path}: {magic!r}")
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape((h, w, c))

def run_cmd(binary: Path, scene: str, mode: str, w: int, h: int, frames: int,
            out_dir: Path, key_frames=None, dump_gbuf=False, dump_epipolar=False,
            warmup=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary), "--scene", scene, "--mode", mode,
        "--width", str(w), "--height", str(h), "--frames", str(frames),
        "--out-dir", str(out_dir), "--warmup", str(warmup)
    ]
    if key_frames is not None:
        cmd += ["--key-frames", ",".join(str(k) for k in key_frames)]
    if dump_gbuf:
        cmd.append("--dump-gbuf")
    if dump_epipolar:
        cmd.append("--dump-epipolar")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    with open(out_dir / "telemetry.json") as f:
        return json.load(f)

def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

def check_finite(arr: np.ndarray):
    return int(np.sum(~np.isfinite(arr)))

def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-16:
        return 120.0
    peak = max(1.0, float(np.max(np.abs(b))))
    return 10.0 * math.log10((peak * peak) / mse)

def sensor1_white_furnace():
    """Scene D @ 1280x720: unit white-furnace + thin glass + rough conductor + micro-grooves."""
    modes = {}
    for mode in ("osabt", "ablated", "baseline", "gt"):
        d = OUT_ROOT / "scene_d" / mode
        tel = run_cmd(GPU_BIN, "d", mode, 1280, 720, 1, d, warmup=8)
        img = read_f32img(d / "frame_0000.f32img")[..., :3]
        nan_cnt = check_finite(img)
        neg_cnt = int(np.sum(img < -1e-6)) if nan_cnt == 0 else -1
        mean_L = float(np.mean(img)) if nan_cnt == 0 else float("nan")
        max_dev = float(np.max(np.abs(img - 1.0))) if nan_cnt == 0 else float("nan")
        rmse = float(np.sqrt(np.mean((img - 1.0) ** 2))) if nan_cnt == 0 else float("nan")
        modes[mode] = {
            "nan_inf": nan_cnt, "neg_pixels": neg_cnt,
            "mean_L": round(mean_L, 6), "abs_mean_err": round(abs(mean_L - 1.0), 6),
            "max_abs_dev": round(max_dev, 6), "rmse_vs_1": round(rmse, 6),
            "gpu_ms_median": round(tel["gpu_ms_median"], 4)
        }
    o, a, b, g = modes["osabt"], modes["ablated"], modes["baseline"], modes["gt"]
    passed = (
        o["nan_inf"] == 0 and o["neg_pixels"] == 0 and
        o["abs_mean_err"] <= 0.003 and o["max_abs_dev"] <= 0.015 and
        g["abs_mean_err"] <= 0.005 and
        o["rmse_vs_1"] < a["rmse_vs_1"] and o["rmse_vs_1"] < b["rmse_vs_1"]
    )
    return {"pass": bool(passed), "modes": modes}

def sensor2_cpu_gpu_parity():
    """Scenes A, B, C (frames 0 & 40), D @ 320x180: deterministic CPU <-> Metal parity."""
    cases = [
        ("a", "osabt", 1, [0]),
        ("b", "osabt", 1, [0]),
        ("c", "osabt", 42, [0, 40, 41]),
        ("d", "osabt", 1, [0]),
    ]
    results = {}
    all_ok = True
    for sc, mode, nf, kfs in cases:
        dc = OUT_ROOT / "parity" / f"cpu_{sc}"
        dg = OUT_ROOT / "parity" / f"gpu_{sc}"
        run_cmd(CPU_BIN, sc, mode, 320, 180, nf, dc, key_frames=kfs, warmup=0)
        run_cmd(GPU_BIN, sc, mode, 320, 180, nf, dg, key_frames=kfs, warmup=1)
        for kf in kfs:
            ic = read_f32img(dc / f"frame_{kf:04d}.f32img")[..., :3]
            ig = read_f32img(dg / f"frame_{kf:04d}.f32img")[..., :3]
            n_bad = check_finite(ic) + check_finite(ig)
            max_err = float(np.max(np.abs(ic - ig))) if n_bad == 0 else float("inf")
            psnr = compute_psnr(ic, ig) if n_bad == 0 else 0.0
            ok = (n_bad == 0 and psnr >= 60.0 and max_err <= 5e-3)
            all_ok = all_ok and ok
            results[f"scene_{sc}_f{kf}"] = {
                "pass": bool(ok), "nan_inf": n_bad,
                "psnr_db": round(psnr, 2), "max_abs_err": round(max_err, 6)
            }
    return {"pass": bool(all_ok), "cases": results}

def sensor3_grazing_aliasing():
    """Scene B @ 1280x720 (16-frame irrational drift): >=18 dB SNR in 82-87 deg grazing band."""
    n_drift = 16
    all_f = list(range(n_drift))
    tels = {}
    for mode in ("gt", "osabt", "ablated", "baseline"):
        d = OUT_ROOT / "scene_b" / mode
        tels[mode] = run_cmd(
            GPU_BIN, "b", mode, 1280, 720, n_drift, d,
            key_frames=all_f, dump_gbuf=(mode == "gt"), warmup=6
        )
    # Build exact 82-87 deg grazing ROI mask from analytical G-buffer (cos_theta in [cos 87, cos 82])
    aux0 = read_f32img(OUT_ROOT / "scene_b" / "gt" / "gbuf_aux_0000.f32img")
    norm0 = read_f32img(OUT_ROOT / "scene_b" / "gt" / "gbuf_norm_0000.f32img")
    cos_t = aux0[..., 0]
    surf_id = norm0[..., 3]
    c_lo, c_hi = math.cos(math.radians(87.0)), math.cos(math.radians(82.0))
    roi = (surf_id > 0.5) & (surf_id < 1.5) & (cos_t >= c_lo) & (cos_t <= c_hi)
    roi_px = int(np.sum(roi))
    if roi_px < 1000:
        raise RuntimeError(f"Scene B grazing ROI too small: {roi_px} px")

    gt_seq = np.stack([read_f32img(OUT_ROOT / "scene_b" / "gt" / f"frame_{f:04d}.f32img")[..., :3] for f in all_f])
    stats = {}
    for mode in ("osabt", "ablated", "baseline"):
        seq = np.stack([read_f32img(OUT_ROOT / "scene_b" / mode / f"frame_{f:04d}.f32img")[..., :3] for f in all_f])
        nan_cnt = check_finite(seq)
        err = (seq - gt_seq)[:, roi, :]
        mse_sp = float(np.mean(err ** 2))
        # Temporal shimmer: frame-to-frame difference error vs GT frame-to-frame difference
        dt_err = (np.diff(seq, axis=0) - np.diff(gt_seq, axis=0))[:, roi, :]
        mse_tm = float(np.mean(dt_err ** 2))
        stats[mode] = {
            "nan_inf": nan_cnt, "spatial_mse": mse_sp, "spatial_rmse": round(math.sqrt(mse_sp), 6),
            "shimmer_mse": mse_tm, "shimmer_rmse": round(math.sqrt(mse_tm), 6),
            "gpu_ms_median": round(tels[mode]["gpu_ms_median"], 4)
        }
    spatial_snr_db = 10.0 * math.log10((stats["ablated"]["spatial_mse"] + 1e-12) / (stats["osabt"]["spatial_mse"] + 1e-12))
    shimmer_snr_db = 10.0 * math.log10((stats["ablated"]["shimmer_mse"] + 1e-12) / (stats["osabt"]["shimmer_mse"] + 1e-12))
    passed = (
        stats["osabt"]["nan_inf"] == 0 and
        spatial_snr_db >= 18.0 and shimmer_snr_db >= 18.0 and
        stats["osabt"]["spatial_rmse"] < stats["baseline"]["spatial_rmse"] and
        stats["osabt"]["shimmer_rmse"] < stats["baseline"]["shimmer_rmse"]
    )
    return {
        "pass": bool(passed), "roi_pixels": roi_px,
        "spatial_snr_db_vs_ablated": round(spatial_snr_db, 2),
        "shimmer_snr_db_vs_ablated": round(shimmer_snr_db, 2),
        "modes": stats
    }

def sensor4_orthogonal_bleed():
    """Scene A @ 1280x720: cross-surface bleed rejection across |n_a.n_b| < 0.15 junctions."""
    tels = {}
    for mode in ("gt", "osabt", "ablated", "baseline"):
        d = OUT_ROOT / "scene_a" / mode
        tels[mode] = run_cmd(
            GPU_BIN, "a", mode, 1280, 720, 1, d,
            dump_gbuf=(mode == "gt"), warmup=8
        )
    gbuf = read_f32img(OUT_ROOT / "scene_a" / "gt" / "gbuf_norm_0000.f32img")
    nrm = gbuf[..., :3]
    sid = np.round(gbuf[..., 3]).astype(np.int32)
    # Detect junction pixels on curved pillar (sid==2) or thin parapet front (sid==3)
    # within 3 px of any orthogonal normal neighbor (|n_p . n_q| < 0.15) or different surface
    target_surf = (sid == 2) | (sid == 3)
    ortho_mask = np.zeros(sid.shape, dtype=bool)
    H, W = sid.shape
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            y0, y1 = max(0, -dy), min(H, H - dy)
            x0, x1 = max(0, -dx), min(W, W - dx)
            dot = np.sum(nrm[y0:y1, x0:x1] * nrm[y0+dy:y1+dy, x0+dx:x1+dx], axis=-1)
            diff_surf = (sid[y0:y1, x0:x1] != sid[y0+dy:y1+dy, x0+dx:x1+dx])
            ortho_mask[y0:y1, x0:x1] |= (np.abs(dot) < 0.15) & diff_surf
    junc = target_surf & ortho_mask
    # Also include thin-parapet interior pixels (sid==3) to catch backlit alcove bleed-through!
    parapet = (sid == 3)
    junc_px = int(np.sum(junc))
    para_px = int(np.sum(parapet))
    if junc_px < 200 or para_px < 500:
        raise RuntimeError(f"Scene A junction/parapet mask too small: junc={junc_px}, para={para_px}")

    gt = read_f32img(OUT_ROOT / "scene_a" / "gt" / "frame_0000.f32img")[..., :3]
    stats = {}
    for mode in ("osabt", "ablated", "baseline"):
        img = read_f32img(OUT_ROOT / "scene_a" / mode / "frame_0000.f32img")[..., :3]
        nan_cnt = check_finite(img)
        diff_rgb = np.abs(img - gt)
        diff_j = diff_rgb[junc]
        diff_p = diff_rgb[parapet]
        stats[mode] = {
            "nan_inf": nan_cnt,
            "junction_mean_err": round(float(np.mean(diff_j)), 6),
            "junction_p95_err": round(float(np.percentile(diff_j, 95)), 6),
            "parapet_mean_err": round(float(np.mean(diff_p)), 6),
            "full_rmse": round(float(np.sqrt(np.mean(diff_rgb ** 2))), 6),
            "gpu_ms_median": round(tels[mode]["gpu_ms_median"], 4)
        }
    o, a, b = stats["osabt"], stats["ablated"], stats["baseline"]
    passed = (
        o["nan_inf"] == 0 and
        o["junction_mean_err"] <= 0.008 and o["junction_p95_err"] <= 0.025 and
        o["parapet_mean_err"] <= 0.008 and
        o["junction_mean_err"] < a["junction_mean_err"] * 0.30 and
        o["junction_mean_err"] < b["junction_mean_err"] * 0.40 and
        o["parapet_mean_err"] < a["parapet_mean_err"] * 0.30 and
        o["parapet_mean_err"] < b["parapet_mean_err"] * 0.40
    )
    return {
        "pass": bool(passed), "junction_pixels": junc_px, "parapet_pixels": para_px,
        "modes": stats
    }

def block_splotch_variance(err_luma: np.ndarray, bs: int = 8) -> float:
    H, W = err_luma.shape
    h8, w8 = (H // bs) * bs, (W // bs) * bs
    blocks = err_luma[:h8, :w8].reshape(h8 // bs, bs, w8 // bs, bs).mean(axis=(1, 3))
    return float(np.var(blocks))

def sensor5_temporal_shock():
    """Scene C @ 1280x720 (120 frames): Dirac shock, whip-pan disocclusion, zero splotches/ringing."""
    tels = {}
    for mode in ("gt", "osabt", "ablated", "baseline"):
        d = OUT_ROOT / "scene_c" / mode
        tels[mode] = run_cmd(
            GPU_BIN, "c", mode, 1280, 720, 120, d,
            key_frames=KEY_FRAMES_C, dump_epipolar=(mode != "gt"), warmup=4
        )
    gt_k = {kf: read_f32img(OUT_ROOT / "scene_c" / "gt" / f"frame_{kf:04d}.f32img")[..., :3] for kf in KEY_FRAMES_C}
    stats = {}
    for mode in ("osabt", "ablated", "baseline"):
        d = OUT_ROOT / "scene_c" / mode
        epi = read_f32img(d / "epipolar.f32img")[..., :3]
        nan_cnt = check_finite(epi)
        per_f = {}
        for kf in KEY_FRAMES_C:
            im = read_f32img(d / f"frame_{kf:04d}.f32img")[..., :3]
            nan_cnt += check_finite(im)
            err = im - gt_k[kf]
            per_f[kf] = {
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "splotch_var": block_splotch_variance(luma(err))
            }
        post_dirac_max = max(per_f[k]["rmse"] for k in (41, 42, 44))
        whip_rmse = float(np.mean([per_f[k]["rmse"] for k in (68, 75, 82)]))
        steady_rmse = float(np.mean([per_f[k]["rmse"] for k in (30, 115)]))
        splotch_max = max(per_f[k]["splotch_var"] for k in (30, 68, 82, 115))
        epi_max = float(np.max(luma(epi))) if nan_cnt == 0 else float("inf")
        stats[mode] = {
            "nan_inf": nan_cnt,
            "dirac_f40_rmse": round(per_f[40]["rmse"], 5),
            "post_dirac_max_rmse": round(post_dirac_max, 5),
            "whip_disocc_rmse": round(whip_rmse, 5),
            "steady_rmse": round(steady_rmse, 5),
            "max_splotch_var": round(splotch_max, 6),
            "epipolar_max_luma": round(epi_max, 4),
            "gpu_ms_median": round(tels[mode]["gpu_ms_median"], 4)
        }
    o, a, b = stats["osabt"], stats["ablated"], stats["baseline"]
    passed = (
        o["nan_inf"] == 0 and o["epipolar_max_luma"] < 25.0 and
        o["dirac_f40_rmse"] <= 0.15 and
        o["dirac_f40_rmse"] < a["dirac_f40_rmse"] * 0.30 and
        o["dirac_f40_rmse"] < b["dirac_f40_rmse"] * 0.30 and
        o["post_dirac_max_rmse"] <= 0.06 and
        o["whip_disocc_rmse"] <= 0.08 and
        o["whip_disocc_rmse"] < a["whip_disocc_rmse"] and
        o["whip_disocc_rmse"] < b["whip_disocc_rmse"] and
        o["steady_rmse"] <= 0.045 and
        o["steady_rmse"] < a["steady_rmse"] and
        o["steady_rmse"] < b["steady_rmse"] and
        o["max_splotch_var"] <= 0.0020 and
        o["max_splotch_var"] < a["max_splotch_var"] and
        o["max_splotch_var"] < b["max_splotch_var"]
    )
    return {"pass": bool(passed), "modes": stats}

def sensor6_budget_and_ablation(s1, s3, s4, s5):
    """Equal-time ablation & Apple M1 <= 4.0 ms/frame 720p budget gate."""
    osabt_ms = {
        "scene_a": s4["modes"]["osabt"]["gpu_ms_median"],
        "scene_b": s3["modes"]["osabt"]["gpu_ms_median"],
        "scene_c": s5["modes"]["osabt"]["gpu_ms_median"],
        "scene_d": s1["modes"]["osabt"]["gpu_ms_median"],
    }
    base_ms = {
        "scene_a": s4["modes"]["baseline"]["gpu_ms_median"],
        "scene_b": s3["modes"]["baseline"]["gpu_ms_median"],
        "scene_c": s5["modes"]["baseline"]["gpu_ms_median"],
        "scene_d": s1["modes"]["baseline"]["gpu_ms_median"],
    }
    abl_ms = {
        "scene_a": s4["modes"]["ablated"]["gpu_ms_median"],
        "scene_b": s3["modes"]["ablated"]["gpu_ms_median"],
        "scene_c": s5["modes"]["ablated"]["gpu_ms_median"],
        "scene_d": s1["modes"]["ablated"]["gpu_ms_median"],
    }
    max_osabt = max(osabt_ms.values())
    avg_osabt = sum(osabt_ms.values()) / 4.0
    avg_base = sum(base_ms.values()) / 4.0
    equal_time_ok = (max_osabt <= 4.0) and (avg_osabt <= avg_base * 1.05)
    beats_all = bool(s1["pass"] and s3["pass"] and s4["pass"] and s5["pass"])
    return {
        "pass": bool(equal_time_ok and beats_all),
        "max_osabt_gpu_ms": round(max_osabt, 4),
        "avg_osabt_gpu_ms": round(avg_osabt, 4),
        "avg_baseline_gpu_ms": round(avg_base, 4),
        "per_scene_gpu_ms": {"osabt": osabt_ms, "baseline": base_ms, "ablated": abl_ms}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "artifacts" / "oracle_report.json"))
    args = parser.parse_args()
    for b in (CPU_BIN, GPU_BIN):
        if not b.exists():
            print(f"ERROR: missing binary {b}", file=sys.stderr)
            sys.exit(2)
    s1 = sensor1_white_furnace()
    s2 = sensor2_cpu_gpu_parity()
    s3 = sensor3_grazing_aliasing()
    s4 = sensor4_orthogonal_bleed()
    s5 = sensor5_temporal_shock()
    s6 = sensor6_budget_and_ablation(s1, s3, s4, s5)
    overall = all(x["pass"] for x in (s1, s2, s3, s4, s5, s6))
    report = {
        "overall_pass": bool(overall),
        "S1_white_furnace": s1,
        "S2_cpu_gpu_parity": s2,
        "S3_grazing_aliasing": s3,
        "S4_orthogonal_bleed": s4,
        "S5_temporal_shock": s5,
        "S6_m1_budget_ablation": s6
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    summary = {
        "overall_pass": overall,
        "S1_white_furnace": "PASS" if s1["pass"] else "FAIL",
        "S2_cpu_gpu_parity": "PASS" if s2["pass"] else "FAIL",
        "S3_grazing_aliasing": "PASS" if s3["pass"] else "FAIL",
        "S4_orthogonal_bleed": "PASS" if s4["pass"] else "FAIL",
        "S5_temporal_shock": "PASS" if s5["pass"] else "FAIL",
        "S6_m1_budget_ablation": "PASS" if s6["pass"] else "FAIL",
        "max_osabt_gpu_ms": s6["max_osabt_gpu_ms"],
        "avg_osabt_gpu_ms": s6["avg_osabt_gpu_ms"],
        "avg_baseline_gpu_ms": s6["avg_baseline_gpu_ms"],
        "S3_spatial_snr_db": s3["spatial_snr_db_vs_ablated"],
        "S3_shimmer_snr_db": s3["shimmer_snr_db_vs_ablated"],
        "S4_osabt_junction_bleed": s4["modes"]["osabt"]["junction_mean_err"],
        "S5_osabt_dirac_rmse": s5["modes"]["osabt"]["dirac_f40_rmse"],
        "S1_osabt_wf_max_dev": s1["modes"]["osabt"]["max_abs_dev"]
    }
    print(json.dumps(summary, indent=2))
    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()
