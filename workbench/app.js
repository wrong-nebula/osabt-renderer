(() => {
  "use strict";

  const state = {
    status: null,
    versionNs: 0,
    autoPoll: true,
    scene: "a",
    layout: "wipe", // wipe | quad | single
    modeL: "osabt",
    modeR: "gt",
    view: "rgb",
    tonemap: "aces",
    exposure: 0.0,
    amp: 10,
    zoom: 1,
    panX: 0,
    panY: 0,
    wipe: 0.5,
    frame: 0,
    playing: false,
    playTimer: null,
    probeX: 640,
    probeY: 360,
    pinned: true,
    scopeMode: "luma", // luma | err
    epiMode: "osabt",
    geoMode: "geodesic",
    scanlineData: null,
    timelineData: null,
    probeData: null,
    notes: [],
  };

  const $ = (id) => document.getElementById(id);

  function showToast(msg) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.classList.remove("show"), 2000);
  }

  function fmtTimeAgo(ts) {
    if (!ts) return "—";
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  }

  function imgUrl(scene, mode, frame, view = state.view) {
    const q = new URLSearchParams({
      scene,
      mode,
      frame: String(frame),
      view,
      ref: "gt",
      exposure: String(state.exposure),
      amp: String(state.amp),
      tonemap: state.tonemap,
      v: String(state.versionNs),
    });
    return `/api/image?${q.toString()}`;
  }

  // =========================================================================
  // 1. Live Artifact Polling & Oracle S1-S6 Sensor Ribbon
  // =========================================================================
  async function fetchStatus(forceReloadImages = false) {
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const changed = data.version_ns !== state.versionNs;
      state.status = data;
      if (changed) {
        state.versionNs = data.version_ns;
      }
      renderTopbar();
      renderSensorRibbon();
      renderSceneDetail();
      syncSceneFrames();

      if (changed || forceReloadImages) {
        updateViewportImages();
        fetchProbe();
        fetchScanline();
        if (state.scene === "b" || state.scene === "c") {
          fetchTimeline();
        }
        updateEpipolarImage();
        if (changed && !forceReloadImages) {
          showToast("Artifacts updated on disk — view refreshed");
        }
      }
      $("liveDot").classList.remove("stale");
      $("liveText").textContent = state.autoPoll ? "LIVE · 2s" : "PAUSED";
    } catch (err) {
      $("liveDot").classList.add("stale");
      $("liveText").textContent = "OFFLINE";
    }
  }

  function renderTopbar() {
    const rep = state.status?.oracle_report;
    const badge = $("overallBadge");
    if (!rep) {
      badge.textContent = "ORACLE: NO REPORT";
      badge.className = "overall-badge fail";
    } else {
      const sensors = [
        rep.S1_white_furnace,
        rep.S2_cpu_gpu_parity,
        rep.S3_grazing_aliasing,
        rep.S4_orthogonal_bleed,
        rep.S5_temporal_shock,
        rep.S6_m1_budget_ablation,
      ];
      const passed = sensors.filter((s) => s && s.pass).length;
      badge.textContent = rep.overall_pass
        ? `ORACLE: PASS (${passed}/6)`
        : `ORACLE: ${passed}/6 PASS`;
      badge.className = `overall-badge ${rep.overall_pass ? "pass" : "fail"}`;
    }

    const b = state.status?.builds || {};
    const metalAge = b.osabt_metal ? fmtTimeAgo(b.osabt_metal.mtime) : "missing";
    const cpuAge = b.osabt_cpu ? fmtTimeAgo(b.osabt_cpu.mtime) : "missing";
    const slice = state.status?.agent_state?.current_slice ?? "?";
    $("buildMeta").textContent = `slice ${slice} · metal ${metalAge} · cpu ${cpuAge}`;
    $("notesBadge").textContent = String(state.status?.notes_count ?? state.notes.length);
  }

  function renderSensorRibbon() {
    const rep = state.status?.oracle_report;
    const el = $("sensorRibbon");
    if (!rep) {
      el.innerHTML = `<div class="sensor-card"><div class="sensor-metric">Waiting for oracle_report.json…</div></div>`;
      return;
    }

    const s1 = rep.S1_white_furnace;
    const s2 = rep.S2_cpu_gpu_parity;
    const s3 = rep.S3_grazing_aliasing;
    const s4 = rep.S4_orthogonal_bleed;
    const s5 = rep.S5_temporal_shock;
    const s6 = rep.S6_m1_budget_ablation;

    const minPsnr = s2?.cases
      ? Math.min(...Object.values(s2.cases).map((c) => c.psnr_db ?? 0))
      : 0;
    const maxParErr = s2?.cases
      ? Math.max(...Object.values(s2.cases).map((c) => c.max_abs_err ?? 0))
      : 0;
    const s4Target = s4?.modes?.baseline
      ? Math.min(0.008, s4.modes.baseline.junction_mean_err * 0.4)
      : 0.008;

    const cards = [
      {
        id: "S1 · FURNACE",
        pass: s1?.pass,
        metric: `dev ${s1?.modes?.osabt?.max_abs_dev?.toFixed(5) ?? "—"}`,
        target: `≤ 0.015 · ${s1?.modes?.osabt?.gpu_ms_median?.toFixed(2) ?? "—"} ms`,
        scene: "d",
        frame: 0,
      },
      {
        id: "S2 · CPU↔GPU",
        pass: s2?.pass,
        metric: `${minPsnr.toFixed(1)} dB`,
        target: `≥ 60 dB · err ${maxParErr.toFixed(5)}`,
        scene: "a",
        frame: 0,
      },
      {
        id: "S3 · GRAZING",
        pass: s3?.pass,
        metric: `${s3?.spatial_snr_db_vs_ablated?.toFixed(1) ?? "—"} / ${s3?.shimmer_snr_db_vs_ablated?.toFixed(1) ?? "—"} dB`,
        target: `≥ 18 dB · ${s3?.modes?.osabt?.gpu_ms_median?.toFixed(2) ?? "—"} ms`,
        scene: "b",
        frame: 0,
      },
      {
        id: "S4 · BLEED",
        pass: s4?.pass,
        metric: `junc ${s4?.modes?.osabt?.junction_mean_err?.toFixed(5) ?? "—"}`,
        target: `< ${s4Target.toFixed(5)} · para ${s4?.modes?.osabt?.parapet_mean_err?.toFixed(5) ?? "—"}`,
        scene: "a",
        frame: 0,
      },
      {
        id: "S5 · SHOCK",
        pass: s5?.pass,
        metric: `f40 ${s5?.modes?.osabt?.dirac_f40_rmse?.toFixed(4) ?? "—"}`,
        target: `≤ 0.15 · post ${s5?.modes?.osabt?.post_dirac_max_rmse?.toFixed(3) ?? "—"} · whip ${s5?.modes?.osabt?.whip_disocc_rmse?.toFixed(3) ?? "—"}`,
        scene: "c",
        frame: 40,
      },
      {
        id: "S6 · M1 ≤4ms",
        pass: s6?.pass,
        metric: `max ${s6?.max_osabt_gpu_ms?.toFixed(2) ?? "—"} ms`,
        target: `≤ 4.00 ms · avg ${s6?.avg_osabt_gpu_ms?.toFixed(2) ?? "—"} vs ${s6?.avg_baseline_gpu_ms?.toFixed(2) ?? "—"}`,
        scene: "c",
        frame: 40,
      },
    ];

    el.innerHTML = cards
      .map(
        (c, idx) => `
      <div class="sensor-card ${c.pass ? "pass" : "fail"} ${state.scene === c.scene && idx !== 1 && idx !== 5 ? "selected" : ""}"
           data-jump-scene="${c.scene}" data-jump-frame="${c.frame}">
        <div class="sensor-head">
          <span class="sensor-id">${c.id}</span>
          <span class="sensor-tag ${c.pass ? "pass" : "fail"}">${c.pass ? "PASS" : "FAIL"}</span>
        </div>
        <div class="sensor-metric">${c.metric}</div>
        <div class="sensor-target">${c.target}</div>
      </div>`
      )
      .join("");

    el.querySelectorAll(".sensor-card").forEach((card) => {
      card.addEventListener("click", () => {
        const sc = card.dataset.jumpScene;
        const fr = parseInt(card.dataset.jumpFrame, 10) || 0;
        selectScene(sc, fr);
      });
    });
  }

  function renderSceneDetail() {
    const rep = state.status?.oracle_report;
    const box = $("sceneDetailBox");
    const scMeta = state.status?.scenes?.[state.scene];
    const tel = scMeta?.modes?.osabt?.telemetry;
    $("sceneGpuLbl").textContent = tel ? `${tel.gpu_ms_median.toFixed(2)} ms med` : "— ms";
    if (!rep) {
      box.textContent = "No oracle report on disk.";
      return;
    }

    const lines = [];
    if (state.scene === "a" && rep.S4_orthogonal_bleed) {
      const m = rep.S4_orthogonal_bleed.modes;
      const o = m.osabt, b = m.baseline, a = m.ablated;
      lines.push(`Junc Err : OS-ABT ${o.junction_mean_err.toFixed(6)} | Base ${b.junction_mean_err.toFixed(6)} | Abl ${a.junction_mean_err.toFixed(6)}`);
      lines.push(`Ratio    : OS/Base = ${(o.junction_mean_err / Math.max(b.junction_mean_err, 1e-9)).toFixed(3)} (gate < 0.400)`);
      lines.push(`Junc P95 : OS-ABT ${o.junction_p95_err.toFixed(6)} (gate ≤ 0.025)`);
      lines.push(`Parapet  : OS-ABT ${o.parapet_mean_err.toFixed(6)} | Base ${b.parapet_mean_err.toFixed(6)} (gate ≤ 0.008)`);
      lines.push(`M1 GPU   : OS ${o.gpu_ms_median.toFixed(2)}ms | Base ${b.gpu_ms_median.toFixed(2)}ms | Abl ${a.gpu_ms_median.toFixed(2)}ms`);
    } else if (state.scene === "b" && rep.S3_grazing_aliasing) {
      const s = rep.S3_grazing_aliasing;
      const o = s.modes.osabt, b = s.modes.baseline, a = s.modes.ablated;
      lines.push(`Spatial SNR : ${s.spatial_snr_db_vs_ablated.toFixed(2)} dB (gate ≥ 18.0 dB)`);
      lines.push(`Shimmer SNR : ${s.shimmer_snr_db_vs_ablated.toFixed(2)} dB (gate ≥ 18.0 dB)`);
      lines.push(`Spatial RMSE: OS ${o.spatial_rmse.toFixed(5)} | Base ${b.spatial_rmse.toFixed(5)} | Abl ${a.spatial_rmse.toFixed(5)}`);
      lines.push(`ROI Pixels  : ${s.roi_pixels.toLocaleString()} px (82°–87° band)`);
      lines.push(`M1 GPU      : OS ${o.gpu_ms_median.toFixed(2)}ms | Base ${b.gpu_ms_median.toFixed(2)}ms`);
    } else if (state.scene === "c" && rep.S5_temporal_shock) {
      const m = rep.S5_temporal_shock.modes;
      const o = m.osabt, b = m.baseline, a = m.ablated;
      lines.push(`Dirac f40  : OS ${o.dirac_f40_rmse.toFixed(4)} (≤0.15) | Base ${b.dirac_f40_rmse.toFixed(4)} | Abl ${a.dirac_f40_rmse.toFixed(4)}`);
      lines.push(`Post-Dirac : OS ${o.post_dirac_max_rmse.toFixed(4)} (≤0.06) | Base ${b.post_dirac_max_rmse.toFixed(4)}`);
      lines.push(`Whip Disocc: OS ${o.whip_disocc_rmse.toFixed(4)} (≤0.08) | Base ${b.whip_disocc_rmse.toFixed(4)}`);
      lines.push(`Steady RMSE: OS ${o.steady_rmse.toFixed(4)} (≤0.045) | Splotch ${o.max_splotch_var.toFixed(6)} (≤0.002)`);
      lines.push(`EpipolarMax: ${o.epipolar_max_luma.toFixed(2)} (<25.0) | M1 GPU: ${o.gpu_ms_median.toFixed(2)}ms`);
    } else if (state.scene === "d" && rep.S1_white_furnace) {
      const m = rep.S1_white_furnace.modes;
      const o = m.osabt, b = m.baseline;
      lines.push(`Mean L     : OS ${o.mean_L.toFixed(6)} | Base ${b.mean_L.toFixed(6)}`);
      lines.push(`Max |L-1|  : OS ${o.max_abs_dev.toFixed(6)} (≤0.015) | Base ${b.max_abs_dev.toFixed(6)}`);
      lines.push(`NaN/Neg    : ${o.nan_inf} / ${o.neg_pixels} | M1 GPU: ${o.gpu_ms_median.toFixed(2)}ms`);
    }
    if (tel && (tel.pass1_ms || tel.pass2_ms)) {
      lines.push(`Passes     : Pass1 ${(tel.pass1_ms || 0).toFixed(2)}ms · Pass2 ${(tel.pass2_ms || 0).toFixed(2)}ms · p95 ${(tel.gpu_ms_p95 || 0).toFixed(2)}ms`);
    }
    box.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
  }

  // =========================================================================
  // 2. Scene, Frame, Layout, Split-Wipe, Pan/Zoom & Viewport Rendering
  // =========================================================================
  function getSceneFrames(sc = state.scene) {
    const fs = state.status?.scenes?.[sc]?.frames;
    return fs && fs.length ? fs : [0];
  }

  function syncSceneFrames() {
    const frames = getSceneFrames();
    if (!frames.includes(state.frame)) {
      state.frame = state.scene === "c" && frames.includes(40) ? 40 : frames[0];
    }
    const pills = $("kfPills");
    const isMulti = frames.length > 1;
    $("btnPlay").style.display = isMulti ? "inline-block" : "none";
    $("timelineBox").style.display = isMulti ? "block" : "none";
    $("stripBody").classList.toggle("temporal-split", isMulti);

    pills.innerHTML = frames
      .map((f) => {
        let cls = "kf-pill";
        if (f === state.frame) cls += " active";
        if (state.scene === "c" && f === 40) cls += " event-dirac";
        if (state.scene === "c" && (f === 68 || f === 75 || f === 82)) cls += " event-whip";
        return `<button class="${cls}" data-f="${f}">f${f}</button>`;
      })
      .join("");
    pills.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => setFrame(parseInt(btn.dataset.f, 10)));
    });
  }

  function selectScene(sc, targetFrame = null) {
    stopPlayback();
    state.scene = sc;
    document.querySelectorAll("#sceneSeg button").forEach((b) => {
      b.classList.toggle("active", b.dataset.scene === sc);
    });
    const frames = getSceneFrames(sc);
    if (targetFrame !== null && frames.includes(targetFrame)) {
      state.frame = targetFrame;
    } else {
      state.frame = sc === "c" && frames.includes(40) ? 40 : frames[0];
    }
    // Reset sensible default probe coordinates per scene
    if (sc === "a") { state.probeX = 560; state.probeY = 470; }
    else if (sc === "b") { state.probeX = 640; state.probeY = 440; }
    else if (sc === "c") { state.probeX = 640; state.probeY = 360; }
    else { state.probeX = 640; state.probeY = 360; }

    syncSceneFrames();
    renderSensorRibbon();
    renderSceneDetail();
    updateViewportImages();
    updateCrosshairDOM();
    fetchProbe();
    fetchScanline();
    if (sc === "b" || sc === "c") {
      fetchTimeline();
    } else {
      state.timelineData = null;
    }
    updateEpipolarLine();
  }

  function setFrame(f) {
    state.frame = f;
    syncSceneFrames();
    updateViewportImages();
    fetchProbe();
    fetchScanline();
    drawTimeline();
    updateEpipolarLine();
  }

  function updateViewportImages() {
    const modeNames = {
      osabt: "OS-ABT",
      gt: "GT (Ref)",
      baseline: "Baseline",
      ablated: "Ablated",
    };
    const viewNames = {
      rgb: "RGB",
      err: `|Err| ×${state.amp}`,
      serr: `Signed ×${state.amp}`,
      luma: "Log Luma",
      norm: "Normals",
      sid: "Surf ID",
      cos: "Grazing θ",
    };

    const isQuad = state.layout === "quad";
    const isSingle = state.layout === "single";
    $("quadGrid").style.display = isQuad ? "grid" : "none";
    $("layerLeft").style.display = isQuad ? "none" : "block";
    $("layerRight").style.display = isQuad || isSingle ? "none" : "block";
    $("wipeDivider").style.display = state.layout === "wipe" ? "block" : "none";
    $("badgeLeft").style.display = isQuad ? "none" : "block";
    $("badgeRight").style.display = state.layout === "wipe" ? "block" : "none";

    $("hudScene").textContent = `Scene ${state.scene.toUpperCase()} · frame ${state.frame} · ${viewNames[state.view]}`;
    $("badgeLeft").textContent = `${modeNames[state.modeL]} · ${viewNames[state.view]}`;
    $("badgeRight").textContent = `${modeNames[state.modeR]} · ${viewNames[state.view]}`;

    if (isQuad) {
      ["gt", "osabt", "baseline", "ablated"].forEach((m) => {
        const v = m === "gt" && (state.view === "err" || state.view === "serr") ? "rgb" : state.view;
        $(`imgQ_${m}`).src = imgUrl(state.scene, m, state.frame, v);
      });
    } else {
      const viewL = state.modeL === "gt" && (state.view === "err" || state.view === "serr") ? "rgb" : state.view;
      const viewR = state.modeR === "gt" && (state.view === "err" || state.view === "serr") ? "rgb" : state.view;
      $("imgLeft").src = imgUrl(state.scene, state.modeL, state.frame, viewL);
      if (!isSingle) {
        $("imgRight").src = imgUrl(state.scene, state.modeR, state.frame, viewR);
      }
    }
    applyTransform();
  }

  function applyTransform() {
    const pct = Math.max(0, Math.min(100, state.wipe * 100));
    $("wipeDivider").style.left = `${pct}%`;
    const z = state.zoom;
    const t = `translate(${state.panX}px, ${state.panY}px) scale(${z})`;
    $("layerLeft").style.transform = t;
    $("layerRight").style.transform = t;
    $("layerLeft").classList.toggle("pixelated", z >= 4);
    $("layerRight").classList.toggle("pixelated", z >= 4);
    // Clip left layer in viewport coordinates so split curtain stays vertical at wipe%
    if (state.layout === "wipe") {
      const inner = $("viewportInner");
      const w = inner.clientWidth || 1;
      // Map curtain screen X back into pre-transformed element space
      const cx = w * 0.5;
      const splitPx = w * state.wipe;
      const localX = (splitPx - cx - state.panX) / z + cx;
      const clipRight = Math.max(0, Math.min(100, (1 - localX / w) * 100));
      $("layerLeft").style.clipPath = `inset(0 ${clipRight}% 0 0)`;
    } else {
      $("layerLeft").style.clipPath = "none";
    }
    updateCrosshairDOM();
  }

  function updateCrosshairDOM() {
    const show = state.layout !== "quad";
    $("crossX").style.display = show ? "block" : "none";
    $("crossY").style.display = show ? "block" : "none";
    $("crossDot").style.display = show ? "block" : "none";
    if (!show) return;

    const inner = $("viewportInner");
    const w = inner.clientWidth || 1280;
    const h = inner.clientHeight || 720;
    const nx = state.probeX / 1280;
    const ny = state.probeY / 720;
    const sx = (nx * w - w * 0.5) * state.zoom + w * 0.5 + state.panX;
    const sy = (ny * h - h * 0.5) * state.zoom + h * 0.5 + state.panY;
    $("crossX").style.top = `${sy}px`;
    $("crossY").style.left = `${sx}px`;
    $("crossDot").style.left = `${sx}px`;
    $("crossDot").style.top = `${sy}px`;
    $("hudCoord").textContent = `px (${state.probeX}, ${state.probeY}) · zoom ${state.zoom}×`;
  }

  function clientToPixel(ev, targetEl = $("viewportInner")) {
    const rect = targetEl.getBoundingClientRect();
    const rx = ev.clientX - rect.left;
    const ry = ev.clientY - rect.top;
    if (targetEl === $("viewportInner")) {
      const lx = (rx - rect.width * 0.5 - state.panX) / state.zoom + rect.width * 0.5;
      const ly = (ry - rect.height * 0.5 - state.panY) / state.zoom + rect.height * 0.5;
      const px = Math.max(0, Math.min(1279, Math.round((lx / rect.width) * 1280)));
      const py = Math.max(0, Math.min(719, Math.round((ly / rect.height) * 720)));
      return { x: px, y: py };
    }
    const px = Math.max(0, Math.min(1279, Math.round((rx / rect.width) * 1280)));
    const py = Math.max(0, Math.min(719, Math.round((ry / rect.height) * 720)));
    return { x: px, y: py };
  }

  // =========================================================================
  // 3. Axis 1: See Inside — Pixel Probe, 8x Loupe, Stiefel Compass & Scanline
  // =========================================================================
  async function fetchProbe() {
    try {
      const q = new URLSearchParams({
        scene: state.scene,
        frame: String(state.frame),
        x: String(state.probeX),
        y: String(state.probeY),
      });
      const r = await fetch(`/api/probe?${q.toString()}`, { cache: "no-store" });
      if (!r.ok) return;
      state.probeData = await r.json();
      renderProbe();
      drawLoupe();
    } catch (_) {}
  }

  function renderProbe() {
    const d = state.probeData;
    if (!d) return;
    $("probeCoordLbl").textContent = `(${d.x}, ${d.y})`;
    $("probeFrameLbl").textContent = `scene ${d.scene.toUpperCase()} · f=${d.frame}`;

    const os = d.modes?.osabt;
    const bs = d.modes?.baseline;
    $("kvOsErr").textContent = os?.abs_err != null ? os.abs_err.toFixed(5) : "—";
    $("kvOsErr").style.color = os?.abs_err != null && bs?.abs_err != null && os.abs_err <= bs.abs_err ? "var(--green)" : "var(--cyan)";
    $("kvBsErr").textContent = bs?.abs_err != null ? bs.abs_err.toFixed(5) : "—";
    $("kvSid").textContent = d.gbuf?.sid != null ? `sid ${d.gbuf.sid}` : "n/a";
    $("kvAngle").textContent = d.gbuf?.angle_deg != null ? `${d.gbuf.angle_deg.toFixed(1)}°` : "n/a";

    const rows = [
      { key: "gt", label: "GT", cls: "dot-gt" },
      { key: "osabt", label: "OS-ABT", cls: "dot-osabt" },
      { key: "baseline", label: "Baseline", cls: "dot-baseline" },
      { key: "ablated", label: "Ablated", cls: "dot-ablated" },
    ];
    $("probeTableBody").innerHTML = rows
      .map((row) => {
        const m = d.modes?.[row.key];
        if (!m) return `<tr><td><span class="mode-dot ${row.cls}"></span>${row.label}</td><td colspan="3" style="color:var(--ink-dim)">—</td></tr>`;
        const rgb = m.rgb.map((v) => v.toFixed(3)).join(", ");
        const err = m.abs_err != null ? m.abs_err.toFixed(5) : "ref";
        return `<tr>
          <td><span class="mode-dot ${row.cls}"></span>${row.label}</td>
          <td>${rgb}</td>
          <td>${m.luma.toFixed(4)}</td>
          <td style="color:${row.key === "osabt" ? "var(--cyan)" : "var(--ink)"}">${err}</td>
        </tr>`;
      })
      .join("");

    // Stiefel Neighbor Compass
    const gb = d.gbuf;
    if (gb && gb.normal) {
      $("stiefelCard").style.display = "block";
      $("normVecLbl").textContent = `n=(${gb.normal.map((v) => v.toFixed(2)).join(",")})`;
      $("scCenter").textContent = `sid ${gb.sid}`;
      const map = [
        ["scUp", "up", "UP"],
        ["scDown", "down", "DN"],
        ["scLeft", "left", "L"],
        ["scRight", "right", "R"],
      ];
      map.forEach(([id, key, lbl]) => {
        const cell = $(id);
        const nb = gb.neighbors?.[key];
        if (!nb) {
          cell.className = "stiefel-cell";
          cell.innerHTML = `${lbl}<br/>edge`;
        } else {
          const isZero = nb.stiefel_w <= 0.0001;
          cell.className = `stiefel-cell ${isZero ? "zero" : "pass"}`;
          cell.innerHTML = `${lbl} Ψ=${nb.stiefel_w.toFixed(2)}<br/>n·n=${nb.ndot.toFixed(2)}`;
        }
      });
    } else {
      $("stiefelCard").style.display = "none";
    }
  }

  function drawLoupe() {
    const img = $("imgLeft");
    const cv = $("loupeCanvas");
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, cv.width, cv.height);
    if (!img.complete || img.naturalWidth === 0) return;
    const rad = 7; // 15x15 pixels zoomed into 112x112
    const sx = Math.max(0, Math.min(img.naturalWidth - 15, state.probeX - rad));
    const sy = Math.max(0, Math.min(img.naturalHeight - 15, state.probeY - rad));
    ctx.drawImage(img, sx, sy, 15, 15, 0, 0, cv.width, cv.height);
    // Center pixel reticle
    const cell = cv.width / 15;
    ctx.strokeStyle = "#38d9c0";
    ctx.lineWidth = 1.5;
    ctx.strokeRect((state.probeX - sx) * cell, (state.probeY - sy) * cell, cell, cell);
  }

  async function fetchScanline() {
    try {
      const q = new URLSearchParams({
        scene: state.scene,
        frame: String(state.frame),
        y: String(state.probeY),
        step: "2",
      });
      const r = await fetch(`/api/profile?${q.toString()}`, { cache: "no-store" });
      if (!r.ok) return;
      state.scanlineData = await r.json();
      drawScanline();
    } catch (_) {}
  }

  function prepCanvas(cv) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = cv.getBoundingClientRect();
    const w = Math.max(100, Math.round(rect.width * dpr));
    const h = Math.max(60, Math.round(rect.height * dpr));
    if (cv.width !== w || cv.height !== h) {
      cv.width = w;
      cv.height = h;
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function drawScanline() {
    const cv = $("scanlineCanvas");
    const { ctx, w, h } = prepCanvas(cv);
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, w, h);

    const d = state.scanlineData;
    const isErr = state.scopeMode === "err";
    $("stripLabel").textContent = `AXIS 1 // 1D SCANLINE ${isErr ? "|ERROR|" : "LUMA"} (y = ${state.probeY}, f = ${state.frame})`;
    if (!d || !d.series) return;

    const keys = isErr
      ? [
          { k: "ablated_err", c: "#f87171", lw: 1.2 },
          { k: "baseline_err", c: "#f5a623", lw: 1.4 },
          { k: "osabt_err", c: "#38d9c0", lw: 2.0 },
        ]
      : [
          { k: "gt", c: "#f4f7fb", lw: 1.6 },
          { k: "ablated", c: "#f87171", lw: 1.1 },
          { k: "baseline", c: "#f5a623", lw: 1.3 },
          { k: "osabt", c: "#38d9c0", lw: 1.8 },
        ];

    let maxV = 0.05;
    let nPts = 0;
    keys.forEach(({ k }) => {
      const arr = d.series[k];
      if (!arr) return;
      nPts = Math.max(nPts, arr.length);
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] > maxV) maxV = arr[i];
      }
    });
    maxV *= 1.1;

    const padL = 38, padR = 10, padT = 10, padB = 18;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    // Optional surface-ID transitions as faint vertical seams
    if (d.sid_row && d.sid_row.length > 1) {
      ctx.strokeStyle = "rgba(154, 168, 188, 0.14)";
      ctx.lineWidth = 1;
      for (let i = 1; i < d.sid_row.length; i++) {
        if (d.sid_row[i] !== d.sid_row[i - 1]) {
          const sx = padL + (i / (d.sid_row.length - 1)) * pw;
          ctx.beginPath();
          ctx.moveTo(sx, padT);
          ctx.lineTo(sx, padT + ph);
          ctx.stroke();
        }
      }
    }

    // Horizontal gridlines
    ctx.strokeStyle = "rgba(45, 55, 72, 0.45)";
    ctx.fillStyle = "#5e6c80";
    ctx.font = "10px IBM Plex Mono, monospace";
    for (let g = 0; g <= 3; g++) {
      const gy = padT + ph * (1 - g / 3);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + pw, gy);
      ctx.stroke();
      const val = (maxV * g) / 3;
      ctx.fillText(val < 1 ? val.toFixed(3) : val.toFixed(1), 4, gy + 3);
    }

    // Curves
    keys.forEach(({ k, c, lw }) => {
      const arr = d.series[k];
      if (!arr || arr.length < 2) return;
      ctx.strokeStyle = c;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = padL + (i / (arr.length - 1)) * pw;
        const y = padT + ph * (1 - Math.min(1, arr[i] / maxV));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Probed X cursor line
    const cx = padL + (state.probeX / 1280) * pw;
    ctx.strokeStyle = "rgba(56, 217, 192, 0.7)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, padT + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#38d9c0";
    ctx.fillText(`x=${state.probeX}`, Math.min(w - 46, cx + 4), padT + 10);
  }

  // =========================================================================
  // 4. Axis 2: See Across Time — Timeline Chart & Epipolar x-t Slice
  // =========================================================================
  async function fetchTimeline() {
    try {
      const r = await fetch(`/api/timeline?scene=${state.scene}`, { cache: "no-store" });
      if (!r.ok) return;
      state.timelineData = await r.json();
      drawTimeline();
      renderTimelineTable();
    } catch (_) {}
  }

  function drawTimeline() {
    const cv = $("timelineCanvas");
    if ($("timelineBox").style.display === "none") return;
    const { ctx, w, h } = prepCanvas(cv);
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, w, h);

    const d = state.timelineData;
    if (!d || !d.frames || !d.frames.length) return;

    const minF = d.frames[0];
    const maxF = Math.max(minF + 1, d.frames[d.frames.length - 1]);
    const padL = 40, padR = 12, padT = 12, padB = 20;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    // Log-ish or clamped Y scale so OS-ABT detail isn't crushed by ablated f40 spike (5.78)
    const maxY = d.scene === "c" ? 1.0 : 0.35;
    const mapX = (f) => padL + ((f - minF) / (maxF - minF)) * pw;
    const mapY = (v) => padT + ph * (1 - Math.min(1, v / maxY));

    // Grid
    ctx.strokeStyle = "rgba(45, 55, 72, 0.45)";
    ctx.fillStyle = "#5e6c80";
    ctx.font = "10px IBM Plex Mono, monospace";
    for (let g = 0; g <= 4; g++) {
      const gy = padT + ph * (1 - g / 4);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + pw, gy);
      ctx.stroke();
      ctx.fillText(((maxY * g) / 4).toFixed(2), 4, gy + 3);
    }

    // Scene C oracle threshold line (0.15 Dirac gate)
    if (d.scene === "c") {
      const y15 = mapY(0.15);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.5)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, y15);
      ctx.lineTo(padL + pw, y15);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#34d399";
      ctx.fillText("gate 0.15", padL + pw - 56, y15 - 3);
    }

    // Events
    (d.events || []).forEach((ev) => {
      if (ev.frame < minF || ev.frame > maxF) return;
      const ex = mapX(ev.frame);
      ctx.strokeStyle = ev.color;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(ex, padT);
      ctx.lineTo(ex, padT + ph);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    const modes = [
      { k: "ablated", c: "#f87171", lw: 1.2 },
      { k: "baseline", c: "#f5a623", lw: 1.5 },
      { k: "osabt", c: "#38d9c0", lw: 2.2 },
    ];
    modes.forEach(({ k, c, lw }) => {
      const pts = d.series?.[k] || [];
      if (!pts.length) return;
      ctx.strokeStyle = c;
      ctx.fillStyle = c;
      ctx.lineWidth = lw;
      ctx.beginPath();
      pts.forEach((p, idx) => {
        const x = mapX(p.frame);
        const y = mapY(p.rmse);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(mapX(p.frame), mapY(p.rmse), 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    // Current frame marker
    const fx = mapX(state.frame);
    ctx.strokeStyle = "#f4f7fb";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(fx, padT);
    ctx.lineTo(fx, padT + ph);
    ctx.stroke();
    ctx.fillStyle = "#f4f7fb";
    ctx.fillText(`f${state.frame} RMSE`, Math.min(w - 64, fx + 4), padT + 10);
  }

  function renderTimelineTable() {
    const d = state.timelineData;
    const tb = $("timeTableBody");
    if (!d || !d.frames) {
      tb.innerHTML = "";
      return;
    }
    $("timeSceneLbl").textContent = `Scene ${d.scene.toUpperCase()} (${d.frames.length} keyframes)`;
    const byF = {};
    ["osabt", "baseline", "ablated"].forEach((m) => {
      (d.series?.[m] || []).forEach((p) => {
        byF[p.frame] = byF[p.frame] || {};
        byF[p.frame][m] = p.rmse;
      });
    });
    tb.innerHTML = d.frames
      .map((f) => {
        const row = byF[f] || {};
        const tag = d.scene === "c" && f === 40 ? " ⚡" : d.scene === "c" && f === 75 ? " 🎨" : "";
        return `<tr data-tf="${f}" style="cursor:pointer;${f === state.frame ? "background:rgba(56,217,192,0.1)" : ""}">
          <td>f${f}${tag}</td>
          <td style="color:var(--cyan)">${row.osabt != null ? row.osabt.toFixed(4) : "—"}</td>
          <td style="color:var(--amber)">${row.baseline != null ? row.baseline.toFixed(4) : "—"}</td>
          <td style="color:var(--coral)">${row.ablated != null ? row.ablated.toFixed(4) : "—"}</td>
        </tr>`;
      })
      .join("");
    tb.querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => setFrame(parseInt(tr.dataset.tf, 10)));
    });
  }

  function updateEpipolarImage() {
    $("imgEpipolar").src = imgUrl("c", state.epiMode, 0, "epipolar");
    updateEpipolarLine();
  }

  function updateEpipolarLine() {
    const pct = state.scene === "c" ? (state.frame / 119) * 100 : (40 / 119) * 100;
    $("epipolarLine").style.top = `${Math.max(0, Math.min(100, pct))}%`;
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
    } else {
      const frames = getSceneFrames();
      if (frames.length <= 1) return;
      state.playing = true;
      $("btnPlay").textContent = "⏸ PAUSE";
      $("btnPlay").classList.add("active");
      state.playTimer = setInterval(() => {
        const fs = getSceneFrames();
        const idx = fs.indexOf(state.frame);
        setFrame(fs[(idx + 1) % fs.length]);
      }, 420);
    }
  }

  function stopPlayback() {
    state.playing = false;
    clearInterval(state.playTimer);
    $("btnPlay").textContent = "▶ PLAY";
    $("btnPlay").classList.remove("active");
  }

  // =========================================================================
  // 5. Axis 3: See Across Possibilities — Live First-Principles Math Labs
  // =========================================================================
  function smoothstep2(t) {
    const c = Math.max(0, Math.min(1, t));
    return c * c * (3 - 2 * c);
  }

  function drawLab1() {
    const c0 = parseFloat($("rngC0").value);
    const curv = parseFloat($("rngCurv").value);
    $("lblC0").textContent = c0.toFixed(2);
    $("lblCurv").textContent = curv.toFixed(2);

    const cv = $("lab1Canvas");
    const ctx = cv.getContext("2d");
    const w = cv.width, h = cv.height;
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, w, h);

    // Plot kernel weight vs angle on curved cylinder (left half) and orthogonal wall junction (right half)
    const R = curv > 0.05 ? 1.0 / curv : 20.0;
    const planeTol = 0.06;
    const padL = 32, padR = 12, padT = 18, padB = 20;
    const pw = w - padL - padR, ph = h - padT - padB;

    // Grid
    ctx.strokeStyle = "rgba(45, 55, 72, 0.5)";
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.fillStyle = "#8a97aa";
    ctx.font = "10px IBM Plex Mono, monospace";
    ctx.fillText("Curved Cylinder Arc (δ_geo ≡ 0)", padL + 4, 12);
    ctx.fillText("90° Seam", padL + pw - 52, 12);

    // Draw weight curve across lateral chord offset s in [0, 0.35] m on cylinder
    const N = 120;
    let avgSupport = 0;
    let orthoBleed = 0;
    ctx.strokeStyle = state.geoMode === "geodesic" ? "#38d9c0" : state.geoMode === "maxplane" ? "#f5a623" : "#f87171";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      let weight = 0;
      if (t < 0.78) {
        // Along curved cylinder arc of radius R
        const arc = (t / 0.78) * 0.28;
        const ang = arc / R;
        const ndot = Math.cos(ang);
        const dxX = R * Math.sin(ang);
        const dxZ = R * (Math.cos(ang) - 1);
        const npX = 0, npZ = 1;
        const nqX = Math.sin(ang), nqZ = Math.cos(ang);
        if (state.geoMode === "gaussian") {
          const dn2 = (npX - nqX) ** 2 + (npZ - nqZ) ** 2;
          weight = Math.exp(-dn2 / 0.55) * Math.exp(-(dxX * dxX + dxZ * dxZ) / 0.35);
        } else {
          const wn = ndot <= c0 ? 0 : smoothstep2((ndot - c0) / Math.max(1 - c0, 1e-4));
          let outPlane = 0;
          if (state.geoMode === "geodesic") {
            const nmX = 0.5 * (npX + nqX), nmZ = 0.5 * (npZ + nqZ);
            outPlane = Math.abs(dxX * nmX + dxZ * nmZ); // identically 0 on circle!
          } else {
            const dp = Math.abs(dxX * npX + dxZ * npZ);
            const dq = Math.abs(dxX * nqX + dxZ * nqZ);
            outPlane = Math.max(dp, dq);
          }
          weight = wn * smoothstep2(1 - outPlane / planeTol);
        }
        avgSupport += weight;
      } else {
        // Across 90° orthogonal junction (ndot = 0.0)
        if (state.geoMode === "gaussian") {
          weight = Math.exp(-2.0 / 0.55); // classical bilateral leaks ~2.6%!
        } else {
          weight = 0.0 <= c0 ? 0.0 : smoothstep2((0.0 - c0) / (1 - c0));
        }
        orthoBleed = weight;
      }
      const x = padL + t * pw;
      const y = padT + ph * (1 - weight);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Seam separator
    const seamX = padL + 0.78 * pw;
    ctx.strokeStyle = "rgba(248, 113, 113, 0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(seamX, padT);
    ctx.lineTo(seamX, padT + ph);
    ctx.stroke();
    ctx.setLineDash([]);

    const suppPct = Math.round((avgSupport / (N * 0.78)) * 100);
    $("lab1Readout").textContent = `bleed=${orthoBleed.toFixed(4)} · cyl=${suppPct}%`;
    $("lab1Readout").style.color = orthoBleed === 0 && suppPct > 70 ? "var(--green)" : "var(--amber)";
  }

  function p2Box(u, d) {
    const k = Math.floor(u);
    const r = u - k;
    const rcl = Math.min(r, d);
    const p1 = d;
    const p2 = d * (1 - 0.5 * d);
    const frac = 0.5 * rcl * rcl + d * Math.max(0, r - d);
    return 0.5 * k * (k - 1) * p1 + k * (r * p1 + p2 - p1) + frac;
  }

  function antiderivA2(u0, du, duty) {
    if (du <= 1e-4) return u0 - Math.floor(u0) < duty ? 1 : 0;
    const a = p2Box(u0 - du, duty);
    const b = p2Box(u0, duty);
    const c = p2Box(u0 + du, duty);
    return Math.max(0, Math.min(1, (c - 2 * b + a) / (du * du)));
  }

  function drawLab2() {
    const theta = parseFloat($("rngTheta").value);
    const freq = parseFloat($("rngFreq").value);
    $("lblTheta").textContent = `${theta.toFixed(1)}°`;
    $("lblFreq").textContent = freq.toFixed(2);

    const cv = $("lab2Canvas");
    const ctx = cv.getContext("2d");
    const w = cv.width, h = cv.height;
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, w, h);

    const cosT = Math.cos((theta * Math.PI) / 180);
    const du = Math.min(25, (freq * 0.045) / Math.max(cosT * cosT, 1e-4));
    const duty = 0.42;
    const padL = 28, padR = 10, padT = 18, padB = 18;
    const pw = w - padL - padR, ph = h - padT - padB;
    ctx.strokeStyle = "rgba(45, 55, 72, 0.5)";
    ctx.strokeRect(padL, padT, pw, ph);

    let msePoint = 0, mseA2 = 0;
    const N = 160;
    const ptArr = [], a2Arr = [], gtArr = [];
    for (let i = 0; i < N; i++) {
      const u = i * 0.137 + 0.03; // irrational stride
      const pt = u - Math.floor(u) < duty ? 1 : 0;
      const a2 = antiderivA2(u, du, duty);
      // 64x super-sampled Bartlett GT
      let gt = 0, wsum = 0;
      for (let s = -32; s <= 32; s++) {
        const q = s / 32;
        const wt = 1 - Math.abs(q);
        const uu = u + q * du;
        gt += wt * (uu - Math.floor(uu) < duty ? 1 : 0);
        wsum += wt;
      }
      gt /= wsum;
      ptArr.push(pt);
      a2Arr.push(a2);
      gtArr.push(gt);
      msePoint += (pt - gt) ** 2;
      mseA2 += (a2 - gt) ** 2;
    }
    const snr = 10 * Math.log10((msePoint + 1e-9) / (mseA2 + 1e-9));
    $("lab2Readout").textContent = `Δu=${du.toFixed(1)} per · SNR +${snr.toFixed(1)} dB`;

    const drawSeries = (arr, color, lw) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = padL + (i / (arr.length - 1)) * pw;
        const y = padT + ph * (1 - arr[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    drawSeries(ptArr, "rgba(248, 113, 113, 0.65)", 1.0);
    drawSeries(gtArr, "#f4f7fb", 2.0);
    drawSeries(a2Arr, "#38d9c0", 1.5);

    ctx.fillStyle = "#8a97aa";
    ctx.font = "10px IBM Plex Mono, monospace";
    ctx.fillText("Coral: 1-spp Alias | White: 64x GT | Cyan: Exact A₂⊥", padL + 4, 12);
  }

  function drawLab3() {
    const dirac = parseFloat($("rngDirac").value);
    const ag = parseFloat($("rngGauge").value);
    $("lblDirac").textContent = `${dirac.toFixed(1)}×`;
    $("lblGauge").textContent = ag.toFixed(2);

    const cv = $("lab3Canvas");
    const ctx = cv.getContext("2d");
    const w = cv.width, h = cv.height;
    ctx.fillStyle = "#050709";
    ctx.fillRect(0, 0, w, h);

    const T = 70;
    const gt = [], taa = [], trap = [], zdf = [];
    let sTaa = 1.0, sTrap = 1.0, sZdf = 1.0;
    let rhoSample = 0;
    for (let f = 0; f < T; f++) {
      const target = f === 25 ? dirac : f >= 45 ? 2.1 : 1.0;
      gt.push(target);
      // Naive Explicit TAA (alpha = 0.1)
      sTaa = 0.9 * sTaa + 0.1 * target;
      taa.push(sTaa);

      // Un-damped Trapezoidal TPT (beta = 0 => rings negative when g >> 1!)
      const gTrap = Math.abs(target - sTrap) > 0.6 ? 4.5 : 0.08;
      const Gtrap = gTrap / (1 + gTrap);
      const vTrap = (target - sTrap) * Gtrap;
      const yTrap = sTrap + vTrap;
      sTrap = yTrap + vTrap; // pole = (1 - g)/(1 + g) < 0 -> overshoot/ringing!
      trap.push(yTrap);

      // OS-ABT Affine Gauge + L-Stable Port-Hamiltonian ZDF-TPT (beta_n = G_n)
      const prevTarget = f > 0 ? gt[f - 1] : 1.0;
      const kappa = target / Math.max(prevTarget, 0.05);
      const sGauge = kappa * sZdf;
      const sBase = sZdf + ag * (sGauge - sZdf);
      const resid = Math.abs(target - sBase);
      const gZdf = 0.08 + (resid > 0.35 ? 2.5 * resid : 0.0);
      const Gz = gZdf / (1 + gZdf);
      const vZ = (target - sBase) * Gz;
      const yZ = sBase + vZ;
      sZdf = yZ + vZ * (1 - Gz); // exact spectral radius rho = (1 + g)^-2 >= 0!
      if (f === 25) rhoSample = 1 / ((1 + gZdf) * (1 + gZdf));
      zdf.push(yZ);
    }

    const padL = 28, padR = 10, padT = 18, padB = 18;
    const pw = w - padL - padR, ph = h - padT - padB;
    const maxY = Math.max(dirac * 1.15, 3.0);
    const minY = -1.0;
    ctx.strokeStyle = "rgba(45, 55, 72, 0.5)";
    ctx.strokeRect(padL, padT, pw, ph);

    const mapY = (v) => padT + ph * (1 - (Math.max(minY, Math.min(maxY, v)) - minY) / (maxY - minY));
    // Zero line
    ctx.strokeStyle = "rgba(154, 168, 188, 0.25)";
    ctx.beginPath();
    ctx.moveTo(padL, mapY(0));
    ctx.lineTo(padL + pw, mapY(0));
    ctx.stroke();

    const drawCurve = (arr, color, lw) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = padL + (i / (arr.length - 1)) * pw;
        const y = mapY(arr[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    drawCurve(gt, "#f4f7fb", 1.8);
    drawCurve(taa, "#f87171", 1.2);
    drawCurve(trap, "#f5a623", 1.3);
    drawCurve(zdf, "#38d9c0", 2.0);

    ctx.fillStyle = "#8a97aa";
    ctx.font = "10px IBM Plex Mono, monospace";
    ctx.fillText("Cyan: ZDF-TPT (0 ring) | Amber: Trap (ring) | Coral: TAA", padL + 4, 12);
    $("lab3Readout").textContent = `f25 err=${Math.abs(zdf[25] - gt[25]).toFixed(3)} · ρ=${rhoSample.toFixed(2)}`;
  }

  async function loadWgslCode() {
    try {
      const r = await fetch("/osabt.wgsl");
      const txt = await r.text();
      $("wgslCode").textContent = txt;
      $("wgslStatus").textContent = navigator.gpu ? "WebGPU available" : "WGSL spec + Canvas2D";
    } catch (_) {
      $("wgslStatus").textContent = "unavailable";
    }
  }

  // =========================================================================
  // 6. Triage Log (P / F / N) -> artifacts/workbench_notes.jsonl
  // =========================================================================
  async function loadNotes() {
    try {
      const r = await fetch("/api/notes", { cache: "no-store" });
      if (!r.ok) return;
      const d = await r.json();
      state.notes = d.notes || [];
      renderNotes();
    } catch (_) {}
  }

  async function submitTriage(verdict) {
    let noteText = $("noteInput").value.trim();
    if (verdict === "N" && !noteText) {
      $("noteInput").focus();
      showToast("Type a note and press Enter or click [N]");
      return;
    }
    const payload = {
      verdict,
      scene: state.scene,
      mode: state.modeL,
      frame: state.frame,
      x: state.probeX,
      y: state.probeY,
      note: noteText || `${verdict === "P" ? "PASS" : "FAIL"} @ scene_${state.scene} f${state.frame} (${state.probeX},${state.probeY})`,
      metrics: state.probeData?.modes?.osabt || null,
    };
    try {
      const r = await fetch("/api/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (r.ok) {
        const res = await r.json();
        state.notes.push(res.record);
        $("noteInput").value = "";
        renderNotes();
        showToast(`Logged [${verdict}] → workbench_notes.jsonl`);
      }
    } catch (_) {
      showToast("Failed to save note");
    }
  }

  function renderNotes() {
    $("notesBadge").textContent = String(state.notes.length);
    const el = $("notesList");
    if (!state.notes.length) {
      el.innerHTML = `<div class="mono" style="color:var(--ink-dim);font-size:11px;padding:8px 2px">No triage entries yet. Press P, F, or N at any frame/pixel.</div>`;
      return;
    }
    el.innerHTML = [...state.notes]
      .reverse()
      .map(
        (n) => `
      <div class="note-item">
        <div class="note-item-head">
          <span><span class="verdict-pill verdict-${n.verdict}">${n.verdict}</span> scene_${n.scene} · ${n.mode} · f${n.frame} (${n.x ?? "—"},${n.y ?? "—"})</span>
          <span style="color:var(--ink-dim)">${(n.ts || "").replace("T", " ").replace("Z", "")}</span>
        </div>
        <div style="color:var(--ink-mid)">${n.note || ""}</div>
      </div>`
      )
      .join("");
  }

  // =========================================================================
  // 7. Wire Up Events & Boot
  // =========================================================================
  function bindEvents() {
    $("btnAutoPoll").addEventListener("click", () => {
      state.autoPoll = !state.autoPoll;
      $("btnAutoPoll").classList.toggle("active", state.autoPoll);
      $("liveText").textContent = state.autoPoll ? "LIVE · 2s" : "PAUSED";
    });
    $("btnRefresh").addEventListener("click", () => {
      fetchStatus(true);
      showToast("Refreshed renders & telemetry");
    });

    // Scene Seg
    document.querySelectorAll("#sceneSeg button").forEach((b) => {
      b.addEventListener("click", () => selectScene(b.dataset.scene));
    });

    // Layout Seg
    document.querySelectorAll("#layoutSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.layout = b.dataset.layout;
        document.querySelectorAll("#layoutSeg button").forEach((x) => x.classList.toggle("active", x === b));
        updateViewportImages();
      });
    });

    // Mode / View / Tonemap selectors
    $("selModeL").addEventListener("change", (e) => { state.modeL = e.target.value; updateViewportImages(); });
    $("selModeR").addEventListener("change", (e) => { state.modeR = e.target.value; updateViewportImages(); });
    $("selView").addEventListener("change", (e) => { state.view = e.target.value; updateViewportImages(); });
    $("selTonemap").addEventListener("change", (e) => { state.tonemap = e.target.value; updateViewportImages(); });

    $("rngExp").addEventListener("input", (e) => {
      state.exposure = parseFloat(e.target.value);
      $("lblExp").textContent = state.exposure.toFixed(1);
      updateViewportImages();
    });
    $("rngAmp").addEventListener("input", (e) => {
      state.amp = parseInt(e.target.value, 10);
      $("lblAmp").textContent = `${state.amp}×`;
      updateViewportImages();
    });

    // Zoom Seg
    document.querySelectorAll("#zoomSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.zoom = parseInt(b.dataset.zoom, 10);
        if (state.zoom === 1) { state.panX = 0; state.panY = 0; }
        document.querySelectorAll("#zoomSeg button").forEach((x) => x.classList.toggle("active", x === b));
        applyTransform();
      });
    });

    // Scope Mode
    document.querySelectorAll("#scopeModeSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.scopeMode = b.dataset.scope;
        document.querySelectorAll("#scopeModeSeg button").forEach((x) => x.classList.toggle("active", x === b));
        drawScanline();
      });
    });

    $("btnPlay").addEventListener("click", togglePlayback);

    // Split Wipe Drag
    let draggingWipe = false;
    $("wipeDivider").addEventListener("pointerdown", (e) => {
      draggingWipe = true;
      e.stopPropagation();
      $("wipeDivider").setPointerCapture(e.pointerId);
    });
    $("wipeDivider").addEventListener("pointermove", (e) => {
      if (!draggingWipe) return;
      const rect = $("viewportInner").getBoundingClientRect();
      state.wipe = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / rect.width));
      applyTransform();
    });
    $("wipeDivider").addEventListener("pointerup", () => { draggingWipe = false; });

    // Viewport Click to Probe & Shift+Drag to Pan
    let panning = false, panStartX = 0, panStartY = 0, panMoved = false;
    $("viewportInner").addEventListener("pointerdown", (e) => {
      if (state.layout === "quad") return;
      if (state.zoom > 1 && (e.shiftKey || e.button === 1)) {
        panning = true;
        panMoved = false;
        panStartX = e.clientX - state.panX;
        panStartY = e.clientY - state.panY;
        $("viewportInner").setPointerCapture(e.pointerId);
      }
    });
    $("viewportInner").addEventListener("pointermove", (e) => {
      if (panning) {
        panMoved = true;
        state.panX = e.clientX - panStartX;
        state.panY = e.clientY - panStartY;
        applyTransform();
      }
    });
    $("viewportInner").addEventListener("pointerup", (e) => {
      if (panning) {
        panning = false;
        if (panMoved) return;
      }
      if (state.layout === "quad") return;
      const pt = clientToPixel(e, $("viewportInner"));
      state.probeX = pt.x;
      state.probeY = pt.y;
      updateCrosshairDOM();
      fetchProbe();
      fetchScanline();
    });

    // Quad Cell click to probe
    document.querySelectorAll(".quad-cell").forEach((cell) => {
      cell.addEventListener("click", (e) => {
        const pt = clientToPixel(e, cell);
        state.probeX = pt.x;
        state.probeY = pt.y;
        fetchProbe();
        fetchScanline();
      });
    });

    // Draw loupe once image finishes loading
    $("imgLeft").addEventListener("load", drawLoupe);

    // Inspector Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${t}`));
        if (t === "poss") { drawLab1(); drawLab2(); drawLab3(); }
        if (t === "time") { updateEpipolarImage(); }
      });
    });

    // Epipolar mode & click-to-jump
    $("selEpiMode").addEventListener("change", (e) => {
      state.epiMode = e.target.value;
      updateEpipolarImage();
    });
    $("epipolarBox").addEventListener("click", (e) => {
      const rect = $("epipolarBox").getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      const targetF = Math.round(frac * 119);
      const fs = getSceneFrames("c");
      const nearest = fs.reduce((best, f) => (Math.abs(f - targetF) < Math.abs(best - targetF) ? f : best), fs[0]);
      selectScene("c", nearest);
    });

    // Lab sliders
    ["rngC0", "rngCurv"].forEach((id) => $(id).addEventListener("input", drawLab1));
    document.querySelectorAll("#geoModeSeg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.geoMode = b.dataset.geo;
        document.querySelectorAll("#geoModeSeg button").forEach((x) => x.classList.toggle("active", x === b));
        drawLab1();
      });
    });
    ["rngTheta", "rngFreq"].forEach((id) => $(id).addEventListener("input", drawLab2));
    ["rngDirac", "rngGauge"].forEach((id) => $(id).addEventListener("input", drawLab3));

    // Triage buttons
    $("btnMarkP").addEventListener("click", () => submitTriage("P"));
    $("btnMarkF").addEventListener("click", () => submitTriage("F"));
    $("btnMarkN").addEventListener("click", () => submitTriage("N"));
    $("noteInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") submitTriage("N");
    });

    // Global keyboard shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) {
        return;
      }
      const k = e.key.toLowerCase();
      if (k === "p") submitTriage("P");
      else if (k === "f") submitTriage("F");
      else if (k === "n") {
        document.querySelector('.tab-btn[data-tab="triage"]').click();
        $("noteInput").focus();
      } else if (k === "r") {
        fetchStatus(true);
        showToast("Refreshed");
      } else if (e.key === "ArrowRight") {
        const fs = getSceneFrames();
        if (fs.length > 1) setFrame(fs[(fs.indexOf(state.frame) + 1) % fs.length]);
      } else if (e.key === "ArrowLeft") {
        const fs = getSceneFrames();
        if (fs.length > 1) setFrame(fs[(fs.indexOf(state.frame) - 1 + fs.length) % fs.length]);
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlayback();
      } else if (k === "1") selectScene("a");
      else if (k === "2") selectScene("b");
      else if (k === "3") selectScene("c");
      else if (k === "4") selectScene("d");
    });

    window.addEventListener("resize", () => {
      applyTransform();
      drawScanline();
      drawTimeline();
    });
  }

  function boot() {
    bindEvents();
    loadWgslCode();
    loadNotes();
    fetchStatus(true);
    drawLab1();
    drawLab2();
    drawLab3();
    setInterval(() => {
      if (state.autoPoll) fetchStatus(false);
    }, 2000);
  }

  boot();
})();
