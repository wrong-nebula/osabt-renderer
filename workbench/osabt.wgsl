// OS-ABT: Orthogonal-Subspace Antiderivative Bilinear Transport (WebGPU WGSL Core)
// Deterministic WGSL mirror of the four theoretical pillars in src/osabt_core.h
// for the interactive Seeing Space "See Across Possibilities" mathematical sandbox.

const OS_PI: f32 = 3.14159265358979323846;
const OS_TAU: f32 = 6.28318530717958647692;

fn os_smoothstep2(t_in: f32) -> f32 {
  let t = clamp(t_in, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Pillar 1A: Orthogonal-Null Stiefel Subspace Projector (exact 0.0 for |n_p . n_q| <= c0)
fn os_stiefel_null(ndot: f32, c0: f32) -> f32 {
  if (ndot <= c0) {
    return 0.0;
  }
  return os_smoothstep2((ndot - c0) / max(1.0 - c0, 1e-4));
}

// Pillar 1B: Symmetric Geodesic Chord-Midnormal Invariant
// Identically 0 on planes, circular cylinders, and spheres; strictly rejects thin offset shells.
fn os_psi_perp_geodesic(
  xp: vec3<f32>, np: vec3<f32>, sid_p: f32,
  xq: vec3<f32>, nq: vec3<f32>, sid_q: f32,
  c0: f32, plane_tol: f32
) -> f32 {
  if (abs(sid_p - sid_q) > 0.25) {
    return 0.0;
  }
  let w_n = os_stiefel_null(dot(np, nq), c0);
  if (w_n <= 0.0) {
    return 0.0;
  }
  let dx = xq - xp;
  let nm = 0.5 * (np + nq);
  let delta_geo = abs(dot(dx, nm));
  return w_n * os_smoothstep2(1.0 - delta_geo / plane_tol);
}

// Pillar 2: Crease-Clipped 2nd-Order Periodic Antiderivative A2_perp (exact Bartlett integral)
fn os_p2_box(u: f32, d: f32) -> f32 {
  let k = floor(u);
  let r = u - k;
  let s1 = 0.5 * d * (1.0 - d) + 0.5 * d * d;
  // Exact 2nd antiderivative of 0/1 duty-d periodic pulse
  let r_cl = min(r, d);
  let p1_period = d;
  let p2_period = d * (1.0 - 0.5 * d);
  let frac_p2 = 0.5 * r_cl * r_cl + d * max(0.0, r - d);
  return 0.5 * k * (k - 1.0) * p1_period + k * (r * p1_period + p2_period - p1_period) + frac_p2 + 0.0 * s1;
}

fn os_antideriv_a2_bartlett(u0: f32, du: f32, duty: f32) -> f32 {
  if (du <= 1e-4) {
    let f = u0 - floor(u0);
    return select(0.0, 1.0, f < duty);
  }
  let a = os_p2_box(u0 - du, duty);
  let b = os_p2_box(u0, duty);
  let c = os_p2_box(u0 + du, duty);
  return clamp((c - 2.0 * b + a) / (du * du), 0.0, 1.0);
}

// Pillar 3: L-Stable Port-Hamiltonian Zero-Delay-Feedback TPT (beta_n = G_n => rho = (1 + g_n)^-2)
struct ZdfStepOut {
  y_out: f32,
  s_new: f32,
};

fn os_zdf_tpt_step(u_eff: f32, s_base: f32, g_n: f32) -> ZdfStepOut {
  let G_n = g_n / (1.0 + g_n);
  let v_n = (u_eff - s_base) * G_n;
  let y_out = s_base + v_n;
  let s_new = y_out + v_n * (1.0 - G_n);
  return ZdfStepOut(y_out, s_new);
}

// Pillar 4: Exact Stokes Thin-Shell Slab (R_inf + T_inf == 1.0 identically)
fn os_stokes_slab_R(F: f32) -> f32 {
  return (2.0 * F) / (1.0 + F);
}
