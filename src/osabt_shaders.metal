/*
 * Slice 3: Apple M1 Metal Compute Kernels for OS-ABT (v4 Architecture)
 * Concatenated directly with src/osabt_core.h at runtime (strict IEEE-754 parity).
 * - Single packed 8B/px half4 G-buffer: (oct_x, oct_y, z_lin, sid)
 * - Per-scene entry kernels avoiding mega-kernel register spill
 * - Hardware threadgroup SRAM tiling for separable H/V (Scene A) & 2A/2B (Scene C)
 * - Hardware simd_sum fused 16x16 affine-gauge tile reduction inside Pass 2A
 */

/* Ground-Truth 256-spp Bartlett kernel */
kernel void k_pass1_gt(
    constant FrameUniforms& u [[buffer(0)]],
    device v4* out_rgb        [[buffer(1)]],
    device v4* out_gn         [[buffer(2)]],
    device v4* out_gp         [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  Pass1Out p1 = os_pass1_pixel(&u, px, py);
  out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
  if (u.pad0 != 0) {
    out_gn[idx] = p1.gbuf_n;
    out_gp[idx] = p1.gbuf_p;
  }
}

/* Direct analytical Scene B kernels (skips Pass 2 fairly across all modes) */
kernel void k_pass1_scene_b_osabt(
    constant FrameUniforms& u [[buffer(0)]],
    device v4* out_rgb        [[buffer(1)]],
    device v4* out_gn         [[buffer(2)]],
    device v4* out_gp         [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_b_ray(&u, ro, rd_c, fx_c, fy_c, MODE_OSABT);
  out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
  if (u.pad0 != 0) {
    out_gn[idx] = p1.gbuf_n;
    out_gp[idx] = p1.gbuf_p;
  }
}

kernel void k_pass1_scene_b(
    constant FrameUniforms& u [[buffer(0)]],
    device v4* out_rgb        [[buffer(1)]],
    device v4* out_gn         [[buffer(2)]],
    device v4* out_gp         [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_b_ray(&u, ro, rd_c, fx_c, fy_c, u.mode);
  out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
  if (u.pad0 != 0) {
    out_gn[idx] = p1.gbuf_n;
    out_gp[idx] = p1.gbuf_p;
  }
}

/* Direct analytical Scene D kernel (skips Pass 2 fairly across all modes) */
kernel void k_pass1_scene_d(
    constant FrameUniforms& u [[buffer(0)]],
    device v4* out_rgb        [[buffer(1)]],
    device v4* out_gn         [[buffer(2)]],
    device v4* out_gp         [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_d_ray(&u, ro, rd_c, fx_c, fy_c, u.mode);
  out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
  if (u.pad0 != 0) {
    out_gn[idx] = p1.gbuf_n;
    out_gp[idx] = p1.gbuf_p;
  }
}

/* Packed 8B G-buffer Pass 1 for Scene A (specialized OS-ABT + generic modes) */
kernel void k_pass1_scene_a_osabt(
    constant FrameUniforms& u [[buffer(0)]],
    device h4* cur_irrad      [[buffer(1)]],
    device h4* cur_direct     [[buffer(2)]],
    device u16_4* cur_albedo  [[buffer(3)]],
    device h4* cur_gbuf       [[buffer(4)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_a_ray(&u, ro, rd_c, fx_c, fy_c, MODE_OSABT, 0);
  cur_irrad[idx]  = v4_to_h4(p1.irrad);
  cur_direct[idx] = v4_to_h4(p1.direct);
  cur_albedo[idx] = v4_to_u16(p1.albedo);
  cur_gbuf[idx]   = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);
}

kernel void k_pass1_scene_a(
    constant FrameUniforms& u [[buffer(0)]],
    device h4* cur_irrad      [[buffer(1)]],
    device h4* cur_direct     [[buffer(2)]],
    device u16_4* cur_albedo  [[buffer(3)]],
    device h4* cur_gbuf       [[buffer(4)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_a_ray(&u, ro, rd_c, fx_c, fy_c, u.mode, 0);
  cur_irrad[idx]  = v4_to_h4(p1.irrad);
  cur_albedo[idx] = v4_to_u16(p1.albedo);
  cur_gbuf[idx]   = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);
  if (u.mode == MODE_OSABT) {
    cur_direct[idx] = v4_to_h4(p1.direct);
  }
}

/* Packed 8B G-buffer Pass 1 for Scene C (1-spp area shadow + 1-bounce GI) */
kernel void k_pass1_scene_c(
    constant FrameUniforms& u [[buffer(0)]],
    device h4* cur_irrad      [[buffer(1)]],
    device u16_4* cur_albedo  [[buffer(2)]],
    device h4* cur_gbuf       [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);
  Pass1Out p1 = os_eval_scene_c_ray(&u, ro, rd_c, fx_c, fy_c, u.mode, 0);
  cur_irrad[idx]  = v4_to_h4(p1.irrad);
  cur_albedo[idx] = v4_to_u16(p1.albedo);
  cur_gbuf[idx]   = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);
}

/* Packed 8B G-buffer Pass 1 for the showcase scenes E/F/G (1-spp rect-light shadow + 1-bounce GI).
 * One entry kernel per scene so the scene dispatch constant-folds away. */
#define OS_SHOW_PASS1_KERNEL(NAME, SCENE)                                             \
kernel void NAME(                                                                     \
    constant FrameUniforms& u [[buffer(0)]],                                          \
    device h4* cur_irrad      [[buffer(1)]],                                          \
    device u16_4* cur_albedo  [[buffer(2)]],                                          \
    device h4* cur_gbuf       [[buffer(3)]],                                          \
    uint2 gid                 [[thread_position_in_grid]])                            \
{                                                                                     \
  int px = (int)gid.x;                                                                \
  int py = (int)gid.y;                                                                \
  if (px >= u.width || py >= u.height) return;                                        \
  int idx = py * u.width + px;                                                        \
  v3 ro = mk3(u.cam_eye[0], u.cam_eye[1], u.cam_eye[2]);                              \
  float fx_c = (float)px + 0.5f;                                                      \
  float fy_c = (float)py + 0.5f;                                                      \
  v3 rd_c = os_camera_ray(&u, fx_c, fy_c);                                            \
  Pass1Out p1 = os_eval_show_ray(&u, ro, rd_c, fx_c, fy_c, u.mode, 0, SCENE);         \
  cur_irrad[idx]  = v4_to_h4(p1.irrad);                                               \
  cur_albedo[idx] = v4_to_u16(p1.albedo);                                             \
  cur_gbuf[idx]   = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);                        \
}
OS_SHOW_PASS1_KERNEL(k_pass1_scene_e, SCENE_E)
OS_SHOW_PASS1_KERNEL(k_pass1_scene_f, SCENE_F)
OS_SHOW_PASS1_KERNEL(k_pass1_scene_g, SCENE_G)

/* Scene H "Sponza" (triangle mesh through a BVH): 1-spp Pass 1 into the packed G-buffer */
kernel void k_pass1_scene_h(
    constant FrameUniforms& u [[buffer(0)]],
    device h4* cur_irrad      [[buffer(1)]],
    device u16_4* cur_albedo  [[buffer(2)]],
    device h4* cur_gbuf       [[buffer(3)]],
    const device v4* nodes    [[buffer(4)]],
    const device v4* tri_a    [[buffer(5)]],
    const device v4* tri_n    [[buffer(6)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int idx = py * u.width + px;
  Pass1Out p1 = os_pass1_pixel_mesh(&u, px, py, nodes, tri_a, tri_n);
  cur_irrad[idx]  = v4_to_h4(p1.irrad);
  cur_albedo[idx] = v4_to_u16(p1.albedo);
  cur_gbuf[idx]   = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);
}

/* Scene H ground truth. Rendered one horizontal band per command buffer (rows band.x .. band.y-1)
 * so a 64-spp frame of BVH rays never becomes one multi-second GPU submission. */
kernel void k_pass1_gt_mesh(
    constant FrameUniforms& u [[buffer(0)]],
    device v4* out_rgb        [[buffer(1)]],
    device v4* out_gn         [[buffer(2)]],
    device v4* out_gp         [[buffer(3)]],
    const device v4* nodes    [[buffer(4)]],
    const device v4* tri_a    [[buffer(5)]],
    const device v4* tri_n    [[buffer(6)]],
    constant int4& band       [[buffer(7)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y + band.x;
  if (px >= u.width || py >= u.height || py >= band.y) return;
  int idx = py * u.width + px;
  Pass1Out p1 = os_pass1_pixel_mesh(&u, px, py, nodes, tri_a, tri_n);
  out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
  if (u.pad0 != 0) {
    out_gn[idx] = p1.gbuf_n;
    out_gp[idx] = p1.gbuf_p;
  }
}

/* Scene A OS-ABT Pass 2H: Threadgroup-tiled 32x8 Horizontal Psi_perp Indirect Filter
 * (6 KB on-chip SRAM tile -> zero DRAM reads inside the 16-tap stencil loop!) */
kernel void k_pass2_osabt_spatial_h(
    constant FrameUniforms& u  [[buffer(0)]],
    const device h4* src_irrad [[buffer(1)]],
    const device h4* cur_gbuf  [[buffer(2)]],
    device h4* dst_irrad       [[buffer(3)]],
    uint2 tgp                  [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]])
{
  threadgroup h4 sh_irr[8][48];
  threadgroup h4 sh_gb[8][48];

  int W = u.width, H = u.height;
  int bx = (int)tgp.x * 32;
  int by = (int)tgp.y * 8;
  int lx = (int)lid.x;
  int ly = (int)lid.y;
  int gy = by + ly;

  int gx0 = bx + lx - 8;
  if (gy < H && gx0 >= 0 && gx0 < W) {
    int gidx = gy * W + gx0;
    sh_irr[ly][lx] = src_irrad[gidx];
    sh_gb[ly][lx]  = cur_gbuf[gidx];
  } else {
    sh_irr[ly][lx] = h4(0.0h);
    sh_gb[ly][lx]  = h4(0.0h);
  }
  if (lx < 16) {
    int gx1 = bx + lx + 24;
    if (gy < H && gx1 >= 0 && gx1 < W) {
      int gidx = gy * W + gx1;
      sh_irr[ly][lx + 32] = src_irrad[gidx];
      sh_gb[ly][lx + 32]  = cur_gbuf[gidx];
    } else {
      sh_irr[ly][lx + 32] = h4(0.0h);
      sh_gb[ly][lx + 32]  = h4(0.0h);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int px = bx + lx;
  int py = gy;
  if (px >= W || py >= H) return;
  int idx = py * W + px;

  h4 ci_h = sh_irr[ly][lx + 8];
  v4 ci   = h4_to_v4(ci_h);
  h4 gp_h = sh_gb[ly][lx + 8];
  v4 gp   = h4_to_v4(gp_h);
  float sid_p = gp.w;
  if (ci.w < 0.5f || sid_p < 0.5f) {
    dst_irrad[idx] = ci_h;
    return;
  }

  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp = os_camera_ray_unnorm(&u, (float)px + 0.5f, (float)py + 0.5f);
  float step_s = (2.0f * u.cam_right[3] * u.cam_fwd[3]) / (float)W;
  float rp_dot_np = v3_dot(rp, np);
  float dR_dot_np = (u.cam_right[0] * np.x + u.cam_right[1] * np.y + u.cam_right[2] * np.z) * step_s;
  float p_plane   = rp_dot_np * zp;

  v3 acc = mk3(ci.x, ci.y, ci.z);
  float w_sum = 1.0f;
  const float inv_tol = 16.666666f;
  for (int d = -8; d <= 8; ++d) {
    if (d == 0) continue;
    int sx = lx + 8 + d;
    h4 gqh = sh_gb[ly][sx];
    if (abs(gp_h.w - gqh.w) > 0.25h) continue;
    float zq = (float)gqh.z;
    float psi;
    if (abs(gqh.x - gp_h.x) + abs(gqh.y - gp_h.y) < 1e-4h) {
      float d_geo = f_abs(fma(zq, fma((float)d, dR_dot_np, rp_dot_np), -p_plane));
      float q = d_geo * inv_tol;
      if (q >= 1.0f) continue;
      float t = 1.0f - q * q;
      psi = t * t;
    } else {
      v3 nq = os_oct_decode((float)gqh.x, (float)gqh.y);
      float w_n = os_stiefel_null(v3_dot(np, nq));
      if (w_n <= 0.0f) continue;
      v3 dR = mk3(u.cam_right[0] * step_s, u.cam_right[1] * step_s, u.cam_right[2] * step_s);
      v3 dx = v3_add(v3_scale(rp, zq - zp), v3_scale(dR, zq * (float)d));
      psi = os_psi_perp_plane(dx, np, nq, w_n, inv_tol);
      if (psi <= 0.0f) continue;
    }
    float w = os_spatial_w(d < 0 ? -d : d) * psi;
    v4 qi = h4_to_v4(sh_irr[ly][sx]);
    acc = v3_add(acc, v3_scale(mk3(qi.x, qi.y, qi.z), w));
    w_sum += w;
  }
  v3 filt = v3_scale(acc, 1.0f / f_max(w_sum, 1e-6f));
  dst_irrad[idx] = v4_to_h4(mk4(filt.x, filt.y, filt.z, 1.0f));
}

/* Scene A OS-ABT Pass 2V: Threadgroup-tiled 16x16 Vertical Psi_perp Indirect Filter
 * (8 KB on-chip SRAM tile, 100% 128B-coalesced loads, zero DRAM reads in loop!) */
kernel void k_pass2_osabt_spatial_v(
    constant FrameUniforms& u  [[buffer(0)]],
    const device h4* src_irrad [[buffer(1)]],
    const device h4* cur_dir   [[buffer(2)]],
    const device u16_4* cur_alb [[buffer(3)]],
    const device h4* cur_gbuf  [[buffer(4)]],
    device v4* out_rgb         [[buffer(5)]],
    uint2 tgp                  [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]])
{
  threadgroup h4 sh_irr[32][16];
  threadgroup h4 sh_gb[32][16];

  int W = u.width, H = u.height;
  int bx = (int)tgp.x * 16;
  int by = (int)tgp.y * 16;
  int lx = (int)lid.x;
  int ly = (int)lid.y;
  int px = bx + lx;

  int gy0 = by + ly - 8;
  if (px < W && gy0 >= 0 && gy0 < H) {
    int gidx = gy0 * W + px;
    sh_irr[ly][lx] = src_irrad[gidx];
    sh_gb[ly][lx]  = cur_gbuf[gidx];
  } else {
    sh_irr[ly][lx] = h4(0.0h);
    sh_gb[ly][lx]  = h4(0.0h);
  }
  int gy1 = by + ly + 8;
  if (px < W && gy1 >= 0 && gy1 < H) {
    int gidx = gy1 * W + px;
    sh_irr[ly + 16][lx] = src_irrad[gidx];
    sh_gb[ly + 16][lx]  = cur_gbuf[gidx];
  } else {
    sh_irr[ly + 16][lx] = h4(0.0h);
    sh_gb[ly + 16][lx]  = h4(0.0h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int py = by + ly;
  if (px >= W || py >= H) return;
  int idx = py * W + px;

  v4 ci = h4_to_v4(sh_irr[ly + 8][lx]);
  h4 gp_h = sh_gb[ly + 8][lx];
  v4 gp = h4_to_v4(gp_h);
  float sid_p = gp.w;
  if (ci.w < 0.5f || sid_p < 0.5f) {
    out_rgb[idx] = mk4(ci.x, ci.y, ci.z, 1.0f);
    return;
  }

  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp = os_camera_ray_unnorm(&u, (float)px + 0.5f, (float)py + 0.5f);
  float step_s = (-2.0f * u.cam_fwd[3]) / (float)H;
  float rp_dot_np = v3_dot(rp, np);
  float dR_dot_np = (u.cam_up[0] * np.x + u.cam_up[1] * np.y + u.cam_up[2] * np.z) * step_s;
  float p_plane   = rp_dot_np * zp;

  v3 acc = mk3(ci.x, ci.y, ci.z);
  float w_sum = 1.0f;
  const float inv_tol = 16.666666f;
  for (int d = -8; d <= 8; ++d) {
    if (d == 0) continue;
    int sy = ly + 8 + d;
    h4 gqh = sh_gb[sy][lx];
    if (abs(gp_h.w - gqh.w) > 0.25h) continue;
    float zq = (float)gqh.z;
    float psi;
    if (abs(gqh.x - gp_h.x) + abs(gqh.y - gp_h.y) < 1e-4h) {
      float d_geo = f_abs(fma(zq, fma((float)d, dR_dot_np, rp_dot_np), -p_plane));
      float q = d_geo * inv_tol;
      if (q >= 1.0f) continue;
      float t = 1.0f - q * q;
      psi = t * t;
    } else {
      v3 nq = os_oct_decode((float)gqh.x, (float)gqh.y);
      float w_n = os_stiefel_null(v3_dot(np, nq));
      if (w_n <= 0.0f) continue;
      v3 dR = mk3(u.cam_up[0] * step_s, u.cam_up[1] * step_s, u.cam_up[2] * step_s);
      v3 dx = v3_add(v3_scale(rp, zq - zp), v3_scale(dR, zq * (float)d));
      psi = os_psi_perp_plane(dx, np, nq, w_n, inv_tol);
      if (psi <= 0.0f) continue;
    }
    float w = os_spatial_w(d < 0 ? -d : d) * psi;
    v4 qi = h4_to_v4(sh_irr[sy][lx]);
    acc = v3_add(acc, v3_scale(mk3(qi.x, qi.y, qi.z), w));
    w_sum += w;
  }
  v3 filt = v3_scale(acc, 1.0f / f_max(w_sum, 1e-6f));
  v4 cd = h4_to_v4(cur_dir[idx]);
  v4 ca = u16_to_v4(cur_alb[idx]);
  v3 col = v3_add(mk3(cd.x, cd.y, cd.z), v3_mul(mk3(ca.x, ca.y, ca.z), filt));
  out_rgb[idx] = mk4(col.x, col.y, col.z, 1.0f);
}

/* Scene C OS-ABT Pass 2A + FUSED Pass 2G1:
 * Threadgroup-tiled 18x18 SRAM 3x3 cell moments + 8B geodesic reprojection +
 * hardware simd_sum 16x16 affine-gauge tile reduction (eliminates Pass 2G1!). */
kernel void k_pass2a_osabt_reproj(
    constant FrameUniforms& u  [[buffer(0)]],
    const device h4* cur_irrad [[buffer(1)]],
    const device h4* cur_gbuf  [[buffer(2)]],
    const device st4* prev_st  [[buffer(3)]],
    const device h4* prev_gbuf [[buffer(4)]],
    device rp4* out_reproj     [[buffer(5)]],
    device h4* out_sp1         [[buffer(6)]],
    device v4* tile_su         [[buffer(7)]],
    device v4* tile_ss         [[buffer(8)]],
    device v4* tile_ss2        [[buffer(9)]],
    device v4* tile_sus        [[buffer(10)]],
    uint2 tgp                  [[threadgroup_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint num_simds             [[simdgroups_per_threadgroup]])
{
  threadgroup h4 sh_irr[18 * 18];
  threadgroup h4 sh_gb[18 * 18];
  threadgroup float sh_red[8 * 14];

  int W = u.width, H = u.height;
  int bx = (int)tgp.x * 16;
  int by = (int)tgp.y * 16;

  for (int s = (int)tid; s < 324; s += 256) {
    int sy = (int)(((uint)s * 3641u) >> 16);
    int sx = s - sy * 18;
    int gx = bx + sx - 1;
    int gy = by + sy - 1;
    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
      int gidx = gy * W + gx;
      sh_irr[s] = cur_irrad[gidx];
      sh_gb[s]  = cur_gbuf[gidx];
    } else {
      sh_irr[s] = h4(0.0h);
      sh_gb[s]  = h4(0.0h);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int lx = (int)lid.x;
  int ly = (int)lid.y;
  int px = bx + lx;
  int py = by + ly;

  v4 v_rep_out = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  h4 h_sp1_out = h4(0.0h);
  int valid_red = 0;

  if (px < W && py < H) {
    int idx = py * W + px;
    int cslot = (ly + 1) * 18 + (lx + 1);
    v4 ci = h4_to_v4(sh_irr[cslot]);
    h4 gp_h = sh_gb[cslot];
    v4 gp = h4_to_v4(gp_h);
    float sid_p = gp.w;

    if (ci.w < 0.5f || sid_p < 0.5f) {
      v4 zero_v = mk4(ci.x, ci.y, ci.z, 0.0f);
      out_reproj[idx] = rp4_store(zero_v);
      out_sp1[idx]    = v4_to_h4(zero_v);
    } else {
      v3 np = os_oct_decode(gp.x, gp.y);
      float zp = gp.z;
      v3 rp_ray = os_camera_ray_unnorm(&u, (float)px + 0.5f, (float)py + 0.5f);
      float rp_dot_np = v3_dot(rp_ray, np);

      /* 1. Single cache-coherent bilinear geodesic Psi_perp-gated history reprojection */
      v3 s_prev = mk3(0.0f, 0.0f, 0.0f);
      float prev_age = 0.0f;
      if (u.frame_idx > 0 && u.total_frames > 1) {
        const float inv_tol_rep = 12.5f;
        v3 xp = mk3(u.cam_eye[0] + rp_ray.x * zp,
                    u.cam_eye[1] + rp_ray.y * zp,
                    u.cam_eye[2] + rp_ray.z * zp);
        v2 rp = os_reproject_prev(&u, xp);
        int x0 = (int)f_floor(rp.x);
        int y0 = (int)f_floor(rp.y);
        float fx = rp.x - (float)x0;
        float fy = rp.y - (float)y0;
        float psx = (2.0f * u.prev_right[3] * u.prev_fwd[3]) / (float)W;
        float psy = (-2.0f * u.prev_fwd[3]) / (float)H;
        float pe_dot_np   = (u.prev_eye[0] - xp.x) * np.x + (u.prev_eye[1] - xp.y) * np.y + (u.prev_eye[2] - xp.z) * np.z;
        float pr00_dot_np = v3_dot(os_prev_camera_ray_unnorm(&u, (float)x0 + 0.5f, (float)y0 + 0.5f), np);
        float pdrx_dot_np = (u.prev_right[0] * np.x + u.prev_right[1] * np.y + u.prev_right[2] * np.z) * psx;
        float pdry_dot_np = (u.prev_up[0] * np.x    + u.prev_up[1] * np.y    + u.prev_up[2] * np.z) * psy;
        float rw_sum = 0.0f;
        for (int b = 0; b < 4; ++b) {
          int ox = b & 1;
          int oy = b >> 1;
          float bwt = (ox ? fx : (1.0f - fx)) * (oy ? fy : (1.0f - fy));
          if (bwt < 1e-4f) continue;
          int rx = x0 + ox;
          int ry = y0 + oy;
          if ((uint)rx >= (uint)W || (uint)ry >= (uint)H) continue;
          int ridx = ry * W + rx;
          h4 pgh = prev_gbuf[ridx];
          if (abs(gp_h.w - pgh.w) > 0.25h) continue;
          float pgz = (float)pgh.z;
          float rpsi;
          if (abs(pgh.x - gp_h.x) + abs(pgh.y - gp_h.y) < 1e-4h) {
            float r_dot = pr00_dot_np + (ox ? pdrx_dot_np : 0.0f) + (oy ? pdry_dot_np : 0.0f);
            float d_geo = f_abs(fma(pgz, r_dot, pe_dot_np));
            float q = d_geo * inv_tol_rep;
            if (q >= 1.0f) continue;
            float t = 1.0f - q * q;
            rpsi = t * t;
          } else {
            v3 pn_p = os_oct_decode((float)pgh.x, (float)pgh.y);
            float w_n = os_stiefel_null(v3_dot(np, pn_p));
            if (w_n <= 0.0f) continue;
            v3 dpos_p = v3_sub(os_pos_from_z_prev(&u, rx, ry, pgz), xp);
            rpsi = os_psi_perp_plane(dpos_p, np, pn_p, w_n, inv_tol_rep);
          }
          float w = bwt * rpsi;
          if (w > 1e-5f) {
            v4 ps4 = st4_load(prev_st[ridx]);
            s_prev = v3_add(s_prev, v3_scale(mk3(ps4.x, ps4.y, ps4.z), w));
            prev_age += ps4.w * w;
            rw_sum += w;
          }
        }
        if (rw_sum >= 0.15f) {
          float inv_rw = 1.0f / rw_sum;
          s_prev = v3_scale(s_prev, inv_rw);
          prev_age = f_max(1.0f, prev_age * inv_rw);
        } else {
          prev_age = 0.0f;
        }
      }
      rp4 rep_st = rp4_store(mk4(s_prev.x, s_prev.y, s_prev.z, prev_age));
      out_reproj[idx] = rep_st;
      v_rep_out = rp4_load(rep_st);  /* tile reduction sees exactly what Pass 2B will load */

      /* 2. 3x3 Stride-1 Co-Surface Psi_perp Cell Mean & Exact Kish Cell Variance from SRAM */
      float sx = (2.0f * u.cam_right[3] * u.cam_fwd[3]) / (float)W;
      float sy = (-2.0f * u.cam_fwd[3]) / (float)H;
      float dRx_dot_np = (u.cam_right[0] * np.x + u.cam_right[1] * np.y + u.cam_right[2] * np.z) * sx;
      float dRy_dot_np = (u.cam_up[0] * np.x    + u.cam_up[1] * np.y    + u.cam_up[2] * np.z) * sy;
      float p_plane    = rp_dot_np * zp;

      const float kw[3] = { 0.80f, 1.0f, 0.80f };
      const float inv_tol = 16.666666f;
      v3 uc = mk3(ci.x, ci.y, ci.z);
      v3 m1 = uc;
      v3 m2 = v3_mul(uc, uc);
      float w_sum = 1.0f, w2_sum = 1.0f;
      for (int dy = -1; dy <= 1; ++dy) {
        int srow = (ly + 1 + dy) * 18 + (lx + 1);
        float wy = kw[dy + 1];
        float r_dot_y = fma((float)dy, dRy_dot_np, rp_dot_np);
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          int slot = srow + dx;
          h4 gqh = sh_gb[slot];
          if (abs(gp_h.w - gqh.w) > 0.25h) continue;
          float zq = (float)gqh.z;
          float psi;
          if (abs(gqh.x - gp_h.x) + abs(gqh.y - gp_h.y) < 1e-4h) {
            float d_geo = f_abs(fma(zq, fma((float)dx, dRx_dot_np, r_dot_y), -p_plane));
            float q = d_geo * inv_tol;
            if (q >= 1.0f) continue;
            float t = 1.0f - q * q;
            psi = t * t;
          } else {
            v3 nq = os_oct_decode((float)gqh.x, (float)gqh.y);
            float w_n = os_stiefel_null(v3_dot(np, nq));
            if (w_n <= 0.0f) continue;
            v3 dRx = mk3(u.cam_right[0] * sx, u.cam_right[1] * sx, u.cam_right[2] * sx);
            v3 dRy = mk3(u.cam_up[0] * sy,    u.cam_up[1] * sy,    u.cam_up[2] * sy);
            v3 dpos = v3_add(v3_scale(rp_ray, zq - zp),
                             v3_scale(v3_add(v3_scale(dRy, (float)dy), v3_scale(dRx, (float)dx)), zq));
            psi = os_psi_perp_plane(dpos, np, nq, w_n, inv_tol);
            if (psi <= 0.0f) continue;
          }
          float w = wy * kw[dx + 1] * psi;
          v4 qi = h4_to_v4(sh_irr[slot]);
          v3 qu = mk3(qi.x, qi.y, qi.z);
          m1 = v3_add(m1, v3_scale(qu, w));
          m2 = v3_add(m2, v3_scale(v3_mul(qu, qu), w));
          w_sum += w;
          w2_sum += w * w;
        }
      }
      float inv_w = 1.0f / f_max(w_sum, 1e-6f);
      v3 u_sp1 = v3_scale(m1, inv_w);
      v3 v3c = v3_max(mk3(0.0f, 0.0f, 0.0f), v3_sub(v3_scale(m2, inv_w), v3_mul(u_sp1, u_sp1)));
      float var_3ch = (v3c.x + v3c.y) + v3c.z;
      float m_eff1 = (w_sum * w_sum) / f_max(w2_sum, 1e-6f);
      float std_cell = f_sqrt(var_3ch / f_max(m_eff1, 1.0f));
      h_sp1_out = v4_to_h4(mk4(u_sp1.x, u_sp1.y, u_sp1.z, std_cell));
      out_sp1[idx] = h_sp1_out;
      if (((lid.x & 1u) == 0u) && ((lid.y & 1u) == 0u)) valid_red = 1;
    }
  }

  int TW = (W + 15) / 16;
  int tidx = (int)tgp.y * TW + (int)tgp.x;
  if (u.frame_idx == 0 || u.total_frames <= 1) {
    if (tid == 0u) {
      tile_su[tidx]  = mk4(0.0f, 0.0f, 0.0f, 0.0f);
      tile_ss[tidx]  = mk4(0.0f, 0.0f, 0.0f, 0.0f);
      tile_ss2[tidx] = mk4(0.0f, 0.0f, 0.0f, 0.0f);
      tile_sus[tidx] = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return;
  }

  /* 3. Fused 2G1 tile reduction staged in 4-float chunks -> minimal register pressure! */
  float wa = 0.0f;
  v4 rep = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  v4 sp  = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  if (valid_red) {
    rep = v_rep_out;
    if (rep.w >= 0.5f) {
      sp = h4_to_v4(h_sp1_out);
      wa = f_min(1.0f, rep.w * 0.25f);
    }
  }
  uint b = simd_id * 14u;
  {
    float s0 = simd_sum(sp.x * wa);
    float s1 = simd_sum(sp.y * wa);
    float s2 = simd_sum(sp.z * wa);
    float s3 = simd_sum(wa);
    if (simd_lane == 0u) { sh_red[b + 0u] = s0; sh_red[b + 1u] = s1; sh_red[b + 2u] = s2; sh_red[b + 3u] = s3; }
  }
  {
    float s0 = simd_sum(rep.x * wa);
    float s1 = simd_sum(rep.y * wa);
    float s2 = simd_sum(rep.z * wa);
    float s3 = simd_sum((wa * wa) * (sp.w * sp.w));
    if (simd_lane == 0u) { sh_red[b + 4u] = s0; sh_red[b + 5u] = s1; sh_red[b + 6u] = s2; sh_red[b + 7u] = s3; }
  }
  {
    float s0 = simd_sum(rep.x * rep.x * wa);
    float s1 = simd_sum(rep.y * rep.y * wa);
    float s2 = simd_sum(rep.z * rep.z * wa);
    if (simd_lane == 0u) { sh_red[b + 8u] = s0; sh_red[b + 9u] = s1; sh_red[b + 10u] = s2; }
  }
  {
    float s0 = simd_sum(sp.x * rep.x * wa);
    float s1 = simd_sum(sp.y * rep.y * wa);
    float s2 = simd_sum(sp.z * rep.z * wa);
    if (simd_lane == 0u) { sh_red[b + 11u] = s0; sh_red[b + 12u] = s1; sh_red[b + 13u] = s2; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float acc[14];
    for (int k = 0; k < 14; ++k) {
      float s = 0.0f;
      for (uint g = 0u; g < num_simds; ++g) s += sh_red[g * 14u + (uint)k];
      acc[k] = s;
    }
    tile_su[tidx]  = mk4(acc[0],  acc[1],  acc[2],  acc[3]);
    tile_ss[tidx]  = mk4(acc[4],  acc[5],  acc[6],  acc[7]);
    tile_ss2[tidx] = mk4(acc[8],  acc[9],  acc[10], 0.0f);
    tile_sus[tidx] = mk4(acc[11], acc[12], acc[13], 0.0f);
  }
}

/* Scene C OS-ABT Pass 2G2: Regional ridge-regularized affine gauge solver (L1-resident) */
kernel void k_pass2g2_tile_gauge(
    constant FrameUniforms& u  [[buffer(0)]],
    const device v4* tile_su   [[buffer(1)]],
    const device v4* tile_ss   [[buffer(2)]],
    const device v4* tile_ss2  [[buffer(3)]],
    const device v4* tile_sus  [[buffer(4)]],
    device v4* tile_kap        [[buffer(5)]],
    device v4* tile_bet        [[buffer(6)]],
    uint2 gid                  [[thread_position_in_grid]])
{
  int TW = (u.width + 15) / 16;
  int TH = (u.height + 15) / 16;
  int tx = (int)gid.x;
  int ty = (int)gid.y;
  if (tx >= TW || ty >= TH) return;
  os_pass2g2_tile_gauge(&u, tx, ty, tile_su, tile_ss, tile_ss2, tile_sus, tile_kap, tile_bet);
}

/* Scene C OS-ABT Pass 2B: Threadgroup-tiled 22x22 SRAM Stride-3 Disjoint Hierarchy +
 * Bilinear Affine Gauge + L-Stable Port-Hamiltonian ZDF-TPT! */
kernel void k_pass2b_osabt_zdf(
    constant FrameUniforms& u   [[buffer(0)]],
    const device u16_4* cur_alb [[buffer(1)]],
    const device h4* cur_gbuf   [[buffer(2)]],
    const device rp4* tmp_reproj [[buffer(3)]],
    const device h4* tmp_sp1    [[buffer(4)]],
    const device v4* tile_kap   [[buffer(5)]],
    const device v4* tile_bet   [[buffer(6)]],
    device st4* out_st          [[buffer(7)]],
    device v4* out_rgb          [[buffer(8)]],
    uint2 tgp                   [[threadgroup_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint tid                    [[thread_index_in_threadgroup]])
{
  threadgroup h4 sh_gb[22 * 22];
  threadgroup h4 sh_sp1[22 * 22];
  threadgroup v4 sh_kap[9];
  threadgroup v4 sh_bet[9];

  int W = u.width, H = u.height;
  int TW = (W + 15) / 16;
  int TH = (H + 15) / 16;
  int bx = (int)tgp.x * 16;
  int by = (int)tgp.y * 16;

  for (int s = (int)tid; s < 484; s += 256) {
    int sy = (int)(((uint)s * 2979u) >> 16);
    int sx = s - sy * 22;
    int gx = bx + sx - 3;
    int gy = by + sy - 3;
    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
      int gidx = gy * W + gx;
      sh_gb[s]  = cur_gbuf[gidx];
      sh_sp1[s] = tmp_sp1[gidx];
    } else {
      sh_gb[s]  = h4(0.0h);
      sh_sp1[s] = h4(0.0h);
    }
  }
  if (tid < 9u) {
    int dy = (int)(tid / 3u) - 1;
    int dx = (int)(tid % 3u) - 1;
    int tx = (int)tgp.x + dx;
    int ty = (int)tgp.y + dy;
    tx = (tx < 0) ? 0 : ((tx >= TW) ? (TW - 1) : tx);
    ty = (ty < 0) ? 0 : ((ty >= TH) ? (TH - 1) : ty);
    int ti = ty * TW + tx;
    sh_kap[tid] = tile_kap[ti];
    sh_bet[tid] = tile_bet[ti];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int lx = (int)lid.x;
  int ly = (int)lid.y;
  int px = bx + lx;
  int py = by + ly;
  if (px >= W || py >= H) return;
  int idx = py * W + px;

  int cslot = (ly + 3) * 22 + (lx + 3);
  v4 ca = u16_to_v4(cur_alb[idx]);
  h4 gp_h = sh_gb[cslot];
  v4 gp = h4_to_v4(gp_h);
  float sid_p = gp.w;
  v4 sp1_c = h4_to_v4(sh_sp1[cslot]);
  if (ca.w < 0.5f || sid_p < 0.5f) {
    v3 col = mk3(sp1_c.x * ca.x, sp1_c.y * ca.y, sp1_c.z * ca.z);
    out_st[idx]  = st4_store(mk4(sp1_c.x, sp1_c.y, sp1_c.z, 1.0f));
    out_rgb[idx] = mk4(col.x, col.y, col.z, 1.0f);
    return;
  }

  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp_ray = os_camera_ray_unnorm(&u, (float)px + 0.5f, (float)py + 0.5f);
  float sx3 = (6.0f * u.cam_right[3] * u.cam_fwd[3]) / (float)W;
  float sy3 = (-6.0f * u.cam_fwd[3]) / (float)H;
  float rp_dot_np   = v3_dot(rp_ray, np);
  float dRx3_dot_np = (u.cam_right[0] * np.x + u.cam_right[1] * np.y + u.cam_right[2] * np.z) * sx3;
  float dRy3_dot_np = (u.cam_up[0] * np.x    + u.cam_up[1] * np.y    + u.cam_up[2] * np.z) * sy3;
  float p_plane     = rp_dot_np * zp;

  v3 alb = mk3(ca.x, ca.y, ca.z);
  v4 rep_c = rp4_load(tmp_reproj[idx]);
  v3 s_prev = mk3(rep_c.x, rep_c.y, rep_c.z);
  float prev_age = rep_c.w;
  v3 u_sp1 = mk3(sp1_c.x, sp1_c.y, sp1_c.z);
  float v_cell_c = sp1_c.w * sp1_c.w;

  const float kw[3] = { 0.72f, 1.0f, 0.72f };
  const float inv_tol = 16.666666f;
  v3 u81_num = u_sp1;
  v3 u81_sq  = v3_mul(u_sp1, u_sp1);
  float w_u = 1.0f, w2_u = 1.0f, v_u81_acc = v_cell_c;

  for (int dy = -1; dy <= 1; ++dy) {
    int srow = (ly + 3 + dy * 3) * 22 + (lx + 3);
    float wy = kw[dy + 1];
    float r_dot_y = fma((float)dy, dRy3_dot_np, rp_dot_np);
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int slot = srow + dx * 3;
      h4 gqh = sh_gb[slot];
      if (abs(gp_h.w - gqh.w) > 0.25h) continue;
      float zq = (float)gqh.z;
      float psi;
      if (abs(gqh.x - gp_h.x) + abs(gqh.y - gp_h.y) < 1e-4h) {
        float d_geo = f_abs(fma(zq, fma((float)dx, dRx3_dot_np, r_dot_y), -p_plane));
        float q = d_geo * inv_tol;
        if (q >= 1.0f) continue;
        float t = 1.0f - q * q;
        psi = t * t;
      } else {
        v3 nq = os_oct_decode((float)gqh.x, (float)gqh.y);
        float w_n = os_stiefel_null(v3_dot(np, nq));
        if (w_n <= 0.0f) continue;
        v3 dRx3 = mk3(u.cam_right[0] * sx3, u.cam_right[1] * sx3, u.cam_right[2] * sx3);
        v3 dRy3 = mk3(u.cam_up[0] * sy3,    u.cam_up[1] * sy3,    u.cam_up[2] * sy3);
        v3 dpos = v3_add(v3_scale(rp_ray, zq - zp),
                         v3_scale(v3_add(v3_scale(dRy3, (float)dy), v3_scale(dRx3, (float)dx)), zq));
        psi = os_psi_perp_plane(dpos, np, nq, w_n, inv_tol);
        if (psi <= 0.0f) continue;
      }
      float w = wy * kw[dx + 1] * psi;
      v4 sq4 = h4_to_v4(sh_sp1[slot]);
      v3 uq = mk3(sq4.x, sq4.y, sq4.z);
      u81_num = v3_add(u81_num, v3_scale(uq, w));
      u81_sq  = v3_add(u81_sq,  v3_scale(v3_mul(uq, uq), w));
      w_u += w;
      w2_u += w * w;
      v_u81_acc += (w * w) * (sq4.w * sq4.w);
    }
  }

  float inv_wu = 1.0f / f_max(w_u, 1e-6f);
  v3 u_81 = v3_scale(u81_num, inv_wu);
  v3 inter_v3 = v3_max(mk3(0.0f, 0.0f, 0.0f), v3_sub(v3_scale(u81_sq, inv_wu), v3_mul(u_81, u_81)));
  float var_inter = (inter_v3.x + inter_v3.y) + inter_v3.z;

  v3 y_out, s_new;
  float age_new;
  if (prev_age < 0.5f) {
    y_out = u_81;
    s_new = u_81;
    age_new = 1.0f;
  } else {
    float gx = ((float)px + 0.5f) * (1.0f / 16.0f) - 0.5f;
    float gy = ((float)py + 0.5f) * (1.0f / 16.0f) - 0.5f;
    int tx0 = (int)f_floor(gx);
    int ty0 = (int)f_floor(gy);
    float tfx = gx - (float)tx0;
    float tfy = gy - (float)ty0;
    int lx0 = tx0 - (int)tgp.x + 1;
    int ly0 = ty0 - (int)tgp.y + 1;
    float w00 = (1.0f - tfx) * (1.0f - tfy), w10 = tfx * (1.0f - tfy);
    float w01 = (1.0f - tfx) * tfy,          w11 = tfx * tfy;
    int i00 = ly0 * 3 + lx0, i10 = i00 + 1;
    int i01 = i00 + 3,       i11 = i00 + 4;

    v4 k00 = sh_kap[i00], k10 = sh_kap[i10], k01 = sh_kap[i01], k11 = sh_kap[i11];
    v4 b00 = sh_bet[i00], b10 = sh_bet[i10], b01 = sh_bet[i01], b11 = sh_bet[i11];
    v3 kappa = mk3(k00.x*w00 + k10.x*w10 + k01.x*w01 + k11.x*w11,
                   k00.y*w00 + k10.y*w10 + k01.y*w01 + k11.y*w11,
                   k00.z*w00 + k10.z*w10 + k01.z*w01 + k11.z*w11);
    float a_g = k00.w*w00 + k10.w*w10 + k01.w*w01 + k11.w*w11;
    v3 beta  = mk3(b00.x*w00 + b10.x*w10 + b01.x*w01 + b11.x*w11,
                   b00.y*w00 + b10.y*w10 + b01.y*w01 + b11.y*w11,
                   b00.z*w00 + b10.z*w10 + b01.z*w01 + b11.z*w11);

    v3 s_gauge = v3_max(mk3(0.0f, 0.0f, 0.0f), v3_add(v3_mul(kappa, s_prev), beta));
    v3 s_base = v3_mix(s_prev, s_gauge, a_g);

    v3 d_loc = v3_sub(u_81, s_base);
    float var_loc = (v_u81_acc + 0.30f * var_inter * w2_u) * (inv_wu * inv_wu) + 0.018f;
    float z2_loc = v3_dot(d_loc, d_loc) / var_loc;
    float r_shock = f_max(0.0f, (z2_loc - 16.0f) * 0.25f);

    float g_min = f_max(0.035f, 0.5f / (prev_age + 1.0f));
    float g_n = g_min + 0.04f * a_g + 0.80f * r_shock;
    float G_n = g_n / (1.0f + g_n);

    float sp_blend = f_clamp(G_n * 2.5f + 0.25f * a_g, 0.0f, 1.0f);
    v3 u_eff = v3_mix(u_sp1, u_81, sp_blend);

    v3 v_n = v3_scale(v3_sub(u_eff, s_base), G_n);
    y_out = v3_add(s_base, v_n);
    s_new = v3_add(y_out, v3_scale(v_n, 1.0f - G_n));

    if (r_shock > 1.0f) {
      age_new = 1.0f;
    } else if (a_g > 0.2f) {
      age_new = f_max(12.0f, prev_age * 0.85f);
    } else {
      age_new = f_min(36.0f, prev_age + 1.0f);
    }
  }

  v3 col = v3_mul(alb, v3_max(mk3(0.0f, 0.0f, 0.0f), y_out));
  out_st[idx]  = st4_store(mk4(s_new.x, s_new.y, s_new.z, age_new));
  out_rgb[idx] = mk4(col.x, col.y, col.z, 1.0f);
}

kernel void k_pass2_ablated(
    constant FrameUniforms& u  [[buffer(0)]],
    const device h4* cur_irrad [[buffer(1)]],
    const device u16_4* cur_alb [[buffer(2)]],
    const device h4* cur_gbuf  [[buffer(3)]],
    const device h4* prev_st   [[buffer(4)]],
    device h4* out_st          [[buffer(5)]],
    device v4* out_rgb         [[buffer(6)]],
    uint2 gid                  [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  os_pass2_ablated_pixel(&u, px, py, cur_irrad, cur_alb, cur_gbuf, prev_st, out_st, out_rgb);
}

kernel void k_pass2a_baseline_taa(
    constant FrameUniforms& u  [[buffer(0)]],
    const device h4* cur_irrad [[buffer(1)]],
    const device u16_4* cur_alb [[buffer(2)]],
    const device h4* cur_gbuf  [[buffer(3)]],
    const device h4* prev_st   [[buffer(4)]],
    device h4* out_taa         [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  os_pass2a_baseline_taa(&u, px, py, cur_irrad, cur_alb, cur_gbuf, prev_st, out_taa);
}

kernel void k_pass2b_baseline_atrous(
    constant FrameUniforms& u [[buffer(0)]],
    const device h4* src_rgb  [[buffer(1)]],
    const device h4* cur_gbuf [[buffer(2)]],
    device h4* dst_pack       [[buffer(3)]],
    device v4* out_rgb        [[buffer(4)]],
    uint2 gid                 [[thread_position_in_grid]])
{
  int px = (int)gid.x;
  int py = (int)gid.y;
  if (px >= u.width || py >= u.height) return;
  int is_final = (u.pass_step == 2) ? 1 : 0;
  os_pass2b_baseline_atrous(&u, px, py, u.pass_step, is_final,
                            src_rgb, cur_gbuf, dst_pack, out_rgb);
}
