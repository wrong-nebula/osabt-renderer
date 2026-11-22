/*
 * OS-ABT: Orthogonal-Subspace Antiderivative Bilinear Transport
 * Dual-compile pure mathematical core (ISO C11 CPU Reference <-> Apple Metal MSL)
 * Zero dynamic allocation, deterministic IEEE-754 float32 parity.
 */
#ifndef OSABT_CORE_H
#define OSABT_CORE_H

#ifdef __METAL_VERSION__
  #ifndef OSABT_PASS2_RELAXED
    #pragma clang fp contract(off)
  #endif
  #include <metal_stdlib>
  using namespace metal;
  #define ABT_INLINE inline
  #define ABT_CONST constant
  #define ABT_GLOBAL device
  #define ABT_RO_GLOBAL const device
  #define ABT_THREAD thread
  typedef float2 v2;
  typedef float3 v3;
  typedef float4 v4;
  typedef half4  h4;
  typedef ushort4 u16_4;
  typedef uint   u32;
  ABT_INLINE v2 mk2(float x, float y) { return float2(x, y); }
  ABT_INLINE v3 mk3(float x, float y, float z) { return float3(x, y, z); }
  ABT_INLINE v4 mk4(float x, float y, float z, float w) { return float4(x, y, z, w); }
  ABT_INLINE h4 mkh4(float x, float y, float z, float w) { return half4(float4(x, y, z, w)); }
  ABT_INLINE h4 v4_to_h4(v4 v) { return half4(v); }
  ABT_INLINE v4 h4_to_v4(h4 h) { return float4(h); }
  ABT_INLINE u16_4 v4_to_u16(v4 v) {
    return ushort4(clamp(v, 0.0f, 1.0f) * 65535.0f + 0.5f);
  }
  ABT_INLINE v4 u16_to_v4(u16_4 u) { return float4(u) * (1.0f / 65535.0f); }
  ABT_INLINE u32 f_bits(float x) { return as_type<uint>(x); }
  ABT_INLINE float f_from_bits(u32 b) { return as_type<float>(b); }
  ABT_INLINE u16_4 mku16(u32 x, u32 y, u32 z, u32 w) { return ushort4(uint4(x, y, z, w)); }
  ABT_INLINE float f_abs(float x) { return fabs(x); }
  ABT_INLINE float f_sqrt(float x) { return sqrt(max(x, 0.0f)); }
  ABT_INLINE float f_floor(float x) { return floor(x); }
  ABT_INLINE float f_ldexp(float x, int e) { return ldexp(x, e); }
  ABT_INLINE float f_min(float a, float b) { return min(a, b); }
  ABT_INLINE float f_max(float a, float b) { return max(a, b); }
  ABT_INLINE float f_clamp(float x, float lo, float hi) { return clamp(x, lo, hi); }
  ABT_INLINE v3 v3_add(v3 a, v3 b) { return a + b; }
  ABT_INLINE v3 v3_sub(v3 a, v3 b) { return a - b; }
  ABT_INLINE v3 v3_mul(v3 a, v3 b) { return a * b; }
  ABT_INLINE v3 v3_scale(v3 a, float s) { return a * s; }
  ABT_INLINE float v3_dot(v3 a, v3 b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
  ABT_INLINE v3 v3_cross(v3 a, v3 b) {
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  }
  ABT_INLINE float v3_len(v3 a) { return f_sqrt(v3_dot(a, a)); }
  ABT_INLINE v3 v3_norm(v3 a) { float l = v3_len(a); return (l > 1e-8f) ? v3_scale(a, 1.0f / l) : float3(0.0f, 1.0f, 0.0f); }
  ABT_INLINE v3 v3_mix(v3 a, v3 b, float t) { return a + (b - a) * t; }
  ABT_INLINE v3 v3_min(v3 a, v3 b) { return min(a, b); }
  ABT_INLINE v3 v3_max(v3 a, v3 b) { return max(a, b); }
  ABT_INLINE v3 v3_clamp(v3 x, v3 lo, v3 hi) { return clamp(x, lo, hi); }
#else
  #include <math.h>
  #include <stdint.h>
  #include <string.h>
  #define ABT_INLINE static inline
  #define ABT_CONST const
  #define ABT_GLOBAL
  #define ABT_RO_GLOBAL const
  #define ABT_THREAD
  typedef uint32_t u32;
  typedef struct { _Float16 x, y, z, w; } h4;
  typedef struct { uint16_t x, y, z, w; } u16_4;
  typedef struct { float x, y; } v2;
  typedef struct { float x, y, z; } v3;
  typedef struct { float x, y, z, w; } v4;
  ABT_INLINE v2 mk2(float x, float y) { v2 r = {x, y}; return r; }
  ABT_INLINE v3 mk3(float x, float y, float z) { v3 r = {x, y, z}; return r; }
  ABT_INLINE v4 mk4(float x, float y, float z, float w) { v4 r = {x, y, z, w}; return r; }
  ABT_INLINE h4 mkh4(float x, float y, float z, float w) {
    h4 r = { (_Float16)x, (_Float16)y, (_Float16)z, (_Float16)w };
    return r;
  }
  ABT_INLINE h4 v4_to_h4(v4 v) { return mkh4(v.x, v.y, v.z, v.w); }
  ABT_INLINE v4 h4_to_v4(h4 h) { return mk4((float)h.x, (float)h.y, (float)h.z, (float)h.w); }
  ABT_INLINE float f_min(float a, float b) { return a < b ? a : b; }
  ABT_INLINE float f_max(float a, float b) { return a > b ? a : b; }
  ABT_INLINE float f_clamp(float x, float lo, float hi) { return f_min(hi, f_max(lo, x)); }
  ABT_INLINE u16_4 v4_to_u16(v4 v) {
    u16_4 r = {
      (uint16_t)(f_clamp(v.x, 0.0f, 1.0f) * 65535.0f + 0.5f),
      (uint16_t)(f_clamp(v.y, 0.0f, 1.0f) * 65535.0f + 0.5f),
      (uint16_t)(f_clamp(v.z, 0.0f, 1.0f) * 65535.0f + 0.5f),
      (uint16_t)(f_clamp(v.w, 0.0f, 1.0f) * 65535.0f + 0.5f)
    };
    return r;
  }
  ABT_INLINE v4 u16_to_v4(u16_4 u) {
    const float inv = 1.0f / 65535.0f;
    return mk4((float)u.x * inv, (float)u.y * inv, (float)u.z * inv, (float)u.w * inv);
  }
  ABT_INLINE u32 f_bits(float x) { u32 b; memcpy(&b, &x, sizeof(b)); return b; }
  ABT_INLINE float f_from_bits(u32 b) { float x; memcpy(&x, &b, sizeof(x)); return x; }
  ABT_INLINE u16_4 mku16(u32 x, u32 y, u32 z, u32 w) {
    u16_4 r = { (uint16_t)x, (uint16_t)y, (uint16_t)z, (uint16_t)w };
    return r;
  }
  ABT_INLINE float f_abs(float x) { return fabsf(x); }
  ABT_INLINE float f_sqrt(float x) { return sqrtf(x > 0.0f ? x : 0.0f); }
  ABT_INLINE float f_floor(float x) { return floorf(x); }
  ABT_INLINE float f_ldexp(float x, int e) { return ldexpf(x, e); }
  ABT_INLINE v3 v3_add(v3 a, v3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
  ABT_INLINE v3 v3_sub(v3 a, v3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
  ABT_INLINE v3 v3_mul(v3 a, v3 b) { return mk3(a.x * b.x, a.y * b.y, a.z * b.z); }
  ABT_INLINE v3 v3_scale(v3 a, float s) { return mk3(a.x * s, a.y * s, a.z * s); }
  ABT_INLINE float v3_dot(v3 a, v3 b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
  ABT_INLINE v3 v3_cross(v3 a, v3 b) {
    return mk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  }
  ABT_INLINE float v3_len(v3 a) { return f_sqrt(v3_dot(a, a)); }
  ABT_INLINE v3 v3_norm(v3 a) { float l = v3_len(a); return (l > 1e-8f) ? v3_scale(a, 1.0f / l) : mk3(0.0f, 1.0f, 0.0f); }
  ABT_INLINE v3 v3_mix(v3 a, v3 b, float t) { return mk3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t); }
  ABT_INLINE v3 v3_min(v3 a, v3 b) { return mk3(f_min(a.x, b.x), f_min(a.y, b.y), f_min(a.z, b.z)); }
  ABT_INLINE v3 v3_max(v3 a, v3 b) { return mk3(f_max(a.x, b.x), f_max(a.y, b.y), f_max(a.z, b.z)); }
  ABT_INLINE v3 v3_clamp(v3 x, v3 lo, v3 hi) { return v3_min(hi, v3_max(lo, x)); }
#endif

/* -------------------------------------------------------------------------
 * E16: shared-exponent HDR history pack, 8 B/px (same footprint as half4).
 *   x,y,z : 16-bit mantissas of max(rgb,0) sharing ONE binary exponent e, value = m * 2^(e-16)
 *   w     : [15:5] age in 1/32-frame steps (0 .. 63.97; the transport caps age at 36)
 *           [4:0]  e + 15, e in [-15, 16]  (same ~65504 ceiling as half-float)
 * Quantum is 2^(e-16) <= max(rgb) * 2^-15, i.e. 32x finer than half4's 2^-10 relative step.
 * Why it exists: the affine gauge multiplies history by kappa (~6.5x at a Dirac shock), so each
 * rounding flip between the CPU reference and the relaxed-math GPU kernels is amplified 6.5x.
 * A half-float flip (2e-3 at s~3) lands at 1.3e-2, far past the 5e-3 parity tolerance; an E16
 * flip (6e-5) lands at 4e-4. Full float would also work but doubles the history bandwidth.
 * Pack/unpack use only exact power-of-two scaling and integer ops: bit-identical CPU <-> Metal.
 * History is non-negative by construction (convex ZDF-TPT updates of non-negative irradiance).
 * ------------------------------------------------------------------------- */
ABT_INLINE u16_4 os_pack_e16(v4 v) {
  float r = f_max(v.x, 0.0f), g = f_max(v.y, 0.0f), b = f_max(v.z, 0.0f);
  float m = f_max(r, f_max(g, b));
  int e = (int)((f_bits(m) >> 23) & 0xffu) - 126;          /* m = f * 2^e, f in [0.5, 1) */
  e = (e < -15) ? -15 : ((e > 16) ? 16 : e);
  float scale = f_from_bits((u32)(143 - e) << 23);         /* 2^(16-e), exact */
  u32 mr = (u32)f_min(r * scale + 0.5f, 65535.0f);
  u32 mg = (u32)f_min(g * scale + 0.5f, 65535.0f);
  u32 mb = (u32)f_min(b * scale + 0.5f, 65535.0f);
  u32 ia = (u32)f_min(f_max(v.w, 0.0f) * 32.0f + 0.5f, 2047.0f);
  return mku16(mr, mg, mb, (ia << 5) | (u32)(e + 15));
}
ABT_INLINE v4 os_unpack_e16(u16_4 p) {
  u32 w = (u32)p.w;
  float scale = f_from_bits((96u + (w & 31u)) << 23);      /* 2^(e-16), exact */
  return mk4((float)p.x * scale, (float)p.y * scale, (float)p.z * scale,
             (float)(w >> 5) * (1.0f / 32.0f));
}

/* Storage formats of the OS-ABT temporal path (Scene C): history state `st4` and reprojected
 * history `rp4`.  2 = E16 shared-exponent pack (default), 1 = full float (16 B/px), 0 = half4. */
#ifndef OSABT_ST_FMT
#define OSABT_ST_FMT 2
#endif
#ifndef OSABT_REP_FMT
#define OSABT_REP_FMT 2
#endif
#if OSABT_ST_FMT == 2
  typedef u16_4 st4;
  ABT_INLINE v4  st4_load(st4 s) { return os_unpack_e16(s); }
  ABT_INLINE st4 st4_store(v4 v) { return os_pack_e16(v); }
#elif OSABT_ST_FMT == 1
  typedef v4 st4;
  ABT_INLINE v4  st4_load(st4 s) { return s; }
  ABT_INLINE st4 st4_store(v4 v) { return v; }
#else
  typedef h4 st4;
  ABT_INLINE v4  st4_load(st4 s) { return h4_to_v4(s); }
  ABT_INLINE st4 st4_store(v4 v) { return v4_to_h4(v); }
#endif
#if OSABT_REP_FMT == 2
  typedef u16_4 rp4;
  ABT_INLINE v4  rp4_load(rp4 s) { return os_unpack_e16(s); }
  ABT_INLINE rp4 rp4_store(v4 v) { return os_pack_e16(v); }
#elif OSABT_REP_FMT == 1
  typedef v4 rp4;
  ABT_INLINE v4  rp4_load(rp4 s) { return s; }
  ABT_INLINE rp4 rp4_store(v4 v) { return v; }
#else
  typedef h4 rp4;
  ABT_INLINE v4  rp4_load(rp4 s) { return h4_to_v4(s); }
  ABT_INLINE rp4 rp4_store(v4 v) { return v4_to_h4(v); }
#endif

#define OS_PI   3.14159265358979323846f
#define OS_TAU  6.28318530717958647692f
#define OS_INF  1e20f

/* Shared bit-exact deterministic elementary functions (0-ULP CPU <-> Apple Metal parity) */
ABT_INLINE float f_sin(float x) {
  float k = f_floor(x * (1.0f / OS_TAU) + 0.5f);
  float t = x - k * OS_TAU;
  if (t > 1.5707963267948966f)       t =  OS_PI - t;
  else if (t < -1.5707963267948966f) t = -OS_PI - t;
  float t2 = t * t;
  return t * (1.0f + t2 * (-0.16666666f + t2 * (0.008333302f + t2 * (-0.0001983953f + t2 * 2.718311e-6f))));
}
ABT_INLINE float f_cos(float x) {
  return f_sin(x + 1.5707963267948966f);
}
ABT_INLINE float f_asin(float x) {
  float s = (x < 0.0f) ? -1.0f : 1.0f;
  float a = f_clamp(f_abs(x), 0.0f, 1.0f);
  float p = 1.5707963050f + a * (-0.2145988016f + a * (0.0889789874f + a * (-0.0501743046f
          + a * (0.0308918810f + a * (-0.0170881256f + a * (0.0066700901f - 0.0012624911f * a))))));
  return s * (1.5707963267948966f - f_sqrt(1.0f - a) * p);
}
ABT_INLINE float f_exp(float x) {
  if (x <= -80.0f) return 0.0f;
  if (x >= 80.0f)  return 5.540622e34f;
  float z = x * 1.4426950408889634f;
  float n = f_floor(z + 0.5f);
  float r = (z - n) * 0.6931471805599453f;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
  return f_ldexp(p, (int)n);
}
/* 3-op C-infinity rational kernel matching exp(-u) through 2nd order (zero transcendental overhead) */
ABT_INLINE float f_rat_exp_neg(float u) {
  float x = f_max(0.0f, u);
  return 1.0f / (1.0f + x + 0.5f * x * x);
}

enum {
  SCENE_A = 0,
  SCENE_B = 1,
  SCENE_C = 2,
  SCENE_D = 3,
  /* Showcase scenes (not part of the frozen oracle); all share os_eval_show_ray */
  SCENE_E = 4,  /* "Gallery": ceiling-lit room with spheres, plinth and columns */
  SCENE_F = 5,  /* "Colonnade": hall lit from a side window, forward dolly */
  SCENE_G = 6,  /* "Still life": close-up of objects on a table */
  SCENE_H = 7   /* "Sponza": triangle-mesh atrium traced through a BVH (needs --mesh) */
};

enum {
  MODE_OSABT    = 0,
  MODE_ABLATED  = 1,
  MODE_BASELINE = 2,
  MODE_GT       = 3
};

typedef struct {
  int   scene_id;
  int   mode;
  int   width;
  int   height;
  int   frame_idx;
  int   total_frames;
  int   pass_step;       /* For baseline a-trous step (1 or 2) */
  int   pad0;
  float cam_eye[4];      /* xyz, fov_rad */
  float cam_fwd[4];
  float cam_right[4];
  float cam_up[4];
  float prev_eye[4];     /* xyz, fov_rad */
  float prev_fwd[4];
  float prev_right[4];
  float prev_up[4];
  float light_color[4];  /* rgb, intensity_mult */
} FrameUniforms;

/* =========================================================================
 * Deterministic PRNG & R2 Antithetic Blue-Noise Hash (CPU <-> GPU Identical)
 * ========================================================================= */
ABT_INLINE u32 os_hash_u32(u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

ABT_INLINE float os_u01(u32 h) {
  return (float)(h & 0x00ffffffu) * (1.0f / 16777216.0f);
}

ABT_INLINE v2 os_sample_2d(int px, int py, int frame, int sidx) {
  /* Spatial blue-noise decorrelation via R2 + checkerboard temporal antithetic pairing */
  u32 seed = os_hash_u32((u32)px * 1973u ^ (u32)py * 9277u ^ (u32)sidx * 26699u);
  float u0 = os_u01(seed);
  float v0 = os_u01(os_hash_u32(seed ^ 0x9e3779b9u));
  /* Golden-ratio R2 plastic sequence across frames with parity inversion for antithetic Nyquist */
  float f = (float)(frame >> 1);
  float u = u0 + f * 0.7548776662466927f + (float)sidx * 0.38196601125f;
  float v = v0 + f * 0.5698402909980532f + (float)sidx * 0.61803398875f;
  u = u - f_floor(u);
  v = v - f_floor(v);
  if (((frame + px + py) & 1) != 0) {
    u = 1.0f - u;
    v = 1.0f - v;
  }
  return mk2(u, v);
}

ABT_INLINE float os_luma(v3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

/* =========================================================================
 * PILLAR 1: Exact Orthogonal-Null Stiefel Subspace Projector (Psi_perp)
 * ========================================================================= */
ABT_INLINE float os_smoothstep2(float u) {
  float t = f_clamp(u, 0.0f, 1.0f);
  return t * t * t * (10.0f - 15.0f * t + 6.0f * t * t);
}

/* Algebraic zero whenever |n_p . n_q| <= 0.15 (cutoff c0 = 0.20) or opposite shell */
ABT_INLINE float os_stiefel_null(float ndot) {
  return os_smoothstep2((ndot - 0.20f) * 1.25f); /* 1 / (1 - 0.20) = 1.25 */
}

/* Full co-surface subspace projector: Stiefel normal dead-zone + symmetric geodesic
 * chord-midnormal invariant delta_geo = |(xq - xp) . 0.5*(np + nq)| + sid.
 * Identically 0 on planes, circular cylinders, and spheres; strictly rejects parallel
 * offset shells (thin parapets/depth steps) and orthogonal junctions (exact 0.0)!
 * Uses inv_plane_tol (multiplication) to eliminate inner-loop IEEE-754 division. */
ABT_INLINE float os_psi_perp_plane(v3 dx, v3 np, v3 nq, float w_n, float inv_plane_tol) {
  v3 nm = v3_scale(v3_add(np, nq), 0.5f);
  float delta_geo = f_abs(v3_dot(dx, nm));
  float w_plane = os_smoothstep2(1.0f - delta_geo * inv_plane_tol);
  return w_n * w_plane;
}

ABT_INLINE float os_psi_perp(v3 xp, v3 np, float sid_p, v3 xq, v3 nq, float sid_q, float plane_tol) {
  if (f_abs(sid_p - sid_q) > 0.25f) return 0.0f;
  float w_n = os_stiefel_null(v3_dot(np, nq));
  if (w_n <= 0.0f) return 0.0f;
  return os_psi_perp_plane(v3_sub(xq, xp), np, nq, w_n, 1.0f / plane_tol);
}

/* =========================================================================
 * Hardware half4 Single-Buffer Geometry Codec: (oct_x, oct_y, z_lin, sid)
 * Packs world-space normal + linear camera depth + surface ID into 8 B/px
 * (zero integer bit-packing; halves G-buffer DRAM traffic on Apple M1!).
 * ========================================================================= */
ABT_INLINE v2 os_oct_encode(v3 n) {
  float l1 = f_abs(n.x) + f_abs(n.y) + f_abs(n.z);
  float inv = (l1 > 1e-8f) ? (1.0f / l1) : 0.0f;
  float ox = n.x * inv;
  float oy = n.y * inv;
  if (n.z < 0.0f) {
    float tx = (1.0f - f_abs(oy)) * ((ox >= 0.0f) ? 1.0f : -1.0f);
    float ty = (1.0f - f_abs(ox)) * ((oy >= 0.0f) ? 1.0f : -1.0f);
    ox = tx;
    oy = ty;
  }
  return mk2(ox, oy);
}

ABT_INLINE v3 os_oct_decode(float ox, float oy) {
  float nz = 1.0f - f_abs(ox) - f_abs(oy);
  float nx = ox;
  float ny = oy;
  if (nz < 0.0f) {
    float tx = (1.0f - f_abs(oy)) * ((ox >= 0.0f) ? 1.0f : -1.0f);
    float ty = (1.0f - f_abs(ox)) * ((oy >= 0.0f) ? 1.0f : -1.0f);
    nx = tx;
    ny = ty;
  }
  return v3_norm(mk3(nx, ny, nz));
}

ABT_INLINE h4 os_pack_gbuf_h4(ABT_CONST FrameUniforms* u, v4 gbuf_p, v4 gbuf_n) {
  if (gbuf_n.w < 0.5f) return mkh4(0.0f, 0.0f, 0.0f, 0.0f);
  v2 oct = os_oct_encode(mk3(gbuf_n.x, gbuf_n.y, gbuf_n.z));
  v3 dp = mk3(gbuf_p.x - u->cam_eye[0], gbuf_p.y - u->cam_eye[1], gbuf_p.z - u->cam_eye[2]);
  float z_lin = (dp.x * u->cam_fwd[0] + dp.y * u->cam_fwd[1]) + dp.z * u->cam_fwd[2];
  return mkh4(oct.x, oct.y, z_lin, gbuf_n.w);
}

/* =========================================================================
 * PILLAR 2: Crease-Clipped Asymmetric 2nd-Order Antiderivative (A2_perp)
 * Exact Bartlett / Asymmetric-Tent integral of periodic piecewise-constant
 * signals in O(1) ALU time with exact removable singularity guard.
 * ========================================================================= */

/* Periodic 1D step: value = 1 on u in [a, b) mod P, value = 0 elsewhere (0 <= a <= b <= P).
 * Exact 2nd antiderivative F2(x) decomposed as 0.5 * m * x^2 + periodic G2(u) - c1 * u,
 * keeping u in [0, P) for full IEEE-754 fp32 precision even at large grazing coordinates!
 */
ABT_INLINE float os_c1_periodic_step(float P, float a, float b) {
  float L = f_max(0.0f, b - a);
  float m = L / P;
  float Pa = P - a;
  float Pb = P - b;
  return 0.5f * (-m) * P + 0.5f * (Pa * Pa - Pb * Pb) / P;
}

ABT_INLINE float os_f2_periodic_step_rel(float x_rel, float P, float a, float b) {
  float L = f_max(0.0f, b - a);
  float m = L / P;
  float q = f_floor(x_rel / P);
  float u = x_rel - q * P;
  float ra = f_max(0.0f, u - a);
  float rb = f_max(0.0f, u - b);
  float c1 = os_c1_periodic_step(P, a, b);
  float g2 = 0.5f * (-m) * u * u + 0.5f * (ra * ra - rb * rb) - c1 * u;
  return 0.5f * m * x_rel * x_rel + g2;
}

/* Exact 1st antiderivative F1(x_rel) matching d/dx F2(x_rel) identically! */
ABT_INLINE float os_f1_periodic_step_rel(float x_rel, float P, float a, float b) {
  float L = f_max(0.0f, b - a);
  float m = L / P;
  float q = f_floor(x_rel / P);
  float u = x_rel - q * P;
  float ra = f_max(0.0f, u - a);
  float rb = f_max(0.0f, u - b);
  return m * x_rel + (-m * u + (ra - rb)) - os_c1_periodic_step(P, a, b);
}

/* Exact cancellation-free small-footprint left ramp integral on [xm - hL, xm] (hL <= 0.5*P) */
ABT_INLINE float os_ramp_l_small(float xm, float hL, float P, float a, float b) {
  float lo = xm - hL;
  float hi = xm;
  float inv_h = 1.0f / hL;
  float acc = 0.0f;
  /* k = -1 */
  float s0 = f_max(lo, a - P);
  float s1 = f_min(hi, b - P);
  if (s1 > s0) {
    float u0 = (s0 - lo) * inv_h;
    float u1 = (s1 - lo) * inv_h;
    acc += (u1 * u1 - u0 * u0);
  }
  /* k = 0 */
  s0 = f_max(lo, a);
  s1 = f_min(hi, b);
  if (s1 > s0) {
    float u0 = (s0 - lo) * inv_h;
    float u1 = (s1 - lo) * inv_h;
    acc += (u1 * u1 - u0 * u0);
  }
  return acc;
}

/* Exact cancellation-free small-footprint right ramp integral on [xm, xm + hR] (hR <= 0.5*P) */
ABT_INLINE float os_ramp_r_small(float xm, float hR, float P, float a, float b) {
  float lo = xm;
  float hi = xm + hR;
  float inv_h = 1.0f / hR;
  float acc = 0.0f;
  /* k = 0 */
  float s0 = f_max(lo, a);
  float s1 = f_min(hi, b);
  if (s1 > s0) {
    float v0 = (hi - s0) * inv_h;
    float v1 = (hi - s1) * inv_h;
    acc += (v0 * v0 - v1 * v1);
  }
  /* k = +1 */
  s0 = f_max(lo, a + P);
  s1 = f_min(hi, b + P);
  if (s1 > s0) {
    float v0 = (hi - s0) * inv_h;
    float v1 = (hi - s1) * inv_h;
    acc += (v0 * v0 - v1 * v1);
  }
  return acc;
}

/* Crease-clipped Asymmetric 2nd-Order ADAA2 for periodic indicator on [a, b] in [0, P).
 * Split into left ramp I_L and right ramp I_R with exact quadratic CDF for h <= 0.5*P,
 * guaranteeing zero fp32 catastrophic cancellation across ALL scales h in [1e-7*P, 1e4*P]!
 */
ABT_INLINE float os_adaa2_step(float xc, float hL, float hR, float P, float a, float b) {
  if (b <= a + 1e-7f) return 0.0f;
  if (b - a >= P - 1e-7f) return 1.0f;
  float x0 = f_floor(xc / P) * P;
  float xm = xc - x0;
  float halfP = 0.5f * P;
  float hL_s = f_max(hL, 1e-6f * P);
  float hR_s = f_max(hR, 1e-6f * P);
  float IL, IR;
  if (hL_s <= halfP) {
    IL = os_ramp_l_small(xm, hL_s, P, a, b);
  } else {
    float f1m = os_f1_periodic_step_rel(xm, P, a, b);
    float f2m = os_f2_periodic_step_rel(xm, P, a, b);
    float f2l = os_f2_periodic_step_rel(xm - hL_s, P, a, b);
    IL = (2.0f / hL_s) * (f1m - (f2m - f2l) / hL_s);
  }
  if (hR_s <= halfP) {
    IR = os_ramp_r_small(xm, hR_s, P, a, b);
  } else {
    float f1m = os_f1_periodic_step_rel(xm, P, a, b);
    float f2m = os_f2_periodic_step_rel(xm, P, a, b);
    float f2r = os_f2_periodic_step_rel(xm + hR_s, P, a, b);
    IR = (2.0f / hR_s) * ((f2r - f2m) / hR_s - f1m);
  }
  return f_clamp((hL_s * IL + hR_s * IR) / (hL_s + hR_s), 0.0f, 1.0f);
}

/* Exact Closed-Form Horizon Landing Solver for H(s) = A * cos(2*pi*f*s) at slope mu.
 * Solves phi + cos(phi)/lam = phi_c + cos(phi_c)/lam on (pi - phi_c, 2*pi)
 * to < 1e-7 in 3 Newton steps using asymptotic minimax seed (zero loops over periods!).
 */
ABT_INLINE float os_solve_horizon_landing(float lam, float phi_c) {
  float xi_c = phi_c + f_cos(phi_c) / lam;
  float lo = OS_PI - phi_c;
  float hi = OS_TAU;
  float s0 = f_sqrt(lam);
  float s1 = f_sqrt(f_max(0.0f, 1.0f - lam));
  float p = 2.8869413f - 3.4815234f * s0 + 2.8214261f * lam
          + 2.8403510f * s1 + 0.0655151f * (1.0f - lam)
          - 0.6570472f * lam * s0 + 0.4870632f * (1.0f - lam) * s1;
  p = f_clamp(p, lo + 1e-5f, hi - 1e-5f);
  for (int i = 0; i < 3; ++i) {
    float f = p + f_cos(p) / lam - xi_c;
    float df = 1.0f - f_sin(p) / lam;
    if (f_abs(df) > 1e-5f) {
      p = f_clamp(p - f / df, lo + 1e-6f, hi - 1e-6f);
    }
  }
  return p;
}

/* Strictly monotonic inversion of xi(phi) = phi + cos(phi)/lam on [vis_lo, vis_hi] */
ABT_INLINE float os_inv_xi_phase(float xi_t, float lam, float vis_lo, float vis_hi, float xi_lo) {
  float u = f_clamp((xi_t - xi_lo) / OS_TAU, 0.0f, 1.0f);
  float p = vis_lo + (vis_hi - vis_lo) * u;
  for (int i = 0; i < 4; ++i) {
    float f = p + f_cos(p) / lam - xi_t;
    float df = f_max(1e-4f, 1.0f - f_sin(p) / lam);
    p = f_clamp(p - f / df, vis_lo, vis_hi);
  }
  return p;
}

typedef struct {
  float w_ridge;
  float s_ridge;
  float s_valley;
  float h_ridge;
} MicroVisResult;

ABT_INLINE float os_eval_wrapped_adaa2_arc(float xc, float hL, float hR, float P,
                                          float phi0, float phi1, float lam, float dir_sign) {
  if (phi1 <= phi0 + 1e-6f) return 0.0f;
  float xi0 = phi0 + f_cos(phi0) / lam;
  float xi1 = phi1 + f_cos(phi1) / lam;
  float a = (xi0 / OS_TAU) * P;
  float b = (xi1 / OS_TAU) * P;
  float qa = f_floor(a / P);
  float a_mod = a - qa * P;
  float len = f_clamp(b - a, 0.0f, P);
  float s_xc = dir_sign * xc;
  float shL = (dir_sign > 0.0f) ? hL : hR;
  float shR = (dir_sign > 0.0f) ? hR : hL;
  if (a_mod + len <= P) {
    return os_adaa2_step(s_xc, shL, shR, P, a_mod, a_mod + len);
  } else {
    float p1 = os_adaa2_step(s_xc, shL, shR, P, a_mod, P);
    float p2 = os_adaa2_step(s_xc, shL, shR, P, 0.0f, a_mod + len - P);
    return f_clamp(p1 + p2, 0.0f, 1.0f);
  }
}

/* 2-point Gauss-Legendre visible-arc stats: returns (mean_ndotl, mean_height) */
ABT_INLINE v2 os_facet_vis_stats(float a0, float a1, float lam, float two_pi_f_A,
                                 float amp, float dir_sign, v3 sun_dir) {
  if (a1 <= a0 + 1e-6f) return mk2(f_max(0.0f, sun_dir.y), 0.0f);
  const float gl_x[2] = { -0.57735027f, 0.57735027f };
  float mid = 0.5f * (a0 + a1);
  float hspan = 0.5f * (a1 - a0);
  float num_s = 0.0f, num_h = 0.0f, den = 0.0f;
  for (int i = 0; i < 2; ++i) {
    float ph = mid + hspan * gl_x[i];
    float sp = f_sin(ph);
    float dh_dz = dir_sign * (-two_pi_f_A * sp);
    float invl = 1.0f / f_sqrt(1.0f + dh_dz * dh_dz);
    float ndl = f_max(0.0f, (sun_dir.y - dh_dz * sun_dir.z) * invl);
    float jac = f_max(0.0f, 1.0f - sp / lam);
    num_s += ndl * jac;
    num_h += (amp * f_cos(ph)) * jac;
    den += jac;
  }
  if (den > 1e-7f) return mk2(num_s / den, num_h / den);
  return mk2(f_max(0.0f, sun_dir.y), 0.0f);
}

ABT_INLINE MicroVisResult os_eval_micro_groove_adaa2(
    float coord, float hL, float hR, float freq, float amp,
    float mu_slope, float dir_sign, v3 sun_dir, int white_furnace) {
  MicroVisResult out;
  if (white_furnace) {
    float P = 1.0f / freq;
    out.w_ridge = os_adaa2_step(dir_sign * coord, hL, hR, P, 0.0f, P);
    out.s_ridge = 1.0f; out.s_valley = 1.0f; out.h_ridge = 0.0f;
    return out;
  }
  float P = 1.0f / freq;
  float two_pi_f_A = OS_TAU * freq * amp;
  float lam = mu_slope / f_max(two_pi_f_A, 1e-6f);
  float vis_lo, vis_hi;
  if (lam >= 0.9999f) {
    vis_lo = -OS_PI;
    vis_hi = OS_PI;
  } else {
    float lcl = f_clamp(lam, 1e-4f, 0.9999f);
    float phi_c = f_asin(lcl);
    float phi_l = os_solve_horizon_landing(lcl, phi_c);
    vis_lo = phi_l - OS_TAU;
    vis_hi = phi_c;
  }
  float band_lo = (dir_sign > 0.0f) ? (-0.24f * OS_TAU) : (-0.14f * OS_TAU);
  float band_hi = (dir_sign > 0.0f) ? ( 0.14f * OS_TAU) : ( 0.24f * OS_TAU);
  float r0 = f_max(vis_lo, band_lo);
  float r1 = f_min(vis_hi, band_hi);
  float v0 = vis_lo;
  float v1 = f_min(vis_hi, band_lo);

  /* Single ADAA2 evaluation: w_valley === 1 - w_ridge identically! */
  out.w_ridge = os_eval_wrapped_adaa2_arc(coord, hL, hR, P, r0, r1, lam, dir_sign);

  v2 r_avg = os_facet_vis_stats(r0, r1, lam, two_pi_f_A, amp, dir_sign, sun_dir);
  v2 v_avg = os_facet_vis_stats(v0, v1, lam, two_pi_f_A, amp, dir_sign, sun_dir);

  /* Smooth sub-period local phase resolution via exact inverse xi^-1(s) -> periodic integral */
  float dxi = OS_TAU * freq * (hL + hR);
  float sub_w = f_clamp(dxi / OS_TAU, 0.0f, 1.0f);
  float blend = sub_w * sub_w * (3.0f - 2.0f * sub_w);
  if (blend < 0.999f) {
    float xi_lo = vis_lo + f_cos(vis_lo) / lam;
    float xi_c_raw = OS_TAU * freq * (dir_sign * coord);
    float q_xi = f_floor((xi_c_raw - xi_lo) / OS_TAU);
    float xi_c_mod = xi_c_raw - q_xi * OS_TAU;
    float phi_pix = os_inv_xi_phase(xi_c_mod, lam, vis_lo, vis_hi, xi_lo);

    float ph_r = f_clamp(phi_pix, r0, r1);
    float dh_r = dir_sign * (-two_pi_f_A * f_sin(ph_r));
    float s_r_loc = f_max(0.0f, (sun_dir.y - dh_r * sun_dir.z) / f_sqrt(1.0f + dh_r * dh_r));
    float h_r_loc = amp * f_cos(ph_r);

    float ph_v = (v1 > v0 + 1e-6f) ? f_clamp(phi_pix, v0, v1) : vis_lo;
    float dh_v = dir_sign * (-two_pi_f_A * f_sin(ph_v));
    float s_v_loc = f_max(0.0f, (sun_dir.y - dh_v * sun_dir.z) / f_sqrt(1.0f + dh_v * dh_v));

    out.s_ridge  = s_r_loc * (1.0f - blend) + r_avg.x * blend;
    out.h_ridge  = h_r_loc * (1.0f - blend) + r_avg.y * blend;
    out.s_valley = s_v_loc * (1.0f - blend) + v_avg.x * blend;
  } else {
    out.s_ridge  = r_avg.x;
    out.h_ridge  = r_avg.y;
    out.s_valley = v_avg.x;
  }
  return out;
}

/* =========================================================================
 * PILLAR 3: Exact Thin-Shell Stokes Reciprocity & Kulla-Conty MS GGX
 * ========================================================================= */
ABT_INLINE float os_fresnel_schlick(float cos_t, float f0) {
  float m = f_clamp(1.0f - f_clamp(cos_t, 0.0f, 1.0f), 0.0f, 1.0f);
  float m2 = m * m;
  return f0 + (1.0f - f0) * (m2 * m2 * m);
}

ABT_INLINE float os_fresnel_dielectric(float cos_i, float ior) {
  float c = f_clamp(f_abs(cos_i), 0.0f, 1.0f);
  float sin2_t = (1.0f - c * c) / (ior * ior);
  if (sin2_t >= 1.0f) return 1.0f;
  float cos_t = f_sqrt(1.0f - sin2_t);
  float rs = (c - ior * cos_t) / (c + ior * cos_t);
  float rp = (ior * c - cos_t) / (ior * c + cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

/* Exact Stokes geometric-series thin-shell slab (R + T === 1.000000 identically) */
ABT_INLINE v2 os_stokes_slab(float cos_i, float ior) {
  float F = os_fresnel_dielectric(cos_i, ior);
  float denom = 1.0f + F;
  return mk2((2.0f * F) / denom, (1.0f - F) / denom);
}

/* Naive 2-interface slab without infinite internal bounces (loses up to 14.8% energy!) */
ABT_INLINE v2 os_naive_slab(float cos_i, float ior) {
  float F = os_fresnel_dielectric(cos_i, ior);
  float omF = 1.0f - F;
  return mk2(F + omF * omF * F, omF * omF);
}

/* Exact single-scattering GGX directional albedo E_ss(mu, alpha) (Heitz/Hill/Karis rational fit) */
ABT_INLINE float os_ggx_directional_albedo(float mu, float alpha) {
  float m = f_clamp(mu, 0.01f, 1.0f);
  float a = f_clamp(alpha, 0.01f, 1.0f);
  /* Accurate smooth analytical fit to Smith-GGX single-scatter hemispherical albedo */
  float a2 = a * a;
  float loss = a2 * (0.42f * (1.0f - m) + 0.24f * (1.0f - m) * (1.0f - m) + 0.14f);
  return f_clamp(1.0f - loss, 0.25f, 1.0f);
}

/* =========================================================================
 * Analytical 3D Ray Primitives & Scene Evaluators (Scenes A, B, C, D)
 * ========================================================================= */
typedef struct {
  float t;
  v3    pos;
  v3    nrm;
  float sid;    /* Surface ID: 0=sky, 1=floor, 2=pillar/cliff, 3=parapet, 4=alcove, 5=ceiling, 6=sphere1, 7=sphere2 */
  int   hit;
} HitInfo;

ABT_INLINE HitInfo os_hit_none(void) {
  HitInfo h;
  h.t = OS_INF; h.pos = mk3(0,0,0); h.nrm = mk3(0,1,0); h.sid = 0.0f; h.hit = 0;
  return h;
}

ABT_INLINE void os_isect_plane_y(v3 ro, v3 rd, float y0, float ny_sign, float sid, ABT_THREAD HitInfo* best) {
  if (f_abs(rd.y) < 1e-7f) return;
  float t = (y0 - ro.y) / rd.y;
  if (t > 1e-4f && t < best->t) {
    best->t = t;
    best->pos = v3_add(ro, v3_scale(rd, t));
    best->nrm = mk3(0.0f, ny_sign, 0.0f);
    best->sid = sid;
    best->hit = 1;
  }
}

ABT_INLINE void os_isect_plane_x(v3 ro, v3 rd, float x0, float nx_sign, float sid, ABT_THREAD HitInfo* best) {
  if (f_abs(rd.x) < 1e-7f) return;
  float t = (x0 - ro.x) / rd.x;
  if (t > 1e-4f && t < best->t) {
    v3 p = v3_add(ro, v3_scale(rd, t));
    if (p.y >= 0.0f) {
      best->t = t; best->pos = p; best->nrm = mk3(nx_sign, 0.0f, 0.0f); best->sid = sid; best->hit = 1;
    }
  }
}

ABT_INLINE void os_isect_plane_z(v3 ro, v3 rd, float z0, float nz_sign, float sid, ABT_THREAD HitInfo* best) {
  if (f_abs(rd.z) < 1e-7f) return;
  float t = (z0 - ro.z) / rd.z;
  if (t > 1e-4f && t < best->t) {
    v3 p = v3_add(ro, v3_scale(rd, t));
    if (p.y >= 0.0f && p.y <= 3.0f) {
      best->t = t; best->pos = p; best->nrm = mk3(0.0f, 0.0f, nz_sign); best->sid = sid; best->hit = 1;
    }
  }
}

ABT_INLINE void os_isect_cylinder_y(v3 ro, v3 rd, float cx, float cz, float R,
                                   float y_min, float y_max, float sid, ABT_THREAD HitInfo* best) {
  float ox = ro.x - cx, oz = ro.z - cz;
  float a = rd.x * rd.x + rd.z * rd.z;
  if (a < 1e-8f) return;
  float b = ox * rd.x + oz * rd.z;
  float c = ox * ox + oz * oz - R * R;
  float disc = b * b - a * c;
  if (disc < 0.0f) return;
  float sd = f_sqrt(disc);
  float t = (-b - sd) / a;
  if (t <= 1e-4f) t = (-b + sd) / a;
  if (t > 1e-4f && t < best->t) {
    v3 p = v3_add(ro, v3_scale(rd, t));
    if (p.y >= y_min && p.y <= y_max) {
      best->t = t;
      best->pos = p;
      best->nrm = v3_norm(mk3(p.x - cx, 0.0f, p.z - cz));
      best->sid = sid;
      best->hit = 1;
    }
  }
}

ABT_INLINE v3 os_make_inv_rd(v3 rd) {
  return mk3((f_abs(rd.x) > 1e-8f) ? (1.0f / rd.x) : 1e8f,
             (f_abs(rd.y) > 1e-8f) ? (1.0f / rd.y) : 1e8f,
             (f_abs(rd.z) > 1e-8f) ? (1.0f / rd.z) : 1e8f);
}

ABT_INLINE void os_isect_box_inv(v3 ro, v3 rd, v3 inv_rd, v3 bmin, v3 bmax, float sid, ABT_THREAD HitInfo* best) {
  float tx1 = (bmin.x - ro.x) * inv_rd.x, tx2 = (bmax.x - ro.x) * inv_rd.x;
  float ty1 = (bmin.y - ro.y) * inv_rd.y, ty2 = (bmax.y - ro.y) * inv_rd.y;
  float tz1 = (bmin.z - ro.z) * inv_rd.z, tz2 = (bmax.z - ro.z) * inv_rd.z;
  float tmin = f_max(f_max(f_min(tx1, tx2), f_min(ty1, ty2)), f_min(tz1, tz2));
  float tmax = f_min(f_min(f_max(tx1, tx2), f_max(ty1, ty2)), f_max(tz1, tz2));
  if (tmax < 1e-4f || tmin > tmax) return;
  float t = (tmin > 1e-4f) ? tmin : tmax;
  if (t < best->t) {
    v3 p = v3_add(ro, v3_scale(rd, t));
    v3 n = mk3(0.0f, 0.0f, 1.0f);
    float eps = 1e-3f;
    if (f_abs(p.x - bmin.x) < eps) n = mk3(-1.0f, 0.0f, 0.0f);
    else if (f_abs(p.x - bmax.x) < eps) n = mk3(1.0f, 0.0f, 0.0f);
    else if (f_abs(p.y - bmin.y) < eps) n = mk3(0.0f, -1.0f, 0.0f);
    else if (f_abs(p.y - bmax.y) < eps) n = mk3(0.0f, 1.0f, 0.0f);
    else if (f_abs(p.z - bmin.z) < eps) n = mk3(0.0f, 0.0f, -1.0f);
    else if (f_abs(p.z - bmax.z) < eps) n = mk3(0.0f, 0.0f, 1.0f);
    best->t = t; best->pos = p; best->nrm = n; best->sid = sid; best->hit = 1;
  }
}

ABT_INLINE void os_isect_box(v3 ro, v3 rd, v3 bmin, v3 bmax, float sid, ABT_THREAD HitInfo* best) {
  os_isect_box_inv(ro, rd, os_make_inv_rd(rd), bmin, bmax, sid, best);
}

ABT_INLINE int os_isect_box_any(v3 ro, v3 inv_rd, v3 bmin, v3 bmax, float max_t) {
  float tx1 = (bmin.x - ro.x) * inv_rd.x, tx2 = (bmax.x - ro.x) * inv_rd.x;
  float ty1 = (bmin.y - ro.y) * inv_rd.y, ty2 = (bmax.y - ro.y) * inv_rd.y;
  float tz1 = (bmin.z - ro.z) * inv_rd.z, tz2 = (bmax.z - ro.z) * inv_rd.z;
  float tmin = f_max(f_max(f_min(tx1, tx2), f_min(ty1, ty2)), f_min(tz1, tz2));
  float tmax = f_min(f_min(f_max(tx1, tx2), f_max(ty1, ty2)), f_max(tz1, tz2));
  if (tmax < 1e-4f || tmin > tmax) return 0;
  float t = (tmin > 1e-4f) ? tmin : tmax;
  return (t > 1e-4f && t < max_t) ? 1 : 0;
}

ABT_INLINE int os_isect_cylinder_y_any(v3 ro, v3 rd, float cx, float cz, float R,
                                       float y_min, float y_max, float max_t) {
  float ox = ro.x - cx, oz = ro.z - cz;
  float a = rd.x * rd.x + rd.z * rd.z;
  if (a < 1e-8f) return 0;
  float b = ox * rd.x + oz * rd.z;
  float c = ox * ox + oz * oz - R * R;
  float disc = b * b - a * c;
  if (disc < 0.0f) return 0;
  float sd = f_sqrt(disc);
  float t = (-b - sd) / a;
  if (t <= 1e-4f) t = (-b + sd) / a;
  if (t > 1e-4f && t < max_t) {
    float py = ro.y + rd.y * t;
    if (py >= y_min && py <= y_max) return 1;
  }
  return 0;
}

ABT_INLINE void os_isect_sphere(v3 ro, v3 rd, v3 center, float R, float sid, ABT_THREAD HitInfo* best) {
  v3 oc = v3_sub(ro, center);
  float b = v3_dot(oc, rd);
  float c = v3_dot(oc, oc) - R * R;
  float disc = b * b - c;
  if (disc < 0.0f) return;
  float sd = f_sqrt(disc);
  float t = -b - sd;
  if (t <= 1e-4f) t = -b + sd;
  if (t > 1e-4f && t < best->t) {
    v3 p = v3_add(ro, v3_scale(rd, t));
    best->t = t; best->pos = p; best->nrm = v3_norm(v3_sub(p, center)); best->sid = sid; best->hit = 1;
  }
}

ABT_INLINE float os_isect_glass_panel(v3 ro, v3 rd, v3 center, v3 nrm, float hx, float hy, float t_max) {
  float denom = v3_dot(rd, nrm);
  if (f_abs(denom) < 1e-6f) return -1.0f;
  float t = v3_dot(v3_sub(center, ro), nrm) / denom;
  if (t <= 1e-4f || t >= t_max) return -1.0f;
  v3 p = v3_add(ro, v3_scale(rd, t));
  v3 d = v3_sub(p, center);
  /* Local panel axes: up=(0,1,0), tang = cross(up, nrm) */
  v3 tang = v3_norm(mk3(nrm.z, 0.0f, -nrm.x));
  float lx = v3_dot(d, tang);
  float ly = d.y;
  if (f_abs(lx) <= hx && f_abs(ly) <= hy) return t;
  return -1.0f;
}

/* Scene-specialized zero-redundant-division raytracers & any-hit shadow tests */
ABT_INLINE HitInfo os_trace_scene_a(v3 ro, v3 rd) {
  HitInfo best = os_hit_none();
  v3 inv_rd = os_make_inv_rd(rd);
  os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
  os_isect_box_inv(ro, rd, inv_rd, mk3(-5.0f, 2.40f, -4.55f), mk3(5.0f, 2.55f, -3.20f), 5.0f, &best);
  os_isect_cylinder_y(ro, rd, 0.0f, -2.55f, 0.62f, 0.0f, 2.40f, 2.0f, &best);
  os_isect_box_inv(ro, rd, inv_rd, mk3(-2.80f, 0.0f, -4.15f), mk3(2.80f, 0.78f, -4.07f), 3.0f, &best);
  os_isect_plane_z(ro, rd, -4.48f, 1.0f, 4.0f, &best);
  os_isect_box_inv(ro, rd, inv_rd, mk3(-2.80f, 2.02f, -3.20f), mk3(2.80f, 2.40f, -2.10f), 5.0f, &best);
  return best;
}

/* Exact any-hit sun shadow test for Scene A (sun_dir = (0.48,0.72,0.50) > 0: floor/alcove cannot occlude) */
ABT_INLINE int os_shadow_scene_a(v3 ro, v3 rd, v3 inv_rd) {
  if (os_isect_cylinder_y_any(ro, rd, 0.0f, -2.55f, 0.62f, 0.0f, 2.40f, OS_INF)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-2.80f, 0.0f, -4.15f), mk3(2.80f, 0.78f, -4.07f), OS_INF)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-2.80f, 2.02f, -3.20f), mk3(2.80f, 2.40f, -2.10f), OS_INF)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-5.0f, 2.40f, -4.55f), mk3(5.0f, 2.55f, -3.20f), OS_INF)) return 1;
  return 0;
}

ABT_INLINE HitInfo os_trace_scene_c(v3 ro, v3 rd) {
  HitInfo best = os_hit_none();
  v3 inv_rd = os_make_inv_rd(rd);
  os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
  os_isect_cylinder_y(ro, rd, -0.32f, -0.55f, 0.42f, 0.0f, 2.20f, 2.0f, &best);
  os_isect_plane_z(ro, rd, -3.60f, 1.0f, 3.0f, &best);
  os_isect_box_inv(ro, rd, inv_rd, mk3(1.15f, 0.0f, -2.40f), mk3(1.55f, 2.10f, -2.00f), 4.0f, &best);
  return best;
}

/* Exact any-hit area-light shadow test for Scene C (light disk inside room: only cylinder & box can occlude) */
ABT_INLINE int os_shadow_scene_c(v3 ro, v3 rd, float max_t) {
  if (os_isect_cylinder_y_any(ro, rd, -0.32f, -0.55f, 0.42f, 0.0f, 2.20f, max_t)) return 1;
  if (os_isect_box_any(ro, os_make_inv_rd(rd), mk3(1.15f, 0.0f, -2.40f), mk3(1.55f, 2.10f, -2.00f), max_t)) return 1;
  return 0;
}

/* -------------------------------------------------------------------------
 * Scene E "Gallery": closed 4.8 x 3.0 m room (open toward the camera) with a visible
 * rectangular ceiling light, three spheres, a low pedestal and two rear columns.
 * Surface IDs: 1 floor, 2 left wall, 3 back wall, 4 right wall, 5 ceiling, 6 ivory sphere,
 * 7 blue sphere, 8 pedestal, 9 amber sphere, 10 rear-left column, 11 rear-right column,
 * 12 emissive light panel.
 * ------------------------------------------------------------------------- */
/* Slab box intersector that takes the face normal from the slab that produced the hit, so a
 * pixel on a box edge can never pick up the neighbouring face's normal. os_isect_box_inv decides
 * the face by position within 1 mm, which leaves dark dashes along box edges; it is left as-is
 * because the frozen oracle's scenes A and C depend on its exact output. */
ABT_INLINE void os_isect_box_face(v3 ro, v3 rd, v3 inv_rd, v3 bmin, v3 bmax, float sid, ABT_THREAD HitInfo* best) {
  float tx1 = (bmin.x - ro.x) * inv_rd.x, tx2 = (bmax.x - ro.x) * inv_rd.x;
  float ty1 = (bmin.y - ro.y) * inv_rd.y, ty2 = (bmax.y - ro.y) * inv_rd.y;
  float tz1 = (bmin.z - ro.z) * inv_rd.z, tz2 = (bmax.z - ro.z) * inv_rd.z;
  float nx = f_min(tx1, tx2), ny = f_min(ty1, ty2), nz = f_min(tz1, tz2);
  float fx = f_max(tx1, tx2), fy = f_max(ty1, ty2), fz = f_max(tz1, tz2);
  float tmin = f_max(f_max(nx, ny), nz);
  float tmax = f_min(f_min(fx, fy), fz);
  if (tmax < 1e-4f || tmin > tmax) return;
  float t;
  v3 n;
  if (tmin > 1e-4f) {
    t = tmin;   /* entering: outward normal opposes the ray on the entry axis */
    if (nx >= ny && nx >= nz) n = mk3((rd.x > 0.0f) ? -1.0f : 1.0f, 0.0f, 0.0f);
    else if (ny >= nz)        n = mk3(0.0f, (rd.y > 0.0f) ? -1.0f : 1.0f, 0.0f);
    else                      n = mk3(0.0f, 0.0f, (rd.z > 0.0f) ? -1.0f : 1.0f);
  } else {
    t = tmax;   /* origin inside the box: exit face */
    if (fx <= fy && fx <= fz) n = mk3((rd.x > 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
    else if (fy <= fz)        n = mk3(0.0f, (rd.y > 0.0f) ? 1.0f : -1.0f, 0.0f);
    else                      n = mk3(0.0f, 0.0f, (rd.z > 0.0f) ? 1.0f : -1.0f);
  }
  if (t < best->t) {
    best->t = t; best->pos = v3_add(ro, v3_scale(rd, t)); best->nrm = n; best->sid = sid; best->hit = 1;
  }
}

ABT_INLINE int os_isect_sphere_any(v3 ro, v3 rd, v3 center, float R, float max_t) {
  v3 oc = v3_sub(ro, center);
  float b = v3_dot(oc, rd);
  float c = v3_dot(oc, oc) - R * R;
  float disc = b * b - c;
  if (disc < 0.0f) return 0;
  float sd = f_sqrt(disc);
  float t = -b - sd;
  if (t <= 1e-4f) t = -b + sd;
  return (t > 1e-4f && t < max_t) ? 1 : 0;
}

ABT_INLINE HitInfo os_trace_scene_e(v3 ro, v3 rd) {
  HitInfo best = os_hit_none();
  v3 inv_rd = os_make_inv_rd(rd);
  os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
  os_isect_plane_y(ro, rd, 3.0f, -1.0f, 5.0f, &best);
  os_isect_plane_x(ro, rd, -2.40f, 1.0f, 2.0f, &best);
  os_isect_plane_x(ro, rd, 2.40f, -1.0f, 4.0f, &best);
  os_isect_plane_z(ro, rd, -3.80f, 1.0f, 3.0f, &best);
  os_isect_sphere(ro, rd, mk3(-0.60f, 0.55f, -1.75f), 0.55f, 6.0f, &best);
  os_isect_sphere(ro, rd, mk3(-0.05f, 0.34f, 0.10f), 0.34f, 7.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.25f, 0.0f, -2.85f), mk3(1.35f, 0.30f, -1.95f), 8.0f, &best);
  os_isect_sphere(ro, rd, mk3(0.80f, 0.62f, -2.40f), 0.32f, 9.0f, &best);
  os_isect_cylinder_y(ro, rd, -1.95f, -3.30f, 0.22f, 0.0f, 3.0f, 10.0f, &best);
  os_isect_cylinder_y(ro, rd, 1.95f, -3.30f, 0.22f, 0.0f, 3.0f, 11.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-0.65f, 2.94f, -2.35f), mk3(0.65f, 3.10f, -1.35f), 12.0f, &best);
  return best;
}

/* Any-hit shadow test toward the ceiling light: the room shell is convex and the light panel
 * is the target, so only the six free-standing objects can occlude. */
ABT_INLINE int os_shadow_scene_e(v3 ro, v3 rd, float max_t) {
  if (os_isect_sphere_any(ro, rd, mk3(-0.60f, 0.55f, -1.75f), 0.55f, max_t)) return 1;
  if (os_isect_sphere_any(ro, rd, mk3(-0.05f, 0.34f, 0.10f), 0.34f, max_t)) return 1;
  if (os_isect_sphere_any(ro, rd, mk3(0.80f, 0.62f, -2.40f), 0.32f, max_t)) return 1;
  if (os_isect_box_any(ro, os_make_inv_rd(rd), mk3(0.25f, 0.0f, -2.85f), mk3(1.35f, 0.30f, -1.95f), max_t)) return 1;
  if (os_isect_cylinder_y_any(ro, rd, -1.95f, -3.30f, 0.22f, 0.0f, 3.0f, max_t)) return 1;
  if (os_isect_cylinder_y_any(ro, rd, 1.95f, -3.30f, 0.22f, 0.0f, 3.0f, max_t)) return 1;
  return 0;
}

/* -------------------------------------------------------------------------
 * Scene F "Colonnade": 4.8 m wide hall, 11 m deep, lit only by one compact window panel in
 * the left wall. Four full-height columns stand between the window and the camera's side of
 * the hall, so their shadows fan out across the floor and up the right wall; a red sphere on
 * a plinth sits at the far end.
 * Surface IDs: 1 floor, 2 left wall, 3 end wall, 4 right wall, 5 ceiling, 6-9 columns
 * (one ID each: overlapping silhouettes must not be filtered together), 10 plinth,
 * 11 sphere, 12 emissive window panel.
 * ------------------------------------------------------------------------- */
ABT_INLINE HitInfo os_trace_scene_f(v3 ro, v3 rd) {
  HitInfo best = os_hit_none();
  v3 inv_rd = os_make_inv_rd(rd);
  os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
  os_isect_plane_y(ro, rd, 3.0f, -1.0f, 5.0f, &best);
  os_isect_plane_x(ro, rd, -2.40f, 1.0f, 2.0f, &best);
  os_isect_plane_x(ro, rd, 2.40f, -1.0f, 4.0f, &best);
  os_isect_plane_z(ro, rd, -7.50f, 1.0f, 3.0f, &best);
  os_isect_cylinder_y(ro, rd, -0.90f, -1.20f, 0.20f, 0.0f, 3.0f, 6.0f, &best);
  os_isect_cylinder_y(ro, rd, -0.90f, -2.60f, 0.20f, 0.0f, 3.0f, 7.0f, &best);
  os_isect_cylinder_y(ro, rd, -0.90f, -4.00f, 0.20f, 0.0f, 3.0f, 8.0f, &best);
  os_isect_cylinder_y(ro, rd, -0.90f, -5.40f, 0.20f, 0.0f, 3.0f, 9.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.60f, 0.0f, -6.30f), mk3(1.40f, 0.50f, -5.50f), 10.0f, &best);
  os_isect_sphere(ro, rd, mk3(1.00f, 1.05f, -5.90f), 0.55f, 11.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-2.50f, 1.00f, -3.20f), mk3(-2.36f, 2.50f, -2.00f), 12.0f, &best);
  return best;
}

ABT_INLINE int os_shadow_scene_f(v3 ro, v3 rd, float max_t) {
  if (os_isect_cylinder_y_any(ro, rd, -0.90f, -1.20f, 0.20f, 0.0f, 3.0f, max_t)) return 1;
  if (os_isect_cylinder_y_any(ro, rd, -0.90f, -2.60f, 0.20f, 0.0f, 3.0f, max_t)) return 1;
  if (os_isect_cylinder_y_any(ro, rd, -0.90f, -4.00f, 0.20f, 0.0f, 3.0f, max_t)) return 1;
  if (os_isect_cylinder_y_any(ro, rd, -0.90f, -5.40f, 0.20f, 0.0f, 3.0f, max_t)) return 1;
  if (os_isect_box_any(ro, os_make_inv_rd(rd), mk3(0.60f, 0.0f, -6.30f), mk3(1.40f, 0.50f, -5.50f), max_t)) return 1;
  if (os_isect_sphere_any(ro, rd, mk3(1.00f, 1.05f, -5.90f), 0.55f, max_t)) return 1;
  return 0;
}

/* -------------------------------------------------------------------------
 * Scene G "Still life": a table carrying three spheres, a cube and a tall carton, seen
 * close up; lit by an off-centre ceiling softbox so shadows fall across the table top.
 * Surface IDs: 1 floor, 2 left wall, 3 back wall, 4 right wall, 5 ceiling, 6 table top,
 * 7 front legs, 8 back legs, 9 red sphere, 10 green sphere, 11 white sphere,
 * 12 emissive softbox, 13 cube, 14 carton.
 * ------------------------------------------------------------------------- */
ABT_INLINE HitInfo os_trace_scene_g(v3 ro, v3 rd) {
  HitInfo best = os_hit_none();
  v3 inv_rd = os_make_inv_rd(rd);
  os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
  os_isect_plane_y(ro, rd, 3.0f, -1.0f, 5.0f, &best);
  os_isect_plane_x(ro, rd, -3.20f, 1.0f, 2.0f, &best);
  os_isect_plane_x(ro, rd, 3.20f, -1.0f, 4.0f, &best);
  os_isect_plane_z(ro, rd, -3.40f, 1.0f, 3.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-1.10f, 0.70f, -2.60f), mk3(1.10f, 0.78f, -1.40f), 6.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-1.04f, 0.0f, -1.54f), mk3(-0.96f, 0.70f, -1.46f), 7.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.96f, 0.0f, -1.54f), mk3(1.04f, 0.70f, -1.46f), 7.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-1.04f, 0.0f, -2.54f), mk3(-0.96f, 0.70f, -2.46f), 8.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.96f, 0.0f, -2.54f), mk3(1.04f, 0.70f, -2.46f), 8.0f, &best);
  os_isect_sphere(ro, rd, mk3(-0.55f, 1.08f, -2.10f), 0.30f, 9.0f, &best);
  os_isect_sphere(ro, rd, mk3(-0.15f, 0.98f, -1.70f), 0.20f, 10.0f, &best);
  os_isect_sphere(ro, rd, mk3(0.35f, 0.90f, -1.60f), 0.12f, 11.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.55f, 0.78f, -2.20f), mk3(0.95f, 1.18f, -1.80f), 13.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(0.10f, 0.78f, -2.50f), mk3(0.30f, 1.50f, -2.30f), 14.0f, &best);
  os_isect_box_face(ro, rd, inv_rd, mk3(-1.50f, 2.94f, -1.60f), mk3(-0.30f, 3.10f, -0.60f), 12.0f, &best);
  return best;
}

ABT_INLINE int os_shadow_scene_g(v3 ro, v3 rd, float max_t) {
  v3 inv_rd = os_make_inv_rd(rd);
  if (os_isect_sphere_any(ro, rd, mk3(-0.55f, 1.08f, -2.10f), 0.30f, max_t)) return 1;
  if (os_isect_sphere_any(ro, rd, mk3(-0.15f, 0.98f, -1.70f), 0.20f, max_t)) return 1;
  if (os_isect_sphere_any(ro, rd, mk3(0.35f, 0.90f, -1.60f), 0.12f, max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-1.10f, 0.70f, -2.60f), mk3(1.10f, 0.78f, -1.40f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(0.55f, 0.78f, -2.20f), mk3(0.95f, 1.18f, -1.80f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(0.10f, 0.78f, -2.50f), mk3(0.30f, 1.50f, -2.30f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-1.04f, 0.0f, -1.54f), mk3(-0.96f, 0.70f, -1.46f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(0.96f, 0.0f, -1.54f), mk3(1.04f, 0.70f, -1.46f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(-1.04f, 0.0f, -2.54f), mk3(-0.96f, 0.70f, -2.46f), max_t)) return 1;
  if (os_isect_box_any(ro, inv_rd, mk3(0.96f, 0.0f, -2.54f), mk3(1.04f, 0.70f, -2.46f), max_t)) return 1;
  return 0;
}

/* Showcase scene dispatch (the scene id is a compile-time constant in each Metal entry kernel) */
ABT_INLINE HitInfo os_trace_show(int sc, v3 ro, v3 rd) {
  if (sc == SCENE_F) return os_trace_scene_f(ro, rd);
  if (sc == SCENE_G) return os_trace_scene_g(ro, rd);
  return os_trace_scene_e(ro, rd);
}

ABT_INLINE int os_shadow_show(int sc, v3 ro, v3 rd, float max_t) {
  if (sc == SCENE_F) return os_shadow_scene_f(ro, rd, max_t);
  if (sc == SCENE_G) return os_shadow_scene_g(ro, rd, max_t);
  return os_shadow_scene_e(ro, rd, max_t);
}

/* =========================================================================
 * Triangle-mesh ray tracing (Scene H). The BVH is built on the host (src/mesh_bvh.h) and reaches
 * both back ends as three flat float4 arrays:
 *   nodes : 2 per node   (bmin.xyz, A) (bmax.xyz, B); interior: A = left child (right = A + 1),
 *           B = 0; leaf: A = first triangle, B = triangle count (> 0). Indices are exact floats.
 *   tri_a : 3 per triangle (v0, e1, e2)      tri_n : 3 per triangle (vertex normals)
 * Stack-based nearest-first traversal with Moller-Trumbore intersection; identical arithmetic
 * on the CPU reference and in the (safe-math) Metal Pass 1 kernels.
 * ========================================================================= */
typedef struct {
  float t;      /* hit distance (or the t_max passed in when nothing was hit) */
  float bu;     /* barycentric weight of vertex 1 */
  float bv;     /* barycentric weight of vertex 2 */
  int   tri;    /* triangle index in leaf order, -1 = miss */
} MeshHit;

/* Entry distance into an AABB, or OS_INF when the ray misses it or it lies beyond t_max */
ABT_INLINE float os_aabb_entry(v4 n0, v4 n1, v3 ro, v3 inv_rd, float t_max) {
  float tx1 = (n0.x - ro.x) * inv_rd.x, tx2 = (n1.x - ro.x) * inv_rd.x;
  float ty1 = (n0.y - ro.y) * inv_rd.y, ty2 = (n1.y - ro.y) * inv_rd.y;
  float tz1 = (n0.z - ro.z) * inv_rd.z, tz2 = (n1.z - ro.z) * inv_rd.z;
  float tmin = f_max(f_max(f_min(tx1, tx2), f_min(ty1, ty2)), f_min(tz1, tz2));
  float tmax = f_min(f_min(f_max(tx1, tx2), f_max(ty1, ty2)), f_max(tz1, tz2));
  if (tmax < 0.0f || tmin > tmax || tmin >= t_max) return OS_INF;
  return tmin;
}

/* Nearest hit in (1e-4, t_max); with any_hit != 0 returns at the first hit found (shadow rays) */
ABT_INLINE MeshHit os_mesh_trace(ABT_RO_GLOBAL v4* nodes, ABT_RO_GLOBAL v4* tri_a,
                                 v3 ro, v3 rd, float t_max, int any_hit) {
  MeshHit best;
  best.t = t_max; best.bu = 0.0f; best.bv = 0.0f; best.tri = -1;
  v3 inv_rd = os_make_inv_rd(rd);
  int   stack_n[48];
  float stack_t[48];
  int sp = 0;
  int ni = 0;
  if (os_aabb_entry(nodes[0], nodes[1], ro, inv_rd, best.t) >= OS_INF) return best;
  for (int guard = 0; guard < 65536; ++guard) {   /* hard bound on visited nodes */
    v4 n0 = nodes[ni * 2];
    v4 n1 = nodes[ni * 2 + 1];
    int descend = 0;
    if (n1.w > 0.5f) {
      int first = (int)n0.w;
      int count = (int)n1.w;
      for (int k = 0; k < count; ++k) {
        int ti = first + k;
        v4 a0 = tri_a[ti * 3], a1 = tri_a[ti * 3 + 1], a2 = tri_a[ti * 3 + 2];
        v3 e1 = mk3(a1.x, a1.y, a1.z);
        v3 e2 = mk3(a2.x, a2.y, a2.z);
        v3 pv = v3_cross(rd, e2);
        float det = v3_dot(e1, pv);
        if (f_abs(det) < 1e-12f) continue;
        float inv_det = 1.0f / det;
        v3 tv = mk3(ro.x - a0.x, ro.y - a0.y, ro.z - a0.z);
        float bu = v3_dot(tv, pv) * inv_det;
        if (bu < 0.0f || bu > 1.0f) continue;
        v3 qv = v3_cross(tv, e1);
        float bv = v3_dot(rd, qv) * inv_det;
        if (bv < 0.0f || bu + bv > 1.0f) continue;
        float t = v3_dot(e2, qv) * inv_det;
        if (t > 1e-4f && t < best.t) {
          best.t = t; best.bu = bu; best.bv = bv; best.tri = ti;
          if (any_hit != 0) return best;
        }
      }
    } else {
      int l = (int)n0.w;
      float tl = os_aabb_entry(nodes[l * 2], nodes[l * 2 + 1], ro, inv_rd, best.t);
      float tr = os_aabb_entry(nodes[l * 2 + 2], nodes[l * 2 + 3], ro, inv_rd, best.t);
      if (tl < OS_INF && tr < OS_INF) {
        int near_n = (tl <= tr) ? l : (l + 1);
        int far_n  = (tl <= tr) ? (l + 1) : l;
        if (sp < 48) { stack_n[sp] = far_n; stack_t[sp] = f_max(tl, tr); ++sp; }
        ni = near_n; descend = 1;
      } else if (tl < OS_INF) {
        ni = l; descend = 1;
      } else if (tr < OS_INF) {
        ni = l + 1; descend = 1;
      }
    }
    if (descend != 0) continue;
    /* pop the next subtree that can still contain a closer hit */
    int found = 0;
    while (sp > 0) {
      --sp;
      if (stack_t[sp] < best.t) { ni = stack_n[sp]; found = 1; break; }
    }
    if (found == 0) break;
  }
  return best;
}

/* Interpolated vertex normal of a hit, flipped to face the incoming ray (two-sided surfaces) */
ABT_INLINE v3 os_mesh_normal(ABT_RO_GLOBAL v4* tri_n, MeshHit h, v3 rd) {
  v4 n0 = tri_n[h.tri * 3], n1 = tri_n[h.tri * 3 + 1], n2 = tri_n[h.tri * 3 + 2];
  float w0 = 1.0f - h.bu - h.bv;
  v3 n = v3_norm(mk3(n0.x * w0 + n1.x * h.bu + n2.x * h.bv,
                     n0.y * w0 + n1.y * h.bu + n2.y * h.bv,
                     n0.z * w0 + n1.z * h.bu + n2.z * h.bv));
  return (v3_dot(n, rd) > 0.0f) ? v3_scale(n, -1.0f) : n;
}

/* Geometric (flat) triangle normal of a hit, flipped to face the incoming ray. Secondary rays start
 * from a point offset along THIS normal: offsetting along the interpolated shading normal can put
 * the origin inside the surface near silhouettes and leaves dark dots there (shadow terminator). */
ABT_INLINE v3 os_mesh_geo_normal(ABT_RO_GLOBAL v4* tri_a, MeshHit h, v3 rd) {
  v4 a1 = tri_a[h.tri * 3 + 1], a2 = tri_a[h.tri * 3 + 2];
  v3 n = v3_norm(v3_cross(mk3(a1.x, a1.y, a1.z), mk3(a2.x, a2.y, a2.z)));
  return (v3_dot(n, rd) > 0.0f) ? v3_scale(n, -1.0f) : n;
}

/* Trace primary or shadow ray in Scene A, B, C, or D */
ABT_INLINE HitInfo os_trace_scene(int scene_id, v3 ro, v3 rd) {
  if (scene_id == SCENE_A) return os_trace_scene_a(ro, rd);
  if (scene_id == SCENE_C) return os_trace_scene_c(ro, rd);
  HitInfo best = os_hit_none();
  if (scene_id == SCENE_B) {
    os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
    os_isect_plane_x(ro, rd, 1.65f, -1.0f, 2.0f, &best);
  } else {
    /* SCENE_D: White Furnace & Thin-Shell Reciprocity Chamber */
    os_isect_plane_y(ro, rd, 0.0f, 1.0f, 1.0f, &best);
    os_isect_plane_x(ro, rd, -1.75f, 1.0f, 2.0f, &best);
    os_isect_sphere(ro, rd, mk3(-0.68f, 0.62f, -1.60f), 0.55f, 6.0f, &best);
    os_isect_sphere(ro, rd, mk3(0.72f, 0.58f, -1.75f), 0.52f, 7.0f, &best);
  }
  return best;
}

/* =========================================================================
 * Camera Ray & Exact Differential Footprint Calculator (Zero Trig!)
 * ========================================================================= */
ABT_INLINE v3 os_camera_ray_unnorm(ABT_CONST FrameUniforms* u, float px, float py) {
  float tan_half = u->cam_fwd[3];
  float aspect   = u->cam_right[3];
  float ndc_x = (2.0f * (px / (float)u->width) - 1.0f) * aspect * tan_half;
  float ndc_y = (1.0f - 2.0f * (py / (float)u->height)) * tan_half;
  v3 fwd = mk3(u->cam_fwd[0], u->cam_fwd[1], u->cam_fwd[2]);
  v3 rgt = mk3(u->cam_right[0], u->cam_right[1], u->cam_right[2]);
  v3 up  = mk3(u->cam_up[0], u->cam_up[1], u->cam_up[2]);
  return v3_add(v3_add(fwd, v3_scale(rgt, ndc_x)), v3_scale(up, ndc_y));
}

ABT_INLINE v3 os_prev_camera_ray_unnorm(ABT_CONST FrameUniforms* u, float px, float py) {
  float tan_half = u->prev_fwd[3];
  float aspect   = u->prev_right[3];
  float ndc_x = (2.0f * (px / (float)u->width) - 1.0f) * aspect * tan_half;
  float ndc_y = (1.0f - 2.0f * (py / (float)u->height)) * tan_half;
  v3 fwd = mk3(u->prev_fwd[0], u->prev_fwd[1], u->prev_fwd[2]);
  v3 rgt = mk3(u->prev_right[0], u->prev_right[1], u->prev_right[2]);
  v3 up  = mk3(u->prev_up[0], u->prev_up[1], u->prev_up[2]);
  return v3_add(v3_add(fwd, v3_scale(rgt, ndc_x)), v3_scale(up, ndc_y));
}

ABT_INLINE v3 os_pos_from_z_cur(ABT_CONST FrameUniforms* u, int px, int py, float z_lin) {
  v3 r = os_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  return mk3(u->cam_eye[0] + r.x * z_lin,
             u->cam_eye[1] + r.y * z_lin,
             u->cam_eye[2] + r.z * z_lin);
}

ABT_INLINE v3 os_pos_from_z_prev(ABT_CONST FrameUniforms* u, int px, int py, float z_lin) {
  v3 r = os_prev_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  return mk3(u->prev_eye[0] + r.x * z_lin,
             u->prev_eye[1] + r.y * z_lin,
             u->prev_eye[2] + r.z * z_lin);
}

ABT_INLINE v3 os_camera_ray(ABT_CONST FrameUniforms* u, float px, float py) {
  return v3_norm(os_camera_ray_unnorm(u, px, py));
}

/* Exact plane-projected footprint half-extents (hx, hz) for horizontal floor y=0 */
ABT_INLINE v2 os_floor_footprint(ABT_CONST FrameUniforms* u, float px, float py, v3 p_center) {
  v3 ro = mk3(u->cam_eye[0], u->cam_eye[1], u->cam_eye[2]);
  v3 r_dx = os_camera_ray_unnorm(u, px + 0.5f, py);
  v3 r_dy = os_camera_ray_unnorm(u, px, py + 0.5f);
  float tx = (r_dx.y < -1e-5f) ? f_clamp(-ro.y / r_dx.y, 0.01f, 80.0f) : 10.0f;
  float ty = (r_dy.y < -1e-5f) ? f_clamp(-ro.y / r_dy.y, 0.01f, 80.0f) : 10.0f;
  float px1_x = ro.x + r_dx.x * tx, px1_z = ro.z + r_dx.z * tx;
  float py1_x = ro.x + r_dy.x * ty, py1_z = ro.z + r_dy.z * ty;
  float hx = f_abs(px1_x - p_center.x) + f_abs(py1_x - p_center.x);
  float hz = f_abs(px1_z - p_center.z) + f_abs(py1_z - p_center.z);
  return mk2(f_clamp(hx, 1e-4f, 6.0f), f_clamp(hz, 1e-4f, 12.0f));
}

/* Exact vertical-plane footprint (hy, hz) on x = x0 */
ABT_INLINE v2 os_wallx_footprint(ABT_CONST FrameUniforms* u, float px, float py, float x0, v3 p_center) {
  v3 ro = mk3(u->cam_eye[0], u->cam_eye[1], u->cam_eye[2]);
  v3 r_dx = os_camera_ray_unnorm(u, px + 0.5f, py);
  v3 r_dy = os_camera_ray_unnorm(u, px, py + 0.5f);
  float tx = (f_abs(r_dx.x) > 1e-5f) ? f_clamp((x0 - ro.x) / r_dx.x, 0.01f, 80.0f) : 10.0f;
  float ty = (f_abs(r_dy.x) > 1e-5f) ? f_clamp((x0 - ro.x) / r_dy.x, 0.01f, 80.0f) : 10.0f;
  float px1_y = ro.y + r_dx.y * tx, px1_z = ro.z + r_dx.z * tx;
  float py1_y = ro.y + r_dy.y * ty, py1_z = ro.z + r_dy.z * ty;
  float hy = f_abs(px1_y - p_center.y) + f_abs(py1_y - p_center.y);
  float hz = f_abs(px1_z - p_center.z) + f_abs(py1_z - p_center.z);
  return mk2(f_clamp(hy, 1e-4f, 6.0f), f_clamp(hz, 1e-4f, 12.0f));
}

/* Exact vertical-plane footprint (hx, hy) on z = z0 */
ABT_INLINE v2 os_wallz_footprint(ABT_CONST FrameUniforms* u, float px, float py, float z0, v3 p_center) {
  v3 ro = mk3(u->cam_eye[0], u->cam_eye[1], u->cam_eye[2]);
  v3 r_dx = os_camera_ray_unnorm(u, px + 0.5f, py);
  v3 r_dy = os_camera_ray_unnorm(u, px, py + 0.5f);
  float tx = (f_abs(r_dx.z) > 1e-5f) ? f_clamp((z0 - ro.z) / r_dx.z, 0.01f, 80.0f) : 10.0f;
  float ty = (f_abs(r_dy.z) > 1e-5f) ? f_clamp((z0 - ro.z) / r_dy.z, 0.01f, 80.0f) : 10.0f;
  float px1_x = ro.x + r_dx.x * tx, px1_y = ro.y + r_dx.y * tx;
  float py1_x = ro.x + r_dy.x * ty, py1_y = ro.y + r_dy.y * ty;
  float hx = f_abs(px1_x - p_center.x) + f_abs(py1_x - p_center.x);
  float hy = f_abs(px1_y - p_center.y) + f_abs(py1_y - p_center.y);
  return mk2(f_clamp(hx, 1e-4f, 6.0f), f_clamp(hy, 1e-4f, 6.0f));
}

/* Exact physical displaced heightfield intersection for Scene B GT (and POM for Baseline):
 * Ray hits floor micro-relief y = A * cos(2*pi*f*z) along ray ro + t*rd.
 */
ABT_INLINE float os_exact_displaced_hit_z(v3 ro, v3 rd, float freq, float amp, int steps, int bin_steps) {
  /* Ray passes through slab y in [-amp, +amp] for t in [t_top, t_bot] */
  if (rd.y >= -1e-6f) return ro.z + rd.z * 50.0f;
  float t0 = f_max(0.0f, (amp - ro.y) / rd.y);
  float t1 = (-amp - ro.y) / rd.y;
  float dt = (t1 - t0) / (float)steps;
  float t_prev = t0;
  for (int i = 1; i <= steps; ++i) {
    float t_cur = t0 + dt * (float)i;
    float y_cur = ro.y + rd.y * t_cur;
    float z_cur = ro.z + rd.z * t_cur;
    float h_cur = amp * f_cos(OS_TAU * freq * z_cur);
    if (y_cur <= h_cur) {
      float lo = t_prev, hi = t_cur;
      for (int b = 0; b < bin_steps; ++b) {
        float mid = 0.5f * (lo + hi);
        float ym = ro.y + rd.y * mid;
        float zm = ro.z + rd.z * mid;
        if (ym <= amp * f_cos(OS_TAU * freq * zm)) hi = mid;
        else lo = mid;
      }
      float tf = 0.5f * (lo + hi);
      return ro.z + rd.z * tf;
    }
    t_prev = t_cur;
  }
  return ro.z + rd.z * (0.5f * (t0 + t1));
}

/* Exact physical displaced hit on fluted wall x = 1.65 - A*cos(2*pi*f*z) */
ABT_INLINE float os_exact_displaced_wall_z(v3 ro, v3 rd, float x0, float freq, float amp, int steps, int bin_steps) {
  if (rd.x <= 1e-6f) return ro.z + rd.z * 50.0f;
  float t0 = f_max(0.0f, ((x0 - amp) - ro.x) / rd.x);
  float t1 = ((x0 + amp) - ro.x) / rd.x;
  float dt = (t1 - t0) / (float)steps;
  float t_prev = t0;
  for (int i = 1; i <= steps; ++i) {
    float t_cur = t0 + dt * (float)i;
    float x_cur = ro.x + rd.x * t_cur;
    float z_cur = ro.z + rd.z * t_cur;
    float wx = x0 - amp * f_cos(OS_TAU * freq * z_cur);
    if (x_cur >= wx) {
      float lo = t_prev, hi = t_cur;
      for (int b = 0; b < bin_steps; ++b) {
        float mid = 0.5f * (lo + hi);
        float xm = ro.x + rd.x * mid;
        float zm = ro.z + rd.z * mid;
        if (xm >= x0 - amp * f_cos(OS_TAU * freq * zm)) hi = mid;
        else lo = mid;
      }
      return ro.z + rd.z * (0.5f * (lo + hi));
    }
    t_prev = t_cur;
  }
  return ro.z + rd.z * (0.5f * (t0 + t1));
}

/* =========================================================================
 * Helper: Scene A Projective Gobo & Checkerboard Material Evaluation
 * ========================================================================= */
ABT_INLINE float os_scene_a_gobo_pattern(float x, float z) {
  float gx = x * 1.15f + 0.35f * z;
  float gz = z * 1.15f - 0.35f * x;
  float sx = f_sin(OS_TAU * gx);
  float sz = f_sin(OS_TAU * gz);
  return (sx * sz > 0.0f) ? 1.0f : 0.15f;
}

ABT_INLINE v3 os_scene_a_floor_albedo(v3 p, v2 fp, int mode) {
  const float fx = 1.2080078125f;       /* 1237/1024 */
  const float fz = 1.4142135623730951f; /* sqrt(2)   */
  const float Px = 1.0f / fx, Pz = 1.0f / fz;
  const v3 hi = mk3(0.95f, 0.91f, 0.82f);
  const v3 lo = mk3(0.06f, 0.07f, 0.09f);
  if (mode == MODE_OSABT) {
    /* Crease-Clipped Asymmetric 2D-ADAA2: clip footprint strictly against curved pillar & parapet! */
    float dx_cyl = p.x - 0.0f;
    float dz_cyl = p.z - (-2.55f);
    float dist_cyl = f_max(1e-4f, f_sqrt(dx_cyl * dx_cyl + dz_cyl * dz_cyl) - 0.62f);
    float dist_para = f_max(1e-4f, p.z - (-4.07f));
    /* Clip left/right half-extents along x and z strictly inside co-planar floor! */
    float hxL = fp.x, hxR = fp.x, hzL = fp.y, hzR = fp.y;
    if (dx_cyl > 0.0f) hxL = f_min(hxL, dist_cyl); else hxR = f_min(hxR, dist_cyl);
    if (dz_cyl > 0.0f) hzL = f_min(hzL, dist_cyl); else hzR = f_min(hzR, dist_cyl);
    hzL = f_min(hzL, dist_para);
    float ax = os_adaa2_step(p.x, hxL, hxR, Px, 0.0f, 0.5f * Px);
    float az = os_adaa2_step(p.z, hzL, hzR, Pz, 0.0f, 0.5f * Pz);
    /* Exact 2D separable checkerboard probability: P(X XOR Z) = ax + az - 2*ax*az! */
    float chk = ax + az - 2.0f * ax * az;
    return v3_mix(lo, hi, f_clamp(chk, 0.0f, 1.0f));
  } else if (mode == MODE_BASELINE) {
    /* Classical 4-tap rotated-grid anisotropic tap without crease clipping */
    const float ox[4] = { -0.35f,  0.35f, -0.15f,  0.15f };
    const float oz[4] = { -0.15f,  0.15f,  0.35f, -0.35f };
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      float u = (p.x + ox[i] * fp.x) * fx; u -= f_floor(u);
      float v = (p.z + oz[i] * fp.y) * fz; v -= f_floor(v);
      int c = ((u < 0.5f) ^ (v < 0.5f)) ? 1 : 0;
      acc += 0.25f * (float)c;
    }
    return v3_mix(lo, hi, acc);
  } else {
    /* Point-sampled (Ablated & per-ray GT) */
    float u = p.x * fx; u -= f_floor(u);
    float v = p.z * fz; v -= f_floor(v);
    float chk = (((u < 0.5f) ^ (v < 0.5f)) ? 1.0f : 0.0f);
    return v3_mix(lo, hi, chk);
  }
}

ABT_INLINE v3 os_scene_a_pillar_albedo(v3 p) {
  /* Dark slate-blue curved marble with subtle smooth veining (high contrast vs floor!) */
  float vein = 0.5f + 0.5f * f_sin(p.y * 5.2f + 1.4f * f_sin(p.x * 4.0f + p.z * 3.0f));
  v3 base = mk3(0.20f, 0.24f, 0.30f);
  return v3_scale(base, 0.88f + 0.24f * vein);
}

/* =========================================================================
 * Helper: Cosine Hemisphere Sample (Deterministic, Zero Divergence)
 * ========================================================================= */
ABT_INLINE v3 os_cosine_hemi(v3 n, v2 xi) {
  float r = f_sqrt(f_clamp(xi.x, 0.0f, 1.0f));
  float phi = OS_TAU * xi.y;
  float x = r * f_cos(phi);
  float y = r * f_sin(phi);
  float z = f_sqrt(f_max(0.0f, 1.0f - xi.x));
  v3 up = (f_abs(n.y) < 0.99f) ? mk3(0.0f, 1.0f, 0.0f) : mk3(1.0f, 0.0f, 0.0f);
  v3 t = v3_norm(v3_cross(up, n));
  v3 b = v3_cross(n, t);
  return v3_add(v3_add(v3_scale(t, x), v3_scale(b, y)), v3_scale(n, z));
}

/* =========================================================================
 * Scene C Backdrop & Floor Albedo (with ADAA2 vs Point)
 * ========================================================================= */
ABT_INLINE v3 os_scene_c_albedo(const HitInfo h, v2 fp, int mode) {
  if (h.sid == 1.0f) {
    /* Architectural stone tiles on floor */
    float Px = 0.78f, Pz = 0.78f;
    if (mode == MODE_OSABT) {
      float ax = os_adaa2_step(h.pos.x, fp.x, fp.x, Px, 0.0f, 0.5f * Px);
      float az = os_adaa2_step(h.pos.z, fp.y, fp.y, Pz, 0.0f, 0.5f * Pz);
      float chk = ax + az - 2.0f * ax * az;
      return v3_mix(mk3(0.24f, 0.26f, 0.29f), mk3(0.68f, 0.64f, 0.58f), chk);
    } else {
      float u = h.pos.x / Px; u -= f_floor(u);
      float v = h.pos.z / Pz; v -= f_floor(v);
      float chk = ((u < 0.5f) ^ (v < 0.5f)) ? 1.0f : 0.0f;
      return v3_mix(mk3(0.24f, 0.26f, 0.29f), mk3(0.68f, 0.64f, 0.58f), chk);
    }
  } else if (h.sid == 2.0f) {
    /* Foreground occluder pillar: warm terracotta bronze */
    return mk3(0.62f, 0.30f, 0.20f);
  } else if (h.sid == 3.0f) {
    /* Detailed backdrop architectural wall with crisp fluted vertical ribs */
    float P = 0.43f;
    if (mode == MODE_OSABT) {
      float rib = os_adaa2_step(h.pos.x, f_clamp(fp.x, 1e-4f, 0.3f), f_clamp(fp.x, 1e-4f, 0.3f), P, 0.0f, 0.45f * P);
      return v3_mix(mk3(0.28f, 0.35f, 0.44f), mk3(0.82f, 0.78f, 0.70f), rib);
    } else {
      float u = h.pos.x / P; u -= f_floor(u);
      return v3_mix(mk3(0.28f, 0.35f, 0.44f), mk3(0.82f, 0.78f, 0.70f), (u < 0.45f) ? 1.0f : 0.0f);
    }
  } else {
    return mk3(0.48f, 0.52f, 0.46f);
  }
}

/* =========================================================================
 * PASS 1 CORE: Evaluate 1 Pixel (Primary Ray + Analytical ADAA2 + 1-spp GI)
 * Output:
 *   out_irrad  = (E_r, E_g, E_b, demod_flag)  (if demod_flag==0, E is direct RGB!)
 *   out_albedo = (rho_r, rho_g, rho_b, 1.0)
 *   out_gbuf_n = (nx, ny, nz, surface_id)
 *   out_gbuf_p = (px, py, pz, cos_theta)
 * ========================================================================= */
typedef struct {
  v4 irrad;
  v4 direct;
  v4 albedo;
  v4 gbuf_n;
  v4 gbuf_p;
} Pass1Out;

/* Single ray evaluation helper for Scene A (used by GT 256-spp and 1-spp modes) */
ABT_INLINE Pass1Out os_eval_scene_a_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                       float px, float py, int mode, int sidx) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  HitInfo h = os_trace_scene_a(ro, rd);
  v3 sky = mk3(0.30f, 0.36f, 0.46f);
  if (!h.hit) {
    o.irrad  = mk4(sky.x, sky.y, sky.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
    return o;
  }
  float cos_t = f_clamp(v3_dot(h.nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
  o.gbuf_n = mk4(h.nrm.x, h.nrm.y, h.nrm.z, h.sid);
  o.gbuf_p = mk4(h.pos.x, h.pos.y, h.pos.z, cos_t);

  if (h.sid == 4.0f) {
    /* Bright backlit recessed alcove emission */
    o.irrad  = mk4(3.8f, 3.0f, 1.9f, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    return o;
  }
  v3 sun_dir = mk3(0.48028825f, 0.72043238f, 0.50030026f); /* exact norm(0.48, 0.72, 0.50) */
  v3 inv_sun = mk3(2.0820830f,  1.3880553f,  1.9987997f);
  v3 sun_col = mk3(1.65f, 1.55f, 1.40f);
  v2 fp = (h.sid == 1.0f) ? os_floor_footprint(u, px, py, h.pos) : mk2(0.01f, 0.01f);

  /* Surface albedo */
  v3 alb = mk3(0.78f, 0.75f, 0.70f);
  if (h.sid == 1.0f) alb = os_scene_a_floor_albedo(h.pos, fp, mode);
  else if (h.sid == 2.0f) alb = os_scene_a_pillar_albedo(h.pos);
  else if (h.sid == 3.0f) alb = mk3(0.16f, 0.18f, 0.22f);

  /* Fast any-hit shadow visibility against sun (zero divisions!) */
  float ndl = f_max(0.0f, v3_dot(h.nrm, sun_dir));
  float vis = (ndl > 0.0f) ? (os_shadow_scene_a(v3_add(h.pos, v3_scale(h.nrm, 2e-3f)), sun_dir, inv_sun) ? 0.0f : 1.0f) : 0.0f;

  /* Overhead floor projective architectural gobo light (subspace-folded in OS-ABT!) */
  float gobo = os_scene_a_gobo_pattern(h.pos.x, h.pos.z);
  float gobo_weight = 0.0f;
  if (mode == MODE_GT || mode == MODE_OSABT) {
    /* Exact Polar-Folded Orthogonal-Null Projection: strictly 0 when |n . e_y| <= 0.20!
       So floor gobo NEVER climbs orthogonal vertical pillar or parapet! */
    gobo_weight = os_stiefel_null(v3_dot(h.nrm, mk3(0.0f, 1.0f, 0.0f)));
  } else if (mode == MODE_ABLATED) {
    /* Un-nulled naive projective camera-map: projects onto floor AND spills onto vertical faces! */
    gobo_weight = 0.38f + 0.62f * f_max(0.0f, h.nrm.y);
  } else {
    /* Classical triplanar/projective blend (max(0, n.y)^2 + oblique spillover near grazing) */
    float ny = f_max(0.0f, h.nrm.y);
    float obl = 1.0f - f_abs(h.nrm.y);
    gobo_weight = ny * ny + 0.22f * obl * (1.0f - 0.5f * f_abs(h.nrm.z));
  }
  v3 direct = v3_scale(sun_col, ndl * vis * (0.55f + 0.65f * gobo_weight * gobo));

  /* Indirect / Ambient & Backlit Alcove Thin-Parapet Shell Transport */
  v3 indirect;
  if (mode == MODE_GT || mode == MODE_OSABT) {
    /* Exact unbiased physical hemisphere bounce ray (zero backlit alcove leak through solid parapet!) */
    v2 xi = os_sample_2d((int)px, (int)py, u->frame_idx, (mode == MODE_GT) ? sidx : 0);
    v3 bdir = os_cosine_hemi(h.nrm, xi);
    HitInfo bh = os_trace_scene_a(v3_add(h.pos, v3_scale(h.nrm, 2e-3f)), bdir);
    if (!bh.hit) {
      indirect = sky;
    } else if (bh.sid == 4.0f) {
      indirect = mk3(3.8f, 3.0f, 1.9f);
    } else {
      v3 balb = (bh.sid == 1.0f) ? os_scene_a_floor_albedo(bh.pos, mk2(0.02f,0.02f), MODE_ABLATED)
              : (bh.sid == 2.0f) ? os_scene_a_pillar_albedo(bh.pos)
              : (bh.sid == 3.0f) ? mk3(0.16f, 0.18f, 0.22f) : mk3(0.78f, 0.75f, 0.70f);
      float bndl = f_max(0.0f, v3_dot(bh.nrm, sun_dir));
      indirect = v3_mul(balb, v3_add(v3_scale(sun_col, bndl * 0.5f), v3_scale(sky, 0.5f)));
    }
  } else {
    /* Classical SDF / World-space probe GI (Ablated & Baseline):
       Standard 0.75-0.95m probe cell straddles 0.08-thick parapet -> backlit alcove light leak! */
    float probe_r = (mode == MODE_ABLATED) ? 0.95f : 0.78f;
    float dist_alcove = f_abs(h.pos.z - (-4.48f));
    float leak = f_max(0.0f, 1.0f - dist_alcove / probe_r);
    leak = leak * leak;
    v3 alcove_leak = v3_scale(mk3(3.8f, 3.0f, 1.9f), leak * ((mode == MODE_ABLATED) ? 0.38f : 0.22f));
    v2 xi = os_sample_2d((int)px, (int)py, u->frame_idx, 0);
    v3 bdir = os_cosine_hemi(h.nrm, xi);
    HitInfo bh = os_trace_scene_a(v3_add(h.pos, v3_scale(h.nrm, 2e-3f)), bdir);
    v3 base_ind = (!bh.hit) ? sky : (bh.sid == 4.0f ? mk3(3.8f, 3.0f, 1.9f) : v3_scale(sky, 0.65f));
    indirect = v3_add(base_ind, alcove_leak);
  }

  if (mode == MODE_GT) {
    v3 E = v3_add(direct, v3_scale(indirect, 0.55f));
    v3 col = v3_mul(alb, E);
    o.irrad  = mk4(col.x, col.y, col.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  } else if (mode == MODE_OSABT) {
    /* Split-transport: noise-free analytical direct radiance + noisy 1-spp diffuse indirect GI */
    v3 dir_rgb = v3_mul(alb, direct);
    v3 ind_e   = v3_scale(indirect, 0.55f);
    o.direct = mk4(dir_rgb.x, dir_rgb.y, dir_rgb.z, 0.0f);
    o.irrad  = mk4(ind_e.x, ind_e.y, ind_e.z, 1.0f);
    o.albedo = mk4(alb.x, alb.y, alb.z, 1.0f);
  } else {
    v3 E = v3_add(direct, v3_scale(indirect, 0.55f));
    o.irrad  = mk4(E.x, E.y, E.z, 1.0f);
    o.albedo = mk4(alb.x, alb.y, alb.z, 1.0f);
  }
  return o;
}

/* Single ray evaluation helper for Scene B (Grazing Micro-Geometry Cliff) */
ABT_INLINE Pass1Out os_eval_scene_b_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                       float px, float py, int mode) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  HitInfo h = os_trace_scene(SCENE_B, ro, rd);
  v3 sky = mk3(0.24f, 0.30f, 0.42f);
  if (!h.hit) {
    o.irrad  = mk4(sky.x, sky.y, sky.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
    return o;
  }
  float cos_t = f_clamp(v3_dot(h.nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
  o.gbuf_n = mk4(h.nrm.x, h.nrm.y, h.nrm.z, h.sid);
  o.gbuf_p = mk4(h.pos.x, h.pos.y, h.pos.z, cos_t);

  v3 sun_dir = v3_norm(mk3(-0.35f, 0.55f, 0.76f));
  v3 sun_col = mk3(1.80f, 1.70f, 1.55f);
  const v3 ridge_c = mk3(0.94f, 0.89f, 0.80f);
  const v3 val_c   = mk3(0.07f, 0.08f, 0.10f);
  const float fz = 1.7083804731470768f; /* (1237/1024)*sqrt(2) */
  const float fx = 1.6290845870917275f; /* (1031/1024)*phi     */
  const float amp = 0.015f;

  v3 col = sky;
  if (h.sid == 1.0f) {
    /* Grazing micro-grooved floor y = A*cos(2*pi*fz*z) + transverse inlay fx */
    if (mode == MODE_OSABT) {
      /* Exact zero-trig perspective-asymmetric footprint + 2D shear + visible ridge height parallax */
      v3 r_top = os_camera_ray_unnorm(u, px, py - 0.5f);
      v3 r_bot = os_camera_ray_unnorm(u, px, py + 0.5f);
      v3 r_lft = os_camera_ray_unnorm(u, px - 0.5f, py);
      v3 r_rgt = os_camera_ray_unnorm(u, px + 0.5f, py);
      float t_top = (r_top.y < -1e-5f) ? f_clamp(-ro.y / r_top.y, 0.01f, 80.0f) : h.t;
      float t_bot = (r_bot.y < -1e-5f) ? f_clamp(-ro.y / r_bot.y, 0.01f, 80.0f) : h.t;
      float t_lft = (r_lft.y < -1e-5f) ? f_clamp(-ro.y / r_lft.y, 0.01f, 80.0f) : h.t;
      float t_rgt = (r_rgt.y < -1e-5f) ? f_clamp(-ro.y / r_rgt.y, 0.01f, 80.0f) : h.t;
      float z_top = ro.z + r_top.z * t_top;
      float z_bot = ro.z + r_bot.z * t_bot;
      float x_top = ro.x + r_top.x * t_top;
      float x_bot = ro.x + r_bot.x * t_bot;
      float x_lft = ro.x + r_lft.x * t_lft;
      float x_rgt = ro.x + r_rgt.x * t_rgt;
      float hz_neg = f_clamp(f_abs(h.pos.z - z_top), 1e-4f, 12.0f);
      float hz_pos = f_clamp(f_abs(z_bot - h.pos.z), 1e-4f, 12.0f);

      float dir_sign = (rd.z < 0.0f) ? -1.0f : 1.0f;
      float hz_mag = f_abs(rd.z);
      float mu_slope = f_max(1e-4f, -rd.y / f_max(hz_mag, 1e-5f));
      MicroVisResult mv = os_eval_micro_groove_adaa2(
          h.pos.z, hz_neg, hz_pos, fz, amp, mu_slope, dir_sign, sun_dir, 0);

      /* Exact ridge elevation parallax onto visible height y = mv.h_ridge! */
      float scale_h = 1.0f - mv.h_ridge / f_max(ro.y, 1e-4f);
      float x_h = ro.x + (h.pos.x - ro.x) * scale_h;
      float dx_shear = 0.5f * f_abs(x_top - x_bot);
      float hx_neg = f_clamp((f_abs(h.pos.x - x_lft) + dx_shear) * scale_h, 1e-4f, 6.0f);
      float hx_pos = f_clamp((f_abs(x_rgt - h.pos.x) + dx_shear) * scale_h, 1e-4f, 6.0f);
      float hxR = f_min(hx_pos, f_max(1e-4f, 1.65f - x_h));

      float Px = 1.0f / fx;
      float inlay = os_adaa2_step(x_h, hx_neg, hxR, Px, 0.0f, 0.52f * Px);
      float w_val = 1.0f - mv.w_ridge;
      v3 alb_ridge = v3_mix(val_c, ridge_c, 0.28f + 0.72f * inlay);
      v3 E_ridge = v3_add(v3_scale(sun_col, mv.s_ridge), v3_scale(sky, 0.35f));
      v3 E_val   = v3_add(v3_scale(sun_col, mv.s_valley), v3_scale(sky, 0.35f));
      col = v3_add(v3_scale(v3_mul(alb_ridge, E_ridge), mv.w_ridge),
                   v3_scale(v3_mul(val_c, E_val), w_val));
    } else {
      float z_hit = h.pos.z;
      float x_hit = h.pos.x;
      if (mode == MODE_GT) {
        z_hit = os_exact_displaced_hit_z(ro, rd, fz, amp, 96, 10);
        float t_z = (f_abs(rd.z) > 1e-6f) ? ((z_hit - ro.z) / rd.z) : h.t;
        x_hit = ro.x + rd.x * t_z;
      } else if (mode == MODE_BASELINE) {
        z_hit = os_exact_displaced_hit_z(ro, rd, fz, amp, 16, 4);
        float t_z = (f_abs(rd.z) > 1e-6f) ? ((z_hit - ro.z) / rd.z) : h.t;
        x_hit = ro.x + rd.x * t_z;
      }
      float ph_z = z_hit * fz; ph_z -= f_floor(ph_z);
      float ph_x = x_hit * fx; ph_x -= f_floor(ph_x);
      float is_ridge = (ph_z < 0.14f || ph_z > 0.76f) ? 1.0f : 0.0f;
      float is_inlay = (ph_x < 0.52f) ? 1.0f : 0.0f;
      float dh_dz = -OS_TAU * fz * amp * f_sin(OS_TAU * fz * z_hit);
      v3 n_m = v3_norm(mk3(0.0f, 1.0f, -dh_dz));
      float ndl = f_max(0.0f, v3_dot(n_m, sun_dir));
      v3 alb = v3_mix(val_c, ridge_c, is_ridge * (0.28f + 0.72f * is_inlay));
      v3 E = v3_add(v3_scale(sun_col, ndl), v3_scale(sky, 0.35f));
      col = v3_mul(alb, E);
    }
  } else {
    /* Grazing fluted cliff wall at x = 1.65 (sid == 2) */
    const float fc = 2.13701171875f;
    const float ac = 0.018f;
    if (mode == MODE_OSABT) {
      v2 fp = os_wallx_footprint(u, px, py, 1.65f, h.pos);
      float dir_sign = (rd.z < 0.0f) ? -1.0f : 1.0f;
      float mu_slope = f_max(1e-4f, rd.x / f_max(f_abs(rd.z), 1e-5f));
      v3 sun_rot = mk3(sun_dir.y, -sun_dir.x, sun_dir.z);
      MicroVisResult mv = os_eval_micro_groove_adaa2(
          h.pos.z, fp.y, fp.y, fc, ac, mu_slope, dir_sign, sun_rot, 0);
      v3 c_lo = mk3(0.18f, 0.20f, 0.24f);
      v3 c_hi = mk3(0.76f, 0.72f, 0.65f);
      float w_val = 1.0f - mv.w_ridge;
      v3 E_r = v3_add(v3_scale(sun_col, mv.s_ridge), v3_scale(sky, 0.35f));
      v3 E_v = v3_add(v3_scale(sun_col, mv.s_valley), v3_scale(sky, 0.35f));
      col = v3_add(v3_scale(v3_mul(c_hi, E_r), mv.w_ridge),
                   v3_scale(v3_mul(c_lo, E_v), w_val));
    } else {
      float z_hit = h.pos.z;
      if (mode == MODE_GT) z_hit = os_exact_displaced_wall_z(ro, rd, 1.65f, fc, ac, 64, 8);
      else if (mode == MODE_BASELINE) z_hit = os_exact_displaced_wall_z(ro, rd, 1.65f, fc, ac, 16, 4);
      float ph = z_hit * fc; ph -= f_floor(ph);
      float is_ridge = (ph < 0.14f || ph > 0.76f) ? 1.0f : 0.0f;
      float dh_dz = -OS_TAU * fc * ac * f_sin(OS_TAU * fc * z_hit);
      v3 n_m = v3_norm(mk3(-1.0f, 0.0f, -dh_dz));
      float ndl = f_max(0.0f, v3_dot(n_m, sun_dir));
      v3 alb = v3_mix(mk3(0.18f, 0.20f, 0.24f), mk3(0.76f, 0.72f, 0.65f), is_ridge);
      v3 E = v3_add(v3_scale(sun_col, ndl), v3_scale(sky, 0.35f));
      col = v3_mul(alb, E);
    }
  }
  o.irrad  = mk4(col.x, col.y, col.z, 0.0f);
  o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  return o;
}

/* Single ray evaluation helper for Scene C (Dirac Impulse & Whip-Pan Disocclusion) */
ABT_INLINE Pass1Out os_eval_scene_c_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                       float px, float py, int mode, int sidx) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  HitInfo h = os_trace_scene_c(ro, rd);
  v3 sky = mk3(0.22f, 0.27f, 0.36f);
  if (!h.hit) {
    o.irrad  = mk4(sky.x, sky.y, sky.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
    return o;
  }
  float cos_t = f_clamp(v3_dot(h.nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
  o.gbuf_n = mk4(h.nrm.x, h.nrm.y, h.nrm.z, h.sid);
  o.gbuf_p = mk4(h.pos.x, h.pos.y, h.pos.z, cos_t);

  v2 fp = (h.sid == 1.0f) ? os_floor_footprint(u, px, py, h.pos)
        : (h.sid == 3.0f) ? os_wallz_footprint(u, px, py, -3.60f, h.pos)
        : mk2(0.015f, 0.015f);
  v3 alb = os_scene_c_albedo(h, fp, mode);

  /* Area light at (1.80, 2.90, -0.60), radius 0.75 -> 1 spp stochastic soft shadow! */
  v2 xi_l = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2);
  float lr = 0.75f * f_sqrt(xi_l.x);
  float lphi = OS_TAU * xi_l.y;
  v3 lpos = mk3(1.80f + lr * f_cos(lphi), 2.90f, -0.60f + lr * f_sin(lphi));
  v3 to_l = v3_sub(lpos, h.pos);
  float dist_l = v3_len(to_l);
  v3 ldir = v3_scale(to_l, 1.0f / f_max(dist_l, 1e-4f));
  float ndl = f_max(0.0f, v3_dot(h.nrm, ldir));
  v3 hit_orig = v3_add(h.pos, v3_scale(h.nrm, 2e-3f));
  float vis = (ndl > 0.0f) ? (os_shadow_scene_c(hit_orig, ldir, dist_l - 1e-3f) ? 0.0f : 1.0f) : 0.0f;
  float att = 10.5f / (1.0f + 0.14f * dist_l * dist_l);
  v3 lcol = v3_scale(mk3(u->light_color[0], u->light_color[1], u->light_color[2]), u->light_color[3]);
  v3 direct = v3_scale(lcol, ndl * vis * att);

  /* 1 spp stochastic cosine-weighted 1-bounce GI */
  v2 xi_g = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2 + 1);
  v3 bdir = os_cosine_hemi(h.nrm, xi_g);
  HitInfo bh = os_trace_scene_c(hit_orig, bdir);
  v3 bounce;
  if (!bh.hit) {
    bounce = v3_scale(sky, 0.85f);
  } else {
    v3 balb = os_scene_c_albedo(bh, mk2(0.05f, 0.05f), MODE_ABLATED);
    v3 b_tol = v3_sub(mk3(1.80f, 2.90f, -0.60f), bh.pos);
    float b_dl = v3_len(b_tol);
    float b_ndl = f_max(0.0f, v3_dot(bh.nrm, v3_scale(b_tol, 1.0f / f_max(b_dl, 1e-4f))));
    float b_att = 8.0f / (1.0f + 0.14f * b_dl * b_dl);
    bounce = v3_mul(balb, v3_add(v3_scale(lcol, b_ndl * b_att * 0.45f), v3_scale(sky, 0.40f)));
  }

  v3 E = v3_add(direct, v3_scale(bounce, 0.55f));
  if (mode == MODE_GT) {
    v3 col = v3_mul(alb, E);
    o.irrad  = mk4(col.x, col.y, col.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  } else {
    o.irrad  = mk4(E.x, E.y, E.z, 1.0f);
    o.albedo = mk4(alb.x, alb.y, alb.z, 1.0f);
  }
  return o;
}

/* =========================================================================
 * Scene E "Gallery" albedo + single-ray evaluation (showcase; not in the oracle)
 * ========================================================================= */
ABT_INLINE v3 os_scene_e_albedo(const HitInfo h, v2 fp, int mode) {
  if (h.sid == 1.0f) {
    /* Slate / warm-marble checker floor (analytic ADAA2 in OS-ABT, point-sampled otherwise) */
    float P = 0.60f;
    float chk;
    if (mode == MODE_OSABT) {
      float ax = os_adaa2_step(h.pos.x, fp.x, fp.x, P, 0.0f, 0.5f * P);
      float az = os_adaa2_step(h.pos.z, fp.y, fp.y, P, 0.0f, 0.5f * P);
      chk = ax + az - 2.0f * ax * az;
    } else {
      float a = h.pos.x / P; a -= f_floor(a);
      float b = h.pos.z / P; b -= f_floor(b);
      chk = ((a < 0.5f) ^ (b < 0.5f)) ? 1.0f : 0.0f;
    }
    return v3_mix(mk3(0.20f, 0.21f, 0.24f), mk3(0.74f, 0.70f, 0.63f), chk);
  } else if (h.sid == 2.0f) {
    return mk3(0.70f, 0.22f, 0.16f);   /* left wall: terracotta */
  } else if (h.sid == 3.0f) {
    /* Back wall: two-tone vertical panelling */
    float P = 0.55f;
    float s;
    if (mode == MODE_OSABT) {
      float hx = f_clamp(fp.x, 1e-4f, 0.3f);
      s = os_adaa2_step(h.pos.x, hx, hx, P, 0.0f, 0.5f * P);
    } else {
      float a = h.pos.x / P; a -= f_floor(a);
      s = (a < 0.5f) ? 1.0f : 0.0f;
    }
    return v3_mix(mk3(0.60f, 0.64f, 0.68f), mk3(0.80f, 0.78f, 0.72f), s);
  } else if (h.sid == 4.0f) {
    return mk3(0.14f, 0.48f, 0.46f);   /* right wall: teal */
  } else if (h.sid == 5.0f) {
    return mk3(0.90f, 0.90f, 0.88f);   /* ceiling */
  } else if (h.sid == 6.0f) {
    return mk3(0.82f, 0.80f, 0.74f);   /* ivory sphere */
  } else if (h.sid == 7.0f) {
    return mk3(0.16f, 0.30f, 0.72f);   /* blue sphere */
  } else if (h.sid == 8.0f) {
    return mk3(0.50f, 0.42f, 0.36f);   /* warm stone pedestal */
  } else if (h.sid == 9.0f) {
    return mk3(0.92f, 0.36f, 0.05f);   /* amber sphere */
  }
  return mk3(0.16f, 0.10f, 0.07f);     /* dark walnut columns */
}

ABT_INLINE v3 os_scene_f_albedo(const HitInfo h, v2 fp, int mode) {
  if (h.sid == 1.0f) {
    /* Low-contrast pale stone tiles, so the column shadows are the dominant floor pattern */
    float P = 0.90f;
    float chk;
    if (mode == MODE_OSABT) {
      float ax = os_adaa2_step(h.pos.x, fp.x, fp.x, P, 0.0f, 0.5f * P);
      float az = os_adaa2_step(h.pos.z, fp.y, fp.y, P, 0.0f, 0.5f * P);
      chk = ax + az - 2.0f * ax * az;
    } else {
      float a = h.pos.x / P; a -= f_floor(a);
      float b = h.pos.z / P; b -= f_floor(b);
      chk = ((a < 0.5f) ^ (b < 0.5f)) ? 1.0f : 0.0f;
    }
    return v3_mix(mk3(0.58f, 0.57f, 0.56f), mk3(0.68f, 0.67f, 0.64f), chk);
  } else if (h.sid == 2.0f) {
    return mk3(0.20f, 0.34f, 0.44f);   /* window wall: deep blue-green (never directly lit) */
  } else if (h.sid == 3.0f) {
    return mk3(0.62f, 0.30f, 0.22f);   /* end wall: terracotta */
  } else if (h.sid == 4.0f) {
    return mk3(0.80f, 0.74f, 0.62f);   /* right wall: cream, catches the column shadows */
  } else if (h.sid == 5.0f) {
    return mk3(0.85f, 0.84f, 0.80f);   /* ceiling */
  } else if (h.sid == 10.0f) {
    return mk3(0.25f, 0.24f, 0.24f);   /* dark plinth */
  } else if (h.sid == 11.0f) {
    return mk3(0.75f, 0.14f, 0.10f);   /* red sphere */
  }
  return mk3(0.80f, 0.78f, 0.74f);     /* stone columns */
}

ABT_INLINE v3 os_scene_g_albedo(const HitInfo h, v2 fp, int mode) {
  if (h.sid == 1.0f) {
    /* Wood planks running toward the back wall */
    float P = 0.36f;
    float s;
    if (mode == MODE_OSABT) {
      s = os_adaa2_step(h.pos.x, fp.x, fp.x, P, 0.0f, 0.5f * P);
    } else {
      float a = h.pos.x / P; a -= f_floor(a);
      s = (a < 0.5f) ? 1.0f : 0.0f;
    }
    return v3_mix(mk3(0.36f, 0.22f, 0.13f), mk3(0.52f, 0.34f, 0.20f), s);
  } else if (h.sid == 2.0f || h.sid == 4.0f) {
    return mk3(0.75f, 0.72f, 0.68f);   /* side walls */
  } else if (h.sid == 3.0f) {
    return mk3(0.42f, 0.52f, 0.56f);   /* back wall: grey-blue */
  } else if (h.sid == 5.0f) {
    return mk3(0.86f, 0.86f, 0.84f);   /* ceiling */
  } else if (h.sid == 6.0f) {
    return mk3(0.72f, 0.60f, 0.44f);   /* pale oak table top */
  } else if (h.sid == 7.0f || h.sid == 8.0f) {
    return mk3(0.22f, 0.18f, 0.15f);   /* dark legs */
  } else if (h.sid == 9.0f) {
    return mk3(0.75f, 0.12f, 0.10f);   /* red sphere */
  } else if (h.sid == 10.0f) {
    return mk3(0.50f, 0.70f, 0.38f);   /* green sphere */
  } else if (h.sid == 11.0f) {
    return mk3(0.88f, 0.87f, 0.84f);   /* white sphere */
  } else if (h.sid == 13.0f) {
    return mk3(0.20f, 0.36f, 0.70f);   /* blue cube */
  }
  return mk3(0.90f, 0.72f, 0.24f);     /* yellow carton */
}

ABT_INLINE v3 os_show_albedo(int sc, const HitInfo h, v2 fp, int mode) {
  if (sc == SCENE_F) return os_scene_f_albedo(h, fp, mode);
  if (sc == SCENE_G) return os_scene_g_albedo(h, fp, mode);
  return os_scene_e_albedo(h, fp, mode);
}

/* Each showcase scene has one axis-aligned rectangular area light with a visible emissive panel
 * (surface ID 12). One uniform light sample gives irradiance-over-pi
 *   K * cos_surface * cos_light / d^2,   K = L * Area / pi,
 * and the panel itself shows the matching radiance L = K * pi / Area.
 *   E: ceiling panel   x in [-0.65, 0.65], z in [-2.35, -1.35], y = 2.94, facing down  (1.3 m^2)
 *   F: window panel    y in [1.00, 2.50],  z in [-3.20, -2.00], x = -2.36, facing +x   (1.8 m^2)
 *   G: ceiling softbox x in [-1.50, -0.30], z in [-1.60, -0.60], y = 2.94, facing down (1.2 m^2)
 *   H: sky portal      x in [-8.00, 7.00],  z in [-2.00, 2.00],  y = 11.8, facing down  (60 m^2),
 *      spanning the roof opening of the Sponza atrium; the roof itself shadows the excess. */
ABT_INLINE v3 os_show_light_pos(int sc, v2 xi) {
  if (sc == SCENE_F) return mk3(-2.36f, 1.00f + 1.50f * xi.x, -3.20f + 1.20f * xi.y);
  if (sc == SCENE_G) return mk3(-1.50f + 1.20f * xi.x, 2.94f, -1.60f + 1.00f * xi.y);
  if (sc == SCENE_H) return mk3(-8.00f + 15.0f * xi.x, 11.8f, -2.00f + 4.00f * xi.y);
  return mk3(-0.65f + 1.30f * xi.x, 2.94f, -2.35f + 1.00f * xi.y);
}
ABT_INLINE v3 os_show_light_nrm(int sc) {
  return (sc == SCENE_F) ? mk3(1.0f, 0.0f, 0.0f) : mk3(0.0f, -1.0f, 0.0f);
}
ABT_INLINE float os_show_light_k(int sc) {
  return (sc == SCENE_F) ? 25.0f : ((sc == SCENE_G) ? 9.0f : ((sc == SCENE_H) ? 280.0f : 9.5f));
}
ABT_INLINE float os_show_light_le(int sc) {
  return (sc == SCENE_F) ? 43.633f : ((sc == SCENE_G) ? 23.562f : ((sc == SCENE_H) ? 14.661f : 22.958f));
}

/* Single-ray evaluation shared by the showcase scenes E, F and G */
ABT_INLINE Pass1Out os_eval_show_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                    float px, float py, int mode, int sidx, int sc) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  HitInfo h = os_trace_show(sc, ro, rd);
  if (!h.hit) {
    o.irrad  = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
    return o;
  }
  float cos_t = f_clamp(v3_dot(h.nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
  o.gbuf_n = mk4(h.nrm.x, h.nrm.y, h.nrm.z, h.sid);
  o.gbuf_p = mk4(h.pos.x, h.pos.y, h.pos.z, cos_t);
  v3 lcol = v3_scale(mk3(u->light_color[0], u->light_color[1], u->light_color[2]), u->light_color[3]);
  float light_k = os_show_light_k(sc);
  v3 light_n = os_show_light_nrm(sc);

  if (h.sid == 12.0f) {
    /* Emissive panel: final radiance, bypasses demodulation and filtering (own surface ID) */
    float le = os_show_light_le(sc);
    o.irrad  = mk4(lcol.x * le, lcol.y * le, lcol.z * le, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    return o;
  }

  v2 fp = (h.sid == 1.0f) ? os_floor_footprint(u, px, py, h.pos)
        : (sc == SCENE_E && h.sid == 3.0f) ? os_wallz_footprint(u, px, py, -3.80f, h.pos)
        : mk2(0.015f, 0.015f);
  v3 alb = os_show_albedo(sc, h, fp, mode);
  v3 hit_orig = v3_add(h.pos, v3_scale(h.nrm, 2e-3f));

  /* 1 spp next-event estimation on the rectangular light -> stochastic soft shadows */
  v2 xi_l = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2);
  v3 to_l = v3_sub(os_show_light_pos(sc, xi_l), h.pos);
  float d2 = f_max(v3_dot(to_l, to_l), 0.05f);
  float dist_l = f_sqrt(d2);
  v3 ldir = v3_scale(to_l, 1.0f / dist_l);
  float g = f_max(0.0f, v3_dot(h.nrm, ldir)) * f_max(0.0f, -v3_dot(ldir, light_n));
  float vis = (g > 0.0f) ? (os_shadow_show(sc, hit_orig, ldir, dist_l - 1e-3f) ? 0.0f : 1.0f) : 0.0f;
  v3 direct = v3_scale(lcol, light_k * g * vis / d2);

  /* 1 spp cosine-weighted one-bounce GI; the bounce point is lit from the light centre with a
   * shadow ray, plus a constant fill standing in for the deeper bounces (the steady-state indirect
   * irradiance rho*D/(1-rho) of these rooms is ~0.4 in these units; 0.15 keeps shadows readable
   * and allows for the open side).
   * A bounce ray that reaches the panel contributes nothing: the light is already counted by the
   * direct sample. */
  v2 xi_g = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2 + 1);
  v3 bdir = os_cosine_hemi(h.nrm, xi_g);
  HitInfo bh = os_trace_show(sc, hit_orig, bdir);
  v3 bounce = mk3(0.0f, 0.0f, 0.0f);
  if (bh.hit && bh.sid != 12.0f) {
    v3 balb = os_show_albedo(sc, bh, mk2(0.05f, 0.05f), MODE_ABLATED);
    v3 b_tol = v3_sub(os_show_light_pos(sc, mk2(0.5f, 0.5f)), bh.pos);
    float b_d2 = f_max(v3_dot(b_tol, b_tol), 0.05f);
    float b_dl = f_sqrt(b_d2);
    v3 b_ldir = v3_scale(b_tol, 1.0f / b_dl);
    float b_g = f_max(0.0f, v3_dot(bh.nrm, b_ldir)) * f_max(0.0f, -v3_dot(b_ldir, light_n));
    v3 b_orig = v3_add(bh.pos, v3_scale(bh.nrm, 2e-3f));
    float b_vis = (b_g > 0.0f) ? (os_shadow_show(sc, b_orig, b_ldir, b_dl - 1e-3f) ? 0.0f : 1.0f) : 0.0f;
    float b_e = light_k * b_g * b_vis / b_d2 + 0.15f;
    bounce = v3_mul(balb, v3_scale(lcol, b_e));
  }

  v3 E = v3_add(direct, bounce);
  if (mode == MODE_GT) {
    v3 col = v3_mul(alb, E);
    o.irrad  = mk4(col.x, col.y, col.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  } else {
    o.irrad  = mk4(E.x, E.y, E.z, 1.0f);
    o.albedo = mk4(alb.x, alb.y, alb.z, 1.0f);
  }
  return o;
}

/* =========================================================================
 * Scene H "Sponza": single-ray evaluation over the triangle mesh. Lighting is the showcase model
 * (one rectangular area light - here the sky through the roof opening - sampled once, one
 * cosine-weighted bounce lit from the light centre, constant fill for deeper bounces) plus a
 * directional sun sampled once at the primary hit and at the bounce point. The source asset has one material
 * and no textures, so the albedo is uniform limestone with a darker floor, and every mesh pixel
 * carries surface ID 1: the filters separate surfaces by normal and plane distance alone.
 * A ray that leaves the mesh has gone up through the roof opening and shows the sky.
 * ========================================================================= */
ABT_INLINE v3 os_scene_h_albedo(v3 pos, v3 n) {
  if (n.y > 0.9f && pos.y < 0.3f) return mk3(0.46f, 0.43f, 0.40f);   /* ground-floor paving */
  return mk3(0.74f, 0.70f, 0.62f);                                   /* limestone */
}

/* Sun for Scene H: a warm directional light about 38 deg off vertical that shines down the long
 * axis of the atrium (the roof opening is 14 m long but only 3 m wide), so it lands as one broad
 * shape on the far end wall and the floor in front of it, grazing the -z wall on the way.
 * xi jitters the direction inside a 1.15 deg cone (soft shadow edges);
 * xi = (0, 0) gives the central direction. OS_H_SUN_E is the sun's irradiance-over-pi on a
 * surface facing it, in the same units as the sky portal (~1 on the floor). */
#define OS_H_SUN_E 12.0f
ABT_INLINE v3 os_h_sun_dir(v2 xi) {
  v3 l = mk3(-0.60362f, 0.78471f, 0.14085f);         /* normalize(-0.60, 0.78, 0.14) */
  v3 t = v3_norm(v3_cross(l, mk3(1.0f, 0.0f, 0.0f)));
  v3 b = v3_cross(l, t);
  float r = 0.02f * f_sqrt(f_clamp(xi.x, 0.0f, 1.0f));
  float phi = OS_TAU * xi.y;
  return v3_norm(v3_add(l, v3_add(v3_scale(t, r * f_cos(phi)), v3_scale(b, r * f_sin(phi)))));
}
ABT_INLINE v3 os_h_sun_color(void) { return mk3(1.0f, 0.78f, 0.52f); }
/* Radiance shown where the camera sees the sky directly. Display only: a deep blue well below the
 * portal's own radiance, so the opening reads as sky instead of clipping to white. */
ABT_INLINE v3 os_h_sky_display(void) { return mk3(1.40f, 2.30f, 4.00f); }

ABT_INLINE Pass1Out os_eval_mesh_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                    float px, float py, int mode, int sidx,
                                    ABT_RO_GLOBAL v4* nodes, ABT_RO_GLOBAL v4* tri_a, ABT_RO_GLOBAL v4* tri_n) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  v3 lcol = v3_scale(mk3(u->light_color[0], u->light_color[1], u->light_color[2]), u->light_color[3]);
  float light_k = os_show_light_k(SCENE_H);
  v3 light_n = os_show_light_nrm(SCENE_H);
  MeshHit h = os_mesh_trace(nodes, tri_a, ro, rd, OS_INF, 0);
  if (h.tri < 0) {
    v3 sky = v3_scale(os_h_sky_display(), u->light_color[3]);
    o.irrad  = mk4(sky.x, sky.y, sky.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
    return o;
  }
  v3 pos = v3_add(ro, v3_scale(rd, h.t));
  v3 nrm = os_mesh_normal(tri_n, h, rd);
  float cos_t = f_clamp(v3_dot(nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
  o.gbuf_n = mk4(nrm.x, nrm.y, nrm.z, 1.0f);
  o.gbuf_p = mk4(pos.x, pos.y, pos.z, cos_t);
  v3 alb = os_scene_h_albedo(pos, nrm);
  v3 hit_orig = v3_add(pos, v3_scale(os_mesh_geo_normal(tri_a, h, rd), 4e-3f));

  /* 1 spp next-event estimation on the sky portal -> stochastic soft shadows */
  v2 xi_l = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2);
  v3 to_l = v3_sub(os_show_light_pos(SCENE_H, xi_l), pos);
  float d2 = f_max(v3_dot(to_l, to_l), 0.05f);
  float dist_l = f_sqrt(d2);
  v3 ldir = v3_scale(to_l, 1.0f / dist_l);
  float g = f_max(0.0f, v3_dot(nrm, ldir)) * f_max(0.0f, -v3_dot(ldir, light_n));
  float vis = 0.0f;
  if (g > 0.0f) {
    MeshHit sh = os_mesh_trace(nodes, tri_a, hit_orig, ldir, dist_l - 1e-3f, 1);
    vis = (sh.tri < 0) ? 1.0f : 0.0f;
  }
  v3 direct = v3_scale(lcol, light_k * g * vis / d2);

  /* 1 spp sun sample (jittered inside its cone) -> hard-edged shadow shapes with a thin penumbra */
  v3 sun_col = v3_scale(os_h_sun_color(), OS_H_SUN_E * u->light_color[3]);
  v3 sdir = os_h_sun_dir(os_sample_2d((int)px, (int)py, u->frame_idx, 200 + sidx));
  float s_ndl = f_max(0.0f, v3_dot(nrm, sdir));
  if (s_ndl > 0.0f) {
    MeshHit ss = os_mesh_trace(nodes, tri_a, hit_orig, sdir, OS_INF, 1);
    if (ss.tri < 0) direct = v3_add(direct, v3_scale(sun_col, s_ndl));
  }

  /* 1 spp cosine-weighted one-bounce GI (bounce point lit from the portal centre) + constant fill */
  v2 xi_g = os_sample_2d((int)px, (int)py, u->frame_idx, sidx * 2 + 1);
  v3 bdir = os_cosine_hemi(nrm, xi_g);
  MeshHit bh = os_mesh_trace(nodes, tri_a, hit_orig, bdir, OS_INF, 0);
  v3 bounce = mk3(0.0f, 0.0f, 0.0f);
  if (bh.tri >= 0) {
    v3 bpos = v3_add(hit_orig, v3_scale(bdir, bh.t));
    v3 bnrm = os_mesh_normal(tri_n, bh, bdir);
    v3 b_orig = v3_add(bpos, v3_scale(os_mesh_geo_normal(tri_a, bh, bdir), 4e-3f));
    v3 balb = os_scene_h_albedo(bpos, bnrm);
    v3 b_tol = v3_sub(os_show_light_pos(SCENE_H, mk2(0.5f, 0.5f)), bpos);
    float b_d2 = f_max(v3_dot(b_tol, b_tol), 0.05f);
    float b_dl = f_sqrt(b_d2);
    v3 b_ldir = v3_scale(b_tol, 1.0f / b_dl);
    float b_g = f_max(0.0f, v3_dot(bnrm, b_ldir)) * f_max(0.0f, -v3_dot(b_ldir, light_n));
    float b_vis = 0.0f;
    if (b_g > 0.0f) {
      MeshHit bs = os_mesh_trace(nodes, tri_a, b_orig, b_ldir, b_dl - 1e-3f, 1);
      b_vis = (bs.tri < 0) ? 1.0f : 0.0f;
    }
    float b_e = light_k * b_g * b_vis / b_d2 + 0.15f;
    v3 b_in = v3_scale(lcol, b_e);
    /* sunlit surfaces are the strong secondary sources: shadow-test the sun at the bounce point */
    v3 b_sdir = os_h_sun_dir(mk2(0.0f, 0.0f));
    float b_sndl = f_max(0.0f, v3_dot(bnrm, b_sdir));
    if (b_sndl > 0.0f) {
      MeshHit bss = os_mesh_trace(nodes, tri_a, b_orig, b_sdir, OS_INF, 1);
      if (bss.tri < 0) b_in = v3_add(b_in, v3_scale(sun_col, b_sndl));
    }
    bounce = v3_mul(balb, b_in);
  }

  v3 E = v3_add(direct, bounce);
  if (mode == MODE_GT) {
    v3 col = v3_mul(alb, E);
    o.irrad  = mk4(col.x, col.y, col.z, 0.0f);
    o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  } else {
    o.irrad  = mk4(E.x, E.y, E.z, 1.0f);
    o.albedo = mk4(alb.x, alb.y, alb.z, 1.0f);
  }
  return o;
}

/* Pass-1 entry for the mesh scene: 1 spp in the real-time modes, 64-spp Bartlett-filtered
 * ground truth otherwise (8 x 8 rather than 16 x 16: each mesh sample costs several BVH rays). */
ABT_INLINE Pass1Out os_pass1_pixel_mesh(ABT_CONST FrameUniforms* u, int px, int py,
                                       ABT_RO_GLOBAL v4* nodes, ABT_RO_GLOBAL v4* tri_a, ABT_RO_GLOBAL v4* tri_n) {
  v3 ro = mk3(u->cam_eye[0], u->cam_eye[1], u->cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(u, fx_c, fy_c);
  if (u->mode != MODE_GT) {
    return os_eval_mesh_ray(u, ro, rd_c, fx_c, fy_c, u->mode, 0, nodes, tri_a, tri_n);
  }
  Pass1Out center = os_eval_mesh_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, 0, nodes, tri_a, tri_n);
  if (center.gbuf_n.w < 0.5f) return center;
  v3 acc = mk3(0.0f, 0.0f, 0.0f);
  float w_sum = 0.0f;
  v3 c_nrm = mk3(center.gbuf_n.x, center.gbuf_n.y, center.gbuf_n.z);
  const int n_side = 8;
  const float inv_n = 1.0f / 8.0f;
  for (int sy = 0; sy < n_side; ++sy) {
    for (int sx = 0; sx < n_side; ++sx) {
      int sidx = sy * n_side + sx;
      v2 j = os_sample_2d(px, py, u->frame_idx + 17, sidx);
      float ox = ((float)sx + j.x) * inv_n * 2.0f - 1.0f;
      float oy = ((float)sy + j.y) * inv_n * 2.0f - 1.0f;
      float w = (1.0f - f_abs(ox)) * (1.0f - f_abs(oy));
      float qx = fx_c + ox * 0.5f;
      float qy = fy_c + oy * 0.5f;
      Pass1Out sp = os_eval_mesh_ray(u, ro, os_camera_ray(u, qx, qy), qx, qy, MODE_GT, sidx, nodes, tri_a, tri_n);
      /* Crease-clipped primary surface, as in the analytic scenes */
      if (sp.gbuf_n.w < 0.5f || v3_dot(mk3(sp.gbuf_n.x, sp.gbuf_n.y, sp.gbuf_n.z), c_nrm) <= 0.25f) {
        sp = os_eval_mesh_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, sidx, nodes, tri_a, tri_n);
      }
      acc = v3_add(acc, v3_scale(mk3(sp.irrad.x, sp.irrad.y, sp.irrad.z), w));
      w_sum += w;
    }
  }
  v3 avg = v3_scale(acc, 1.0f / f_max(w_sum, 1e-6f));
  center.irrad = mk4(avg.x, avg.y, avg.z, 0.0f);
  center.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  return center;
}

/* Single ray evaluation helper for Scene D (White Furnace & Thin-Shell Reciprocity Chamber) */
ABT_INLINE Pass1Out os_eval_scene_d_ray(ABT_CONST FrameUniforms* u, v3 ro, v3 rd,
                                       float px, float py, int mode) {
  Pass1Out o;
  o.direct = mk4(0.0f, 0.0f, 0.0f, 0.0f);
  HitInfo h = os_trace_scene(SCENE_D, ro, rd);
  float t_bg = h.hit ? h.t : OS_INF;

  /* Evaluate opaque/dielectric background hit inside unit white furnace (L_env === 1.0) */
  v3 bg_rad = mk3(1.0f, 1.0f, 1.0f);
  if (h.hit) {
    float cos_t = f_clamp(v3_dot(h.nrm, v3_scale(rd, -1.0f)), 0.0f, 1.0f);
    o.gbuf_n = mk4(h.nrm.x, h.nrm.y, h.nrm.z, h.sid);
    o.gbuf_p = mk4(h.pos.x, h.pos.y, h.pos.z, cos_t);
    if (h.sid == 1.0f) {
      /* Lossless grazing micro-grooved floor (rho === 1.0) */
      if (mode == MODE_OSABT) {
        v2 fp = os_floor_footprint(u, px, py, h.pos);
        float hxL = f_min(fp.x, f_max(1e-4f, h.pos.x - (-1.75f)));
        /* Asymmetric crease-clipped white-furnace ADAA2 -> exact 1.000000! */
        float wf = os_adaa2_step(h.pos.x, hxL, fp.x, 1.0f / 1.70838047f, 0.0f, 1.0f / 1.70838047f);
        bg_rad = mk3(wf, wf, wf);
      } else {
        bg_rad = mk3(1.0f, 1.0f, 1.0f);
      }
    } else if (h.sid == 2.0f) {
      bg_rad = mk3(1.0f, 1.0f, 1.0f);
    } else if (h.sid == 6.0f) {
      /* Lossless rough conductor sphere (alpha = 0.62, F0 = 1.0) */
      float E_ss = os_ggx_directional_albedo(cos_t, 0.62f);
      if (mode == MODE_OSABT || mode == MODE_GT) {
        float E_ms = 1.0f - E_ss;
        float tot = E_ss + E_ms;
        bg_rad = mk3(tot, tot, tot);
      } else {
        bg_rad = mk3(E_ss, E_ss, E_ss);
      }
    } else if (h.sid == 7.0f) {
      float F = os_fresnel_dielectric(cos_t, 1.5f);
      float tot = F + (1.0f - F);
      bg_rad = mk3(tot, tot, tot);
    }
  } else {
    o.gbuf_n = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    o.gbuf_p = mk4(0.0f, 0.0f, 0.0f, 1.0f);
  }

  /* Two tilted thin transparent glass panels (ior = 1.5) overlapping the chamber! */
  const v3 p1_c = mk3(-0.15f, 0.75f, -0.55f);
  const v3 p1_n = mk3(0.42261826f, 0.0f, 0.90630779f);
  const v3 p2_c = mk3(0.55f, 0.68f, -0.35f);
  const v3 p2_n = mk3(-0.95105652f, 0.0f, 0.30901699f);

  float t1 = os_isect_glass_panel(ro, rd, p1_c, p1_n, 0.65f, 0.55f, t_bg);
  float t2 = os_isect_glass_panel(ro, rd, p2_c, p2_n, 0.60f, 0.50f, t_bg);

  float ts[2];
  v3 ns[2];
  int count = 0;
  if (t1 > 0.0f) { ts[count] = t1; ns[count] = p1_n; count++; }
  if (t2 > 0.0f) { ts[count] = t2; ns[count] = p2_n; count++; }
  if (count == 2 && ts[0] < ts[1]) {
    float tmp = ts[0]; ts[0] = ts[1]; ts[1] = tmp;
    v3 tn = ns[0]; ns[0] = ns[1]; ns[1] = tn;
  }

  v3 rad = bg_rad;
  v3 env = mk3(1.0f, 1.0f, 1.0f);
  for (int i = 0; i < count; ++i) {
    float ci = f_abs(v3_dot(rd, ns[i]));
    v2 rt;
    if (mode == MODE_OSABT) {
      rt = os_stokes_slab(ci, 1.5f);
    } else if (mode == MODE_GT) {
      float F = os_fresnel_dielectric(ci, 1.5f);
      float omF2 = (1.0f - F) * (1.0f - F);
      float R = F, T = 0.0f, fp = 1.0f;
      for (int b = 0; b < 16; ++b) {
        T += omF2 * fp;
        R += omF2 * F * fp;
        fp *= F * F;
      }
      rt = mk2(R, T);
    } else {
      rt = os_naive_slab(ci, 1.5f);
    }
    rad = v3_add(v3_scale(env, rt.x), v3_scale(rad, rt.y));
  }

  o.irrad  = mk4(rad.x, rad.y, rad.z, 0.0f);
  o.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
  return o;
}

/* =========================================================================
 * PASS 1 ENTRY: Dispatch Single Pixel (Handles 256-spp Bartlett GT vs 1-spp)
 * ========================================================================= */
ABT_INLINE Pass1Out os_pass1_pixel(ABT_CONST FrameUniforms* u, int px, int py) {
  v3 ro = mk3(u->cam_eye[0], u->cam_eye[1], u->cam_eye[2]);
  float fx_c = (float)px + 0.5f;
  float fy_c = (float)py + 0.5f;
  v3 rd_c = os_camera_ray(u, fx_c, fy_c);

  if (u->mode == MODE_GT) {
    Pass1Out center;
    if (u->scene_id == SCENE_A)      center = os_eval_scene_a_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, 0);
    else if (u->scene_id == SCENE_B) center = os_eval_scene_b_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT);
    else if (u->scene_id == SCENE_C) center = os_eval_scene_c_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, 0);
    else if (u->scene_id >= SCENE_E) center = os_eval_show_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, 0, u->scene_id);
    else                             center = os_eval_scene_d_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT);

    /* Analytical sky / emissive alcove / light panel on Scenes A, C & E need zero MC supersampling */
    if ((u->scene_id == SCENE_A && (center.gbuf_n.w < 0.5f || center.gbuf_n.w == 4.0f)) ||
        (u->scene_id == SCENE_C && center.gbuf_n.w < 0.5f) ||
        (u->scene_id >= SCENE_E && (center.gbuf_n.w < 0.5f || center.gbuf_n.w == 12.0f))) {
      return center;
    }

    v3 acc = mk3(0.0f, 0.0f, 0.0f);
    float w_sum = 0.0f;
    v3 c_nrm = mk3(center.gbuf_n.x, center.gbuf_n.y, center.gbuf_n.z);
    float c_sid = center.gbuf_n.w;
    int n_side = (u->scene_id == SCENE_D) ? 4 : 16;
    float inv_n = 1.0f / (float)n_side;
    for (int sy = 0; sy < n_side; ++sy) {
      for (int sx = 0; sx < n_side; ++sx) {
        int sidx = sy * n_side + sx;
        v2 j = os_sample_2d(px, py, u->frame_idx + 17, sidx);
        float ox = ((float)sx + j.x) * inv_n * 2.0f - 1.0f;
        float oy = ((float)sy + j.y) * inv_n * 2.0f - 1.0f;
        float w = (1.0f - f_abs(ox)) * (1.0f - f_abs(oy));
        float qx = fx_c + ox * 0.5f;
        float qy = fy_c + oy * 0.5f;
        v3 rd = os_camera_ray(u, qx, qy);
        Pass1Out sp;
        if (u->scene_id == SCENE_B) {
          /* Full unconstrained physical micro-geometry raytracing on Scene B */
          sp = os_eval_scene_b_ray(u, ro, rd, qx, qy, MODE_GT);
        } else if (u->scene_id == SCENE_D) {
          sp = os_eval_scene_d_ray(u, ro, rd, qx, qy, MODE_GT);
        } else if (u->scene_id == SCENE_A) {
          sp = os_eval_scene_a_ray(u, ro, rd, qx, qy, MODE_GT, sidx);
          /* Crease-clipped primary-surface GT: avoid silhouette cross-contamination into G-buffer masks */
          if (sp.gbuf_n.w != c_sid || v3_dot(mk3(sp.gbuf_n.x, sp.gbuf_n.y, sp.gbuf_n.z), c_nrm) <= 0.25f) {
            sp = os_eval_scene_a_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, sidx);
          }
        } else if (u->scene_id >= SCENE_E) {
          sp = os_eval_show_ray(u, ro, rd, qx, qy, MODE_GT, sidx, u->scene_id);
          if (sp.gbuf_n.w != c_sid || v3_dot(mk3(sp.gbuf_n.x, sp.gbuf_n.y, sp.gbuf_n.z), c_nrm) <= 0.25f) {
            sp = os_eval_show_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, sidx, u->scene_id);
          }
        } else {
          sp = os_eval_scene_c_ray(u, ro, rd, qx, qy, MODE_GT, sidx);
          if (sp.gbuf_n.w != c_sid || v3_dot(mk3(sp.gbuf_n.x, sp.gbuf_n.y, sp.gbuf_n.z), c_nrm) <= 0.25f) {
            sp = os_eval_scene_c_ray(u, ro, rd_c, fx_c, fy_c, MODE_GT, sidx);
          }
        }
        acc = v3_add(acc, v3_scale(mk3(sp.irrad.x, sp.irrad.y, sp.irrad.z), w));
        w_sum += w;
      }
    }
    v3 avg = v3_scale(acc, 1.0f / f_max(w_sum, 1e-6f));
    center.irrad = mk4(avg.x, avg.y, avg.z, 0.0f);
    center.albedo = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    return center;
  }

  if (u->scene_id == SCENE_A)      return os_eval_scene_a_ray(u, ro, rd_c, fx_c, fy_c, u->mode, 0);
  else if (u->scene_id == SCENE_B) return os_eval_scene_b_ray(u, ro, rd_c, fx_c, fy_c, u->mode);
  else if (u->scene_id == SCENE_C) return os_eval_scene_c_ray(u, ro, rd_c, fx_c, fy_c, u->mode, 0);
  else if (u->scene_id >= SCENE_E) return os_eval_show_ray(u, ro, rd_c, fx_c, fy_c, u->mode, 0, u->scene_id);
  else                             return os_eval_scene_d_ray(u, ro, rd_c, fx_c, fy_c, u->mode);
}

/* =========================================================================
 * Exact Previous-Frame Camera Reprojection (Zero Trig, World -> Prev Pixel)
 * ========================================================================= */
ABT_INLINE v2 os_reproject_prev(ABT_CONST FrameUniforms* u, v3 pos) {
  v3 pe = mk3(u->prev_eye[0], u->prev_eye[1], u->prev_eye[2]);
  v3 pf = mk3(u->prev_fwd[0], u->prev_fwd[1], u->prev_fwd[2]);
  v3 pr = mk3(u->prev_right[0], u->prev_right[1], u->prev_right[2]);
  v3 pu = mk3(u->prev_up[0], u->prev_up[1], u->prev_up[2]);
  v3 d = v3_sub(pos, pe);
  float z = v3_dot(d, pf);
  if (z <= 1e-4f) return mk2(-100.0f, -100.0f);
  float tan_half = u->prev_fwd[3];
  float aspect   = u->prev_right[3];
  float inv_zt = 1.0f / (z * tan_half);
  float ndc_x = (v3_dot(d, pr) * inv_zt) / aspect;
  float ndc_y =  v3_dot(d, pu) * inv_zt;
  float px = (0.5f * ndc_x + 0.5f) * (float)u->width - 0.5f;
  float py = (0.5f - 0.5f * ndc_y) * (float)u->height - 0.5f;
  return mk2(px, py);
}

/* =========================================================================
 * PASS 2 (OS-ABT SINGLE-FRAME): 2-Pass Separable H + V Strictly Co-Surface
 * Psi_perp Indirect Reconstruction (Single Packed 8B half4 G-Buffer!)
 * ========================================================================= */
ABT_INLINE float os_spatial_w(int ad) {
  const float wt[9] = {
    1.00000000f, 0.98216197f, 0.93058574f, 0.85097547f,
    0.75217830f, 0.64464142f, 0.53822699f, 0.44034218f, 0.35516991f
  };
  return wt[ad];
}

ABT_INLINE void os_pass2_osabt_spatial(
    ABT_CONST FrameUniforms* u, int px, int py, int horiz, int is_final,
    ABT_RO_GLOBAL h4* src_irrad,
    ABT_RO_GLOBAL h4* cur_direct,
    ABT_RO_GLOBAL u16_4* cur_albedo,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_GLOBAL h4* dst_irrad,
    ABT_GLOBAL v4* out_rgb)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 ci = h4_to_v4(src_irrad[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  float sid_p = gp.w;
  if (ci.w < 0.5f || sid_p < 0.5f) {
    if (!is_final) dst_irrad[idx] = src_irrad[idx];
    else out_rgb[idx] = mk4(ci.x, ci.y, ci.z, 1.0f);
    return;
  }
  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp = os_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  float step_s = horiz ? ((2.0f * u->cam_right[3] * u->cam_fwd[3]) / (float)W)
                       : ((-2.0f * u->cam_fwd[3]) / (float)H);
  v3 dR = horiz ? mk3(u->cam_right[0] * step_s, u->cam_right[1] * step_s, u->cam_right[2] * step_s)
                : mk3(u->cam_up[0] * step_s,    u->cam_up[1] * step_s,    u->cam_up[2] * step_s);

  v3 acc = mk3(ci.x, ci.y, ci.z);
  float w_sum = 1.0f;
  const float inv_tol = 16.666666f; /* 1 / 0.06 */
  for (int d = -8; d <= 8; ++d) {
    if (d == 0) continue;
    int qx = horiz ? (px + d) : px;
    int qy = horiz ? py : (py + d);
    if (qx < 0 || qx >= W || qy < 0 || qy >= H) continue;
    int qidx = qy * W + qx;
    v4 gq = h4_to_v4(cur_gbuf[qidx]);
    if (f_abs(sid_p - gq.w) > 0.25f) continue;
    v3 nq;
    float w_n;
    if (f_abs(gq.x - gp.x) + f_abs(gq.y - gp.y) < 1e-4f) {
      nq = np;
      w_n = 1.0f;
    } else {
      nq = os_oct_decode(gq.x, gq.y);
      w_n = os_stiefel_null(v3_dot(np, nq));
      if (w_n <= 0.0f) continue;
    }
    float zq = gq.z;
    v3 dx = v3_add(v3_scale(rp, zq - zp), v3_scale(dR, zq * (float)d));
    float psi = os_psi_perp_plane(dx, np, nq, w_n, inv_tol);
    if (psi <= 0.0f) continue;
    float w = os_spatial_w(d < 0 ? -d : d) * psi;
    v4 qi = h4_to_v4(src_irrad[qidx]);
    acc = v3_add(acc, v3_scale(mk3(qi.x, qi.y, qi.z), w));
    w_sum += w;
  }
  v3 filt = v3_scale(acc, 1.0f / f_max(w_sum, 1e-6f));
  if (!is_final) {
    dst_irrad[idx] = v4_to_h4(mk4(filt.x, filt.y, filt.z, 1.0f));
  } else {
    v4 cd = h4_to_v4(cur_direct[idx]);
    v4 ca = u16_to_v4(cur_albedo[idx]);
    v3 col = v3_add(mk3(cd.x, cd.y, cd.z), v3_mul(mk3(ca.x, ca.y, ca.z), filt));
    out_rgb[idx] = mk4(col.x, col.y, col.z, 1.0f);
  }
}

/* =========================================================================
 * PASS 2A (OS-ABT TEMPORAL): Cache-Coherent Bilinear Geodesic Reprojection +
 * 3x3 Stride-1 Co-Surface Psi_perp Cell Moments (Single 8B Packed G-Buffer!)
 * ========================================================================= */
ABT_INLINE void os_pass2a_osabt_reproj(
    ABT_CONST FrameUniforms* u, int px, int py,
    ABT_RO_GLOBAL h4* cur_irrad,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_RO_GLOBAL st4* prev_state,
    ABT_RO_GLOBAL h4* prev_gbuf,
    ABT_GLOBAL rp4* out_reproj,
    ABT_GLOBAL h4* out_sp1)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 ci = h4_to_v4(cur_irrad[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  float sid_p = gp.w;
  if (ci.w < 0.5f || sid_p < 0.5f) {
    v4 zero_v = mk4(ci.x, ci.y, ci.z, 0.0f);
    out_reproj[idx] = rp4_store(zero_v);
    out_sp1[idx]    = v4_to_h4(zero_v);
    return;
  }
  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp_ray = os_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  v3 xp = mk3(u->cam_eye[0] + rp_ray.x * zp,
              u->cam_eye[1] + rp_ray.y * zp,
              u->cam_eye[2] + rp_ray.z * zp);

  /* 1. Single cache-coherent bilinear geodesic Psi_perp-gated history reprojection */
  v3 s_prev = mk3(0.0f, 0.0f, 0.0f);
  float prev_age = 0.0f;
  if (u->frame_idx > 0 && u->total_frames > 1) {
    v2 rp = os_reproject_prev(u, xp);
    int x0 = (int)f_floor(rp.x);
    int y0 = (int)f_floor(rp.y);
    float fx = rp.x - (float)x0;
    float fy = rp.y - (float)y0;
    float bw[4] = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy };
    int bx[4] = { x0, x0 + 1, x0, x0 + 1 };
    int by[4] = { y0, y0, y0 + 1, y0 + 1 };
    float rw_sum = 0.0f;
    const float inv_tol_rep = 12.5f; /* 1 / 0.08 */
    for (int b = 0; b < 4; ++b) {
      if (bw[b] < 1e-4f) continue;
      if (bx[b] < 0 || bx[b] >= W || by[b] < 0 || by[b] >= H) continue;
      int ridx = by[b] * W + bx[b];
      v4 pg = h4_to_v4(prev_gbuf[ridx]);
      if (f_abs(sid_p - pg.w) > 0.25f) continue;
      v3 pn_p;
      float w_n;
      if (f_abs(pg.x - gp.x) + f_abs(pg.y - gp.y) < 1e-4f) {
        pn_p = np;
        w_n = 1.0f;
      } else {
        pn_p = os_oct_decode(pg.x, pg.y);
        w_n = os_stiefel_null(v3_dot(np, pn_p));
        if (w_n <= 0.0f) continue;
      }
      v3 x_prev = os_pos_from_z_prev(u, bx[b], by[b], pg.z);
      float rpsi = os_psi_perp_plane(v3_sub(x_prev, xp), np, pn_p, w_n, inv_tol_rep);
      float w = bw[b] * rpsi;
      if (w > 1e-5f) {
        v4 ps4 = st4_load(prev_state[ridx]);
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
  out_reproj[idx] = rp4_store(mk4(s_prev.x, s_prev.y, s_prev.z, prev_age));

  /* 2. 3x3 Stride-1 Co-Surface Psi_perp Cell Mean & Exact Kish Cell Variance */
  float sx = (2.0f * u->cam_right[3] * u->cam_fwd[3]) / (float)W;
  float sy = (-2.0f * u->cam_fwd[3]) / (float)H;
  v3 dRx = mk3(u->cam_right[0] * sx, u->cam_right[1] * sx, u->cam_right[2] * sx);
  v3 dRy = mk3(u->cam_up[0] * sy,    u->cam_up[1] * sy,    u->cam_up[2] * sy);

  const float kw[3] = { 0.80f, 1.0f, 0.80f };
  const float inv_tol = 16.666666f;
  v3 uc = mk3(ci.x, ci.y, ci.z);
  v3 m1 = uc;
  v3 m2 = v3_mul(uc, uc);
  float w_sum = 1.0f, w2_sum = 1.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    int qy = py + dy;
    if (qy < 0 || qy >= H) continue;
    float wy = kw[dy + 1];
    v3 dyR = v3_scale(dRy, (float)dy);
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int qx = px + dx;
      if (qx < 0 || qx >= W) continue;
      int qidx = qy * W + qx;
      v4 gq = h4_to_v4(cur_gbuf[qidx]);
      if (f_abs(sid_p - gq.w) > 0.25f) continue;
      v3 nq;
      float w_n;
      if (f_abs(gq.x - gp.x) + f_abs(gq.y - gp.y) < 1e-4f) {
        nq = np;
        w_n = 1.0f;
      } else {
        nq = os_oct_decode(gq.x, gq.y);
        w_n = os_stiefel_null(v3_dot(np, nq));
        if (w_n <= 0.0f) continue;
      }
      float zq = gq.z;
      v3 dpos = v3_add(v3_scale(rp_ray, zq - zp), v3_scale(v3_add(dyR, v3_scale(dRx, (float)dx)), zq));
      float psi = os_psi_perp_plane(dpos, np, nq, w_n, inv_tol);
      if (psi <= 0.0f) continue;
      float w = wy * kw[dx + 1] * psi;
      v4 qi = h4_to_v4(cur_irrad[qidx]);
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
  out_sp1[idx] = v4_to_h4(mk4(u_sp1.x, u_sp1.y, u_sp1.z, std_cell));
}

/* =========================================================================
 * PASS 2G1 (OS-ABT TEMPORAL): Deterministic 16x16 Tile Affine-Gauge Reduction
 * (CPU reference; fused into Pass 2A via hardware simd_sum on Apple M1 GPU!)
 * ========================================================================= */
ABT_INLINE void os_pass2g1_tile_reduce(
    ABT_CONST FrameUniforms* u, int tx, int ty,
    ABT_RO_GLOBAL rp4* tmp_reproj,
    ABT_RO_GLOBAL h4* tmp_sp1,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_GLOBAL v4* tile_su,
    ABT_GLOBAL v4* tile_ss,
    ABT_GLOBAL v4* tile_ss2,
    ABT_GLOBAL v4* tile_sus)
{
  int W = u->width, H = u->height;
  int TW = (W + 15) / 16;
  int tidx = ty * TW + tx;
  v3 su = mk3(0.0f, 0.0f, 0.0f), ss = mk3(0.0f, 0.0f, 0.0f);
  v3 ss2 = mk3(0.0f, 0.0f, 0.0f), sus = mk3(0.0f, 0.0f, 0.0f);
  float w_cnt = 0.0f, w2_v = 0.0f;

  if (u->frame_idx > 0 && u->total_frames > 1) {
    int bx = tx * 16;
    int by = ty * 16;
    for (int iy = 0; iy < 16; iy += 2) {
      int py = by + iy;
      if (py >= H) break;
      for (int ix = 0; ix < 16; ix += 2) {
        int px = bx + ix;
        if (px >= W) break;
        int idx = py * W + px;
        v4 gp = h4_to_v4(cur_gbuf[idx]);
        if (gp.w < 0.5f) continue;
        v4 rep = rp4_load(tmp_reproj[idx]);
        if (rep.w < 0.5f) continue;
        v4 sp = h4_to_v4(tmp_sp1[idx]);
        float wa = f_min(1.0f, rep.w * 0.25f);
        v3 uv = mk3(sp.x, sp.y, sp.z);
        v3 sv = mk3(rep.x, rep.y, rep.z);
        su  = v3_add(su,  v3_scale(uv, wa));
        ss  = v3_add(ss,  v3_scale(sv, wa));
        ss2 = v3_add(ss2, v3_scale(v3_mul(sv, sv), wa));
        sus = v3_add(sus, v3_scale(v3_mul(uv, sv), wa));
        w_cnt += wa;
        w2_v  += (wa * wa) * (sp.w * sp.w);
      }
    }
  }
  tile_su[tidx]  = mk4(su.x,  su.y,  su.z,  w_cnt);
  tile_ss[tidx]  = mk4(ss.x,  ss.y,  ss.z,  w2_v);
  tile_ss2[tidx] = mk4(ss2.x, ss2.y, ss2.z, 0.0f);
  tile_sus[tidx] = mk4(sus.x, sus.y, sus.z, 0.0f);
}

/* =========================================================================
 * PASS 2G2 (OS-ABT TEMPORAL): Regional Ridge-Regularized Affine Gauge Solver
 * Fits E_n = kappa * s_{n-1} + beta over ~5,000 pixels per tile (L1-resident!)
 * ========================================================================= */
ABT_INLINE void os_pass2g2_tile_gauge(
    ABT_CONST FrameUniforms* u, int tx, int ty,
    ABT_RO_GLOBAL v4* tile_su,
    ABT_RO_GLOBAL v4* tile_ss,
    ABT_RO_GLOBAL v4* tile_ss2,
    ABT_RO_GLOBAL v4* tile_sus,
    ABT_GLOBAL v4* tile_kap,
    ABT_GLOBAL v4* tile_bet)
{
  int TW = (u->width + 15) / 16;
  int TH = (u->height + 15) / 16;
  int tidx = ty * TW + tx;
  if (u->frame_idx == 0 || u->total_frames <= 1) {
    tile_kap[tidx] = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    tile_bet[tidx] = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  v3 Su = mk3(0.0f, 0.0f, 0.0f), Ss = mk3(0.0f, 0.0f, 0.0f);
  v3 Ss2 = mk3(0.0f, 0.0f, 0.0f), Sus = mk3(0.0f, 0.0f, 0.0f);
  float Wc = 0.0f, Wv = 0.0f;

  /* 9x9 strided-2 tile window (~288x288 px = up to 5,184 sampled pixels) */
  for (int dy = -4; dy <= 4; ++dy) {
    int qy = ty + dy * 2;
    if (qy < 0 || qy >= TH) continue;
    for (int dx = -4; dx <= 4; ++dx) {
      int qx = tx + dx * 2;
      if (qx < 0 || qx >= TW) continue;
      int qidx = qy * TW + qx;
      float wt = f_rat_exp_neg(0.08f * (float)(dx * dx + dy * dy));
      v4 tu  = tile_su[qidx];
      v4 ts  = tile_ss[qidx];
      v4 ts2 = tile_ss2[qidx];
      v4 tus = tile_sus[qidx];
      Su  = v3_add(Su,  v3_scale(mk3(tu.x,  tu.y,  tu.z),  wt));
      Ss  = v3_add(Ss,  v3_scale(mk3(ts.x,  ts.y,  ts.z),  wt));
      Ss2 = v3_add(Ss2, v3_scale(mk3(ts2.x, ts2.y, ts2.z), wt));
      Sus = v3_add(Sus, v3_scale(mk3(tus.x, tus.y, tus.z), wt));
      Wc += wt * tu.w;
      Wv += (wt * wt) * ts.w;
    }
  }

  if (Wc < 8.0f) {
    tile_kap[tidx] = mk4(1.0f, 1.0f, 1.0f, 0.0f);
    tile_bet[tidx] = mk4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  float inv_W = 1.0f / Wc;
  v3 u_bar = v3_scale(Su, inv_W);
  v3 s_bar = v3_scale(Ss, inv_W);
  v3 d_bar = v3_sub(u_bar, s_bar);
  float d2 = v3_dot(d_bar, d_bar);
  float var_ubar = (2.5f * Wv) * (inv_W * inv_W) + 0.0025f;
  float z2_tile = d2 / var_ubar;
  float a_g = f_clamp((z2_tile - 6.0f) * 0.125f, 0.0f, 1.0f);

  /* Exact Ridge-regularized per-channel affine gauge anchored at ratio kappa0 = u_bar/s_bar */
  float ux[3] = { u_bar.x, u_bar.y, u_bar.z };
  float sx[3] = { s_bar.x, s_bar.y, s_bar.z };
  float s2x[3] = { Ss2.x * inv_W, Ss2.y * inv_W, Ss2.z * inv_W };
  float usx[3] = { Sus.x * inv_W, Sus.y * inv_W, Sus.z * inv_W };
  float kv[3], bv[3];
  for (int c = 0; c < 3; ++c) {
    float var_s  = f_max(0.0f, s2x[c] - sx[c] * sx[c]);
    float cov_su = usx[c] - sx[c] * ux[c];
    float k0     = (ux[c] + 0.01f) / (sx[c] + 0.01f);
    float lam    = 0.06f * (sx[c] * sx[c] + 0.02f);
    float k      = f_clamp((cov_su + lam * k0) / (var_s + lam), 0.05f, 20.0f);
    kv[c] = k;
    bv[c] = ux[c] - k * sx[c];
  }
  tile_kap[tidx] = mk4(kv[0], kv[1], kv[2], a_g);
  tile_bet[tidx] = mk4(bv[0], bv[1], bv[2], 0.0f);
}

/* =========================================================================
 * PASS 2B (OS-ABT TEMPORAL): Stride-3 Disjoint-Cell Co-Surface Hierarchy +
 * Bilinear Regional Affine Gauge + L-Stable Port-Hamiltonian ZDF-TPT!
 * ========================================================================= */
ABT_INLINE void os_pass2b_osabt_zdf(
    ABT_CONST FrameUniforms* u, int px, int py,
    ABT_RO_GLOBAL u16_4* cur_albedo,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_RO_GLOBAL rp4* tmp_reproj,
    ABT_RO_GLOBAL h4* tmp_sp1,
    ABT_RO_GLOBAL v4* tile_kap,
    ABT_RO_GLOBAL v4* tile_bet,
    ABT_GLOBAL st4* out_state,
    ABT_GLOBAL v4* out_rgb)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 ca = u16_to_v4(cur_albedo[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  float sid_p = gp.w;
  v4 sp1_c = h4_to_v4(tmp_sp1[idx]);
  if (ca.w < 0.5f || sid_p < 0.5f) {
    v3 col = mk3(sp1_c.x * ca.x, sp1_c.y * ca.y, sp1_c.z * ca.z);
    out_state[idx] = st4_store(mk4(sp1_c.x, sp1_c.y, sp1_c.z, 1.0f));
    out_rgb[idx]   = mk4(col.x, col.y, col.z, 1.0f);
    return;
  }

  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp_ray = os_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  float sx = (2.0f * u->cam_right[3] * u->cam_fwd[3]) / (float)W;
  float sy = (-2.0f * u->cam_fwd[3]) / (float)H;
  v3 dRx3 = mk3(u->cam_right[0] * (3.0f * sx), u->cam_right[1] * (3.0f * sx), u->cam_right[2] * (3.0f * sx));
  v3 dRy3 = mk3(u->cam_up[0] * (3.0f * sy),    u->cam_up[1] * (3.0f * sy),    u->cam_up[2] * (3.0f * sy));

  v3 alb = mk3(ca.x, ca.y, ca.z);
  v4 rep_c = rp4_load(tmp_reproj[idx]);
  v3 s_prev = mk3(rep_c.x, rep_c.y, rep_c.z);
  float prev_age = rep_c.w;
  v3 u_sp1 = mk3(sp1_c.x, sp1_c.y, sp1_c.z);
  float v_cell_c = sp1_c.w * sp1_c.w;

  /* 9-tap stride-3 gather over DISJOINT 3x3 co-surface cells (81 pixels, zero reproj reads!) */
  const float kw[3] = { 0.72f, 1.0f, 0.72f };
  const float inv_tol = 16.666666f;
  v3 u81_num = u_sp1;
  v3 u81_sq  = v3_mul(u_sp1, u_sp1);
  float w_u = 1.0f, w2_u = 1.0f, v_u81_acc = v_cell_c;

  for (int dy = -1; dy <= 1; ++dy) {
    int qy = py + dy * 3;
    if (qy < 0 || qy >= H) continue;
    float wy = kw[dy + 1];
    v3 dyR = v3_scale(dRy3, (float)dy);
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int qx = px + dx * 3;
      if (qx < 0 || qx >= W) continue;
      int qidx = qy * W + qx;
      v4 gq = h4_to_v4(cur_gbuf[qidx]);
      if (f_abs(sid_p - gq.w) > 0.25f) continue;
      v3 nq;
      float w_n;
      if (f_abs(gq.x - gp.x) + f_abs(gq.y - gp.y) < 1e-4f) {
        nq = np;
        w_n = 1.0f;
      } else {
        nq = os_oct_decode(gq.x, gq.y);
        w_n = os_stiefel_null(v3_dot(np, nq));
        if (w_n <= 0.0f) continue;
      }
      float zq = gq.z;
      v3 dpos = v3_add(v3_scale(rp_ray, zq - zp), v3_scale(v3_add(dyR, v3_scale(dRx3, (float)dx)), zq));
      float psi = os_psi_perp_plane(dpos, np, nq, w_n, inv_tol);
      if (psi <= 0.0f) continue;
      float w = wy * kw[dx + 1] * psi;
      v4 sq4 = h4_to_v4(tmp_sp1[qidx]);
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
    /* Smooth bilinear read from L1-resident 80x45 regional affine gauge field */
    int TW = (W + 15) / 16;
    int TH = (H + 15) / 16;
    float gx = ((float)px + 0.5f) * (1.0f / 16.0f) - 0.5f;
    float gy = ((float)py + 0.5f) * (1.0f / 16.0f) - 0.5f;
    int tx0 = (int)f_floor(gx);
    int ty0 = (int)f_floor(gy);
    float tfx = gx - (float)tx0;
    float tfy = gy - (float)ty0;
    int cx0 = (tx0 < 0) ? 0 : ((tx0 >= TW) ? (TW - 1) : tx0);
    int cx1 = (tx0 + 1 < 0) ? 0 : ((tx0 + 1 >= TW) ? (TW - 1) : (tx0 + 1));
    int cy0 = (ty0 < 0) ? 0 : ((ty0 >= TH) ? (TH - 1) : ty0);
    int cy1 = (ty0 + 1 < 0) ? 0 : ((ty0 + 1 >= TH) ? (TH - 1) : (ty0 + 1));
    float w00 = (1.0f - tfx) * (1.0f - tfy), w10 = tfx * (1.0f - tfy);
    float w01 = (1.0f - tfx) * tfy,          w11 = tfx * tfy;
    int i00 = cy0 * TW + cx0, i10 = cy0 * TW + cx1;
    int i01 = cy1 * TW + cx0, i11 = cy1 * TW + cx1;

    v4 k00 = tile_kap[i00], k10 = tile_kap[i10], k01 = tile_kap[i01], k11 = tile_kap[i11];
    v4 b00 = tile_bet[i00], b10 = tile_bet[i10], b01 = tile_bet[i01], b11 = tile_bet[i11];
    v3 kappa = mk3(k00.x*w00 + k10.x*w10 + k01.x*w01 + k11.x*w11,
                   k00.y*w00 + k10.y*w10 + k01.y*w01 + k11.y*w11,
                   k00.z*w00 + k10.z*w10 + k01.z*w01 + k11.z*w11);
    float a_g = k00.w*w00 + k10.w*w10 + k01.w*w01 + k11.w*w11;
    v3 beta  = mk3(b00.x*w00 + b10.x*w10 + b01.x*w01 + b11.x*w11,
                   b00.y*w00 + b10.y*w10 + b01.y*w01 + b11.y*w11,
                   b00.z*w00 + b10.z*w10 + b01.z*w01 + b11.z*w11);

    v3 s_gauge = v3_max(mk3(0.0f, 0.0f, 0.0f), v3_add(v3_mul(kappa, s_prev), beta));
    v3 s_base = v3_mix(s_prev, s_gauge, a_g);

    /* Local co-surface Kish shock check on 81-pixel mean u_81 vs gauge-transported s_base */
    v3 d_loc = v3_sub(u_81, s_base);
    float var_loc = (v_u81_acc + 0.30f * var_inter * w2_u) * (inv_wu * inv_wu) + 0.018f;
    float z2_loc = v3_dot(d_loc, d_loc) / var_loc;
    float r_shock = f_max(0.0f, (z2_loc - 16.0f) * 0.25f);

    float g_min = f_max(0.035f, 0.5f / (prev_age + 1.0f));
    float g_n = g_min + 0.04f * a_g + 0.80f * r_shock;
    float G_n = g_n / (1.0f + g_n);

    float sp_blend = f_clamp(G_n * 2.5f + 0.25f * a_g, 0.0f, 1.0f);
    v3 u_eff = v3_mix(u_sp1, u_81, sp_blend);

    /* Theorem 4: Exact L-Stable Port-Hamiltonian ZDF-TPT (beta_n = G_n => rho = (1 + g_n)^-2) */
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
  out_state[idx] = st4_store(mk4(s_new.x, s_new.y, s_new.z, age_new));
  out_rgb[idx]   = mk4(col.x, col.y, col.z, 1.0f);
}

/* =========================================================================
 * PASS 2 (ABLATED NEGATIVE CONTROL): Naive Gaussian Bilateral + Explicit TAA
 * ========================================================================= */
ABT_INLINE void os_pass2_ablated_pixel(
    ABT_CONST FrameUniforms* u, int px, int py,
    ABT_RO_GLOBAL h4* cur_irrad,
    ABT_RO_GLOBAL u16_4* cur_albedo,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_RO_GLOBAL h4* prev_state,
    ABT_GLOBAL h4* out_state,
    ABT_GLOBAL v4* out_rgb)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 ci = h4_to_v4(cur_irrad[idx]);
  v4 ca = u16_to_v4(cur_albedo[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  if (ci.w < 0.5f || gp.w < 0.5f) {
    v3 col = mk3(ci.x * ca.x, ci.y * ca.y, ci.z * ca.z);
    out_state[idx] = v4_to_h4(mk4(col.x, col.y, col.z, 1.0f));
    out_rgb[idx]   = mk4(col.x, col.y, col.z, 1.0f);
    return;
  }
  v3 xp = os_pos_from_z_cur(u, px, py, gp.z);
  v3 np = os_oct_decode(gp.x, gp.y);

  v3 acc = mk3(0.0f, 0.0f, 0.0f);
  float wsum = 0.0f;
  for (int dy = -2; dy <= 2; ++dy) {
    int qy = py + dy;
    if (qy < 0 || qy >= H) continue;
    for (int dx = -2; dx <= 2; ++dx) {
      int qx = px + dx;
      if (qx < 0 || qx >= W) continue;
      int qidx = qy * W + qx;
      v4 gq = h4_to_v4(cur_gbuf[qidx]);
      v3 nq = (gq.w < 0.5f) ? mk3(0.0f, 0.0f, 0.0f) : os_oct_decode(gq.x, gq.y);
      v3 dn = v3_sub(np, nq);
      float w = f_rat_exp_neg(v3_dot(dn, dn) * 1.1111111f) * f_rat_exp_neg(0.18f * (float)(dx*dx + dy*dy));
      v4 qi = h4_to_v4(cur_irrad[qidx]);
      v4 qa = u16_to_v4(cur_albedo[qidx]);
      v3 qrgb = mk3(qi.x * qa.x, qi.y * qa.y, qi.z * qa.z);
      acc = v3_add(acc, v3_scale(qrgb, w));
      wsum += w;
    }
  }
  v3 cur_rgb = v3_scale(acc, 1.0f / f_max(wsum, 1e-6f));
  if (u->frame_idx > 0 && u->total_frames > 1) {
    v2 rp = os_reproject_prev(u, xp);
    int rx = (int)f_floor(rp.x + 0.5f);
    int ry = (int)f_floor(rp.y + 0.5f);
    if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
      v4 ps = h4_to_v4(prev_state[ry * W + rx]);
      cur_rgb = v3_mix(mk3(ps.x, ps.y, ps.z), cur_rgb, 0.10f);
    }
  }
  out_state[idx] = v4_to_h4(mk4(cur_rgb.x, cur_rgb.y, cur_rgb.z, 1.0f));
  out_rgb[idx]   = mk4(cur_rgb.x, cur_rgb.y, cur_rgb.z, 1.0f);
}

/* =========================================================================
 * PASS 2 (CLASSICAL REAL-TIME BASELINE): TAA Variance Clamp + 2-Pass SVGF A-Trous
 * ========================================================================= */
ABT_INLINE void os_pass2a_baseline_taa(
    ABT_CONST FrameUniforms* u, int px, int py,
    ABT_RO_GLOBAL h4* cur_irrad,
    ABT_RO_GLOBAL u16_4* cur_albedo,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_RO_GLOBAL h4* prev_state,
    ABT_GLOBAL h4* out_taa)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 ci = h4_to_v4(cur_irrad[idx]);
  v4 ca = u16_to_v4(cur_albedo[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  v3 cur_rgb = mk3(ci.x * ca.x, ci.y * ca.y, ci.z * ca.z);
  if (ci.w < 0.5f || gp.w < 0.5f || u->frame_idx == 0 || u->total_frames <= 1) {
    out_taa[idx] = v4_to_h4(mk4(cur_rgb.x, cur_rgb.y, cur_rgb.z, ci.w));
    return;
  }
  v3 xp = os_pos_from_z_cur(u, px, py, gp.z);

  v3 m1 = mk3(0.0f, 0.0f, 0.0f), m2 = mk3(0.0f, 0.0f, 0.0f);
  float cnt = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    int qy = py + dy;
    if (qy < 0 || qy >= H) continue;
    for (int dx = -1; dx <= 1; ++dx) {
      int qx = px + dx;
      if (qx < 0 || qx >= W) continue;
      int qidx = qy * W + qx;
      v4 qi = h4_to_v4(cur_irrad[qidx]);
      v4 qa = u16_to_v4(cur_albedo[qidx]);
      v3 c = mk3(qi.x * qa.x, qi.y * qa.y, qi.z * qa.z);
      m1 = v3_add(m1, c);
      m2 = v3_add(m2, v3_mul(c, c));
      cnt += 1.0f;
    }
  }
  float inv_cnt = 1.0f / cnt;
  v3 mu = v3_scale(m1, inv_cnt);
  v3 sg = mk3(f_sqrt(f_max(0.0f, m2.x * inv_cnt - mu.x * mu.x)),
              f_sqrt(f_max(0.0f, m2.y * inv_cnt - mu.y * mu.y)),
              f_sqrt(f_max(0.0f, m2.z * inv_cnt - mu.z * mu.z)));
  v3 box_lo = v3_sub(mu, v3_scale(sg, 1.5f));
  v3 box_hi = v3_add(mu, v3_scale(sg, 1.5f));

  v2 rp = os_reproject_prev(u, xp);
  int rx = (int)f_floor(rp.x + 0.5f);
  int ry = (int)f_floor(rp.y + 0.5f);
  v3 out_c = cur_rgb;
  if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
    v4 ps = h4_to_v4(prev_state[ry * W + rx]);
    v3 hist = v3_clamp(mk3(ps.x, ps.y, ps.z), box_lo, box_hi);
    out_c = v3_mix(hist, cur_rgb, 0.10f);
  }
  out_taa[idx] = v4_to_h4(mk4(out_c.x, out_c.y, out_c.z, ci.w));
}

ABT_INLINE void os_pass2b_baseline_atrous(
    ABT_CONST FrameUniforms* u, int px, int py, int step, int is_final,
    ABT_RO_GLOBAL h4* src_rgb,
    ABT_RO_GLOBAL h4* cur_gbuf,
    ABT_GLOBAL h4* dst_pack,
    ABT_GLOBAL v4* out_rgb)
{
  int W = u->width, H = u->height;
  int idx = py * W + px;
  v4 center = h4_to_v4(src_rgb[idx]);
  v4 gp = h4_to_v4(cur_gbuf[idx]);
  if (center.w < 0.5f || gp.w < 0.5f) {
    dst_pack[idx] = src_rgb[idx];
    if (is_final) out_rgb[idx] = mk4(center.x, center.y, center.z, 1.0f);
    return;
  }
  v3 np = os_oct_decode(gp.x, gp.y);
  float zp = gp.z;
  v3 rp_ray = os_camera_ray_unnorm(u, (float)px + 0.5f, (float)py + 0.5f);
  float sx = (2.0f * u->cam_right[3] * u->cam_fwd[3]) / (float)W;
  float sy = (-2.0f * u->cam_fwd[3]) / (float)H;
  v3 dRx = mk3(u->cam_right[0] * (sx * (float)step), u->cam_right[1] * (sx * (float)step), u->cam_right[2] * (sx * (float)step));
  v3 dRy = mk3(u->cam_up[0] * (sy * (float)step),    u->cam_up[1] * (sy * (float)step),    u->cam_up[2] * (sy * (float)step));

  float lc = os_luma(mk3(center.x, center.y, center.z));
  const float k1d[3] = { 0.25f, 0.50f, 0.25f };
  v3 acc = mk3(0.0f, 0.0f, 0.0f);
  float wsum = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    int qy = py + dy * step;
    if (qy < 0 || qy >= H) continue;
    v3 dyR = v3_scale(dRy, (float)dy);
    for (int dx = -1; dx <= 1; ++dx) {
      int qx = px + dx * step;
      if (qx < 0 || qx >= W) continue;
      int qidx = qy * W + qx;
      v4 qv = h4_to_v4(src_rgb[qidx]);
      v4 gq = h4_to_v4(cur_gbuf[qidx]);
      v3 nq = (gq.w < 0.5f) ? mk3(0.0f, 0.0f, 0.0f)
            : ((f_abs(gq.x - gp.x) + f_abs(gq.y - gp.y) < 1e-4f) ? np : os_oct_decode(gq.x, gq.y));
      v3 dn = v3_sub(np, nq);
      v3 dp = v3_add(v3_scale(rp_ray, gq.z - zp), v3_scale(v3_add(dyR, v3_scale(dRx, (float)dx)), gq.z));
      float dl = f_abs(lc - os_luma(mk3(qv.x, qv.y, qv.z)));
      float w = k1d[dx + 1] * k1d[dy + 1]
              * f_rat_exp_neg(v3_dot(dn, dn) * 1.8181818f + v3_dot(dp, dp) * 2.8571429f + dl * 1.5384615f);
      acc = v3_add(acc, v3_scale(mk3(qv.x, qv.y, qv.z), w));
      wsum += w;
    }
  }
  v3 res = v3_scale(acc, 1.0f / f_max(wsum, 1e-6f));
  dst_pack[idx] = v4_to_h4(mk4(res.x, res.y, res.z, center.w));
  if (is_final) out_rgb[idx] = mk4(res.x, res.y, res.z, 1.0f);
}

#endif /* OSABT_CORE_H */
