/*
 * Slice 3: Native Apple M1 Metal Runner for OS-ABT (v4 Architecture)
 * Hardware GPU timestamps (GPUEndTime - GPUStartTime), Private GPU-local 8B half4 scratch,
 * threadgroup SRAM tiling + fused simd_sum tile reduction, strict IEEE-754 parity.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "scene_setup.h"
#include "mesh_bvh.h"

static NSString* load_combined_shader(void) {
  NSError* err = nil;
  NSString* core = [NSString stringWithContentsOfFile:@"src/osabt_core.h"
                                             encoding:NSUTF8StringEncoding error:&err];
  if (!core) {
    fprintf(stderr, "Failed to read src/osabt_core.h: %s\n", [[err localizedDescription] UTF8String]);
    return nil;
  }
  NSString* kernels = [NSString stringWithContentsOfFile:@"src/osabt_shaders.metal"
                                                encoding:NSUTF8StringEncoding error:&err];
  if (!kernels) {
    fprintf(stderr, "Failed to read src/osabt_shaders.metal: %s\n", [[err localizedDescription] UTF8String]);
    return nil;
  }
  return [NSString stringWithFormat:@"%@\n%@\n", core, kernels];
}

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
  NSError* err = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) { fprintf(stderr, "Missing Metal kernel %s\n", [name UTF8String]); exit(2); }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    fprintf(stderr, "PSO error %s: %s\n", [name UTF8String], [[err localizedDescription] UTF8String]);
    exit(2);
  }
  return pso;
}

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pso_p1_gt;
  id<MTLComputePipelineState> pso_p1_b;
  id<MTLComputePipelineState> pso_p1_d;
  id<MTLComputePipelineState> pso_p1_a;
  id<MTLComputePipelineState> pso_p1_c;
  id<MTLComputePipelineState> pso_osabt_sph;
  id<MTLComputePipelineState> pso_osabt_spv;
  id<MTLComputePipelineState> pso_osabt_2a;
  id<MTLComputePipelineState> pso_osabt_g2;
  id<MTLComputePipelineState> pso_osabt_2b;
  id<MTLComputePipelineState> pso_abl;
  id<MTLComputePipelineState> pso_taa;
  id<MTLComputePipelineState> pso_atr;
  /* GPU-Private hardware half4 scratch buffers (8 B/px; single packed G-buffer!) */
  id<MTLBuffer> buf_irrad;
  id<MTLBuffer> buf_direct;
  id<MTLBuffer> buf_albedo;
  id<MTLBuffer> buf_gbuf0;
  id<MTLBuffer> buf_gbuf1;
  id<MTLBuffer> buf_st0;
  id<MTLBuffer> buf_st1;
  id<MTLBuffer> buf_tmp_a;
  id<MTLBuffer> buf_tmp_b;
  id<MTLBuffer> buf_reproj;  /* rp4 reprojected history (Scene C OS-ABT temporal path only) */
  size_t st_bytes;           /* byte length of each history buffer (st4 on the OS-ABT temporal path, else h4) */
  /* GPU-Private L2-resident coarse tile buffers (80x45 x 16B = 57.6 KB each) */
  id<MTLBuffer> buf_tile_su;
  id<MTLBuffer> buf_tile_ss;
  id<MTLBuffer> buf_tile_ss2;
  id<MTLBuffer> buf_tile_sus;
  id<MTLBuffer> buf_tile_kap;
  id<MTLBuffer> buf_tile_bet;
  /* Shared CPU-readable output & optional G-buffer dump */
  id<MTLBuffer> buf_gn_dump;
  id<MTLBuffer> buf_gp_dump;
  id<MTLBuffer> buf_out;
  /* Mesh scene (Scene H): BVH nodes + triangle arrays, read-only on the GPU */
  id<MTLBuffer> buf_mesh_nodes;
  id<MTLBuffer> buf_mesh_tri_a;
  id<MTLBuffer> buf_mesh_tri_n;
  int is_mesh;
  MTLSize grid;
  MTLSize tg;
  MTLSize tg_grid16;
  MTLSize tg_grid_h;
  MTLSize tg_h;
  MTLSize tile_grid;
  MTLSize tile_tg;
} MetalCtx;

static int is_direct_pass(const RunConfig* cfg) {
  return (cfg->mode == MODE_GT || cfg->scene_id == SCENE_B || cfg->scene_id == SCENE_D) ? 1 : 0;
}

static void zero_private_history(const MetalCtx* m, size_t sz8) {
  size_t st_sz = m->st_bytes;
  id<MTLCommandBuffer> cb = [m->queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl fillBuffer:m->buf_st0   range:NSMakeRange(0, st_sz) value:0];
  [bl fillBuffer:m->buf_st1   range:NSMakeRange(0, st_sz) value:0];
  [bl fillBuffer:m->buf_gbuf0 range:NSMakeRange(0, sz8) value:0];
  [bl fillBuffer:m->buf_gbuf1 range:NSMakeRange(0, sz8) value:0];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

static void bind_mesh(const MetalCtx* m, id<MTLComputeCommandEncoder> e) {
  [e setBuffer:m->buf_mesh_nodes offset:0 atIndex:4];
  [e setBuffer:m->buf_mesh_tri_a offset:0 atIndex:5];
  [e setBuffer:m->buf_mesh_tri_n offset:0 atIndex:6];
}

/* Mesh ground truth: rows [y0, y1) of the frame in one encoder */
static void encode_mesh_gt_band(const MetalCtx* m, const RunConfig* cfg, const FrameUniforms* u,
                                int y0, int y1, id<MTLComputeCommandEncoder> e1) {
  int band[4] = { y0, y1, 0, 0 };
  [e1 setComputePipelineState:m->pso_p1_gt];
  [e1 setBytes:u length:sizeof(*u) atIndex:0];
  [e1 setBuffer:m->buf_out     offset:0 atIndex:1];
  [e1 setBuffer:m->buf_gn_dump offset:0 atIndex:2];
  [e1 setBuffer:m->buf_gp_dump offset:0 atIndex:3];
  bind_mesh(m, e1);
  [e1 setBytes:band length:sizeof(band) atIndex:7];
  [e1 dispatchThreads:MTLSizeMake((NSUInteger)cfg->width, (NSUInteger)(y1 - y0), 1) threadsPerThreadgroup:m->tg];
}

static void encode_pass1_enc(const MetalCtx* m, const RunConfig* cfg, const FrameUniforms* u, int cur, id<MTLComputeCommandEncoder> e1) {
  if (m->is_mesh && cfg->mode == MODE_GT) {
    encode_mesh_gt_band(m, cfg, u, 0, cfg->height, e1);
    return;
  }
  if (is_direct_pass(cfg)) {
    if (cfg->mode == MODE_GT)          [e1 setComputePipelineState:m->pso_p1_gt];
    else if (cfg->scene_id == SCENE_B) [e1 setComputePipelineState:m->pso_p1_b];
    else                               [e1 setComputePipelineState:m->pso_p1_d];
    [e1 setBytes:u length:sizeof(*u) atIndex:0];
    [e1 setBuffer:m->buf_out     offset:0 atIndex:1];
    [e1 setBuffer:m->buf_gn_dump offset:0 atIndex:2];
    [e1 setBuffer:m->buf_gp_dump offset:0 atIndex:3];
  } else {
    id<MTLBuffer> gb_cur = cur ? m->buf_gbuf1 : m->buf_gbuf0;
    if (cfg->scene_id == SCENE_A) {
      [e1 setComputePipelineState:m->pso_p1_a];
      [e1 setBytes:u length:sizeof(*u) atIndex:0];
      [e1 setBuffer:m->buf_irrad  offset:0 atIndex:1];
      [e1 setBuffer:m->buf_direct offset:0 atIndex:2];
      [e1 setBuffer:m->buf_albedo offset:0 atIndex:3];
      [e1 setBuffer:gb_cur        offset:0 atIndex:4];
    } else {
      [e1 setComputePipelineState:m->pso_p1_c];
      [e1 setBytes:u length:sizeof(*u) atIndex:0];
      [e1 setBuffer:m->buf_irrad  offset:0 atIndex:1];
      [e1 setBuffer:m->buf_albedo offset:0 atIndex:2];
      [e1 setBuffer:gb_cur        offset:0 atIndex:3];
      if (m->is_mesh) bind_mesh(m, e1);
    }
  }
  [e1 dispatchThreads:m->grid threadsPerThreadgroup:m->tg];
}

static void encode_pass2_enc(const MetalCtx* m, const RunConfig* cfg, FrameUniforms u, int cur, int prv, id<MTLComputeCommandEncoder> e2) {
  if (is_direct_pass(cfg)) return;

  id<MTLBuffer> gb_cur = cur ? m->buf_gbuf1 : m->buf_gbuf0;
  id<MTLBuffer> st_cur = cur ? m->buf_st1   : m->buf_st0;
  id<MTLBuffer> gb_prv = prv ? m->buf_gbuf1 : m->buf_gbuf0;
  id<MTLBuffer> st_prv = prv ? m->buf_st1   : m->buf_st0;

  if (cfg->mode == MODE_OSABT) {
    if (cfg->scene_id == SCENE_A || cfg->frames <= 1) {
      /* 2-Pass Threadgroup-Tiled Separable H + V Strictly Co-Surface Psi_perp Filter */
      [e2 setComputePipelineState:m->pso_osabt_sph];
      [e2 setBytes:&u length:sizeof(u) atIndex:0];
      [e2 setBuffer:m->buf_irrad  offset:0 atIndex:1];
      [e2 setBuffer:gb_cur        offset:0 atIndex:2];
      [e2 setBuffer:m->buf_tmp_a  offset:0 atIndex:3];
      [e2 dispatchThreadgroups:m->tg_grid_h threadsPerThreadgroup:m->tg_h];

      [e2 setComputePipelineState:m->pso_osabt_spv];
      [e2 setBytes:&u length:sizeof(u) atIndex:0];
      [e2 setBuffer:m->buf_tmp_a  offset:0 atIndex:1];
      [e2 setBuffer:m->buf_direct offset:0 atIndex:2];
      [e2 setBuffer:m->buf_albedo offset:0 atIndex:3];
      [e2 setBuffer:gb_cur        offset:0 atIndex:4];
      [e2 setBuffer:m->buf_out    offset:0 atIndex:5];
      [e2 dispatchThreadgroups:m->tg_grid16 threadsPerThreadgroup:m->tg];
    } else {
      /* Scene C: 2A+2G1 fused (SRAM 18x18 + simd_sum) -> 2G2 (tile gauge) -> 2B (SRAM 22x22 ZDF-TPT) */
      [e2 setComputePipelineState:m->pso_osabt_2a];
      [e2 setBytes:&u length:sizeof(u) atIndex:0];
      [e2 setBuffer:m->buf_irrad    offset:0 atIndex:1];
      [e2 setBuffer:gb_cur          offset:0 atIndex:2];
      [e2 setBuffer:st_prv          offset:0 atIndex:3];
      [e2 setBuffer:gb_prv          offset:0 atIndex:4];
      [e2 setBuffer:m->buf_reproj   offset:0 atIndex:5];
      [e2 setBuffer:m->buf_tmp_b    offset:0 atIndex:6];
      [e2 setBuffer:m->buf_tile_su  offset:0 atIndex:7];
      [e2 setBuffer:m->buf_tile_ss  offset:0 atIndex:8];
      [e2 setBuffer:m->buf_tile_ss2 offset:0 atIndex:9];
      [e2 setBuffer:m->buf_tile_sus offset:0 atIndex:10];
      [e2 dispatchThreadgroups:m->tg_grid16 threadsPerThreadgroup:m->tg];

      [e2 setComputePipelineState:m->pso_osabt_g2];
      [e2 setBytes:&u length:sizeof(u) atIndex:0];
      [e2 setBuffer:m->buf_tile_su  offset:0 atIndex:1];
      [e2 setBuffer:m->buf_tile_ss  offset:0 atIndex:2];
      [e2 setBuffer:m->buf_tile_ss2 offset:0 atIndex:3];
      [e2 setBuffer:m->buf_tile_sus offset:0 atIndex:4];
      [e2 setBuffer:m->buf_tile_kap offset:0 atIndex:5];
      [e2 setBuffer:m->buf_tile_bet offset:0 atIndex:6];
      [e2 dispatchThreads:m->tile_grid threadsPerThreadgroup:m->tile_tg];

      [e2 setComputePipelineState:m->pso_osabt_2b];
      [e2 setBytes:&u length:sizeof(u) atIndex:0];
      [e2 setBuffer:m->buf_albedo   offset:0 atIndex:1];
      [e2 setBuffer:gb_cur          offset:0 atIndex:2];
      [e2 setBuffer:m->buf_reproj   offset:0 atIndex:3];
      [e2 setBuffer:m->buf_tmp_b    offset:0 atIndex:4];
      [e2 setBuffer:m->buf_tile_kap offset:0 atIndex:5];
      [e2 setBuffer:m->buf_tile_bet offset:0 atIndex:6];
      [e2 setBuffer:st_cur          offset:0 atIndex:7];
      [e2 setBuffer:m->buf_out      offset:0 atIndex:8];
      [e2 dispatchThreadgroups:m->tg_grid16 threadsPerThreadgroup:m->tg];
    }
  } else if (cfg->mode == MODE_ABLATED) {
    [e2 setComputePipelineState:m->pso_abl];
    [e2 setBytes:&u length:sizeof(u) atIndex:0];
    [e2 setBuffer:m->buf_irrad  offset:0 atIndex:1];
    [e2 setBuffer:m->buf_albedo offset:0 atIndex:2];
    [e2 setBuffer:gb_cur        offset:0 atIndex:3];
    [e2 setBuffer:st_prv        offset:0 atIndex:4];
    [e2 setBuffer:st_cur        offset:0 atIndex:5];
    [e2 setBuffer:m->buf_out    offset:0 atIndex:6];
    [e2 dispatchThreads:m->grid threadsPerThreadgroup:m->tg];
  } else {
    [e2 setComputePipelineState:m->pso_taa];
    [e2 setBytes:&u length:sizeof(u) atIndex:0];
    [e2 setBuffer:m->buf_irrad  offset:0 atIndex:1];
    [e2 setBuffer:m->buf_albedo offset:0 atIndex:2];
    [e2 setBuffer:gb_cur        offset:0 atIndex:3];
    [e2 setBuffer:st_prv        offset:0 atIndex:4];
    [e2 setBuffer:m->buf_tmp_a  offset:0 atIndex:5];
    [e2 dispatchThreads:m->grid threadsPerThreadgroup:m->tg];

    u.pass_step = 1;
    [e2 setComputePipelineState:m->pso_atr];
    [e2 setBytes:&u length:sizeof(u) atIndex:0];
    [e2 setBuffer:m->buf_tmp_a  offset:0 atIndex:1];
    [e2 setBuffer:gb_cur        offset:0 atIndex:2];
    [e2 setBuffer:m->buf_tmp_b  offset:0 atIndex:3];
    [e2 setBuffer:m->buf_out    offset:0 atIndex:4];
    [e2 dispatchThreads:m->grid threadsPerThreadgroup:m->tg];

    u.pass_step = 2;
    [e2 setBytes:&u length:sizeof(u) atIndex:0];
    [e2 setBuffer:m->buf_tmp_b  offset:0 atIndex:1];
    [e2 setBuffer:gb_cur        offset:0 atIndex:2];
    [e2 setBuffer:st_cur        offset:0 atIndex:3];
    [e2 setBuffer:m->buf_out    offset:0 atIndex:4];
    [e2 dispatchThreads:m->grid threadsPerThreadgroup:m->tg];
  }
}

static double run_one_frame(const MetalCtx* m, const RunConfig* cfg, int f) {
  int cur = f & 1;
  int prv = 1 - cur;
  FrameUniforms u;
  os_fill_uniforms(&u, cfg, f);
  if (m->is_mesh && cfg->mode == MODE_GT) {
    /* 64-spp BVH ground truth: one command buffer per 16-row band keeps every submission short */
    double ms_total = 0.0;
    for (int y0 = 0; y0 < cfg->height; y0 += 16) {
      int y1 = (y0 + 16 < cfg->height) ? (y0 + 16) : cfg->height;
      id<MTLCommandBuffer> cbb = [m->queue commandBuffer];
      id<MTLComputeCommandEncoder> eb = [cbb computeCommandEncoder];
      encode_mesh_gt_band(m, cfg, &u, y0, y1, eb);
      [eb endEncoding];
      [cbb commit];
      [cbb waitUntilCompleted];
      if (cbb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "GPU error in mesh ground-truth band %d: %s\n", y0,
                [[cbb.error localizedDescription] UTF8String]);
        exit(3);
      }
      double ms = (cbb.GPUEndTime - cbb.GPUStartTime) * 1000.0;
      if (ms > 0.0) ms_total += ms;
    }
    return ms_total;
  }
  id<MTLCommandBuffer> cb = [m->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  encode_pass1_enc(m, cfg, &u, cur, enc);
  encode_pass2_enc(m, cfg, u, cur, prv, enc);
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  return (ms > 0.0) ? ms : 0.0;
}

static void measure_pass_split(const MetalCtx* m, const RunConfig* cfg, double* out_p1, double* out_p2) {
  FrameUniforms u;
  int f = (cfg->frames > 1) ? 1 : 0;
  os_fill_uniforms(&u, cfg, f);
  id<MTLCommandBuffer> cb1 = [m->queue commandBuffer];
  id<MTLComputeCommandEncoder> e1 = [cb1 computeCommandEncoder];
  encode_pass1_enc(m, cfg, &u, f & 1, e1);
  [e1 endEncoding];
  [cb1 commit];
  [cb1 waitUntilCompleted];
  *out_p1 = fmax(0.0, (cb1.GPUEndTime - cb1.GPUStartTime) * 1000.0);
  if (is_direct_pass(cfg)) {
    *out_p2 = 0.0;
    return;
  }
  id<MTLCommandBuffer> cb2 = [m->queue commandBuffer];
  id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
  encode_pass2_enc(m, cfg, u, f & 1, 1 - (f & 1), e2);
  [e2 endEncoding];
  [cb2 commit];
  [cb2 waitUntilCompleted];
  *out_p2 = fmax(0.0, (cb2.GPUEndTime - cb2.GPUStartTime) * 1000.0);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    RunConfig cfg;
    if (os_parse_args(argc, argv, &cfg) != 0) {
      fprintf(stderr, "Usage: osabt_metal --scene a|b|c|d|e|f|g|h --mode osabt|ablated|baseline|gt ...\n");
      return 1;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "No Metal GPU device available.\n"); return 1; }

    NSString* src = load_combined_shader();
    if (!src) return 1;
    /* Keep the runtime-compiled kernels on the same storage formats as this binary. */
    NSDictionary* fmt_macros = @{ @"OSABT_ST_FMT": @(OSABT_ST_FMT), @"OSABT_REP_FMT": @(OSABT_REP_FMT) };
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.preprocessorMacros = fmt_macros;
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
      opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
    } else
#endif
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
      fprintf(stderr, "Metal compile error:\n%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLLibrary> lib_p2 = lib;
    if (!is_direct_pass(&cfg) || (cfg.scene_id == SCENE_B && cfg.mode == MODE_OSABT)) {
      MTLCompileOptions* opts2 = [[MTLCompileOptions alloc] init];
      opts2.preprocessorMacros = @{ @"OSABT_PASS2_RELAXED": @1,
                                    @"OSABT_ST_FMT": @(OSABT_ST_FMT), @"OSABT_REP_FMT": @(OSABT_REP_FMT) };
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_15_0
      if (@available(macOS 15.0, *)) {
        opts2.mathMode = MTLMathModeRelaxed;
        opts2.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
      } else
#endif
      {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts2.fastMathEnabled = YES;
#pragma clang diagnostic pop
      }
      lib_p2 = [dev newLibraryWithSource:src options:opts2 error:&err];
      if (!lib_p2) {
        fprintf(stderr, "Metal Pass2 compile error:\n%s\n", [[err localizedDescription] UTF8String]);
        return 1;
      }
    }

    size_t npix  = (size_t)cfg.width * (size_t)cfg.height;
    size_t sz8   = npix * sizeof(h4);
    size_t sz16  = npix * sizeof(v4);
    int TW = (cfg.width + 15) / 16;
    int TH = (cfg.height + 15) / 16;
    size_t sz_tile = (size_t)TW * (size_t)TH * sizeof(v4);

    MTLResourceOptions ropt_priv = MTLResourceStorageModePrivate;
    MTLResourceOptions ropt_sh   = MTLResourceStorageModeShared;

    MetalCtx m;
    memset(&m, 0, sizeof(m));
    m.queue = [dev newCommandQueue];
    if (is_direct_pass(&cfg)) {
      if (cfg.mode == MODE_GT)          m.pso_p1_gt = make_pso(dev, lib, (cfg.scene_id == SCENE_H) ? @"k_pass1_gt_mesh" : @"k_pass1_gt");
      else if (cfg.scene_id == SCENE_B) m.pso_p1_b  = make_pso(dev, (cfg.mode == MODE_OSABT) ? lib_p2 : lib,
                                                               (cfg.mode == MODE_OSABT) ? @"k_pass1_scene_b_osabt" : @"k_pass1_scene_b");
      else                              m.pso_p1_d  = make_pso(dev, lib, @"k_pass1_scene_d");
    } else {
      if (cfg.scene_id == SCENE_A)      m.pso_p1_a  = make_pso(dev, lib, (cfg.mode == MODE_OSABT) ? @"k_pass1_scene_a_osabt" : @"k_pass1_scene_a");
      else                              m.pso_p1_c  = make_pso(dev, lib, (cfg.scene_id == SCENE_E) ? @"k_pass1_scene_e" :
                                                                    (cfg.scene_id == SCENE_F) ? @"k_pass1_scene_f" :
                                                                    (cfg.scene_id == SCENE_G) ? @"k_pass1_scene_g" :
                                                                    (cfg.scene_id == SCENE_H) ? @"k_pass1_scene_h" : @"k_pass1_scene_c");
      if (cfg.mode == MODE_OSABT) {
        if (cfg.scene_id == SCENE_A || cfg.frames <= 1) {
          m.pso_osabt_sph = make_pso(dev, lib_p2, @"k_pass2_osabt_spatial_h");
          m.pso_osabt_spv = make_pso(dev, lib_p2, @"k_pass2_osabt_spatial_v");
        } else {
          m.pso_osabt_2a  = make_pso(dev, lib_p2, @"k_pass2a_osabt_reproj");
          m.pso_osabt_g2  = make_pso(dev, lib_p2, @"k_pass2g2_tile_gauge");
          m.pso_osabt_2b  = make_pso(dev, lib_p2, @"k_pass2b_osabt_zdf");
        }
      } else if (cfg.mode == MODE_ABLATED) {
        m.pso_abl = make_pso(dev, lib_p2, @"k_pass2_ablated");
      } else {
        m.pso_taa = make_pso(dev, lib_p2, @"k_pass2a_baseline_taa");
        m.pso_atr = make_pso(dev, lib_p2, @"k_pass2b_baseline_atrous");
      }
    }

    m.buf_irrad     = [dev newBufferWithLength:sz8     options:ropt_priv];
    m.buf_direct    = [dev newBufferWithLength:sz8     options:ropt_priv];
    m.buf_albedo    = [dev newBufferWithLength:sz8     options:ropt_priv];
    m.buf_gbuf0     = [dev newBufferWithLength:sz8     options:ropt_priv];
    m.buf_gbuf1     = [dev newBufferWithLength:sz8     options:ropt_priv];
    int osabt_temporal = (cfg.mode == MODE_OSABT && cfg.scene_id != SCENE_A && cfg.frames > 1 && !is_direct_pass(&cfg)) ? 1 : 0;
    m.st_bytes      = osabt_temporal ? npix * sizeof(st4) : sz8;
    m.buf_st0       = [dev newBufferWithLength:m.st_bytes options:ropt_priv];
    m.buf_st1       = [dev newBufferWithLength:m.st_bytes options:ropt_priv];
    m.buf_tmp_a     = [dev newBufferWithLength:sz8     options:ropt_priv];
    m.buf_tmp_b     = [dev newBufferWithLength:sz8     options:ropt_priv];
    if (osabt_temporal)
      m.buf_reproj  = [dev newBufferWithLength:npix * sizeof(rp4) options:ropt_priv];

    m.buf_tile_su   = [dev newBufferWithLength:sz_tile options:ropt_priv];
    m.buf_tile_ss   = [dev newBufferWithLength:sz_tile options:ropt_priv];
    m.buf_tile_ss2  = [dev newBufferWithLength:sz_tile options:ropt_priv];
    m.buf_tile_sus  = [dev newBufferWithLength:sz_tile options:ropt_priv];
    m.buf_tile_kap  = [dev newBufferWithLength:sz_tile options:ropt_priv];
    m.buf_tile_bet  = [dev newBufferWithLength:sz_tile options:ropt_priv];

    m.buf_gn_dump   = [dev newBufferWithLength:(cfg.dump_gbuf ? sz16 : 64) options:ropt_sh];
    m.buf_gp_dump   = [dev newBufferWithLength:(cfg.dump_gbuf ? sz16 : 64) options:ropt_sh];
    m.buf_out       = [dev newBufferWithLength:sz16    options:ropt_sh];

    if (cfg.scene_id == SCENE_H) {
      OsMesh mesh;
      if (os_mesh_load(cfg.mesh_path, &mesh) != 0) return 1;
      m.is_mesh = 1;
      m.buf_mesh_nodes = [dev newBufferWithBytes:mesh.nodes length:(size_t)mesh.n_nodes * 8 * sizeof(float) options:ropt_sh];
      m.buf_mesh_tri_a = [dev newBufferWithBytes:mesh.tri_a length:(size_t)mesh.n_tris * 12 * sizeof(float) options:ropt_sh];
      m.buf_mesh_tri_n = [dev newBufferWithBytes:mesh.tri_n length:(size_t)mesh.n_tris * 12 * sizeof(float) options:ropt_sh];
      fprintf(stderr, "mesh: %u triangles, %u BVH nodes, depth %d\n", mesh.n_tris, mesh.n_nodes, mesh.max_depth);
      os_mesh_free(&mesh);
    }
    m.grid          = MTLSizeMake((NSUInteger)cfg.width, (NSUInteger)cfg.height, 1);
    m.tg            = MTLSizeMake(16, 16, 1);
    m.tg_grid16     = MTLSizeMake((NSUInteger)TW, (NSUInteger)TH, 1);
    m.tg_grid_h     = MTLSizeMake((NSUInteger)((cfg.width + 31) / 32), (NSUInteger)((cfg.height + 7) / 8), 1);
    m.tg_h          = MTLSizeMake(32, 8, 1);
    m.tile_grid     = MTLSizeMake((NSUInteger)TW, (NSUInteger)TH, 1);
    m.tile_tg       = MTLSizeMake(8, 8, 1);

    v4* epipolar = cfg.dump_epipolar ? (v4*)calloc((size_t)cfg.width * (size_t)cfg.frames, sizeof(v4)) : NULL;
    v4* aux_host = cfg.dump_gbuf ? (v4*)calloc(npix, sizeof(v4)) : NULL;

    zero_private_history(&m, sz8);
    int n_warm = (cfg.mode == MODE_GT) ? 0 : cfg.warmup;
    for (int w = 0; w < n_warm; ++w) {
      run_one_frame(&m, &cfg, 0);
    }
    zero_private_history(&m, sz8);

    double frame_ms[1024];
    int n_ms = 0;

    int timing_reps = (cfg.frames == 1 && cfg.mode != MODE_GT && cfg.warmup > 0) ? 15 : 1;
    for (int r = 0; r < timing_reps; ++r) {
      for (int f = 0; f < cfg.frames; ++f) {
        if (cfg.mode == MODE_GT && cfg.has_key_frames && !os_should_save_frame(&cfg, f)) {
          continue;
        }
        double tot = run_one_frame(&m, &cfg, f);
        if (n_ms < 1024) frame_ms[n_ms++] = tot;

        if (r == timing_reps - 1) {
          v4* out_ptr = (v4*)[m.buf_out contents];
          if (epipolar) {
            int cy = cfg.height / 2;
            memcpy(&epipolar[(size_t)f * (size_t)cfg.width],
                   &out_ptr[(size_t)cy * (size_t)cfg.width],
                   (size_t)cfg.width * sizeof(v4));
          }
          if (os_should_save_frame(&cfg, f)) {
            char path[600];
            snprintf(path, sizeof(path), "%s/frame_%04d.f32img", cfg.out_dir, f);
            os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)out_ptr);
            if (cfg.dump_gbuf) {
              v4* gn_ptr = (v4*)[m.buf_gn_dump contents];
              v4* gp_ptr = (v4*)[m.buf_gp_dump contents];
              snprintf(path, sizeof(path), "%s/gbuf_norm_%04d.f32img", cfg.out_dir, f);
              os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)gn_ptr);
              for (size_t i = 0; i < npix; ++i) {
                aux_host[i] = mk4(gp_ptr[i].w, gp_ptr[i].x, gp_ptr[i].y, gp_ptr[i].z);
              }
              snprintf(path, sizeof(path), "%s/gbuf_aux_%04d.f32img", cfg.out_dir, f);
              os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)aux_host);
            }
          }
        }
      }
    }

    double p1_ms = 0.0, p2_ms = 0.0;
    if (cfg.mode != MODE_GT) {
      measure_pass_split(&m, &cfg, &p1_ms, &p2_ms);
    }

    if (epipolar) {
      char path[600];
      snprintf(path, sizeof(path), "%s/epipolar.f32img", cfg.out_dir);
      os_write_f32img(path, cfg.width, cfg.frames, 4, (const float*)epipolar);
    }
    os_write_telemetry(&cfg, frame_ms, n_ms, p1_ms, p2_ms);

    free(epipolar);
    free(aux_host);
  }
  return 0;
}
