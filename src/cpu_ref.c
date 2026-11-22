/*
 * Slice 2/3: Pure Mathematical CPU Reference Renderer (ISO C11, Deterministic)
 * Zero GPU/UI coupling. Compiled with -std=c11 -O3 -ffp-contract=off.
 * Bit-identical IEEE-754 packed 8B half4 G-buffer & pass schedule to src/metal_runner.m.
 */
#include "scene_setup.h"
#include "mesh_bvh.h"
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static int is_direct_pass(const RunConfig* cfg) {
  return (cfg->mode == MODE_GT || cfg->scene_id == SCENE_B || cfg->scene_id == SCENE_D) ? 1 : 0;
}

int main(int argc, char** argv) {
  RunConfig cfg;
  if (os_parse_args(argc, argv, &cfg) != 0) {
    fprintf(stderr, "Usage: osabt_cpu --scene a|b|c|d|e|f|g|h --mode osabt|ablated|baseline|gt ...\n");
    return 1;
  }
  size_t npix = (size_t)cfg.width * (size_t)cfg.height;
  int TW = (cfg.width + 15) / 16;
  int TH = (cfg.height + 15) / 16;
  size_t ntile = (size_t)TW * (size_t)TH;

  h4* cur_irrad    = (h4*)calloc(npix, sizeof(h4));
  h4* cur_direct   = (h4*)calloc(npix, sizeof(h4));
  u16_4* cur_albedo= (u16_4*)calloc(npix, sizeof(u16_4));
  h4* gbuf_buf[2]  = { (h4*)calloc(npix, sizeof(h4)), (h4*)calloc(npix, sizeof(h4)) };
  h4* st_buf[2]    = { (h4*)calloc(npix, sizeof(h4)), (h4*)calloc(npix, sizeof(h4)) };
  h4* tmp_a        = (h4*)calloc(npix, sizeof(h4));
  h4* tmp_b        = (h4*)calloc(npix, sizeof(h4));
  /* OS-ABT temporal path (Scene C): history state + reprojected history in their own formats */
  st4* sto_buf[2]  = { (st4*)calloc(npix, sizeof(st4)), (st4*)calloc(npix, sizeof(st4)) };
  rp4* tmp_rep     = (rp4*)calloc(npix, sizeof(rp4));

  v4* tile_su    = (v4*)calloc(ntile, sizeof(v4));
  v4* tile_ss    = (v4*)calloc(ntile, sizeof(v4));
  v4* tile_ss2   = (v4*)calloc(ntile, sizeof(v4));
  v4* tile_sus   = (v4*)calloc(ntile, sizeof(v4));
  v4* tile_kap   = (v4*)calloc(ntile, sizeof(v4));
  v4* tile_bet   = (v4*)calloc(ntile, sizeof(v4));

  v4* gn_dump    = cfg.dump_gbuf ? (v4*)calloc(npix, sizeof(v4)) : NULL;
  v4* gp_dump    = cfg.dump_gbuf ? (v4*)calloc(npix, sizeof(v4)) : NULL;
  v4* aux_dump   = cfg.dump_gbuf ? (v4*)calloc(npix, sizeof(v4)) : NULL;
  v4* out_rgb    = (v4*)calloc(npix, sizeof(v4));
  v4* epipolar   = cfg.dump_epipolar ? (v4*)calloc((size_t)cfg.width * (size_t)cfg.frames, sizeof(v4)) : NULL;

  double frame_ms[1024];
  int n_ms = 0;
  double p1_acc = 0.0, p2_acc = 0.0;
  int direct = is_direct_pass(&cfg);

  /* Mesh scene (Scene H): load the triangles and build the BVH once */
  OsMesh mesh;
  memset(&mesh, 0, sizeof(mesh));
  int use_mesh = (cfg.scene_id == SCENE_H) ? 1 : 0;
  if (use_mesh && os_mesh_load(cfg.mesh_path, &mesh) != 0) return 1;
  const v4* mesh_nodes = (const v4*)mesh.nodes;
  const v4* mesh_tri_a = (const v4*)mesh.tri_a;
  const v4* mesh_tri_n = (const v4*)mesh.tri_n;

  for (int f = 0; f < cfg.frames; ++f) {
    if (cfg.mode == MODE_GT && cfg.has_key_frames && !os_should_save_frame(&cfg, f)) {
      continue;
    }
    int cur = f & 1;
    int prv = 1 - cur;
    FrameUniforms u;
    os_fill_uniforms(&u, &cfg, f);

    double t0 = now_ms();
    if (direct) {
      for (int py = 0; py < cfg.height; ++py) {
        for (int px = 0; px < cfg.width; ++px) {
          int idx = py * cfg.width + px;
          Pass1Out p1 = use_mesh ? os_pass1_pixel_mesh(&u, px, py, mesh_nodes, mesh_tri_a, mesh_tri_n)
                                 : os_pass1_pixel(&u, px, py);
          out_rgb[idx] = mk4(p1.irrad.x, p1.irrad.y, p1.irrad.z, 1.0f);
          if (cfg.dump_gbuf) {
            gn_dump[idx] = p1.gbuf_n;
            gp_dump[idx] = p1.gbuf_p;
          }
        }
      }
    } else {
      for (int py = 0; py < cfg.height; ++py) {
        for (int px = 0; px < cfg.width; ++px) {
          int idx = py * cfg.width + px;
          Pass1Out p1 = use_mesh ? os_pass1_pixel_mesh(&u, px, py, mesh_nodes, mesh_tri_a, mesh_tri_n)
                                 : os_pass1_pixel(&u, px, py);
          cur_irrad[idx]      = v4_to_h4(p1.irrad);
          cur_albedo[idx]     = v4_to_u16(p1.albedo);
          gbuf_buf[cur][idx]  = os_pack_gbuf_h4(&u, p1.gbuf_p, p1.gbuf_n);
          if (cfg.scene_id == SCENE_A && cfg.mode == MODE_OSABT) {
            cur_direct[idx] = v4_to_h4(p1.direct);
          }
        }
      }
    }
    double t1 = now_ms();

    if (!direct) {
      if (cfg.mode == MODE_OSABT) {
        if (cfg.scene_id == SCENE_A || cfg.frames <= 1) {
          for (int py = 0; py < cfg.height; ++py)
            for (int px = 0; px < cfg.width; ++px)
              os_pass2_osabt_spatial(&u, px, py, 1, 0, cur_irrad, cur_direct, cur_albedo,
                                     gbuf_buf[cur], tmp_a, out_rgb);
          for (int py = 0; py < cfg.height; ++py)
            for (int px = 0; px < cfg.width; ++px)
              os_pass2_osabt_spatial(&u, px, py, 0, 1, tmp_a, cur_direct, cur_albedo,
                                     gbuf_buf[cur], tmp_b, out_rgb);
        } else {
          for (int py = 0; py < cfg.height; ++py)
            for (int px = 0; px < cfg.width; ++px)
              os_pass2a_osabt_reproj(&u, px, py, cur_irrad, gbuf_buf[cur],
                                     sto_buf[prv], gbuf_buf[prv], tmp_rep, tmp_b);
          for (int ty = 0; ty < TH; ++ty)
            for (int tx = 0; tx < TW; ++tx)
              os_pass2g1_tile_reduce(&u, tx, ty, tmp_rep, tmp_b, gbuf_buf[cur],
                                     tile_su, tile_ss, tile_ss2, tile_sus);
          for (int ty = 0; ty < TH; ++ty)
            for (int tx = 0; tx < TW; ++tx)
              os_pass2g2_tile_gauge(&u, tx, ty, tile_su, tile_ss, tile_ss2, tile_sus,
                                    tile_kap, tile_bet);
          for (int py = 0; py < cfg.height; ++py)
            for (int px = 0; px < cfg.width; ++px)
              os_pass2b_osabt_zdf(&u, px, py, cur_albedo, gbuf_buf[cur],
                                  tmp_rep, tmp_b, tile_kap, tile_bet, sto_buf[cur], out_rgb);
        }
      } else if (cfg.mode == MODE_ABLATED) {
        for (int py = 0; py < cfg.height; ++py)
          for (int px = 0; px < cfg.width; ++px)
            os_pass2_ablated_pixel(&u, px, py, cur_irrad, cur_albedo, gbuf_buf[cur],
                                   st_buf[prv], st_buf[cur], out_rgb);
      } else {
        for (int py = 0; py < cfg.height; ++py)
          for (int px = 0; px < cfg.width; ++px)
            os_pass2a_baseline_taa(&u, px, py, cur_irrad, cur_albedo, gbuf_buf[cur],
                                   st_buf[prv], tmp_a);
        for (int py = 0; py < cfg.height; ++py)
          for (int px = 0; px < cfg.width; ++px)
            os_pass2b_baseline_atrous(&u, px, py, 1, 0, tmp_a, gbuf_buf[cur],
                                      tmp_b, out_rgb);
        for (int py = 0; py < cfg.height; ++py)
          for (int px = 0; px < cfg.width; ++px)
            os_pass2b_baseline_atrous(&u, px, py, 2, 1, tmp_b, gbuf_buf[cur],
                                      st_buf[cur], out_rgb);
      }
    }
    double t2 = now_ms();
    if (n_ms < 1024) frame_ms[n_ms++] = (t2 - t0);
    p1_acc += (t1 - t0); p2_acc += (t2 - t1);

    if (epipolar) {
      int cy = cfg.height / 2;
      memcpy(&epipolar[(size_t)f * (size_t)cfg.width], &out_rgb[(size_t)cy * (size_t)cfg.width],
             (size_t)cfg.width * sizeof(v4));
    }

    if (os_should_save_frame(&cfg, f)) {
      char path[600];
      snprintf(path, sizeof(path), "%s/frame_%04d.f32img", cfg.out_dir, f);
      os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)out_rgb);
      if (cfg.dump_gbuf && gn_dump && gp_dump) {
        snprintf(path, sizeof(path), "%s/gbuf_norm_%04d.f32img", cfg.out_dir, f);
        os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)gn_dump);
        for (size_t i = 0; i < npix; ++i) {
          aux_dump[i] = mk4(gp_dump[i].w, gp_dump[i].x, gp_dump[i].y, gp_dump[i].z);
        }
        snprintf(path, sizeof(path), "%s/gbuf_aux_%04d.f32img", cfg.out_dir, f);
        os_write_f32img(path, cfg.width, cfg.height, 4, (const float*)aux_dump);
      }
    }
  }

  if (epipolar) {
    char path[600];
    snprintf(path, sizeof(path), "%s/epipolar.f32img", cfg.out_dir);
    os_write_f32img(path, cfg.width, cfg.frames, 4, (const float*)epipolar);
  }
  os_write_telemetry(&cfg, frame_ms, n_ms,
                     n_ms ? p1_acc / n_ms : 0.0, n_ms ? p2_acc / n_ms : 0.0);

  free(cur_irrad); free(cur_direct); free(cur_albedo);
  free(gbuf_buf[0]); free(gbuf_buf[1]);
  free(st_buf[0]); free(st_buf[1]);
  if (use_mesh) os_mesh_free(&mesh);
  free(tmp_a); free(tmp_b); free(tmp_rep);
  free(sto_buf[0]); free(sto_buf[1]);
  free(tile_su); free(tile_ss); free(tile_ss2); free(tile_sus);
  free(tile_kap); free(tile_bet);
  free(gn_dump); free(gp_dump); free(aux_dump);
  free(out_rgb); free(epipolar);
  return 0;
}
