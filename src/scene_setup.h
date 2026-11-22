#ifndef OSABT_SCENE_SETUP_H
#define OSABT_SCENE_SETUP_H

#include "osabt_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>

typedef struct {
  int scene_id;
  int mode;
  int width;
  int height;
  int frames;
  int warmup;
  int dump_gbuf;
  int dump_epipolar;
  char out_dir[512];
  int has_key_frames;
  int key_mask[1024];
} RunConfig;

static void os_build_cam(v3 eye, v3 target, float fov_deg,
                         float out_eye[4], float out_fwd[4],
                         float out_rgt[4], float out_up[4]) {
  v3 fwd = v3_norm(v3_sub(target, eye));
  v3 rgt = v3_norm(v3_cross(fwd, mk3(0.0f, 1.0f, 0.0f)));
  v3 up  = v3_cross(rgt, fwd);
  float fov_rad = fov_deg * (OS_PI / 180.0f);
  out_eye[0] = eye.x; out_eye[1] = eye.y; out_eye[2] = eye.z;
  out_eye[3] = fov_rad;
  out_fwd[0] = fwd.x; out_fwd[1] = fwd.y; out_fwd[2] = fwd.z; out_fwd[3] = tanf(fov_rad * 0.5f);
  out_rgt[0] = rgt.x; out_rgt[1] = rgt.y; out_rgt[2] = rgt.z; out_rgt[3] = 0.0f;
  out_up[0]  = up.x;  out_up[1]  = up.y;  out_up[2]  = up.z;  out_up[3]  = 0.0f;
}

static void os_scene_camera_at(int scene_id, int frame_idx,
                               float out_eye[4], float out_fwd[4],
                               float out_rgt[4], float out_up[4],
                               float out_lcol[4]) {
  out_lcol[0] = 1.0f; out_lcol[1] = 0.92f; out_lcol[2] = 0.80f; out_lcol[3] = 1.0f;
  if (scene_id == SCENE_A) {
    os_build_cam(mk3(0.95f, 0.42f, 1.85f), mk3(-0.10f, 0.52f, -2.80f), 48.0f,
                 out_eye, out_fwd, out_rgt, out_up);
  } else if (scene_id == SCENE_B) {
    /* 16-frame irrational sub-pixel camera drift from scene_b.json */
    float dx = 0.00371f * (float)frame_idx;
    float dz = -0.01137f * (float)frame_idx;
    os_build_cam(mk3(0.0f + dx, 0.24f, 0.0f + dz),
                 mk3(0.0f + dx, 0.14f, -4.50f + dz), 45.0f,
                 out_eye, out_fwd, out_rgt, out_up);
  } else if (scene_id == SCENE_C) {
    /* 120-frame schedule from scene_c.json:
       0..39 steady slow pan, 40 Dirac +6.5x shock, 60..90 whip-pan 34 deg,
       75 chromatic step shock (0.55, 0.88, 1.45)*2.1, 91..119 settle */
    float yaw_deg = 0.0f;
    float x_shift = 0.0f;
    if (frame_idx < 60) {
      yaw_deg = 0.08f * (float)frame_idx;
      x_shift = 0.0025f * (float)frame_idx;
    } else if (frame_idx <= 90) {
      float t = (float)(frame_idx - 60) / 30.0f;
      /* Smooth C1 S-curve whip-pan sweeping 34 deg & lateral parallax across foreground pillar */
      float s = t * t * (3.0f - 2.0f * t);
      yaw_deg = 4.8f + 34.0f * s;
      x_shift = 0.15f + 0.95f * s;
    } else {
      float t = (float)(frame_idx - 90) / 29.0f;
      yaw_deg = 38.8f + 1.2f * t;
      x_shift = 1.10f + 0.04f * t;
    }
    float yaw = yaw_deg * (OS_PI / 180.0f);
    v3 eye = mk3(0.15f - x_shift, 0.85f, 2.40f);
    /* Look direction rotated around Y by yaw */
    v3 base_d = v3_norm(mk3(-0.15f, -0.13f, -5.40f));
    float cy = cosf(yaw), sy = sinf(yaw);
    v3 dir = mk3(base_d.x * cy + base_d.z * sy, base_d.y, -base_d.x * sy + base_d.z * cy);
    os_build_cam(eye, v3_add(eye, dir), 50.0f, out_eye, out_fwd, out_rgt, out_up);

    if (frame_idx == 40) {
      out_lcol[0] = 1.0f; out_lcol[1] = 0.92f; out_lcol[2] = 0.80f; out_lcol[3] = 6.5f;
    } else if (frame_idx >= 75) {
      out_lcol[0] = 0.55f; out_lcol[1] = 0.88f; out_lcol[2] = 1.45f; out_lcol[3] = 2.1f;
    }
  } else {
    os_build_cam(mk3(0.0f, 0.85f, 2.60f), mk3(0.0f, 0.55f, -2.00f), 48.0f,
                 out_eye, out_fwd, out_rgt, out_up);
  }
}

static void os_fill_uniforms(FrameUniforms* u, const RunConfig* cfg, int frame_idx) {
  memset(u, 0, sizeof(*u));
  u->scene_id = cfg->scene_id;
  u->mode = cfg->mode;
  u->width = cfg->width;
  u->height = cfg->height;
  u->frame_idx = frame_idx;
  u->total_frames = cfg->frames;
  u->pad0 = cfg->dump_gbuf;
  float aspect = (float)cfg->width / (float)cfg->height;
  os_scene_camera_at(cfg->scene_id, frame_idx,
                     u->cam_eye, u->cam_fwd, u->cam_right, u->cam_up, u->light_color);
  u->cam_right[3] = aspect;
  int prev_f = (frame_idx > 0) ? (frame_idx - 1) : 0;
  float dummy_l[4];
  os_scene_camera_at(cfg->scene_id, prev_f,
                     u->prev_eye, u->prev_fwd, u->prev_right, u->prev_up, dummy_l);
  u->prev_right[3] = aspect;
}

static int os_parse_args(int argc, char** argv, RunConfig* cfg) {
  memset(cfg, 0, sizeof(*cfg));
  cfg->width = 1280; cfg->height = 720; cfg->frames = 1; cfg->warmup = 4;
  strcpy(cfg->out_dir, "artifacts/out");
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
      const char* s = argv[++i];
      if (s[0] == 'a' || s[0] == 'A') cfg->scene_id = SCENE_A;
      else if (s[0] == 'b' || s[0] == 'B') cfg->scene_id = SCENE_B;
      else if (s[0] == 'c' || s[0] == 'C') cfg->scene_id = SCENE_C;
      else if (s[0] == 'd' || s[0] == 'D') cfg->scene_id = SCENE_D;
      else return -1;
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      const char* m = argv[++i];
      if (strcmp(m, "osabt") == 0) cfg->mode = MODE_OSABT;
      else if (strcmp(m, "ablated") == 0) cfg->mode = MODE_ABLATED;
      else if (strcmp(m, "baseline") == 0) cfg->mode = MODE_BASELINE;
      else if (strcmp(m, "gt") == 0) cfg->mode = MODE_GT;
      else return -1;
    } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
      cfg->width = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
      cfg->height = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      cfg->frames = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg->warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
      snprintf(cfg->out_dir, sizeof(cfg->out_dir), "%s", argv[++i]);
    } else if (strcmp(argv[i], "--dump-gbuf") == 0) {
      cfg->dump_gbuf = 1;
    } else if (strcmp(argv[i], "--dump-epipolar") == 0) {
      cfg->dump_epipolar = 1;
    } else if (strcmp(argv[i], "--key-frames") == 0 && i + 1 < argc) {
      cfg->has_key_frames = 1;
      char buf[512];
      snprintf(buf, sizeof(buf), "%s", argv[++i]);
      char* tok = strtok(buf, ",");
      while (tok) {
        int k = atoi(tok);
        if (k >= 0 && k < 1024) cfg->key_mask[k] = 1;
        tok = strtok(NULL, ",");
      }
    }
  }
  return 0;
}

static int os_should_save_frame(const RunConfig* cfg, int f) {
  if (cfg->has_key_frames) return (f >= 0 && f < 1024) ? cfg->key_mask[f] : 0;
  return 1;
}

static int os_write_f32img(const char* path, int w, int h, int c, const float* data) {
  FILE* fp = fopen(path, "wb");
  if (!fp) { fprintf(stderr, "Failed to open %s\n", path); return -1; }
  char magic[4] = {'O','S','A','B'};
  uint32_t hdr[3] = { (uint32_t)w, (uint32_t)h, (uint32_t)c };
  fwrite(magic, 1, 4, fp);
  fwrite(hdr, sizeof(uint32_t), 3, fp);
  fwrite(data, sizeof(float), (size_t)w * (size_t)h * (size_t)c, fp);
  fclose(fp);
  return 0;
}

static int os_cmp_double(const void* a, const void* b) {
  double da = *(const double*)a, db = *(const double*)b;
  return (da < db) ? -1 : (da > db ? 1 : 0);
}

static void os_write_telemetry(const RunConfig* cfg, double* ms_arr, int n_ms,
                               double pass1_ms, double pass2_ms) {
  double sorted[1024];
  int n = (n_ms > 1024) ? 1024 : n_ms;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) { sorted[i] = ms_arr[i]; sum += ms_arr[i]; }
  qsort(sorted, (size_t)n, sizeof(double), os_cmp_double);
  double med = (n > 0) ? sorted[n / 2] : 0.0;
  double p95 = (n > 0) ? sorted[(int)((n - 1) * 0.95)] : 0.0;
  /* Contention-robust floor estimates (other GPU clients only ever ADD time to a frame). */
  double mn  = (n > 0) ? sorted[0] : 0.0;
  double p10 = (n > 0) ? sorted[(int)((n - 1) * 0.10)] : 0.0;
  double mean = (n > 0) ? (sum / (double)n) : 0.0;
  const char* sc_names[4] = {"a","b","c","d"};
  const char* m_names[4] = {"osabt","ablated","baseline","gt"};
  char path[600];
  snprintf(path, sizeof(path), "%s/telemetry.json", cfg->out_dir);
  FILE* fp = fopen(path, "w");
  if (!fp) return;
  fprintf(fp,
    "{\n"
    "  \"scene\": \"%s\",\n"
    "  \"mode\": \"%s\",\n"
    "  \"width\": %d,\n"
    "  \"height\": %d,\n"
    "  \"frames\": %d,\n"
    "  \"gpu_ms_median\": %.6f,\n"
    "  \"gpu_ms_p95\": %.6f,\n"
    "  \"gpu_ms_mean\": %.6f,\n"
    "  \"gpu_ms_min\": %.6f,\n"
    "  \"gpu_ms_p10\": %.6f,\n"
    "  \"pass1_ms\": %.6f,\n"
    "  \"pass2_ms\": %.6f\n"
    "}\n",
    sc_names[cfg->scene_id], m_names[cfg->mode],
    cfg->width, cfg->height, cfg->frames,
    med, p95, mean, mn, p10, pass1_ms, pass2_ms);
  fclose(fp);
}

#endif /* OSABT_SCENE_SETUP_H */
