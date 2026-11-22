/*
 * Host-side triangle mesh loader + BVH builder for the OS-ABT mesh scenes (CPU reference and
 * Metal runner). Not compiled into the Metal kernels: it only produces the three flat float
 * arrays that the shared core (osabt_core.h: os_mesh_trace) traverses on both back ends.
 *
 * Input file ("OSMS", written by tools/decode_glb_draco.cjs):
 *   char[4] "OSMS" | uint32 triangle_count | per triangle 9 float positions + 9 float normals
 *
 * Output arrays (all groups of float4, so they upload unchanged into `device float4*` buffers):
 *   nodes : 2 x float4 per node   (bmin.xyz, A) (bmax.xyz, B)
 *             interior: A = index of the left child (right child = A + 1), B = 0
 *             leaf    : A = first triangle,                                B = triangle count (> 0)
 *           Indices are stored as exact floats (< 2^24), so no bit reinterpretation is needed.
 *   tri_a : 3 x float4 per triangle (v0, e1 = v1 - v0, e2 = v2 - v0) in leaf order
 *   tri_n : 3 x float4 per triangle (vertex normals n0, n1, n2) in leaf order
 */
#ifndef OSABT_MESH_BVH_H
#define OSABT_MESH_BVH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* The CPU reference and the Metal runner each build the BVH themselves and are compiled with
 * different flags; forbid fused multiply-add here so both produce the identical tree (and hence
 * the identical leaf order the traversal visits). */
#pragma STDC FP_CONTRACT OFF

#define OS_MESH_LEAF_MAX   4
#define OS_MESH_SAH_BINS   16
#define OS_MESH_MAX_DEPTH  44   /* os_mesh_trace keeps a 48-entry traversal stack */

typedef struct {
  float*   nodes;     /* 8 floats per node */
  float*   tri_a;     /* 12 floats per triangle */
  float*   tri_n;     /* 12 floats per triangle */
  uint32_t n_nodes;
  uint32_t n_tris;
  int      max_depth;
} OsMesh;

typedef struct {
  const float* src;   /* 18 floats per source triangle */
  float*    cen;      /* 3 per source triangle */
  float*    tmin;     /* 3 per source triangle */
  float*    tmax;     /* 3 per source triangle */
  uint32_t* idx;
  float*    nodes;
  uint32_t  n_nodes;
  int       max_depth;
} OsBvhBuild;

static void os_bvh_build_node(OsBvhBuild* b, uint32_t node, uint32_t first, uint32_t count, int depth) {
  float bmin[3] = { 1e30f, 1e30f, 1e30f }, bmax[3] = { -1e30f, -1e30f, -1e30f };
  float cmin[3] = { 1e30f, 1e30f, 1e30f }, cmax[3] = { -1e30f, -1e30f, -1e30f };
  for (uint32_t i = first; i < first + count; ++i) {
    uint32_t t = b->idx[i];
    for (int a = 0; a < 3; ++a) {
      if (b->tmin[t * 3 + a] < bmin[a]) bmin[a] = b->tmin[t * 3 + a];
      if (b->tmax[t * 3 + a] > bmax[a]) bmax[a] = b->tmax[t * 3 + a];
      if (b->cen[t * 3 + a] < cmin[a]) cmin[a] = b->cen[t * 3 + a];
      if (b->cen[t * 3 + a] > cmax[a]) cmax[a] = b->cen[t * 3 + a];
    }
  }
  float* n = &b->nodes[(size_t)node * 8];
  n[0] = bmin[0]; n[1] = bmin[1]; n[2] = bmin[2];
  n[4] = bmax[0]; n[5] = bmax[1]; n[6] = bmax[2];
  if (depth > b->max_depth) b->max_depth = depth;

  if (count <= OS_MESH_LEAF_MAX || depth >= OS_MESH_MAX_DEPTH) {
    n[3] = (float)first; n[7] = (float)count;
    return;
  }

  /* Binned surface-area heuristic over all three axes */
  int best_axis = -1, best_bin = 0;
  float best_cost = 1e30f;
  for (int a = 0; a < 3; ++a) {
    float ext = cmax[a] - cmin[a];
    if (!(ext > 1e-9f)) continue;
    float lo[OS_MESH_SAH_BINS][3], hi[OS_MESH_SAH_BINS][3];
    uint32_t cnt[OS_MESH_SAH_BINS];
    for (int k = 0; k < OS_MESH_SAH_BINS; ++k) {
      cnt[k] = 0;
      for (int c = 0; c < 3; ++c) { lo[k][c] = 1e30f; hi[k][c] = -1e30f; }
    }
    float scale = (float)OS_MESH_SAH_BINS / ext;
    for (uint32_t i = first; i < first + count; ++i) {
      uint32_t t = b->idx[i];
      int k = (int)((b->cen[t * 3 + a] - cmin[a]) * scale);
      if (k >= OS_MESH_SAH_BINS) k = OS_MESH_SAH_BINS - 1;
      if (k < 0) k = 0;
      cnt[k]++;
      for (int c = 0; c < 3; ++c) {
        if (b->tmin[t * 3 + c] < lo[k][c]) lo[k][c] = b->tmin[t * 3 + c];
        if (b->tmax[t * 3 + c] > hi[k][c]) hi[k][c] = b->tmax[t * 3 + c];
      }
    }
    float right_area[OS_MESH_SAH_BINS];
    uint32_t right_cnt[OS_MESH_SAH_BINS];
    float rlo[3] = { 1e30f, 1e30f, 1e30f }, rhi[3] = { -1e30f, -1e30f, -1e30f };
    uint32_t rc = 0;
    for (int k = OS_MESH_SAH_BINS - 1; k > 0; --k) {
      rc += cnt[k];
      for (int c = 0; c < 3; ++c) { if (lo[k][c] < rlo[c]) rlo[c] = lo[k][c]; if (hi[k][c] > rhi[c]) rhi[c] = hi[k][c]; }
      float dx = rhi[0] - rlo[0], dy = rhi[1] - rlo[1], dz = rhi[2] - rlo[2];
      right_area[k] = (rc > 0) ? (dx * dy + dy * dz + dz * dx) : 0.0f;
      right_cnt[k] = rc;
    }
    float llo[3] = { 1e30f, 1e30f, 1e30f }, lhi[3] = { -1e30f, -1e30f, -1e30f };
    uint32_t lc = 0;
    for (int k = 0; k < OS_MESH_SAH_BINS - 1; ++k) {
      lc += cnt[k];
      for (int c = 0; c < 3; ++c) { if (lo[k][c] < llo[c]) llo[c] = lo[k][c]; if (hi[k][c] > lhi[c]) lhi[c] = hi[k][c]; }
      if (lc == 0 || right_cnt[k + 1] == 0) continue;
      float dx = lhi[0] - llo[0], dy = lhi[1] - llo[1], dz = lhi[2] - llo[2];
      float cost = (dx * dy + dy * dz + dz * dx) * (float)lc + right_area[k + 1] * (float)right_cnt[k + 1];
      if (cost < best_cost) { best_cost = cost; best_axis = a; best_bin = k; }
    }
  }

  uint32_t mid;
  if (best_axis >= 0) {
    float ext = cmax[best_axis] - cmin[best_axis];
    float scale = (float)OS_MESH_SAH_BINS / ext;
    uint32_t i = first, j = first + count;
    while (i < j) {
      uint32_t t = b->idx[i];
      int k = (int)((b->cen[t * 3 + best_axis] - cmin[best_axis]) * scale);
      if (k >= OS_MESH_SAH_BINS) k = OS_MESH_SAH_BINS - 1;
      if (k < 0) k = 0;
      if (k <= best_bin) { ++i; } else { --j; b->idx[i] = b->idx[j]; b->idx[j] = t; }
    }
    mid = i;
  } else {
    mid = first + count / 2;   /* all centroids coincide: split the index range evenly */
  }
  if (mid == first || mid == first + count) mid = first + count / 2;

  uint32_t left = b->n_nodes;
  b->n_nodes += 2;
  n = &b->nodes[(size_t)node * 8];
  n[3] = (float)left; n[7] = 0.0f;
  os_bvh_build_node(b, left, first, mid - first, depth + 1);
  os_bvh_build_node(b, left + 1, mid, first + count - mid, depth + 1);
}

/* Returns 0 on success. */
static int os_mesh_load(const char* path, OsMesh* m) {
  memset(m, 0, sizeof(*m));
  FILE* fp = fopen(path, "rb");
  if (!fp) { fprintf(stderr, "Cannot open mesh file %s\n", path); return -1; }
  char magic[4];
  uint32_t n = 0;
  if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "OSMS", 4) != 0 || fread(&n, 4, 1, fp) != 1 || n == 0) {
    fprintf(stderr, "Bad mesh file %s\n", path); fclose(fp); return -1;
  }
  if (n >= (1u << 23)) { fprintf(stderr, "Mesh too large (%u triangles)\n", n); fclose(fp); return -1; }
  float* src = (float*)malloc((size_t)n * 18 * sizeof(float));
  if (!src || fread(src, sizeof(float) * 18, n, fp) != n) {
    fprintf(stderr, "Truncated mesh file %s\n", path); fclose(fp); free(src); return -1;
  }
  fclose(fp);

  OsBvhBuild b;
  memset(&b, 0, sizeof(b));
  b.src  = src;
  b.cen  = (float*)malloc((size_t)n * 3 * sizeof(float));
  b.tmin = (float*)malloc((size_t)n * 3 * sizeof(float));
  b.tmax = (float*)malloc((size_t)n * 3 * sizeof(float));
  b.idx  = (uint32_t*)malloc((size_t)n * sizeof(uint32_t));
  b.nodes = (float*)calloc((size_t)n * 2 * 8, sizeof(float));
  for (uint32_t t = 0; t < n; ++t) {
    const float* p = &src[(size_t)t * 18];
    for (int a = 0; a < 3; ++a) {
      float lo = p[a], hi = p[a];
      if (p[3 + a] < lo) lo = p[3 + a];
      if (p[3 + a] > hi) hi = p[3 + a];
      if (p[6 + a] < lo) lo = p[6 + a];
      if (p[6 + a] > hi) hi = p[6 + a];
      b.tmin[t * 3 + a] = lo; b.tmax[t * 3 + a] = hi;
      b.cen[t * 3 + a] = 0.5f * (lo + hi);
    }
    b.idx[t] = t;
  }
  b.n_nodes = 1;
  os_bvh_build_node(&b, 0, 0, n, 0);

  m->n_tris = n;
  m->n_nodes = b.n_nodes;
  m->max_depth = b.max_depth;
  m->nodes = (float*)malloc((size_t)b.n_nodes * 8 * sizeof(float));
  memcpy(m->nodes, b.nodes, (size_t)b.n_nodes * 8 * sizeof(float));
  m->tri_a = (float*)calloc((size_t)n * 12, sizeof(float));
  m->tri_n = (float*)calloc((size_t)n * 12, sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    const float* p = &src[(size_t)b.idx[i] * 18];
    float* a = &m->tri_a[(size_t)i * 12];
    float* nn = &m->tri_n[(size_t)i * 12];
    for (int c = 0; c < 3; ++c) {
      a[c]     = p[c];
      a[4 + c] = p[3 + c] - p[c];
      a[8 + c] = p[6 + c] - p[c];
      nn[c]     = p[9 + c];
      nn[4 + c] = p[12 + c];
      nn[8 + c] = p[15 + c];
    }
  }
  free(b.cen); free(b.tmin); free(b.tmax); free(b.idx); free(b.nodes); free(src);
  return 0;
}

static void os_mesh_free(OsMesh* m) {
  free(m->nodes); free(m->tri_a); free(m->tri_n);
  memset(m, 0, sizeof(*m));
}

#endif /* OSABT_MESH_BVH_H */
