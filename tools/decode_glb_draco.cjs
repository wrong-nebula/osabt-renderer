#!/usr/bin/env node
/*
 * One-off asset bake: decode a Draco-compressed .glb (KHR_draco_mesh_compression) into the flat
 * triangle file the OS-ABT runners load for mesh scenes.
 *
 *   node tools/decode_glb_draco.cjs <in.glb> <draco_decoder.js (CommonJS build)> <out.bin>
 *
 * The Draco decoder is Google's reference decoder as bundled with three.js
 * (three/examples/jsm/libs/draco/gltf/draco_decoder.js); copy it to a .cjs file so Node loads it
 * as CommonJS. Node transforms (translation / rotation / scale / matrix) are applied.
 *
 * Output layout (little endian):
 *   char[4] "OSMS" | uint32 triangle_count | per triangle: 9 x float32 positions (v0,v1,v2)
 *                                                          9 x float32 vertex normals (n0,n1,n2)
 */
const fs = require('fs');
const path = require('path');

const [, , glbPath, decoderPath, outPath] = process.argv;
if (!glbPath || !decoderPath || !outPath) {
  console.error('usage: decode_glb_draco.cjs <in.glb> <draco_decoder.cjs> <out.bin>');
  process.exit(2);
}

const buf = fs.readFileSync(glbPath);
if (buf.toString('latin1', 0, 4) !== 'glTF') throw new Error('not a GLB file');
let json = null, bin = null;
for (let off = 12; off < buf.readUInt32LE(8);) {
  const len = buf.readUInt32LE(off), type = buf.toString('latin1', off + 4, off + 8);
  const body = buf.subarray(off + 8, off + 8 + len);
  if (type === 'JSON') json = JSON.parse(body.toString('utf8'));
  else if (type.startsWith('BIN')) bin = body;
  off += 8 + len;
}
if (!json || !bin) throw new Error('GLB is missing its JSON or BIN chunk');

/* ---- 4x4 column-major node matrices ---- */
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; ++c) for (let r = 0; r < 4; ++r)
    for (let k = 0; k < 4; ++k) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
function localMatrix(n) {
  if (n.matrix) return n.matrix.slice();
  const [tx, ty, tz] = n.translation || [0, 0, 0];
  const [x, y, z, w] = n.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = n.scale || [1, 1, 1];
  return [
    (1 - 2 * (y * y + z * z)) * sx, (2 * (x * y + z * w)) * sx, (2 * (x * z - y * w)) * sx, 0,
    (2 * (x * y - z * w)) * sy, (1 - 2 * (x * x + z * z)) * sy, (2 * (y * z + x * w)) * sy, 0,
    (2 * (x * z + y * w)) * sz, (2 * (y * z - x * w)) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1];
}

require(path.resolve(decoderPath))().then((draco) => {
  const decoder = new draco.Decoder();
  const chunks = [];
  let triCount = 0;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];

  function decodePrimitive(prim, m) {
    const ext = prim.extensions && prim.extensions.KHR_draco_mesh_compression;
    if (!ext) throw new Error('primitive is not Draco-compressed (not supported by this tool)');
    const bv = json.bufferViews[ext.bufferView];
    const data = bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
    const dbuf = new draco.DecoderBuffer();
    dbuf.Init(new Int8Array(data.buffer, data.byteOffset, data.byteLength), data.byteLength);
    const mesh = new draco.Mesh();
    const status = decoder.DecodeBufferToMesh(dbuf, mesh);
    if (!status.ok()) throw new Error('Draco decode failed: ' + status.error_msg());
    const nv = mesh.num_points(), nf = mesh.num_faces();

    function attr(id, comps) {
      const a = decoder.GetAttributeByUniqueId(mesh, id);
      const arr = new draco.DracoFloat32Array();
      decoder.GetAttributeFloatForAllPoints(mesh, a, arr);
      const out = new Float32Array(nv * comps);
      for (let i = 0; i < out.length; ++i) out[i] = arr.GetValue(i);
      draco.destroy(arr);
      return out;
    }
    const pos = attr(ext.attributes.POSITION, 3);
    const nrm = ext.attributes.NORMAL !== undefined ? attr(ext.attributes.NORMAL, 3) : null;

    const out = new Float32Array(nf * 18);
    const ia = new draco.DracoInt32Array();
    for (let f = 0; f < nf; ++f) {
      decoder.GetFaceFromMesh(mesh, f, ia);
      const p = [];
      for (let k = 0; k < 3; ++k) {
        const i = ia.GetValue(k) * 3;
        const x = pos[i], y = pos[i + 1], z = pos[i + 2];
        p.push([m[0] * x + m[4] * y + m[8] * z + m[12],
                m[1] * x + m[5] * y + m[9] * z + m[13],
                m[2] * x + m[6] * y + m[10] * z + m[14]]);
      }
      /* geometric normal as fallback, and to orient nothing: winding is kept as authored */
      const e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
      const e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
      let g = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const gl = Math.hypot(g[0], g[1], g[2]);
      if (!(gl > 0)) continue;                      /* drop degenerate triangles */
      g = [g[0] / gl, g[1] / gl, g[2] / gl];
      const o = triCount * 18 - chunkBase;
      for (let k = 0; k < 3; ++k) {
        out[o + k * 3] = p[k][0]; out[o + k * 3 + 1] = p[k][1]; out[o + k * 3 + 2] = p[k][2];
        for (let c = 0; c < 3; ++c) { lo[c] = Math.min(lo[c], p[k][c]); hi[c] = Math.max(hi[c], p[k][c]); }
        let n = g;
        if (nrm) {
          const i = ia.GetValue(k) * 3;
          /* rotation/uniform-scale part of the node matrix is enough for these assets */
          const x = nrm[i], y = nrm[i + 1], z = nrm[i + 2];
          const t = [m[0] * x + m[4] * y + m[8] * z, m[1] * x + m[5] * y + m[9] * z, m[2] * x + m[6] * y + m[10] * z];
          const tl = Math.hypot(t[0], t[1], t[2]);
          if (tl > 1e-12) n = [t[0] / tl, t[1] / tl, t[2] / tl];
        }
        out[o + 9 + k * 3] = n[0]; out[o + 9 + k * 3 + 1] = n[1]; out[o + 9 + k * 3 + 2] = n[2];
      }
      ++triCount;
    }
    draco.destroy(ia); draco.destroy(mesh); draco.destroy(dbuf);
    chunks.push(Buffer.from(out.buffer, 0, (triCount * 18 - chunkBase) * 4));
  }

  let chunkBase = 0;
  function visit(ni, parent) {
    const n = json.nodes[ni];
    const m = mul(parent, localMatrix(n));
    if (n.mesh !== undefined) {
      for (const prim of json.meshes[n.mesh].primitives) {
        if ((prim.mode === undefined ? 4 : prim.mode) !== 4) continue;
        chunkBase = triCount * 18;
        decodePrimitive(prim, m);
      }
    }
    for (const c of n.children || []) visit(c, m);
  }
  const ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const scene = json.scenes[json.scene || 0];
  for (const ni of scene.nodes) visit(ni, ident);

  const header = Buffer.alloc(8);
  header.write('OSMS', 0, 'latin1');
  header.writeUInt32LE(triCount, 4);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.concat([header, ...chunks]));
  console.log(`triangles ${triCount}`);
  console.log(`bounds min ${lo.map((v) => v.toFixed(3)).join(', ')}  max ${hi.map((v) => v.toFixed(3)).join(', ')}`);
  console.log(`wrote ${outPath} (${8 + triCount * 72} bytes)`);
}).catch((e) => { console.error(e); process.exit(1); });
