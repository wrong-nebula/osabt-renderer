# OS-ABT

Orthogonal-Subspace Antiderivative Bilinear Transport: a research renderer that reconstructs a
clean, temporally stable image from one sample per pixel, using only surface normals and depth.
It has a C11 CPU reference and an Apple Metal compute implementation that share one math core
(`src/osabt_core.h`), and a frozen oracle that checks one against the other and both against
ground truth.

**Sponza walkthrough:** https://wrong-nebula.github.io/osabt-renderer/

The derivation is in [THEORY.md](THEORY.md).

## Build

Requires macOS on Apple Silicon with the Xcode command-line tools.

```bash
mkdir -p build
clang -std=c11 -O3 -ffp-contract=off -Wall -Wextra -Werror -o build/osabt_cpu src/cpu_ref.c -lm
clang -x objective-c -framework Metal -framework Foundation -O3 -Wall -Wextra -o build/osabt_metal src/metal_runner.m -lm
```

The Metal runner compiles its kernels from the sources in `src/` at start-up, so run it from the
repository root.

## Run

```bash
mkdir -p artifacts/out
build/osabt_metal --scene e --mode osabt --width 1280 --height 720 --frames 120 --key-frames 60,119 --out-dir artifacts/out
```

- `--scene` is one of `a`–`h` (below); `--mode` is `osabt`, `ablated`, `baseline` or `gt` (ground truth).
- Frames are written as raw float images (`.f32img`: `"OSAB"`, width, height, channels, then float32 data).

| Scene | What it is |
|---|---|
| `a`–`d` | The four oracle scenes: normal bleed, grazing aliasing, temporal shock and whip-pan, white furnace |
| `e` | Gallery: ceiling-lit room with spheres, a plinth and columns |
| `f` | Colonnade: hall lit from one side window |
| `g` | Still life: close-up of objects on a table |
| `h` | Sponza: triangle mesh traced through a BVH, sun and sky |

Scenes `a`–`g` are analytic (planes, spheres, boxes, cylinders). Scene `h` needs a mesh file, which
is not in this repository; see below.

## The oracle

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
shasum -a 256 -c ORACLE.sha256 && .venv/bin/python verify_render.py
```

`verify_render.py` and `oracle_scenes/` are frozen (`ORACLE.sha256`); the oracle covers scenes `a`–`d`
only. Its timing sensor assumes nothing else is using the GPU.

## Sponza (scene `h`)

The mesh is not redistributed here. To reproduce it you need a Draco-compressed glTF binary of the
Sponza atrium and Google's Draco decoder as bundled with three.js
(`three/examples/jsm/libs/draco/gltf/draco_decoder.js`, copied to a `.cjs` file):

```bash
node tools/decode_glb_draco.cjs sponza.glb draco_decoder.cjs artifacts/sponza/sponza_mesh.bin
build/osabt_metal --scene h --mode osabt --frames 120 --key-frames 119 --out-dir artifacts/out
```

The BVH is built at load time (`src/mesh_bvh.h`). Expect roughly 0.2 s per 1280×720 frame on an
Apple M1: this scene traces six rays per pixel through about 262,000 triangles.

Sponza atrium model: Crytek Sponza by Frank Meinl, after the original by Marko Dabrovic.
