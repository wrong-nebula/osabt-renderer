# Workspace Execution Rules & Typed Commands

## Hardware & Safety Constraints
- **Machine**: Apple M1, 16 GB unified memory (`arm64`, macOS 24.6.0).
- **GPU**: Apple M1 integrated GPU (`MTLCreateSystemDefaultDevice()` -> `Apple M1`, 1024 max threads/threadgroup).
- **Serialization Rule**: NEVER launch parallel GPU captures, parallel headless browsers, or concurrent heavy compiles/renders. Run one Metal/browser job at a time.
- **Image Isolation Rule**: NEVER call `read` on `.png`/`.jpg`/`.webp` files in the root session. Always inspect images via `subagent(agent="visual-inspector", ...)`.

## Typed Commands
- **Python Environment (`.venv`)**:
  ```bash
  .venv/bin/python <script.py>
  ```
- **Slice 0 Falsifier**:
  ```bash
  .venv/bin/python slice0_falsifier.py
  ```
- **Build Native C11 CPU Reference**:
  ```bash
  clang -std=c11 -O3 -ffp-contract=off -Wall -Wextra -Werror -o build/osabt_cpu src/cpu_ref.c -lm
  ```
  *(Note: `-ffp-contract=off` guarantees deterministic IEEE-754 parity with Metal/WGSL.)*
- **Build Native Apple M1 Metal Runner**:
  ```bash
  clang -x objective-c -framework Metal -framework Foundation -O3 -Wall -Wextra -o build/osabt_metal src/metal_runner.m -lm
  ```
- **Frozen Adversarial Oracle Verification**:
  ```bash
  shasum -a 256 -c ORACLE.sha256 && .venv/bin/python verify_render.py
  ```
- **Seeing Space Local Server**:
  ```bash
  .venv/bin/python workbench/server.py --port 8765
  ```
- **Context Meter Check**:
  ```bash
  conductor-meter current --compact
  ```
