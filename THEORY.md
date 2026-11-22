# OS-ABT: Orthogonal-Subspace Antiderivative Bilinear Transport

> **Living Mathematical Derivation, Invariant Proofs, Roofline Bounds, and Rejected Dead-Ends**

---

## 1. Unified Operator Definition

We address four coupled real-time rendering failure modes simultaneously—**(1) grazing-angle micro-geometry & non-divisor discontinuity aliasing**, **(2) cross-surface projective & basis bleed across orthogonal junctions**, **(3) low-budget (1–2 spp) spatio-temporal instability under Dirac impulses and whip-pan disocclusions**, and **(4) thin-shell dielectric / white-furnace energy loss**—using a single unified closed-form mathematical operator:

$$\mathcal{T}_{\text{OS-ABT}}[L](p, t) = \underbrace{\mathcal{A}_2^{\perp}\!\left[\rho_{\text{vis}}\right](p)}_{\substack{\text{Crease-Clipped Visible-Projected}\\ \text{2nd-Order Antiderivative}}} \cdot \underbrace{\mathcal{Z}_{\text{TPT}}^{\perp}\!\left[E_{\text{inc}}\right](p, t)}_{\substack{\text{L-Stable Port-Hamiltonian ZDF-TPT}\\ \text{in Orthogonal Normal Subspace}}} + \underbrace{\mathcal{S}_{\text{Stokes}}\!\left[L_{\text{spec}}, L_{\text{trans}}\right](p, t)}_{\substack{\text{Exact Infinite-Bounce}\\ \text{Thin-Shell Reciprocity}}}$$

Zero neural networks, zero backpropagation, zero brute-force supersampling, and zero ad-hoc clamping heuristics are used anywhere in $\mathcal{T}_{\text{OS-ABT}}$.

---

## 2. Slice 0: First-Principles Invention Tournament (`H1` vs `H2` vs `H3`)

Three mutually incompatible mathematical candidates were formulated from first principles and falsified head-to-head via `slice0_falsifier.py` (193 lines, deterministic CPU double precision) on four adversarial tests:
- **T1**: Grazing $\theta = 85^\circ$ ($\mu = \cot 85^\circ \approx 0.08749$) micro-relief $H(s) = 0.015\cos(2\pi f s)$ with irrational non-divisor pitch $f = \frac{1237}{1024}\sqrt{2} \approx 1.70826$, exhibiting genuine horizon self-occlusion ($2\pi f A = 0.161 > \mu$). Footprint $\Delta x = (\mu \cdot 5)^{-1} \approx 2.286$ world units ($\approx 3.91$ irrational periods/pixel).
- **T2**: Orthogonal normal junction ($|\mathbf{n}_{\text{floor}} \cdot \mathbf{n}_{\text{pillar}}| = 0.12 < 0.15$) separating $L_{\text{floor}} = 1.0$ from a curved pillar $L_{\text{pillar}} = 0.20$.
- **T3**: 120-frame temporal sequence with 1-spp antithetic alternating noise ($\pm 0.25$) + Gaussian jitter ($\sigma = 0.06$), a $+5.0$ **Dirac impulse** at frame 40, a $+1.5$ **step shock** at frame 70, and a **disocclusion** to $L=0.15$ at frame 95.
- **T4**: Thin-shell glass slab ($\eta = 1.5$) in a unit white furnace across $\cos\theta \in [0.01, 1.0]$.

### Tournament Falsification Table (Measured by `slice0_falsifier.py`)

| Test / Metric | Unfiltered / TAA Baseline | **H1: GCWT** *(Killed)* | **H2: SCFR** *(Killed)* | **H3: OS-ABT** *(Survivor)* |
|---|---:|---:|---:|---:|
| **T1 Grazing SNR (dB)** | `0.00 dB` (`RMSE 0.22231`) | `7.98 dB` (`RMSE 0.08875`) | `35.05 dB` (`RMSE 0.00393`) | **`53.19 dB`** (`RMSE 0.000487`) |
| **T1 Step Bounds** *(true `[0.08, 0.95]`)* | — | — | `[0.0112, 1.0188]` (*±7.9% Gibbs!*) | **`[0.0800, 0.9500]`** *(Zero Gibbs)* |
| **T1 White-Furnace Max Err** | — | `4.0e-3` | `0` *(broken if Gibbs clamped)* | **`1.95e-13`** |
| **T2 Orthogonal Bleed ($|\mathbf{n}_a\!\cdot\!\mathbf{n}_b|=0.12$)** | — | `0.059259` | `0.015481` | **`0.000000`** *(Algebraic Zero)* |
| **T2 Asymmetric Crease WF Err** | — | — | — | **`1.11e-16`** |
| **T3 Steady-State RMSE (1 spp)** | `0.04199` | `0.08851` | `0.03343` | **`0.00889`** *(4.7× < TAA)* |
| **T3 Dirac Impulse Err (Frame 40)** | `4.4750` | `12.7978` | `4.7088` | **`0.0372`** *(120× < TAA)* |
| **T3 Post-Dirac Ringing (Frames 41–55)** | `0.4507` | `2.05e6` *(Diverged!)* | `0.5456` | **`0.0235`** *(Zero Ringing)* |
| **T3 Step Shock RMSE (Frames 70–80)** | `0.9093` | `3.45e15` *(Diverged!)* | `0.9192` | **`0.0245`** *(37× < TAA)* |
| **T3 Disocclusion Err (Frame 95)** | `1.3570` | `1.58e22` *(Diverged!)* | `1.4700` | **`0.0400`** *(34× < TAA)* |
| **T4 Thin-Shell Slab WF Max Err** | `0.148146` *(14.8% loss)* | — | — | **`2.22e-16`** *(Exact)* |

---

## 3. Exact Derivations & Load-Bearing Invariant Proofs (`H3: OS-ABT`)

### Theorem 1 — Visible-Projected Change-of-Variables & Exact White-Furnace Conservation
Consider a micro-heightfield $z = H(s)$ viewed at grazing elevation slope $\mu = \cot\theta = (\mathbf{v}\cdot\mathbf{n}) / \|\mathbf{v}_\perp\| > 0$. The screen-projected ray coordinate of micro-surface point $s$ is $x_{\text{proj}}(s) = s - H(s)/\mu$. The visible set is the strict upper envelope $\mathcal{V}_\mu = \{s : x_{\text{proj}}(s) > \sup_{s' < s} x_{\text{proj}}(s')\}$, partitioning each period into disjoint visible intervals $\bigcup_k [s_k^-, s_k^+]$ where each occluded gap $[s_k^+, s_{k+1}^-]$ satisfies the grazing tangency condition $x_{\text{proj}}(s_{k+1}^-) = x_{\text{proj}}(s_k^+)$.

Integrating the foreshortened visible radiance over screen footprint $[x_a, x_b]$ with Jacobian $dx_{\text{proj}} = (1 - H'(s)/\mu)\,ds$:
$$\bar{L} = \frac{1}{x_b - x_a}\sum_k \int_{s_k^-}^{s_k^+} L_{\text{micro}}(s)\left(1 - \frac{H'(s)}{\mu}\right) ds = \frac{1}{x_b - x_a}\int_{x_a}^{x_b} L_{\text{vis}}(x)\,dx$$
**Proof of White-Furnace Exactness**: When $L_{\text{micro}}(s) \equiv 1$, the sum telescopes identically: $\sum_k (x_{\text{proj}}(s_k^+) - x_{\text{proj}}(s_k^-)) = x_b - x_a$, yielding $\bar{L} = 1$ identically for every grazing angle $\theta \in [0^\circ, 90^\circ)$ and every relief height $H(s)$.

### Theorem 2 — Crease-Clipped Asymmetric 2nd-Order Antiderivative ($\mathcal{A}_2^\perp$)
Let $P = [s_m - h_L, s_m + h_R]$ be the pixel footprint along SVD major axis $\mathbf{e}_1$ clipped analytically to the co-planar side of any orthogonal crease ($h_L, h_R > 0$), and define the normalized asymmetric Bartlett kernel:
$$K_{h_L, h_R}(\tau) = \frac{2}{h_L + h_R}\begin{cases} 1 - \frac{s_m - \tau}{h_L}, & \tau \in [s_m - h_L, s_m] \\ 1 - \frac{\tau - s_m}{h_R}, & \tau \in [s_m, s_m + h_R] \end{cases}$$
Let $F_1(s) = \int_0^s f(\tau)\,d\tau$ and $F_2(s) = \int_0^s F_1(\tau)\,d\tau$. Integrating by parts twice (boundary terms vanish because $K(s_m - h_L) = K(s_m + h_R) = 0$):
$$\mathcal{A}_2^\perp[f](s_m; h_L, h_R) \equiv \int_{s_m - h_L}^{s_m + h_R} K_{h_L, h_R}(\tau) f(\tau)\,d\tau = \frac{2}{h_L + h_R}\left(\frac{F_2(s_m + h_R) - F_2(s_m)}{h_R} - \frac{F_2(s_m) - F_2(s_m - h_L)}{h_L}\right)$$

**Corollaries**:
1. **Exact Non-Negativity & Monotonicity (Zero Gibbs)**: Since $K_{h_L, h_R} \ge 0$ and $\int K_{h_L, h_R} = 1$, $f \in [m, M] \implies \mathcal{A}_2^\perp[f] \in [m, M]$.
2. **Exact White-Furnace Conservation at Arbitrary Creases**: For $f \equiv c$, $F_2(s) = \frac{c}{2}s^2$, giving $\frac{2}{h_L+h_R}\left(\frac{c}{2}(2s_m+h_R) - \frac{c}{2}(2s_m-h_L)\right) = c$ identically for all $h_L, h_R > 0$.
3. **Aliasing Roll-Off**: For symmetric $h_L = h_R = h$, $|\widehat{K_h}(\omega)| = \operatorname{sinc}^2(\omega h / 2\pi) = O(\omega^{-2})$ ($-40\text{ dB/decade}$), suppressing arbitrary incommensurate/irrational grazing frequencies in $O(1)$ ALU time (3 evaluations of $F_2$ per axis, or 9 for full 2D tensor-product $F_{2,2}$).
4. **Removable Singularity Guard**: For $h_L < \epsilon_h$ (or $h_R < \epsilon_h$), replace the divided difference with its exact Taylor limit $F_1(s_m) \pm \frac{h}{2}f(s_m) + \frac{h^2}{6}f'(s_m)$, preventing IEEE-754 catastrophic cancellation.

### Theorem 3 — Exact Orthogonal-Null Stiefel Subspace Projector ($\Psi_\perp$)
For surface frames $(\mathbf{x}_p, \mathbf{n}_p)$ and $(\mathbf{x}_q, \mathbf{n}_q)$, define:
$$\eta(c; c_0) = S_2\!\left(\frac{c - c_0}{1 - c_0}\right), \qquad S_2(u) = \operatorname{clamp}(u,0,1)^3(10 - 15\operatorname{clamp}(u,0,1) + 6\operatorname{clamp}(u,0,1)^2)$$
with dead-zone cutoff $c_0 = 0.20 > 0.15$, coupled with the tangent-plane out-of-plane barrier $\zeta_\perp(p, q) = S_2(1 - |(\mathbf{x}_q - \mathbf{x}_p)\cdot \mathbf{n}_p| / \tau_{\text{plane}})$ and projective polar frame folding $\mathbf{w}_{\text{proj}}(\mathbf{n}) = \frac{\eta(\mathbf{n}\cdot \mathbf{e}_k)}{\sum_j \eta(\mathbf{n}\cdot \mathbf{e}_j)}$ in the exact polar basis $\mathbf{Q}^* = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1/2} \in SO(3)$:
$$\Psi_\perp(p, q) = \eta(\mathbf{n}_p \cdot \mathbf{n}_q;\, c_0) \cdot \zeta_\perp(p, q)$$
**Proof of Algebraic Zero Orthogonal Bleed**: Whenever $|\mathbf{n}_p \cdot \mathbf{n}_q| \le 0.15 < c_0 = 0.20$ (orthogonal floor/wall/curved-pillar junction) or opposite sides of a thin parapet shell ($\mathbf{n}_p \cdot \mathbf{n}_q \le 0$), $(c - c_0)/(1 - c_0) \le 0 \implies \Psi_\perp(p, q) \equiv 0$ bit-for-bit, while adjacent pixels on smooth curved surfaces ($\Delta\theta \le 10^\circ \implies c \ge 0.985$) enjoy $C^2$-smooth full coupling $\eta(c) > 0.999$.

### Theorem 4 — L-Stable Port-Hamiltonian ZDF-TPT Temporal Integrator ($\mathcal{Z}_{\text{TPT}}^\perp$)
To eliminate Crank-Nicolson / naive trapezoidal ringing ($R(-\infty) = -1$) while preserving zero-delay implicit Nyquist suppression, define the adaptive-damping bilinear integrator with dimensionless conductance $g_n \in [g_{\min}, \infty)$, $G_n = \frac{g_n}{1 + g_n} \in (0, 1]$, and **critical port-Hamiltonian shock damping $\beta_n = G_n$**:
$$v_n = G_n (u_n - s_{n-1}), \qquad y_n = s_{n-1} + v_n, \qquad s_n = y_n + (1 - G_n) v_n$$
**Proof of All Four Temporal Invariants**:
1. **Unconditional Monotone L-Stability (Zero Ringing / Zero Overshoot)**: Eliminating $y_n, v_n$ gives $s_n = (1 - G_n)^2 s_{n-1} + G_n(2 - G_n) u_n$. The state spectral radius is:
   $$\rho(g_n) = (1 - G_n)^2 = \frac{1}{(1 + g_n)^2} \in (0, 1) \quad \forall g_n \in (0, \infty), \qquad \lim_{g_n \to \infty} \rho(g_n) = 0$$
   Because $\rho(g_n) > 0$ strictly everywhere, the impulse/step response is **monotone non-oscillatory** for EVERY conductance jump, and as $g_n \to \infty$ (Dirac shock or disocclusion), history vanishes in 1 frame with **zero overshoot**.
2. **Convexity & Exact White-Furnace DC Unity Gain**: Both $y_n = (1-G_n)s_{n-1} + G_n u_n$ and $s_n = (1-G_n)^2 s_{n-1} + (2G_n - G_n^2) u_n$ are exact convex combinations with positive weights summing to $1$. No nonlinear AABB history clamping is required; constant signals pass with error $0$.
3. **$O(g^2)$ Nyquist Antithetic Noise Rejection**: At steady-state $g$, the transfer function is $H(z) = G\frac{1 + (1-G)z^{-1}}{1 - (1-G)^2 z^{-1}}$ with $H(1) = 1$ and $|H(-1)| = \frac{G^2}{1 + (1-G)^2} \approx \frac{g^2}{2}$, beating Explicit Euler/TAA's $O(g)$ Nyquist leakage by $\sim 20\times$ ($26\text{ dB}$) at identical steady-state pole $\rho \approx 1 - 2g$.
4. **Co-Planar Kish Shock Conductance**: Computing pooled innovation $\bar{\Delta}_n = \frac{\sum_q \Psi_\perp(p,q)(u_n(q)-s_{n-1}(q))}{\sum_q \Psi_\perp(p,q)}$ and effective sample count $M_{\text{eff}} = (\sum w_q)^2 / \sum w_q^2$ purely inside the orthogonal-null subspace separates coherent lighting shocks ($z^2 = \|\bar{\Delta}_n\|^2 / (\hat{\sigma}^2/M_{\text{eff}} + \epsilon) \gg \tau_{\chi^2}$) from 1–2 spp stochastic noise ($z^2 \approx 1$) with zero cross-corner contamination.

### Theorem 5 — Exact Infinite-Bounce Thin-Shell Stokes Reciprocity ($\mathcal{S}_{\text{Stokes}}$)
For a lossless thin dielectric shell with single-interface Fresnel reflectance $F(\theta_i)$, summing all internal reflections gives:
$$R_{\text{slab}}(\theta_i) = F + \sum_{k=0}^{\infty}(1-F)^2 F^{2k+1} = \frac{2F}{1+F}, \qquad T_{\text{slab}}(\theta_i) = \sum_{k=0}^{\infty}(1-F)^2 F^{2k} = \frac{1-F}{1+F}$$
satisfying $R_{\text{slab}} + T_{\text{slab}} \equiv 1$ identically across all incidence angles $\theta_i \in [0, \pi/2)$ and all indices $\eta > 1$.

---

## 4. Explicit Prior-Art Separation Proof

| Prior Art | Mathematical Mechanism | Proven Failure Mode Solved Simultaneously by OS-ABT |
|---|---|---|
| **EWA Splatting** | Elliptical Gaussian filter on unbounded tangent plane | Gaussian tails bleed across $|\mathbf{n}_a\cdot\mathbf{n}_b|=0$ creases; Jensen error on nonlinear grazing displacement/BRDF; no temporal operator. |
| **LEAN / LEADR** | Gaussian NDF 1st/2nd slope moments in mipmaps | Fails on non-Gaussian discontinuous grazing grooves with horizon self-occlusion; mipmaps bleed across orthogonal corners. |
| **Relief / POM** | Point-ray iterative heightfield march | Zero footprint integration $\implies$ severe moiré on irrational frequencies ($1237/1024, \sqrt{2}$); loop cost explodes at $85^\circ\text{–}88^\circ$. |
| **SVGF** | Explicit Euler IIR + à-trous bilateral filter | Explicit Euler leaks $O(\alpha)$ Nyquist noise & lags Dirac shocks; Gaussian normal weights $\exp(-\Delta n/\sigma) > 0$ never vanish at creases. |
| **ReSTIR** | Point-sampled RIS reservoirs + spatial reuse | Zero analytical grazing footprint anti-aliasing; reservoir reuse leaks across thin shells without extra visibility rays. |
| **Radiance Cascades** | Multi-scale probe grid | Probe cells straddle thin parapets/orthogonal walls causing light leaks; isotropic cells fail at $85^\circ$ grazing anisotropy. |
| **SDF Cone Tracing** | Isotropic sphere-marching cone | Isotropic approximation breaks on $15\times$ anisotropic grazing footprints and bleeds across sub-cone-radius thin shells. |
| **Projective Camera Mapping** | Ray-aligned 2D texture projection | Stretches floor textures vertically up orthogonal walls/pillars; lacks orthogonal polar subspace nulling. |
| **3D Gaussian Splatting** | Volumetric 3D Gaussian primitives | Smooth Gaussian support bleeds across sharp creases and breaks analytical thin-shell Fresnel & grazing horizon conservation. |

---

## 5. Apple M1 Roofline Analysis (1280×720 = 921,600 px)
- **Hardware Ceiling**: Apple M1 GPU = `2.6 TFLOPS` FP32, `68.25 GB/s` unified LPDDR4X.
- **Pass 1 (Analytical Scene + Crease-Clipped 2D-ADAA$_2$ + 1–2 spp Direct/Indirect + Stokes Thin Shell)**: ~420 FLOPs/px (`0.15 ms` ALU floor), 24 B/px output (`0.32 ms` BW floor).
- **Pass 2 (Co-Planar $\Psi_\perp$ Shock Detection + L-Stable ZDF-TPT + Remodulation)**: ~180 FLOPs/px (`0.07 ms` ALU floor), 32 B/px DRAM traffic (`0.43 ms` BW floor).
- **Combined Roofline Floor**: ~`0.75 ms` (expected empirical wall-clock `0.9–1.8 ms/frame`, well under the `4.0 ms` hard budget).

---

## 6. Rejected Dead-Ends

### Dead-End 1 — `H1`: Gauge-Covariant Clifford Wavelet Transport (GCWT) + Explicit RK2
- **Killed in Slice 0 (`slice0_falsifier.py`)**.
- **Exact Measured Failures**:
  1. **Grazing Non-Divisor Aliasing**: Achieved only `7.98 dB` SNR (`RMSE = 0.088745`, failing the `>= 18 dB` gate by `> 10 dB`) because snapping footprints to dyadic wavelet scales $2^j$ introduces shift-variance and spectral leakage against irrational pitch $f = \frac{1237}{1024}\sqrt{2}$.
  2. **Orthogonal Bleed**: Soft Clifford scalar product $\max(0, \mathbf{n}_a\cdot\mathbf{n}_b)$ leaked `0.059259` luminance across $|\mathbf{n}_a\cdot\mathbf{n}_b|=0.12$.
  3. **Explicit RK2 Instability Cliff**: Under adaptive shock gain ($k > 2$), explicit RK2 violated its stability region $|1 + z + \frac{1}{2}z^2| \le 1$, exploding catastrophically (`post_dirac_max_ring = 2.05e6`, `step_rmse = 3.45e15`, `disocc_err = 1.58e22`).

### Dead-End 2 — `H2`: Spectral Characteristic-Function Rendering (SCFR) + Fixed-Pole [1/1] Padé
- **Killed in Slice 0 (`slice0_falsifier.py`)**.
- **Exact Measured Failures**:
  1. **Gibbs Overshoot & Positivity Violation**: Even with $K=6$ exact Fourier harmonics multiplied by $\operatorname{sinc}$ characteristic function, sharp step discontinuities oscillated outside true `[0.08, 0.95]` bounds (`[0.0112, 1.0188]`, $\pm 7.9\%$ Gibbs error), producing negative radiance on dark steps and breaking white-furnace conservation if clamped.
  2. **Orthogonal Bleed**: Gaussian Steklov normal barrier leaked `0.015481` across $|\mathbf{n}_a\cdot\mathbf{n}_b|=0.12$.
  3. **Fixed-Pole Shock Lag**: Linear time-invariant Padé ($g = 0.06$) had massive lag under shocks (`dirac_err = 4.7088`, `step_rmse = 0.9192`, `disocc_err = 1.4700`).

### Dead-End 3 — Naive Un-damped Trapezoidal ZDF-TPT ($\beta_n = 0$) & Single-Pixel Cold Start
- **Caught analytically & empirically during Slice 0 derivation**.
- **Exact Failure Mechanism**:
  1. Pure Crank-Nicolson ($\beta = 0$) has state update $s_n = 2y_n - s_{n-1}$ with eigenvalue $\rho(g) = \frac{1-g}{1+g} \to -1$ as $g \to \infty$, producing $+100\%$ alternating-sign ringing after any Dirac shock ($g > 1$). Replacing $\beta_n = 0$ with critical port-Hamiltonian damping $\beta_n = G_n = \frac{g_n}{1+g_n}$ eliminated ringing completely ($\rho(g) = (1+g)^{-2} \in (0, 1)$).
  2. Initializing the neighborhood state $s_{\text{nb}}$ from a single noisy antithetic pixel caused a spurious Frame-1 innovation spike (`ss_rmse = 0.05695` vs `0.00889` after fixing to co-planar initialization).

### Dead-End 4 — Slice 3 v1 Monolithic FP32 Kernel (5/6 Sensors Failed on First Full Oracle Run)
- **Measured on Apple M1 (`artifacts/oracle_report.json` v1)**: Only `S1` passed (`max_abs_dev = 0.0`); `S2`, `S3`, `S4`, `S5`, `S6` all failed with cleanly isolated root causes:
  1. **S3 Grazing Aliasing (`7.83 dB` spatial, `8.97 dB` shimmer vs `>= 18 dB` gate)**: Evaluating the transverse inlay at planar $y = 0$ (`h.pos.x`) instead of the visible ridge elevation $y = H_{\text{vis}}$ (`x_hit = ro.x + (h.pos.x - ro.x) * (1 - H_vis / ro.y)`) shifted the transverse inlay by `0.077 m` at grazing edges (`MAE = 0.17` on ridge rows vs `0.002` on valleys), and omitting local $\xi^{-1}$ phase resolution when $\Delta\xi < 2\pi$ blurred resolved sub-period ridge slopes. Fixing both raised offline spatial SNR from `7.83 dB` to `37.23 dB` (`RMSE = 0.00490` vs baseline `0.03694`) and shimmer SNR to `27.17 dB`.
  2. **S4 Orthogonal Bleed (`junction_mean_err = 0.0572` vs `<= 0.008` gate)**: An infinite `y = 2.40` ceiling plane blocked 100% of Scene A sunlight (`vis === 0`), while jittering primary rays in `MODE_GT` across macroscopic foreground/alcove silhouettes mixed the `3.8` backlit alcove into `sid = 2, 3` G-buffer pixels (`parapet near alcove = 1.026` error), inverting the bleed metric in favor of blurring.
  3. **S5 Temporal Shock (`dirac_f40_rmse = 0.5444`, `steady_rmse = 0.0905`, `whip_disocc_rmse = 0.1109`)**: Additive ZDF-TPT reset destroyed 39 frames of converged soft-shadow visibility at Frame 40 (`6.5x` Dirac shock), leaving `5.0 / sqrt(13) = 1.38` 1-spp shadow variance in penumbras; comparing 3-channel $\|\bar{\Delta}\|^2$ against 1-channel luma variance triggered spurious steady-state resets; and a 13-tap radius-2 spatial gather was too narrow for first-frame whip-pan disocclusion. Resolved via **Gauge-Covariant Port-Hamiltonian ZDF-TPT** ($\kappa_c = \bar{u}_{\perp,c}/\bar{s}_{\perp,c}$ weighted by history similarity inside $\Psi_\perp$) + 2-stage hierarchical $\Psi_\perp$ (stride-1 + stride-3, effective `9x9` = 81 co-planar pixels).
  4. **S2 CPU↔GPU Parity (`max_abs_err = 0.0154` on Scene B, `0.177` on Scene C f40 vs `<= 5e-3` gate)**: `MTLMathModeSafe` left `mathFloatingPointFunctions` at default `Fast` (12–16 bit trig approximations on macOS 15+), and divided differences `(F2(x+h)-F2(x))/h` in `os_adaa2_step` suffered fp32 catastrophic cancellation for small footprints `h/P < 0.05`. Resolved via `MTLMathFloatingPointFunctionsPrecise` + exact closed-form Bartlett CDF $B(t)$ for $h_L + h_R \le P$.
  5. **S6 Apple M1 Budget (`max_osabt_gpu_ms = 8.92 ms` vs `<= 4.0 ms` gate)**: Dispatching Pass 2 on analytical Scenes B/D (`~2.4 ms` wasted bandwidth), 14 redundant `sin/cos` calls per pixel for `tan(fov/2)`, and 13 scattered random `prev_*` reprojection fetches per pixel thrashing M1's L2 cache. Resolved by skipping Pass 2 on analytical scenes, precomputing scaled camera basis vectors, replacing 13 scattered reprojections with 1 coherent reprojection in Pass 2A + pixel-aligned stride-3 Pass 2B, and compacting intermediate G-buffer/irradiance traffic.
