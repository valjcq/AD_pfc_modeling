# Fitting Initialization — Design Rationale

This document traces the step-by-step reasoning used to derive good initialization parameter sets for the ring attractor optimization, following the switch to the W&W 2006 physics-based transfer function.

---

## Context: the new transfer function

The transfer function for each population is:

```
Φ(I) = A · α(I − Θ) / (1 − exp(−g · α · (I − Θ)))
```

with **fixed** W&W 2006 constants:
- PYR: α=310 Hz/nA, Θ=125/310≈0.403 nA, g=0.16 s
- Interneurons: α=615 Hz/nA, Θ=177/615≈0.288 nA, g=0.087 s

The only free parameter per population is the **output scaler A**. This is the core difference from the old implementation, where A was hardcoded to 1.0 everywhere — with the physics constants, A=1 gives rates of ~25–40 Hz at typical inputs, far above the 8–10 Hz target.

---

## Step 1 — Single-node operating point

**Goal**: find `(I0_pyr, A_pyr)` such that the single node runs at ~8.5 Hz PYR and the Turing instability condition is satisfiable for the ring.

### The saturation problem with the original (pre-fix) values

The first attempt used `I0_pyr=0.51 nA, A_pyr=0.31`, placing the PYR neuron at:

```
z = g · α · (I_syn − Θ) ≈ 0.16 × 310 × (0.49 − 0.403) ≈ 4.7
```

At z=4.7, the W&W slope `dΦ/dI` is already at **96% of its asymptote** `A·α`. The slope at rest (≈92.7 Hz/nA) and at 10× cue (≈96.1 Hz/nA) are nearly identical — only a **3.5% Turing window**. No `w_pyr_pyr_inter` can simultaneously satisfy:
- `Φ'(I*_rest) · w < 1` (uniform rest state stable)
- `Φ'(I*_cue) · w > 1` (bump nucleates under cue)

### Solution: lower I0_pyr to sit on the slope's rising portion

Reducing `I0_pyr` to 0.44 nA puts the operating point at z≈1.23, where the slope is at only ~70% of asymptote. With `A_pyr=0.76` to compensate and maintain the 8.5 Hz target (deterministic):

| Parameter | Before | Intermediate | Effect |
|-----------|--------|--------------|--------|
| I0_pyr | 0.51 nA | **0.44 nA** | Lower operating point, more slope dynamic range |
| A_pyr | 0.31 | **0.76** | Compensates for lower phi_core to maintain ~8.5 Hz det |
| Turing window | 3.5% | **30.5%** | Analytically feasible range |

**Turing window** (analytical, deterministic, 10× cue factor):
- slope_rest = 163.8 Hz/nA → w_thresh_rest = 6.1e-3 nA/Hz
- slope_cue = 235.6 Hz/nA → w_thresh_cue = 4.2e-3 nA/Hz
- Window: [4.2e-3, 6.1e-3] nA/Hz (30.5% wide)

---

## Step 1b — Recalibrating A_pyr for noisy simulations

The optimizer evaluates fitness under **noise** (`sigma_noise=0.3`, white current noise into PYR). This changes the effective operating point via Jensen's inequality.

### Jensen's inequality effect

The W&W transfer function is **convex** at low z (z≈1.2 at our operating point). For a convex function, `E[Φ(I + ξ)] > Φ(E[I])` — the mean firing rate under noise exceeds the deterministic rate. Concretely:

```
noise_std = sigma_noise × I0_pyr = 0.3 × 0.44 = 0.132 nA
```

With `A_pyr=0.76`, the noisy mean PYR rate is **~15.76 Hz** — nearly double the target 8.5 Hz. Initializing the optimizer here would incur a large loss penalty from the start.

### Solution: lower A_pyr to match the noisy operating point

Reducing `A_pyr` from 0.76 to **0.40** brings the noisy mean rate close to the 8.5 Hz target while keeping I0_pyr=0.44 nA (preserving the Turing window):

| Parameter | Deterministic target | **Noisy default** | Notes |
|-----------|---------------------|-------------------|-------|
| I0_pyr | 0.44 nA | **0.44 nA** | Unchanged — controls Turing window |
| A_pyr | 0.76 | **0.40** | Calibrated so noisy mean ≈ 8.6 Hz |
| PYR rate (det) | 8.25 Hz | ~4.4 Hz | Jensen gap: det rate is ~half noisy rate |
| PYR rate (noisy, 8 trials) | — | **~8.6 Hz** | Matches 8.214 Hz target |

**Important**: deterministic ring simulations (for Turing window analysis) will show ~4.4 Hz PYR — this is expected. The Turing analysis remains valid as an approximation of the noisy-regime slope.

These are the **current defaults** in `CircuitParams()`. Calling `CircuitParams()` with no arguments yields the noisy-calibrated operating point.

---

## Step 2 — Ring connectivity initialization

### Why the analytical Turing threshold is only an upper bound

The simple condition `Φ'(I*) · w_inter < 1` describes the **uniform (k=0) spatial mode** of the ring. For bump formation (Turing instability), we need the k=1 mode to be super-critical while k=0 remains sub-critical.

The key mechanism: **global PV inhibition selectively suppresses the uniform mode**.
- When all 64 nodes fire together (k=0), global PV sees full population activity and brakes the network strongly.
- When a spatial bump forms (k≠0), the global PV averages over active and silent nodes, providing less net inhibition to the bump.

Consequence: with sufficient `w_pv_global`, `w_pyr_pyr_inter` can be set **above the naive single-node Turing threshold** while the uniform rest state remains stable. The empirical feasible range extends beyond [4.2e-3, 6.1e-3].

### 2D scan protocol

A scan over (w_inter, w_pv) space with:
- **Noisy scan (T=3000ms, burn_in=1000ms, N_SEEDS=3)**: faster than deterministic because noise helps convergence from zero IC, reducing the slow-convergence false-positive problem.
- **Why noise helps**: at high w_pv (≥6e-3), the ring converges slowly from r=0 under deterministic dynamics (~3–4 s). Using T<4s deterministically causes false positives (configs appear OK in transient but saturate at steady state). With `sigma_noise=0.3`, stochastic perturbations kick the network out of the zero fixed point faster — T=3000ms with burn_in=1000ms is sufficient.
- Grid: w_inter ∈ [5e-4, 1e-2], w_pv ∈ [5e-4, 2e-2]
- Status criterion: OK if 4 ≤ mean_pyr ≤ 15 Hz and spatial CV < 0.10 (averaged over N_SEEDS)

### Empirical bump test

The analytical Turing condition (Φ'·w at cue) is reported but **is not the selection criterion**. Instead, a direct empirical test is performed:
- Apply 10× baseline cue at 0° for 500ms (after 2s steady state)
- Check if spatial bump persists: max_pyr / mean_pyr > 2.0 post-cue

The 10× cue factor is arbitrary but biologically plausible (strong, brief sensory input).

### Validated initialization

**Selected**: `w_pyr_pyr_inter = 4e-3 nA/Hz`, `w_pv_global = 8e-3 nA/Hz`, `sigma_pyr_deg = 15°`

| Metric | Value | Method | Target |
|--------|-------|--------|--------|
| PYR rest | ~8.6 Hz | noisy (5 seeds) | ~8.5 Hz |
| Spatial CV | ~0.02 | noisy | < 0.10 |
| Bump ratio (10× cue, det) | ~5.3 | deterministic | > 2.0 |
| Bump ratio (10× cue, noisy) | ~1.1 | noisy | informational |

The noisy bump ratio (~1.1) is expected at initialization: the ring is near the Turing threshold, and noise blurs spatial structure. The deterministic ratio (~5.3) confirms the ring is in the bump-capable regime. The optimizer will push toward clearer, more stable bumps.

**Why this point**: it is the largest `w_inter` that remains in a stable uniform rest state under noisy simulation, placing the initialization as close as possible to the Turing threshold. This minimizes the distance the optimizer needs to travel to reach the bump-forming regime.

---

## Step 3 — A-bug fix in the ring simulation

The ring Numba-compiled loop (`_fast_ring_loop.py`) called `_phi_scalar(I, Θ, α, g)` without the `A` parameter, defaulting to A=1.0 for all populations. This made the ring run at ~24 Hz PYR instead of 8 Hz, and made the Turing analysis invalid (wrong operating point).

Fix: added `A` as a parameter to `_phi_scalar` and `_ring_euler_loop`, propagated from `simulation.py`.

**Important**: Numba caches compiled functions. After this fix, the `__pycache__` directory retains stale `.nbi/.nbc` files that override the new code silently. The notebook includes a cache-clearing cell that must run before any ring simulation.

---

## Summary: initialization files

| File | Key parameters |
|------|---------------|
| `params/fit_init.json` | I0_pyr=0.44, A_pyr=0.40, sigma_noise=0.3 |
| `params/init/single_ring_init.json` | I0_pyr=0.44, A_pyr=0.40 |
| `params/init/network_ring_init.json` | w_inter=4e-3, w_pv=8e-3, sigma=15° |

These values place the optimizer in the Turing-feasible regime from the start, with noisy operating-point rates ≈ 8.5 Hz PYR. The deterministic rates at these parameters (~4.4 Hz PYR) are intentionally below target — Jensen's inequality will boost them to ~8.6 Hz when sigma_noise=0.3 is applied.
