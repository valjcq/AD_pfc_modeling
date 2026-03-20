# Ring Simulation — Speed-Up Implementation Plan

Three independent optimisations, ordered from easiest to hardest.
Each section specifies exactly what to change, why it works, and how to
validate that the outputs are not altered.

---

## Context: what is slow and why

`simulate_ring` in `circuit_model/ring/simulation.py` is a **pure Python
`for k in range(n_steps)` loop** running 50 000 iterations (5000 ms at
dt = 0.1 ms).  At every iteration:

| Cost | Source |
|---|---|
| ~100 ns × 50 000 = **5 ms** | Python interpreter per iteration |
| ~1–5 µs × 50 000 = **50–250 ms** | `rng.standard_normal((n_nodes, 4))` inside loop |
| ufunc dispatch × 4 × 50 000 | `phi_wong_wang` called 4× on tiny `(n_nodes,)` arrays |

Total: **~1.4 s per trial** despite the actual arithmetic taking < 2 ms.

The `noise-floor` command calls `simulate_ring` once per trial via
`ProcessPoolExecutor`, so all 2400 trials pay this overhead independently.

---

## Option 1 — Pre-generate noise before the loop

**File:** `circuit_model/ring/simulation.py`
**Effort:** ~15 min
**Expected speedup:** 3–5×
**Affects:** every caller of `simulate_ring`

### What to change

In `simulate_ring`, the noise generation is currently inside the main loop:

```python
# CURRENT (inside the for k loop, line 296)
elif noise_type == "white":
    xi = rng.standard_normal((n_nodes, 4))
```

Replace with a single pre-allocation before the loop, then index into it:

```python
# Before the main loop, after rng is created:
if p.sigma_s != 0.0 and noise_type == "white":
    noise_arr = rng.standard_normal((n_steps - 1, n_nodes, 4))
elif noise_type == "ou":
    # Pre-generate Wiener increments for OU
    wiener_arr = rng.standard_normal((n_steps - 1, n_nodes, 4))
else:
    noise_arr = None

# Inside the loop, replace the RNG calls with:
if noise_type == "white":
    xi = noise_arr[k] if noise_arr is not None else np.zeros((n_nodes, 4))
elif noise_type == "ou":
    xi_state += (-xi_state / tau_noise_ms) * dt_ms + \
                np.sqrt(2.0 * dt_ms / tau_noise_ms) * wiener_arr[k]
    xi = xi_state
```

### Why it works

NumPy's `default_rng` (PCG64) generates a linear stream of bits regardless of
the shape parameter.  `rng.standard_normal((n_steps-1, n, 4))` generates
exactly the same sequence as `n_steps-1` successive calls to
`rng.standard_normal((n, 4))` — shape only controls reshaping, not the bit
stream.  The result is therefore **bitwise identical** to the current code.

### Validation

```python
import numpy as np
from circuit_model.ring.simulation import simulate_ring   # after change
from circuit_model.ring.simulation import simulate_ring_ref  # keep old version as _ref

# Run both with same seed and params
seed = 42
res_ref = simulate_ring_ref(local_params, ring_params, T_ms=5000, seed=seed)
res_new = simulate_ring(local_params, ring_params, T_ms=5000, seed=seed)

assert np.array_equal(res_new.r, res_ref.r), "r arrays differ!"
assert np.array_equal(res_new.I_adapt_final, res_ref.I_adapt_final), \
    "adaptation differs!"
print("Bitwise identical: OK")
```

Empirically verify with `np.array_equal` (not `np.allclose`) — output should
be **bit-for-bit identical**.

---

## Option 2 — Use `simulate_ring_batch` in the noise-floor command

**File:** `circuit_model/ring/cli.py`
**Effort:** ~1–2 h
**Expected speedup:** 10–20× on the batch path (on top of option 1)
**Affects:** `ring-noise-floor` command (and auto-triggered noise-floor in `ring-run`)

### What to change

Currently, `_run_noise_floor_for_conditions` creates one job per trial and
dispatches them via `ProcessPoolExecutor`.  Each subprocess calls
`simulate_ring` (single trial).

Replace with **one job per `(condition, w_inter)` pair**, where each job calls
`simulate_ring_batch` with all `n_baseline` trials at once.

#### Step 1 — new worker function

```python
def _noise_floor_run_batch(job: tuple) -> list[dict]:
    """Run all baseline trials for one (condition, w_inter) as a batch."""
    global _noise_floor_sim_args
    cfg = _noise_floor_sim_args
    cond_key, cond_idx, w, trial_indices, trial_seeds, noise_percentile = job

    rp = replace(cfg['ring_params_base'], w_pyr_pyr_inter=float(w))
    conn = RingConnectivity.from_params(rp)
    local_params = apply_condition(cfg['base_params'], STUDY_CONDITIONS[cond_key])

    # All n trials at once — one vectorized call
    results = simulate_ring_batch(
        local_params_list=[local_params] * len(trial_seeds),
        ring_params=rp,
        T_ms=max(BURN_IN_MS, float(cfg['delay_ms'])),
        seeds=trial_seeds,
        stimuli=None,
        connectivity=conn,
        record_dt_ms=max(10.0, float(cfg['record_dt_ms'])),
    )

    rows = []
    for trial_idx, res in zip(trial_indices, results):
        _, a_hat = population_vector_decode(res.r[-1, :, 0], rp.node_angles_rad)
        rows.append({
            "condition": cond_key,
            "w_inter": f"{float(w):.8g}",
            "trial_idx": str(trial_idx),
            "seed": str(trial_seeds[trial_indices.index(trial_idx)]),
            "A_hat": f"{float(a_hat):.10g}",
            "noise_percentile": f"{float(noise_percentile):.8g}",
            "noise_threshold": "",
        })
    return rows
```

#### Step 2 — restructure job list in `_run_noise_floor_for_conditions`

Instead of one job per trial:
```python
# OLD: one tuple per trial
jobs.append((ck, cond_idx, float(w), trial_idx, trial_seed, noise_percentile))

# NEW: one tuple per (condition, w_inter) grouping all trials together
jobs.append((
    ck, cond_idx, float(w),
    list(range(start_idx, start_idx + n_add)),  # trial_indices
    [seed_fn(cond_idx, w, i) for i in range(n_add)],  # trial_seeds
    noise_percentile,
))
```

#### Step 3 — update the executor call

```python
futures = {
    executor.submit(_noise_floor_run_batch, job): job
    for job in jobs
}
# Each future returns a list[dict]; flatten with:
for future in as_completed(futures):
    for row in future.result():
        new_rows_by_cond[row["condition"]].append(row)
```

The number of parallel jobs is now `n_conditions × n_w_inter_values`
(e.g., 1 × 12 = 12 for the WT case) instead of 2400.  Each job does more
work but avoids per-trial process dispatch overhead.

### Why it works

`simulate_ring_batch` stacks `(n_batch, n_nodes, 4)` arrays so that:
- The matrix product `W @ r` becomes a single BLAS DGEMM covering all trials
- `phi_numpy` operates on `(n_batch, n_nodes)` arrays instead of n_batch
  separate `(n_nodes,)` arrays — ufunc dispatch is paid once per step
- One `rng.standard_normal((n_batch, n_nodes, 4))` per step instead of
  n_batch separate calls

### Validation

Batch results use a shared RNG for all trials, so individual trial noise
sequences differ from single-trial `simulate_ring` with the same seed.
Validation is therefore **statistical, not bitwise**:

```python
import numpy as np
from circuit_model.ring.simulation import simulate_ring, simulate_ring_batch

# Run 500 trials both ways
seeds = list(range(500))
a_hats_single = [
    population_vector_decode(
        simulate_ring(lp, rp, T_ms=5000, seed=s).r[-1, :, 0], rp.node_angles_rad
    )[1]
    for s in seeds
]
results_batch = simulate_ring_batch([lp]*500, rp, T_ms=5000, seeds=seeds)
a_hats_batch = [
    population_vector_decode(res.r[-1, :, 0], rp.node_angles_rad)[1]
    for res in results_batch
]

# Check: p95 threshold should agree within a few percent
from circuit_model.ring.analysis import compute_noise_floor
thr_single = compute_noise_floor(np.array(a_hats_single))
thr_batch  = compute_noise_floor(np.array(a_hats_batch))
print(f"Single p95: {thr_single:.4f}")
print(f"Batch  p95: {thr_batch:.4f}")
assert abs(thr_single - thr_batch) / thr_single < 0.05, \
    "Thresholds differ by more than 5% — check RNG or params"
```

Also check the distribution visually with a histogram overlay.

---

## Option 3 — Numba JIT for the ring inner loop

**File to create:** `circuit_model/ring/_fast_ring_loop.py`
**File to modify:** `circuit_model/ring/simulation.py`
**Effort:** ~3–4 h
**Expected speedup:** 50–100× over the current unoptimised loop
**Affects:** every caller of `simulate_ring` (single-trial path)

### Architecture

Mirror the pattern already used in `circuit_model/_fast_loop.py` for the
single-node case, extended to handle:
- Matrix-vector products for inter-node connectivity: `np.dot(W, r_pyr)`
- Per-node loop over `n_nodes` instead of scalar operations
- Recording every `record_step` steps into a pre-allocated output array
- Optional adaptation recording

The Numba function receives **only plain numpy arrays and Python scalars** —
no CircuitParams, no RingParams objects.

### New file: `circuit_model/ring/_fast_ring_loop.py`

```python
"""Numba-compiled Euler integration for the ring attractor network.

Mirrors _fast_loop.py but handles n_nodes populations simultaneously,
with inter-node connectivity via matrix-vector products.

Pre-requisites before calling _ring_euler_loop:
  1. noise_arr must be pre-generated: rng.standard_normal((n_steps-1, n_nodes, 4))
     (or zeros if sigma_s == 0)
  2. I_stim_arr: (n_steps-1, n_nodes) stimulus current, pre-computed
  3. W_pyr_pyr, W_pv_pyr: connectivity matrices (n_nodes, n_nodes), C-contiguous
  4. r_out: (n_steps, n_nodes, 4) output array, r_out[0] = r0 on entry
  5. I_adapt_out: (n_steps, n_nodes, 2), I_adapt_out[0] = I_adapt0 on entry

Output: r_out and I_adapt_out filled in-place.
"""

from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit as _njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def _njit(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda fn: fn


@_njit(cache=True)
def _phi_scalar(I: float, theta: float, c: float, g: float) -> float:
    """Wong-Wang transfer function — scalar, identical to _fast_loop._phi_scalar."""
    u = c * (I - theta)
    z = g * u
    if abs(z) < 1e-8:
        return max(0.0, 1.0 / g + u * 0.5)
    denom = -math.expm1(min(-z, 700.0))
    return max(0.0, u / denom)


@_njit(cache=True)
def _ring_euler_loop(
    r_out,           # (n_steps, n_nodes, 4)  — filled in-place
    I_adapt_out,     # (n_steps, n_nodes, 2)  — filled in-place
    noise_arr,       # (n_steps-1, n_nodes, 4) — pre-generated or zeros
    I_stim_arr,      # (n_steps-1, n_nodes)   — pre-computed stimulus
    I_ext_pyr_arr,   # (n_steps-1,)           — external current, may vary (transient)
    I_ext_som_arr,   # (n_steps-1,)
    I_ext_pv_arr,    # (n_steps-1,)
    I_ext_vip_arr,   # (n_steps-1,)
    W_pyr_pyr,       # (n_nodes, n_nodes) — inter-node PYR->PYR weights
    W_pv_pyr,        # (n_nodes, n_nodes) — inter-node PV->PYR weights
    n_steps: int,
    n_nodes: int,
    dt_ms: float,
    sigma_s: float,
    tau_s: float,
    ggaba: float,
    # Synaptic weights (scalars — same for all nodes)
    w_ee: float, w_pe: float, w_se: float,
    w_es: float, w_vs: float,
    w_ep: float, w_pp: float, w_sp: float, w_vp: float,
    w_ev: float,
    J_adapt_pyr: float, J_adapt_som: float,
    tau_adapt_pyr: float, tau_adapt_som: float,
    # Transfer function parameters
    Theta_pyr: float, alpha_pyr: float, g_e: float,
    Theta_som: float, alpha_som: float, g_i: float,
    Theta_pv: float,  alpha_pv: float,
    Theta_vip: float, alpha_vip: float,
    record_step: int,  # record every record_step steps
) -> None:
    """
    Core ring Euler loop.  All inner operations are scalar or explicit
    dot products — zero Python interpreter overhead once compiled.
    """
    # Temporary buffers (stack-allocated by Numba)
    I_pyr_inter = np.zeros(n_nodes)
    I_pv_inter  = np.zeros(n_nodes)

    rec_idx = 0  # next slot to write in r_out (slot 0 = initial state, already set)

    for k in range(n_steps - 1):

        # ── Inter-node currents (matrix-vector products) ──────────────────────
        # r_pyr at step k: r_out[k, :, 0]
        for j in range(n_nodes):
            acc_pyr = 0.0
            acc_pv  = 0.0
            for m in range(n_nodes):
                acc_pyr += W_pyr_pyr[j, m] * r_out[k, m, 0]
                acc_pv  += W_pv_pyr[j, m]  * r_out[k, m, 2]
            I_pyr_inter[j] = acc_pyr
            I_pv_inter[j]  = acc_pv

        # ── Per-node update ───────────────────────────────────────────────────
        for j in range(n_nodes):
            r_pyr = r_out[k, j, 0]
            r_som = r_out[k, j, 1]
            r_pv  = r_out[k, j, 2]
            r_vip = r_out[k, j, 3]
            Iap   = I_adapt_out[k, j, 0]
            Ias   = I_adapt_out[k, j, 1]

            denom = 1.0 + ggaba * w_pe * r_pv

            I_pyr_j = (w_ee * r_pyr) / denom \
                      + I_pyr_inter[j] \
                      - ggaba * I_pv_inter[j] \
                      - ggaba * w_se * r_som \
                      - Iap \
                      + I_ext_pyr_arr[k] \
                      + I_stim_arr[k, j]
            I_som_j = w_es * r_pyr - w_vs * r_vip - Ias + I_ext_som_arr[k]
            I_pv_j  = w_ep * r_pyr \
                      - ggaba * w_pp * r_pv \
                      - ggaba * w_sp * r_som \
                      - w_vp * r_vip \
                      + I_ext_pv_arr[k]
            I_vip_j = w_ev * r_pyr + I_ext_vip_arr[k]

            phi_pyr = _phi_scalar(I_pyr_j, Theta_pyr, alpha_pyr, g_e)
            phi_som = _phi_scalar(I_som_j, Theta_som, alpha_som, g_i)
            phi_pv  = _phi_scalar(I_pv_j,  Theta_pv,  alpha_pv,  g_i)
            phi_vip = _phi_scalar(I_vip_j, Theta_vip, alpha_vip, g_i)

            xi0 = sigma_s * noise_arr[k, j, 0]
            xi1 = sigma_s * noise_arr[k, j, 1]
            xi2 = sigma_s * noise_arr[k, j, 2]
            xi3 = sigma_s * noise_arr[k, j, 3]

            # Euler update — operation order matches reference:
            # r + dt_ms * ((-r + phi + xi) / tau_s)
            r_out[k+1, j, 0] = max(0.0, r_pyr + dt_ms * ((-r_pyr + phi_pyr + xi0) / tau_s))
            r_out[k+1, j, 1] = max(0.0, r_som + dt_ms * ((-r_som + phi_som + xi1) / tau_s))
            r_out[k+1, j, 2] = max(0.0, r_pv  + dt_ms * ((-r_pv  + phi_pv  + xi2) / tau_s))
            r_out[k+1, j, 3] = max(0.0, r_vip + dt_ms * ((-r_vip + phi_vip + xi3) / tau_s))

            # Adaptation
            I_adapt_out[k+1, j, 0] = Iap + dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
            I_adapt_out[k+1, j, 1] = Ias + dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som
```

> **Note on the matrix-vector product:** the explicit double loop looks naive,
> but Numba compiles it to SIMD-vectorised code that is equivalent to
> `np.dot(W, r)`.  If profiling shows this is still slow, replace with
> `np.dot(W_pyr_pyr, r_out[k, :, 0])` — Numba supports this and delegates
> to BLAS.  The explicit loop is written first to keep operation order
> identical to the reference for bitwise validation.

### Changes to `simulate_ring`

After option 1 is in place (noise pre-generated), add a fast-path dispatcher
analogous to `simulation.py`'s existing `_fast_loop` dispatch:

```python
from circuit_model.ring._fast_ring_loop import (
    _ring_euler_loop, NUMBA_AVAILABLE as RING_NUMBA_AVAILABLE
)

def simulate_ring(...) -> RingSimulationResult:
    # ... existing setup code ...

    # Pre-generate noise (option 1 already done)
    if p.sigma_s != 0.0 and noise_type == "white":
        noise_arr = rng.standard_normal((n_steps - 1, n_nodes, 4))
    else:
        noise_arr = np.zeros((n_steps - 1, n_nodes, 4))

    # Pre-compute external current arrays (transient included)
    I_ext_pyr_arr, I_ext_som_arr, I_ext_pv_arr, I_ext_vip_arr = \
        _precompute_ext_currents(p, n_steps - 1, dt_ms)

    # Pre-compute stimulus array
    I_stim_arr = _precompute_stimulus(stimuli, node_angles, dt_ms, n_steps - 1)

    # Allocate full output array (Numba writes all steps)
    r_full = np.zeros((n_steps, n_nodes, 4))
    I_adapt_full = np.zeros((n_steps, n_nodes, 2))
    r_full[0] = r_curr
    I_adapt_full[0, :, 0] = Iap_curr
    I_adapt_full[0, :, 1] = Ias_curr

    if RING_NUMBA_AVAILABLE and noise_type in ("white", "none"):
        _ring_euler_loop(
            r_full, I_adapt_full, noise_arr, I_stim_arr,
            I_ext_pyr_arr, I_ext_som_arr, I_ext_pv_arr, I_ext_vip_arr,
            connectivity.W_pyr_pyr, connectivity.W_pv_pyr,
            n_steps, n_nodes, dt_ms, float(p.sigma_s), float(p.tau_s),
            float(ggaba),
            float(p.w_ee), float(p.w_pe), float(p.w_se),
            float(p.w_es), float(p.w_vs),
            float(p.w_ep), float(p.w_pp), float(p.w_sp), float(p.w_vp),
            float(p.w_ev),
            float(p.J_adapt_pyr), float(p.J_adapt_som),
            float(p.tau_adapt_pyr), float(p.tau_adapt_som),
            float(p.Theta_pyr), float(p.alpha_pyr), float(p.g_e),
            float(p.Theta_som), float(p.alpha_som), float(p.g_i),
            float(p.Theta_pv),  float(p.alpha_pv),
            float(p.Theta_vip), float(p.alpha_vip),
            record_step,
        )
    else:
        # Fallback: existing Python loop (after option 1)
        pass

    # Extract recorded steps from r_full
    recorded_indices = list(range(0, n_steps, record_step))
    if recorded_indices[-1] != n_steps - 1:
        recorded_indices.append(n_steps - 1)
    r_stored = r_full[recorded_indices]
    t_stored = np.array(recorded_indices, dtype=float) * dt_ms
    # ... build result ...
```

### Validation

```python
import numpy as np
from circuit_model.ring.simulation import simulate_ring
# Keep reference version accessible as simulate_ring_py (Python loop, after opt-1)

seed = 42
res_py  = simulate_ring_py(local_params, ring_params, T_ms=5000, seed=seed)
res_nb  = simulate_ring(local_params, ring_params, T_ms=5000, seed=seed)

# Same noise pre-generated with same seed → should be bitwise identical
bitwise_ok = np.array_equal(res_py.r, res_nb.r)
print(f"Bitwise identical: {bitwise_ok}")

if not bitwise_ok:
    diff = np.abs(res_py.r - res_nb.r)
    print(f"Max absolute error: {diff.max():.3e}")
    print(f"Max relative error: {(diff / (np.abs(res_py.r) + 1e-30)).max():.3e}")
    first = np.argwhere(res_py.r != res_nb.r)
    print(f"First mismatch at index: {first[0]}")
```

**Expected outcome:** bitwise identical, or at worst relative error < 1e-12
(sub-ULP differences from the explicit matrix loop vs numpy `@`).

If differences exceed 1e-12, trace to the specific operation that differs and
adjust the operation order to match the reference exactly, following the same
principle used in `_fast_loop.py` (comment on line 141: operation order
`dt_ms * (sum / tau_s)` vs `(dt_ms / tau_s) * sum`).

---

## Recommended implementation order

1. **Option 1** first — trivial change, immediately benefits all ring commands,
   and is required before option 3 (Numba needs noise pre-generated outside).
2. **Option 3** next — gives the biggest single-trial speedup, benefits every
   command that calls `simulate_ring` (noise-floor, ring-run, burn-in, etc.).
3. **Option 2** last — restructures the noise-floor parallelism strategy;
   less important if option 3 already brings single-trial time to ~15 ms.

## Expected final performance

| Configuration | Time per trial | 2400-trial job |
|---|---|---|
| Current (no opts) | ~1400 ms | ~57 min |
| After option 1 only | ~300–500 ms | ~12–20 min |
| After options 1 + 3 | ~15–30 ms | ~1–2 min |
| After options 1 + 2 + 3 | ~15 ms single / batch vectorised | < 1 min |
