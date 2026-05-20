# Ring Attractor Network — Experiments & Analysis

This document describes the experimental analysis commands and protocols for the ring attractor network. For the model architecture and implementation see [ring_attractor.md](ring_attractor.md).

---

## Table of Contents

10. [Analysis Methods](#10-analysis-methods)
    - [10.1 Population Vector Decoding](#101-population-vector-decoding)
    - [10.2 Distractor-Induced Drift Field Analysis](#102-distractor-induced-drift-field-analysis)
    - [10.3–10.8 Other metrics](#103-bump-width-estimation)
11. [Parameter Calibration](#11-parameter-calibration)
    - [11.1 Purpose](#111-purpose)
    - [11.2 Phase 1 — Noise Floor Estimation](#112-phase-1--noise-floor-estimation)
    - [11.3 Phase 2 — Grid Exploration](#113-phase-2--grid-exploration)
    - [11.4 Phase 3 — Aggregation and Recommendation](#114-phase-3--aggregation-and-recommendation)
    - [11.5 Outputs](#115-outputs)
    - [11.6 Caching](#116-caching)
12. [Bump Drift Analysis](#12-bump-drift-analysis)
    - [12.1 Purpose](#121-purpose)
    - [12.2 Simulation Protocol](#122-simulation-protocol)
    - [12.3 Bump Center Tracking](#123-bump-center-tracking)
    - [12.4 Displacement Metric](#124-displacement-metric)
    - [12.5 Bump Collapse Detection](#125-bump-collapse-detection)
    - [12.6 Aggregation](#126-aggregation)
    - [12.7 Outputs](#127-outputs)
    - [12.8 Caching](#128-caching)
13. [Bump Asymmetry Analysis](#13-bump-asymmetry-analysis)
    - [13.1 Purpose](#131-purpose)
    - [13.2 Asymmetry Index](#132-asymmetry-index)
    - [13.3 Simulation Protocol](#133-simulation-protocol)
    - [13.4 Measured Quantities](#134-measured-quantities)
    - [13.5 Statistical Tests](#135-statistical-tests)
    - [13.6 Outputs](#136-outputs)
    - [13.7 Caching](#137-caching)
|---|---|---|---|
| $\hat{\varphi}$, $\hat{A}$ (pop. vector) | — | All timepoints | Bump position and integrity |
| $S$ (normalization constant) | Eqs. 18–19 (static: Eq. 19 simplified) | Once per condition, clean delay | Analytical predictor of distractor susceptibility |
| $A(\Delta\varphi)$ curve (theoretical) | Eq. 7 | From bump profile, no simulation | Predicted drift field per condition |
| $\hat{A}(\Delta\varphi)$ curve (empirical) | Fig. 7B; §"Distractor analysis" p. 40 | Distractor trials, vary $\Delta\varphi$ | Measured susceptibility fingerprint |
| Bump collapse probability | — | Distractor trials | Fraction of trials where $\hat{A}$ drops below threshold |

### 10.3 Bump Width Estimation

The bump width is estimated using the **circular standard deviation**. Given the decoded center $\hat{\theta}$:

1. Compute angular deviations: $\delta_i = d(\theta_i, \hat{\theta})$
2. Compute circular resultant length:
$$R = \sqrt{\left(\frac{\sum_i r_i \cos \delta_i}{\sum_i r_i}\right)^2 + \left(\frac{\sum_i r_i \sin \delta_i}{\sum_i r_i}\right)^2}$$

3. Convert to standard deviation (von Mises approximation):
$$\sigma_{\text{bump}} = \sqrt{-2 \ln(R)}$$

$R = 1$ corresponds to a perfectly peaked distribution; $R = 0$ to uniform.

### 10.4 Drift Rate

Systematic drift during the delay is estimated by fitting a linear trend to the unwrapped bump center trajectory:

$$v_{\text{drift}} = \frac{\hat{\theta}(t_{\text{end}}) - \hat{\theta}(t_{\text{start}})}{t_{\text{end}} - t_{\text{start}}}$$

expressed in degrees/second.

### 10.5 Bump Drift

Stochastic wandering of the bump during the delay is quantified by the **final displacement** from the cue location. See [§12](#12-bump-drift-analysis) for the full description of the `ring-diffusion` command, which implements this analysis across conditions.

### 10.6 Working Memory Error

The angular error between the decoded bump position at the end of the delay and the original cue location:

$$\varepsilon = d(\hat{\theta}_{\text{final}}, \theta_{\text{stim}})$$

A bump is considered **maintained** if $\hat{A} > \tau$ at the evaluation time, where $\tau$ is the noise floor threshold from calibration ([§11.2](#112-phase-1--noise-floor-estimation)). Falls back to $\tau = 0.2$ if no calibration data is available.

### 10.7 Metrics Summary

| Metric | Key | Unit | Description |
|--------|-----|------|-------------|
| Bump center | `center_mean_deg` | degrees | Circular mean of decoded position |
| Center variability | `center_std_deg` | degrees | Circular std of decoded position |
| Decoding confidence | `amplitude_mean` | [0, 1] | Mean population vector length |
| Bump width | `width_mean_deg` | degrees | Circular std (von Mises approx.) |
| Drift rate | `drift_rate_deg_per_s` | deg/s | Systematic drift velocity |
| Displacement std | `std_deg` | degrees | Std of final displacement across trials (see [§12](#12-bump-drift-analysis)) |
| Error from cue | `error_from_cue_deg` | degrees | Angular distance to cue location |

### 10.8 Multi-Trial Aggregation

When running multiple trials, metrics are aggregated across trials as mean $\pm$ SEM (standard error of the mean):

$$\bar{m} = \frac{1}{K}\sum_{k=1}^{K} m_k, \quad \text{SEM} = \frac{s}{\sqrt{K}}$$

where $K$ is the number of trials and $s$ is the sample standard deviation.

---

## 11. Parameter Calibration

### 11.1 Purpose

Before running any ring attractor analysis, it is necessary to choose two critical parameters:

- **Stimulus amplitude** $A$ — the peak current injected into PYR neurons to encode the memory cue.
- **Inter-node coupling** $w_{\text{pyr}}^{\text{inter}}$ — the total row-sum weight of PYR→PYR connections across ring nodes (see [§2.2](ring_attractor.md#22-pyr--pyr-excitation)).

Both parameters jointly determine whether a working memory bump forms and survives the delay period. Too weak a stimulus or too little recurrent coupling → no bump forms; too strong → network saturates and peak firing rates become biologically implausible. The `ring-calibrate` command (`python -m circuit_model ring-calibrate`) sweeps a 2D grid over these two parameters, estimates the noise floor from baseline trials, and recommends the combination with the highest bump maintenance rate.

The calibration also provides the **noise floor threshold** $\tau$ that is consumed downstream by `ring-diffusion` to decide whether a bump has collapsed.

---

### 11.2 Phase 1 — Noise Floor Estimation

**Goal**: establish a data-driven threshold $\tau(w_{\text{inter}})$ that separates genuine memory bumps from spontaneous fluctuations.

**Procedure**:

1. For each value of $w_{\text{inter}}$ in the sweep, run $N_{\text{baseline}}$ trials (default: 100) **without any stimulus** (amplitude = 0). Each trial uses the same pre-computed burn-in state (network at baseline equilibrium) plus a different noise seed.

2. For each baseline trial, decode the population vector amplitude $\hat{A}$ ([§10.1](#101-population-vector-decoding)):
$$\hat{A} = |\bar{z}|, \quad \bar{z} = \frac{\sum_i r_i\, e^{i\theta_i}}{\sum_i r_i}$$

3. Collect all $\hat{A}$ values across trials. Compute the threshold as the $p_{\text{noise}}$-th percentile (default: 95th):
$$\tau(w_{\text{inter}}) = \text{percentile}_{95}\!\left(\{\hat{A}_k^{\text{baseline}}\}_{k=1}^{N_{\text{baseline}}}\right)$$

**Interpretation**: $\tau$ is the maximum $\hat{A}$ that spontaneous network noise can produce without any stimulus. Any trial with $\hat{A}_{\text{final}} > \tau$ is declared a **success** (genuine bump maintained). Because background activity depends on network coupling, $\tau$ is computed separately for each $w_{\text{inter}}$.

**Output**: `baseline_A_hat.csv` with one row per baseline trial; columns: `condition_key`, `w_inter`, `a_hat_value`.

---

### 11.3 Phase 2 — Grid Exploration

The calibration sweeps a Cartesian grid of (amplitude, $w_{\text{inter}}$) pairs. Default ranges:

| Parameter | Default sweep |
|-----------|--------------|
| Stimulus amplitude $A$ | $\{5, 10, 15, 20, 25, 30, 35, 40, 45, 50\}$ (× $I_{\text{ext}}^{\text{PYR}}$ baseline) |
| $w_{\text{inter}}$ | $\{2, 3, 4, 5, 6\}$ |

The stimulus protocol is fixed and identical to the working memory protocol in [§6.4](ring_attractor.md#64-working-memory-protocol):

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Network relaxes to baseline (computed once, cached) |
| Pre-cue baseline | 500 ms | Continued baseline |
| Cue | 250 ms | Gaussian stimulus at $\theta_{\text{stim}} = 180°$, amplitude $A$, $\sigma_{\text{stim}} = 20°$ |
| Delay | 3,000 ms (default) | No input; bump must be self-sustained |

For each grid point $(A,\, w_{\text{inter}})$, the simulation is repeated $N_{\text{trials}}$ times (default: 50) with different noise seeds. After each trial, three metrics are extracted at the end of the delay:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Population vector amplitude | $\hat{A}_{\text{final}}$ | Confidence of bump presence $\in [0,1]$ |
| Peak PYR firing rate | $r_{\text{peak}}$ | Maximum pyramidal rate (Hz) during delay |
| Angular error | $\varepsilon$ | Distance between decoded center and cue at $180°$ |

$\hat{A}$ is also recorded at regular intervals (every 200 ms) during the delay to produce timecourse plots.

---

### 11.4 Phase 3 — Aggregation and Recommendation

**Per grid point**, aggregate across $N_{\text{trials}}$ trials:

$$\text{success\_rate}(A,\, w_{\text{inter}}) = \frac{1}{N_{\text{trials}}} \sum_{k=1}^{N_{\text{trials}}} \mathbf{1}\!\left[\hat{A}_{\text{final},k} > \tau(w_{\text{inter}})\right]$$

$$\overline{\hat{A}}(A,\, w_{\text{inter}}) = \frac{1}{N_{\text{trials}}} \sum_{k=1}^{N_{\text{trials}}} \hat{A}_{\text{final},k}$$

**Recommended parameters** are selected by:

1. **Primary criterion**: highest `success_rate` in the grid.
2. **Tiebreaker**: among all grid points sharing the maximum success rate, choose the one with the highest $\overline{\hat{A}}$.
3. **Quality check**: if the recommended point has $r_{\text{peak}} > 100\,\text{Hz}$, a warning is emitted (possible over-excitation).

The recommendation is saved to `calibration_recommended.json`.

---

### 11.5 Outputs

All files are written to `figs/calibration/<n_nodes>/<params_stem>/<base_conn_label>/<condition>/`.

**Data files**:

| File | Content |
|------|---------|
| `baseline_A_hat.csv` | Per-trial baseline $\hat{A}$ values (inputs to noise floor computation) |
| `calibration_results.csv` | Per-trial results: `condition_key`, `amplitude`, `w_inter`, `trial_idx`, `seed`, `A_hat_final`, `peak_pyr_rate`, `center_final_deg`, `error_from_cue_deg` |
| `calibration_summary.csv` | Aggregated per grid point: `condition_key`, `amplitude`, `w_inter`, `success_rate`, `mean_A_hat`, `peak_pyr_rate`, `mean_error_deg`, `noise_threshold`, `n_trials` |
| `calibration_recommended.json` | Best $(A,\, w_{\text{inter}})$ with metadata |

**Diagnostic figures**:

| Figure | Description |
|--------|-------------|
| `noise_floor.png` | Histogram of baseline $\hat{A}$ for each $w_{\text{inter}}$, with threshold $\tau$ shown |
| `heatmap_success_rate.png` | 2D heatmap of success rate over the (amplitude × $w_{\text{inter}}$) grid |
| `heatmap_A_hat.png` | 2D heatmap of mean $\overline{\hat{A}}$ |
| `heatmap_peak_pyr.png` | 2D heatmap of mean peak PYR rate — used to flag over-excitation |
| `timecourses_<band>.png` | $\hat{A}(t)$ timecourses during the delay for all high-success points (success_rate ≥ 0.9), with mean ± SEM/SD |
| `scatter_summary.png` | Scatter of $\overline{\hat{A}}$ vs success rate, colored by peak PYR rate, across all conditions |

---

### 11.6 Caching

Calibration is computationally expensive (default: $6 \times 5 \times 50 = 1500$ trials per condition, plus 500 baseline trials). To avoid redundant computation:

- If `calibration_summary.csv` already exists and covers the requested (amplitude, $w_{\text{inter}}$) grid, the cached results are reused automatically.
- Each condition is cached independently, so adding a new condition to `--conditions` only runs the new simulations.
- Baseline trials are similarly cached per $w_{\text{inter}}$ value.
- Use `--no_cache` to force a complete rerun.

---

## 12. Bump Drift Analysis

### 12.1 Purpose

The `ring-calibrate` command identifies parameter regimes where a stable bump forms, but it does not measure how well that bump maintains the cue location over time. The `ring-diffusion` command addresses this: it quantifies how far the bump center **drifts** from the encoded cue across the full delay period, comparing this across experimental conditions.

The primary summary statistic is the **standard deviation of the final displacement** across trials. A small standard deviation indicates tight memory fidelity; a large one indicates that the bump wanders substantially.

---

### 12.2 Simulation Protocol

Each trial follows the standard working memory protocol ([§6.4](ring_attractor.md#64-working-memory-protocol)), identical to calibration:

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Network at baseline equilibrium (pre-computed, cached) |
| Pre-cue baseline | 500 ms | Continued baseline |
| Cue | 250 ms | Gaussian stimulus at $\theta_{\text{stim}} = 180°$, amplitude $A$, $\sigma_{\text{stim}} = 20°$ |
| Delay | configurable (default 3,000 ms) | No input; bump must be self-sustained |

For each condition in the sweep, $N_{\text{trials}}$ independent trials are run in parallel (default: 10 per condition), each with a different noise seed. The bump center trajectory is recorded at 1 ms resolution throughout the delay period (400 ms after cue offset onward, to skip the initial transient).

---

### 12.3 Bump Center Tracking

At every recorded timestep during the delay, the bump center $\hat{\theta}(t)$ and decoding confidence $\hat{A}(t)$ are extracted using the population vector ([§10.1](#101-population-vector-decoding)):

$$\hat{\theta}(t) = \arg\!\left(\frac{\sum_i r_i^{\text{PYR}}(t)\, e^{i\theta_i}}{\sum_i r_i^{\text{PYR}}(t)}\right) \mod 2\pi$$

$$\hat{A}(t) = \left|\frac{\sum_i r_i^{\text{PYR}}(t)\, e^{i\theta_i}}{\sum_i r_i^{\text{PYR}}(t)}\right|$$

The trajectory $\hat{\theta}(t)$ is then **phase-unwrapped** (i.e. $2\pi$ jumps caused by wraparound on the ring are removed), yielding a continuous, monotone-compatible signal. This makes arithmetic differences between timepoints meaningful.

---

### 12.4 Displacement Metric

The displacement of the bump at the end of the delay is its signed angular shift from the **known cue location** $\theta_{\text{stim}} = 180°$.

**Reference position**: the fixed stimulus location:

$$\hat{\theta}_{\text{ref}} = \theta_{\text{stim}} = \pi\,\text{rad}$$

Using the known cue location rather than the empirical bump position at the start of the delay avoids contamination by the transient immediately following cue offset, during which the bump is still forming and oscillating strongly (see [§13](#13-drift-field-analysis)).

**End position** (final portion of delay):

The displacement is computed over a trailing window of $W_{\text{end}} = 500\,\text{ms}$ (~5 oscillation cycles at ~10 Hz):

$$\delta(t) = \hat{\theta}(t) - \hat{\theta}_{\text{ref}}, \quad t \in [t_{\text{end}} - W_{\text{end}},\; t_{\text{end}}]$$

All $\delta(t)$ are wrapped to $(-\pi, \pi]$, then the frame with the **smallest absolute value** is selected:

$$\Delta\hat{\theta} = \delta\!\left(\arg\min_t |\delta(t)|\right)$$

**Rationale:** The bump oscillates around its attractor position (see [§13](#13-drift-field-analysis)). By finding the oscillation zero-crossing (minimum $|\delta|$), the estimate captures the DC drift component and discards the oscillatory excursion. Using a 500 ms trailing window samples ~5 full cycles, ensuring at least one zero-crossing is available.

The final result is a signed displacement in degrees:
- $\Delta\hat{\theta} > 0$: bump drifted counter-clockwise from cue
- $\Delta\hat{\theta} < 0$: bump drifted clockwise from cue
- $|\Delta\hat{\theta}|$ small: faithful memory retention

---

### 12.5 Bump Collapse Detection

If a calibration file is found (`calibration_summary.csv`, [§11.5](#115-outputs)), the noise floor threshold $\tau$ for the matching condition, amplitude, and $w_{\text{inter}}$ is retrieved automatically. A trial is declared **valid** (bump survived the full delay) if the decoding confidence at the end of the delay satisfies:

$$\hat{A}(t_{\text{end}}) \geq \tau$$

Trials that fail this criterion (bump collapsed) are excluded from the displacement distribution and flagged as `valid = 0` in the output. The fraction of surviving trials is tracked over time as a **survival curve**:

$$P_{\text{survive}}(t) = \frac{1}{N_{\text{trials}}} \sum_{k=1}^{N_{\text{trials}}} \mathbf{1}\!\left[\hat{A}_k(t) \geq \tau\right]$$

---

### 12.6 Aggregation

For each condition, the following statistics are computed across all valid trials:

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| Mean displacement | $\bar{\Delta} = \frac{1}{K}\sum_{k=1}^{K} \Delta\hat{\theta}_k$ | Systematic bias; near 0 for symmetric noise |
| Std of displacement | $\sigma_\Delta = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(\Delta\hat{\theta}_k - \bar{\Delta})^2}$ | Spread of drift — **primary measure of memory fidelity** |
| Mean absolute displacement | $\overline{\|\Delta\|} = \frac{1}{K}\sum_{k=1}^{K} \|\Delta\hat{\theta}_k\|$ | Unsigned drift magnitude |
| Valid / Total trials | $n_{\text{valid}} / n_{\text{total}}$ | Collapse rate |

The standard deviation $\sigma_\Delta$ is the most informative quantity: it is insensitive to any symmetric bias (which cancels in the mean) and directly reflects trial-to-trial variability in the final bump position.

---

### 12.7 Outputs

All files are written to `figs/diffusion/<n_nodes>/<params_stem>/<conn_label>/`.

**Data files**:

| File | Content |
|------|---------|
| `diffusion_displacement_trials.csv` | Per-trial: `condition_key`, `trial_idx`, `displacement_deg`, `valid` |
| `diffusion_displacement_summary.csv` | Per condition: `condition_key`, `mean_deg`, `std_deg`, `abs_mean_deg`, `n_valid`, `n_total`, `delay_ms`, `amplitude_factor`, `seed`, `n_trials` |
| `diffusion_amplitude.csv` | Per (condition, time): `condition_key`, `t_s`, `amp_mean`, `amp_sem`, `survival_frac`, `noise_threshold` |

**Figures**:

| Figure | Description |
|--------|-------------|
| `diffusion_displacement.png` | Violin + strip plot of signed displacement (degrees) per condition, with mean marker and sample size annotation. Primary comparison figure across conditions. |
| `diffusion_ring_snapshot.png` | For each condition: (top) PYR activity heatmap (node angle × time) during the delay with decoded center overlaid in cyan and cue position as a white dashed line; (bottom) amplitude $\hat{A}(t)$ trace with noise threshold marked in red. Shows one random valid trial (instead of the most extreme displacement trial). |

---

### 12.8 Caching

The per-trial trajectories are expensive to compute (default: $8\,\text{conditions} \times N_{\text{trials}}$ simulations of ~14 s each). The command caches its results:

- If `diffusion_displacement_summary.csv` and `diffusion_displacement_trials.csv` already exist with matching parameters (same conditions, delay, amplitude, seed, $\geq N_{\text{trials}}$ rows), the simulations are skipped and results are loaded from disk.
- Use `--no_cache` to force a full rerun.

---

## 13. Bump Asymmetry Analysis

### 13.1 Purpose

The bump asymmetry experiment tests whether the activity bump drifts **systematically** to one side of the cue location during the delay period. In an ideal, perfectly symmetric ring attractor the expected asymmetry is zero; any consistent left or right bias would reveal a structural symmetry-breaking caused by the nAChR condition (WT, APP, KO variants). Comparing the asymmetry distribution across conditions thus probes whether APP or receptor knockout introduces a directional bias on top of the stochastic diffusion already quantified in [§12](#12-bump-drift-analysis).

### 13.2 Asymmetry Index

For each time step the activity pattern at the cue location is split into a **left half** (nodes with negative signed angular offset from the cue) and a **right half** (positive offset):

$$\text{offset}_i = \bigl((\theta_i - \theta_{\text{cue}} + 180°) \bmod 360°\bigr) - 180°$$

$$A(t) = \frac{\sum_{i:\,\text{offset}_i > 0} r_i^{\text{PYR}}(t) \;-\; \sum_{i:\,\text{offset}_i < 0} r_i^{\text{PYR}}(t)}{\sum_{i:\,\text{offset}_i \neq 0} r_i^{\text{PYR}}(t)}$$

By default (`--correct_asymmetry` on), pre-cue and delay metrics use amplitude-weighted normalized means:

$$a_{\mathcal{T},\text{corr}} = \frac{\sum_{t\in\mathcal{T}} A(t)\,\text{Amp}(t)}{\sum_{t\in\mathcal{T}} \text{Amp}(t)}$$

where $\text{Amp}(t)$ is the decoded bump amplitude. This reduces bias from condition-dependent bump strength (weak bumps are more noise-sensitive) and normalizes by total bump strength in the window. Use `--no_correct_asymmetry` to recover the raw (unweighted) mean of $A(t)$.

The index $A(t) \in [-1, +1]$:

| Value | Interpretation |
|-------|----------------|
| $+1$ | all activity on the right (clockwise) side of the cue |
| $0$ | perfectly symmetric |
| $-1$ | all activity on the left (counter-clockwise) side |

When the bump is perfectly centred on the cue the numerator is zero, so $A = 0$. Any persistent non-zero value during the delay indicates a systematic bias.

### 13.3 Simulation Protocol

Each trial runs **independently from zero initial conditions** to ensure that pre-cue spontaneous states are uncorrelated across trials and between conditions.

| Phase | Duration | Description |
|-------|----------|-------------|
| Noisy burn-in | 6 000 ms (`ASYM_SETTLING_MS`) | Noisy spontaneous activity with a trial-unique seed, no stimulus |
| Cue | 250 ms | Gaussian stimulus at the chosen cue location, $\sigma = 18°$ |
| Delay | `--delay_ms` (user-specified) | Noise-driven maintenance, no external input |

The unique per-trial seed drives both the burn-in and the delay, so each trial provides a fresh, independent spontaneous state. This is in contrast to the shared burn-in cache used in `ring-study`; the asymmetry experiment cannot reuse a single burn-in state because that would create identical pre-cue conditions across all trials.

**Key constants:**
- `ASYM_SETTLING_MS = 6000.0` ms
- `ASYM_PRE_CUE_WINDOW_MS = 500.0` ms
- `TRANSIENT_SKIP_TIME_MS = 400.0` ms (skipped after cue offset before measuring delay asymmetry)

#### Cue location and structural pre-cue bias

The asymmetry index excludes the node exactly at the cue (offset = 0) and always counts the antipodal node (offset = −180°) as "left". For even $N$ with the cue on a node this gives left = $N/2$, right = $N/2 - 1$, and a structural pre-cue bias of $-1/(N-1)$. For odd $N$, the antipodal position is never on a node so snapping to a node already produces left = right = $(N-1)/2$ — no bias.

| $N$ parity | Cue placement | Left count | Right count | Structural bias |
|------------|--------------|-----------|------------|----------------|
| Even | On node | $N/2$ | $N/2 - 1$ | $-1/(N-1)$ ≈ −0.008 (N=128), −0.004 (N=256) |
| Even | Between nodes | $N/2$ | $N/2$ | **0** |
| Odd | On node | $(N-1)/2$ | $(N-1)/2$ | **0** |

**Balance correction (on by default, disable with `--no_cue_balance`)**:
- **Even N** — cue placed at `nearest_node_angle + step/2` (halfway between two adjacent nodes), guaranteeing left = right = $N/2$.
- **Odd N** — cue snapped to the nearest node (already balanced by parity).

A diagnostic line is always printed when $N$ is even:
- Balance on: `[N=128 is even] Cue placed at 181.4063° (half-step between nodes) to balance left/right counts.`
- `--no_cue_balance`: `WARNING: N=128 is even … structural pre-cue bias ≈ -0.0079.`

**`--random_cue_location`** (default: off) — each trial draws a uniformly random cue angle in $[0°, 360°)$ using a separate RNG seeded from `seed ⊕ 0xA5A5A5A5`. A continuous random angle is never exactly on a node, so left = right = $N/2$ naturally; the balance correction is not applied.

> **Interpretation**: if the pre-cue bias disappears with the balance correction, it was entirely structural. If it persists, there may be an additional source (e.g. numerical asymmetry of the connectivity matrix).

### 13.4 Measured Quantities

For each trial the following scalar values are extracted:

**Pre-cue asymmetry** (`pre_cue_asym`) — amplitude-weighted mean of $A(t)$ over the last 500 ms of the burn-in period (before cue onset). This serves as a per-trial baseline: in the absence of any stimulus the ring should be approximately symmetric, so any pre-existing asymmetry in the spontaneous state can be tracked and correlated with the delay outcome.

**Last pre-cue asymmetry** (`last_pre_cue_asym`) — instantaneous $A(t)$ at the single time step immediately before cue onset. Captures the network state without any time-averaging.

**Delay asymmetry** (`delay_asym`) — amplitude-weighted mean of $A(t)$ from cue offset + 400 ms to the end of the delay. The 400 ms skip discards the large SOM/PYR transient that follows stimulus offset (see [§9.1](ring_attractor.md#91-mechanism)).

$$a_{\text{pre,corr}} = \frac{\sum_{t \in \mathcal{T}_{\text{pre}}} A(t)\,\text{Amp}(t)}{\sum_{t \in \mathcal{T}_{\text{pre}}} \text{Amp}(t)}, \qquad \mathcal{T}_{\text{pre}} = [t_{\text{cue}} - 500\,\text{ms},\; t_{\text{cue}})$$

$$a_{\text{delay,corr}} = \frac{\sum_{t \in \mathcal{T}_{\text{delay}}} A(t)\,\text{Amp}(t)}{\sum_{t \in \mathcal{T}_{\text{delay}}} \text{Amp}(t)}, \qquad \mathcal{T}_{\text{delay}} = [t_{\text{off}} + 400\,\text{ms},\; T]$$

With `--no_correct_asymmetry`, the code uses the raw unweighted means over each window.

**Temporal magnitude metrics** — two non-cancelling metrics are computed on the raw (unweighted) $A(t)$ timecourse within each window. Unlike the signed weighted mean, these cannot be cancelled by oscillations that alternate between R and L:

| Metric | CSV column | Formula | Interpretation |
|--------|-----------|---------|----------------|
| Mean absolute asymmetry — delay | `mean_abs_asym` | $\langle|A(t)|\rangle_{\mathcal{T}_{\text{delay}}}$ | Average magnitude of L/R imbalance during the delay, regardless of side |
| Std of asymmetry — delay | `asym_std` | $\sigma(A(t))_{\mathcal{T}_{\text{delay}}}$ | Trial-level variability of $A(t)$ during the delay; large for oscillating conditions |
| Mean absolute asymmetry — pre-cue | `mean_abs_asym_precue` | $\langle|A(t)|\rangle_{\mathcal{T}_{\text{pre}}}$ | Average magnitude of spontaneous L/R imbalance before cue onset |
| Std of asymmetry — pre-cue | `asym_std_precue` | $\sigma(A(t))_{\mathcal{T}_{\text{pre}}}$ | Trial-level variability of spontaneous $A(t)$ before cue onset |

These are computed by `compute_asymmetry_temporal_metrics()` in `analysis.py`.

### 13.5 Statistical Tests

After collecting $n_{\text{trials}}$ values of $a_{\text{delay}}$ per condition, two one-sample tests against zero are performed:

**One-sample t-test** (`scipy.stats.ttest_1samp`):

$$t = \frac{\bar{a}_{\text{delay}}}{s / \sqrt{n}}, \qquad H_0: \mu = 0$$

**Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`, two-sided, applied when $n \geq 10$): a non-parametric alternative that makes no normality assumption, appropriate given that the distribution of delay asymmetry values can be non-Gaussian and may be affected by bump collapses.

Results are printed in a table with significance markers ($* p < 0.05$, $** p < 0.01$, $*** p < 0.001$) and stored in `stats_by_condition` for annotation of summary figures.

The p-value displayed on plots uses the Wilcoxon result when available (more robust), falling back to the t-test p-value otherwise.

**Pairwise Mann-Whitney U tests** are also run for the four temporal magnitude metrics (`mean_abs_asym`, `asym_std`, `mean_abs_asym_precue`, `asym_std_precue`) across all condition pairs. Results are stored in `pairwise_stats` and displayed as significance brackets on the corresponding panels of `asymmetry_summary.png`.

### 13.6 Outputs

All outputs are written to:

```
figs/asymmetry/{N}/{params_label}/{connectivity_label}/amp{amp}_{mode}/
```

| File | Description |
|------|-------------|
| *(path mode)* | `{mode} = corrected` when `--correct_asymmetry` is enabled (default), else `uncorrected` |
| `asymmetry_trials.csv` | Per-trial data: `condition`, `trial_idx`, `seed`, `cue_deg`, `pre_cue_asym`, `last_pre_cue_asym`, `delay_asym`, `delay_ms`, `amplitude`, `random_cue` (0/1), `balance_cue` (0/1), `correct_asymmetry` (0/1), `mean_abs_asym`, `asym_std`, `mean_abs_asym_precue`, `asym_std_precue` |
| `asymmetry_distribution.png` | Violin + jittered strip plots of pre-cue and delay asymmetry per condition. Each point is coloured by asymmetry value (blue → right, yellow → left). The delay violin is annotated with the p-value and significance stars from [§13.5](#135-statistical-tests). Plot title indicates cue mode and whether asymmetry is corrected. |
| `asymmetry_correlation.png` | Two-row scatter plot: top row shows mean pre-cue (500 ms window) vs. delay asymmetry; bottom row shows instantaneous last pre-cue step vs. delay asymmetry. Each panel is annotated with Pearson $r$ and significance stars. |
| `asymmetry_summary.png` | 2×3 bar chart grouped by period. **Row 0 — Delay**: (1) mean \|asymmetry\| ± SEM, (2) mean \|A(t)\| ± SEM, (3) std(A(t)) ± SEM. **Row 1 — Pre-cue**: (4) mean \|asymmetry\| ± SEM, (5) mean \|A(t)\| ± SEM, (6) std(A(t)) ± SEM. All panels with pairwise MWU brackets. |
| `worst_case/{cond}/dashboard.png` | Ring attractor dashboard for the trial with the largest $|a_{\text{delay}}|$ per condition. |
| `worst_case/{cond}/bump_metrics.png` | Bump metrics over time (including asymmetry panel) for the worst-case trial. |
| `worst_case/{cond}/snapshot_evolution.mp4` | Animated ring snapshot evolution for the worst-case trial. |

### 13.7 Caching

The trial CSV is used as a cache on subsequent runs. Before launching simulations the command reads `asymmetry_trials.csv` and validates:
- `delay_ms` matches the current `--delay_ms` argument
- `amplitude` matches the current `--amplitude` argument
- `random_cue` (0/1) matches the current `--random_cue_location` flag
- `balance_cue` (0/1) matches whether `--no_cue_balance` is absent (1) or present (0)
- `correct_asymmetry` (0/1) matches whether `--correct_asymmetry` is enabled (1) or disabled (0)

If validation passes, already-computed trial indices are loaded and only the **missing trials** are simulated (top-up logic). This allows incremental runs: e.g. running with `--n_trials 50` after a previous `--n_trials 20` will run only 30 additional trials and merge them into the CSV. If any parameter does not match, all trials are rerun from scratch and the CSV is overwritten.

CSVs missing `correct_asymmetry` are treated as legacy caches: mode is inferred from the folder suffix (`amp*_uncorrected` → raw, `amp*_corrected` → corrected; no suffix defaults to raw). CSVs that predate `cue_deg`/`random_cue`/`balance_cue` load `cue_deg` defaulting to `STIM_CENTER_DEG` and skip flag validation (defaulting `balance_cue=1`).

---

## 14. Burn-in Stationarity Analysis

**Command**: `ring-burnin-stability`

### 14.1 Purpose

Verify that a spontaneous burn-in period of a given duration is sufficient for the ring attractor to reach stationarity. The experiment divides the burn-in into equal windows and tests whether successive windows are statistically indistinguishable. If the last windows do not differ significantly, the network has settled into its equilibrium distribution and the burn-in length is adequate.

This is complementary to `ring-study`, which uses a single shared *deterministic* (noiseless) burn-in state. Here, the burn-in is **noisy** (white noise, different seed per trial), so the question is whether the *distribution* of spontaneous states has converged, not just a single trajectory.

### 14.2 Protocol

```
Zero initial conditions
        │
        ▼
  Noisy burn-in  ──────────────────────────────────────────────────►
  (T = burnin_ms, default 10 000 ms)
  │         │         │         │         │         │    ...
  │  win 0  │  win 1  │  win 2  │  win 3  │  win 4  │
  0      1000 ms   2000 ms   3000 ms   4000 ms   5000 ms ...
```

- **100 independent trials** (default `--n_trials 100`), each starting from zero initial conditions with a unique noise seed.
- **No stimulus** — purely spontaneous dynamics throughout.
- Total duration: `burnin_ms` (default 10 000 ms) divided into `n_periods = burnin_ms / period_ms` windows (default 10 windows of 1 000 ms).
- Asymmetry is measured relative to a fixed reference angle (`--ref_deg`, default 0°) rather than a stimulus location.

| Phase | Duration | Noise |
|-------|----------|-------|
| Spontaneous activity | `burnin_ms` (default 10 000 ms) | White noise, trial-unique seed |

Key constants:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_trials` | 100 | Independent noisy trials |
| `--burnin_ms` | 10 000 ms | Total duration per trial |
| `--period_ms` | 1 000 ms | Duration of each comparison window |
| `--ref_deg` | 0.0° | Fixed reference angle for asymmetry |

### 14.3 Measured Quantities

For each trial and each window $w$ covering $[w \cdot T_p,\, (w+1) \cdot T_p)$:

| Metric | Symbol | Definition |
|--------|--------|------------|
| Mean amplitude | $\bar{A}_w$ | $\langle \text{Amp}(t) \rangle_{t \in w}$ — mean of the population-vector amplitude over the window |
| Mean absolute asymmetry | $\overline{|A|}_w$ | $\langle |A(t)| \rangle_{t \in w}$ — mean of $|A(t)|$ where $A(t)$ is computed relative to `--ref_deg` |

Amplitude is computed by `population_vector_decode()` (PYR population, `analysis.py`). Asymmetry uses `compute_bump_asymmetry()` with `stim_angle_deg` set to `ref_deg`.

### 14.4 Statistical Tests

Two complementary tests are run per condition per metric.

**Kruskal-Wallis test** (global, `scipy.stats.kruskal`): tests whether the $K$ window distributions all share the same median. Each group is the set of $n_{\text{trials}}$ per-window values for one window index.

$$H_0: \text{all } K \text{ window distributions are identical}$$

A non-significant result ($p \geq 0.05$) indicates that no window differs from the others — the metric is stationary across the burn-in.

**Pairwise Mann-Whitney U tests** (`scipy.stats.mannwhitneyu`, two-sided): run for every pair of **adjacent** windows $(w, w+1)$. This shows *where* in the burn-in the dynamics transition from non-stationary to stationary. Results are displayed as significance brackets above the box plots.

| Symbol | Threshold |
|--------|-----------|
| `***` | $p < 0.001$ |
| `**` | $p < 0.01$ |
| `*` | $p < 0.05$ |
| `n.s.` | $p \geq 0.05$ |

Brackets between adjacent boxes are drawn in black when significant and grey when not.

### 14.5 Outputs

```
figs/burnin_stability/{N}/{connectivity_label}/
├── burnin_stability_trials.csv
├── burnin_stability_summary.csv
└── burnin_stability_{cond}.png   (one per condition)
```

| File | Contents |
|------|----------|
| `burnin_stability_trials.csv` | Per-trial, per-window data: `condition`, `trial_idx`, `seed`, `window_idx`, `window_start_ms`, `window_end_ms`, `amp_mean`, `abs_asym_mean`, `burnin_ms`, `period_ms`, `ref_deg` |
| `burnin_stability_summary.csv` | Per-condition Kruskal-Wallis results: `condition`, `metric` (`amplitude` or `abs_asymmetry`), `H`, `p` |
| `burnin_stability_{cond}.png` | Two-panel figure: left panel = amplitude box plots per window; right panel = $\|A(t)\|$ box plots per window. Title shows KW p-value; adjacent-window MWU brackets overlaid. |

### 14.6 Caching

The trial CSV is used as a cache on subsequent runs. Before launching simulations the command reads `burnin_stability_trials.csv` and validates that `burnin_ms`, `period_ms`, and `ref_deg` match the current arguments. If validation passes, already-computed trial indices are skipped and only missing trials are run (top-up logic). If any parameter does not match, all trials are rerun from scratch.

---

---

---

## 17. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cerebral Cortex*, 10(9), 910-923.

3. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
