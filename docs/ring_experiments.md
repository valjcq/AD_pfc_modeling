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
<<<<<<< HEAD
=======
14. [Burn-in Stationarity Analysis](#14-burn-in-stationarity-analysis)
    - [14.1 Purpose](#141-purpose)
    - [14.2 Protocol](#142-protocol)
    - [14.3 Measured Quantities](#143-measured-quantities)
    - [14.4 Statistical Tests](#144-statistical-tests)
    - [14.5 Outputs](#145-outputs)
    - [14.6 Caching](#146-caching)
15. [Oscillation-Distractor Experiment](#15-oscillation-distractor-experiment)
    - [15.1 Purpose](#151-purpose)
    - [15.2 Protocol Timeline](#152-protocol-timeline)
    - [15.3 Measured Quantities](#153-measured-quantities)
    - [15.4 Analysis Methods](#154-analysis-methods)
    - [15.5 Outputs](#155-outputs)
    - [15.6 Caching](#156-caching)
16. [Phase-Dependent Distractor Experiment](#16-phase-dependent-distractor-experiment)
    - [16.1 Purpose](#161-purpose)
    - [16.2 Core Design Principle: Identical Pre-Distractor State](#162-core-design-principle-identical-pre-distractor-state)
    - [16.3 Oscillation Frequency Auto-Detection](#163-oscillation-frequency-auto-detection)
    - [16.4 Protocol Timeline](#164-protocol-timeline)
    - [16.5 Measured Quantities](#165-measured-quantities)
    - [16.6 Outputs](#166-outputs)
    - [16.7 Caching](#167-caching)
17. [References](#17-references)

---

## 10. Analysis Methods

### 10.1 Population Vector Decoding
(Quite common in the literature, e.g. Wimmer et al. 2014)

The bump center is estimated using the **circular mean** (population vector) method. For activity $r_i$ at nodes with angles $\theta_i$:

$$\bar{z} = \frac{\sum_{i=0}^{N-1} r_i \, e^{i\theta_i}}{\sum_{i=0}^{N-1} r_i}$$

where $i = \sqrt{-1}$. The decoded center and amplitude are:

$$\hat{\theta} = \arg(\bar{z}) \mod 2\pi$$

$$\hat{A} = |\bar{z}| \in [0, 1]$$

$\hat{A}$ is a confidence measure: $\hat{A} = 1$ for a perfect delta-function bump, $\hat{A} \approx 0$ for uniform activity.

**Note:** 
- $\arg(\bar{z})$ is the **angle (phase)** of the complex mean vector, not an argmax over $\theta$.
- If there's two bump peaks of equal height on opposite sides of the ring, $\hat{A}$ will be low and $\hat{\theta}$ will be the circular mean between them.

### 10.2 Distractor-Induced Drift Field Analysis

> **Primary reference**: Seeholzer, Deger & Gerstner (2019), "Stability of working memory in continuous attractor networks under the control of short-term plasticity", *PLOS Computational Biology*, https://doi.org/10.1371/journal.pcbi.1006928.
> All equations below follow their notation directly. Deviations are noted explicitly.

---

#### Motivation and choice of method

The population vector (Section 10.1) is the natural readout during clean delay periods, but it conflates two qualitatively different distractor outcomes: (i) the bump shifted toward the distractor and settled at a new position, and (ii) two competing bumps coexist and the population vector returns their spurious circular mean. The maximum likelihood estimator of Compte et al. (2000) fits the post-distractor profile to a shifted pre-distractor template and identifies the dominant peak, but it is purely descriptive: it tells you *where* the bump ended up, not *why*, and it does not generalize across conditions with different bump shapes.

We instead adopt the **drift field framework of Seeholzer et al. (2019, §"Linking theory to experiments: Distractors and network size", p. 17–18)**, which reframes the distractor problem mechanistically. Rather than asking "where is the bump after the distractor?", we ask: "what force did the distractor exert on the bump, as a function of their angular separation?". This yields a quantity, the drift field $A(\varphi)$, that is directly predictable from the bump profile shape and connectable to theory, making it the appropriate analysis for comparing nAChR conditions.

---

#### Theoretical background

The key move, formalized in Seeholzer et al. (2019, §"Analysis of drift and diffusion with STP", p. 24–27), is to reduce the full $N$-dimensional network state to a single slow variable: the angular position $\varphi(t)$ of the bump center on the ring. This is valid because all states on the attractor manifold are related by translation (the network has a continuous symmetry), so any state can be written as a shifted copy of the canonical bump profile $\tilde{\phi}_0$.

Under this reduction, Seeholzer et al. (their **Eq. 4**) show that the bump center evolves according to a **one-dimensional Langevin equation**:

$$\dot{\varphi} = \sqrt{B}\, \eta(t) + A(\varphi)$$

where:
- $\eta(t)$ is white Gaussian noise with $\langle \eta(t) \rangle = 0$ and $\langle \eta(t)\,\eta(t') \rangle = \delta(t - t')$
- $B$ is the **diffusion strength** (their **Eq. 5**), set by the bump profile shape and the short-term plasticity parameters $C_i/S$ (see below)
- $A(\varphi)$ is the **drift field** (their **Eq. 7**), describing deterministic displacement of the bump due to any symmetry-breaking input — including distractors

> **Notation note:** Seeholzer et al. write the Langevin equation using $\dot{\varphi}$ (continuous time). In our discrete-time simulations we use the Euler-Maruyama form $\Delta\varphi = A(\varphi)\Delta t + \sqrt{B\,\Delta t}\,\xi$, where $\xi \sim \mathcal{N}(0,1)$. The diffusion convention follows their footnote 1: $D_{\text{Einstein}} = B/2$.

---

#### Drift field from a distractor input

In the absence of any heterogeneity, $A(\varphi) = 0$ everywhere, and the bump diffuses symmetrically. A distractor breaks this symmetry. Seeholzer et al. (2019, p. 17–18) treat the distractor as an **additional heterogeneity**: they assume it causes neurons $i$ centered around position $\varphi_D$ to fire at elevated rates $\phi_{0,i} + \Delta\phi_i$ above their steady-state bump value. (Here we follow Seeholzer et al.'s own notation: $\phi_{0,i}$ denotes the steady-state firing rate of neuron $i$ in the canonical bump, and $\Delta\phi_i$ the distractor-induced rate perturbation.)

This elevated activity "will introduce a drift field according to Eq. (7)" (Seeholzer et al., p. 18), which in the distractor context reads:

$$A(\varphi) = \sum_{i=0}^{N-1} \frac{C_i}{S} \cdot \frac{dJ_{0,i}}{d\varphi} \cdot \Delta\phi_i(\varphi_D) \tag{Eq. 7, Seeholzer et al.}$$

where:
- $\frac{dJ_{0,i}}{d\varphi}$ is the spatial derivative of the steady-state input to neuron $i$ under a shift of the bump center — an antisymmetric function that acts as a **sensitivity kernel** identifying which neurons' activity changes push the bump left vs. right (their Fig. 1C–E)
- $\Delta\phi_i(\varphi_D)$ is the firing rate increase at neuron $i$ caused by a distractor centered at $\varphi_D$
- $C_i/S$ are weighting factors from short-term plasticity (their **Eq. 6** for $C_i$ and **Eq. 18–19** for $S$; see below)

The angular separation $\Delta\varphi = \varphi_D - \varphi$ fully determines the drift magnitude, so we write $A(\Delta\varphi)$ as the **distractor-induced drift field** — a scalar function of the bump–distractor angular offset.

> **Notation note:** In our implementation, we follow the paper's notation and call this function $A(\Delta\varphi)$. Some other works (e.g., Wimmer et al. 2014) use $v_D$ for drift velocity; these are equivalent.

---

#### The normalization constant $S$ and the weighting factors $C_i$

The factor $S$ in the denominator of Eq. (7) is the **proportionality constant of the left eigenvector** $e_l$ of the linearized network Jacobian (Seeholzer et al., **Eq. 18–19**, p. 25–26). It has a clear geometric interpretation: it is the squared norm of the translational mode of the bump,

$$S = \sum_{i} \frac{\left(\frac{dJ_{0,i}}{d\varphi}\right)^2 \cdot [\ldots C_i\text{-terms}\ldots]}{[\ldots]}$$

(see their full Eq. 19 for the STP-dependent expression). In the static case (no short-term plasticity, $U = 1$, $\tau_u = \tau_x \to 0$), the expression simplifies to $S_{\text{static}} = \tau_s \sum_i (d\phi_{0,i}/d\varphi)^2$ (their p. 13). 

Geometrically, $S$ measures **how sharp the bump's flanks are**: a narrow, peaked bump has steep flanks and a large $S$, which means the $C_i/S$ prefactors are small and the drift field amplitude is reduced (the bump resists displacement). A broad, flat bump has a small $S$ and is more susceptible to drift.

The factors $C_i$ capture the **STP-dependent modulation** of each neuron's contribution (their **Eq. 6**):

$$C_i = \frac{U(1 + 2\tau_u\phi_{0,i} + U\tau_u^2\phi_{0,i}^2)}{(1 + U\phi_{0,i}(\tau_u + \tau_x) + U\tau_u\tau_x\phi_{0,i}^2)^2}$$

For our rate model without STP, all $C_i = 1$ and $S$ reduces to the static expression above.

> **Notation note:** In our model we do not implement short-term plasticity. We therefore use the simplified static forms: $C_i = 1$ for all $i$, and $S = \tau_s \sum_i (d\phi_{0,i}/d\varphi)^2$ (where $\tau_s$ is the dominant synaptic time constant). This means Eq. (7) reduces to the simpler projection formula used below.

---

#### Shape of the drift field

For a symmetric Mexican-hat network, the curve $A(\Delta\varphi)$ (Seeholzer et al. Fig. 7A, purple dashed line) has a characteristic profile consistent with their predictions:

- $A(0) = 0$: a distractor directly on top of the bump is symmetric; no net drift
- $A(\Delta\varphi) > 0$ for small $\Delta\varphi > 0$: the bump is attracted toward the distractor
- $A$ peaks near $\Delta\varphi \approx \sigma_{\text{bump}}$ (one bump half-width away)
- $A \to 0$ as $\Delta\varphi \to \pi$: a diametrically opposite distractor has negligible spatial overlap with the bump flanks
- The function is antisymmetric: $A(-\Delta\varphi) = -A(\Delta\varphi)$

The **amplitude** of this curve is the key quantity that differs across nAChR conditions:
- **α7-KO**: reduced PV drive → broader, lower-amplitude bump → smaller $S$ → larger $A(\Delta\varphi)$ amplitude → increased distractor susceptibility. This is directly analogous to Seeholzer et al. Fig. 7D, which shows that broader bumps ($\sigma_g = 0.8$ rad) have a larger radial reach than narrow ones ($\sigma_g = 0.5$ rad).
- **APP**: network hyperactivity → different bump shape and amplitude → shifted drift field (whether this stays within the linear perturbation regime is an open question; see *Assumptions* below).

---

#### Empirical measurement of $A(\Delta\varphi)$ in simulations

Seeholzer et al. (2019, §"Distractor analysis", p. 40) describe the following procedure, which we adapt directly. For each distractor angular distance $\Delta\varphi \in [0°, 180°]$:

1. Run $n_{\text{trials}}$ trials with the bump initialized at $\varphi_0$ and a distractor at $\varphi_0 + \Delta\varphi$, presented for duration $T_D$
2. Measure the bump displacement $\Delta\varphi_{\text{bump}} = \hat{\varphi}_{\text{post}} - \varphi_0$ using the population vector
3. The mean displacement normalized by distractor duration estimates the drift velocity:

$$\hat{A}(\Delta\varphi) \approx \frac{\langle \hat{\varphi}_{\text{post}} - \varphi_0 \rangle}{T_D}$$

This is equivalent to their procedure of generating 1000 trajectories starting from $\varphi_0 = 0$ by integrating Eq. (4) for 250 ms and measuring the mean final position $\varphi_1$ (their Fig. 7B).

The full curve $\hat{A}(\Delta\varphi)$ over $\Delta\varphi \in [0°, 180°]$ is the **distractor susceptibility fingerprint** of the network. Comparing this curve across the four biological conditions (WT, α7-KO, β2-KO, APP) is the primary distractor analysis.

---

#### Analytical shortcut: estimating $S$ from a clean delay trial

Before running any distractor simulations, $S$ (in the static, no-STP limit) can be estimated directly from the firing rate profile of a clean delay trial:

$$S \approx \tau_s \sum_{i=0}^{N-1} \left( \frac{\phi_{0,i+1} - \phi_{0,i-1}}{2\Delta\theta} \right)^2 \cdot \Delta\theta$$

where $\Delta\theta = 2\pi/N$ is the angular spacing between neurons. A lower $S$ predicts a larger $A(\Delta\varphi)$ amplitude, providing a fast diagnostic of distractor susceptibility without running distractor trials.

---

#### Assumptions and validity

Following Seeholzer et al. (2019, p. 22–23), the drift field formula Eq. (7) relies on a **linear perturbation assumption**: the distractor is treated as weak, meaning the bump shape is approximately unchanged during distractor presentation and only the center position shifts. The paper states explicitly: *"we assume the system to remain at approximately steady-state, i.e. that the bump shape is unaffected by the additional external input, except for a shift of the center position"* (p. 23).

This assumption may be violated in the **APP condition**, where network hyperactivity already distorts the baseline bump shape. In that case, Eq. (7) provides an approximation, and the empirical $\hat{A}(\Delta\varphi)$ from simulations should be the primary quantity reported.

---

#### Summary: what to compute and when

| Quantity | Seeholzer et al. ref. | When to compute | What it reveals |
>>>>>>> origin/main
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

<<<<<<< HEAD
=======
## 16. Phase-Dependent Distractor Experiment

CLI command: `ring-osc-phase-distractor`

### 16.1 Purpose

This experiment addresses a specific question left open by `ring-osc-distractor-study`: **does the oscillation phase at the moment of distractor arrival matter?** Two distractors that are identical in space and amplitude but arrive at opposite phases of the ongoing oscillation (e.g., π apart) may have very different effects on the memory trace.

Key questions:
- Does PLV between cue and distractor nodes depend on the oscillation phase at distractor onset?
- Does the cue node lose or maintain its oscillatory power differently depending on whether the distractor arrives at a peak vs. trough of the oscillation?
- Is there a preferred phase (or anti-phase) at which the distractor is maximally disruptive?

### 16.2 Core Design Principle: Identical Pre-Distractor State

Unlike `ring-osc-distractor-study`, which varies the distractor timing freely, this experiment enforces an **identical pre-distractor network state** for all trials at a given phase value:

1. A single burn-in simulation is run with a **fixed seed**.
2. For each target phase `φ` (in π units), a separate *pre-distractor simulation* runs from the same burn-in state with the **same fixed seed**, ending at:
   ```
   delay1(φ) = delay1_base + φ × T_osc / 2
   ```
   Because the seed is fixed, the trajectory is deterministic — all pre-distractor states are snapshots of the **same noise realisation** at different points in time.
3. `n_trials` distractor simulations then branch from each pre-distractor state with **independent noise seeds**, providing stochastic replicates for statistics.

This ensures that within a given phase value, all trials share the exact same initial conditions for the distractor period, making the phase effect interpretable as the sole cause of any observed differences.

### 16.3 Oscillation Frequency Auto-Detection

Before building the phase grid, a reference no-distractor simulation is run to estimate the dominant oscillation frequency `f_osc`:
- PYR rate at the cue node is extracted over the post-cue delay.
- `compute_oscillation_band_timecourse` is applied; the median dominant frequency is taken as `f_osc`.
- If auto-detection fails (no finite frequency in the band), the value from `--osc_freq_hz` (default 5 Hz) is used as fallback.

The oscillation period is `T_osc = 1000 / f_osc` ms. The phase grid then maps:

| `phase_pi` | Phase (radians) | `delay1` |
|------------|-----------------|----------|
| 0          | 0               | `delay1_base` |
| 0.5        | π/2             | `delay1_base + T_osc/4` |
| 1.0        | π               | `delay1_base + T_osc/2` |
| 1.5        | 3π/2            | `delay1_base + 3·T_osc/4` |
| 2.0        | 2π              | `delay1_base + T_osc` (≡ phase 0) |

`n_phase_sweep` (default 16) equally-spaced values are generated in [0, 2π). The four values 0, 0.5, 1.0, 1.5 are always included for the 4-panel figures.

### 16.4 Protocol Timeline

```
[burn-in 10 s] → [pre-cue 0.5 s] → [cue 0.25 s] → [delay1(φ)] → [distractor 0.2 s] → [delay2 2 s]
                ↑_________________________________________________↑
                      identical deterministic trajectory (fixed seed)
```

### 16.5 Measured Quantities

Identical to Experiment 15: STFT dominant power at cue and distractor nodes, and PLV between them.

The key output is the **summary scalar per trial**:
- `plv_mean_delay2`: mean PLV over the post-distractor delay
- `cue_power_mean_delay2`: mean cue node dominant power over the post-distractor delay
- `dist_power_mean_delay2`: mean distractor node dominant power over the post-distractor delay

These scalars are computed per trial and then averaged across the `n_trials` replicates to obtain the phase-tuning curves.

### 16.6 Outputs

Output root: `figs/ring/osc_phase_distractor/{network_label}/{condition_key}/factor{F}/amp{X}/offset{Y}/`

| File | Description |
|------|-------------|
| `osc_phase_trials.csv` | Trial-level table in the experiment root |
| `.osc_phase_cache_{key}.pkl` | Pickle cache |
| **4-panel timecourse grids** | |
| `phase_plv_4panel.png` | 2×2 grid of PLV timecourses for phases 0, π/2, π, 3π/2 |
| `phase_cue_power_4panel.png` | Same for cue node dominant power |
| `phase_dist_power_4panel.png` | Same for distractor node dominant power |
| **Continuous phase sweep** | |
| `phase_sweep.png` | 3-row linear plot: PLV / cue power / dist power (mean ± SEM over delay₂) vs. phase from 0 to 2π |
| `phase_polar.png` | Polar rose version of `phase_sweep.png` (one subplot per metric) |
| **Phase × time heatmaps** | |
| `phase_heatmap_plv.png` | Heatmap: phase (y) × time relative to distractor onset (x), color = mean PLV |
| `phase_heatmap_cue_power.png` | Same for cue node power |
| `phase_heatmap_dist_power.png` | Same for distractor node power |

### 16.7 Caching

Results are pickled to `.osc_phase_cache_{key}.pkl`. The cache key encodes network parameters, conditions, amplitudes, phase grid, timing, and analysis band. Pass `--no_cache` to force re-simulation.

---

>>>>>>> origin/main
## 17. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cerebral Cortex*, 10(9), 910-923.

3. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
