# Ring Attractor Network — Experiments & Analysis

This document describes the experimental analysis commands and protocols for the ring attractor network. For the model architecture and implementation see [ring_attractor.md](ring_attractor.md).

---

## Table of Contents

10. [Analysis Methods](#10-analysis-methods)
    - [10.1 Population Vector Decoding](#101-population-vector-decoding)
    - [10.2 Distractor-Induced Drift Field Analysis](#102-distractor-induced-drift-field-analysis)
    - [10.2b 2D Distractor Sweep](#102b-2d-distractor-sweep)
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
13. [Drift Field Analysis](#13-drift-field-analysis)
    - [13.1 Purpose](#131-purpose)
    - [13.2 Simulation Protocol](#132-simulation-protocol)
    - [13.3 Bump Position Measurement](#133-bump-position-measurement)
    - [13.4 Drift Velocity Computation](#134-drift-velocity-computation)
    - [13.5 Outputs](#135-outputs)
    - [13.6 Caching](#136-caching)
14. [2D Distractor Sweep](#14-2d-distractor-sweep)
    - [14.1 Purpose](#141-purpose)
    - [14.2 Simulation Protocol](#142-simulation-protocol)
    - [14.3 Bump Position Measurement](#143-bump-position-measurement)
    - [14.4 Collapse Detection](#144-collapse-detection)
    - [14.5 Aggregation](#145-aggregation)
    - [14.6 Outputs](#146-outputs)
    - [14.7 Caching](#147-caching)
15. [Lesion Study](#15-lesion-study)
    - [15.1 Purpose](#151-purpose)
    - [15.2 Protocol](#152-protocol)
    - [15.3 Knockdown Mechanism](#153-knockdown-mechanism)
    - [15.4 Analysis](#154-analysis)
    - [15.5 Outputs](#155-outputs)
16. [τ_adapt Sweep](#16-τ_adapt-sweep)
    - [16.1 Purpose](#161-purpose)
    - [16.2 Protocol](#162-protocol)
    - [16.3 Analysis](#163-analysis)
    - [16.4 Outputs](#164-outputs)
17. [Phase Plane Analysis](#17-phase-plane-analysis)
    - [17.1 Purpose](#171-purpose)
    - [17.2 Single-Node Decoupled Model](#172-single-node-decoupled-model)
    - [17.3 Hysteresis Sweep](#173-hysteresis-sweep)
    - [17.4 Outputs](#174-outputs)
18. [Temporal Dissection](#18-temporal-dissection)
    - [18.1 Purpose](#181-purpose)
    - [18.2 Protocol](#182-protocol)
    - [18.3 Recorded Quantities](#183-recorded-quantities)
    - [18.4 Outputs](#184-outputs)
19. [Bump Asymmetry Analysis](#19-bump-asymmetry-analysis)
    - [19.1 Purpose](#191-purpose)
    - [19.2 Asymmetry Index](#192-asymmetry-index)
    - [19.3 Simulation Protocol](#193-simulation-protocol)
    - [19.4 Measured Quantities](#194-measured-quantities)
    - [19.5 Statistical Tests](#195-statistical-tests)
    - [19.6 Outputs](#196-outputs)
    - [19.7 Caching](#197-caching)
20. [References](#20-references)

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
|---|---|---|---|
| $\hat{\varphi}$, $\hat{A}$ (pop. vector) | — | All timepoints | Bump position and integrity |
| $S$ (normalization constant) | Eqs. 18–19 (static: Eq. 19 simplified) | Once per condition, clean delay | Analytical predictor of distractor susceptibility |
| $A(\Delta\varphi)$ curve (theoretical) | Eq. 7 | From bump profile, no simulation | Predicted drift field per condition |
| $\hat{A}(\Delta\varphi)$ curve (empirical) | Fig. 7B; §"Distractor analysis" p. 40 | Distractor trials, vary $\Delta\varphi$ | Measured susceptibility fingerprint |
| Bump collapse probability | — | Distractor trials | Fraction of trials where $\hat{A}$ drops below threshold |

### 10.2b 2D Distractor Sweep

While Section 10.2 estimates the drift field $\hat{A}(\Delta\varphi)$ at a **fixed** distractor amplitude across conditions, the `ring-distractor-sweep` command provides a complementary analysis: it holds the condition fixed (typically WT) and jointly varies both $\Delta\varphi$ and distractor amplitude $\alpha$ to map out a full **2D susceptibility landscape**.

#### Measured quantities

For each $(\Delta\varphi, \alpha)$ cell and $K$ trials:

| Quantity | Definition | Output column |
|----------|-----------|---------------|
| Mean bump shift | $\langle \Delta\hat{\theta} \rangle$ (degrees) | `drift_mean_deg` |
| Shift SEM | $\text{SEM}(\Delta\hat{\theta})$ | `drift_sem_deg` |
| Collapse probability | $P(\hat{A}_{\text{after}} < \tau)$ | `collapse_prob` |
| Pre-distractor $\hat{A}$ | Mean pop-vector amplitude before | `pre_amp_mean` |
| Post-distractor $\hat{A}$ | Mean pop-vector amplitude after | `post_amp_mean` |

Collapse threshold $\tau$ is auto-detected from `calibration_summary.csv` (matching on condition, cue amplitude, `w_inter`). Falls back to 0.2 with a warning if no calibration data is found. Can be overridden with `--collapse_threshold`.

#### Output figures

**Drift heatmap** (`distractor_sweep_drift.png`)
— Mean $\Delta\hat{\theta}$ on a diverging (RdBu_r) colormap symmetric about 0. Positive = bump pulled toward distractor; negative = repelled. Reveals the angular range and amplitude window where the distractor exerts significant attractive force.

**Collapse probability heatmap** (`distractor_sweep_collapse.png`)
— Fraction of trials with $\hat{A}_{\text{after}} < \tau$ on a sequential (YlOrRd) colormap. Separates the regime where the distractor merely shifts the bump (drift without collapse) from the regime where it destroys working memory entirely.

**Bump trajectories** (`distractor_sweep_timecourses.png`)
— Full $\hat{\theta}(t)$ decoded at every timestep for 6 representative $(\Delta\varphi, \alpha)$ cells. Shaded grey region = cue window; shaded orange = distractor window. Useful to distinguish: (i) gradual drift during distractor, (ii) snap to new location, (iii) collapse and recovery.

#### Relationship to drift-field analysis

The $\Delta\varphi$-sweep at a fixed $\alpha$ in `ring-distractor-sweep` is equivalent to one row of the `ring-drift-field` output, except that `ring-drift-field` normalises by distractor duration $T_D$ to yield $\hat{A}$ in rad/s, while `ring-distractor-sweep` reports the raw displacement in degrees. The heatmap view adds the amplitude dimension and collapse probability, making it more informative for hypothesis generation about which distractor regime the network operates in.

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

The calibration also provides the **noise floor threshold** $\tau$ that is consumed downstream by `ring-distractor-sweep` and `ring-diffusion` to decide whether a bump has collapsed.

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

## 13. Drift Field Analysis

### 13.1 Purpose

The `ring-drift-field` command implements the empirical measurement of the **distractor-induced drift field** $\hat{A}(\Delta\varphi)$ described in [§10.2](#102-distractor-induced-drift-field-analysis). It sweeps the angular offset $\Delta\varphi$ of a distractor relative to the cue and measures the resulting bump displacement, normalized by distractor duration to give a drift velocity in rad/s. This is repeated across all experimental conditions, producing one curve $\hat{A}(\Delta\varphi)$ per condition that serves as the distractor susceptibility fingerprint.

Before interpreting the results, it is important to asses that the network dynamics does shift the bump center in response to the distractor, but does not significantly distort the bump shape (i.e. the linear perturbation assumption holds). This can be verified by plotting the full $\hat{\theta}(t)$ trajectory during the distractor window for a few representative trials.
(It isn't the case in our network so far)

---

### 13.2 Simulation Protocol

Each trial follows a two-phase sequence. Unlike the clean delay protocol ([§6.4](ring_attractor.md#64-working-memory-protocol)), a **distractor stimulus** is injected during the delay:

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Network at baseline (pre-computed, cached) |
| Pre-cue baseline | 500 ms | Continued baseline |
| Cue | 250 ms | Gaussian stimulus at $\theta_{\text{stim}} = 180°$, amplitude $A$ |
| Delay | `distractor_onset_ms` (default 1,500 ms) | Bump forms and stabilizes |
| Distractor | `distractor_duration_ms` (default 200 ms) | Competing stimulus at $180° + \Delta\varphi$, amplitude $A_{\text{dist}}$ |
| Post-distractor buffer | 500 ms | Recovery window; measurement taken here |

The **distractor onset** (in absolute simulation time) is:

$$t_{\text{dist}} = t_{\text{stim,off}} + \Delta t_{\text{delay}} = 10750\,\text{ms} + 1500\,\text{ms} = 12250\,\text{ms}$$

**Sweep dimensions**: distractor angular offsets $\Delta\varphi \in [0°, 180°]$ in steps of `--distractor_steps` (default 10°), giving 19 offset values. For each offset, `--n_trials` independent trials (default 50) are run per condition, parallelised across workers.

---

### 13.3 Bump Position Measurement

The bump center is decoded via population vector at two timepoints:

| Measurement | Timing | Description |
|-------------|--------|-------------|
| **Pre-distractor** $\hat{\theta}_{\text{pre}}$ | $t_{\text{dist}} - 50\,\text{ms}$ | Bump position 50 ms before distractor onset |
| **Post-distractor** $\hat{\theta}_{\text{post}}$ | $t_{\text{dist}} + T_D + 500\,\text{ms}$ | Bump position 500 ms after distractor offset |

The signed displacement is:

$$\Delta\hat{\theta} = \hat{\theta}_{\text{post}} - \hat{\theta}_{\text{pre}}$$

wrapped to $(-\pi, \pi]$. Positive $\Delta\hat{\theta}$ means the bump drifted **toward** the distractor.

---

### 13.4 Drift Velocity Computation

The per-trial displacement is normalized by distractor duration $T_D$ to yield a **drift velocity** estimate:

$$\hat{A}(\Delta\varphi) \approx \frac{\langle \Delta\hat{\theta} \rangle}{T_D}$$

expressed in rad/s. Across $N_{\text{trials}}$ trials per offset:

| Statistic | Formula |
|-----------|---------|
| Mean drift velocity | $\hat{A}(\Delta\varphi) = \bar{\Delta\hat{\theta}} / T_D$ |
| Standard deviation | $\sigma_A = \text{std}(\Delta\hat{\theta},\, \text{ddof}=1) / T_D$ |
| Standard error | $\text{SEM}_A = \sigma_A / \sqrt{N_{\text{trials}}}$ |

This procedure is equivalent to Seeholzer et al. (2019, §"Distractor analysis", p. 40): running many trajectories from $\varphi_0$ with a distractor at $\varphi_0 + \Delta\varphi$ and measuring the mean final position.

---

### 13.5 Outputs

All files are written to `figs/drift_field/<n_nodes>/<params_stem>/<conn_label>/`.

**Data files**:

| File | Content |
|------|---------|
| `drift_field_trials.csv` | Per-trial: `condition_key`, `offset_deg`, `trial_idx`, `seed`, `displacement_rad`, `pre_amp`, `post_amp` |
| `drift_field_summary.csv` | Per (condition, offset): `condition_key`, `offset_deg`, `A_hat_rad_per_s`, `A_hat_sem`, `A_hat_sd`, `n_trials`, `distractor_amplitude_factor`, `distractor_duration_ms`, `distractor_onset_ms` |

**Figures**:

| Figure | Description |
|--------|-------------|
| `drift_field_sem.png` / `drift_field_sd.png` | $\hat{A}(\Delta\varphi)$ vs offset in degrees, one colored line per condition with mean ± SEM or SD shading. Positive values indicate bump attraction toward distractor. |

---

### 13.6 Caching

If both CSV files exist and every requested (condition, offset) pair already has $\geq N_{\text{trials}}$ rows in `drift_field_trials.csv`, and the distractor parameters (`amplitude_factor`, `duration_ms`, `onset_ms`) match, the simulations are skipped. Use `--no_cache` to force a full rerun.

---

## 14. 2D Distractor Sweep

### 14.1 Purpose

The `ring-distractor-sweep` command extends the drift field analysis by jointly varying both the distractor **angular offset** $\Delta\varphi$ and its **amplitude** relative to the cue, for a single experimental condition. The result is a 2D landscape that simultaneously reveals (i) how much the bump shifts and (ii) where in the ($\Delta\varphi$, amplitude) space the bump collapses entirely. See [§10.2b](#102b-2d-distractor-sweep) for the theoretical framing.

---

### 14.2 Simulation Protocol

The protocol uses an explicit two-delay structure that cleanly separates bump stabilization, distractor presentation, and post-distractor recovery:

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Baseline equilibrium (cached) |
| Pre-cue baseline | 500 ms | Continued baseline |
| Cue | 250 ms | Gaussian stimulus at $\theta_{\text{stim}} = 180°$, amplitude $A_{\text{cue}}$ |
| Delay₁ | `--delay1_ms` (default 1,000 ms) | Bump forms and stabilizes |
| Distractor | `--distractor_duration_ms` (default 250 ms) | Competing stimulus at $180° + \Delta\varphi$, amplitude $\alpha \cdot A_{\text{cue}}$ |
| Delay₂ | `--delay2_ms` (default 1,000 ms) | Post-distractor recovery |

**Distractor onset** (absolute):

$$t_{\text{dist}} = 10500 + 250 + \Delta t_{\text{delay1}} = 11750\,\text{ms}\,\text{(default)}$$

**Sweep grid** (both dimensions are configurable):

| Dimension | Default values |
|-----------|---------------|
| Angular offset $\Delta\varphi$ | $\{0°, 5°, 10°, 15°, 20°, 30°, 40°, 60°, 80°, 100°, 130°, 150°, 180°\}$ |
| Amplitude factor $\alpha$ | $\{0.5\times, 0.75\times, 1.0\times, 1.25\times, 1.5\times\}$ |

For each grid cell $(\Delta\varphi, \alpha)$, `--n_trials` independent trials (default 10) are run, totalling $13 \times 5 \times 10 = 650$ simulations by default. Only a single condition is swept at a time.

---

### 14.3 Bump Position Measurement

The same measurement windows as the drift field command are used:

| Measurement | Timing | Description |
|-------------|--------|-------------|
| **Pre-distractor** $\hat{\theta}_{\text{pre}}$ | $t_{\text{dist}} - 50\,\text{ms}$ | Bump position 50 ms before distractor onset |
| **Post-distractor** $\hat{\theta}_{\text{post}}$ | $t_{\text{dist,off}} + 500\,\text{ms}$ | Bump position 500 ms after distractor offset |

The signed bump shift in degrees:

$$\Delta\hat{\theta} = (\hat{\theta}_{\text{post}} - \hat{\theta}_{\text{pre}} + \pi) \bmod 2\pi - \pi$$

---

### 14.4 Collapse Detection

A bump is declared **collapsed** in a given trial if the post-distractor decoding confidence falls below the noise floor threshold $\tau$:

$$\text{collapsed}_k = \mathbf{1}\!\left[\hat{A}_k^{\text{post}} < \tau\right]$$

The threshold $\tau$ is resolved in this order:

1. **`--collapse_threshold`** flag (manual override).
2. **Calibration file** — reads `calibration_summary.csv` ([§11.5](#115-outputs)) for the matching condition, cue amplitude, and $w_{\text{inter}}$, and uses its `noise_threshold` column.
3. **Fallback** — $\tau = 0.2$ with a warning if no calibration data is found.

---

### 14.5 Aggregation

For each grid cell $(\Delta\varphi, \alpha)$ across $K$ trials:

| Statistic | Formula | Output column |
|-----------|---------|---------------|
| Mean bump shift | $\bar{\Delta} = \frac{1}{K}\sum_k \Delta\hat{\theta}_k$ | `drift_mean_deg` |
| Shift SD | $\text{SD}(\Delta\hat{\theta})$ | `drift_sd_deg` |
| Shift SEM | $\text{SD} / \sqrt{K}$ | `drift_sem_deg` |
| Collapse probability | $P_{\text{collapse}} = \frac{1}{K}\sum_k \mathbf{1}[\hat{A}_k^{\text{post}} < \tau]$ | `collapse_prob` |
| Pre-distractor $\bar{\hat{A}}$ | Mean pop-vector amplitude before distractor | `pre_amp_mean` |
| Post-distractor $\bar{\hat{A}}$ | Mean pop-vector amplitude after distractor | `post_amp_mean` |

---

### 14.6 Outputs

All files are written to `figs/distractor_sweep/<n_nodes>/<params_stem>/<conn_label>/`.

**Data files**:

| File | Content |
|------|---------|
| `distractor_sweep_trials.csv` | Per-trial: `offset_deg`, `amp_factor`, `trial_idx`, `displacement_deg`, `pre_amp`, `post_amp` |
| `distractor_sweep_summary.csv` | Per grid cell: `condition_key`, `offset_deg`, `amp_factor`, `n_trials`, `drift_mean_deg`, `drift_sd_deg`, `drift_sem_deg`, `collapse_prob`, `pre_amp_mean`, `post_amp_mean`, `distractor_duration_ms`, `delay1_ms`, `delay2_ms`, `cue_amp_factor` |

**Figures**:

| Figure | Description |
|--------|-------------|
| `distractor_sweep_drift.png` | 2D heatmap of mean bump shift (degrees). Axes: $\Delta\varphi$ (x) × distractor amplitude factor (y). Diverging colormap (RdBu_r), symmetric about 0. Positive = bump attracted toward distractor. |
| `distractor_sweep_collapse.png` | 2D heatmap of collapse probability $[0, 1]$. Sequential colormap (YlOrRd). Cells annotated with percentage. Separates the regime where the distractor merely shifts the bump from the regime where it destroys working memory. |
| `distractor_sweep_timecourses.png` | Full $\hat{\theta}(t)$ decoded at every timestep for 6 representative $(\Delta\varphi, \alpha)$ cells. Grey shading = cue window; orange shading = distractor window. Distinguishes gradual drift, snap-to-new-location, and collapse. |
| `activity_grid.png` | PYR activity rasters (node angle × time) for all non-zero offsets at a single amplitude (~0.75× cue), showing the spatial pattern of distractor competition. |

---

### 14.7 Caching

If both CSV files exist and all requested $(\Delta\varphi, \alpha)$ cells have $\geq N_{\text{trials}}$ matching rows, and the protocol parameters (condition, `distractor_duration_ms`, `delay1_ms`, `delay2_ms`, `cue_amp_factor`) match, the simulations are skipped. Use `--no_cache` to force a full rerun.

---

## 15. Lesion Study

### 15.1 Purpose

The `ring-lesion` command identifies the **functional contribution** of each population to bump formation and maintenance. By scaling down the synaptic weight of one population at a time while keeping all others intact, it dissociates:

- **Formation** — whether a bump forms at all after the cue (measured at 50 ms post-cue-offset)
- **Maintenance** — how long the bump survives before collapsing (bump survival time)

The result is a 4×2 panel figure across the four populations and five knockdown levels.

---

### 15.2 Protocol

Standard working memory protocol ([§6.4](ring_attractor.md#64-working-memory-protocol)) is used. For each population × knockdown level combination, `n_trials` independent trials are run with different noise seeds. The delay is fixed at `--delay_ms` (default 2000 ms).

**Knockdown sweep**: [0%, 25%, 50%, 75%, 100%] (configurable via `--knockdown_levels`).

---

### 15.3 Knockdown Mechanism

Each knockdown percentage $\kappa$ scales the relevant weight multiplicatively by $(1 - \kappa/100)$:

| Population | Parameter scaled | Effect |
|------------|-----------------|--------|
| `PYR_recurrence` | `w_pyr_pyr_inter` | Weakens inter-node PYR→PYR excitation; reduces bump persistence |
| `PV` | `w_pv_global` | Weakens global lateral inhibition; bump less sharply tuned |
| `SOM` | `w_se` | Weakens SOM→PYR subtractive inhibition; alters adaptation feedback |
| `VIP` | `w_vs` | Weakens VIP→SOM disinhibition; alters SOM gating |

---

### 15.4 Analysis

**Formation rate**: fraction of trials where bump amplitude $\hat{A} > $ `noise_floor` at 300 ms after cue offset.

**Bump survival time**: first moment during the delay where $\hat{A}$ drops below `noise_floor` and **stays below** for ≥100 ms. If the bump survives the full delay, survival time is reported as `delay_ms`. Trials that never formed (formation failure) are excluded from survival statistics.

---

### 15.5 Outputs

Files in `figs/lesion/<n_nodes>/default/<conn_label>/`:

| File | Content |
|------|---------|
| `lesion_results.csv` | Per-trial: `population`, `knockdown_pct`, `trial_idx`, `seed`, `formation_ok`, `survival_time_ms` |
| `lesion_figure.png` | 4×2 panel figure: formation rate (left) and survival time ± SEM (right) vs knockdown %, one row per population |

---

## 16. τ_adapt Sweep

### 16.1 Purpose

The `ring-tau-sweep` command probes the role of **adaptation timescale** (`tau_adapt_pyr`) in shaping three aspects of bump dynamics:

1. **Bump survival time** — does slower adaptation destabilize the bump?
2. **Diffusion** — does τ_adapt affect how fast the bump position wanders?
3. **Oscillation frequency** — the dominant frequency of bump amplitude oscillations ([§9](ring_attractor.md#9-bump-amplitude-oscillations)) scales inversely with τ_adapt

By sweeping τ_adapt over [50, 100, 200, 400, 600, 1000, 2000] ms while keeping `J_adapt_pyr` fixed, the experiment isolates timescale from strength effects.

---

### 16.2 Protocol

Standard working memory protocol ([§6.4](ring_attractor.md#64-working-memory-protocol)). For each τ_adapt value, `n_trials` independent trials are run. The delay defaults to `--delay_ms` (default 3000 ms, use `--delay_ms 2000` for a standard run).

---

### 16.3 Analysis

For each τ value:

1. **Survival time**: per trial, `compute_bump_survival_time` → aggregate mean ± SEM (same definition as [§15.4](#154-analysis))
2. **Diffusion coefficient**: extract delay bump-center trajectory → `compute_msd_curve` → `fit_diffusion_coefficient` → D in deg²/s
3. **Oscillation frequency**: FFT of per-trial bump amplitude → `compute_oscillation_spectrum` → dominant frequency in Hz (NaN if no peak detected)

---

### 16.4 Outputs

Files in `figs/tau_sweep/<n_nodes>/default/<conn_label>/`:

| File | Content |
|------|---------|
| `tau_sweep_results.csv` | Per-trial: `tau_ms`, `trial_idx`, `seed`, `survival_time_ms` |
| `tau_sweep_figure.png` | 3-panel figure (shared log x-axis): survival time, diffusion D, oscillation frequency vs τ_adapt |

---

## 17. Phase Plane Analysis

### 17.1 Purpose

The `ring-phase-plane` command performs a **bifurcation analysis** of the local circuit by decoupling a single node from the ring and sweeping the external PYR drive $I_0^{\text{PYR}} + \Delta I$ over a range. The UP/DOWN hysteresis sweep reveals whether the circuit has a **bistable** operating regime — a coexistence of low-activity (spontaneous) and high-activity (persistent) fixed points — and shows how this changes across conditions (WT, α7-KO, β2-KO, WT_APP).

---

### 17.2 Single-Node Decoupled Model

All inter-node connectivity is zeroed (`w_pyr_pyr_inter = 0`, `w_pv_global = 0`), leaving only the **local circuit** dynamics ([§3](ring_attractor.md#3-local-circuit-dynamics)). This isolates the circuit's intrinsic input-output relationship from network-level amplification.

The additive offset $\Delta I$ is applied exclusively to the PYR external current:

$$I_{\text{ext}}^{\text{PYR}} \to I_0^{\text{PYR}} + \Delta I$$

All other populations (SOM, PV, VIP) receive their standard $I_0$ values.

---

### 17.3 Hysteresis Sweep

The sweep uses **two passes**:

| Pass | Direction | Initial state | Purpose |
|------|-----------|--------------|---------|
| UP sweep | $\Delta I_{\min} \to \Delta I_{\max}$ | All rates = 0 | Finds the lower branch (spontaneous → active transition) |
| DOWN sweep | $\Delta I_{\max} \to \Delta I_{\min}$ | All rates = 0 | Finds the upper branch (active → spontaneous transition) |

State (rates + adaptation currents) is **carried forward** between consecutive steps, so hysteresis is captured. At each step the network is integrated for `step_ms` ms; the firing rate is averaged over the last `settle_ms` ms.

**Bistability detection**: a step is declared bistable when $|r_{\text{PYR}}^{\text{UP}} - r_{\text{PYR}}^{\text{DOWN}}| > $ `bistable_threshold`.

**Operating point markers** (dashed vertical lines on plots):

| Marker | $\Delta I$ value | Interpretation |
|--------|-----------------|----------------|
| Spontaneous/delay | 0 | No external cue; background drive only |
| Cue | $\approx A_{\text{factor}} \times I_0^{\text{PYR}}$ | Peak stimulus drive during cue |

---

### 17.4 Outputs

Files in `figs/phase_plane/default/<conn_label>/`:

| File | Content |
|------|---------|
| `<condition>_phase_plane.csv` | Per step: `delta_I`, `up_PYR`, `up_SOM`, `up_PV`, `up_VIP`, `down_PYR`, `down_SOM`, `down_PV`, `down_VIP`, `bistable` |
| `phase_plane_grid.png` | 4-condition × 4-population S-curve grid. Blue = UP sweep, red dashed = DOWN sweep, grey shading = bistable region, vertical dashed lines = operating points |

---

## 18. Temporal Dissection

### 18.1 Purpose

The `ring-temporal-dissection` command provides a mechanistic, **location-resolved** view of a single working memory trial by recording full time courses at three nodes simultaneously: the stimulus center, its quadrature (+90°), and the antipodal node (+180°). Unlike `ring-run` (which shows a whole-network heatmap), this command shows **how each population's firing rate and adaptation current evolve differentially** across the ring — directly linking network dynamics to the local circuit equations ([§3](ring_attractor.md#3-local-circuit-dynamics)).

---

### 18.2 Protocol

A single deterministic trial (`noise_type='none'`, `sigma_s=0`) is run with `record_adaptation=True` so that the PYR adaptation current $I_{\text{adapt}}^{\text{PYR}}(t)$ is stored at each timestep alongside the firing rates. This removes stochastic variability and exposes the underlying attractor dynamics.

The three tracked nodes are:

| Panel column | Node index | Angle |
|-------------|-----------|-------|
| Center (0°) | `stim_node` | Stimulus location |
| +90° | `(stim_node + N/4) % N` | Quadrature |
| Antipodal (+180°) | `(stim_node + N/2) % N` | Opposite side of ring |

---

### 18.3 Recorded Quantities

| Row | Quantity | Unit |
|-----|---------|------|
| 0 | PYR firing rate $r^{\text{PYR}}$ | Hz |
| 1 | SOM firing rate $r^{\text{SOM}}$ | Hz |
| 2 | PV firing rate $r^{\text{PV}}$ | Hz |
| 3 | VIP firing rate $r^{\text{VIP}}$ | Hz |
| 4 | PYR adaptation current $I_{\text{adapt}}^{\text{PYR}}$ | a.u. |

Vertical markers: red dashed = cue onset, red solid = cue offset, blue dotted = collapse time (only if the bump collapses during the delay).

---

### 18.4 Outputs

Files in `figs/temporal_dissection/<n_nodes>/default/<conn_label>/`:

| File | Content |
|------|---------|
| `temporal_dissection.png` | 5×3 panel figure (rows: PYR, SOM, PV, VIP, I_adapt_PYR; columns: center, +90°, antipodal) |

---

## 19. Bump Asymmetry Analysis

### 19.1 Purpose

The bump asymmetry experiment tests whether the activity bump drifts **systematically** to one side of the cue location during the delay period. In an ideal, perfectly symmetric ring attractor the expected asymmetry is zero; any consistent left or right bias would reveal a structural symmetry-breaking caused by the nAChR condition (WT, APP, KO variants). Comparing the asymmetry distribution across conditions thus probes whether APP or receptor knockout introduces a directional bias on top of the stochastic diffusion already quantified in [§12](#12-bump-drift-analysis).

### 19.2 Asymmetry Index

For each time step the activity pattern at the cue location is split into a **left half** (nodes with negative signed angular offset from the cue) and a **right half** (positive offset):

$$\text{offset}_i = \bigl((\theta_i - \theta_{\text{cue}} + 180°) \bmod 360°\bigr) - 180°$$

$$A(t) = \frac{\sum_{i:\,\text{offset}_i > 0} r_i^{\text{PYR}}(t) \;-\; \sum_{i:\,\text{offset}_i < 0} r_i^{\text{PYR}}(t)}{\sum_{i:\,\text{offset}_i \neq 0} r_i^{\text{PYR}}(t)}$$

By default (`--correct_asymmetry` on), the reported asymmetry is amplitude-weighted at each time step:

$$A_{\text{corr}}(t) = A(t) \times \text{Amp}(t)$$

where $\text{Amp}(t)$ is the decoded bump amplitude. This reduces bias from condition-dependent bump strength (weak bumps are more noise-sensitive). Use `--no_correct_asymmetry` to recover the raw index $A(t)$.

The index $A(t) \in [-1, +1]$:

| Value | Interpretation |
|-------|----------------|
| $+1$ | all activity on the right (clockwise) side of the cue |
| $0$ | perfectly symmetric |
| $-1$ | all activity on the left (counter-clockwise) side |

When the bump is perfectly centred on the cue the numerator is zero, so $A = 0$. Any persistent non-zero value during the delay indicates a systematic bias.

### 19.3 Simulation Protocol

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

### 19.4 Measured Quantities

For each trial two scalar values are extracted from either $A_{\text{corr}}(t)$ (default) or $A(t)$ (with `--no_correct_asymmetry`):

**Pre-cue asymmetry** — mean of $A(t)$ over the last 500 ms of the burn-in period (before cue onset). This serves as a per-trial baseline: in the absence of any stimulus the ring should be approximately symmetric, so any pre-existing asymmetry in the spontaneous state can be tracked and correlated with the delay outcome.

**Delay asymmetry** — mean of $A(t)$ from cue offset + 400 ms to the end of the delay. The 400 ms skip discards the large SOM/PYR transient that follows stimulus offset (see [§9.1](ring_attractor.md#91-mechanism)).

$$a_{\text{pre}} = \frac{1}{|\mathcal{T}_{\text{pre}}|}\sum_{t \in \mathcal{T}_{\text{pre}}} A(t), \qquad \mathcal{T}_{\text{pre}} = [t_{\text{cue}} - 500\,\text{ms},\; t_{\text{cue}})$$

$$a_{\text{delay}} = \frac{1}{|\mathcal{T}_{\text{delay}}|}\sum_{t \in \mathcal{T}_{\text{delay}}} A(t), \qquad \mathcal{T}_{\text{delay}} = [t_{\text{off}} + 400\,\text{ms},\; T]$$

### 19.5 Statistical Tests

After collecting $n_{\text{trials}}$ values of $a_{\text{delay}}$ per condition, two one-sample tests against zero are performed:

**One-sample t-test** (`scipy.stats.ttest_1samp`):

$$t = \frac{\bar{a}_{\text{delay}}}{s / \sqrt{n}}, \qquad H_0: \mu = 0$$

**Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`, two-sided, applied when $n \geq 10$): a non-parametric alternative that makes no normality assumption, appropriate given that the distribution of delay asymmetry values can be non-Gaussian and may be affected by bump collapses.

Results are printed in a table with significance markers ($* p < 0.05$, $** p < 0.01$, $*** p < 0.001$) and stored in `stats_by_condition` for annotation of summary figures.

The p-value displayed on plots uses the Wilcoxon result when available (more robust), falling back to the t-test p-value otherwise.

### 19.6 Outputs

All outputs are written to:

```
figs/asymmetry/{N}/{params_label}/{connectivity_label}/amp{amp}_{mode}/
```

| File | Description |
|------|-------------|
| *(path mode)* | `{mode} = corrected` when `--correct_asymmetry` is enabled (default), else `uncorrected` |
| `asymmetry_trials.csv` | Per-trial data: `condition`, `trial_idx`, `seed`, `cue_deg`, `pre_cue_asym`, `delay_asym`, `delay_ms`, `amplitude`, `random_cue` (0/1), `balance_cue` (0/1), `correct_asymmetry` (0/1) |
| `asymmetry_distribution.png` | Violin + jittered strip plots of pre-cue and delay asymmetry per condition. Each point is coloured by asymmetry value (blue → right, yellow → left). The delay violin is annotated with the p-value and significance stars from [§19.5](#195-statistical-tests). Plot title indicates cue mode and whether asymmetry is corrected. |
| `asymmetry_correlation.png` | Scatter plot of pre-cue vs. delay asymmetry per condition, with Pearson $r$ annotated. |
| `asymmetry_summary.png` | Three-panel bar chart: (1) mean delay asymmetry ± SEM with significance stars, (2) fraction of rightward trials, (3) mean absolute asymmetry ± SEM. |
| `worst_case/{cond}/dashboard.png` | Ring attractor dashboard for the trial with the largest $|a_{\text{delay}}|$ per condition. |
| `worst_case/{cond}/bump_metrics.png` | Bump metrics over time (including asymmetry panel) for the worst-case trial. |
| `worst_case/{cond}/snapshot_evolution.mp4` | Animated ring snapshot evolution for the worst-case trial. |

### 19.7 Caching

The trial CSV is used as a cache on subsequent runs. Before launching simulations the command reads `asymmetry_trials.csv` and validates:
- `delay_ms` matches the current `--delay_ms` argument
- `amplitude` matches the current `--amplitude` argument
- `random_cue` (0/1) matches the current `--random_cue_location` flag
- `balance_cue` (0/1) matches whether `--no_cue_balance` is absent (1) or present (0)
- `correct_asymmetry` (0/1) matches whether `--correct_asymmetry` is enabled (1) or disabled (0)

If validation passes, already-computed trial indices are loaded and only the **missing trials** are simulated (top-up logic). This allows incremental runs: e.g. running with `--n_trials 50` after a previous `--n_trials 20` will run only 30 additional trials and merge them into the CSV. If any parameter does not match, all trials are rerun from scratch and the CSV is overwritten.

CSVs missing `correct_asymmetry` are treated as legacy caches: mode is inferred from the folder suffix (`amp*_uncorrected` → raw, `amp*_corrected` → corrected; no suffix defaults to raw). CSVs that predate `cue_deg`/`random_cue`/`balance_cue` load `cue_deg` defaulting to `STIM_CENTER_DEG` and skip flag validation (defaulting `balance_cue=1`).

---

## 20. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cerebral Cortex*, 10(9), 910-923.

3. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
