# Ring Attractor Network — Model & Implementation

This document describes the mathematical formulation and implementation of the ring attractor network model used for spatial working memory simulations. The model builds on the 4-population PFC circuit by arranging $N$ identical local circuits on a ring with distance-dependent inter-node connectivity.

For experimental analysis commands see [ring_experiments.md](ring_experiments.md).

---

## Table of Contents

- [Ring Attractor Network — Model \& Implementation](#ring-attractor-network--model--implementation)
  - [Table of Contents](#table-of-contents)
  - [1. Network Architecture](#1-network-architecture)
  - [2. Inter-Node Connectivity](#2-inter-node-connectivity)
    - [2.1 Angular Distance](#21-angular-distance)
    - [2.2 PYR to PYR Excitation](#22-pyr-to-pyr-excitation)
      - [Inter-node excitatory input](#inter-node-excitatory-input)
    - [2.3 PV to PYR Global Inhibition](#23-pv-to-pyr-global-inhibition)
    - [2.4 Connectivity Parameters](#24-connectivity-parameters)
  - [3. Local Circuit Dynamics](#3-local-circuit-dynamics)
    - [3.1 NMDA Gating Variable](#31-nmda-gating-variable)
    - [3.2 Input Current Equations](#32-input-current-equations)
    - [3.3 Weight Notation](#33-weight-notation)
    - [3.4 GABA Scaling](#34-gaba-scaling)
    - [3.5 External Currents](#35-external-currents)
    - [3.6 Transient Current](#36-transient-current)
  - [4. Spike-Frequency Adaptation](#4-spike-frequency-adaptation)
  - [5. Transfer Function](#5-transfer-function)
  - [6. Stimulus \& Distractor Protocol](#6-stimulus--distractor-protocol)
    - [6.1 Spatial Profile](#61-spatial-profile)
    - [6.2 Temporal Profile](#62-temporal-profile)
    - [6.3 Total Stimulus Current](#63-total-stimulus-current)
    - [6.4 Working Memory Protocol](#64-working-memory-protocol)
    - [6.5 Stimulus Parameters](#65-stimulus-parameters)
    - [6.6 Distractor Stimulus](#66-distractor-stimulus)
  - [7. Noise](#7-noise)
    - [Noise equation](#noise-equation)
    - [Noise processes](#noise-processes)
    - [Relation to Seeholzer et al. (2019)](#relation-to-seeholzer-et-al-2019)
  - [8. Experimental Conditions](#8-experimental-conditions)
  - [9. Bump Amplitude Oscillations](#9-bump-amplitude-oscillations)
    - [9.1 Mechanism](#91-mechanism)
    - [9.2 Effect on MSD](#92-effect-on-msd)
    - [9.3 Approaches to Correct or Mitigate the Problem](#93-approaches-to-correct-or-mitigate-the-problem)
    - [9.4 Oscillation Detection](#94-oscillation-detection)
  - [10. Joint Ring + Circuit Optimization](#10-joint-ring--circuit-optimization)
    - [10.1 Motivation](#101-motivation)
    - [10.2 Parameter Space](#102-parameter-space)
    - [10.3 Loss Function](#103-loss-function)
      - [Trace-based Turing bistability loss (optional, `--turing_weight`)](#trace-based-turing-bistability-loss-optional---turing_weight)
      - [Deprecated bump mode](#deprecated-bump-mode)
    - [10.4 Computational Cost](#104-computational-cost)
    - [10.5 Output](#105-output)
  - [11. References](#11-references)

---

## 1. Network Architecture

The network consists of $N$ nodes (default $N = 64$) arranged uniformly on a ring. Each node $i \in \{0, 1, \ldots, N-1\}$ is assigned a preferred angle:

$$\theta_i = \frac{2\pi \, i}{N}, \quad i = 0, \ldots, N-1$$

Each node contains a full local circuit with four neural populations:
- **PYR** (pyramidal, excitatory)
- **PV** (parvalbumin, fast-spiking inhibitory)
- **SOM** (somatostatin, dendritic inhibitory)
- **VIP** (vasoactive intestinal peptide, disinhibitory)

The state of the network at time $t$ is described by the firing rates $r_i^X(t)$ for population $X \in \{\text{PYR}, \text{SOM}, \text{PV}, \text{VIP}\}$ at node $i$, plus adaptation currents $I_{\text{adapt},i}^X(t)$ for PYR and SOM.

**Inter-node connections:**
- PYR $\to$ PYR: distance-dependent excitation (Gaussian profile, see [§2.2](#22-pyr--pyr-excitation))
- PV $\to$ PYR: global inhibition (uniform, see [§2.3](#23-pv--pyr-global-inhibition))
- SOM, VIP: local only (no inter-node connections)

---

## 2. Inter-Node Connectivity

### Row-sum normalisation principle

All three inter-node kernels are row-sum normalised to the corresponding **single-node fitted scalar** from `CircuitParams`. This guarantees that a homogeneous ring (all nodes identical) reproduces the single-node fixed point exactly — the ring model is a consistent extension of the single-node fit, not a separate parameterisation.

| Kernel | Row-sum | Derived from |
|--------|---------|--------------|
| PYR→PYR | $J_{\text{NMDA}}$ | `local_params.J_NMDA` |
| PV→PYR | $w_{pe}$ | `local_params.w_pe` |
| SOM→PYR | $w_{se}$ | `local_params.w_se` |

There are **no additional free parameters** for connection strengths. The only structural free parameters are the Gaussian widths $\sigma_{\text{pyr}}$ and $\sigma_{\text{som}}$.

### 2.1 Angular Distance

The angular distance between two nodes $i$ and $j$ on the ring, handling the wraparound:

$$d(\theta_i, \theta_j) = \min\bigl(|\theta_i - \theta_j|,\; 2\pi - |\theta_i - \theta_j|\bigr)$$

This gives $d \in [0, \pi]$.

### 2.2 PYR $\to$ PYR Excitation (unified NMDA kernel)

The PYR→PYR weight matrix is a **unified** Gaussian kernel that includes the diagonal (self-weight). It replaces the previous separation between local NMDA recurrence and inter-node coupling:

1. **Raw profile (including self at distance 0):**
$$\tilde{W}_{ij} = \exp\!\left(-\frac{d(\theta_i, \theta_j)^2}{2\,\sigma_{\text{pyr}}^2}\right), \quad \text{for all } i, j$$

2. **Row-sum normalization to $J_{\text{NMDA}}$:**
$$W_{ij}^{\text{PYR}\to\text{PYR}} = J_{\text{NMDA}} \cdot \frac{\tilde{W}_{ij}}{\sum_{k} \tilde{W}_{ik}}$$

This guarantees $\sum_j W_{ij} = J_{\text{NMDA}}$ for every node $i$.

The NMDA drive at node $i$ uses the **gating variable** $S_j^{\text{NMDA}}$ (not the raw rate) — see §3.1:

$$I_{\text{NMDA},i} = \sum_{j=0}^{N-1} W_{ij}^{\text{PYR}\to\text{PYR}} \cdot S_j^{\text{NMDA}}$$

This term enters the PYR input as $I_{\text{NMDA},i} / \text{denom}$ where the denominator carries the PV divisive inhibition (see §3.2).

### 2.3 PV $\to$ PYR Global Inhibition (divisive)

PV interneurons from **all** nodes (including self) inhibit PYR at each node uniformly:

$$W_{ij}^{\text{PV}\to\text{PYR}} = \frac{w_{pe}}{N}, \quad \text{for all } i, j$$

Row-sum = $w_{pe}$. The total PV drive enters as a **divisive** (shunting) denominator:

$$\text{denom}_i = 1 + g_{\text{GABA}} \cdot \sum_{j=0}^{N-1} W_{ij}^{\text{PV}\to\text{PYR}} \cdot r_j^{\text{PV}}$$

This models perisomatic GABAergic inhibition that reduces input resistance. At the homogeneous fixed point, $\sum_j W_{ij}^{\text{PV}} r_j = w_{pe} \cdot r^{\text{PV}}$, recovering the single-node denominator exactly.

### 2.4 SOM $\to$ PYR Lateral Inhibition

SOM neurons project **laterally** to PYR nodes via a Gaussian kernel with **zero diagonal** (no self-inhibition):

1. **Raw profile (zero at $i = j$):**
$$\tilde{W}_{ij} = \begin{cases} \exp\!\left(-\dfrac{d(\theta_i,\theta_j)^2}{2\,\sigma_{\text{som}}^2}\right) & i \neq j \\ 0 & i = j \end{cases}$$

2. **Row-sum normalization to $w_{se}$:**
$$W_{ij}^{\text{SOM}\to\text{PYR}} = w_{se} \cdot \frac{\tilde{W}_{ij}}{\sum_{k} \tilde{W}_{ik}}$$

The lateral SOM contribution enters as a **subtractive** inhibition:

$$I_{\text{SOM-lat},i} = g_{\text{GABA}} \cdot \sum_{j=0}^{N-1} W_{ij}^{\text{SOM}\to\text{PYR}} \cdot r_j^{\text{SOM}}$$

At the homogeneous fixed point the row-sum is $w_{se}$ and all nodes contribute equally, reproducing the single-node subtractive term $g_{\text{GABA}} \cdot w_{se} \cdot r^{\text{SOM}}$ exactly.

### 2.5 Connectivity Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `n_nodes` | $N$ | 64 | Number of nodes on the ring |
| `sigma_pyr_deg` | $\sigma_{\text{pyr}}$ | 15.0 deg | Gaussian width of unified PYR→PYR kernel |
| `sigma_som_deg` | $\sigma_{\text{som}}$ | 15.0 deg | Gaussian width of lateral SOM→PYR kernel |

Connection strengths (`J_NMDA`, `w_pe`, `w_se`) are taken directly from `CircuitParams` and are **not** free parameters at the ring level.

**Network-size invariance:** All kernels are row-sum normalised, so the total drive per node is independent of $N$. Simulations at $N=32$, $N=64$, $N=128$ produce comparable homogeneous fixed points.

---

## 3. Local Circuit Dynamics

Each node follows rate dynamics governed by:

$$\tau_s \frac{dr_i^X}{dt} = -r_i^X + \Phi^X(I_i^X)$$

where $\tau_s$ is the synaptic time constant, $\Phi^X$ is the transfer function for population $X$, and $I_i^X$ is the total input current. Noise is injected into the input current of all populations, each scaled by its own baseline drive (see [Section 7](#7-noise)).

Firing rates are clamped to $r_i^X \in [0,\, 200]$ Hz at each integration step. The upper bound acts as a safety net against numerical overflow in large networks while remaining well above physiological firing rates.

### 3.1 NMDA Gating Variable

Local recurrent self-excitation in PYR is mediated through NMDA receptors. Instead of an instantaneous weight $w_{ee} \cdot r_i^{\text{PYR}}$, the recurrent drive is proportional to a **saturable NMDA gating variable** $S_i^{\text{NMDA}} \in [0, 1]$ that evolves according to:

$$\tau_{\text{NMDA}} \frac{dS_i^{\text{NMDA}}}{dt} = -S_i^{\text{NMDA}} + (1 - S_i^{\text{NMDA}}) \cdot \gamma_{\text{NMDA}} \cdot r_i^{\text{PYR}}$$

The $(1 - S_i^{\text{NMDA}})$ factor captures **saturation**: at high firing rates, the fraction of open NMDA channels approaches 1 and the gating variable cannot increase further. This is the kinetic model from Wong & Wang (2006).

| Constant | Symbol | Value | Description |
|----------|--------|-------|-------------|
| `GAMMA_NMDA` | $\gamma_{\text{NMDA}}$ | 0.641 | Kinetic opening rate (dimensionless) |
| `TAU_NMDA_MS` | $\tau_{\text{NMDA}}$ | 100 ms | NMDA decay time constant |

These are **fixed physics constants**, not fitted parameters. The fitted parameter is $J_{\text{NMDA}}$ (the synaptic coupling strength).

**Steady state:** At a fixed firing rate $r^*$, the gating variable converges to:

$$S^* = \frac{\gamma_{\text{NMDA}} \cdot \tau_{\text{NMDA}} \cdot r^*}{1 + \gamma_{\text{NMDA}} \cdot \tau_{\text{NMDA}} \cdot r^*}$$

This saturating nonlinearity is what replaces the linear $w_{ee} \cdot r_i^{\text{PYR}}$ used in earlier versions of the model.

**Initialization:** $S_i^{\text{NMDA}}(0)$ is set to the steady-state value $S^*$ computed from the initial PYR firing rates, so the network starts without a NMDA transient.

### 3.2 Input Current Equations

**PYR** receives unified NMDA excitation from all nodes (via $W^{\text{PYR}}$), divisive PV inhibition from all nodes (via $W^{\text{PV}}$), lateral SOM inhibition (via $W^{\text{SOM}}$), adaptation, external drive, stimulus, and noise:

$$I_i^{\text{PYR}} = \frac{\displaystyle\sum_j W_{ij}^{\text{PYR}} S_j^{\text{NMDA}}}{1 + g_{\text{GABA}} \displaystyle\sum_j W_{ij}^{\text{PV}} r_j^{\text{PV}}} - g_{\text{GABA}} \sum_j W_{ij}^{\text{SOM}} r_j^{\text{SOM}} - I_{\text{adapt},i}^{\text{PYR}} + I_{\text{ext}}^{\text{PYR}} + I_{\text{stim},i}(t) + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$$

- **Numerator** $(W^{\text{PYR}} S^{\text{NMDA}})$: unified Gaussian kernel (including self-weight) gates all NMDA drive — local recurrence and inter-node excitation are handled by a single matrix product.
- **Denominator** $(1 + g_{\text{GABA}} W^{\text{PV}} r^{\text{PV}})$: all PV nodes (local + inter-node) enter divisively, modelling perisomatic shunting inhibition.
- **Subtractive SOM** $(g_{\text{GABA}} W^{\text{SOM}} r^{\text{SOM}})$: purely lateral (zero diagonal), models dendritic SOM inhibition.

At the homogeneous fixed point, the three matrix products reduce exactly to the single-node scalars $J_{\text{NMDA}} S^*$, $w_{pe} r^{\text{PV}}$, and $w_{se} r^{\text{SOM}}$.

The noise term $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$ injects stochastic current into each PYR node. Injecting noise at the current level (before the transfer function $\Phi^{\text{PYR}}$) means its effect on firing rate is naturally filtered by the transfer function slope $\Phi'$, consistent with a diffusion-approximation interpretation of Poisson spiking variability. The proportionality to $I_{\text{ext}}^{\text{PYR}}$ ensures noise scales automatically across experimental conditions.

**SOM** (local connections only):

$$I_i^{\text{SOM}} = w_{es} \, r_i^{\text{PYR}} - w_{vs} \, r_i^{\text{VIP}} - I_{\text{adapt},i}^{\text{SOM}} + I_{\text{ext}}^{\text{SOM}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{SOM}} \cdot \xi_i(t)$$

**PV** (local connections only; PV's global effect is on PYR, not on other PV):

$$I_i^{\text{PV}} = w_{ep} \, r_i^{\text{PYR}} - g_{\text{GABA}} \, w_{pp} \, r_i^{\text{PV}} - g_{\text{GABA}} \, w_{sp} \, r_i^{\text{SOM}} - w_{vp} \, r_i^{\text{VIP}} + I_{\text{ext}}^{\text{PV}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PV}} \cdot \xi_i(t)$$

**VIP** (local connections only):

$$I_i^{\text{VIP}} = w_{ev} \, r_i^{\text{PYR}} + I_{\text{ext}}^{\text{VIP}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{VIP}} \cdot \xi_i(t)$$

### 3.3 Weight Notation

Weights follow the convention $w_{XY}$ = connection **from** population $Y$ **to** population $X$:
- e = PYR (excitatory), p = PV, s = SOM, v = VIP
- Example: $w_{ep}$ = weight from PYR to PV
- $J_{\text{NMDA}}$ is the recurrent PYR→PYR NMDA coupling strength (replaces the former $w_{ee}$)

### 3.4 GABA Scaling

Inhibitory weights are multiplied by a GABA scaling factor:

$$g_{\text{GABA}} = g_{\text{GABA}}^{\text{base}} + \text{act}_{\alpha 7} \cdot g_{\alpha 7}$$

where $\text{act}_{\alpha 7} \in [0, 1]$ is the alpha7 nAChR activation level (0 under knockout).

### 3.5 External Currents

Each population receives baseline tonic drive plus receptor-mediated currents:

$$I_{\text{ext}}^{\text{PYR}} = I_0^{\text{PYR}}$$

$$I_{\text{ext}}^{\text{PV}} = I_0^{\text{PV}} + \text{act}_{\alpha 7} \cdot I_{\alpha 7}^{\text{PV}}$$

$$I_{\text{ext}}^{\text{SOM}} = I_0^{\text{SOM}} + \text{act}_{\alpha 7} \cdot I_{\alpha 7}^{\text{SOM}} + \text{act}_{\beta 2} \cdot I_{\beta 2}^{\text{SOM}}$$

$$I_{\text{ext}}^{\text{VIP}} = I_0^{\text{VIP}} + \text{act}_{\alpha 5} \cdot I_{\alpha 5}^{\text{VIP}}$$

### 3.6 Transient Current

An optional nonspecific transient current can be applied to all populations simultaneously during a window $[t_{\text{start}}, t_{\text{start}} + \Delta t_{\text{trans}})$:

$$I_{\text{ext}}^{X}(t) = I_{\text{ext}}^{X} + \begin{cases} f_{\text{trans}} \cdot I_0^X & \text{if } t_{\text{start}} \leq t < t_{\text{start}} + \Delta t_{\text{trans}} \\ 0 & \text{otherwise} \end{cases}$$

where $f_{\text{trans}}$ is the transient factor (fraction of baseline I0).

---

## 4. Spike-Frequency Adaptation

PYR exhibits spike-frequency adaptation, which provides slow negative feedback:

$$\tau_{\text{adapt}}^{\text{PYR}} \frac{dI_{\text{adapt},i}^{\text{PYR}}}{dt} = -I_{\text{adapt},i}^{\text{PYR}} + J_{\text{adapt}}^{\text{PYR}} \cdot r_i^{\text{PYR}}$$

SOM adaptation is present in the code (`J_adapt_som`) but **disabled by default during optimization** (`J_adapt_som = 0` in the bistable fit; freeze with `--freeze J_adapt_som`). The thesis model does not include SOM adaptation.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tau_adapt_pyr` | $\tau_{\text{adapt}}^{\text{PYR}}$ | 600 ms | PYR adaptation time constant |
| `J_adapt_pyr` | $J_{\text{adapt}}^{\text{PYR}}$ | 0.27 | PYR adaptation strength |
| `J_adapt_som` | $J_{\text{adapt}}^{\text{SOM}}$ | 0.0 | SOM adaptation strength (off by default) |

Adaptation prevents runaway excitation and creates temporal dynamics in the bump.

---

## 5. Transfer Function

The model uses the **Wong-Wang transfer function** (Wong & Wang, 2006):

$$\Phi(I) = \frac{u}{1 - e^{-g \, u}}, \quad \text{where } u = \alpha \cdot (I - \theta)$$

**Interneuron soft ceiling.** To prevent pathological runaway firing, PV, SOM, and VIP use a **hyperbolic soft ceiling** applied post-hoc:

$$\Phi_{\text{cap}}(I) = \frac{r_{\max} \cdot \Phi(I)}{r_{\max} + \Phi(I)}$$

This asymptotes to $r_{\max}$ as $\Phi \to \infty$, while leaving low-rate behavior ($\Phi \ll r_{\max}$) unchanged. PYR uses the uncapped $\Phi$.

| Population | Transfer function | $r_{\max}$ (Hz) |
|------------|------------------|----------------|
| PYR | $\Phi(I)$ (uncapped) | — |
| PV | $\Phi_{\text{cap}}(I)$ | 53 |
| SOM | $\Phi_{\text{cap}}(I)$ | 53 |
| VIP | $\Phi_{\text{cap}}(I)$ | 103 |

The ceilings are 1.5× the Rooy (2021) high-state targets for each interneuron type.

**Parameters per population $X$:**

| Parameter | Symbol | Population | Value |
|-----------|--------|------------|-------|
| `Theta_pyr` | $\theta^{pyr}$ | PYR | $125/310 \approx 0.403\ \text{nA}$ |
| `Theta_pv` / `Theta_som` / `Theta_vip` | $\theta^{inh}$ | PV, SOM, VIP | $177/615 \approx 0.288\ \text{nA}$ |
| `alpha_pyr` | $\alpha^{pyr}$ | PYR | $310\ \text{Hz/nA}$ |
| `alpha_pv` / `alpha_som` / `alpha_vip` | $\alpha^{inh}$ | PV, SOM, VIP | $615\ \text{Hz/nA}$ |
| `g_exc` | $g_e$ | PYR | $0.16\ \text{s}$ |
| `g_inh` | $g_i$ | PV, SOM, VIP | $0.087\ \text{s}$ |

**Numerical stability:** For $|g \cdot u| < \epsilon$ (near zero), a Taylor expansion is used: $\Phi \approx 1/g + u/2$.

---

## 6. Stimulus & Distractor Protocol

### 6.1 Spatial Profile

The stimulus is a current injection to PYR neurons with a Gaussian spatial profile centered at angle $\theta_{\text{stim}}$:

$$S_{\text{spatial}}(\theta_i) = \exp\!\left(-\frac{d(\theta_i, \theta_{\text{stim}})^2}{2\,\sigma_{\text{stim}}^2}\right)$$

where $\sigma_{\text{stim}}$ is the stimulus spatial width (default: $18\degree$ from Yang et Liu, 2023).

### 6.2 Temporal Profile

The stimulus is active as a square pulse during the window $[t_{\text{on}}, t_{\text{off}})$ where $t_{\text{off}} = t_{\text{on}} + \Delta t_{\text{stim}}$:

$$M(t) = \begin{cases} 1 & \text{if } t_{\text{on}} \leq t < t_{\text{off}} \\ 0 & \text{otherwise} \end{cases}$$

### 6.3 Total Stimulus Current

The stimulus current at node $i$ and time $t$ is:

$$I_{\text{stim},i}(t) = \begin{cases} A \cdot S_{\text{spatial}}(\theta_i) \cdot M(t) & \text{if } t_{\text{on}} \leq t < t_{\text{off}} \\ 0 & \text{otherwise} \end{cases}$$

where $A$ is the peak amplitude.

### 6.4 Working Memory Protocol

The standard working memory protocol consists of:

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Network settles to baseline state (no stimulus) |
| Pre-cue baseline | 500 ms | Continued baseline after burn-in |
| Cue presentation | 250 ms | Stimulus at $\theta_{\text{stim}} = 180\degree$, amplitude $A$ |
| Delay period | 3,000 ms (default) | No stimulus; memory retention |
| (Optional) Response transient | configurable | Nonspecific current boost to all populations |

### 6.5 Stimulus Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `center_deg` | $\theta_{\text{stim}}$ | 180 deg | Stimulus angular location |
| `amplitude` | $A$ | 150.0 | Peak current amplitude |
| `sigma_deg` | $\sigma_{\text{stim}}$ | 18.0 deg | Spatial width (Gaussian sigma) |
| `onset_ms` | $t_{\text{on}}$ | 10500 ms | Stimulus onset time |
| `duration_ms` | $\Delta t_{\text{stim}}$ | 250 ms | Stimulus duration |

---

### 6.6 Distractor Stimulus

An optional distractor stimulus can be presented during the delay period using the `WorkingMemoryProtocol` class. The distractor is a second `RingStimulus` with its own angular location, amplitude, timing, and spatial width:

$$I_{\text{distract},i}(t) = A_{\text{dist}} \cdot \exp\!\left(-\frac{d(\theta_i, \theta_{\text{dist}})^2}{2\,\sigma_{\text{stim}}^2}\right) \cdot M_{\text{dist}}(t)$$

Multiple stimuli are summed: $I_{\text{stim},i}(t) = \sum_k I_{\text{stim},i}^{(k)}(t)$

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distractor_location_deg` | `None` | Distractor angle (None = no distractor) |
| `distractor_amplitude` | 3.0 | Distractor current amplitude |
| `distractor_onset_ms` | 1500.0 | Distractor onset (absolute time, ms) |
| `distractor_duration_ms` | 200.0 | Distractor duration (ms) |


---

---

## 7. Noise

Noise is injected as a shared stochastic current perturbation into **all four populations** (PYR, SOM, PV, VIP) at each node independently. All populations share the same noise process $\xi_i(t)$ at each node, but each population's noise amplitude is proportional to its own baseline external drive. This ensures correlated variability across populations while keeping the relative noise level consistent for each population. This models the variability in synaptic drive (diffusion approximation of Poisson spike trains).

### Noise equation

The noisy input current for each population $X \in \{\text{PYR, SOM, PV, VIP}\}$ at node $i$ is:

$$I_i^{X}(t) = I_i^{X,\text{det}}(t) + \underbrace{\sigma_{\text{noise}} \cdot I_{\text{ext}}^{X}}_{\text{noise scale (nA)}} \cdot \xi_i(t)$$

where $I_i^{X,\text{det}}$ is the deterministic part (all synaptic, adaptation, and stimulus terms), $\sigma_{\text{noise}}$ is the dimensionless noise amplitude, and $\xi_i(t)$ is the shared noise process (see below). Each population's noise scale is proportional to its own baseline drive $I_{\text{ext}}^{X}$, so the dimensionless noise amplitude $\sigma_{\text{noise}}$ has the same meaning across all populations.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `sigma_noise` | $\sigma_{\text{noise}}$ | `0.3` | Dimensionless noise amplitude. Noise current std for population $X$ = `sigma_noise × I_ext_X` (nA) |

### Noise processes

Two stochastic processes are available for $\xi_i(t)$:

**White noise** (default): i.i.d. Gaussian samples at each integration step
$$\xi_i(t) \sim \mathcal{N}(0, 1) \quad \text{independent across nodes and time steps}$$

**Ornstein-Uhlenbeck (OU) noise**: temporally correlated noise with time constant $\tau_{\text{noise}}$
$$d\xi_i = -\frac{\xi_i}{\tau_{\text{noise}}} \, dt + \sqrt{\frac{2}{\tau_{\text{noise}}}} \, dW_i$$

Discretized (Euler-Maruyama):
$$\xi_i(t + \Delta t) = \xi_i(t) - \frac{\xi_i(t)}{\tau_{\text{noise}}} \Delta t + \sqrt{\frac{2\,\Delta t}{\tau_{\text{noise}}}} \, \eta_i, \quad \eta_i \sim \mathcal{N}(0, 1)$$

**No noise**: $\xi_i(t) = 0$ (set `sigma_noise = 0`).

---

### Relation to Seeholzer et al. (2019)

The Langevin equation (their Eq. 4) contains a single white noise term $\sqrt{B}\,\eta(t)$ with
$\langle\eta(t)\,\eta(t')\rangle = \delta(t - t')$. This is **not** a noise injected into the network
directly; it is the *emergent* effective noise on the bump center $\varphi(t)$, obtained by projecting
the full $N$-dimensional spiking variability onto the 1D attractor manifold via the left eigenvector
$e_l$ (their §"Diffusion", p. 24).

The underlying noise source in the paper is **independent white Gaussian noise per neuron** —
a diffusion approximation to Poisson spike emission:

$$\xi_i(t) \approx \phi_{0,i} + \sqrt{\phi_{0,i}}\,\eta_i(t), \qquad
\langle\eta_i(t)\,\eta_j(t')\rangle = \delta(t-t')\,\delta_{ij}$$

(their p. 24, lines immediately before Eq. 20). Our model injects current-space noise into each population $X$ with amplitude $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{X}$, which after passing through the transfer function slope $\Phi'$ produces effective rate noise consistent with this formulation. The $\sqrt{\phi_{0,i}}$ amplitude scaling of the original is absorbed into $\sigma_{\text{noise}}$.

**OU noise** introduces temporal correlations with timescale $\tau_{\text{noise}}$ and falls outside
the Seeholzer et al. derivation, which requires white (delta-correlated) noise for the diffusion
coefficient $B$ in Eq. (5) to hold. OU noise is instead motivated as a model of slowly fluctuating
background input from other cortical areas, independent of the bump attractor theory.

**In practice:** white noise is the appropriate mode for comparing empirical diffusion coefficients
$\hat{B}$ to the theoretical prediction of Eq. (5). OU noise produces trial-to-trial variability
with a characteristic timescale but will not match the $\langle[\varphi(t)-\varphi(0)]^2\rangle = B \cdot t$
scaling predicted by the theory.

## 8. Experimental Conditions

The model can simulate 8 conditions by combining two fitted parameter families (WT and WT_APP) with knockout toggles. APP is represented by using the WT_APP fitted parameters, not by sampling receptor desensitization multipliers.

| Condition | Parameter family | $\text{act}_{\alpha 7}$ | $\text{act}_{\beta 2}$ | $\text{act}_{\alpha 5}$ | $g_{\alpha 7}$ |
|-----------|------------------|:---:|:---:|:---:|:---:|
| **WT** | WT | 1.0 | 1.0 | 1.0 | default |
| **WT_APP** | WT_APP | 1.0 | 1.0 | 1.0 | default |
| **$\alpha 7$ KO** | WT | 0.0 | 1.0 | 1.0 | 0.0 |
| **$\alpha 7$ KO_APP** | WT_APP | 0.0 | 1.0 | 1.0 | 0.0 |
| **$\beta 2$ KO** | WT | 1.0 | 0.0 | 1.0 | default |
| **$\beta 2$ KO_APP** | WT_APP | 1.0 | 0.0 | 1.0 | default |
| **$\alpha 5$ KO** | WT | 1.0 | 1.0 | 0.0 | default |
| **$\alpha 5$ KO_APP** | WT_APP | 1.0 | 1.0 | 0.0 | default |

---

## 9. Bump Amplitude Oscillations

### 9.1 Mechanism

After the stimulus offset, the bump amplitude does not settle immediately to a steady value. Instead, it undergoes **damped oscillations** driven by the slow negative feedback of spike-frequency adaptation (SFA). The sequence is:

1. Stimulus drives strong activation → bump forms, amplitude rises.
2. Adaptation builds up during the stimulus → suppresses activity slightly after offset.
3. When the stimulus turns off, adaptation current is still elevated → amplitude undershoots.
4. Adaptation decays → amplitude recovers, overshoots.
5. This bounce repeats with exponentially decaying amplitude until the attractor settles.

In practice (default parameters, 128 nodes), the dominant oscillation frequency is around **~9–10 Hz (period ≈ 100 ms)**. The oscillations are visible in the trial-averaged amplitude and are systematic (not noise-driven): they represent a genuine resonance of the bump attractor.

### 9.2 Effect on MSD

The bump position $\varphi(t)$ is estimated as the phase of the population vector. When the amplitude oscillates, the effective signal-to-noise on the phase estimate also oscillates. Moreover, the position itself may experience small correlated displacements at the oscillation frequency.

The oscillation adds a periodic term to the theoretical MSD:

$$\text{MSD}(\tau) \approx B\,\tau + C\left(1 - \cos\!\left(\frac{2\pi\tau}{T_\text{osc}}\right)\right) + \text{offset}$$

where $B$ is the true diffusion coefficient, $C$ is the oscillation contribution, and $T_\text{osc} = 1/f_\text{osc}$. Fitting a pure line $B\tau$ in the early regime ($\tau < T_\text{osc}$) **overestimates $\hat{B}$** because the oscillation increases apparent displacement at short lags.

### 9.3 Approaches to Correct or Mitigate the Problem

Five strategies are available, from simplest to most principled:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Exclude early transient** | Start MSD fit range after $N$ oscillation periods (e.g. `fit_range_s[0] = 3 × T_osc`) | Simple, no pre-processing | Wastes early data; requires knowing $T_\text{osc}$ |
| **B. Low-pass filter position** ✓ | Apply zero-phase Butterworth LP filter to $\varphi(t)$ at $f_\text{cut} < f_\text{osc}$ before MSD | Clean, preserves slow drift, intuitive | Introduces slight edge effects; needs $f_\text{cut}$ choice |
| **C. Oscillation-corrected fit** ✓ | Fit $\text{MSD} = B\tau + C(1-\cos(2\pi f\tau))$ with $f$ fixed from FFT | Separates diffusion and oscillation rigorously | Requires prior knowledge of $f_\text{osc}$; 3-param fit |
| **D. Time-windowed averaging** | Replace instantaneous $\varphi$ with running mean over 1 cycle | Simple, no filter needed | Introduces temporal smearing of genuine drift |
| **E. Fit only long lags** | Restrict fit to $\tau \gg T_\text{osc}$ where cosine term averages out | No preprocessing | Greatly reduces usable lag range; noisier fit |

**Current implementation**: strategies **B** (low-pass filter) and **C** (oscillation-corrected fit) are applied automatically when a dominant oscillation is detected by FFT of the per-trial amplitude. The filter cutoff defaults to $0.4 \times f_\text{osc}$ and can be overridden with `--filter_cutoff_hz` (set to `0` to disable).

### 9.4 Oscillation Detection

The `compute_oscillation_spectrum` function (in `analysis.py`) computes the power spectrum of the bump amplitude for each trial, averages across trials, and identifies the dominant frequency as the peak exceeding **3× the median power** in the band $[1, 50]$ Hz. The result is reported in `diffusion_oscillation.csv` and visualised in `diffusion_oscillation_spectrum.png`.

---

## 10. Joint Ring + Circuit Optimization

The `ring-optimize` command jointly fits `CircuitParams` and `RingParams` so the ring network at rest reproduces target firing rates while enforcing bump-supporting bistability constraints.

### 10.1 Motivation

Rate matching alone is necessary but not sufficient for working-memory behavior: a parameter set can match quiet-wakefulness means and still fail to sustain a bump after cue offset. The current optimizer therefore combines:

1. rate + KO + Jacobian terms,
2. an optional **trace-based Turing bistability loss** computed from a deterministic cue simulation,
3. optional additional regularizers (spatial uniformity, ACh ratio).

### 10.2 Parameter Space

`ring-optimize` searches over `CircuitParams` together with the ring structural parameters:

| Parameter | Symbol | Default bounds | Description |
|-----------|--------|---------------|-------------|
| `sigma_pyr_deg` | $\sigma_\text{pyr}$ | [5°, 40°] | Gaussian width of unified PYR→PYR kernel |
| `sigma_som_deg` | $\sigma_\text{som}$ | [5°, 40°] | Gaussian width of lateral SOM→PYR kernel |

Connection strengths ($J_\text{NMDA}$, $w_{pe}$, $w_{se}$) are taken from the fitted `CircuitParams` and are **not** additional free parameters. `n_nodes` is fixed by CLI and not optimized.

### 10.3 Loss Function

Base objective:

$$\mathcal{L}_\text{base} = \mathcal{L}_\text{rate} + \frac{1}{N_\text{KO}}\sum_k \mathcal{L}_k^\text{KO} + \mathcal{L}_\text{Jacobian}$$

where

$$\mathcal{L}_\text{rate} = \frac{1}{4} \sum_{X \in \{\text{PYR},\text{SOM},\text{PV},\text{VIP}\}} \left(\frac{\bar r^X - r^X_\text{target}}{r^X_\text{target}}\right)^2$$

and

$$\bar r^X = \frac{1}{N}\sum_{i=1}^N \langle r_i^X \rangle_{\text{window}}.$$

KO conditions (alpha7, alpha5, beta2) are evaluated on single-node by default, or on ring with `--ko_on_ring`.

#### Trace-based Turing bistability loss (optional, `--turing_weight`)

The current implementation is simulation-based (not analytical). For each candidate:

1. Run one deterministic cue simulation (`noise_type="none"`) with cue parameters:
   - `--turing_cue_amplitude`
   - `--turing_cue_duration_ms`
   - `--turing_cue_sigma_deg`
2. Reconstruct gain traces from recorded rates/adaptation and inter-node currents.
3. Score rest-vs-delay constraints with margin `--turing_margin`:
   - rest gain below threshold,
   - late-delay bump-node rate band (`--turing_bump_min_hz`, `--turing_bump_max_hz`),
   - late-delay sustain floor on gain,
   - anti-runaway ceilings (gain/background activity).

Top bump-support nodes are chosen from late-delay PYR activity using `--turing_topk_nodes`; late-delay window size is `--turing_late_delay_ms`.

The combined objective is:

$$\mathcal{L} = \mathcal{L}_\text{base} + w_T \cdot \mathcal{L}_\text{Turing,trace} + w_U \cdot \mathcal{L}_\text{uniformity} + w_A \cdot \mathcal{L}_\text{ACh}.$$

#### Deprecated bump mode

`--bump_mode` is deprecated and ignored. Bump constraints are integrated in the trace-based Turing term.

### 10.4 Computational Cost

Ring optimization remains expensive. Practical defaults:

- `n_nodes = 64` during optimization,
- moderate `n_trials_ring` for stochastic averaging,
- `ko_on_ring = False` unless strict consistency is required,
- nonzero `turing_weight` adds one deterministic cue simulation per evaluation.

### 10.5 Output

```
ring_optim_output/
├── best_circuit_params.json   # Best CircuitParams (same format as optimize)
└── best_ring_params.json      # Best RingParams as JSON
```

These can be passed directly to `ring-run` or `ring-study` via `--params_json` and optional `--sigma_pyr_deg / --sigma_som_deg` overrides.

See [CLI.md — ring-optimize](CLI.md#ring-optimize) for the full, up-to-date argument reference.

---

## 11. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
