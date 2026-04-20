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
    - [2.2 PYR $\\to$ PYR Excitation](#22-pyr-to-pyr-excitation)
      - [Inter-node excitatory input](#inter-node-excitatory-input)
    - [2.3 PV $\\to$ PYR Global Inhibition](#23-pv-to-pyr-global-inhibition)
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

### 2.1 Angular Distance

The angular distance between two nodes $i$ and $j$ on the ring, handling the wraparound:

$$d(\theta_i, \theta_j) = \min\bigl(|\theta_i - \theta_j|,\; 2\pi - |\theta_i - \theta_j|\bigr)$$

This gives $d \in [0, \pi]$.

### 2.2 PYR $\to$ PYR Excitation

The raw Gaussian weights are computed and then **row-sum normalized** to ensure the total coupling strength is independent of $N$:

1. **Raw profile:**
$$\tilde{W}_{ij} = \begin{cases} \exp\!\left(-\dfrac{d(\theta_i, \theta_j)^2}{2\,\sigma_{\text{pyr}}^2}\right) & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

2. **Row-sum normalization:**
$$W_{ij}^{\text{PYR}\to\text{PYR}} = w_{\text{pyr}}^{\text{inter}} \cdot \frac{\tilde{W}_{ij}}{\sum_{k} \tilde{W}_{ik}}$$

This guarantees $\sum_j W_{ij} = w_{\text{pyr}}^{\text{inter}}$ for every node $i$, so the total excitatory drive is invariant to network size $N$.

#### Inter-node excitatory input

For both profiles, the inter-node excitatory input to PYR at node $i$ is:

$$I_{\text{inter},i}^{\text{PYR}} = \sum_{j=0}^{N-1} W_{ij}^{\text{PYR}\to\text{PYR}} \cdot r_j^{\text{PYR}}$$

Self-connections ($i = j$) are always zero because local PYR $\to$ PYR recurrence is handled by the within-node NMDA gating term $J_{\text{NMDA}} \cdot S_i^{\text{NMDA}}$.

### 2.3 PV $\to$ PYR Global Inhibition

PV interneurons from all nodes inhibit PYR at each node, creating an excitation-inhibition loop: local PYR excites local PV (via $w_{ep}$), then PV activity is broadcast globally to suppress PYR everywhere. This mechanism provides the competitive inhibition needed for bump formation.

All PV-to-PYR connections have equal weight (excluding self):

$$W_{ij}^{\text{PV}\to\text{PYR}} = \begin{cases} \displaystyle\frac{w_{\text{PV}}^{\text{global}}}{N-1} & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

The inter-node inhibitory input from PV to PYR at node $i$ is:

$$I_{\text{inter},i}^{\text{PV}\to\text{PYR}} = \sum_{j=0}^{N-1} W_{ij}^{\text{PV}\to\text{PYR}} \cdot r_j^{\text{PV}}$$

### 2.4 Connectivity Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `n_nodes` | $N$ | 64 | Number of nodes on the ring |
| `w_pyr_pyr_inter` | $w_{\text{pyr}}^{\text{inter}}$ | 18.55 | Total row-sum of PYR→PYR weights |
| `sigma_pyr_deg` | $\sigma_{\text{pyr}}$ | 30.0 deg | Gaussian width of PYR→PYR profile |
| `w_pv_global` | $w_{\text{PV}}^{\text{global}}$ | 0.3 | Strength of PV→PYR global inhibition |

**Note on network-size invariance:** The Gaussian profile (row-sum normalization) ensures that the effective coupling is independent of $N$, so simulations at $N=64$, $N=128$, and $N=1024$ produce comparable dynamics.

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

**PYR** receives local recurrent NMDA excitation (with divisive PV inhibition), inter-node excitation, inter-node PV inhibition, SOM subtractive inhibition, adaptation, external drive, stimulus, and noise:

$$I_i^{\text{PYR}} = \frac{J_{\text{NMDA}} \cdot S_i^{\text{NMDA}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}} + I_{\text{inter},i}^{\text{PYR}} - g_{\text{GABA}} \, I_{\text{inter},i}^{\text{PV}\to\text{PYR}} - g_{\text{GABA}} \, w_{se} \, r_i^{\text{SOM}} - I_{\text{adapt},i}^{\text{PYR}} + I_{\text{ext}}^{\text{PYR}} + I_{\text{stim},i}(t) + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$$

The term $\frac{J_{\text{NMDA}} \cdot S_i^{\text{NMDA}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}}$ combines two effects:
- **NMDA saturation** (via $S_i^{\text{NMDA}}$): recurrent drive saturates at high firing rates instead of growing unboundedly.
- **Divisive (shunting) PV inhibition** (denominator): models the effect of perisomatic GABAergic synapses on input resistance.

The noise term $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$ injects stochastic current into each PYR node. Injecting noise at the current level (before the transfer function $\Phi^{\text{PYR}}$) means its effect on firing rate is naturally filtered by the transfer function slope $\Phi'$, consistent with a diffusion-approximation interpretation of Poisson spiking variability. The proportionality to $I_{\text{ext}}^{\text{PYR}}$ ensures noise scales automatically across experimental conditions.

**SOM** (local connections only):

$$I_i^{\text{SOM}} = w_{es} \, r_i^{\text{PYR}} - w_{vs} \, r_i^{\text{VIP}} - I_{\text{adapt},i}^{\text{SOM}} + I_{\text{ext}}^{\text{SOM}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{SOM}} \cdot \xi_i(t)$$

**PV** (local connections only; PV's global effect is on PYR, not on other PV):

$$I_i^{\text{PV}} = w_{ep} \, r_i^{\text{PYR}} - g_{\text{GABA}} \, w_{pp} \, r_i^{\text{PV}} - g_{\text{GABA}} \, w_{sp} \, r_i^{\text{SOM}} - w_{vp} \, r_i^{\text{VIP}} + I_{\text{ext}}^{\text{PV}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PV}} \cdot \xi_i(t)$$

**VIP** (local connections only):

$$I_i^{\text{VIP}} = w_{ev} \, r_i^{\text{PYR}} + I_{\text{ext}}^{\text{VIP}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{VIP}} \cdot \xi_i(t)$$

### 3.3 Weight Notation

Weights follow the convention $w_{XY}$ = connection **from** population $X$ **to** population $Y$:
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

PYR and SOM populations exhibit spike-frequency adaptation, which provides slow negative feedback:

$$\tau_{\text{adapt}}^X \frac{dI_{\text{adapt},i}^X}{dt} = -I_{\text{adapt},i}^X + J_{\text{adapt}}^X \cdot r_i^X$$

for $X \in \{\text{PYR}, \text{SOM}\}$.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tau_adapt_pyr` | $\tau_{\text{adapt}}^{\text{PYR}}$ | 600 ms | PYR adaptation time constant |
| `tau_adapt_som` | $\tau_{\text{adapt}}^{\text{SOM}}$ | 150 ms | SOM adaptation time constant (slow) |
| `J_adapt_pyr` | $J_{\text{adapt}}^{\text{PYR}}$ | 0.27 | PYR adaptation strength |
| `J_adapt_som` | $J_{\text{adapt}}^{\text{SOM}}$ | 27.24 | SOM adaptation strength (strong) |

Adaptation prevents runaway excitation and creates temporal dynamics in the bump.

---

## 5. Transfer Function

The model uses the **Wong-Wang transfer function** (Wong & Wang, 2006), derived from a mean-field reduction of spiking neural networks:

$$\Phi(I) = \frac{u}{1 - e^{-g \, u}}, \quad \text{where } u = c \cdot (I - \theta)$$

**Parameters per population $X$:**

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `Theta_X` | $\theta^X$ | Threshold current |
| `alpha_X` | $c^X$ | Gain / slope parameter |
| `g_e` / `g_i` | $g$ | Curvature parameter ($g_e$ for PYR, $g_i$ for PV/SOM/VIP) |

**Properties:**
- Monotonically increasing
- Bounded below at 0 (firing rates are non-negative)
- Approximately linear near threshold
- Saturates at high inputs
- Reduces to ReLU-like behavior as $g \to \infty$

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

`ring-optimize` searches over `CircuitParams` together with ring-specific parameters:

| Parameter | Symbol | Default bounds | Description |
|-----------|--------|---------------|-------------|
| `w_pyr_pyr_inter` | $w_\text{pyr}^\text{inter}$ | [1, 30] | Total inter-node PYR→PYR coupling (row-sum normalized) |
| `w_pv_global` | $w_\text{PV}^\text{global}$ | [0.5, 20] | Total global PV→PYR inhibition (uniform all-to-all) |
| `sigma_pyr_deg` | $\sigma_\text{pyr}$ | [10°, 60°] | Gaussian width of PYR→PYR profile |

`n_nodes` is fixed by CLI and not optimized.

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

These can be passed directly to `ring-run` or `ring-study` via `--params_json` and a manual `--w_pyr_pyr_inter / --w_pv_global / --sigma_pyr_deg` override.

See [CLI.md — ring-optimize](CLI.md#ring-optimize) for the full, up-to-date argument reference.

---

## 11. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
