# Ring Attractor Network for Working Memory

This document describes the mathematical formulation and implementation of the ring attractor network model used for spatial working memory simulations. The model builds on the 4-population PFC circuit by arranging $N$ identical local circuits on a ring with distance-dependent inter-node connectivity.

---

## Table of Contents

1. [Network Architecture](#1-network-architecture)
2. [Inter-Node Connectivity](#2-inter-node-connectivity)
3. [Local Circuit Dynamics](#3-local-circuit-dynamics)
4. [Spike-Frequency Adaptation](#4-spike-frequency-adaptation)
5. [Transfer Function](#5-transfer-function)
6. [Stimulus Protocol](#6-stimulus-protocol)
7. [Distractor Mechanism](#7-distractor-mechanism)
8. [Noise](#8-noise)
9. [Experimental Conditions](#9-experimental-conditions)
10. [Analysis Methods](#10-analysis-methods)
    - [10.1 Population Vector Decoding](#101-population-vector-decoding)
    - [10.2 Distractor-Induced Drift Field Analysis](#102-distractor-induced-drift-field-analysis)
    - [10.2b 2D Distractor Sweep](#102b-2d-distractor-sweep)
    - [10.3–10.8 Other metrics](#103-bump-width-estimation)
11. [Bump Amplitude Oscillations](#11-bump-amplitude-oscillations)
12. [References](#12-references)

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
- PYR $\to$ PYR: distance-dependent excitation (Gaussian or Compte profile, see §2.2)
- PV $\to$ PYR: global inhibition (uniform or distance-dependent, see §2.3)
- SOM, VIP: local only (no inter-node connections)

---

## 2. Inter-Node Connectivity

### 2.1 Angular Distance

The angular distance between two nodes $i$ and $j$ on the ring, handling the wraparound:

$$d(\theta_i, \theta_j) = \min\bigl(|\theta_i - \theta_j|,\; 2\pi - |\theta_i - \theta_j|\bigr)$$

This gives $d \in [0, \pi]$.

### 2.2 PYR $\to$ PYR Excitation

Two connectivity profiles are available, selectable via the `pyr_profile_type` parameter.

#### Gaussian Profile (default)

The raw Gaussian weights are computed and then **row-sum normalized** to ensure the total coupling strength is independent of $N$:

1. **Raw profile:**
$$\tilde{W}_{ij} = \begin{cases} \exp\!\left(-\dfrac{d(\theta_i, \theta_j)^2}{2\,\sigma_{\text{pyr}}^2}\right) & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

2. **Row-sum normalization:**
$$W_{ij}^{\text{PYR}\to\text{PYR}} = w_{\text{pyr}}^{\text{inter}} \cdot \frac{\tilde{W}_{ij}}{\sum_{k} \tilde{W}_{ik}}$$

This guarantees $\sum_j W_{ij} = w_{\text{pyr}}^{\text{inter}}$ for every node $i$, so the total excitatory drive is invariant to network size $N$.

#### Compte et al. (2000) Profile

An alternative profile following Compte et al. (2000) combines local excitation with surround inhibition ("Mexican hat"):

1. **Gaussian envelope** (with zero diagonal):
$$G_{ij} = \begin{cases} \exp\!\left(-\dfrac{d(\theta_i, \theta_j)^2}{2\,\sigma_{\text{pyr}}^2}\right) & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

2. **Normalization constraint.** Define $S = \sum_{j \neq 0} G_{0j}$ (identical for all rows by symmetry). The inhibitory baseline $J_-$ is derived so that the mean weight equals $1/N$:
$$J_- = \frac{1 - J_+ \, S}{N - 1 - S}$$

   When $J_+ > 1$, $J_-$ becomes negative, producing surround inhibition.

3. **Weight matrix** (scaled by $1/N$ for network-size invariance):
$$W_{ij}^{\text{PYR}\to\text{PYR}} = \frac{1}{N}\bigl[J_- + (J_+ - J_-)\,G_{ij}\bigr], \quad W_{ii} = 0$$

   $J_+$ controls the peak excitation strength at nearby nodes (default: 1.6).

#### Inter-node excitatory input

For both profiles, the inter-node excitatory input to PYR at node $i$ is:

$$I_{\text{inter},i}^{\text{PYR}} = \sum_{j=0}^{N-1} W_{ij}^{\text{PYR}\to\text{PYR}} \cdot r_j^{\text{PYR}}$$

Self-connections ($i = j$) are always zero because local PYR $\to$ PYR recurrence is handled by the within-node weight $w_{ee}$.

### 2.3 PV $\to$ PYR Global Inhibition

PV interneurons from all nodes inhibit PYR at each node, creating an excitation-inhibition loop: local PYR excites local PV (via $w_{ep}$), then PV activity is broadcast globally to suppress PYR everywhere. This mechanism provides the competitive inhibition needed for bump formation.

#### Uniform Mode (default)

All PV-to-PYR connections have equal weight (excluding self):

$$W_{ij}^{\text{PV}\to\text{PYR}} = \begin{cases} \displaystyle\frac{w_{\text{PV}}^{\text{global}}}{N-1} & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

#### Gaussian Mode

Distance-dependent PV inhibition with row-sum normalization (same principle as the Gaussian PYR profile):

$$\tilde{W}_{ij} = \begin{cases} \exp\!\left(-\dfrac{d(\theta_i, \theta_j)^2}{2\,\sigma_{\text{PV}}^2}\right) & \text{if } i \neq j \\[6pt] 0 & \text{if } i = j \end{cases}$$

$$W_{ij}^{\text{PV}\to\text{PYR}} = w_{\text{PV}}^{\text{global}} \cdot \frac{\tilde{W}_{ij}}{\sum_k \tilde{W}_{ik}}$$

The inter-node inhibitory input from PV to PYR at node $i$ is:

$$I_{\text{inter},i}^{\text{PV}\to\text{PYR}} = \sum_{j=0}^{N-1} W_{ij}^{\text{PV}\to\text{PYR}} \cdot r_j^{\text{PV}}$$

### 2.4 Connectivity Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `n_nodes` | $N$ | 64 | Number of nodes on the ring |
| `pyr_profile_type` | -- | `"gaussian"` | PYR→PYR profile: `"gaussian"` or `"compte"` |
| `w_pyr_pyr_inter` | $w_{\text{pyr}}^{\text{inter}}$ | 18.55 | Total row-sum of PYR→PYR weights (Gaussian profile only) |
| `sigma_pyr_deg` | $\sigma_{\text{pyr}}$ | 30.0 deg | Gaussian width of PYR→PYR profile |
| `J_plus` | $J_+$ | 1.6 | Local excitation peak (Compte profile only) |
| `w_pv_global` | $w_{\text{PV}}^{\text{global}}$ | 2.0 | Strength of PV→PYR global inhibition |
| `pv_global_type` | -- | `"uniform"` | `"uniform"` or `"gaussian"` |
| `sigma_pv_deg` | $\sigma_{\text{PV}}$ | 180.0 deg | Width of PV profile (if Gaussian) |

**Note on network-size invariance:** Both the Gaussian (row-sum normalization) and Compte ($1/N$ scaling) profiles ensure that the effective coupling is independent of $N$, so simulations at $N=64$, $N=128$, and $N=1024$ produce comparable dynamics.

---

## 3. Local Circuit Dynamics

Each node follows rate dynamics governed by:

$$\tau_s \frac{dr_i^X}{dt} = -r_i^X + \Phi^X(I_i^X) + \sigma_s \, \xi_i^X(t)$$

where $\tau_s$ is the synaptic time constant, $\Phi^X$ is the transfer function for population $X$, $I_i^X$ is the total input current, $\sigma_s$ is the noise amplitude, and $\xi_i^X(t)$ is a noise process.

Firing rates are clamped to $r_i^X \in [0,\, 200]$ Hz at each integration step. The upper bound acts as a safety net against numerical overflow in large networks while remaining well above physiological firing rates.

### 3.1 Input Current Equations

**PYR** receives local recurrent excitation (with divisive PV inhibition), inter-node excitation, inter-node PV inhibition, SOM subtractive inhibition, adaptation, external drive, and stimulus:

$$I_i^{\text{PYR}} = \frac{w_{ee} \, r_i^{\text{PYR}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}} + I_{\text{inter},i}^{\text{PYR}} - g_{\text{GABA}} \, I_{\text{inter},i}^{\text{PV}\to\text{PYR}} - g_{\text{GABA}} \, w_{se} \, r_i^{\text{SOM}} - I_{\text{adapt},i}^{\text{PYR}} + I_{\text{ext}}^{\text{PYR}} + I_{\text{stim},i}(t)$$

The term $\frac{w_{ee} \, r_i^{\text{PYR}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}}$ implements **divisive (shunting) inhibition** from PV interneurons, modeling the effect of perisomatic GABAergic synapses on input resistance.

**SOM** (local connections only):

$$I_i^{\text{SOM}} = w_{es} \, r_i^{\text{PYR}} - g_{\text{GABA}} \, w_{ps} \, r_i^{\text{PV}} - w_{vs} \, r_i^{\text{VIP}} - I_{\text{adapt},i}^{\text{SOM}} + I_{\text{ext}}^{\text{SOM}}$$

**PV** (local connections only; PV's global effect is on PYR, not on other PV):

$$I_i^{\text{PV}} = w_{ep} \, r_i^{\text{PYR}} - g_{\text{GABA}} \, w_{pp} \, r_i^{\text{PV}} - g_{\text{GABA}} \, w_{sp} \, r_i^{\text{SOM}} - w_{vp} \, r_i^{\text{VIP}} + I_{\text{ext}}^{\text{PV}}$$

**VIP** (local connections only):

$$I_i^{\text{VIP}} = w_{ev} \, r_i^{\text{PYR}} - w_{vv} \, r_i^{\text{VIP}} + I_{\text{ext}}^{\text{VIP}}$$

### 3.2 Weight Notation

Weights follow the convention $w_{XY}$ = connection **from** population $Y$ **to** population $X$:
- e = PYR (excitatory), p = PV, s = SOM, v = VIP
- Example: $w_{ep}$ = weight from PYR to PV

### 3.3 GABA Scaling

Inhibitory weights are multiplied by a GABA scaling factor:

$$g_{\text{GABA}} = g_{\text{GABA}}^{\text{base}} + \text{act}_{\alpha 7} \cdot g_{\alpha 7}$$

where $\text{act}_{\alpha 7} \in [0, 1]$ is the alpha7 nAChR activation level (0 under knockout).

### 3.4 External Currents

Each population receives baseline tonic drive plus receptor-mediated currents:

$$I_{\text{ext}}^{\text{PYR}} = I_0^{\text{PYR}}$$

$$I_{\text{ext}}^{\text{PV}} = I_0^{\text{PV}} + \text{act}_{\alpha 7} \cdot I_{\alpha 7}^{\text{PV}}$$

$$I_{\text{ext}}^{\text{SOM}} = I_0^{\text{SOM}} + \text{act}_{\alpha 7} \cdot I_{\alpha 7}^{\text{SOM}} + \text{act}_{\beta 2} \cdot I_{\beta 2}^{\text{SOM}}$$

$$I_{\text{ext}}^{\text{VIP}} = I_0^{\text{VIP}} + \text{act}_{\alpha 5} \cdot I_{\alpha 5}^{\text{VIP}}$$

### 3.5 Transient Current

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
| `tau_adapt_pyr` | $\tau_{\text{adapt}}^{\text{PYR}}$ | 186.6 ms | PYR adaptation time constant |
| `tau_adapt_som` | $\tau_{\text{adapt}}^{\text{SOM}}$ | 2320.5 ms | SOM adaptation time constant (slow) |
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

## 6. Stimulus Protocol

### 6.1 Spatial Profile

The stimulus is a current injection to PYR neurons with a Gaussian spatial profile centered at angle $\theta_{\text{stim}}$:

$$S_{\text{spatial}}(\theta_i) = \exp\!\left(-\frac{d(\theta_i, \theta_{\text{stim}})^2}{2\,\sigma_{\text{stim}}^2}\right)$$

where $\sigma_{\text{stim}}$ is the stimulus spatial width (default: $20\degree$).

### 6.2 Temporal Modulation

The stimulus has a temporal profile $M(t)$ active during the window $[t_{\text{on}}, t_{\text{off}})$ where $t_{\text{off}} = t_{\text{on}} + \Delta t_{\text{stim}}$. Four shapes are available:

**Square** (default):
$$M(t) = 1$$

**Ramp on** (gradual onset):
$$M(t) = \min\!\left(\frac{t - t_{\text{on}}}{0.1 \cdot \Delta t_{\text{stim}}},\; 1\right)$$

**Ramp off** (gradual offset):
$$M(t) = \min\!\left(\frac{t_{\text{off}} - t}{0.1 \cdot \Delta t_{\text{stim}}},\; 1\right)$$

**Gaussian** (bell-shaped):
$$M(t) = \exp\!\left(-\frac{(t - t_{\text{on}} - \Delta t_{\text{stim}}/2)^2}{2 \cdot (\Delta t_{\text{stim}}/4)^2}\right)$$

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
| `sigma_deg` | $\sigma_{\text{stim}}$ | 20.0 deg | Spatial width (Gaussian sigma) |
| `onset_ms` | $t_{\text{on}}$ | 10500 ms | Stimulus onset time |
| `duration_ms` | $\Delta t_{\text{stim}}$ | 250 ms | Stimulus duration |
| `shape` | -- | `"square"` | Temporal shape |

---

## 7. Distractor Mechanism

An optional distractor stimulus can be presented during the delay period using the `WorkingMemoryProtocol` class. The distractor is a second `RingStimulus` with its own angular location, amplitude, timing, and spatial width:

$$I_{\text{distract},i}(t) = A_{\text{dist}} \cdot \exp\!\left(-\frac{d(\theta_i, \theta_{\text{dist}})^2}{2\,\sigma_{\text{stim}}^2}\right) \cdot M_{\text{dist}}(t)$$

Multiple stimuli are summed: $I_{\text{stim},i}(t) = \sum_k I_{\text{stim},i}^{(k)}(t)$

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distractor_location_deg` | `None` | Distractor angle (None = no distractor) |
| `distractor_amplitude` | 3.0 | Distractor current amplitude |
| `distractor_onset_ms` | 1500.0 | Distractor onset (absolute time, ms) |
| `distractor_duration_ms` | 200.0 | Distractor duration (ms) |

### 7.1 Standard Distractor Protocol (ring-distractor-sweep)

The `ring-distractor-sweep` command uses a two-delay protocol that cleanly separates bump formation, distractor presentation, and post-distractor recovery:

| Phase | Duration | Description |
|-------|----------|-------------|
| Burn-in | 10,000 ms | Network settles to baseline |
| Pre-cue baseline | 500 ms | Continued baseline |
| Cue | 250 ms | Spatial stimulus at $180°$, amplitude $A_{\text{cue}}$ |
| Delay₁ | 1,000 ms (default) | Memory consolidation; bump forms and stabilizes |
| Distractor | 250 ms (default) | Competing stimulus at $180° + \Delta\varphi$, amplitude $\alpha \cdot A_{\text{cue}}$ |
| Delay₂ | 1,000 ms (default) | Post-distractor recovery; measure final bump state |

The two swept dimensions are:
- $\Delta\varphi \in \{0°, 45°, 90°, 135°, 180°\}$ — angular offset of distractor from cue
- $\alpha \in \{0.5\times, 0.75\times, 1.0\times, 1.25\times, 1.5\times\}$ — distractor amplitude relative to cue

For each cell $(\Delta\varphi, \alpha)$, bump position is measured 50 ms before distractor onset ($\hat{\theta}_{\text{before}}$) and 100 ms after distractor offset ($\hat{\theta}_{\text{after}}$). The signed bump shift is:

$$\Delta\hat{\theta} = (\hat{\theta}_{\text{after}} - \hat{\theta}_{\text{before}} + \pi) \bmod 2\pi - \pi$$

positive values indicate drift toward the distractor. Bump collapse is declared when $\hat{A}_{\text{after}} < \tau$, where $\tau$ is the **noise floor** read from the nearest matching row of `calibration_summary.csv` (keyed on condition, cue amplitude, and `w_inter`). This makes the threshold parameter-dependent: a tighter bump produced by a high-amplitude cue has a higher noise floor than a weaker one, so the criterion scales appropriately. If no calibration file is found, $\tau = 0.2$ is used with a warning.

---

## 8. Noise

Three noise modes are available, added to each population at each node:

### White Noise
$$\xi_i^X(t) \sim \mathcal{N}(0, 1) \quad \text{(i.i.d. at each time step)}$$

### Ornstein-Uhlenbeck (OU) Noise
$$d\xi_i^X = -\frac{\xi_i^X}{\tau_{\text{noise}}} \, dt + \sqrt{\frac{2}{\tau_{\text{noise}}}} \, dW_i^X$$

Discretized (Euler-Maruyama):
$$\xi_i^X(t + \Delta t) = \xi_i^X(t) - \frac{\xi_i^X(t)}{\tau_{\text{noise}}} \Delta t + \sqrt{\frac{2\,\Delta t}{\tau_{\text{noise}}}} \, \eta_i^X, \quad \eta_i^X \sim \mathcal{N}(0, 1)$$

### No Noise
$$\xi_i^X(t) = 0$$

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

(their p. 24, lines immediately before Eq. 20). This is exactly our **white noise** mode (up to 
the $\sqrt{\phi_{0,i}}$ amplitude scaling, which in our rate model is absorbed into $\sigma_{\text{noise}}$).

**OU noise** introduces temporal correlations with timescale $\tau_{\text{noise}}$ and falls outside 
the Seeholzer et al. derivation, which requires white (delta-correlated) noise for the diffusion 
coefficient $B$ in Eq. (5) to hold. OU noise is instead motivated as a model of slowly fluctuating 
background input from other cortical areas, independent of the bump attractor theory.

**In practice:** white noise is the appropriate mode for comparing empirical diffusion coefficients 
$\hat{B}$ to the theoretical prediction of Eq. (5). OU noise produces trial-to-trial variability 
with a characteristic timescale but will not match the $\langle[\varphi(t)-\varphi(0)]^2\rangle = B \cdot t$ 
scaling predicted by the theory.

## 9. Experimental Conditions

The model simulates 8 conditions by modifying receptor activation multipliers $(\text{act}_{\alpha 7}, \text{act}_{\beta 2}, \text{act}_{\alpha 5})$ and the GABA scaling parameter $g_{\alpha 7}$:

| Condition | $\text{act}_{\alpha 7}$ | $\text{act}_{\beta 2}$ | $\text{act}_{\alpha 5}$ | $g_{\alpha 7}$ |
|-----------|:---:|:---:|:---:|:---:|
| **WT** | 1.0 | 1.0 | 1.0 | default |
| **WT + APP** | $\sim \mathcal{N}(0.10, 0.03)$ | $\sim \mathcal{N}(0.875, 0.06)$ | $\sim \mathcal{N}(0.60, 0.05)$ | default |
| **$\alpha 7$ KO** | 0.0 | 1.0 | 1.0 | 0.0 |
| **$\alpha 7$ KO + APP** | 0.0 | $\sim \mathcal{N}(0.875, 0.06)$ | $\sim \mathcal{N}(0.60, 0.05)$ | 0.0 |
| **$\beta 2$ KO** | 1.0 | 0.0 | 1.0 | default |
| **$\beta 2$ KO + APP** | $\sim \mathcal{N}(0.10, 0.03)$ | 0.0 | $\sim \mathcal{N}(0.60, 0.05)$ | default |
| **$\alpha 5$ KO** | 1.0 | 1.0 | 0.0 | default |
| **$\alpha 5$ KO + APP** | $\sim \mathcal{N}(0.10, 0.03)$ | $\sim \mathcal{N}(0.875, 0.06)$ | 0.0 | default |

APP desensitization distributions are clipped to biologically plausible ranges:
- $\text{act}_{\alpha 7}$: [0.02, 0.20] (80-98% inactivated)
- $\text{act}_{\beta 2}$: [0.75, 1.00] (0-25% inactivated)
- $\text{act}_{\alpha 5}$: [0.45, 0.75] (25-55% inactivated)

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

### 10.5 Diffusion Coefficient

The diffusion coefficient quantifies stochastic wandering of the bump, computed from the mean squared displacement (MSD):

$$D = \frac{\langle [\hat{\theta}(t + \tau) - \hat{\theta}(t)]^2 \rangle}{2\tau}$$

where $\tau = 100$ ms and the average is over all time pairs in the analysis window. Expressed in degrees$^2$/second.

### 10.6 Working Memory Error

The angular error between the decoded bump position at the end of the delay and the original cue location:

$$\varepsilon = d(\hat{\theta}_{\text{final}}, \theta_{\text{stim}})$$

A bump is considered **maintained** if $\hat{A} > 0.3$ at the evaluation time.

### 10.7 Metrics Summary

| Metric | Key | Unit | Description |
|--------|-----|------|-------------|
| Bump center | `center_mean_deg` | degrees | Circular mean of decoded position |
| Center variability | `center_std_deg` | degrees | Circular std of decoded position |
| Decoding confidence | `amplitude_mean` | [0, 1] | Mean population vector length |
| Bump width | `width_mean_deg` | degrees | Circular std (von Mises approx.) |
| Drift rate | `drift_rate_deg_per_s` | deg/s | Systematic drift velocity |
| Diffusion | `diffusion_deg2_per_s` | deg$^2$/s | Stochastic diffusion coefficient |
| Error from cue | `error_from_cue_deg` | degrees | Angular distance to cue location |

### 10.8 Multi-Trial Aggregation

When running multiple trials, metrics are aggregated across trials as mean $\pm$ SEM (standard error of the mean):

$$\bar{m} = \frac{1}{K}\sum_{k=1}^{K} m_k, \quad \text{SEM} = \frac{s}{\sqrt{K}}$$

where $K$ is the number of trials and $s$ is the sample standard deviation.

---

## 11. Bump Amplitude Oscillations

### 11.1 Mechanism

After the stimulus offset, the bump amplitude does not settle immediately to a steady value. Instead, it undergoes **damped oscillations** driven by the slow negative feedback of spike-frequency adaptation (SFA). The sequence is:

1. Stimulus drives strong activation → bump forms, amplitude rises.
2. Adaptation builds up during the stimulus → suppresses activity slightly after offset.
3. When the stimulus turns off, adaptation current is still elevated → amplitude undershoots.
4. Adaptation decays → amplitude recovers, overshoots.
5. This bounce repeats with exponentially decaying amplitude until the attractor settles.

In practice (default parameters, 128 nodes), the dominant oscillation frequency is around **~9–10 Hz (period ≈ 100 ms)**. The oscillations are visible in the trial-averaged amplitude and are systematic (not noise-driven): they represent a genuine resonance of the bump attractor.

### 11.2 Effect on MSD

The bump position $\varphi(t)$ is estimated as the phase of the population vector. When the amplitude oscillates, the effective signal-to-noise on the phase estimate also oscillates. Moreover, the position itself may experience small correlated displacements at the oscillation frequency.

The oscillation adds a periodic term to the theoretical MSD:

$$\text{MSD}(\tau) \approx B\,\tau + C\left(1 - \cos\!\left(\frac{2\pi\tau}{T_\text{osc}}\right)\right) + \text{offset}$$

where $B$ is the true diffusion coefficient, $C$ is the oscillation contribution, and $T_\text{osc} = 1/f_\text{osc}$. Fitting a pure line $B\tau$ in the early regime ($\tau < T_\text{osc}$) **overestimates $\hat{B}$** because the oscillation increases apparent displacement at short lags.

### 11.3 Approaches to Correct or Mitigate the Problem

Five strategies are available, from simplest to most principled:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Exclude early transient** | Start MSD fit range after $N$ oscillation periods (e.g. `fit_range_s[0] = 3 × T_osc`) | Simple, no pre-processing | Wastes early data; requires knowing $T_\text{osc}$ |
| **B. Low-pass filter position** ✓ | Apply zero-phase Butterworth LP filter to $\varphi(t)$ at $f_\text{cut} < f_\text{osc}$ before MSD | Clean, preserves slow drift, intuitive | Introduces slight edge effects; needs $f_\text{cut}$ choice |
| **C. Oscillation-corrected fit** ✓ | Fit $\text{MSD} = B\tau + C(1-\cos(2\pi f\tau))$ with $f$ fixed from FFT | Separates diffusion and oscillation rigorously | Requires prior knowledge of $f_\text{osc}$; 3-param fit |
| **D. Time-windowed averaging** | Replace instantaneous $\varphi$ with running mean over 1 cycle | Simple, no filter needed | Introduces temporal smearing of genuine drift |
| **E. Fit only long lags** | Restrict fit to $\tau \gg T_\text{osc}$ where cosine term averages out | No preprocessing | Greatly reduces usable lag range; noisier fit |

**Current implementation**: strategies **B** (low-pass filter) and **C** (oscillation-corrected fit) are applied automatically when a dominant oscillation is detected by FFT of the per-trial amplitude. The filter cutoff defaults to $0.4 \times f_\text{osc}$ and can be overridden with `--filter_cutoff_hz` (set to `0` to disable).

### 11.4 Oscillation Detection

The `compute_oscillation_spectrum` function (in `analysis.py`) computes the power spectrum of the bump amplitude for each trial, averages across trials, and identifies the dominant frequency as the peak exceeding **3× the median power** in the band $[1, 50]$ Hz. The result is reported in `diffusion_oscillation.csv` and visualised in `diffusion_oscillation_spectrum.png`.

---

## 12. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cerebral Cortex*, 10(9), 910-923.

3. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
