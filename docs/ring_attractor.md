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
11. [References](#11-references)

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
| `w_pv_global` | $w_{\text{PV}}^{\text{global}}$ | 0.3 | Strength of PV→PYR global inhibition |
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

The bump center is estimated using the **circular mean** (population vector) method. For activity $r_i$ at nodes with angles $\theta_i$:

$$\bar{z} = \frac{\sum_{i=0}^{N-1} r_i \, e^{j\theta_i}}{\sum_{i=0}^{N-1} r_i}$$

where $j = \sqrt{-1}$. The decoded center and amplitude are:

$$\hat{\theta} = \arg(\bar{z}) \mod 2\pi$$

$$\hat{A} = |\bar{z}| \in [0, 1]$$

$\hat{A}$ is a confidence measure: $\hat{A} = 1$ for a perfect delta-function bump, $\hat{A} \approx 0$ for uniform activity.

### 10.2 Bump Width Estimation

The bump width is estimated using the **circular standard deviation**. Given the decoded center $\hat{\theta}$:

1. Compute angular deviations: $\delta_i = d(\theta_i, \hat{\theta})$
2. Compute circular resultant length:
$$R = \sqrt{\left(\frac{\sum_i r_i \cos \delta_i}{\sum_i r_i}\right)^2 + \left(\frac{\sum_i r_i \sin \delta_i}{\sum_i r_i}\right)^2}$$

3. Convert to standard deviation (von Mises approximation):
$$\sigma_{\text{bump}} = \sqrt{-2 \ln(R)}$$

$R = 1$ corresponds to a perfectly peaked distribution; $R = 0$ to uniform.

### 10.3 Drift Rate

Systematic drift during the delay is estimated by fitting a linear trend to the unwrapped bump center trajectory:

$$v_{\text{drift}} = \frac{\hat{\theta}(t_{\text{end}}) - \hat{\theta}(t_{\text{start}})}{t_{\text{end}} - t_{\text{start}}}$$

expressed in degrees/second.

### 10.4 Diffusion Coefficient

The diffusion coefficient quantifies stochastic wandering of the bump, computed from the mean squared displacement (MSD):

$$D = \frac{\langle [\hat{\theta}(t + \tau) - \hat{\theta}(t)]^2 \rangle}{2\tau}$$

where $\tau = 100$ ms and the average is over all time pairs in the analysis window. Expressed in degrees$^2$/second.

### 10.5 Working Memory Error

The angular error between the decoded bump position at the end of the delay and the original cue location:

$$\varepsilon = d(\hat{\theta}_{\text{final}}, \theta_{\text{stim}})$$

A bump is considered **maintained** if $\hat{A} > 0.3$ at the evaluation time.

### 10.6 Metrics Summary

| Metric | Key | Unit | Description |
|--------|-----|------|-------------|
| Bump center | `center_mean_deg` | degrees | Circular mean of decoded position |
| Center variability | `center_std_deg` | degrees | Circular std of decoded position |
| Decoding confidence | `amplitude_mean` | [0, 1] | Mean population vector length |
| Bump width | `width_mean_deg` | degrees | Circular std (von Mises approx.) |
| Drift rate | `drift_rate_deg_per_s` | deg/s | Systematic drift velocity |
| Diffusion | `diffusion_deg2_per_s` | deg$^2$/s | Stochastic diffusion coefficient |
| Error from cue | `error_from_cue_deg` | degrees | Angular distance to cue location |

### 10.7 Multi-Trial Aggregation

When running multiple trials, metrics are aggregated across trials as mean $\pm$ SEM (standard error of the mean):

$$\bar{m} = \frac{1}{K}\sum_{k=1}^{K} m_k, \quad \text{SEM} = \frac{s}{\sqrt{K}}$$

where $K$ is the number of trials and $s$ is the sample standard deviation.

---

## 11. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000). Synaptic mechanisms and network dynamics underlying spatial working memory in a cortical network model. *Cerebral Cortex*, 10(9), 910-923.

3. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
