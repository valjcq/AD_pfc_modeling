# Ring Attractor Network — Model & Implementation

This document describes the mathematical formulation and implementation of the ring attractor network model used for spatial working memory simulations. The model builds on the 4-population PFC circuit by arranging $N$ identical local circuits on a ring with distance-dependent inter-node connectivity.

For experimental analysis commands see [ring_experiments.md](ring_experiments.md).

---

## Table of Contents

1. [Network Architecture](#1-network-architecture)
2. [Inter-Node Connectivity](#2-inter-node-connectivity)
   - [2.1 Angular Distance](#21-angular-distance)
   - [2.2 PYR → PYR Excitation](#22-pyr--pyr-excitation)
   - [2.3 PV → PYR Global Inhibition](#23-pv--pyr-global-inhibition)
   - [2.4 Connectivity Parameters](#24-connectivity-parameters)
3. [Local Circuit Dynamics](#3-local-circuit-dynamics)
4. [Spike-Frequency Adaptation](#4-spike-frequency-adaptation)
5. [Transfer Function](#5-transfer-function)
6. [Stimulus & Distractor Protocol](#6-stimulus--distractor-protocol)
   - [6.6 Distractor Stimulus](#66-distractor-stimulus)
7. [Noise](#7-noise)
8. [Experimental Conditions](#8-experimental-conditions)
9. [Bump Amplitude Oscillations](#9-bump-amplitude-oscillations)
10. [References](#10-references)

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

Self-connections ($i = j$) are always zero because local PYR $\to$ PYR recurrence is handled by the within-node weight $w_{ee}$.

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

The standard distractor sweep protocol (`ring-distractor-sweep`) is documented in [ring_experiments.md](ring_experiments.md#14-2d-distractor-sweep).

---

---

## 7. Noise

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

## 8. Experimental Conditions

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

## 10. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
