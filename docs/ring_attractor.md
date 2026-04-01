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
10. [Joint Ring + Circuit Optimization](#10-joint-ring--circuit-optimization)
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

$$\tau_s \frac{dr_i^X}{dt} = -r_i^X + \Phi^X(I_i^X)$$

where $\tau_s$ is the synaptic time constant, $\Phi^X$ is the transfer function for population $X$, and $I_i^X$ is the total input current. Noise is injected into the PYR input current (see [Section 7](#7-noise)).

Firing rates are clamped to $r_i^X \in [0,\, 200]$ Hz at each integration step. The upper bound acts as a safety net against numerical overflow in large networks while remaining well above physiological firing rates.

### 3.1 Input Current Equations

**PYR** receives local recurrent excitation (with divisive PV inhibition), inter-node excitation, inter-node PV inhibition, SOM subtractive inhibition, adaptation, external drive, stimulus, and noise:

$$I_i^{\text{PYR}} = \frac{w_{ee} \, r_i^{\text{PYR}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}} + I_{\text{inter},i}^{\text{PYR}} - g_{\text{GABA}} \, I_{\text{inter},i}^{\text{PV}\to\text{PYR}} - g_{\text{GABA}} \, w_{se} \, r_i^{\text{SOM}} - I_{\text{adapt},i}^{\text{PYR}} + I_{\text{ext}}^{\text{PYR}} + I_{\text{stim},i}(t) + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$$

The term $\frac{w_{ee} \, r_i^{\text{PYR}}}{1 + g_{\text{GABA}} \, w_{pe} \, r_i^{\text{PV}}}$ implements **divisive (shunting) inhibition** from PV interneurons, modeling the effect of perisomatic GABAergic synapses on input resistance.

The noise term $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$ injects stochastic current into each PYR node. Injecting noise at the current level (before the transfer function $\Phi^{\text{PYR}}$) means its effect on firing rate is naturally filtered by the transfer function slope $\Phi'$, consistent with a diffusion-approximation interpretation of Poisson spiking variability. The proportionality to $I_{\text{ext}}^{\text{PYR}}$ ensures noise scales automatically across experimental conditions.

**SOM** (local connections only):

$$I_i^{\text{SOM}} = w_{es} \, r_i^{\text{PYR}} - w_{vs} \, r_i^{\text{VIP}} - I_{\text{adapt},i}^{\text{SOM}} + I_{\text{ext}}^{\text{SOM}}$$

**PV** (local connections only; PV's global effect is on PYR, not on other PV):

$$I_i^{\text{PV}} = w_{ep} \, r_i^{\text{PYR}} - g_{\text{GABA}} \, w_{pp} \, r_i^{\text{PV}} - g_{\text{GABA}} \, w_{sp} \, r_i^{\text{SOM}} - w_{vp} \, r_i^{\text{VIP}} + I_{\text{ext}}^{\text{PV}}$$

**VIP** (local connections only):

$$I_i^{\text{VIP}} = w_{ev} \, r_i^{\text{PYR}} + I_{\text{ext}}^{\text{VIP}}$$

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

Noise is injected as a shared stochastic current perturbation into **all four populations** (PYR, SOM, PV, VIP) at each node independently. The same noise factor is applied to each population, ensuring correlated variability across populations at each node. This models the variability in synaptic drive (diffusion approximation of Poisson spike trains).

### Noise equation

The noisy input current for each population $X \in \{\text{PYR, SOM, PV, VIP}\}$ at node $i$ is:

$$I_i^{X}(t) = I_i^{X,\text{det}}(t) + \underbrace{\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}}}_{\text{noise scale (nA)}} \cdot \xi_i(t)$$

where $I_i^{X,\text{det}}$ is the deterministic part (all synaptic, adaptation, and stimulus terms), $\sigma_{\text{noise}}$ is the dimensionless noise amplitude, and $\xi_i(t)$ is the shared noise process (see below). Note that the noise scale is proportional to the baseline PYR drive $I_{\text{ext}}^{\text{PYR}}$ but applied equally to all populations, ensuring that noise amplitudes scale together across all population types. This automatically adjusts across experimental conditions with different drive levels.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `sigma_noise` | $\sigma_{\text{noise}}$ | `0.3` | Dimensionless noise amplitude. Noise current std = `sigma_noise × I_ext_pyr` (nA) |

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

(their p. 24, lines immediately before Eq. 20). Our model injects current-space noise into PYR with amplitude $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}}$, which after passing through the transfer function slope $\Phi'$ produces effective rate noise consistent with this formulation. The $\sqrt{\phi_{0,i}}$ amplitude scaling of the original is absorbed into $\sigma_{\text{noise}}$.

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

The `ring-optimize` command jointly fits `CircuitParams` and `RingParams` so the ring network at rest (no stimulus) reproduces experimentally measured quiet-wakefulness firing rates.

### 10.1 Motivation

The standard workflow optimizes `CircuitParams` against single-node firing rates and then tunes ring parameters (`w_pyr_pyr_inter`, `w_pv_global`, `sigma_pyr_deg`) separately via calibration sweeps. This two-step approach can produce inconsistencies: the local circuit was fit assuming no inter-node coupling, but ring connectivity alters the effective input each node receives. Joint optimization resolves this by optimizing all parameters simultaneously against ring-level measurements.

A deeper problem is that matching firing rates alone does not guarantee that the fitted network can support a working-memory bump. Rate matching is a necessary but not sufficient condition: a network can reproduce quiet-wakefulness rates while being unable to sustain localized activity. Two additional constraints address this:

1. **Turing instability** (`--turing_weight`): a necessary condition for bump formation, derived analytically from the parameters without any additional simulation.
2. **Adaptation control** (`--no_adapt`): spike-frequency adaptation can mask whether a bump is truly self-sustaining or merely decaying; disabling it tests robustness under the most permissive conditions for bump stability.

### 10.2 Parameter Space

The joint search space is `CircuitParams` (~60 parameters) extended with three ring-specific parameters:

| Parameter | Symbol | Default bounds | Description |
|-----------|--------|---------------|-------------|
| `w_pyr_pyr_inter` | $w_\text{pyr}^\text{inter}$ | [1, 30] | Total inter-node PYR→PYR coupling (row-sum normalized) |
| `w_pv_global` | $w_\text{PV}^\text{global}$ | [0.5, 20] | Total global PV→PYR inhibition (uniform all-to-all) |
| `sigma_pyr_deg` | $\sigma_\text{pyr}$ | [10°, 60°] | Gaussian width of PYR→PYR connectivity profile |

`n_nodes` is fixed by the user and not optimized.

### 10.3 Loss Function

All modes share the base loss terms:

$$\mathcal{L}_\text{base} = \mathcal{L}_\text{rate} + \frac{1}{N_\text{KO}} \sum_k \mathcal{L}_k^\text{KO} + \mathcal{L}_\text{Jacobian}$$

where $\mathcal{L}_\text{rate}$ is the MSPE between node-averaged ring firing rates and the target rates from quiet wakefulness:

$$\mathcal{L}_\text{rate} = \frac{1}{4} \sum_{X \in \{P,S,V,I\}} \left(\frac{\bar{r}^X - r^X_\text{target}}{r^X_\text{target}}\right)^2$$

The node-averaged rate $\bar{r}^X$ is:

$$\bar{r}^X = \frac{1}{N} \sum_{i=1}^N \langle r_i^X \rangle_{\text{window}}$$

KO conditions (alpha7, alpha5, beta2) are run on single-node by default or on the ring with `--ko_on_ring`.

#### Turing instability penalty (optional, `--turing_weight`)

For a ring network to support cue-triggered bump states without generating spontaneous bumps at rest, the network must satisfy a **three-regime bistable attractor condition**. This ensures that:
- The uniform state is stable at rest (no spontaneous bump nucleation);
- A stable self-sustained bump fixed point exists in the intermediate firing-rate regime (~20–40 Hz PYR);
- The transfer function self-limits at cue-driven rates, preventing runaway growth to saturation.

These three conditions define the full bistable attractor geometry and are expressed via the **corrected Turing instability criterion** for the full 4-population ring network.

**Step 1 — W&W transfer-function derivative.** The Wong-Wang transfer function for population $x$ is:

$$\Phi^x(I) = A_x \cdot \frac{u}{1 - e^{-g_x u}}, \qquad u = c_x (I - \Theta_x)$$

Its derivative with respect to $I$ is:

$${\Phi^x}'(I) = A_x \cdot c_x \cdot \frac{1 - e^{-g_x u}(1 + g_x u)}{(1 - e^{-g_x u})^2}$$

evaluated at the steady-state input current $I^*_x$ recovered from the circuit equations at the operating point. The shape parameters $c_x$, $\Theta_x$, $g_x$ are the fixed W&W 2006 constants ($c_e$, $\Theta_e$, $g_e$ for PYR; $c_i$, $\Theta_i$, $g_i$ for PV/SOM/VIP).

**Step 2 — Effective PYR gain.** The divisive PV→PYR inhibition in the PYR input equation reduces the effective PYR gain. Accounting for this local feedback loop:

$$G_\text{eff}(I^*) = \frac{{\Phi'}_\text{PYR}(I^*_\text{PYR})}{1 + g_\text{GABA} \cdot w_{pe} \cdot {\Phi'}_\text{PV}(I^*_\text{PV}) \cdot w_{ep} \cdot {\Phi'}_\text{PYR}(I^*_\text{PYR})}$$

where $w_{pe}$ is the PV→PYR divisive weight, $w_{ep}$ is the PYR→PV weight, and $g_\text{GABA}$ is the GABA scaling factor.

**Step 3 — Corrected Turing gain product.** The uniform state is linearly unstable to spatial (bump-mode, $k=1$) perturbations when:

$$G_\text{eff}(I^*) \cdot w^\text{inter}_\text{pyr} > 1$$

where $w^\text{inter}_\text{pyr}$ is the PYR→PYR inter-node spatial excitation (Mexican-hat Gaussian profile).

The global all-to-all PV→PYR inter-node inhibition does not appear because uniform kernels (zero spatial structure) have zero Fourier coefficient at all spatial modes $k \geq 1$. This kernel only affects the $k=0$ homogeneous mode, which is not part of the Turing criterion for bump formation. Thus the spatial drive is simply $w^\text{inter}_\text{pyr}$.

**Role of SOM and VIP.** SOM and VIP are local-only populations — they have no inter-node connections and therefore do not add new terms to the spatial Jacobian. They are consequently absent from the criterion above. However, they do influence the operating-point currents $I^*_\text{PYR}$ and $I^*_\text{PV}$ through the full 4-population circuit equations. This effect is already correctly accounted for: both $I^*_\text{PYR}$ and $I^*_\text{PV}$ are recovered via the full 4-population fixed-point computation at the fitted firing rates, so SOM and VIP contributions are implicitly included in the derivatives ${\Phi'}_\text{PYR}$ and ${\Phi'}_\text{PV}$.

The desired working-memory regime therefore requires three gain-product crossings:

$$G_\text{eff}(I^*_\text{rest}) \cdot w^\text{inter}_\text{pyr} < 1 \qquad \text{(stable at rest — no spontaneous bump)}$$

$$G_\text{eff}(I^*_\text{bump}) \cdot w^\text{inter}_\text{pyr} > 1 \qquad \text{(bump fixed point exists — self-sustained)}$$

$$G_\text{eff}(I^*_\text{cue}) \cdot w^\text{inter}_\text{pyr} < 1 \qquad \text{(self-limiting at cue — no runaway)}$$

**Computing the three operating points.** The rest operating point $I^*_\text{rest}$ is recovered directly from the circuit equations at the fitted baseline firing rates (~8 Hz PYR, consistent with Koukouli et al. 2025). The bump operating point is **fixed at $r_\text{PYR} = 40$ Hz**, consistent with self-sustained WM delay activity in rodent PFC. $I^*_\text{bump}$ is found by numerically inverting the PYR transfer function:

$$\Phi_\text{PYR}(I^*_\text{bump}) = 40 \text{ Hz} \quad \text{(bisection)}$$

The PV input $I^*_\text{PV}^\text{bump}$ is then derived from the full circuit equations at $r_\text{PYR} = 40$ Hz (other populations held at rest values). This approach directly targets a biologically motivated operating point rather than depending on a scale factor. The cue operating point $I^*_\text{cue}$ is obtained by scaling the PYR external drive by `--turing_cue_scale` (default 5.0):

$$I_0^\text{PYR} \to 5 \cdot I_0^\text{PYR}$$

This produces PYR firing rates at the cue-driven level (~50–60 Hz, clamped to 80 Hz max; required by inhibitory feedback, Pfeffer et al. 2013). For the cue operating point, the elevated PYR firing increases the input to PV via the recurrent connection $w_{ep}$, so $I^*_\text{PV}^\text{cue} > I^*_\text{PV}^\text{rest}$. All operating-point currents ($I^*_\text{PYR}$, $I^*_\text{PV}$) must be recomputed for each regime before evaluating ${\Phi'}_\text{PYR}$ and ${\Phi'}_\text{PV}$.

These operating points are grounded in the dynamics of the system: rest corresponds to baseline activity in quiet wakefulness; bump corresponds to the target self-sustained bump rate during the working-memory delay; and cue corresponds to the stimulus-driven activity that initiates and drives the bump. No additional simulation is required — all three operating points and their transfer-function derivatives are computed analytically from the current parameter set.

**Penalty.** The three conditions are enforced as a combined soft penalty with safety margin $m$ (default 0.05, `--turing_margin`):

$$\mathcal{L}_\text{rest} = \max\!\left(0,\; G_\text{eff}(I^*_\text{rest}) \cdot w^\text{inter}_\text{pyr} - (1 - m)\right)^2 \qquad \text{(penalise spontaneous bumps at rest)}$$

$$\mathcal{L}_\text{bump} = \max\!\left(0,\; 1 + m - G_\text{eff}(I^*_\text{bump}) \cdot w^\text{inter}_\text{pyr}\right)^2 \qquad \text{(penalise missing bump fixed point at 40 Hz)}$$

$$\mathcal{L}_\text{above} = \max\!\left(0,\; G_\text{eff}(I^*_\text{cue}) \cdot w^\text{inter}_\text{pyr} - (1 - m)\right)^2 \qquad \text{(penalise runaway at cue — ceiling, same form as } \mathcal{L}_\text{rest}\text{)}$$

$$\mathcal{L}_\text{Turing} = \mathcal{L}_\text{rest} + \mathcal{L}_\text{bump} + \mathcal{L}_\text{above}$$

$\mathcal{L}_\text{rest}$ is zero when the network is safely below the Turing threshold at rest; it activates as soon as the gain product approaches 1, pushing the optimizer away from parameter regions that produce spontaneous bumps. $\mathcal{L}_\text{bump}$ is zero when the network is safely above threshold at the self-sustained bump rate (fixed at 40 Hz, Koukouli et al. 2025); it activates when the bump-driven gain is below the threshold, penalising parameter regions where no stable bump fixed point exists. $\mathcal{L}_\text{above}$ is zero when the network is safely below threshold at cue-driven rates (same ceiling form as $\mathcal{L}_\text{rest}$); it activates when the cue-driven gain exceeds the threshold, penalising runaway growth to saturation (Pfeffer et al. 2013).

Diagnostic log line format: `[TURING] gp_rest=X gp_bump=X gp_cue=X L_rest=X L_bump=X L_above=X L_turing=X L_rate=X`

Together, these three terms enforce the full bistable attractor geometry: gain product must cross 1 three times (rest → bump threshold → bump fixed point → cue threshold), creating a self-sustaining but bounded bump state.

Note that satisfying this penalty guarantees the correct **geometry** for bistability — the three operating points properly straddle the Turing threshold — but does not guarantee bump persistence through the full delay period, which additionally depends on adaptation dynamics and noise. Bump persistence is verified empirically via simulation. The corrected three-term criterion properly accounts for PV feedback in the divisive inhibition (via $G_\text{eff}$) and enforces all three regime crossings, making it a more accurate proxy for the true bistable attractor dynamics of the 4-population ring.

#### Adaptation (`--no_adapt`)

When `--no_adapt` is set, $J_\text{adapt,PYR}$ and $J_\text{adapt,SOM}$ are fixed to zero and excluded from the search space. Spike-frequency adaptation introduces a slow negative feedback that can compensate for strong recurrent excitation and mask whether a bump is truly self-sustaining. Fitting without adaptation is a conservative choice: parameters that support a bump without adaptation will also support one in the presence of adaptation (which only helps stability). This mode is useful to establish a baseline that is independent of the adaptation time constants.

#### Summary of modes

| Mode | `ring-optimize` flags | Extra loss terms |
|------|----------------------|-----------------|
| 1 — rates only | *(none)* | — |
| 2 — rates + Turing | `--turing_weight 2.0` | $w_T \cdot \mathcal{L}_\text{Turing}$ |
| 3 — rates + Turing + no adapt | `--turing_weight 2.0 --no_adapt` | $w_T \cdot \mathcal{L}_\text{Turing}$ |
| 4 — single-node + no adapt + Turing | `optimize --turing_weight 2.0 --no_adapt` | $w_T \cdot \mathcal{L}_\text{Turing}$ (fixed $w_\text{ref}$) |
| 5 — rates + bump quality | `--bump_mode` | $w_B \cdot \mathcal{L}_\text{bump}$ |

The full loss for mode 2/3 is:

$$\mathcal{L} = \mathcal{L}_\text{base} + w_T \cdot \mathcal{L}_\text{Turing}$$

The full loss for mode 5 (bump quality) is:

$$\mathcal{L} = \mathcal{L}_\text{base} + w_B \cdot \mathcal{L}_\text{bump}$$

where:

$$\mathcal{L}_\text{bump} = \max\!\left(0,\; A_\text{min} - \langle A \rangle_\text{window}\right)^2$$

$\langle A \rangle_\text{window}$ is the mean population-vector amplitude during a post-stimulus window and $A_\text{min}$ (default 0.3) is the minimum acceptable amplitude. Mode 5 requires running an additional bump simulation per candidate, making it significantly more expensive than modes 2/3.

### 10.4 Computational Cost

Ring simulations are ~10–50× slower than single-node. To stay tractable:
- `n_trials_ring = 3` (vs 8 for single-node)
- KO conditions on single-node (same `CircuitParams`, no ring overhead)
- `n_nodes = 64` recommended during optimization (smaller and faster)
- The Turing penalty adds no simulation overhead — it is a pure analytical computation from the rest rates already computed in step 1

Each evaluation is roughly: 3 ring sims + 3 single-node KO sims (+ 1 bump sim in mode 5).

### 10.5 Output

```
ring_optim_output/
├── best_circuit_params.json   # Best CircuitParams (same format as optimize)
└── best_ring_params.json      # Best RingParams as JSON
```

These can be passed directly to `ring-run` or `ring-study` via `--params_json` and a manual `--w_pyr_pyr_inter / --w_pv_global / --sigma_pyr_deg` override.

See [CLI.md — ring-optimize](CLI.md#ring-optimize) for the full argument reference.

---

## 11. References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Wimmer, K., Nykamp, D. Q., Constantinidis, C., & Bhattacharyya, A. (2014). Bump attractor dynamics in prefrontal cortex explains behavioral precision in spatial working memory. *Nature Neuroscience*, 17(3), 431-439.
