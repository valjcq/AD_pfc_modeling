# PFC Circuit Model: 4-Population Rate Model with Ring Attractor

A computational model of the prefrontal cortex (PFC) microcircuit implementing a 4-population rate model with Nevergrad-based parameter optimization and a ring attractor network for spatial working memory.

**Documentation:**
- [CLI Reference](docs/CLI.md) -- All commands and parameters
- [Ring Attractor Model](docs/ring_attractor.md) -- Mathematical formulation of the ring network

---

## Table of Contents

1. [Biological Background](#biological-background)
2. [Model Architecture](#model-architecture)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Nicotinic Receptor Modulation](#nicotinic-receptor-modulation)
5. [Code Structure](#code-structure)
6. [Quick Start](#quick-start)
7. [Parameter Reference](#parameter-reference)
8. [Data Structures](#data-structures)
9. [References](#references)

---

## Biological Background

### Prefrontal Cortex Microcircuit Organization

The prefrontal cortex (PFC) is critical for executive functions including working memory, decision-making, and cognitive flexibility. Its computational properties emerge from the interplay of distinct neuronal populations:

### Cell Type Functions

| Population | Full Name | Markers | Layer | Function |
|------------|-----------|---------|-------|----------|
| **PYR** | Pyramidal cells | CaMKII, Tbr1 | 2/3, 5 | Principal excitatory neurons; provide output and recurrent excitation |
| **PV** | Parvalbumin interneurons | Parvalbumin | 4, 2/3 | Fast-spiking; provide perisomatic inhibition; regulate spike timing and gamma oscillations |
| **SOM** | Somatostatin interneurons | Somatostatin | 1, 2/3 | Target dendrites; provide dendritic inhibition; gate synaptic inputs |
| **VIP** | VIP interneurons | Vasoactive intestinal peptide | 1, 2/3 | Preferentially inhibit SOM cells; mediate disinhibition of PYR |

### Key Circuit Motifs

#### 1. Feedback Inhibition (PV → PYR)
PV interneurons receive excitation from PYR cells and provide fast, perisomatic inhibition back to PYR. This creates a negative feedback loop that:
- Stabilizes network activity
- Sharpens response timing
- Regulates network synchronization

#### 2. Disinhibition (VIP → SOM → PYR)
VIP interneurons inhibit SOM cells, which in turn inhibit PYR dendrites. When VIP is active:
- SOM inhibition is reduced
- PYR dendritic inputs are disinhibited
- Network gain increases

This motif is crucial for attention and learning, allowing selective amplification of inputs.

#### 3. Lateral Inhibition (SOM → PYR)
SOM interneurons provide dendritic inhibition that:
- Implements gain control
- Mediates surround suppression
- Regulates synaptic plasticity

---

## Model Architecture

### Population Connectivity Matrix

The model implements the following synaptic weight matrix (w_XY = weight from population Y to population X):

```
            FROM
         PYR    PV     SOM    VIP
      ┌─────────────────────────────┐
  PYR │ w_ee   w_pe   w_se    -     │  (recurrent, inhibition from PV/SOM)
  PV  │ w_ep   w_pp   w_sp   w_vp   │  (excitation from PYR, inhibition)
T SOM │ w_es    -      -     w_vs   │  (excitation from PYR, inhib from VIP)
O VIP │ w_ev    -      -     w_vv   │  (weak from PYR, self-inhibition)
      └─────────────────────────────┘
```

### Notation Convention
- **First letter**: target population (e = excitatory/PYR, p = PV, s = SOM, v = VIP)
- **Second letter**: source population

Example: `w_es` = weight from PYR (e) to SOM (s)

---

## Mathematical Formulation

### Rate Equation

Each population's firing rate r follows:

```
τ_s · dr/dt = -r + Φ(I_det) + σ_s · ξ(t)
```

Where:
- `τ_s`: Synaptic time constant (membrane + synaptic integration)
- `Φ`: Transfer function (Wong-Wang form)
- `I_det`: Deterministic input current
- `σ_s`: Noise amplitude
- `ξ(t)`: Gaussian white noise (or Ornstein-Uhlenbeck process)

### Wong-Wang Transfer Function

The model uses the Wong-Wang transfer function, derived from a spiking network mean-field reduction:

```
Φ(I) = u / (1 - exp(-g·u))

where: u = c · (I - θ)
```

**Parameters:**
- `θ` (Theta): Threshold current
- `c` (alpha): Gain/slope parameter
- `g`: Curvature parameter (g_e for excitatory, g_i for inhibitory)

**Properties:**
- Monotonically increasing
- Bounded below at 0
- Approximately linear for small inputs
- Saturates for large inputs
- Reduces to ReLU-like behavior when g → ∞

**Origin:** This form arises from mean-field analysis of integrate-and-fire networks (Wong & Wang, 2006, J. Neurosci.).

### Input Currents

#### PYR (Pyramidal) Input:
```
I_PYR = (w_ee · r_PYR) / (1 + g_GABA · w_pe · r_PV)
        - g_GABA · w_se · r_SOM
        - I_adapt_PYR
        + I_ext_PYR
```

The divisive term `(1 + g_GABA · w_pe · r_PV)` implements **shunting inhibition** from PV interneurons, modeling the effect of perisomatic GABAergic synapses on input resistance.

#### PV (Parvalbumin) Input:
```
I_PV = w_ep · r_PYR
       - g_GABA · w_pp · r_PV
       - g_GABA · w_sp · r_SOM
       - w_vp · r_VIP
       + I_ext_PV
```

#### SOM (Somatostatin) Input:
```
I_SOM = w_es · r_PYR
        - g_GABA · w_ps · r_PV
        - w_vs · r_VIP
        - I_adapt_SOM
        + I_ext_SOM
```

#### VIP Input:
```
I_VIP = w_ev · r_PYR
        - w_vv · r_VIP
        + I_ext_VIP
```

### Spike-Frequency Adaptation

PYR and SOM populations exhibit spike-frequency adaptation:

```
τ_adapt · dI_adapt/dt = -I_adapt + J_adapt · r
```

Where:
- `τ_adapt`: Adaptation time constant (slower than synaptic)
- `J_adapt`: Adaptation strength
- `r`: Current firing rate

Adaptation provides negative feedback that:
- Prevents runaway excitation
- Creates bistability (UP/DOWN states)
- Shapes temporal dynamics

---

## Nicotinic Receptor Modulation

The model includes modulation by three nicotinic acetylcholine receptor (nAChR) subtypes:

### α7 nAChR (act_alpha7)

**Location:** Interneurons (PV, SOM)

**Effects:**
1. Increases external current to PV: `I_alpha7_pv`
2. Increases external current to SOM: `I_alpha7_som`
3. Enhances GABAergic transmission: `g_alpha7` (adds to `g_gaba_base`)

**Mechanism:** α7 receptors are high-affinity, fast-desensitizing homomeric receptors. They enhance interneuron excitability and GABA release.

**Knockout (act_alpha7 = 0):**
- Removes receptor-mediated currents to PV/SOM
- Reduces GABA scaling (g_alpha7 → 0)
- Typically increases PYR firing due to reduced inhibition

### β2 nAChR (act_beta2)

**Location:** SOM interneurons

**Effects:**
1. Increases external current to SOM: `I_beta2_som`

**Mechanism:** β2-containing receptors (typically α4β2) are high-affinity, slowly desensitizing heteromeric receptors on SOM cells.

**Knockout (act_beta2 = 0):**
- Removes current to SOM
- Reduces SOM activity
- May increase PYR firing via reduced dendritic inhibition

### α5 nAChR (act_alpha5)

**Location:** VIP interneurons

**Effects:**
1. Increases external current to VIP: `I_alpha5_vip`

**Mechanism:** α5 subunits (often in α4β2α5 receptors) modulate VIP interneuron excitability.

**Knockout (act_alpha5 = 0):**
- Removes current to VIP
- Reduces VIP activity
- May increase SOM activity (less disinhibition)
- Complex effect on PYR depending on circuit state

---

## Code Structure

The code is organized as a unified Python package:

```
circuit_model/
├── __init__.py          # Public API exports
├── __main__.py          # Entry point: python -m circuit_model
├── params.py            # CircuitParams, ParamBound, default_bounds()
├── transfer.py          # phi_wong_wang() transfer function
├── simulation.py        # simulate_circuit(), mean_rates()
├── loss.py              # TargetRates, FitConfig, loss functions
├── optimization.py      # nevergrad_optimize(), evaluate_params()
├── study.py             # Batch study across 8 experimental conditions
├── plotting.py          # Visualization (dashboard, box plots)
├── io.py                # JSON I/O, output_dir()
├── cli.py               # Unified CLI (run, optimize, study, ring-run, ring-study, ring-oscillation-study)
│
└── ring/                # Ring attractor subpackage
    ├── __init__.py      # Ring API exports
    ├── params.py        # RingParams (network geometry)
    ├── connectivity.py  # Weight matrices (PYR-PYR, PV-PYR)
    ├── stimulus.py      # RingStimulus, WorkingMemoryProtocol
    ├── simulation.py    # simulate_ring(), RingSimulationResult
    ├── analysis.py      # Bump decoding, drift, diffusion metrics
    ├── plotting.py      # Ring-specific visualization
    └── cli.py           # Ring CLI logic (cmd_run, cmd_study)

docs/
├── CLI.md               # Full CLI reference with parameter tables
└── ring_attractor.md    # Mathematical formulation of the ring model

tests/
└── test_ring.py         # Ring attractor tests
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m circuit_model run` | Single circuit simulation with plotting |
| `python -m circuit_model optimize` | Nevergrad parameter optimization |
| `python -m circuit_model study` | Batch study across 8 conditions |
| `python -m circuit_model ring-run` | Ring attractor single-condition simulation |
| `python -m circuit_model ring-study` | Ring attractor multi-condition comparison |
| `python -m circuit_model ring-oscillation-study` | Cue-only oscillation analysis (dominant 2-12 Hz) |

See [docs/CLI.md](docs/CLI.md) for full parameter documentation.

---

## Parameter Reference

### Time Constants

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `tau_s` | 37.35 | ms | Synaptic/membrane time constant for all populations |
| `tau_adapt_pyr` | 186.6 | ms | Adaptation time constant for pyramidal cells |
| `tau_adapt_som` | 2320.5 | ms | Adaptation time constant for SOM cells (much slower) |

### Adaptation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `J_adapt_pyr` | 0.27 | Adaptation strength for PYR (self-inhibition from spiking) |
| `J_adapt_som` | 27.24 | Adaptation strength for SOM (stronger due to slower kinetics) |

### Noise

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_s` | 5.89 | Noise amplitude (standard deviation of Gaussian noise) |

### GABA Scaling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_gaba_base` | 3.93 | Baseline GABA scaling factor |
| `g_alpha7` | 0.96 | α7-receptor-dependent GABA enhancement |

Total GABA scaling: `g_gaba = g_gaba_base + g_alpha7`

### Synaptic Weights

#### Excitatory Connections (from PYR)

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `w_ee` | 6.27 | PYR → PYR | Recurrent excitation; maintains persistent activity |
| `w_ep` | 42.53 | PYR → PV | Drives feedback inhibition |
| `w_es` | 6.57 | PYR → SOM | Recruits dendritic inhibition |
| `w_ev` | 2.96e-6 | PYR → VIP | Weak; VIP driven more by other inputs |

#### Inhibitory Connections

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `w_pe` | 2.22 | PV → PYR | Perisomatic inhibition (divisive) |
| `w_se` | 2.62 | SOM → PYR | Dendritic inhibition (subtractive) |
| `w_pp` | 105.44 | PV → PV | Self-inhibition; limits PV firing |
| `w_sp` | 6.13e-6 | SOM → PV | Weak cross-inhibition |
| `w_vp` | 0.011 | VIP → PV | Weak disinhibition of PV |
| `w_ps` | 2.22 | PV → SOM | Cross-inhibition between interneuron types |
| `w_vs` | 1.27 | VIP → SOM | Core disinhibition pathway |
| `w_vv` | 24.80 | VIP → VIP | Self-inhibition; regulates VIP activity |

### External Currents

#### Baseline Currents

| Parameter | Default | Target | Description |
|-----------|---------|--------|-------------|
| `I0_pyr` | 1.79 | PYR | Tonic drive to pyramidal cells |
| `I_trans` | 5.04 | PYR | Transient/task-related input |
| `I0_pv` | 5.58 | PV | Tonic drive to PV cells |
| `I0_som` | 5.49 | SOM | Tonic drive to SOM cells |
| `I0_vip` | 7.57 | VIP | Tonic drive to VIP cells |

#### Receptor-Mediated Currents

| Parameter | Default | Target | Receptor | Description |
|-----------|---------|--------|----------|-------------|
| `I_alpha7_pv` | 9.90 | PV | α7 nAChR | Cholinergic enhancement of PV |
| `I_alpha7_som` | 5.85 | SOM | α7 nAChR | Cholinergic enhancement of SOM |
| `I_beta2_som` | 9.06 | SOM | β2 nAChR | β2-mediated SOM activation |
| `I_alpha5_vip` | 1.45 | VIP | α5 nAChR | Cholinergic modulation of VIP |

### Receptor Activation Multipliers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `act_alpha7` | 1.0 | α7 receptor activation (0 = knockout) |
| `act_beta2` | 1.0 | β2 receptor activation (0 = knockout) |
| `act_alpha5` | 1.0 | α5 receptor activation (0 = knockout) |

### Transfer Function Parameters

| Parameter | Default | Population | Description |
|-----------|---------|------------|-------------|
| `Theta_pyr` | 5.02 | PYR | Threshold current |
| `alpha_pyr` | 0.69 | PYR | Gain/slope |
| `Theta_pv` | 16.38 | PV | Threshold current (higher = less excitable) |
| `alpha_pv` | 1.48 | PV | Gain/slope |
| `Theta_som` | 5.88 | SOM | Threshold current |
| `alpha_som` | 0.82 | SOM | Gain/slope |
| `Theta_vip` | 13.91 | VIP | Threshold current |
| `alpha_vip` | 0.10 | VIP | Gain/slope (low = gradual response) |
| `g_e` | 0.38 | PYR | Curvature for excitatory cells |
| `g_i` | 0.40 | PV,SOM,VIP | Curvature for inhibitory cells |

---

## Quick Start

### Run a simulation

```bash
python -m circuit_model run
python -m circuit_model run --params_json my_params.json --noise_type ou
```

### Optimize parameters

```bash
python -m circuit_model optimize \
    --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 \
    --n_samples 5000
```

### Batch study across conditions

```bash
python -m circuit_model study --n_runs 50 --noise_type white
```

### Ring attractor simulation

```bash
# Single condition
python -m circuit_model ring-run --condition WT --amplitude 150

# Multi-condition comparison
python -m circuit_model ring-study --conditions WT WT_APP a7_KO --n_trials 10
```

---

## References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Pfeffer, C. K., Xue, M., He, M., Bhattacharyya, A., & Bhattacharyya, S. (2013). Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons. *Nature Neuroscience*, 16(8), 1068-1076.

3. Pi, H.-J., Hangya, B., Kvitsiani, D., Sanders, J. I., Huang, Z. J., & Bhattacharyya, A. (2013). Cortical interneurons that specialize in disinhibitory control. *Nature*, 503(7477), 521-524.
