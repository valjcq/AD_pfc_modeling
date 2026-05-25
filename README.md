# PFC Circuit Model: 5-Population Rate Model (NDNF branch)

A single-node rate model of the prefrontal cortex (PFC) microcircuit with five populations (PYR, SOM, PV, VIP, NDNF) and Nevergrad-based parameter optimization.

**Documentation:**
- [CLI Reference](docs/CLI.md) — all commands and parameters
- [Loss Math Deep Dive](docs/loss_math_deep_dive.md) — step-by-step derivation of every loss term
- [Transfer Function Ceiling](docs/transfer_function_ceiling.md) — NMDA gating and interneuron soft caps

---

## Table of Contents

1. [Biological Background](#biological-background)
2. [Model Architecture](#model-architecture)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Nicotinic Receptor Modulation](#nicotinic-receptor-modulation)
5. [Code Structure](#code-structure)
6. [Parameter Reference](#parameter-reference)
7. [Quick Start](#quick-start)
8. [References](#references)

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
| **NDNF** | NDNF interneurons | Neuron-derived neurotrophic factor | 1 | Subtractive dendritic inhibition (like SOM); expresses α7 + β2 nAChRs; receives PYR + SOM; projects to PYR dendrites, PV, VIP |

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

The model implements the following synaptic weight matrix (`w_XY` = weight from population Y to population X). PYR self-excitation is NMDA-gated via the saturable variable `S^*` and carries the scalar `J_NMDA` instead of a linear `w_ee`:

```
                              FROM
              PYR        PV      SOM     VIP    NDNF
          ┌────────────────────────────────────────────┐
   T  PYR │ J_NMDA·S*   w_pe    w_se    —      w_en    │  (NMDA ÷ PV shunting − SOM/NDNF dendritic)
   O  PV  │ w_ep        w_pp    w_sp    w_vp   w_pn    │  (PYR drive, mutual + NDNF inhibition)
      SOM │ w_es        —       —       w_vs   —       │  (PYR drive, VIP inhibition)
      VIP │ w_ev        —       —       —      w_vn    │  (PYR drive, NDNF inhibition)
      NDNF│ w_ne        —       w_ns    —      —       │  (PYR drive, SOM inhibition)
          └────────────────────────────────────────────┘
```

PV→PYR enters the PYR input *divisively* (shunting) rather than as a subtraction. All other inhibitory connections are subtractive and scaled by `g_gaba`.

### Notation Convention
- **First letter**: target population (e = excitatory/PYR, p = PV, s = SOM, v = VIP, n = NDNF)
- **Second letter**: source population

Example: `w_es` = weight from PYR (e) to SOM (s); `w_en` = weight from NDNF (n) to PYR (e). `J_NMDA` replaces the legacy `w_ee` (a one-shot JSON migration in `io.load_params_json` rescales any old `w_ee` field by ×10; see [io.py](circuit_model/io.py)).

---

## Mathematical Formulation

### Rate Equation

Each population's firing rate r follows:

```
τ_s · dr/dt = -r + Φ(I_pop)
```

where `I_pop` is the population-specific input current (defined below). Stochastic noise is injected **in current-space**: when `noise_type != "none"`, each `I_pop` carries an additional term `sigma_noise · I_ext_pop · ξ(t)`, where `ξ(t)` is a shared Gaussian (white) or Ornstein-Uhlenbeck process and each population's amplitude is scaled by its own baseline drive. This is the diffusion-approximation form of Poisson input variability.

### Wong-Wang Transfer Function (PYR)

PYR uses the Wong-Wang transfer function (Wong & Wang, 2006), derived from a spiking-network mean-field reduction:

```
Φ(I) = u / (1 - exp(-g·u)),   u = c · (I - θ)
```

Parameters:
- `θ` (Theta): threshold current
- `c` (alpha): gain/slope parameter
- `g`: curvature parameter (`g_exc` for PYR, `g_inh` for interneurons)

All six constants are **fixed from W&W 2006** and not optimised: `alpha_pyr = 310 Hz/nA`, `Theta_pyr = 125/310 ≈ 0.403 nA`, `g_exc = 0.16 s`, `alpha_{pv,som,vip,ndnf} = 615 Hz/nA`, `Theta_{pv,som,vip,ndnf} = 177/615 ≈ 0.288 nA`, `g_inh = 0.087 s`.

### Hyperbolic Soft Ceiling (PV, SOM, VIP, NDNF)

To prevent pathological interneuron over-activation in extreme parameter regimes, all four interneurons apply a soft ceiling to the Wong-Wang output:

```
Φ_cap(I) = r_max · Φ(I) / (r_max + Φ(I))
```

With `Φ ≪ r_max` the curve is unchanged; as `Φ → ∞` the rate asymptotes to `r_max`. The ceilings are 2 × the Rooy (2021) active-state targets: `r_max_PV = 70.6`, `r_max_SOM = 70.4`, `r_max_VIP = 137.6`, `r_max_NDNF = 70.0` Hz (NDNF is a placeholder, TODO: refine). PYR is **uncapped** — its saturation comes from NMDA gating below. See [docs/transfer_function_ceiling.md](docs/transfer_function_ceiling.md).

### NMDA Gating Variable (PYR self-excitation)

PYR→PYR recurrence is mediated by NMDA receptors with a saturable gating variable `S ∈ [0, 1]`:

```
τ_NMDA · dS/dt = -S + γ · (1 - S) · r_PYR
```

with fixed kinetics `τ_NMDA = 100 ms`, `γ = 0.641` (Wong & Wang 2006). At steady state:

```
S* = γ · τ_NMDA · r_PYR / (1 + γ · τ_NMDA · r_PYR)
```

`S*` saturates as `r_PYR` grows, which is what folds the PYR nullcline and enables single-node bistability. The fitted scalar `J_NMDA` is the recurrent NMDA coupling strength; it replaces the linear `w_ee · r_PYR` of earlier model versions.

### Input Currents

#### PYR (Pyramidal):
```
I_PYR = (J_NMDA · S*) / (1 + g_GABA · w_pe · r_PV)
        - g_GABA · w_se · r_SOM
        - g_GABA · w_en · r_NDNF
        - I_adapt_PYR
        + I_ext_PYR
        + sigma_noise · I_ext_PYR · ξ(t)
```

The divisive denominator implements **PV shunting inhibition** (perisomatic GABAergic synapses act on input resistance). Both SOM and NDNF deliver subtractive dendritic inhibition.

#### PV (Parvalbumin):
```
I_PV = w_ep · r_PYR
       - g_GABA · w_pp · r_PV
       - g_GABA · w_sp · r_SOM
       - w_vp · r_VIP
       - g_GABA · w_pn · r_NDNF
       + I_ext_PV
       + sigma_noise · I_ext_PV · ξ(t)
```

#### SOM (Somatostatin):
```
I_SOM = w_es · r_PYR
        - w_vs · r_VIP
        - I_adapt_SOM
        + I_ext_SOM
        + sigma_noise · I_ext_SOM · ξ(t)
```

#### VIP:
```
I_VIP = w_ev · r_PYR
        - g_GABA · w_vn · r_NDNF
        + I_ext_VIP
        + sigma_noise · I_ext_VIP · ξ(t)
```

#### NDNF:
```
I_NDNF = w_ne · r_PYR
         - g_GABA · w_ns · r_SOM
         + I_ext_NDNF
         + sigma_noise · I_ext_NDNF · ξ(t)
```

### Spike-Frequency Adaptation

PYR (and optionally SOM) exhibit spike-frequency adaptation:

```
τ_adapt · dI_adapt/dt = -I_adapt + J_adapt · r
```

Adaptation provides slow negative feedback that prevents runaway excitation and shapes the temporal envelope of cue-driven bumps. SOM adaptation (`J_adapt_som`) is **off by default** (`J_adapt_som = 0`); the thesis model uses PYR adaptation only.

---

## Nicotinic Receptor Modulation

The model includes modulation by three nicotinic acetylcholine receptor (nAChR) subtypes:

### α7 nAChR — per-cell activation

**Location:** PV, SOM, and NDNF interneurons. Each has its own activation multiplier so cell-type-selective α7 knockouts are possible.

| Parameter | Effect |
|-----------|--------|
| `act_alpha7_pv`   | Gates `I_alpha7_pv` on PV |
| `act_alpha7_som`  | Gates `I_alpha7_som` on SOM |
| `act_alpha7_ndnf` | Gates `I_alpha7_ndnf` on NDNF |

GABA scaling: `g_gaba = g_gaba_base + g_alpha7 · mean(act_alpha7_pv, act_alpha7_som, act_alpha7_ndnf)`. Global α7-KO ⇒ all three at 0 ⇒ g_alpha7 term vanishes; a single-cell-type α7-KO reduces it by ~1/3.

**Knockouts simulated by the optimizer:**
- Global α7-KO  (all three per-cell α7 = 0)
- NDNF-selective α7-KO (`act_alpha7_ndnf = 0` only)
- PV-selective   α7-KO (`act_alpha7_pv   = 0` only)

### β2 nAChR (act_beta2)

**Location:** SOM and NDNF interneurons.

**Effects:**
1. Increases external current to SOM: `I_beta2_som`
2. Increases external current to NDNF: `I_beta2_ndnf`

**Mechanism:** β2-containing receptors (typically α4β2) are high-affinity, slowly desensitizing heteromeric receptors.

**Knockout (act_beta2 = 0):**
- Removes β2 currents to SOM and NDNF
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
├── __init__.py             # Public API exports
├── __main__.py             # Entry point: python -m circuit_model
├── constants.py            # R_MAX_PHYS, NMDA kinetics, interneuron ceilings (R_MAX_*)
├── defaults.py             # Default WT / WT_APP parameter JSON paths
├── params.py               # CircuitParams (5-pop), ParamBound, default_bounds()
├── transfer.py             # phi_wong_wang(), phi_capped()
├── simulation.py           # simulate_circuit(), mean_rates(), validate_fast_loop()
├── _fast_loop.py           # Numba-compiled 5-pop Euler integrator
├── loss.py                 # TargetRates, FitConfig, loss functions
├── jacobian.py             # 5×5 effective-gain Jacobian + sanity-check report
├── optimization.py         # nevergrad_optimize(), evaluate_params(), LossBreakdown
├── study.py                # Batch study across experimental conditions
├── plotting.py             # Visualization (dashboard, box plots, transfer functions)
├── loss_evolution_plot.py  # Live loss-curve plots during optimization
├── io.py                   # JSON I/O, fit-summary writers, output_dir()
└── cli.py                  # CLI: run / optimize / study / plot-transfer

docs/                       # See the "Documentation" links at the top of this README
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m circuit_model plot-transfer` | Plot Φ(I) for all 5 populations |
| `python -m circuit_model run` | Single 5-population simulation with plotting |
| `python -m circuit_model optimize` | Nevergrad parameter optimization (baseline rates + KOs) |
| `python -m circuit_model study` | Batch study across experimental conditions |

See [docs/CLI.md](docs/CLI.md) for the full parameter documentation.

---

## Parameter Reference

These are the **dataclass defaults** in [`circuit_model/params.py`](circuit_model/params.py) — the starting point for optimization. Fitted JSON files (e.g. `params/new/ring_firing_rate/WT_1mo_article_ko.json`) override most of them.

### Time Constants

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `tau_s` | 20.0 | ms | Synaptic/membrane time constant (all populations; fixed, not optimised) |
| `tau_adapt_pyr` | 600.0 | ms | PYR adaptation time constant |
| `tau_adapt_som` | 150.0 | ms | SOM adaptation time constant |

NMDA gating kinetics (fixed from W&W 2006, in `constants.py`): `τ_NMDA = 100 ms`, `γ_NMDA = 0.641`.

### Adaptation

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `J_adapt_pyr` | 0.002 | nA/Hz | PYR adaptation strength |
| `J_adapt_som` | 0.0 | nA/Hz | SOM adaptation strength (off by default) |

### Noise

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_noise` | 0.3 | Dimensionless. Per-population noise current std = `sigma_noise × I_ext_pop` (nA). |

### GABA Scaling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_gaba_base` | 1.0 | Baseline GABA scaling factor (dimensionless) |
| `g_alpha7` | 0.0 | α7-receptor-dependent GABA enhancement (fitted) |

Total GABA scaling: `g_gaba = g_gaba_base + g_alpha7 · mean(act_alpha7_pv, act_alpha7_som, act_alpha7_ndnf)`.

### Synaptic Weights

All weights are in nA/Hz (weight × rate → nA of input current). Dataclass defaults are intentionally uniform (`0.002`) as a flat optimisation start; fitted values vary by 1–2 orders of magnitude across connections.

#### Excitatory Connections (from PYR)

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `J_NMDA` | 0.3 | PYR → PYR (NMDA-gated) | Recurrent NMDA coupling; replaces legacy `w_ee` |
| `w_ep` | 0.002 | PYR → PV | Drives feedback inhibition |
| `w_es` | 0.002 | PYR → SOM | Recruits dendritic inhibition |
| `w_ev` | 0.002 | PYR → VIP | Disinhibitory drive |
| `w_ne` | 0.002 | PYR → NDNF | Excitatory drive to NDNF |

#### Inhibitory Connections

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `w_pe` | 0.002 | PV → PYR | Perisomatic inhibition (divisive / shunting) |
| `w_se` | 0.002 | SOM → PYR | Dendritic inhibition (subtractive) |
| `w_pp` | 0.002 | PV → PV | Self-inhibition |
| `w_sp` | 0.002 | SOM → PV | Cross-inhibition |
| `w_vp` | 0.002 | VIP → PV | Weak disinhibition of PV |
| `w_vs` | 0.002 | VIP → SOM | Core disinhibition pathway (VIP→SOM→PYR) |
| `w_ns` | 0.002 | SOM → NDNF | SOM gates NDNF activity |
| `w_en` | 0.002 | NDNF → PYR | Dendritic inhibition (subtractive, parallel to SOM) |
| `w_pn` | 0.002 | NDNF → PV | NDNF inhibition of PV |
| `w_vn` | 0.002 | NDNF → VIP | NDNF inhibition of VIP |

### External Currents (nA)

#### Baseline Currents

| Parameter | Default | Target | Description |
|-----------|---------|--------|-------------|
| `I0_pyr`  | 0.44 | PYR  | Baseline tonic drive |
| `I0_pv`   | 0.35 | PV   | Baseline tonic drive |
| `I0_som`  | 0.35 | SOM  | Baseline tonic drive |
| `I0_vip`  | 0.35 | VIP  | Baseline tonic drive |
| `I0_ndnf` | 0.35 | NDNF | Baseline tonic drive (**placeholder**, TODO: refine from literature) |

#### Receptor-Mediated Currents

| Parameter | Default | Target | Receptor | Description |
|-----------|---------|--------|----------|-------------|
| `I_alpha7_pv`   | 0.20 | PV   | α7 nAChR | Cholinergic enhancement of PV |
| `I_alpha7_som`  | 0.20 | SOM  | α7 nAChR | Cholinergic enhancement of SOM |
| `I_alpha7_ndnf` | 0.20 | NDNF | α7 nAChR | Cholinergic enhancement of NDNF |
| `I_beta2_som`   | 0.20 | SOM  | β2 nAChR | β2-mediated SOM activation |
| `I_beta2_ndnf`  | 0.20 | NDNF | β2 nAChR | β2-mediated NDNF activation |
| `I_alpha5_vip`  | 0.20 | VIP  | α5 nAChR | Cholinergic modulation of VIP |

#### Transient currents

`trans_*` and `trans2_*` fields define an optional **PYR-only** transient input (a square pulse of magnitude `trans_factor × I0_pyr` over `[trans_start_ms, trans_start_ms + trans_duration_ms)`). The other populations' external currents are unaffected. Two independent transients are available; both default to disabled.

### Receptor Activation Multipliers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `act_alpha7_pv`   | 1.0 | α7 activation on PV   (per-cell, 0 = selective KO) |
| `act_alpha7_som`  | 1.0 | α7 activation on SOM  (per-cell, 0 = selective KO) |
| `act_alpha7_ndnf` | 1.0 | α7 activation on NDNF (per-cell, 0 = selective KO) |
| `act_beta2`       | 1.0 | β2 activation (affects SOM and NDNF) |
| `act_alpha5`      | 1.0 | α5 activation (affects VIP) |

### Transfer Function Parameters (Wong & Wang 2006, fixed)

These constants are **not optimised**:

| Parameter | Default | Population | Description |
|-----------|---------|------------|-------------|
| `alpha_pyr` | 310.0 Hz/nA | PYR | Gain (`c_e`) |
| `Theta_pyr` | 125/310 ≈ 0.403 nA | PYR | Threshold |
| `g_exc` | 0.16 s | PYR | Curvature |
| `alpha_{pv,som,vip,ndnf}` | 615.0 Hz/nA | PV, SOM, VIP, NDNF | Gain (`c_i`) |
| `Theta_{pv,som,vip,ndnf}` | 177/615 ≈ 0.288 nA | PV, SOM, VIP, NDNF | Threshold |
| `g_inh` | 0.087 s | PV, SOM, VIP, NDNF | Curvature |

Interneuron soft ceilings (constants, not in `CircuitParams`): `R_MAX_PV = 70.6`, `R_MAX_SOM = 70.4`, `R_MAX_VIP = 137.6`, `R_MAX_NDNF = 70.0` Hz (NDNF placeholder).

---

## Quick Start

### Install

```bash
# from the repository root, with the project venv active
pip install -e .
```

`numba` is optional; if missing, the integrator falls back to a slower plain-Python loop (still functional).

### Run a simulation

```bash
# Default circuit (loads params/new/ring_firing_rate/WT_1mo_article_ko.json if available)
python -m circuit_model run

# With OU-correlated noise
python -m circuit_model run --noise_type ou

# Apply a knockout preset
python -m circuit_model run --condition a7_KO
```

### Optimize parameters

```bash
# Rate-matching fit (5 populations) + global KO targets on PYR + selective α7 KOs on NDNF/PV.
python -m circuit_model optimize \
    --target_pyr 1.7328 --target_som 1.3564 --target_pv 1.5281 --target_vip 2.9791 \
    --target_ndnf 2.5309 \
    --target_alpha7_ko_pyr 2.1928 --target_beta2_ko_pyr 1.0825 --target_alpha5_ko_pyr 0.4762 \
    --target_alpha7_ndnf_ko_ndnf 3.0767 --target_alpha7_pv_ko_pv 1.3966 \
    --optimizer twopointde --n_samples 50000 \
    --output_dir fits/WT_NDNF_5pop
```

### Batch study across conditions

```bash
python -m circuit_model study --n_trials 50 --noise_type white
```

### Verify the fast/reference integrators agree

```python
from circuit_model import validate_fast_loop
validate_fast_loop()   # prints "OK — bit-identical" on success
```

---

## References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Pfeffer, C. K., Xue, M., He, M., Bhattacharyya, A., & Bhattacharyya, S. (2013). Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons. *Nature Neuroscience*, 16(8), 1068-1076.

3. Pi, H.-J., Hangya, B., Kvitsiani, D., Sanders, J. I., Huang, Z. J., & Bhattacharyya, A. (2013). Cortical interneurons that specialize in disinhibitory control. *Nature*, 503(7477), 521-524.
