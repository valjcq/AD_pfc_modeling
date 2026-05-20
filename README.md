# PFC Circuit Model: 4-Population Rate Model with Ring Attractor

A computational model of the prefrontal cortex (PFC) microcircuit implementing a 4-population rate model with Nevergrad-based parameter optimization and a ring attractor network for spatial working memory.

**Documentation:**
- [CLI Reference](docs/CLI.md) — all commands and parameters
- [Ring Attractor Model](docs/ring_attractor.md) — mathematical formulation of the ring network
- [Ring Experiments](docs/ring_experiments.md) — analysis protocols and metrics
- [Bistable Loss Guide](docs/bistable_loss_guide.md) — single-node bistable optimizer
- [Loss Math Deep Dive](docs/loss_math_deep_dive.md) — step-by-step derivation of every loss term
- [Transfer Function Ceiling](docs/transfer_function_ceiling.md) — NMDA gating and interneuron soft caps
- [Self-Consistent Interneuron Solve](docs/interneuron_selfconsistent_solve.md) — how the nullcline solver works
- [Single-Node → Ring Conversion](docs/sing_node_fit_to_ring.md) — row-sum derivation of ring kernels

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
              PYR        PV      SOM     VIP
          ┌──────────────────────────────────┐
   T  PYR │ J_NMDA·S*   w_pe    w_se    —    │  (NMDA recurrence ÷ PV shunting − SOM dendritic)
   O  PV  │ w_ep        w_pp    w_sp    w_vp │  (excitation from PYR, mutual inhibition)
      SOM │ w_es        —       —       w_vs │  (excitation from PYR, inhibition from VIP)
      VIP │ w_ev        —       —       —    │  (weak excitation from PYR, no local inhibition)
          └──────────────────────────────────┘
```

PV→PYR enters the PYR input *divisively* (shunting) rather than as a subtraction; see the input-current equations below.

### Notation Convention
- **First letter**: target population (e = excitatory/PYR, p = PV, s = SOM, v = VIP)
- **Second letter**: source population

Example: `w_es` = weight from PYR (e) to SOM (s). `J_NMDA` replaces the legacy `w_ee` (a one-shot JSON migration in `io.load_params_json` rescales any old `w_ee` field by ×10; see [io.py](circuit_model/io.py)).

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

All six constants are **fixed from W&W 2006** and not optimised: `alpha_pyr = 310 Hz/nA`, `Theta_pyr = 125/310 ≈ 0.403 nA`, `g_exc = 0.16 s`, `alpha_{pv,som,vip} = 615 Hz/nA`, `Theta_{pv,som,vip} = 177/615 ≈ 0.288 nA`, `g_inh = 0.087 s`.

### Hyperbolic Soft Ceiling (PV, SOM, VIP)

To prevent pathological interneuron over-activation in extreme parameter regimes, PV, SOM, and VIP apply a soft ceiling to the Wong-Wang output:

```
Φ_cap(I) = r_max · Φ(I) / (r_max + Φ(I))
```

With `Φ ≪ r_max` the curve is unchanged; as `Φ → ∞` the rate asymptotes to `r_max`. The ceilings are 2 × the Rooy (2021) active-state targets: `r_max_PV = 70.6`, `r_max_SOM = 70.4`, `r_max_VIP = 137.6` Hz. PYR is **uncapped** — its saturation comes from NMDA gating below. See [docs/transfer_function_ceiling.md](docs/transfer_function_ceiling.md).

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
        - I_adapt_PYR
        + I_ext_PYR
        + sigma_noise · I_ext_PYR · ξ(t)
```

The divisive denominator implements **PV shunting inhibition** (perisomatic GABAergic synapses act on input resistance).

#### PV (Parvalbumin):
```
I_PV = w_ep · r_PYR
       - g_GABA · w_pp · r_PV
       - g_GABA · w_sp · r_SOM
       - w_vp · r_VIP
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
        + I_ext_VIP
        + sigma_noise · I_ext_VIP · ξ(t)
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
├── __init__.py             # Public API exports
├── __main__.py             # Entry point: python -m circuit_model
├── constants.py            # R_MAX_PHYS, NMDA kinetics, interneuron ceilings
├── defaults.py             # Default WT / WT_APP parameter JSON paths
├── params.py               # CircuitParams, ParamBound, default_bounds()
├── transfer.py             # phi_wong_wang(), phi_capped()
├── simulation.py           # simulate_circuit(), mean_rates(), validate_fast_loop()
├── _fast_loop.py           # Numba-compiled Euler integrator for the single node
├── loss.py                 # TargetRates, FitConfig, loss functions
├── jacobian.py             # Effective-gain Jacobian + sanity-check report
├── optimization.py         # nevergrad_optimize(), evaluate_params(), LossBreakdown
├── bistable_loss.py        # Nullcline-based bistable loss (single-node mode)
├── random_search.py        # Parallel random search for bistable parameter sets
├── study.py                # Batch study across 8 experimental conditions
├── diagnostic.py           # Analytical Turing gain + transfer-function plots
├── plotting.py             # Visualization (dashboard, box plots)
├── loss_evolution_plot.py  # Live loss-curve plots during optimization
├── io.py                   # JSON I/O, fit-summary writers, output_dir()
├── cli.py                  # Unified single-node + study CLI
│
└── ring/                   # Ring attractor subpackage
    ├── __init__.py
    ├── constants.py        # Ring-only constants (e.g. TRANSIENT_SKIP_TIME_MS)
    ├── params.py           # RingParams (n_nodes, sigma_pyr_deg, sigma_som_deg, som_pattern)
    ├── connectivity.py     # PYR→PYR, PV→PYR, SOM→PYR kernels (row-sum normalised)
    ├── stimulus.py         # RingStimulus, WorkingMemoryProtocol
    ├── simulation.py       # simulate_ring(), simulate_ring_batch()
    ├── _fast_ring_loop.py  # Numba-compiled ring Euler integrator
    ├── analysis.py         # Bump decoding, drift, diffusion, asymmetry metrics
    ├── optimization.py     # Joint ring + circuit optimization (RingFitConfig, BumpTarget)
    ├── plotting.py         # Ring-specific visualization
    └── cli.py              # Ring CLI logic for all ring-* commands

docs/                       # See the "Documentation" links at the top of this README
tests/
└── test_ring.py            # Ring attractor tests (28 tests)
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m circuit_model plot-transfer` | Plot Φ(I) for all 4 populations |
| `python -m circuit_model diagnostic` | Analytical Turing gain + transfer-function dashboard |
| `python -m circuit_model run` | Single 4-population simulation with plotting |
| `python -m circuit_model optimize` | Nevergrad parameter optimization (rate / KO / bistable modes) |
| `python -m circuit_model study` | Batch study across 8 experimental conditions |
| `python -m circuit_model random-bistable-search` | Parallel random search for bistable parameter sets |
| `python -m circuit_model ring-run` | Ring attractor single-condition simulation |
| `python -m circuit_model ring-calibrate` | 3D parameter sweep (w_pv_global × w_pyr_pyr_inter × amplitude) |
| `python -m circuit_model ring-bump-decay-study` | Is a post-cue bump a sustained attractor or a transient? |
| `python -m circuit_model ring-optimize` | Joint optimization of CircuitParams + RingParams |

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

Total GABA scaling: `g_gaba = g_gaba_base + g_alpha7`.

### Synaptic Weights

All weights are in nA/Hz (weight × rate → nA of input current). Dataclass defaults are intentionally uniform (`0.002`) as a flat optimisation start; fitted values vary by 1–2 orders of magnitude across connections.

#### Excitatory Connections (from PYR)

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `J_NMDA` | 0.3 | PYR → PYR (NMDA-gated) | Recurrent NMDA coupling; replaces legacy `w_ee` |
| `w_ep` | 0.002 | PYR → PV | Drives feedback inhibition |
| `w_es` | 0.002 | PYR → SOM | Recruits dendritic inhibition |
| `w_ev` | 0.002 | PYR → VIP | Disinhibitory drive |

#### Inhibitory Connections

| Parameter | Default | Connection | Biological Role |
|-----------|---------|------------|-----------------|
| `w_pe` | 0.002 | PV → PYR | Perisomatic inhibition (divisive / shunting) |
| `w_se` | 0.002 | SOM → PYR | Dendritic inhibition (subtractive) |
| `w_pp` | 0.002 | PV → PV | Self-inhibition |
| `w_sp` | 0.002 | SOM → PV | Cross-inhibition |
| `w_vp` | 0.002 | VIP → PV | Weak disinhibition of PV |
| `w_vs` | 0.002 | VIP → SOM | Core disinhibition pathway (VIP→SOM→PYR) |

### External Currents (nA)

#### Baseline Currents

| Parameter | Default | Target | Description |
|-----------|---------|--------|-------------|
| `I0_pyr` | 0.44 | PYR | Baseline tonic drive |
| `I0_pv` | 0.35 | PV | Baseline tonic drive |
| `I0_som` | 0.35 | SOM | Baseline tonic drive |
| `I0_vip` | 0.35 | VIP | Baseline tonic drive |

#### Receptor-Mediated Currents

| Parameter | Default | Target | Receptor | Description |
|-----------|---------|--------|----------|-------------|
| `I_alpha7_pv` | 0.20 | PV | α7 nAChR | Cholinergic enhancement of PV |
| `I_alpha7_som` | 0.20 | SOM | α7 nAChR | Cholinergic enhancement of SOM |
| `I_beta2_som` | 0.20 | SOM | β2 nAChR | β2-mediated SOM activation |
| `I_alpha5_vip` | 0.20 | VIP | α5 nAChR | Cholinergic modulation of VIP |

#### Transient currents

`trans_*` and `trans2_*` fields define an optional **PYR-only** transient input (a square pulse of magnitude `trans_factor × I0_pyr` over `[trans_start_ms, trans_start_ms + trans_duration_ms)`). PV/SOM/VIP external currents are unaffected. Two independent transients are available; both default to disabled.

### Receptor Activation Multipliers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `act_alpha7` | 1.0 | α7 receptor activation (0 = knockout, intermediate values = partial blockade) |
| `act_beta2` | 1.0 | β2 receptor activation |
| `act_alpha5` | 1.0 | α5 receptor activation |

### Transfer Function Parameters (Wong & Wang 2006, fixed)

These six constants are **not optimised**:

| Parameter | Default | Population | Description |
|-----------|---------|------------|-------------|
| `alpha_pyr` | 310.0 Hz/nA | PYR | Gain (`c_e`) |
| `Theta_pyr` | 125/310 ≈ 0.403 nA | PYR | Threshold |
| `g_exc` | 0.16 s | PYR | Curvature |
| `alpha_pv` / `alpha_som` / `alpha_vip` | 615.0 Hz/nA | PV, SOM, VIP | Gain (`c_i`) |
| `Theta_pv` / `Theta_som` / `Theta_vip` | 177/615 ≈ 0.288 nA | PV, SOM, VIP | Threshold |
| `g_inh` | 0.087 s | PV, SOM, VIP | Curvature |

Interneuron soft ceilings (constants, not in `CircuitParams`): `R_MAX_PV = 70.6`, `R_MAX_SOM = 70.4`, `R_MAX_VIP = 137.6` Hz (= 2 × Rooy 2021 high-state targets).

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
# Standard rate-matching fit with KO targets, chained DE → Nelder-Mead
python -m circuit_model optimize \
    --target_pyr 4.143 --target_som 3.423 --target_pv 2.079 --target_vip 1.933 \
    --target_alpha7_ko_pyr 3.513 --target_beta2_ko_pyr 4.8 --target_alpha5_ko_pyr 3.79 \
    --optimizer chaining --n_samples 50000 --n_workers 4 \
    --save_best_json params/new/WT_1mo.json
```

### Batch study across conditions

```bash
python -m circuit_model study --n_trials 50 --noise_type white
```

### Ring attractor simulation

```bash
# Single condition (defaults pull params/ring params from the project WT fit)
python -m circuit_model ring-run --condition WT

# Bump-decay analysis across conditions and cue amplitudes
python -m circuit_model ring-bump-decay-study \
    --conditions WT WT_APP --amplitudes 10 20 30 --n_trials 50 --no_show

# Joint circuit + ring optimization to match resting firing-rate targets
python -m circuit_model ring-optimize \
    --target_pyr 8 --target_som 5 --target_pv 3 --target_vip 2 \
    --n_samples 5000
```

### Verify the fast/reference integrators agree

```python
from circuit_model import validate_fast_loop
validate_fast_loop()   # prints "OK — bit-identical" on success
```

### Tests

```bash
python -m pytest tests/ -q
```

---

## References

1. Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

2. Pfeffer, C. K., Xue, M., He, M., Bhattacharyya, A., & Bhattacharyya, S. (2013). Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons. *Nature Neuroscience*, 16(8), 1068-1076.

3. Pi, H.-J., Hangya, B., Kvitsiani, D., Sanders, J. I., Huang, Z. J., & Bhattacharyya, A. (2013). Cortical interneurons that specialize in disinhibitory control. *Nature*, 503(7477), 521-524.
