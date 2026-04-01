# Model Changes and Refinements

**Date:** 2026-03-19

This document summarizes three major modifications to the model:
1. **Removal of biologically unsupported local connections** (VIP → VIP, PV → SOM)
2. **Stochastic noise architecture redesign** (from rate-space to current-space injection)
3. **Transfer function parameterization** (W&W-grounded approach with explicit shape parameters)

---

## Table of Contents

- [Part 1: Removal of Biologically Unsupported Local Connections](#part-1-removal-of-biologically-unsupported-local-connections)
  - [VIP → VIP Self-Inhibition](#vip--vip-self-inhibition-w_vv)
  - [PV → SOM Cross-Inhibition](#pv--som-cross-inhibition-w_ps)
- [Part 2: Stochastic Noise Architecture](#part-2-stochastic-noise-architecture-shift-from-rate-space-to-current-space)
- [Part 3: Transfer Function W&W-Grounded Parameterisation](#part-3-transfer-function-ww-grounded-parameterisation)
  - [Background: The Provenance Problem](#background-the-provenance-problem)
  - [Transfer Function Definition](#transfer-function-definition)
  - [Parameter Values and Units](#parameter-values-and-units)
  - [Unit Consistency Check](#unit-consistency-check)
  - [Comparison with Koukouli et al. 2025](#comparison-with-koukouli-et-al-2025)
  - [References](#references)

---

# Part 1: Removal of Biologically Unsupported Local Connections

Two synaptic connections have been removed from the model because they are not supported by the biological literature on PFC microcircuit anatomy: the VIP → VIP self-inhibitory connection (`w_vv`) and the PV → SOM cross-inhibitory connection (`w_ps`). Both had been flagged in the original code comments as absent from the reference schematic diagram.

---

## VIP → VIP Self-Inhibition (`w_vv`)

### Biological Motivation

VIP interneurons implement the canonical disinhibitory motif by inhibiting SOM (and weakly PV) cells. However, **local VIP → VIP autoinhibition is not an established connection in the PFC microcircuit literature**. The previous implementation included it as a pragmatic stabilization device (preventing VIP runaway under strong external drive), but this was at odds with the schematic connectivity the model is meant to represent.

### Equation Change

**Before:**

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} - w_{vv}\,r_\text{VIP} + I_\text{ext}^\text{VIP}$$

**After:**

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} + I_\text{ext}^\text{VIP}$$

The self-inhibitory term $-w_{vv}\,r_\text{VIP}$ is removed. VIP activity is now regulated solely by the transfer function saturation and the external drive $I_\text{ext}^\text{VIP}$.

---

## PV → SOM Cross-Inhibition (`w_ps`)

### Biological Motivation

PV interneurons provide fast perisomatic inhibition primarily to pyramidal cells. **A direct PV → SOM inhibitory connection is not a canonical feature of the PFC microcircuit** and is absent from the reference schematic. The previous default value (`w_ps = 2.22`, identical to `w_pe`) was high enough to meaningfully suppress SOM activity, making it a significant — and unjustified — circuit element.

### Equation Change

**Before:**

$$I_\text{SOM} = w_{es}\,r_\text{PYR} - g_\text{GABA}\,w_{ps}\,r_\text{PV} - w_{vs}\,r_\text{VIP} - I_\text{adapt}^\text{SOM} + I_\text{ext}^\text{SOM}$$

**After:**

$$I_\text{SOM} = w_{es}\,r_\text{PYR} - w_{vs}\,r_\text{VIP} - I_\text{adapt}^\text{SOM} + I_\text{ext}^\text{SOM}$$

The cross-inhibitory term $-g_\text{GABA}\,w_{ps}\,r_\text{PV}$ is removed.

---

# Part 2: Stochastic Noise Architecture: Shift from Rate-Space to Current-Space

## Biological Motivation

The previous model injected noise directly into the rate equation of all four populations (additive noise in Hz), applied **outside** the transfer function. This means the noise bypassed the transfer function filtering that converts input current variability into firing rate variability. Additionally, the noise scaling ($\sigma_s$) was independent of the external drive level, making it difficult to tune across different circuit operating points.

The revised model injects noise **into the input current of all four populations** (PYR, SOM, PV, VIP) with a **shared noise factor** scaled by the external drive. This approach:
1. **Grounds noise in circuit anatomy**: variability in synaptic drive (Poisson spike trains, synaptic noise) is a primary source of stochasticity in cortical networks.
2. **Naturally filters noise through the transfer function**: the effective firing-rate noise depends on the local slope $\Phi'(I)$ of each population's transfer function, consistent with diffusion-approximation treatments of neural variability.
3. **Automatically scales across conditions**: by proportionality to $I_{\text{ext}}^{\text{PYR}}$, the noise amplitude adjusts based on external drive level without requiring manual re-tuning per experiment.
4. **Ensures population coherence**: the shared noise factor means all populations at a given node receive the same stochastic perturbation, creating correlated trial-to-trial variability across the local circuit.

## Equation Changes

### Before: Rate-space noise (all populations)

The rate equation for each population $x \in \{\text{PYR}, \text{PV}, \text{SOM}, \text{VIP}\}$ was:

$$\tau_s \frac{dr_x}{dt} = -r_x + \Phi^x(I_x^\text{syn}) + \sigma_s \, \xi_x(t)$$

where $\sigma_s$ (in Hz) was a **noise amplitude in rate space** applied uniformly to all populations.

**Problems with this formulation:**
- Noise is added to the rate **outside** the transfer function, so it bypasses the transfer function filtering that converts input current variability into firing rate variability.
- The noise scaling ($\sigma_s$) was independent of the external drive level, making it difficult to tune across different circuit operating points.

### After: Current-space noise (all populations with shared factor)

Noise is injected **into all four populations** (PYR, SOM, PV, VIP), at the input current level, using a **shared noise factor** at each node:

$$I_i^{X}(t) = I_i^{X,\text{det}}(t) + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t) \quad \text{for } X \in \{\text{PYR}, \text{SOM}, \text{PV}, \text{VIP}\}$$

where:
- $I_i^{X,\text{det}}$ is the deterministic input (all synaptic terms, adaptation, stimulus)
- $\sigma_{\text{noise}}$ is a dimensionless amplitude (default: 0.3)
- $I_{\text{ext}}^{\text{PYR}}$ is the baseline external drive to PYR (in nA) — used as the scaling reference for all populations
- $\xi_i(t)$ is the shared stochastic process (white noise, OU, or zero) applied equally to all populations at node $i$

The rate equation for each population becomes:

$$\tau_s \frac{dr_i^{X}}{dt} = -r_i^{X} + \Phi^{X}\bigl(I_i^{X,\text{det}} + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)\bigr) \quad \text{for } X \in \{\text{PYR}, \text{SOM}, \text{PV}, \text{VIP}\}$$

**Key difference from the previous approach:**
- All four populations receive the same instantaneous noise perturbation $\xi_i(t)$ at each node, ensuring **correlated stochasticity** across populations.
- However, each population experiences a different **effective firing-rate noise** due to its own transfer function slope $\Phi'_X(I)$, which varies depending on whether the population is excitatory (PYR) or inhibitory (PV, SOM, VIP), and where it currently operates on its f-I curve.

## Parameter Mapping

| Aspect | Old | New |
|--------|-----|-----|
| **Parameter name** | `sigma_s` | `sigma_noise` |
| **Units** | Hz | dimensionless |
| **Applies to** | All populations independently (PYR, PV, SOM, VIP) | All populations with shared factor |
| **Injection point** | Added to rate $r_x(t)$ | Added to current $I^{X}(t)$ for all $X$ |
| **Noise process** | Independent $\xi_x(t)$ per population | Shared $\xi_i(t)$ per node across all populations |
| **Scaling** | Constant, independent of drive | Proportional to $I_{\text{ext}}^{\text{PYR}}$ (reference drive) |
| **Noise processes supported** | White, OU, none | White, OU, none |

---

# Part 3: Transfer Function: W&W-Grounded Parameterisation

This part records the rationale and exact parameter choices for the transfer function used in
this model, following a redesign away from the Koukouli et al. 2025 convention toward a cleaner,
more directly W&W-grounded approach.

---

## Background: The Provenance Problem

The transfer function in Koukouli et al. 2025 is stated to use parameters
$c_e, c_i, I_e, I_i, g_e, g_i$ "derived from Wong & Wang (2006)". On inspection, this statement
is misleading in three ways:

1. **Collapsed threshold.** W&W 2006 yields distinct threshold currents for excitatory and
   inhibitory populations — $\Theta_E = I_e/c_e \approx 0.403\ \text{nA}$ vs
   $\Theta_I = I_i/c_i \approx 0.288\ \text{nA}$. Koukouli collapses both to a single value
   $\Theta = 7$ (dimensionless) applied to all four populations identically. This erases a
   meaningful biological distinction: pyramidal cells require more input current to begin firing
   than fast-spiking interneurons at the population level.

2. **Dimensionless $g$.** W&W's $g$ values ($g_e = 0.16\ \text{s}$, $g_i = 0.087\ \text{s}$)
   are in physical units and cannot be imported into a dimensionless model. We tried before with a g=1 dimensionless curvature parameter, but this doesn't take into account for the difference of curvature between excitatory and inhibitory populations.

The present model adopts a cleaner approach: use the **exact W&W functional form**, with the
**exact W&W parameter values**, applied per population class (E vs I), with a single free
**output-scaling factor $A_x$** per population. The only fitting targets are firing rates in Hz.

---

## Transfer Function Definition

### Functional Form (W&W 2006, Eq. 2 — unmodified)

$$\boxed{\Phi^x(I) = A_x \cdot \frac{c_x \, I - I_{0,x}}{1 - \exp\!\bigl[-g_x\,(c_x \, I - I_{0,x})\bigr]}}$$

With $u_x = c_x \, I - I_{0,x}$:

$$\Phi^x(I) = A_x \cdot \frac{u_x}{1 - e^{-g_x \, u_x}}$$

**Behaviour of the core function $u/(1-e^{-gu})$:**
- For $u \gg 0$ (well above threshold): $\approx u$ — approximately linear
- For $u \approx 0$ (near threshold): $\approx 1/g$ — smooth, finite
- For $u < 0$ (below threshold): negative, but network dynamics keep $r \geq 0$ via the $-r$ leak
  term in the rate equation
- Large $g$: sharp linear-threshold behaviour (ReLU-like)
- Small $g$: smooth sigmoid-like onset

The factor $A_x$ scales the overall output amplitude without changing the shape; it is the only
per-population free parameter in the transfer function.

---

## Parameter Values and Units

### W&W 2006 Shape Parameters (fixed, from literature)

These are fitted in W&W 2006 by matching the Abbott & Chance (2005) closed-form expression to the
first-passage time formula of a single LIF neuron driven by AMPA-mediated Gaussian noise
(Poisson input at 2.4 kHz). They are derived from macaque cortical cell parameters (Wang 2002).

| Parameter | Population class | Symbol | Value | Units | Source |
|---|---|---|---|---|---|
| Gain | Excitatory (PYR) | $c_e$ | 310 | Hz/nA  [= $(V \cdot nC)^{-1}$ in W&W notation] | W&W 2006 |
| Gain | Inhibitory (PV, SST, VIP) | $c_i$ | 615 | Hz/nA  [= $(V \cdot nC)^{-1}$ in W&W notation] | W&W 2006 |
| Bias | Excitatory (PYR) | $I_{0,e}$ | 125 | Hz  (rate-domain bias, not a current) | W&W 2006 |
| Bias | Inhibitory (PV, SST, VIP) | $I_{0,i}$ | 177 | Hz  (rate-domain bias, not a current) | W&W 2006 |
| Curvature | Excitatory (PYR) | $g_e$ | 0.16 | s | W&W 2006 |
| Curvature | Inhibitory (PV, SST, VIP) | $g_i$ | 0.087 | s | W&W 2006 |

**Derived threshold currents** (for reference — $\Theta_x = I_{0,x}/c_x$):

| Population | $\Theta_x = I_{0,x}/c_x$ | Interpretation |
|---|---|---|
| PYR | $125 / 310 \approx 0.403\ \text{nA}$ | Input current at which PYR begins to fire |
| PV, SST, VIP | $177 / 615 \approx 0.288\ \text{nA}$ | Input current at which interneurons begin to fire |

Interneurons have a lower threshold than PYR, consistent with the higher excitability of
fast-spiking cells — a distinction that Koukouli's uniform $\Theta = 7$ erases.

> **Note on SST and VIP.** W&W 2006 had only one interneuron class, parameterised from
> fast-spiking (FS) cells closest to PV. Applying the same $(c_i, I_{0,i}, g_i)$ to SST and VIP
> is an approximation. It is however a better-justified approximation than using a shared
> threshold with PYR, because: (a) SST and VIP are still GABAergic
> interneurons whose f-I curves belong to the same functional family; (b) the free $A_x$ per
> population absorbs the output-range differences between PV, SST, and VIP; (c) no published
> mouse mPFC mean-field transfer function fit exists for these subtypes, to my knowledge, that could be used to derive separate shape parameters.

### Dynamics Parameters (fixed, or fitted or removed as needed)

| Parameter | Symbol | Value | Units | Source |
|---|---|---|---|---|
| Population time constant | $\tau_s$ | 20 | ms | Beierlein et al. 2003; Koukouli et al. 2025 |
| PYR adaptation time constant | $\tau_\text{adapt}^\text{PYR}$ | 600 | ms | set to produce ~10 Hz oscillations |
| SOM adaptation time constant | $\tau_\text{adapt}^\text{SOM}$ | 150 | ms | Pospischil et al. 2008 (AdEx LTS; approximate) to verify |

We could try to have multiple time constant classes in the future, distinguishing the NMDA, GABA_A, and GABA_B components. For now, a single $\tau_s$ is used for all synaptic inputs, and the W&W shape parameters are fitted to match the resulting effective time constant.

### Free Parameters — Fitted to Firing Rate Targets

#### Transfer function output scaling

| Parameter | Symbol | Units | Population | Interpretation |
|---|---|---|---|---|
| `A_pyr` | $A_\text{PYR}$ | Hz | PYR | Maximum firing rate scale for PYR |
| `A_pv` | $A_\text{PV}$ | Hz | PV | Maximum firing rate scale for PV |
| `A_som` | $A_\text{SOM}$ | Hz | SST | Maximum firing rate scale for SST |
| `A_vip` | $A_\text{VIP}$ | Hz | VIP | Maximum firing rate scale for VIP |

These factors are the same as the role of $A_x$ from Koukouli et al. 2025: because the W&W
shape parameters fully specify the curve, $A_x$ is the only degree of freedom remaining.

But they were fitting the alpha for each population, which we don't.

#### Synaptic weights

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| `w_ee` | $\omega_{ee}$ | nA/Hz | PYR → PYR local recurrent excitation |
| `w_ep` | $\omega_{ep}$ | nA/Hz | PYR → PV |
| `w_es` | $\omega_{es}$ | nA/Hz | PYR → SST |
| `w_ev` | $\omega_{ev}$ | nA/Hz | PYR → VIP |
| `w_pe` | $\omega_{pe}$ | nA/Hz | PV → PYR (divisive numerator) |
| `w_pp` | $\omega_{pp}$ | nA/Hz | PV → PV self-inhibition |
| `w_se` | $\omega_{se}$ | nA/Hz | SST → PYR (subtractive) |
| `w_ps` | $\omega_{ps}$ | nA/Hz | PV → SST |
| `w_vs` | $\omega_{vs}$ | nA/Hz | VIP → SST |
| `w_vp` | $\omega_{vp}$ | nA/Hz | VIP → PV |

#### External and cholinergic currents

> **Notation warning.** $I_0^x$ here (tonic drive, **nA**) is distinct from $I_{0,x}$ inside
> the transfer function (rate-domain bias, **Hz**). Both are called "$I_0$" in different
> contexts in the literature. In this model: anything that enters the synaptic sum $I_{syn}$
> is in nA; the bias inside $\Phi^x$ is in Hz and is fixed from W&W.

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| `I0_pyr` | $I_0^\text{PYR}$ | nA | Baseline tonic drive to PYR (enters $I_{syn}$) |
| `I0_pv` | $I_0^\text{PV}$ | nA | Baseline tonic drive to PV (enters $I_{syn}$) |
| `I0_som` | $I_0^\text{SOM}$ | nA | Baseline tonic drive to SST (enters $I_{syn}$) |
| `I0_vip` | $I_0^\text{VIP}$ | nA | Baseline tonic drive to VIP (enters $I_{syn}$) |
| `I_alpha7_pv` | $I_{\alpha7}^\text{PV}$ | nA | α7 nAChR current onto PV |
| `I_alpha7_som` | $I_{\alpha7}^\text{SOM}$ | nA | α7 nAChR current onto SST |
| `I_beta2_som` | $I_{\beta2}^\text{SOM}$ | nA | β2 nAChR current onto SST |
| `I_alpha5_vip` | $I_{\alpha5}^\text{VIP}$ | nA | α5 nAChR current onto VIP |

**"For physiological in vivo concentrations of ACh set to 1.77 μM, the cholinergic current strength  should be 35 times greater for α4β2 (including α5α4β2) compared to α7 receptors, due to their  high affinity to ACh."**

This is a statement from Koukouli et al. 2025, based on the known pharmacology of nAChR subtypes. We can use this as a rough guide for setting the relative magnitudes of the cholinergic currents, while still fitting them freely to match the firing rate targets. 

#### GABA modulation

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| `g_gaba_base` | $g_\text{GABA}^\text{base}$ | dimensionless | Baseline GABA scaling factor |
| `g_alpha7` | $g_{\alpha7}$ | dimensionless | α7-mediated enhancement of GABA transmission |

#### Adaptation

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| `J_adapt_pyr` | $J_\text{adapt}^\text{PYR}$ | nA/Hz | PYR adaptation strength |
| `J_adapt_som` | $J_\text{adapt}^\text{SOM}$ | nA/Hz | SST adaptation strength (if used) |

#### Noise

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| `sigma_noise` | $\sigma_{\text{noise}}$ | dimensionless | Noise amplitude in current space (applied to all four populations). Effective noise current = `sigma_noise × I_ext_pyr` (nA). Shared factor across populations; each population experiences its own effective firing-rate noise via its transfer function slope. Supports white or OU processes. |

---

## Unit Consistency Check

The rate equation for each population $x$ is:

$$\tau_s \frac{dr_x}{dt} = -r_x + \Phi^x(I_x^\text{syn}(t))$$

where the PYR input current includes the stochastic noise term:

$$I_i^{\text{PYR}}(t) = I_i^{\text{PYR,det}}(t) + \sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$$

and other populations receive deterministic input only: $I_i^X(t) = I_i^{X,\text{det}}(t)$ for $X \in \{\text{PV}, \text{SOM}, \text{VIP}\}$.

### Check 1 — Rate equation (deterministic component)

| Term | Units | Check |
|---|---|---|
| $\tau_s$ | ms | — |
| $dr_x/dt$ | Hz/ms = kHz | — |
| $\tau_s \cdot dr_x/dt$ | ms × kHz = **Hz** ✓ | matches $r_x$ |
| $r_x$ | Hz | ✓ |
| $\Phi^x(\cdot)$ | Hz | verified in Check 2 |

### Check 2 — Noise term in current space

The noise injected into the PYR current is:

$$\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$$

| Term | Units | Check |
|---|---|---|
| $\sigma_{\text{noise}}$ | dimensionless | — |
| $I_{\text{ext}}^{\text{PYR}}$ | nA | — |
| $\xi_i(t)$ | dimensionless | — |
| $\sigma_{\text{noise}} \cdot I_{\text{ext}}^{\text{PYR}} \cdot \xi_i(t)$ | dimensionless × nA × dimensionless = **nA** ✓ | Noise perturbation has units of current |

### Check 3 — Transfer function: resolving the units of $c_x$ and $I_{0,x}$

$$\Phi^x(I) = A_x \cdot \frac{c_x I - I_{0,x}}{1 - \exp[-g_x(c_x I - I_{0,x})]}$$

**Step 1 — units of $c_x$.**

W&W report $c_x$ in $(V \cdot nC)^{-1}$. Expanding:

$$V \cdot nC = V \cdot nA \cdot s \quad \Rightarrow \quad (V \cdot nC)^{-1} = \frac{1}{V \cdot nA \cdot s}$$

Using Ohm's law $V = nA \cdot G\Omega$ (where conductance absorbs the resistance of the LIF
membrane), the fitting procedure of W&W implicitly sets $1/V = nA \cdot \text{const}$, and
the product $c_x \cdot I_{syn}$ is constrained to output Hz by the LIF first-passage time
formula it was fitted to. The operationally correct unit assignment — consistent with dimensional
analysis of the full equation — is therefore:

$$\boxed{c_x \ \text{has units}\ \text{Hz/nA}}$$

This is the unit in which $c_x$ should be understood throughout this model.

**Step 2 — units of $I_{0,x}$ follow from $c_x$.**

For the subtraction $c_x I - I_{0,x}$ to be dimensionally consistent, $I_{0,x}$ must have
the same units as $c_x \cdot I$:

$$[c_x \cdot I] = \frac{\text{Hz}}{\text{nA}} \times \text{nA} = \text{Hz}$$

$$\therefore \boxed{I_{0,x} \ \text{has units}\ \text{Hz}}$$

This is why W&W report $I_{0,e} = 125\ \text{Hz}$ and $I_{0,i} = 177\ \text{Hz}$: they are
biases in the **rate domain**, not input currents, despite the suggestive notation.

**Step 3 — threshold $\Theta_x = I_{0,x} / c_x$ is in nA.**

$$[\Theta_x] = \frac{[I_{0,x}]}{[c_x]} = \frac{\text{Hz}}{\text{Hz/nA}} = \text{nA} \checkmark$$

This is the true current threshold — the value of $I_{syn}$ (in nA) at which the neuron begins
to respond. The two W&W thresholds are:

$$\Theta_e = \frac{125\ \text{Hz}}{310\ \text{Hz/nA}} \approx 0.403\ \text{nA}, \qquad
\Theta_i = \frac{177\ \text{Hz}}{615\ \text{Hz/nA}} \approx 0.288\ \text{nA}$$

**Step 4 — full unit chain for $\Phi^x$.**

| Term | Units | Check |
|---|---|---|
| $c_x$ | Hz/nA | ✓ |
| $I_{syn}$ | nA | ✓ |
| $c_x \cdot I_{syn}$ | Hz/nA × nA = **Hz** | ✓ |
| $I_{0,x}$ | Hz | ✓ |
| $c_x I - I_{0,x}$ | Hz − Hz = **Hz** | ✓ |
| $g_x$ | s | ✓ |
| $g_x \cdot (c_x I - I_{0,x})$ | s × Hz = **dimensionless** | ✓ exponent consistent |
| $\exp[\cdot]$ | dimensionless | ✓ |
| $1 - \exp[\cdot]$ | dimensionless | ✓ |
| $(c_x I - I_{0,x})/(1-\exp[\cdot])$ | Hz / dimensionless = **Hz** | ✓ |
| $A_x$ | dimensionless | ✓ (pure rescaling factor) |
| $\Phi^x = A_x \cdot (\ldots)$ | dimensionless × Hz = **Hz** | ✓ |

### Check 4 — Synaptic input current

The PYR input (simplified, excluding divisive PV and adaptation terms):

$$I^\text{PYR} = \omega_{ee} \, r_\text{PYR} + \omega_{se}^\dagger \, r_\text{SOM} + I_0^\text{PYR} + I_\text{nAChR}$$

| Term | Units | Check |
|---|---|---|
| $\omega_{xj}$ | nA/Hz | — |
| $r_j$ | Hz | — |
| $\omega_{xj} \cdot r_j$ | nA/Hz × Hz = **nA** ✓ | |
| $I_0^x$ | nA | ✓ |
| $I_\text{nAChR}$ | nA | ✓ |
| $I^\text{PYR}$ total | nA | ✓ — enters $\Phi^x(I)$ as nA ✓ |

### Check 5 — Adaptation current

$$\tau_\text{adapt} \frac{dI_\text{adapt}}{dt} = -I_\text{adapt} + J_\text{adapt} \cdot r$$

| Term | Units | Check |
|---|---|---|
| $\tau_\text{adapt}$ | ms | — |
| $I_\text{adapt}$ | nA | — |
| $dI_\text{adapt}/dt$ | nA/ms | — |
| $\tau_\text{adapt} \cdot dI_\text{adapt}/dt$ | ms × nA/ms = **nA** ✓ | matches $I_\text{adapt}$ |
| $J_\text{adapt}$ | nA/Hz | — |
| $r$ | Hz | — |
| $J_\text{adapt} \cdot r$ | nA/Hz × Hz = **nA** ✓ | matches $I_\text{adapt}$ |

All terms consistent.

---

## Comparison with Koukouli et al. 2025

| Aspect | Koukouli et al. 2025 | This model |
|---|---|---|
| Threshold | $\Theta = 7$ for all populations (dimensionless, **uniform**) | $\Theta_E \approx 0.403$ nA (PYR), $\Theta_I \approx 0.288$ nA (PV/SST/VIP) — **differentiated** |
| Gain $c$ | $\alpha_x \in \{1.9, 2.6, 1.5, 1.2\}$ per population (dimensionless) | $c_e = 310\ \text{nA}^{-1}$ (PYR), $c_i = 615\ \text{nA}^{-1}$ (inhibitory) — **W&W values** |
| Free TF params | $A_x$ × 4 populations | $A_x$ × 4 populations |
| Fixed TF params | $\alpha_x$ × 4 + $\Theta$ × 4 + $g$ × 1 = **9 parameters** | $(c, I_0, g)$ × 2 classes = **6 parameters** |
| Unit convention | Dimensionless currents | nA throughout |

The present approach is strictly more parsimonious in fixed parameters and more directly anchored
to the original W&W derivation.

---

## References

- Wong, K.-F. & Wang, X.-J. (2006). A recurrent network mechanism of time integration in
  perceptual decisions. *J. Neurosci.* 26(4):1314–1328.
- Abbott, L.F. & Chance, F.S. (2005). Drivers and modulators from push-pull and balanced
  synaptic input. *Prog. Brain Res.* 149:147–155.
- Koukouli, F. et al. (2025). Nicotinic acetylcholine receptor subtypes in prefrontal cortex
  layer 2/3 circuit. [supplementary methods]
- Beierlein, M. et al. (2003). Two dynamically distinct inhibitory networks in layer 4 of the
  neocortex. *J. Neurophysiol.* 90(5):2987–3000.
- Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits.
  *Neuron* 36(5):955–968.