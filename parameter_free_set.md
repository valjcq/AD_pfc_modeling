# Model Parameters: Fixed vs Free

Parameters are classified as **fixed** (set from literature before any fitting) or **free** (optimized to match experimental firing rates).

The transfer function used is the Wong-Wang form:

$$\Phi^X(I) = A_x \cdot \frac{u}{1 - e^{-g \, u}}, \quad u = \alpha_x \cdot (I - \Theta_x)$$

where $\Theta_x$ is the threshold, $\alpha_x$ the gain, $g$ the curvature (shared across all populations), and $A_x$ the output scaling. The model operates in dimensionless (arbitrary) current units; $g$ is therefore not imported from the physically-scaled Wong & Wang 2006 paper but is set independently to control curve shape in this unit convention.

---

## Fixed Parameters — Set from Literature

### Dynamics

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| `tau_s` | $\tau_s$ | 20 ms | Beierlein et al. 2003 — single value shared across all populations |
| `tau_adapt_pyr` | $\tau_\text{adapt}^\text{PYR}$ | 600 ms | Destexhe 2009, as used in Koukouli et al. 2025 |

> **Note on SOM adaptation:** Koukouli et al. 2025 do not model SOM spike-frequency adaptation. If added to this model, $\tau_\text{adapt}^\text{SOM}$ and $J_\text{adapt}^\text{SOM}$ become additional free parameters with no prior literature anchor. The only available computational anchor is ~200 ms from AdEx models of LTS cells (Pospischil et al. 2008), but this is not a direct measurement from SOM neurons.

### Transfer function shape

| Parameter | Symbol | Value | Source / Rationale |
|---|---|---|---|
| `Theta_e/p/s/v` | $\Theta_x$ | 7.0 (all) | Koukouli et al. 2025 Table 1 (from Wong & Wang 2006) |
| `alpha_e` | $\alpha_e$ | 1.9 | Koukouli et al. 2025 Table 1 |
| `alpha_p` | $\alpha_p$ | 2.6 | Koukouli et al. 2025 Table 1 |
| `alpha_s` | $\alpha_s$ | 1.5 | Koukouli et al. 2025 Table 1 |
| `alpha_v` | $\alpha_v$ | 1.2 | Koukouli et al. 2025 Table 1 |
| `g` | $g$ | **1.0 (all populations)** | Shape parameter — single value for all populations. The original Wong & Wang values ($g_e=310$, $g_i=615$ Hz/nA) are in physical units incompatible with this model's dimensionless convention. A single shared $g$ is used here because (1) the difference between $g_e$ and $g_i$ in the original reflects a PYR vs. FS f-I curve shape difference already partially captured by the different $\alpha_x$ values, and (2) the output scaling $A_x$ absorbs residual per-population output range differences. Value set to 1.0 as a neutral shape parameter; verified that the function operates in its near-linear regime at expected firing rates. |

> **Important:** $\Theta_x$ and $\alpha_x$ are listed in Koukouli et al. Table 1 and are inherited from Wong & Wang 2006. The curvature parameter $g$ is **absent from Koukouli et al.** — the paper states it is "from Wong & Wang 2006" but provides no numerical value. The Wong & Wang values are in physical units (Hz/nA) and cannot be directly used in this dimensionless model. The choice of $g = 1$ is therefore an independent modeling decision, not a literature-anchored value.

---

## Free Parameters — Optimized to Match Firing Rate Targets

### Transfer function output scaling

| Parameter | Symbol | Koukouli et al. reference value | Description |
|---|---|---|---|
| `A_pyr` | $A_e$ | 4.2 | PYR maximum output scaling |
| `A_pv` | $A_p$ | 10.1 | PV maximum output scaling |
| `A_som` | $A_s$ | 17.1 | SOM maximum output scaling |
| `A_vip` | $A_v$ | 15.5 | VIP maximum output scaling |

These are free parameters in Koukouli et al., fitted by the optimizer to match experimental firing rate targets. The Koukouli values are provided as reference only — they are not used as fixed values here because they were obtained in a differently scaled model. The optimizer will recover its own values for this model's unit convention.

### Synaptic weights

| Parameter | Symbol | Description |
|---|---|---|
| `w_ee` | $w_{ee}$ | PYR → PYR local recurrent excitation |
| `w_ep` | $w_{ep}$ | PYR → PV |
| `w_es` | $w_{es}$ | PYR → SOM |
| `w_ev` | $w_{ev}$ | PYR → VIP |
| `w_pe` | $w_{pe}$ | PV → PYR (divisive) |
| `w_pp` | $w_{pp}$ | PV → PV self-inhibition |
| `w_se` | $w_{se}$ | SOM → PYR (subtractive) |
| `w_ps` | $w_{ps}$ | PV → SOM |
| `w_vs` | $w_{vs}$ | VIP → SOM |
| `w_vp` | $w_{vp}$ | VIP → PV |

### External currents

| Parameter | Symbol | Description |
|---|---|---|
| `I0_pyr` | $I_0^\text{PYR}$ | Baseline tonic drive to PYR |
| `I0_pv` | $I_0^\text{PV}$ | Baseline tonic drive to PV |
| `I0_som` | $I_0^\text{SOM}$ | Baseline tonic drive to SOM |
| `I0_vip` | $I_0^\text{VIP}$ | Baseline tonic drive to VIP |
| `I_alpha7_pv` | $I_{\alpha7}^\text{PV}$ | α7 nAChR current onto PV |
| `I_alpha7_som` | $I_{\alpha7}^\text{SOM}$ | α7 nAChR current onto SOM |
| `I_beta2_som` | $I_{\beta2}^\text{SOM}$ | β2 nAChR current onto SOM |
| `I_alpha5_vip` | $I_{\alpha5}^\text{VIP}$ | α5 nAChR current onto VIP |

### GABA modulation

| Parameter | Symbol | Description |
|---|---|---|
| `g_gaba_base` | $g_\text{GABA}^\text{base}$ | Baseline GABA scaling factor |
| `g_alpha7` | $g_{\alpha7}$ | α7-mediated enhancement of GABA transmission |

### Adaptation

| Parameter | Symbol | Description |
|---|---|---|
| `J_adapt_pyr` | $J_\text{adapt}^\text{PYR}$ | PYR adaptation strength |

### Noise

| Parameter | Symbol | Description |
|---|---|---|
| `sigma_s` | $\sigma_s$ | Noise amplitude (free in Koukouli et al.) |

---

## Ring-Specific Parameters

These apply only to the ring attractor and are not part of the point-circuit firing rate fit. They should be set by a separate principled grid search targeting bump stability and biologically plausible bump width (~40–60°).

| Parameter | Symbol | Description |
|---|---|---|
| `w_pyr_pyr_inter` | $w_\text{pyr}^\text{inter}$ | Total row-sum of inter-node PYR→PYR Gaussian weights |
| `sigma_pyr_deg` | $\sigma_\text{pyr}$ | Width of PYR→PYR Gaussian connectivity profile |
| `w_pv_global` | $w_\text{PV}^\text{global}$ | Strength of uniform global PV→PYR inhibition |