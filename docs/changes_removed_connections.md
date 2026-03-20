# Removal of Biologically Unsupported Local Connections

**Date:** 2026-03-19

Two synaptic connections have been removed from the model because they are not supported by the biological literature on PFC microcircuit anatomy: the VIP → VIP self-inhibitory connection (`w_vv`) and the PV → SOM cross-inhibitory connection (`w_ps`). Both had been flagged in the original code comments as absent from the reference schematic diagram.

---

## 1. VIP → VIP Self-Inhibition (`w_vv`)

### Biological Motivation

VIP interneurons implement the canonical disinhibitory motif by inhibiting SOM (and weakly PV) cells. However, **local VIP → VIP autoinhibition is not an established connection in the PFC microcircuit literature**. The previous implementation included it as a pragmatic stabilization device (preventing VIP runaway under strong external drive), but this was at odds with the schematic connectivity the model is meant to represent.

### Equation Change

**Before:**

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} - w_{vv}\,r_\text{VIP} + I_\text{ext}^\text{VIP}$$

**After:**

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} + I_\text{ext}^\text{VIP}$$

The self-inhibitory term $-w_{vv}\,r_\text{VIP}$ is removed. VIP activity is now regulated solely by the transfer function saturation and the external drive $I_\text{ext}^\text{VIP}$.

### Files Changed

| File | Change |
|------|--------|
| `circuit_model/params.py` | Removed `w_vv` field and its optimization bound |
| `circuit_model/simulation.py` | Removed `- params.w_vv * r_vip` from `I_vip` |
| `circuit_model/ring/simulation.py` | Removed `w_vv` from single-batch and batch loops |
| `circuit_model/cli.py` | Removed `w_vv` from display group and help example |
| `circuit_model/ring/cli.py` | Removed `w_vv` from parameter print line |
| `README.md` | Updated connectivity matrix, VIP equation, parameter table |
| `docs/ring_attractor.md` | Updated VIP input equation (§3.1) |
| `docs/CLI.md` | Updated `--set` example |

---

## 2. PV → SOM Cross-Inhibition (`w_ps`)

### Biological Motivation

PV interneurons provide fast perisomatic inhibition primarily to pyramidal cells. **A direct PV → SOM inhibitory connection is not a canonical feature of the PFC microcircuit** and is absent from the reference schematic. The previous default value (`w_ps = 2.22`, identical to `w_pe`) was high enough to meaningfully suppress SOM activity, making it a significant — and unjustified — circuit element.

### Equation Change

**Before:**

$$I_\text{SOM} = w_{es}\,r_\text{PYR} - g_\text{GABA}\,w_{ps}\,r_\text{PV} - w_{vs}\,r_\text{VIP} - I_\text{adapt}^\text{SOM} + I_\text{ext}^\text{SOM}$$

**After:**

$$I_\text{SOM} = w_{es}\,r_\text{PYR} - w_{vs}\,r_\text{VIP} - I_\text{adapt}^\text{SOM} + I_\text{ext}^\text{SOM}$$

The cross-inhibitory term $-g_\text{GABA}\,w_{ps}\,r_\text{PV}$ is removed.

### Files Changed

| File | Change |
|------|--------|
| `circuit_model/params.py` | Removed `w_ps` field and its optimization bound |
| `circuit_model/simulation.py` | Removed `- ggaba * params.w_ps * r_pv` from `I_som` |
| `circuit_model/ring/simulation.py` | Removed `w_ps` from single-batch and batch loops |
| `README.md` | Updated connectivity matrix, SOM equation, parameter table |
| `docs/ring_attractor.md` | Updated SOM input equation (§3.1) |

---

## Impact on Existing JSON Parameter Files

Existing parameter JSON files (e.g. `params/old/code.json`, optimization logs in `figs/`) may still contain `"w_vv"` and `"w_ps"` keys. This is harmless: `load_params_json()` silently ignores any key not present in `CircuitParams`. No migration is required.
