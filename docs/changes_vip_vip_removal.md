# Removal of VIP → VIP Local Connection

**Date:** 2026-03-18

## Summary

The VIP → VIP self-inhibitory connection (`w_vv`) has been removed from the model. This parameter was not grounded in the biological literature for local PFC microcircuit connectivity and had been flagged in the code itself as absent from the reference schematic diagram.

---

## Biological Motivation

VIP interneurons are known to inhibit SOM and, to a lesser extent, PV interneurons, implementing the canonical disinhibitory circuit motif. However, **local VIP → VIP autoinhibition is not established as a functional connection in the PFC microcircuit literature**. The previous implementation included it as a pragmatic stabilization device (preventing VIP runaway under strong external drive), but this was at odds with the schematic connectivity the model is meant to represent.

---

## Equation Change

### Before

$$\tau_s \frac{dr_\text{VIP}}{dt} = -r_\text{VIP} + \Phi\!\left(I_\text{VIP}\right) + \sigma_s\,\xi(t)$$

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} - w_{vv}\,r_\text{VIP} + I_\text{ext}^\text{VIP}$$

### After

$$\tau_s \frac{dr_\text{VIP}}{dt} = -r_\text{VIP} + \Phi\!\left(I_\text{VIP}\right) + \sigma_s\,\xi(t)$$

$$I_\text{VIP} = w_{ev}\,r_\text{PYR} + I_\text{ext}^\text{VIP}$$

The self-inhibitory term $-w_{vv}\,r_\text{VIP}$ is removed. VIP activity is now regulated solely by the transfer function saturation and the external drive $I_\text{ext}^\text{VIP}$.

---

## Files Changed

| File | Change |
|------|--------|
| `circuit_model/params.py` | Removed `w_vv` field from `CircuitParams`; removed its bound from `default_bounds()` |
| `circuit_model/simulation.py` | Removed `- params.w_vv * r_vip` from `I_vip` computation |
| `circuit_model/ring/simulation.py` | Removed `- p.w_vv * r_vip` (single-batch loop); removed `w_vv` array preallocation and `- w_vv * r_vip` (batch loop) |
| `circuit_model/cli.py` | Removed `w_vv` from inhibitory weight display group and help example |
| `circuit_model/ring/cli.py` | Removed `w_vv` from parameter print line |
| `README.md` | Updated connectivity matrix, VIP input equation, and parameter table |
| `docs/CLI.md` | Updated `--set` example |

---

## Impact on Existing JSON Parameter Files

Existing parameter JSON files (e.g. `params/code.json`, `params/best_param_optim.json`, optimization logs) may still contain a `"w_vv"` key. This is harmless: the `load_params_json()` function silently ignores any key not present in `CircuitParams`. No migration is required.

---

## Impact on Model Dynamics

Removing `w_vv` changes VIP steady-state behavior. With `w_vv = 24.8` and typical VIP rates, the self-inhibition term contributed approximately `~25 × r_VIP` to reduce the input current. Without it, VIP firing is determined by the balance of `w_ev · r_PYR` and `I_ext_VIP` alone, modulated by the transfer function nonlinearity.

**If VIP rates become unstable after this change, the recommended approach is to:**
1. Re-optimize the remaining parameters (especially `I0_vip`, `alpha_vip`, `Theta_vip`) to achieve the desired steady-state VIP rate.
2. Do not reintroduce `w_vv` as a compensatory mechanism.
