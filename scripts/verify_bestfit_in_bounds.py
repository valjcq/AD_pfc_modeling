#!/usr/bin/env python3
"""
Verify that best-fit parameters from best_params.json stay within new bounds.
"""

import json
from circuit_model.params import default_bounds, CircuitParams

# Load best-fit values
with open("best_params.json") as f:
    best_params = json.load(f)

# Create dummy base for getting bounds (actual values don't matter for bounds)
dummy_base = CircuitParams(
    I0_pyr=0.53, I0_pv=0.48, I0_som=0.53, I0_vip=0.20,
    w_ep=0.004, w_pp=0.36, w_se=0.49, w_es=0.035,
    w_vs=0.34, w_ev=0.001, w_sp=0.053, w_vp=0.092,
    w_pe=0.027, J_NMDA=0.057, J_adapt_pyr=0.001, J_adapt_som=0.095,
    g_gaba_base=2.41, g_alpha7=1.63, I_alpha7_pv=0.149, I_alpha7_som=0.177,
    I_beta2_som=0.155, I_alpha5_vip=0.035, tau_adapt_pyr=269, tau_adapt_som=300,
    trans_factor=0.66,
)

bounds = default_bounds(dummy_base)

# Parameters that should be optimized
optimized_params = {
    "w_ep", "w_pp", "w_se", "w_es", "w_vs", "w_ev", "w_sp", "w_vp", "w_pe",
    "J_NMDA", "I0_pyr", "I0_pv", "I0_som", "I0_vip",
    "J_adapt_pyr", "J_adapt_som", "tau_adapt_pyr", "tau_adapt_som",
    "g_gaba_base", "g_alpha7", "I_alpha7_pv", "I_alpha7_som",
    "I_beta2_som", "I_alpha5_vip", "trans_factor"
}

print(f"{'='*80}")
print(f"BEST-FIT PARAMETER VALIDATION")
print(f"{'='*80}\n")

all_ok = True
for param_name in optimized_params:
    if param_name not in best_params:
        print(f"⚠️  {param_name:20s} — NOT IN best_params.json (skipped)")
        continue

    value = best_params[param_name]
    bound = bounds[param_name]
    lo, hi = bound.lo, bound.hi

    in_bounds = lo <= value <= hi
    status = "✓" if in_bounds else "✗ OUT OF BOUNDS"

    if not in_bounds:
        all_ok = False

    print(f"{status} {param_name:20s} = {value:.6f}  [bounds: {lo:.6f}, {hi:.6f}]")

print(f"\n{'='*80}")
if all_ok:
    print("✓ All best-fit parameters are within the new bounds!")
else:
    print("✗ Some best-fit parameters exceed the new bounds!")
print(f"{'='*80}")
