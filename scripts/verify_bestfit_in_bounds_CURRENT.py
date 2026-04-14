#!/usr/bin/env python3
"""
RE-VERIFY bounds using CURRENT best_params.json with J_NMDA (not w_ee).
This is the authoritative check — my bounds must accommodate ALL current best-fit values.
"""

import json
from circuit_model.params import default_bounds, CircuitParams

# Load CURRENT best-fit values (should use J_NMDA, not w_ee)
with open("best_params.json") as f:
    best_params = json.load(f)

print("=" * 80)
print("CURRENT best_params.json SCHEMA CHECK")
print("=" * 80)
print(f"Parameters: {len(best_params)} total")
print(f"Uses 'w_ee': {'w_ee' in best_params}")
print(f"Uses 'J_NMDA': {'J_NMDA' in best_params}")
print()

# Create dummy base for getting bounds
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
print(f"BOUNDS VERIFICATION (CURRENT SCHEMA)")
print(f"{'='*80}\n")

all_ok = True
issues = []

for param_name in sorted(optimized_params):
    if param_name not in best_params:
        print(f"⚠️  {param_name:20s} — NOT IN best_params.json")
        continue

    value = best_params[param_name]
    bound = bounds[param_name]
    lo, hi = bound.lo, bound.hi

    in_bounds = lo <= value <= hi
    status = "✓" if in_bounds else "✗ OUT OF BOUNDS"

    if not in_bounds:
        all_ok = False
        issues.append((param_name, value, lo, hi))

    print(f"{status} {param_name:20s} = {value:10.6f}  [bounds: {lo:8.4f}, {hi:8.4f}]")

print(f"\n{'='*80}")
if all_ok:
    print("✓✓✓ ALL BEST-FIT PARAMETERS ARE WITHIN BOUNDS ✓✓✓")
else:
    print("✗✗✗ SOME PARAMETERS OUT OF BOUNDS - MUST FIX BOUNDS ✗✗✗")
    print("\nOut-of-bounds parameters:")
    for name, val, lo, hi in issues:
        if val < lo:
            print(f"  {name}: {val:.6f} < lower bound {lo:.6f}")
        else:
            print(f"  {name}: {val:.6f} > upper bound {hi:.6f}")

print(f"{'='*80}")
