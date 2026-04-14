#!/usr/bin/env python3
"""
Verification script: Check viable region fraction improvement after bounds tightening.

Uses 50,000 random samples from NEW bounds with:
- Proxy steady-state rates (WT best-fit)
- Exact I_PYR computation from jacobian.py formula
- Pure numpy (no simulation), < 10 seconds runtime
"""

import numpy as np
from circuit_model.params import default_bounds, CircuitParams
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS

# --- Setup ---
np.random.seed(42)
n_samples = 50000
THETA_PYR = 0.40323  # Transfer function threshold

# Proxy steady-state rates from best-fit WT solution
r_pyr_proxy = 8.214
r_som_proxy = 4.295
r_pv_proxy = 4.073
r_vip_proxy = 6.051

# Create a dummy base params object to get bounds
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

# --- Generate 50,000 random samples ---
print("Generating 50,000 random samples from NEW bounds...")

samples = {}
for param_name, bound in bounds.items():
    if param_name not in [
        "w_ep", "w_pp", "w_se", "w_es", "w_vs", "w_ev", "w_sp", "w_vp",
        "w_pe", "J_NMDA", "I0_pyr", "I0_pv", "I0_som", "I0_vip",
        "J_adapt_pyr", "J_adapt_som", "tau_adapt_pyr", "tau_adapt_som",
        "g_gaba_base", "g_alpha7", "I_alpha7_pv", "I_alpha7_som",
        "I_beta2_som", "I_alpha5_vip", "trans_factor"
    ]:
        continue  # Skip non-optimized parameters

    lo, hi = bound.lo, bound.hi
    if bound.mode == "log":
        # Log-uniform sampling: log(x) ~ Uniform[log(lo), log(hi)]
        if lo > 0:
            samples[param_name] = np.exp(np.random.uniform(np.log(lo), np.log(hi), n_samples))
        else:
            # Fallback to linear for [0, hi]
            samples[param_name] = np.random.uniform(lo, hi, n_samples)
    else:  # mode == "lin"
        samples[param_name] = np.random.uniform(lo, hi, n_samples)

# --- Compute I_PYR for each sample ---
print("Computing I_PYR for each sample...")

viable = 0
for i in range(n_samples):
    # Extract parameters for this sample
    w_ep = samples["w_ep"][i]
    w_pp = samples["w_pp"][i]
    w_se = samples["w_se"][i]
    w_sp = samples["w_sp"][i]
    w_vp = samples["w_vp"][i]
    w_pe = samples["w_pe"][i]
    J_NMDA = samples["J_NMDA"][i]
    J_adapt_pyr = samples["J_adapt_pyr"][i]
    I0_pyr = samples["I0_pyr"][i]
    g_gaba_base = samples["g_gaba_base"][i]
    I_alpha7_pv = samples["I_alpha7_pv"][i]
    I_alpha7_som = samples["I_alpha7_som"][i]
    I_beta2_som = samples["I_beta2_som"][i]

    # Compute GABA modulation (g() method)
    ggaba = g_gaba_base

    # NMDA gating at steady state
    S_star = (GAMMA_NMDA * r_pyr_proxy * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr_proxy * TAU_NMDA_MS)

    # Compute I_PYR using exact formula from jacobian.py
    denom = 1.0 + ggaba * w_pe * r_pv_proxy
    I_adapt_pyr = J_adapt_pyr * r_pyr_proxy
    I_ext_pyr = I0_pyr

    I_pyr = (J_NMDA * S_star) / denom - ggaba * w_se * r_som_proxy - I_adapt_pyr + I_ext_pyr

    # Check if viable (above threshold)
    if I_pyr > THETA_PYR:
        viable += 1

viable_fraction = viable / n_samples * 100

print(f"\n{'='*60}")
print(f"VERIFICATION RESULTS")
print(f"{'='*60}")
print(f"Total samples: {n_samples:,}")
print(f"Viable samples (I_PYR > {THETA_PYR}): {viable:,}")
print(f"Viable fraction: {viable_fraction:.2f}%")
print(f"\nComparison:")
print(f"  Old viable fraction: 4.07%")
print(f"  New viable fraction: {viable_fraction:.2f}%")
print(f"  Improvement: {viable_fraction / 4.07:.1f}× better")
print(f"\nTarget: > 15% ✓" if viable_fraction > 15 else f"\nTarget: > 15% ✗ (below target)")
print(f"{'='*60}")
