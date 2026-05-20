"""
Shared constants for single-node circuit model.

These values are used across optimization, loss computation, and analysis scripts.
"""

# Maximum physiological firing rate for PYR neurons (Hz)
# Used by nullcline analysis (bistable_loss) to filter out spurious fixed points
# created by hard clamping at high rates. PYR is uncapped in the integrator, so
# this is an analysis-only ceiling, not a clamp.
R_MAX_PHYS = 100.0  # Hz

# NMDA gating constants (Wong & Wang 2006) — fixed physics, not fitted
TAU_NMDA_MS = 100.0   # ms
GAMMA_NMDA = 0.641    # dimensionless

# Hyperbolic soft ceiling for interneuron transfer functions (Hz)
# Set to 2 × Rooy 2021 high-state targets (see docs/transfer_function_ceiling.md)
R_MAX_PV  = 70.6   # Hz  (2 × 35.3)
R_MAX_SOM = 70.4   # Hz  (2 × 35.2)
R_MAX_VIP = 137.6  # Hz  (2 × 68.8)
