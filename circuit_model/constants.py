"""
Shared constants for single-node circuit model.

These values are used across optimization, loss computation, and analysis scripts.
"""

# Maximum physiological firing rate for PYR neurons (Hz)
# The model clamps Phi to [0, 200] Hz, but the clamp ceiling creates spurious
# fixed points in nullcline analysis. This defines the biologically plausible
# upper bound for PYR firing rates, excluding clamp artifacts.
R_MAX_PHYS = 100.0  # Hz

# Physiological ceiling for the upper stable fixed point in bistability mode (Hz)
# The optimizer should not be allowed to "solve" bistability by pushing the
# upper fixed point into the clamp region (above R_MAX_PHYS).
R_HIGH_MAX = 80.0  # Hz

# NMDA gating constants (Wong & Wang 2006) — fixed physics, not fitted
TAU_NMDA_MS = 100.0   # ms
GAMMA_NMDA = 0.641    # dimensionless

# Hyperbolic soft ceiling for interneuron transfer functions (Hz)
# Set to 2 × Rooy 2021 high-state targets (see docs/transfer_function_ceiling.md)
R_MAX_PV  = 70.6   # Hz  (2 × 35.3)
R_MAX_SOM = 70.4   # Hz  (2 × 35.2)
R_MAX_VIP = 137.6  # Hz  (2 × 68.8)
