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
