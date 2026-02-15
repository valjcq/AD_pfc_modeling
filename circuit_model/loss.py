"""
Loss functions for parameter optimization.

This module contains:
- TargetRates: Target firing rates for optimization
- FitConfig: Configuration for fitting/optimization
- loss_from_means: Loss for base condition
- loss_from_ko_pyr: Loss for knockout conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .simulation import NoiseType


@dataclass(frozen=True)
class TargetRates:
    """Target mean firing rates for optimization."""
    mean_r_pyr: float
    mean_r_som: float
    mean_r_pv: float
    mean_r_vip: float

    # Optional knockout targets
    alpha7_ko_pyr: Optional[float] = None
    alpha5_ko_pyr: Optional[float] = None
    beta2_ko_pyr: Optional[float] = None

    def as_array(self) -> np.ndarray:
        """Return base targets as array [pyr, som, pv, vip]."""
        return np.array(
            [self.mean_r_pyr, self.mean_r_som, self.mean_r_pv, self.mean_r_vip],
            dtype=float,
        )


@dataclass(frozen=True)
class FitConfig:
    """Configuration for simulation and optimization."""
    T_ms: float = 2500.0          # Simulation duration
    dt_ms: float = 0.1            # Time step
    burn_in_ms: float = 1800.0    # Burn-in period (skip transients)
    window_ms: float = 500.0      # Averaging window

    n_trials: int = 8             # Number of trials per parameter set
    init_rate_scale: float = 0.2  # Scale for random initial conditions

    noise_type: NoiseType = "none"
    tau_noise_ms: float = 5.0

    max_rate: float = 200.0       # Maximum allowed rate (stability check)

    ko_min_effect_penalty: float = 5.0      # Penalty for weak KO effect
    ko_wrong_direction_penalty: float = 10.0  # Penalty for wrong direction


def loss_from_means(
    means: np.ndarray,
    target: TargetRates,
    *,
    near_zero_threshold: float = 0.1,
    near_zero_weight: float = 10.0,
) -> float:
    """
    Compute loss between simulated mean firing rates and targets.

    Uses relative MSE (normalized by target magnitude) plus a penalty
    for rates that are too close to zero (to avoid silent solutions).

    Loss = mean((actual - target)^2 / target^2) + penalty_weight * sum(near_zero_penalties)
    """
    tgt = target.as_array()
    denom = np.maximum(np.abs(tgt), 1e-3)  # Avoid division by zero
    rel = (means - tgt) / denom  # Relative error
    mse = float(np.mean(rel**2))

    # Penalize rates that are too close to zero (biologically unrealistic)
    below = np.maximum(near_zero_threshold - means, 0.0)
    near_zero = float(np.sum((below / near_zero_threshold) ** 2))
    return mse + near_zero_weight * near_zero


def loss_from_ko_pyr(
    pyr_mean: float,
    target_pyr: float,
    base_pyr: float,
    *,
    near_zero_threshold: float = 0.1,
    near_zero_weight: float = 10.0,
    min_effect_weight: float = 5.0,
    wrong_direction_weight: float = 10.0,
) -> float:
    """
    Compute loss for knockout (KO) condition targeting PYR firing rate.

    This loss function ensures the model correctly captures receptor knockout effects:
    1. The KO firing rate should match the target
    2. The change from baseline should be in the correct direction
    3. The effect magnitude should be similar to expected

    Example: If alpha7 KO should increase PYR firing from 5 to 7:
        - target_pyr = 7.0 (expected under KO)
        - base_pyr = 5.0 (baseline condition)
        - expected change = +2
        - If actual change is negative, apply wrong_direction penalty
        - If actual change is too small, apply min_effect penalty
    """
    # Standard relative MSE term
    denom = max(abs(target_pyr), 1e-3)
    mse = ((pyr_mean - target_pyr) / denom) ** 2

    # Penalty for near-zero firing (biologically unrealistic)
    below = max(near_zero_threshold - pyr_mean, 0.0)
    near_zero = (below / near_zero_threshold) ** 2

    # Calculate expected vs actual effect of knockout
    expected = target_pyr - base_pyr  # Expected change due to KO
    actual = pyr_mean - base_pyr      # Actual change observed
    exp_mag = abs(expected)
    act_mag = abs(actual)

    min_effect = 0.0
    wrong_dir = 0.0
    if exp_mag > 0.1:  # Only penalize if expected effect is substantial
        # Penalty if effect is too weak (should see at least some change)
        ratio = act_mag / exp_mag
        min_effect = max(0.0, 1.0 - ratio) ** 2

        # Penalty if effect is in wrong direction (e.g., decrease when should increase)
        same_sign = (expected > 0 and actual > 0) or (expected < 0 and actual < 0) or act_mag < 0.01
        if not same_sign:
            wrong_dir = (act_mag / exp_mag) ** 2

    return mse + near_zero_weight * near_zero + min_effect_weight * min_effect + wrong_direction_weight * wrong_dir
