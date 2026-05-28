"""
Loss functions for parameter optimization (5-population NDNF model).

All loss terms are squared log-relative-errors (fold-change in dex):

    L_term = ( log( max(actual, EPS) / target ) )^2

This is target-normalised (a fold-change) AND symmetric: a 2× over- or
under-shoot contribute equally, and as sim → 0 the loss diverges (no
saturation), so the optimiser cannot park populations at zero. EPS=0.01 Hz
is a numerical floor.

No absolute "near-zero" / "wrong direction" / "min effect" ad-hoc penalties.

Buckets exposed by `loss_from_means_normalized` / `loss_from_ko_normalized`:
    - `base`           : 5 baseline firing-rate targets
    - `global_ko`      : PYR rate under each global KO (α7, α5, β2)
    - `selective_ko`   : NDNF / PV rate under their selective α7 KOs
    - `drug`           : per-drug measurements (Stage 2)

Per-bucket CLI weights:  --weight_base, --weight_global_ko,
                          --weight_selective_ko, --weight_drug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .simulation import NoiseType


@dataclass(frozen=True)
class DrugTarget:
    """Measured firing rate under a drug condition.

    Each `DrugTarget` carries the drug label and the value measured on a
    specific cell type (which population's rate to compare against).
    """
    drug: str         # "MLA" | "PNU" | "nicotine"
    population: str   # "PYR" | "SOM" | "PV" | "VIP" | "NDNF"
    target_hz: float


@dataclass(frozen=True)
class TargetRates:
    """Target mean firing rates for optimization (5 populations).

    Global KO targets (all measured on PYR):
      - alpha7_ko_pyr: global α7-KO (all per-cell α7 zeroed)
      - alpha5_ko_pyr, beta2_ko_pyr: global α5 / β2 KO

    Cell-type-selective α7 KO targets (measured on the *deleted* cell type):
      - alpha7_ndnf_ko_ndnf: NDNF firing rate under NDNF-selective α7 KO
      - alpha7_pv_ko_pv:     PV firing rate   under PV-selective   α7 KO

    Drug-condition targets (Stage 2):
      - drug_targets: list of (drug, population, target_hz) tuples.
        Each one contributes ((actual - target_hz)/target_hz)^2 to the loss.
    """
    mean_r_pyr: float
    mean_r_som: float
    mean_r_pv: float
    mean_r_vip: float
    mean_r_ndnf: float = 0.0

    alpha7_ko_pyr: Optional[float] = None
    alpha5_ko_pyr: Optional[float] = None
    beta2_ko_pyr: Optional[float] = None
    alpha7_ndnf_ko_ndnf: Optional[float] = None
    alpha7_pv_ko_pv: Optional[float] = None

    drug_targets: tuple[DrugTarget, ...] = field(default_factory=tuple)

    def as_array(self) -> np.ndarray:
        """Return base targets as array [pyr, som, pv, vip, ndnf]."""
        return np.array(
            [self.mean_r_pyr, self.mean_r_som, self.mean_r_pv,
             self.mean_r_vip, self.mean_r_ndnf],
            dtype=float,
        )


@dataclass(frozen=True)
class FitConfig:
    """Configuration for simulation and optimization."""
    T_ms: float = 2500.0
    dt_ms: float = 0.1
    burn_in_ms: float = 1200.0
    window_ms: float = 500.0
    record_dt_ms: float = 2.0

    n_trials: int = 8
    init_rate_scale: float = 0.2

    noise_type: NoiseType = "white"
    tau_noise_ms: float = 5.0

    max_rate: float = 200.0


# Numerical floor on simulated rates (Hz). Below this we treat the rate as
# `LOG_EPS` for the log-loss to keep the gradient sane while still strongly
# penalising silenced solutions.
LOG_EPS = 0.01


def _log_sq(actual: float, target: float, *, eps: float = LOG_EPS) -> float:
    """Squared log fold-change between actual and target.

    L = ( log( max(actual, eps) / max(target, eps) ) )^2

    Symmetric in over/undershoot. Returns 0 when actual == target,
    grows unboundedly as actual → 0 or actual → ∞.
    """
    a = max(float(actual), eps)
    t = max(float(target), eps)
    return float(np.log(a / t) ** 2)


def loss_from_means_normalized(means: np.ndarray, target: TargetRates) -> float:
    """Sum of per-population log-fold-change² over the 5 baseline rates."""
    tgt = target.as_array()
    out = 0.0
    for i in range(len(tgt)):
        out += _log_sq(float(means[i]), float(tgt[i]))
    return out


def loss_from_ko_normalized(actual: float, target_value: float) -> float:
    """Single log-fold-change² for one KO/drug measurement."""
    return _log_sq(actual, target_value)


def drug_loss(means: np.ndarray, drug_targets: list[DrugTarget],
              drug_name: str) -> float:
    """Sum of squared log-fold-changes for every DrugTarget of `drug_name`.

    `means` is a (5,) array indexed [PYR, SOM, PV, VIP, NDNF].
    Used by Stage 2 (per-drug receptor-activation fitting).
    """
    pop_idx = {"PYR": 0, "SOM": 1, "PV": 2, "VIP": 3, "NDNF": 4}
    total = 0.0
    for dt in drug_targets:
        if dt.drug != drug_name:
            continue
        total += _log_sq(float(means[pop_idx[dt.population]]), dt.target_hz)
    return total
