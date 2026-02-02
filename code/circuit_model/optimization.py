"""
Nevergrad-based parameter optimization.

This module contains:
- KOMeans: Container for knockout condition results
- Candidate: Optimization result candidate
- run_trials: Run multiple simulation trials
- run_condition: Run a single condition (for parallel execution)
- evaluate_params: Evaluate parameters under all conditions
- build_nevergrad_parametrization: Build Nevergrad search space
- params_from_ng_dict: Convert Nevergrad dict to CircuitParams
- nevergrad_optimize: Main optimization function
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from typing import Any, Optional
import os

import nevergrad as ng
import numpy as np

from .params import CircuitParams, ParamBound
from .simulation import simulate_circuit, mean_rates, NoiseType
from .loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr
from .io import log_best_result


@dataclass
class KOMeans:
    """Container for knockout condition mean firing rates."""
    alpha7_ko: Optional[np.ndarray] = None
    alpha5_ko: Optional[np.ndarray] = None
    beta2_ko: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Candidate:
    """A candidate parameter set with its evaluation results."""
    loss: float
    means: np.ndarray
    ko_means: KOMeans
    params: CircuitParams


# Type alias for condition results
ConditionResult = tuple[str, bool, np.ndarray]  # (name, ok, means)


def run_trials(params: CircuitParams, cfg: FitConfig, base_seed: int) -> tuple[bool, np.ndarray]:
    """
    Run multiple simulation trials and return mean firing rates.

    Returns:
        Tuple of (success, means) where success is False if any trial
        produced invalid results (NaN, too high rates, etc.)
    """
    rng = np.random.default_rng(base_seed)
    means_trials: list[np.ndarray] = []

    for _ in range(cfg.n_trials):
        r0 = cfg.init_rate_scale * rng.lognormal(mean=0.0, sigma=0.6, size=4)
        seed = int(rng.integers(0, 2**31 - 1))

        res = simulate_circuit(
            params,
            T_ms=cfg.T_ms,
            dt_ms=cfg.dt_ms,
            r0=r0,
            seed=seed,
            noise_type=cfg.noise_type,
            tau_noise_ms=cfg.tau_noise_ms,
        )
        m = mean_rates(res, burn_in_ms=cfg.burn_in_ms, window_ms=cfg.window_ms)

        if not np.all(np.isfinite(m)) or np.any(m > cfg.max_rate):
            return False, m

        means_trials.append(m)

    means = np.mean(np.stack(means_trials, axis=0), axis=0)
    return True, means


def run_condition(args: tuple[str, CircuitParams, FitConfig, int]) -> ConditionResult:
    """Run a single condition (for use with ProcessPoolExecutor)."""
    name, params, cfg, seed = args
    ok, means = run_trials(params, cfg, seed)
    return name, ok, means


def evaluate_params(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    *,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[float, np.ndarray, KOMeans]:
    """
    Evaluate a parameter set under baseline and knockout conditions.

    Runs simulations for:
    1. Base condition (all receptors active)
    2. alpha7 KO (if target specified): removes alpha7 currents AND GABA enhancement
    3. alpha5 KO (if target specified): removes alpha5 currents to VIP
    4. beta2 KO (if target specified): removes beta2 currents to SOM

    Returns total loss, base mean rates, and knockout mean rates.
    """
    conditions: list[tuple[str, CircuitParams, FitConfig, int]] = [
        ("base", params, cfg, int(rng.integers(0, 2**31 - 1))),
    ]

    # alpha7 KO: Simulates genetic knockout or pharmacological blockade of alpha7 nAChRs
    # Effects: (1) Remove alpha7-mediated currents to PV and SOM
    #          (2) Remove alpha7-dependent GABA enhancement (g_alpha7 -> 0)
    # Typically causes disinhibition -> increased PYR firing
    if target.alpha7_ko_pyr is not None:
        conditions.append(
            (
                "alpha7_ko",
                replace(params, act_alpha7=0.0, g_alpha7=0.0),
                cfg,
                int(rng.integers(0, 2**31 - 1)),
            )
        )

    # alpha5 KO: Removes alpha5 nAChR contribution to VIP
    # Effects: Reduced VIP activity -> less SOM inhibition -> more SOM -> less PYR
    if target.alpha5_ko_pyr is not None:
        conditions.append(("alpha5_ko", replace(params, act_alpha5=0.0), cfg, int(rng.integers(0, 2**31 - 1))))

    # beta2 KO: Removes beta2 nAChR contribution to SOM
    # Effects: Reduced SOM activity -> less dendritic inhibition -> increased PYR
    if target.beta2_ko_pyr is not None:
        conditions.append(("beta2_ko", replace(params, act_beta2=0.0), cfg, int(rng.integers(0, 2**31 - 1))))

    if executor is not None and len(conditions) > 1:
        results = list(executor.map(run_condition, conditions))
    else:
        results = [run_condition(c) for c in conditions]

    ko_means = KOMeans()
    base_means = np.zeros(4, dtype=float)

    for name, ok, means in results:
        if not ok:
            return 1e9, base_means, ko_means
        if name == "base":
            base_means = means
        elif name == "alpha7_ko":
            ko_means.alpha7_ko = means
        elif name == "alpha5_ko":
            ko_means.alpha5_ko = means
        elif name == "beta2_ko":
            ko_means.beta2_ko = means

    total = loss_from_means(base_means, target)
    base_pyr = float(base_means[0])

    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.alpha7_ko[0]),
            target.alpha7_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.alpha5_ko[0]),
            target.alpha5_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.beta2_ko[0]),
            target.beta2_ko_pyr,
            base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )

    return total, base_means, ko_means


def build_nevergrad_parametrization(
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    freeze: Optional[set[str]] = None,
) -> ng.p.Dict:
    """Build Nevergrad parametrization from bounds."""
    freeze = freeze or set()
    params_dict: dict[str, Any] = {}

    for f in fields(CircuitParams):
        name = f.name

        if name in freeze or name not in bounds:
            params_dict[name] = getattr(base, name)
            continue

        bound = bounds[name]
        if bound.mode == "log" and bound.lo > 0:
            params_dict[name] = ng.p.Log(lower=bound.lo, upper=bound.hi)
        else:
            params_dict[name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi)

    return ng.p.Dict(**params_dict)


def params_from_ng_dict(ng_dict: dict[str, Any], base: CircuitParams) -> CircuitParams:
    """Convert Nevergrad dict to CircuitParams."""
    allowed = {f.name for f in fields(CircuitParams)}
    clean = {k: v for k, v in ng_dict.items() if k in allowed}
    return replace(base, **clean)


def nevergrad_optimize(
    target: TargetRates,
    *,
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    fit_cfg: FitConfig,
    n_samples: int,
    top_k: int,
    seed: Optional[int],
    freeze: Optional[set[str]] = None,
    early_stop_loss: Optional[float] = 1e-4,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    n_workers: Optional[int] = None,
) -> list[Candidate]:
    """
    Run Nevergrad optimization to find parameters matching target firing rates.

    Uses TwoPointsDE (Two-Point Differential Evolution) optimizer, which:
    - Is derivative-free (good for noisy, discontinuous loss landscapes)
    - Uses differential evolution with two-point crossover
    - Balances exploration and exploitation

    The optimization loop:
    1. Ask optimizer for a candidate parameter set
    2. Run simulations under base + KO conditions
    3. Compute total loss
    4. Tell optimizer the loss
    5. Track top-k best candidates
    6. Optionally stop early if loss is below threshold

    Parameters can be frozen (excluded from optimization) and searched
    in linear or log space depending on their nature.
    """
    rng = np.random.default_rng(seed)

    parametrization = build_nevergrad_parametrization(base, bounds, freeze)
    optimizer = ng.optimizers.TwoPointsDE(
        parametrization=parametrization,
        budget=n_samples,
        num_workers=1,
    )

    if seed is not None:
        optimizer.parametrization.random_state = np.random.RandomState(seed)

    n_conditions = 1 + sum(
        [
            target.alpha7_ko_pyr is not None,
            target.alpha5_ko_pyr is not None,
            target.beta2_ko_pyr is not None,
        ]
    )
    use_parallel = n_conditions > 1 and (n_workers is None or n_workers not in (0, 1))

    if n_workers is None:
        max_workers = min(n_conditions, os.cpu_count() or 4)
    else:
        max_workers = min(n_conditions, n_workers)

    if log_file:
        open(log_file, "w", encoding="utf-8").close()

    pool_cm = ProcessPoolExecutor(max_workers=max_workers) if use_parallel else nullcontext(None)

    best: list[Candidate] = []

    with pool_cm as executor:
        if use_parallel:
            print(f"Using {max_workers} workers for {n_conditions} conditions")

        last_step = 0
        stopped_early = False

        for step in range(1, n_samples + 1):
            last_step = step
            x = optimizer.ask()
            params = params_from_ng_dict(x.value, base)

            L, means, ko_means = evaluate_params(params, target, fit_cfg, rng=rng, executor=executor)
            optimizer.tell(x, L)

            cand = Candidate(loss=L, means=means, ko_means=ko_means, params=params)

            ko_str = ""
            if ko_means.alpha7_ko is not None:
                ko_str += f" a7KO_pyr={ko_means.alpha7_ko[0]:.4g}"
            if ko_means.alpha5_ko is not None:
                ko_str += f" a5KO_pyr={ko_means.alpha5_ko[0]:.4g}"
            if ko_means.beta2_ko is not None:
                ko_str += f" b2KO_pyr={ko_means.beta2_ko[0]:.4g}"

            print(
                f"[{step}/{n_samples}] loss={L:.6g} "
                f"means=[pyr={means[0]:.4g}, som={means[1]:.4g}, pv={means[2]:.4g}, vip={means[3]:.4g}]"
                f"{ko_str}"
            )

            if len(best) < top_k:
                best.append(cand)
                best.sort(key=lambda c: c.loss)
            elif L < best[-1].loss:
                best[-1] = cand
                best.sort(key=lambda c: c.loss)

            if log_file and step % log_interval == 0 and best:
                _log_candidate(log_file, step, best[0], target)

            if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
                if log_file:
                    _log_candidate(log_file, step, best[0], target)
                stopped_early = True
                break

        if log_file and best and (not stopped_early) and last_step % log_interval != 0:
            _log_candidate(log_file, last_step, best[0], target)

    return best


def _log_candidate(path: str, step: int, cand: Candidate, target: TargetRates) -> None:
    """Helper to log a candidate result."""
    means_dict = {
        "pyr": float(cand.means[0]),
        "som": float(cand.means[1]),
        "pv": float(cand.means[2]),
        "vip": float(cand.means[3]),
    }
    ko_means_dict = {
        "alpha7_ko": cand.ko_means.alpha7_ko.tolist() if cand.ko_means.alpha7_ko is not None else None,
        "alpha5_ko": cand.ko_means.alpha5_ko.tolist() if cand.ko_means.alpha5_ko is not None else None,
        "beta2_ko": cand.ko_means.beta2_ko.tolist() if cand.ko_means.beta2_ko is not None else None,
    }
    log_best_result(path, step, cand.loss, means_dict, ko_means_dict, cand.params, target)
