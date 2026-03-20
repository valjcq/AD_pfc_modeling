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

from dataclasses import dataclass, fields, replace
from typing import Any, Optional
from pathlib import Path

import nevergrad as ng
import numpy as np
from tqdm import tqdm

from .params import CircuitParams, ParamBound
from .simulation import simulate_circuit, mean_rates, NoiseType
from .loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr, jacobian_connectivity_penalty
from .io import log_best_result, save_params_json


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


def _build_conditions(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    rng: np.random.Generator,
) -> list[tuple[str, CircuitParams, FitConfig, int]]:
    """Build list of (name, params, cfg, seed) conditions to simulate."""
    conditions: list[tuple[str, CircuitParams, FitConfig, int]] = [
        ("base", params, cfg, int(rng.integers(0, 2**31 - 1))),
    ]
    # alpha7 KO: remove alpha7-mediated currents AND GABA enhancement
    if target.alpha7_ko_pyr is not None:
        conditions.append(("alpha7_ko", replace(params, act_alpha7=0.0, g_alpha7=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    # alpha5 KO: remove alpha5 contribution to VIP
    if target.alpha5_ko_pyr is not None:
        conditions.append(("alpha5_ko", replace(params, act_alpha5=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    # beta2 KO: remove beta2 contribution to SOM
    if target.beta2_ko_pyr is not None:
        conditions.append(("beta2_ko", replace(params, act_beta2=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    return conditions


def _loss_from_results(
    results: list[ConditionResult],
    target: TargetRates,
    cfg: FitConfig,
    params: CircuitParams,
) -> tuple[float, np.ndarray, KOMeans]:
    """Compute total loss from a list of condition simulation results."""
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
            float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        total += loss_from_ko_pyr(
            float(ko_means.beta2_ko[0]), target.beta2_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )

    total += jacobian_connectivity_penalty(params, base_means)

    return total, base_means, ko_means


def evaluate_params(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    *,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[float, np.ndarray, KOMeans]:
    """Evaluate a parameter set under baseline and knockout conditions."""
    conditions = _build_conditions(params, target, cfg, rng)
    if executor is not None and len(conditions) > 1:
        results = list(executor.map(run_condition, conditions))
    else:
        results = [run_condition(c) for c in conditions]
    return _loss_from_results(results, target, cfg, params)


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
        # Clamp the base value into [lo, hi] so it's always a valid init point.
        # This makes --resume with CMA-ES warm-start from the loaded best params
        # rather than from the geometric/arithmetic centre of the search space.
        raw = float(getattr(base, name))
        init = float(np.clip(raw, bound.lo, bound.hi))
        if bound.mode == "log" and bound.lo > 0:
            params_dict[name] = ng.p.Log(lower=bound.lo, upper=bound.hi, init=init)
        else:
            params_dict[name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi, init=init)

    return ng.p.Dict(**params_dict)


def params_from_ng_dict(ng_dict: dict[str, Any], base: CircuitParams) -> CircuitParams:
    """Convert Nevergrad dict to CircuitParams."""
    allowed = {f.name for f in fields(CircuitParams)}
    clean = {k: v for k, v in ng_dict.items() if k in allowed}
    return replace(base, **clean)


def _build_optimizer(
    name: str,
    parametrization: ng.p.Dict,
    budget: int,
    num_workers: int,
) -> Any:
    """Instantiate a Nevergrad optimizer by name.

    Available options:
    - ``de``       — TwoPointsDE: derivative-free differential evolution (default).
                     Good global explorer, robust on discontinuous landscapes.
    - ``cma``      — CMA-ES: covariance-matrix adaptation evolution strategy.
                     Best local refiner for ~10–100 continuous parameters; learns
                     parameter correlations and converges fast once in a good basin.
    - ``chaining`` — TwoPointsDE (global) → Nelder-Mead (local refinement).
                     Matches the pipeline from the reference paper: gradient-free
                     global search followed by a gradient-free local optimizer.
    - ``auto``     — NGOpt: Nevergrad's meta-optimizer that selects the algorithm
                     based on problem dimension and budget automatically.
    """
    if name == "de":
        return ng.optimizers.TwoPointsDE(
            parametrization=parametrization, budget=budget, num_workers=num_workers,
        )
    elif name == "cma":
        return ng.optimizers.CMA(
            parametrization=parametrization, budget=budget, num_workers=num_workers,
        )
    elif name == "chaining":
        # DE converges fast (~3000-5000 steps); cap its share at min(budget//5, 10000).
        # Nelder-Mead then refines from the best point found by DE.
        de_budget = max(500, min(budget // 5, 10000))
        ChainedCls = ng.optimizers.Chaining(
            [ng.optimizers.TwoPointsDE, ng.optimizers.NelderMead],
            [de_budget],
        )
        return ChainedCls(
            parametrization=parametrization, budget=budget, num_workers=num_workers,
        )
    elif name == "auto":
        return ng.optimizers.NGOpt(
            parametrization=parametrization, budget=budget, num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: de, cma, chaining, auto.")


def nevergrad_optimize(
    target: TargetRates,
    *,
    base: CircuitParams,
    bounds: dict[str, ParamBound],
    fit_cfg: FitConfig,
    n_samples: int,
    top_k: int,
    seed: Optional[int],
    optimizer: str = "de",
    freeze: Optional[set[str]] = None,
    early_stop_loss: Optional[float] = 1e-4,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    save_best_json: Optional[str] = None,
    step_offset: int = 0,
    append_log: bool = False,
) -> list[Candidate]:
    """
    Run Nevergrad optimization to find parameters matching target firing rates.

    Optimizer choices (``optimizer`` argument):
    - ``de``       — TwoPointsDE, derivative-free differential evolution (default).
    - ``cma``      — CMA-ES, learns parameter correlations; fast local convergence.
    - ``chaining`` — TwoPointsDE (global) → Nelder-Mead (local). Matches the
                     reference paper pipeline.
    - ``auto``     — NGOpt, Nevergrad's automatic algorithm selector.

    The optimization loop:
    1. Ask optimizer for a candidate parameter set
    2. Run simulations under base + KO conditions
    3. Compute total loss
    4. Tell optimizer the loss
    5. Track top-k best candidates
    6. Optionally stop early if loss is below threshold
    """
    rng = np.random.default_rng(seed)

    parametrization = build_nevergrad_parametrization(base, bounds, freeze)
    ng_optimizer = _build_optimizer(optimizer, parametrization, n_samples, num_workers=1)

    if seed is not None:
        ng_optimizer.parametrization.random_state = np.random.RandomState(seed)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not append_log:
            open(log_file, "w", encoding="utf-8").close()

    if save_best_json:
        Path(save_best_json).parent.mkdir(parents=True, exist_ok=True)

    if optimizer == "chaining":
        de_budget = max(500, min(n_samples // 5, 10000))
        print(f"Optimizer: chaining (DE for {de_budget} steps → Nelder-Mead for {n_samples - de_budget} steps)")
    else:
        print(f"Optimizer: {optimizer}")

    best: list[Candidate] = []
    last_step = 0
    stopped_early = False

    pbar = tqdm(range(1, n_samples + 1), desc="Optimizing", unit="step")
    for step in pbar:
        last_step = step

        x = ng_optimizer.ask()
        p = params_from_ng_dict(x.value, base)
        cond_results = [run_condition(c) for c in _build_conditions(p, target, fit_cfg, rng)]

        L, means, ko_means = _loss_from_results(cond_results, target, fit_cfg, p)
        ng_optimizer.tell(x, L)

        prev_best_loss = best[0].loss if best else float("inf")
        cand = Candidate(loss=L, means=means, ko_means=ko_means, params=p)
        if len(best) < top_k:
            best.append(cand)
            best.sort(key=lambda c: c.loss)
        elif L < best[-1].loss:
            best[-1] = cand
            best.sort(key=lambda c: c.loss)

        pbar.set_postfix(loss=f"{best[0].loss:.4g}" if best else "N/A", step=step)

        if save_best_json and best and best[0].loss < prev_best_loss:
            save_params_json(save_best_json, best[0].params)

        if log_file and step % log_interval == 0 and best:
            _log_candidate(log_file, step + step_offset, best[0], target)

        if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
            if log_file:
                _log_candidate(log_file, step + step_offset, best[0], target)
            stopped_early = True
            break

    pbar.close()

    if log_file and best and (not stopped_early) and last_step % log_interval != 0:
        _log_candidate(log_file, last_step + step_offset, best[0], target)

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
