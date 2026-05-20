"""
Nevergrad-based parameter optimization.

This module contains:
- KOMeans: Container for knockout condition mean firing rates
- Candidate: Optimization result candidate
- LossBreakdown: Per-component loss breakdown returned by the loss helpers
- run_trials: Run multiple simulation trials
- run_condition: Run a single condition (for parallel execution)
- evaluate_params: Evaluate parameters under all conditions
- build_nevergrad_parametrization: Build Nevergrad search space
- params_from_ng_dict: Convert Nevergrad dict to CircuitParams
- nevergrad_optimize: Main optimization function
"""

from __future__ import annotations

<<<<<<< HEAD
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Any, Optional
=======
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from typing import Any, Optional
import os
>>>>>>> origin/main
from pathlib import Path

import nevergrad as ng
import numpy as np
from tqdm import tqdm

from .params import CircuitParams, ParamBound
from .simulation import simulate_circuit, mean_rates, NoiseType
from .loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr, jacobian_connectivity_penalty, ach_ratio_penalty, transfer_function_slope
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
    breakdown: Optional["LossBreakdown"] = None
    simulated: bool = False  # True if means come from an actual simulation


# Type alias for condition results
ConditionResult = tuple[str, bool, np.ndarray]  # (name, ok, means)


@dataclass
class LossBreakdown:
    """Breakdown of loss components."""
    firing_rate: float
    ko_firing_rate: float
    jacobian: float
    turing: float
    ach_ratio: float
    total: float

    def __str__(self) -> str:
        return (f"loss=[fr={self.firing_rate:.3g}, ko={self.ko_firing_rate:.3g}, "
                f"jac={self.jacobian:.3g}, turing={self.turing:.3g}, "
                f"ach={self.ach_ratio:.3g}, total={self.total:.3g}]")


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
        # KO conditions always simulated for display; only enter the loss if target has KO fields set
        ("alpha7_ko", replace(params, act_alpha7=0.0, g_alpha7=0.0), cfg, int(rng.integers(0, 2**31 - 1))),
        ("alpha5_ko", replace(params, act_alpha5=0.0), cfg, int(rng.integers(0, 2**31 - 1))),
        ("beta2_ko",  replace(params, act_beta2=0.0),  cfg, int(rng.integers(0, 2**31 - 1))),
    ]
<<<<<<< HEAD
=======
    # alpha7 KO: remove alpha7-mediated currents AND GABA enhancement
    if target.alpha7_ko_pyr is not None:
        conditions.append(("alpha7_ko", replace(params, act_alpha7=0.0, g_alpha7=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    # alpha5 KO: remove alpha5 contribution to VIP
    if target.alpha5_ko_pyr is not None:
        conditions.append(("alpha5_ko", replace(params, act_alpha5=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
    # beta2 KO: remove beta2 contribution to SOM
    if target.beta2_ko_pyr is not None:
        conditions.append(("beta2_ko", replace(params, act_beta2=0.0), cfg, int(rng.integers(0, 2**31 - 1))))
>>>>>>> origin/main
    return conditions


def _loss_from_results(
    results: list[ConditionResult],
    target: TargetRates,
    cfg: FitConfig,
<<<<<<< HEAD
    params: CircuitParams,
    *,
    squared_loss: bool = True,
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_w_inter_ref: float = 10.0,
    turing_cue_scale: float = 0.4,
    ach_ratio_weight: float = 2.0,
) -> tuple[float, np.ndarray, KOMeans, LossBreakdown]:
    """Compute total loss from a list of condition simulation results.
    
    Returns:
        Tuple of (total_loss, base_means, ko_means, breakdown).
    """
=======
) -> tuple[float, np.ndarray, KOMeans]:
    """Compute total loss from a list of condition simulation results."""
>>>>>>> origin/main
    ko_means = KOMeans()
    base_means = np.zeros(4, dtype=float)

    for name, ok, means in results:
        if not ok:
            return 1e9, base_means, ko_means, LossBreakdown(1e9, 0., 0., 0., 0., 1e9)
        if name == "base":
            base_means = means
        elif name == "alpha7_ko":
            ko_means.alpha7_ko = means
        elif name == "alpha5_ko":
            ko_means.alpha5_ko = means
        elif name == "beta2_ko":
            ko_means.beta2_ko = means

    # Firing rate loss
    fr_loss = loss_from_means(base_means, target, squared=squared_loss)
    base_pyr = float(base_means[0])

    # Knockout penalty
    ko_loss = 0.0
    n_ko = 0
    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
<<<<<<< HEAD
        ko_loss += loss_from_ko_pyr(
=======
        total += loss_from_ko_pyr(
>>>>>>> origin/main
            float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
<<<<<<< HEAD
        ko_loss += loss_from_ko_pyr(
=======
        total += loss_from_ko_pyr(
>>>>>>> origin/main
            float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
<<<<<<< HEAD
        ko_loss += loss_from_ko_pyr(
=======
        total += loss_from_ko_pyr(
>>>>>>> origin/main
            float(ko_means.beta2_ko[0]), target.beta2_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1

    # Normalise KO loss by number of KO conditions
    ko_firing_rate = ko_loss / n_ko if n_ko > 0 else 0.0

    # Jacobian connectivity penalty
    jac_loss = jacobian_connectivity_penalty(params, base_means) * jacobian_weight

    # Turing instability penalty
    turing_loss = 0.0
    if turing_weight > 0.0:
        from dataclasses import replace as _replace
        w = turing_w_inter_ref
        slope_rest = transfer_function_slope(params, base_means, population="PYR")
        params_cue = _replace(params, I0_pyr=turing_cue_scale * params.I0_pyr)
        slope_cue  = transfer_function_slope(params_cue, base_means, population="PYR")
        t_loss = (max(0.0, slope_rest * w - (1.0 - turing_margin)) ** 2
                + max(0.0, 1.0 + turing_margin - slope_cue * w) ** 2)
        turing_loss = turing_weight * t_loss

    # ACh β2/α7 ratio penalty (Koukouli et al. 2025: β2 should be ~35× α7 on SOM)
    ach_loss = ach_ratio_penalty(params, weight=ach_ratio_weight)

    total = fr_loss + ko_firing_rate + jac_loss + turing_loss + ach_loss
    breakdown = LossBreakdown(firing_rate=fr_loss, ko_firing_rate=ko_firing_rate,
                               jacobian=jac_loss, turing=turing_loss, ach_ratio=ach_loss, total=total)

    return total, base_means, ko_means, breakdown


def evaluate_params(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    *,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor] = None,
    squared_loss: bool = True,
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_w_inter_ref: float = 10.0,
    turing_cue_scale: float = 0.4,
    ach_ratio_weight: float = 2.0,
) -> tuple[float, np.ndarray, KOMeans, LossBreakdown]:
    """Evaluate a parameter set under baseline and knockout conditions.
    
    Returns:
        Tuple of (total_loss, means, ko_means, breakdown).
    """
    conditions = _build_conditions(params, target, cfg, rng)
    if executor is not None and len(conditions) > 1:
        results = list(executor.map(run_condition, conditions))
    else:
        results = [run_condition(c) for c in conditions]
    return _loss_from_results(results, target, cfg, params, squared_loss=squared_loss, jacobian_weight=jacobian_weight, turing_weight=turing_weight, turing_margin=turing_margin, turing_w_inter_ref=turing_w_inter_ref, turing_cue_scale=turing_cue_scale, ach_ratio_weight=ach_ratio_weight)


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
    return _loss_from_results(results, target, cfg)


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
        # DE runs for a fixed 5000 steps (global search); Nelder-Mead refines the rest.
        de_budget = min(5000, budget - 1)
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
    log_file: Optional[str] = None,
    log_interval: int = 50,
    save_best_json: Optional[str] = None,
    step_offset: int = 0,
    append_log: bool = False,
<<<<<<< HEAD
    squared_loss: bool = True,
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_w_inter_ref: float = 10.0,
    turing_cue_scale: float = 0.4,
    ach_ratio_weight: float = 2.0,
    bistable_cfg: Optional[Any] = None,
=======
>>>>>>> origin/main
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

    n_conditions = 1 + sum([
        target.alpha7_ko_pyr is not None,
        target.alpha5_ko_pyr is not None,
        target.beta2_ko_pyr is not None,
    ])

    # Compute how many candidates to evaluate in parallel (batch mode).
    # total_workers = total concurrent simulation slots;
    # batch_size = candidates per step = total_workers // n_conditions.
    if n_workers in (None, 0):
        total_workers = os.cpu_count() or 4
    elif n_workers == 1:
        total_workers = 1
    else:
        total_workers = n_workers

    use_parallel = total_workers > 1
    batch_size = max(1, total_workers // n_conditions)
    max_workers = batch_size * n_conditions

    parametrization = build_nevergrad_parametrization(base, bounds, freeze)
<<<<<<< HEAD
    ng_optimizer = _build_optimizer(optimizer, parametrization, n_samples, num_workers=1)

    if seed is not None:
        ng_optimizer.parametrization.random_state = np.random.RandomState(seed)
=======
    optimizer = ng.optimizers.TwoPointsDE(
        parametrization=parametrization,
        budget=n_samples,
        num_workers=batch_size,
    )

    if seed is not None:
        optimizer.parametrization.random_state = np.random.RandomState(seed)
>>>>>>> origin/main

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not append_log:
            open(log_file, "w", encoding="utf-8").close()
<<<<<<< HEAD
=======

    if save_best_json:
        Path(save_best_json).parent.mkdir(parents=True, exist_ok=True)
>>>>>>> origin/main

    if save_best_json:
        Path(save_best_json).parent.mkdir(parents=True, exist_ok=True)

    if optimizer == "chaining":
        de_budget = max(500, min(n_samples // 5, 10000))
        print(f"Optimizer: chaining (DE for {de_budget} steps → Nelder-Mead for {n_samples - de_budget} steps)")
    else:
        print(f"Optimizer: {optimizer}")

    if use_parallel:
        print(f"Using {max_workers} workers ({batch_size} candidates × {n_conditions} conditions)")

    best: list[Candidate] = []
    last_step = 0

<<<<<<< HEAD
    interrupted = False
    pbar = tqdm(range(1, n_samples + 1), desc="Optimizing", unit="step")
    try:
        for step in pbar:
            last_step = step

            x = ng_optimizer.ask()
            p = params_from_ng_dict(x.value, base)
            sim_ran = False  # Track if simulation ran at this step

            if bistable_cfg is not None:
                # Bistable mode: use bistable loss directly with component breakdown.
                # We no longer run periodic simulation inside the main loop.
                from .bistable_loss import bistable_loss as _bistable_loss
                L, bistable_components = _bistable_loss(p, bistable_cfg, return_components=True)
                L = float(L)
                means = np.full(4, np.nan)
                ko_means = KOMeans()

                # For bistable mode, pass the detailed component breakdown dict
                breakdown = bistable_components
            else:
                # Standard mode: simulate and compute loss
                cond_results = [run_condition(c) for c in _build_conditions(p, target, fit_cfg, rng)]
                L, means, ko_means, breakdown = _loss_from_results(cond_results, target, fit_cfg, p, squared_loss=squared_loss, jacobian_weight=jacobian_weight, turing_weight=turing_weight, turing_margin=turing_margin, turing_w_inter_ref=turing_w_inter_ref, turing_cue_scale=turing_cue_scale, ach_ratio_weight=ach_ratio_weight)
                sim_ran = True  # Standard mode always simulates

            ng_optimizer.tell(x, L)

            prev_best_loss = best[0].loss if best else float("inf")
            cand = Candidate(loss=L, means=means, ko_means=ko_means, params=p, breakdown=breakdown, simulated=sim_ran)
            if len(best) < top_k:
                best.append(cand)
                best.sort(key=lambda c: c.loss)
            elif L < best[-1].loss:
                best[-1] = cand
                best.sort(key=lambda c: c.loss)

            if best[0].loss < prev_best_loss:
                # In bistable mode, log immediately when we find a new best.
                if bistable_cfg is not None and log_file:
                    # Ensure best[0] has firing rates before logging
                    best_to_log = _ensure_means_from_simulation(best[0], target, fit_cfg, rng)
                    _log_candidate(log_file, step + step_offset, best_to_log, target, best_to_log.breakdown)

            # Update progress bar: show best loss only
            pbar.set_postfix({"loss": f"{best[0].loss:.4g}" if best else "N/A"})
=======
    with pool_cm as executor:
        last_step = 0
        stopped_early = False
        # Steps counted per candidate, not per batch
        pbar = tqdm(range(1, n_samples + 1, batch_size), desc="Optimizing", unit="step")

        for step in pbar:
            last_step = step

            # Ask batch_size candidates from the optimizer
            xs = [optimizer.ask() for _ in range(batch_size)]
            params_list = [params_from_ng_dict(x.value, base) for x in xs]

            if use_parallel:
                # Submit all candidate × condition tasks at once for maximum throughput
                tagged_futures: list[tuple[int, Future[ConditionResult]]] = []
                for i, p in enumerate(params_list):
                    for cond in _build_conditions(p, target, fit_cfg, rng):
                        tagged_futures.append((i, executor.submit(run_condition, cond)))

                results_by_cand: list[list[ConditionResult]] = [[] for _ in range(batch_size)]
                for i, fut in tagged_futures:
                    results_by_cand[i].append(fut.result())
            else:
                results_by_cand = [
                    [run_condition(c) for c in _build_conditions(p, target, fit_cfg, rng)]
                    for p in params_list
                ]

            prev_best_loss = best[0].loss if best else float("inf")

            for x, p, cond_results in zip(xs, params_list, results_by_cand):
                L, means, ko_means = _loss_from_results(cond_results, target, fit_cfg)
                optimizer.tell(x, L)

                cand = Candidate(loss=L, means=means, ko_means=ko_means, params=p)
                if len(best) < top_k:
                    best.append(cand)
                    best.sort(key=lambda c: c.loss)
                elif L < best[-1].loss:
                    best[-1] = cand
                    best.sort(key=lambda c: c.loss)

            pbar.set_postfix(loss=f"{best[0].loss:.4g}" if best else "N/A", step=step)
>>>>>>> origin/main

            if save_best_json and best and best[0].loss < prev_best_loss:
                save_params_json(save_best_json, best[0].params)

            if log_file and step % log_interval == 0 and best:
<<<<<<< HEAD
                # In bistable mode, ensure firing rates are available before logging
                best_to_log = best[0]
                if bistable_cfg is not None:
                    best_to_log = _ensure_means_from_simulation(best[0], target, fit_cfg, rng)
                _log_candidate(log_file, step + step_offset, best_to_log, target, best_to_log.breakdown)
                # Generate loss evolution plots every log_interval steps
                try:
                    _generate_loss_plots(log_file)
                except Exception:
                    # Don't break optimization if plotting fails
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\nOptimization interrupted by user (Ctrl+C). Finalizing best-so-far results...")
    finally:
        pbar.close()

    if log_file and best and last_step % log_interval != 0:
        # In bistable mode, ensure firing rates are available before logging
        best_to_log = best[0]
        if bistable_cfg is not None:
            best_to_log = _ensure_means_from_simulation(best[0], target, fit_cfg, rng)
        _log_candidate(log_file, last_step + step_offset, best_to_log, target, best_to_log.breakdown)
        try:
            _generate_loss_plots(log_file)
        except Exception:
            pass

    # Fill in firing rates for any top candidates that were found on non-simulation steps
    if bistable_cfg is not None:
        best = [_ensure_means_from_simulation(c, target, fit_cfg, rng) for c in best]

    if interrupted and best:
        print(f"Best-so-far loss after interruption: {best[0].loss:.4g}")
=======
                _log_candidate(log_file, step + step_offset, best[0], target)

            if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
                if log_file:
                    _log_candidate(log_file, step + step_offset, best[0], target)
                stopped_early = True
                break

        pbar.close()

        if log_file and best and (not stopped_early) and last_step % log_interval != 0:
            _log_candidate(log_file, last_step + step_offset, best[0], target)
>>>>>>> origin/main

    return best


def _generate_loss_plots(log_file: str) -> None:
    """Generate loss evolution plots from the current log file.
    
    Called periodically during optimization to update live visualizations.
    Saves plots to the same directory as the log file.
    """
    try:
        from .loss_evolution_plot import plot_loss_evolution, plot_loss_evolution_ratios
        
        log_path = Path(log_file)
        output_dir = str(log_path.parent)
        
        # Generate both plots
        plot_loss_evolution(log_file, output_dir=output_dir, dpi=72)
        plot_loss_evolution_ratios(log_file, output_dir=output_dir, dpi=72)
    except ImportError:
        # Matplotlib or plotting module not available
        pass
    except Exception:
        # Silently fail - don't interrupt optimization
        pass


def _ensure_means_from_simulation(
    cand: Candidate,
    target: TargetRates,
    fit_cfg: FitConfig,
    rng: np.random.Generator,
) -> Candidate:
    """If candidate hasn't been simulated yet, run simulation to get firing rates.

    Returns:
        New candidate with firing rates, or original if already simulated.
    """
    if cand.simulated:
        # Already has firing rates from simulation
        return cand

    # Run simulation to get firing rates
    try:
        cond_results = [run_condition(c) for c in _build_conditions(cand.params, target, fit_cfg, rng)]
        _, means_sim, ko_means, _ = _loss_from_results(
            cond_results, target, fit_cfg, cand.params,
            squared_loss=True,
        )
        return Candidate(loss=cand.loss, means=means_sim, ko_means=ko_means, params=cand.params, breakdown=cand.breakdown, simulated=True)
    except Exception:
        # If simulation fails, return original with NaN
        return cand


def _log_candidate(
    path: str,
    step: int,
    cand: Candidate,
    target: TargetRates,
    breakdown: Optional[LossBreakdown] = None,
) -> None:
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
    breakdown_dict = None
    if breakdown is not None:
        # Handle both LossBreakdown objects (standard mode) and dicts (bistable mode)
        if isinstance(breakdown, LossBreakdown):
            # Only include components that have non-zero values (non-zero weight)
            breakdown_dict = {
                "total": float(breakdown.total),
            }
            # Add loss components only if they're non-zero (indicating active weight)
            if breakdown.firing_rate > 0:
                breakdown_dict["firing_rate"] = float(breakdown.firing_rate)
            if breakdown.ko_firing_rate > 0:
                breakdown_dict["ko_firing_rate"] = float(breakdown.ko_firing_rate)
            if breakdown.jacobian > 0:
                breakdown_dict["jacobian"] = float(breakdown.jacobian)
            if breakdown.turing > 0:
                breakdown_dict["turing"] = float(breakdown.turing)
            if breakdown.ach_ratio > 0:
                breakdown_dict["ach_ratio"] = float(breakdown.ach_ratio)
        elif isinstance(breakdown, dict):
            # Bistable mode: use the dict directly (already has L_bistab, L_rate, etc.)
            breakdown_dict = breakdown
    log_best_result(path, step, cand.loss, means_dict, ko_means_dict, cand.params, target, breakdown_dict)
