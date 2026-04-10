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
from dataclasses import dataclass, fields, replace
from typing import Any, Optional
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


# Type alias for condition results
ConditionResult = tuple[str, bool, np.ndarray]  # (name, ok, means)


@dataclass
class LossBreakdown:
    """Breakdown of loss components."""
    firing_rate: float
    ko_firing_rate: float
    jacobian: float
    turing: float
    total: float

    def __str__(self) -> str:
        return (f"loss=[fr={self.firing_rate:.3g}, ko={self.ko_firing_rate:.3g}, "
                f"jac={self.jacobian:.3g}, turing={self.turing:.3g}, "
                f"total={self.total:.3g}]")


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
    return conditions


def _loss_from_results(
    results: list[ConditionResult],
    target: TargetRates,
    cfg: FitConfig,
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
    ko_means = KOMeans()
    base_means = np.zeros(4, dtype=float)

    for name, ok, means in results:
        if not ok:
            return 1e9, base_means, ko_means, LossBreakdown(1e9, 0., 0., 0., 1e9)
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
        ko_loss += loss_from_ko_pyr(
            float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        ko_loss += loss_from_ko_pyr(
            float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr, base_pyr,
            min_effect_weight=cfg.ko_min_effect_penalty,
            wrong_direction_weight=cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        ko_loss += loss_from_ko_pyr(
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
                               jacobian=jac_loss, turing=turing_loss, total=total)

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
    early_stop_loss: Optional[float] = 1e-4,
    plateau_patience: int = 5000,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    save_best_json: Optional[str] = None,
    step_offset: int = 0,
    append_log: bool = False,
    squared_loss: bool = True,
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_w_inter_ref: float = 10.0,
    turing_cue_scale: float = 0.4,
    ach_ratio_weight: float = 2.0,
    bistable_cfg: Optional[Any] = None,
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
        de_budget = 0
        print(f"Optimizer: {optimizer}")

    plateau_start_step = de_budget + 1  # plateau only active after DE phase

    best: list[Candidate] = []
    last_step = 0
    stopped_early = False
    steps_since_improvement = 0

    pbar = tqdm(range(1, n_samples + 1), desc="Optimizing", unit="step")
    for step in pbar:
        last_step = step

        x = ng_optimizer.ask()
        p = params_from_ng_dict(x.value, base)

        if bistable_cfg is not None:
            # Bistable mode: use bistable loss directly
            from .bistable_loss import bistable_loss as _bistable_loss
            L = float(_bistable_loss(p, bistable_cfg))
            means = np.zeros(4)
            ko_means = KOMeans()
            breakdown = LossBreakdown(firing_rate=0., ko_firing_rate=0., jacobian=0., turing=L, total=L)
        else:
            # Standard mode: simulate and compute loss
            cond_results = [run_condition(c) for c in _build_conditions(p, target, fit_cfg, rng)]
            L, means, ko_means, breakdown = _loss_from_results(cond_results, target, fit_cfg, p, squared_loss=squared_loss, jacobian_weight=jacobian_weight, turing_weight=turing_weight, turing_margin=turing_margin, turing_w_inter_ref=turing_w_inter_ref, turing_cue_scale=turing_cue_scale, ach_ratio_weight=ach_ratio_weight)

        ng_optimizer.tell(x, L)
        
        # Update progress bar with loss breakdown
        pbar.set_postfix_str(str(breakdown))

        prev_best_loss = best[0].loss if best else float("inf")
        cand = Candidate(loss=L, means=means, ko_means=ko_means, params=p, breakdown=breakdown)
        if len(best) < top_k:
            best.append(cand)
            best.sort(key=lambda c: c.loss)
        elif L < best[-1].loss:
            best[-1] = cand
            best.sort(key=lambda c: c.loss)

        if best[0].loss < prev_best_loss:
            steps_since_improvement = 0
        elif step >= plateau_start_step:
            steps_since_improvement += 1

        pbar.set_postfix(loss=f"{best[0].loss:.4g}" if best else "N/A", pyr=f"{best[0].means[0]:.2f}" if best else "N/A", step=step, plateau=steps_since_improvement)

        if save_best_json and best and best[0].loss < prev_best_loss:
            save_params_json(save_best_json, best[0].params)

        if log_file and step % log_interval == 0 and best:
            _log_candidate(log_file, step + step_offset, best[0], target, best[0].breakdown)
            # Generate loss evolution plots every log_interval steps
            try:
                _generate_loss_plots(log_file)
            except Exception as e:
                # Don't break optimization if plotting fails
                pass

        if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
            if log_file:
                _log_candidate(log_file, step + step_offset, best[0], target, best[0].breakdown)
                try:
                    _generate_loss_plots(log_file)
                except Exception:
                    pass
            stopped_early = True
            break

        if plateau_patience > 0 and steps_since_improvement >= plateau_patience:
            print(f"\nEarly stop: no improvement for {plateau_patience} steps.")
            if log_file and best:
                _log_candidate(log_file, step + step_offset, best[0], target, best[0].breakdown)
                try:
                    _generate_loss_plots(log_file)
                except Exception:
                    pass
            stopped_early = True
            break

    pbar.close()

    if log_file and best and (not stopped_early) and last_step % log_interval != 0:
        _log_candidate(log_file, last_step + step_offset, best[0], target, best[0].breakdown)
        try:
            _generate_loss_plots(log_file)
        except Exception:
            pass

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
        breakdown_dict = {
            "firing_rate": float(breakdown.firing_rate),
            "ko_firing_rate": float(breakdown.ko_firing_rate),
            "jacobian": float(breakdown.jacobian),
            "turing": float(breakdown.turing),
            "total": float(breakdown.total),
        }
    log_best_result(path, step, cand.loss, means_dict, ko_means_dict, cand.params, target, breakdown_dict)
