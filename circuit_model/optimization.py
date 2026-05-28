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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Any, Optional
from pathlib import Path

import nevergrad as ng
import numpy as np
from tqdm import tqdm

from .params import CircuitParams, ParamBound
from .simulation import simulate_circuit, mean_rates, NoiseType
from .loss import TargetRates, FitConfig, loss_from_means_normalized, loss_from_ko_normalized, drug_loss, DrugTarget
from .io import log_best_result, save_params_json


@dataclass
class KOMeans:
    """Container for knockout/drug condition mean firing rates (shape (5,) each).

    Global KOs zero a receptor everywhere it acts.
    Selective α7 KOs zero α7 only on the named cell type.
    Drug entries are keyed by drug name in `drug` dict.
    """
    alpha7_ko: Optional[np.ndarray] = None
    alpha5_ko: Optional[np.ndarray] = None
    beta2_ko: Optional[np.ndarray] = None
    alpha7_ndnf_ko: Optional[np.ndarray] = None
    alpha7_pv_ko: Optional[np.ndarray] = None
    drug: dict[str, np.ndarray] = None  # drug name -> mean rates (5,)

    def __post_init__(self):
        if self.drug is None:
            self.drug = {}


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
    """Breakdown of loss components.

    All four buckets are sums of normalised squared relative errors,
    multiplied by their per-bucket CLI weight.
    """
    base: float
    global_ko: float
    selective_ko: float
    drug: float
    total: float

    def __str__(self) -> str:
        return (f"loss=[base={self.base:.3g}, gko={self.global_ko:.3g}, "
                f"sko={self.selective_ko:.3g}, drug={self.drug:.3g}, "
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
        r0 = cfg.init_rate_scale * rng.lognormal(mean=0.0, sigma=0.6, size=5)
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
    *,
    drug_param_overrides: Optional[dict[str, dict[str, float]]] = None,
) -> list[tuple[str, CircuitParams, FitConfig, int]]:
    """Build list of (name, params, cfg, seed) conditions to simulate.

    `drug_param_overrides` maps a drug name (e.g. "MLA") to a dict of
    CircuitParams field overrides applied via `dataclasses.replace`. Each
    drug listed in `target.drug_targets` will be simulated.
    """
    alpha7_all_off = dict(act_alpha7_pv=0.0, act_alpha7_som=0.0, act_alpha7_ndnf=0.0)
    conditions: list[tuple[str, CircuitParams, FitConfig, int]] = [
        ("base", params, cfg, int(rng.integers(0, 2**31 - 1))),
        # Global KOs — measured on PYR
        ("alpha7_ko", replace(params, **alpha7_all_off),     cfg, int(rng.integers(0, 2**31 - 1))),
        ("alpha5_ko", replace(params, act_alpha5=0.0),       cfg, int(rng.integers(0, 2**31 - 1))),
        ("beta2_ko",  replace(params, act_beta2=0.0),        cfg, int(rng.integers(0, 2**31 - 1))),
        # Cell-type-selective α7 KOs — measured on the deleted cell type itself
        ("alpha7_ndnf_ko", replace(params, act_alpha7_ndnf=0.0), cfg, int(rng.integers(0, 2**31 - 1))),
        ("alpha7_pv_ko",   replace(params, act_alpha7_pv=0.0),   cfg, int(rng.integers(0, 2**31 - 1))),
    ]
    # Drug conditions (Stage 2)
    if target.drug_targets and drug_param_overrides:
        seen: set[str] = set()
        for dt in target.drug_targets:
            if dt.drug in seen:
                continue
            seen.add(dt.drug)
            overrides = drug_param_overrides.get(dt.drug, {})
            conditions.append((
                f"drug_{dt.drug}",
                replace(params, **overrides),
                cfg,
                int(rng.integers(0, 2**31 - 1)),
            ))
    return conditions


_POP_INDEX = {"PYR": 0, "SOM": 1, "PV": 2, "VIP": 3, "NDNF": 4}


def _loss_from_results(
    results: list[ConditionResult],
    target: TargetRates,
    cfg: FitConfig,
    params: CircuitParams,
    *,
    weight_base: float = 1.0,
    weight_global_ko: float = 1.0,
    weight_selective_ko: float = 1.0,
    weight_drug: float = 1.0,
) -> tuple[float, np.ndarray, KOMeans, LossBreakdown]:
    """Compute total loss from a list of condition simulation results.

    All buckets are sums of normalised squared relative errors, scaled by
    their respective per-bucket weight.
    """
    ko_means = KOMeans()
    base_means = np.zeros(5, dtype=float)

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
        elif name == "alpha7_ndnf_ko":
            ko_means.alpha7_ndnf_ko = means
        elif name == "alpha7_pv_ko":
            ko_means.alpha7_pv_ko = means
        elif name.startswith("drug_"):
            ko_means.drug[name[len("drug_"):]] = means

    # --- base : 5 baseline firing-rate targets ---
    base_loss = loss_from_means_normalized(base_means, target)

    # --- global_ko : PYR rate under each global KO ---
    global_ko_loss = 0.0
    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
        global_ko_loss += loss_from_ko_normalized(float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr)
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        global_ko_loss += loss_from_ko_normalized(float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr)
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        global_ko_loss += loss_from_ko_normalized(float(ko_means.beta2_ko[0]), target.beta2_ko_pyr)

    # --- selective_ko : NDNF / PV rate under their selective α7 KOs ---
    selective_ko_loss = 0.0
    if target.alpha7_ndnf_ko_ndnf is not None and ko_means.alpha7_ndnf_ko is not None:
        selective_ko_loss += loss_from_ko_normalized(
            float(ko_means.alpha7_ndnf_ko[4]), target.alpha7_ndnf_ko_ndnf
        )
    if target.alpha7_pv_ko_pv is not None and ko_means.alpha7_pv_ko is not None:
        selective_ko_loss += loss_from_ko_normalized(
            float(ko_means.alpha7_pv_ko[2]), target.alpha7_pv_ko_pv
        )

    # --- drug : per-drug per-population measurements ---
    drug_loss = 0.0
    for dt in target.drug_targets:
        rates = ko_means.drug.get(dt.drug)
        if rates is None:
            continue
        idx = _POP_INDEX[dt.population]
        drug_loss += loss_from_ko_normalized(float(rates[idx]), dt.target_hz)

    base_w   = weight_base        * base_loss
    gko_w    = weight_global_ko   * global_ko_loss
    sko_w    = weight_selective_ko * selective_ko_loss
    drug_w   = weight_drug        * drug_loss
    total    = base_w + gko_w + sko_w + drug_w

    breakdown = LossBreakdown(base=base_w, global_ko=gko_w, selective_ko=sko_w,
                              drug=drug_w, total=total)
    return total, base_means, ko_means, breakdown


def evaluate_params(
    params: CircuitParams,
    target: TargetRates,
    cfg: FitConfig,
    *,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor] = None,
    weight_base: float = 1.0,
    weight_global_ko: float = 1.0,
    weight_selective_ko: float = 1.0,
    weight_drug: float = 1.0,
    drug_param_overrides: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[float, np.ndarray, KOMeans, LossBreakdown]:
    """Evaluate a parameter set under baseline + KO + drug conditions."""
    conditions = _build_conditions(params, target, cfg, rng,
                                    drug_param_overrides=drug_param_overrides)
    if executor is not None and len(conditions) > 1:
        results = list(executor.map(run_condition, conditions))
    else:
        results = [run_condition(c) for c in conditions]
    return _loss_from_results(
        results, target, cfg, params,
        weight_base=weight_base,
        weight_global_ko=weight_global_ko,
        weight_selective_ko=weight_selective_ko,
        weight_drug=weight_drug,
    )


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
    if name in ("de", "twopointde", "TwoPointsDE"):
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
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose: de (alias: twopointde), cma, chaining, auto."
        )


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
    weight_base: float = 1.0,
    weight_global_ko: float = 1.0,
    weight_selective_ko: float = 1.0,
    weight_drug: float = 1.0,
    drug_param_overrides: Optional[dict[str, dict[str, float]]] = None,
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

    n_drug_conditions = len({dt.drug for dt in target.drug_targets})
    n_conditions = 1 + sum([
        target.alpha7_ko_pyr is not None,
        target.alpha5_ko_pyr is not None,
        target.beta2_ko_pyr is not None,
        target.alpha7_ndnf_ko_ndnf is not None,
        target.alpha7_pv_ko_pv is not None,
    ]) + n_drug_conditions

    # Parallel batching is not currently wired up to the main loop below.
    # Run sequentially.
    use_parallel = False
    max_workers = 1
    batch_size = 1

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

    if use_parallel:
        print(f"Using {max_workers} workers ({batch_size} candidates × {n_conditions} conditions)")

    best: list[Candidate] = []
    last_step = 0

    interrupted = False
    pbar = tqdm(range(1, n_samples + 1), desc="Optimizing", unit="step")
    try:
        for step in pbar:
            last_step = step

            x = ng_optimizer.ask()
            p = params_from_ng_dict(x.value, base)

            cond_results = [run_condition(c) for c in _build_conditions(
                p, target, fit_cfg, rng, drug_param_overrides=drug_param_overrides,
            )]
            L, means, ko_means, breakdown = _loss_from_results(
                cond_results, target, fit_cfg, p,
                weight_base=weight_base,
                weight_global_ko=weight_global_ko,
                weight_selective_ko=weight_selective_ko,
                weight_drug=weight_drug,
            )

            ng_optimizer.tell(x, L)

            prev_best_loss = best[0].loss if best else float("inf")
            cand = Candidate(loss=L, means=means, ko_means=ko_means, params=p,
                             breakdown=breakdown, simulated=True)
            if len(best) < top_k:
                best.append(cand)
                best.sort(key=lambda c: c.loss)
            elif L < best[-1].loss:
                best[-1] = cand
                best.sort(key=lambda c: c.loss)

            pbar.set_postfix({"loss": f"{best[0].loss:.4g}" if best else "N/A"})

            if save_best_json and best and best[0].loss < prev_best_loss:
                save_params_json(save_best_json, best[0].params)

            if log_file and step % log_interval == 0 and best:
                _log_candidate(log_file, step + step_offset, best[0], target, best[0].breakdown)
                try:
                    _generate_loss_plots(log_file)
                except Exception:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\nOptimization interrupted by user (Ctrl+C). Finalizing best-so-far results...")
    finally:
        pbar.close()

    if log_file and best and last_step % log_interval != 0:
        _log_candidate(log_file, last_step + step_offset, best[0], target, best[0].breakdown)
        try:
            _generate_loss_plots(log_file)
        except Exception:
            pass

    if interrupted and best:
        print(f"Best-so-far loss after interruption: {best[0].loss:.4g}")

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
        "pyr":  float(cand.means[0]),
        "som":  float(cand.means[1]),
        "pv":   float(cand.means[2]),
        "vip":  float(cand.means[3]),
        "ndnf": float(cand.means[4]),
    }
    ko_means_dict = {
        "alpha7_ko":      cand.ko_means.alpha7_ko.tolist()      if cand.ko_means.alpha7_ko      is not None else None,
        "alpha5_ko":      cand.ko_means.alpha5_ko.tolist()      if cand.ko_means.alpha5_ko      is not None else None,
        "beta2_ko":       cand.ko_means.beta2_ko.tolist()       if cand.ko_means.beta2_ko       is not None else None,
        "alpha7_ndnf_ko": cand.ko_means.alpha7_ndnf_ko.tolist() if cand.ko_means.alpha7_ndnf_ko is not None else None,
        "alpha7_pv_ko":   cand.ko_means.alpha7_pv_ko.tolist()   if cand.ko_means.alpha7_pv_ko   is not None else None,
    }
    if cand.ko_means.drug:
        ko_means_dict["drugs"] = {drug: rates.tolist() for drug, rates in cand.ko_means.drug.items()}
    breakdown_dict = None
    if breakdown is not None:
        breakdown_dict = {"total": float(breakdown.total)}
        if breakdown.base > 0:
            breakdown_dict["base"] = float(breakdown.base)
        if breakdown.global_ko > 0:
            breakdown_dict["global_ko"] = float(breakdown.global_ko)
        if breakdown.selective_ko > 0:
            breakdown_dict["selective_ko"] = float(breakdown.selective_ko)
        if breakdown.drug > 0:
            breakdown_dict["drug"] = float(breakdown.drug)
    log_best_result(path, step, cand.loss, means_dict, ko_means_dict, cand.params, target, breakdown_dict)


# ============================================================================
# STAGE 2: per-drug receptor-activation fitting
# ============================================================================

# Free CircuitParams fields in Stage 2 (g_alpha7 is frozen per Phase C decision).
STAGE2_FREE_FIELDS = (
    "act_alpha7_pv",
    "act_alpha7_som",
    "act_alpha7_ndnf",
    "act_beta2",
    "act_alpha5",
)


def _stage2_bounds() -> dict[str, ParamBound]:
    """Biology-informed bounds on receptor activations for Stage 2 fits."""
    # Activations are dimensionless multipliers on receptor-driven currents.
    # 0 = full block, 1 = baseline, >1 = potentiation.
    return {
        "act_alpha7_pv":   ParamBound(0.0, 5.0, mode="lin"),
        "act_alpha7_som":  ParamBound(0.0, 5.0, mode="lin"),
        "act_alpha7_ndnf": ParamBound(0.0, 5.0, mode="lin"),
        "act_beta2":       ParamBound(0.0, 5.0, mode="lin"),
        "act_alpha5":      ParamBound(0.0, 5.0, mode="lin"),
    }


def _stage2_eval(params: CircuitParams, cfg: FitConfig, rng: np.random.Generator) -> tuple[bool, np.ndarray]:
    """Average per-trial mean rates for a single CircuitParams set (no KOs)."""
    return run_trials(params, cfg, int(rng.integers(0, 2**31 - 1)))


def optimize_drug_activations(
    base: CircuitParams,
    drug_targets: list[DrugTarget],
    fit_cfg: FitConfig,
    *,
    n_samples: int,
    optimizer: str = "twopointde",
    seed: Optional[int] = 0,
    log_file: Optional[str] = None,
    log_interval: int = 50,
) -> dict[str, dict]:
    """Per-drug fit of receptor activations.

    For each distinct drug in `drug_targets`, runs an independent nevergrad
    optimization that varies only `act_alpha7_pv/_som/_ndnf`, `act_beta2`,
    `act_alpha5` (5 free params per drug, bounded [0, 5]). All other circuit
    parameters are frozen.

    Returns
    -------
    dict mapping drug_name -> {
        "activations": {act_*: value},
        "loss": float,
        "predicted_means": [pyr, som, pv, vip, ndnf],
        "targets": [{"population": ..., "target_hz": ..., "predicted_hz": ...}],
    }
    """
    rng_master = np.random.default_rng(seed)
    bounds = _stage2_bounds()
    drugs_in_order: list[str] = []
    for dt in drug_targets:
        if dt.drug not in drugs_in_order:
            drugs_in_order.append(dt.drug)

    out: dict[str, dict] = {}
    for drug in drugs_in_order:
        this_drug_targets = [dt for dt in drug_targets if dt.drug == drug]

        # Build per-drug nevergrad parametrization: only act_* fields are free.
        # All other CircuitParams stay at `base`.
        params_dict: dict[str, Any] = {}
        for f in fields(CircuitParams):
            if f.name in STAGE2_FREE_FIELDS:
                bound = bounds[f.name]
                init = float(np.clip(float(getattr(base, f.name)), bound.lo, bound.hi))
                params_dict[f.name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi, init=init)
            else:
                params_dict[f.name] = getattr(base, f.name)
        parametrization = ng.p.Dict(**params_dict)

        ng_opt = _build_optimizer(optimizer, parametrization, n_samples, num_workers=1)
        ng_opt.parametrization.random_state = np.random.RandomState(
            int(rng_master.integers(0, 2**31 - 1))
        )

        rng_eval = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        best_loss = float("inf")
        best_means: Optional[np.ndarray] = None
        best_acts: dict[str, float] = {}

        print(f"\n--- Stage 2 fit: drug = {drug} ({len(this_drug_targets)} measurement(s)) ---")
        pbar = tqdm(range(1, n_samples + 1), desc=f"  {drug}", unit="step")
        for step in pbar:
            x = ng_opt.ask()
            cand = params_from_ng_dict(x.value, base)
            ok, means = _stage2_eval(cand, fit_cfg, rng_eval)
            if not ok:
                L = 1e9
            else:
                L = drug_loss(means, this_drug_targets, drug)
            ng_opt.tell(x, L)
            if L < best_loss:
                best_loss = L
                best_means = means.copy() if ok else None
                best_acts = {k: float(v) for k, v in x.value.items() if k in STAGE2_FREE_FIELDS}
            pbar.set_postfix({"loss": f"{best_loss:.4g}"})

            if log_file and step % log_interval == 0:
                with open(log_file, "a", encoding="utf-8") as fh:
                    import json as _json
                    fh.write(_json.dumps({
                        "drug": drug, "step": step, "loss": best_loss,
                        "activations": best_acts,
                    }) + "\n")
        pbar.close()

        predicted = best_means.tolist() if best_means is not None else [float("nan")] * 5
        out[drug] = {
            "activations": best_acts,
            "loss": float(best_loss),
            "predicted_means": predicted,
            "targets": [
                {
                    "population": dt.population,
                    "target_hz": dt.target_hz,
                    "predicted_hz": predicted[{"PYR": 0, "SOM": 1, "PV": 2, "VIP": 3, "NDNF": 4}[dt.population]],
                }
                for dt in this_drug_targets
            ],
        }
    return out
