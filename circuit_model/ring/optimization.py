"""
Joint ring + circuit parameter optimization.

This module provides:
- RingFitConfig: Configuration wrapping FitConfig with ring-specific settings
- BumpTarget: Soft targets for bump quality (Mode 2 only)
- RingCandidate: Optimization result holding both CircuitParams and RingParams
- run_ring_trials: Run multiple ring simulations at rest, return mean node-averaged rates
- run_bump_trial: Run one ring simulation with a cue stimulus, return bump loss
- evaluate_ring_params: Full evaluation (rates + optional bump quality)
- build_ring_parametrization: Nevergrad parametrization for joint circuit+ring search space
- ring_params_from_ng_dict: Extract RingParams from Nevergrad dict
- nevergrad_optimize_ring: Main joint optimization function

Modes:
- Mode 1 (bump_target=None): Optimize so ring at rest matches TargetRates
- Mode 2 (bump_target set): Same + soft constraint that a bump forms after stimulus
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace, asdict
from pathlib import Path
from typing import Any, Optional

import nevergrad as ng
import numpy as np
from tqdm import tqdm

from ..params import CircuitParams, ParamBound, default_bounds
from ..loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr, jacobian_connectivity_penalty, transfer_function_slope
from ..optimization import (
    KOMeans,
    run_condition,
    _build_conditions,
    _build_optimizer,
    build_nevergrad_parametrization,
    params_from_ng_dict,
)
from ..io import save_params_json

from .params import RingParams, default_ring_bounds
from .simulation import simulate_ring, mean_rates_ring, NoiseType
from .connectivity import RingConnectivity
from .stimulus import RingStimulus
from .analysis import decode_bump_center


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingFitConfig:
    """
    Configuration for ring-based optimization.

    Wraps a FitConfig (used for single-node KO evaluations) and adds
    ring-specific settings.
    """
    fit_cfg: FitConfig = None  # type: ignore[assignment]  # set post-init via __post_init__
    n_trials_ring: int = 3     # Number of ring trials per candidate (fewer than single-node due to cost)
    ko_on_ring: bool = False   # If True, run KO conditions on ring (slower but fully consistent)

    def __post_init__(self):
        if self.fit_cfg is None:
            object.__setattr__(self, 'fit_cfg', FitConfig())

    @classmethod
    def from_fit_cfg(cls, fit_cfg: FitConfig, n_trials_ring: int = 3, ko_on_ring: bool = False) -> "RingFitConfig":
        """Build a RingFitConfig from an existing FitConfig."""
        return cls(fit_cfg=fit_cfg, n_trials_ring=n_trials_ring, ko_on_ring=ko_on_ring)


@dataclass(frozen=True)
class BumpTarget:
    """
    Soft targets for bump quality (Mode 2 only).

    These are independent of the firing rate targets: the firing rate targets
    come from quiet wakefulness data, while bump targets are biophysical
    constraints on working memory network function.
    """
    min_amplitude: float = 0.3       # Minimum acceptable bump amplitude [0,1]
    bump_loss_weight: float = 2.0    # Weight of bump loss relative to rate loss
    stim_amplitude: float = 5.0      # Stimulus peak current (applied to PYR)
    stim_sigma_deg: float = 20.0     # Gaussian width of stimulus (degrees)
    stim_duration_ms: float = 250.0  # Stimulus duration (ms)
    eval_window_ms: float = 500.0    # Post-stimulus window to evaluate bump amplitude


@dataclass(frozen=True)
class RingCandidate:
    """Optimization result holding both CircuitParams and RingParams."""
    loss: float
    ring_means: np.ndarray   # Mean firing rates averaged across ring nodes, shape (4,)
    ko_means: KOMeans        # KO condition results (single-node or ring)
    params: CircuitParams
    ring_params: RingParams


# ---------------------------------------------------------------------------
# Ring simulation helpers
# ---------------------------------------------------------------------------

def run_ring_trials(
    params: CircuitParams,
    ring_params: RingParams,
    cfg: RingFitConfig,
    rng: np.random.Generator,
    connectivity: Optional[RingConnectivity] = None,
) -> tuple[bool, np.ndarray]:
    """
    Run n_trials_ring ring simulations at rest (no stimulus).

    Returns:
        (success, means) where means has shape (4,) — firing rates averaged
        over nodes and time for [pyr, som, pv, vip]. Returns (False, zeros)
        if any trial produces NaN or rates above max_rate.
    """
    fit_cfg = cfg.fit_cfg
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params)

    means_trials: list[np.ndarray] = []

    for _ in range(cfg.n_trials_ring):
        seed = int(rng.integers(0, 2**31 - 1))

        result = simulate_ring(
            params,
            ring_params,
            T_ms=fit_cfg.T_ms,
            dt_ms=fit_cfg.dt_ms,
            stimuli=None,
            seed=seed,
            noise_type=fit_cfg.noise_type,
            tau_noise_ms=fit_cfg.tau_noise_ms,
            connectivity=connectivity,
        )

        # shape (n_nodes, 4) → average over nodes → (4,)
        rates_per_node = mean_rates_ring(result, burn_in_ms=fit_cfg.burn_in_ms, window_ms=fit_cfg.window_ms)
        means = rates_per_node.mean(axis=0)  # (4,)

        if not np.all(np.isfinite(means)) or np.any(means > fit_cfg.max_rate):
            return False, np.zeros(4)

        means_trials.append(means)

    means_avg = np.mean(np.stack(means_trials, axis=0), axis=0)
    return True, means_avg


def run_bump_trial(
    params: CircuitParams,
    ring_params: RingParams,
    cfg: RingFitConfig,
    bump_target: BumpTarget,
    rng: np.random.Generator,
    connectivity: Optional[RingConnectivity] = None,
) -> float:
    """
    Run one ring simulation with a cue stimulus at 0°.

    Evaluates bump amplitude during eval_window_ms after stimulus offset.
    Returns bump loss: max(0, min_amplitude - mean_amplitude)^2.
    No penalty when amplitude >= min_amplitude (one-sided soft constraint).
    """
    fit_cfg = cfg.fit_cfg
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params)

    onset_ms = fit_cfg.burn_in_ms
    stim = RingStimulus(
        center_deg=0.0,
        amplitude=bump_target.stim_amplitude,
        sigma_deg=bump_target.stim_sigma_deg,
        onset_ms=onset_ms,
        duration_ms=bump_target.stim_duration_ms,
    )
    T_ms = onset_ms + bump_target.stim_duration_ms + bump_target.eval_window_ms

    seed = int(rng.integers(0, 2**31 - 1))
    result = simulate_ring(
        params,
        ring_params,
        T_ms=T_ms,
        dt_ms=fit_cfg.dt_ms,
        stimuli=[stim],
        seed=seed,
        noise_type=fit_cfg.noise_type,
        tau_noise_ms=fit_cfg.tau_noise_ms,
        connectivity=connectivity,
    )

    if not np.all(np.isfinite(result.r)):
        return bump_target.min_amplitude ** 2  # maximum penalty

    # Decode bump amplitude during the post-stimulus eval window
    _, amplitude = decode_bump_center(result, population=0)  # shape (n_steps,)

    # Find timesteps in the eval window [onset + duration, T_ms]
    eval_start_ms = onset_ms + bump_target.stim_duration_ms
    t_ms = result.t_ms
    eval_mask = t_ms >= eval_start_ms

    if not np.any(eval_mask):
        return bump_target.min_amplitude ** 2

    mean_amplitude = float(amplitude[eval_mask].mean())

    # One-sided penalty: 0 if amplitude >= min_amplitude, quadratic below
    shortfall = max(0.0, bump_target.min_amplitude - mean_amplitude)
    return shortfall ** 2


# ---------------------------------------------------------------------------
# Turing instability loss
# ---------------------------------------------------------------------------

def turing_instability_loss(
    params: CircuitParams,
    ring_params: RingParams,
    r_ss: np.ndarray,
    margin: float = 0.1,
) -> float:
    """Soft penalty enforcing the Turing instability condition for bump formation.

    The condition Φ'(I*_PYR) · w_pyr_pyr_inter > 1 is necessary for the ring to
    support a spatially localised bump state (ignoring PV feedback).

    Penalty:  max(0,  1 + margin − turing)²
    which is 0 once  turing ≥ 1 + margin  (one-sided soft constraint).

    Parameters
    ----------
    params    : CircuitParams — local circuit parameters
    ring_params : RingParams — ring connectivity parameters
    r_ss      : steady-state firing rates [pyr, som, pv, vip] from ring rest trials
    margin    : safety margin above the instability threshold (default 0.1)
    """
    slope = transfer_function_slope(params, r_ss, population="PYR")
    turing = slope * ring_params.w_pyr_pyr_inter
    return max(0.0, 1.0 + margin - turing) ** 2


# ---------------------------------------------------------------------------
# Joint evaluation
# ---------------------------------------------------------------------------

def evaluate_ring_params(
    params: CircuitParams,
    ring_params: RingParams,
    target: TargetRates,
    cfg: RingFitConfig,
    bump_target: Optional[BumpTarget],
    rng: np.random.Generator,
    turing_weight: float = 0.0,
    turing_margin: float = 0.1,
) -> tuple[float, np.ndarray, KOMeans]:
    """
    Evaluate a (CircuitParams, RingParams) pair.

    Steps:
    1. Run ring at rest (n_trials_ring) → ring rate loss
    2. Run KO conditions on single-node (default) or ring (cfg.ko_on_ring)
    3. Jacobian connectivity penalty
    4. (optional) Turing instability penalty (turing_weight > 0)
    5. (Mode 2) Run bump trial → bump loss

    Returns:
        (total_loss, ring_means, ko_means)
    """
    # Pre-compute connectivity once for all ring simulations in this evaluation
    connectivity = RingConnectivity.from_params(ring_params)

    # --- Step 1: Ring baseline ---
    ok, ring_means = run_ring_trials(params, ring_params, cfg, rng, connectivity=connectivity)
    if not ok:
        return 1e9, ring_means, KOMeans()

    ring_rate_loss = loss_from_means(ring_means, target)

    # --- Step 2: KO conditions ---
    ko_means = KOMeans()
    ko_loss = 0.0
    n_ko = 0

    if cfg.ko_on_ring:
        # Run each KO condition on the ring
        ko_conditions = [
            ("alpha7_ko", replace(params, act_alpha7=0.0, g_alpha7=0.0)),
            ("alpha5_ko", replace(params, act_alpha5=0.0)),
            ("beta2_ko",  replace(params, act_beta2=0.0)),
        ]
        for ko_name, ko_params in ko_conditions:
            ko_ok, ko_m = run_ring_trials(ko_params, ring_params, cfg, rng, connectivity=connectivity)
            if not ko_ok:
                return 1e9, ring_means, ko_means
            if ko_name == "alpha7_ko":
                ko_means.alpha7_ko = ko_m
            elif ko_name == "alpha5_ko":
                ko_means.alpha5_ko = ko_m
            elif ko_name == "beta2_ko":
                ko_means.beta2_ko = ko_m
    else:
        # Run KO conditions on single-node (cheaper, same CircuitParams)
        fit_cfg = cfg.fit_cfg
        ko_conditions_sn = _build_conditions(params, target, fit_cfg, rng)
        for name, ok_sn, means_sn in [run_condition(c) for c in ko_conditions_sn]:
            if not ok_sn:
                return 1e9, ring_means, ko_means
            if name == "alpha7_ko":
                ko_means.alpha7_ko = means_sn
            elif name == "alpha5_ko":
                ko_means.alpha5_ko = means_sn
            elif name == "beta2_ko":
                ko_means.beta2_ko = means_sn

    # Compute KO losses
    base_pyr = float(ring_means[0])
    fit_cfg = cfg.fit_cfg

    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
        ko_loss += loss_from_ko_pyr(
            float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr, base_pyr,
            min_effect_weight=fit_cfg.ko_min_effect_penalty,
            wrong_direction_weight=fit_cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        ko_loss += loss_from_ko_pyr(
            float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr, base_pyr,
            min_effect_weight=fit_cfg.ko_min_effect_penalty,
            wrong_direction_weight=fit_cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1
    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        ko_loss += loss_from_ko_pyr(
            float(ko_means.beta2_ko[0]), target.beta2_ko_pyr, base_pyr,
            min_effect_weight=fit_cfg.ko_min_effect_penalty,
            wrong_direction_weight=fit_cfg.ko_wrong_direction_penalty,
        )
        n_ko += 1

    total = ring_rate_loss
    if n_ko > 0:
        total += ko_loss / n_ko

    # --- Step 3: Jacobian penalty (evaluated at ring-averaged rates) ---
    total += jacobian_connectivity_penalty(params, ring_means)

    # --- Step 4: Turing instability penalty (analytical, from ring rest rates) ---
    if turing_weight > 0.0:
        t_loss = turing_instability_loss(params, ring_params, ring_means, turing_margin)
        total += turing_weight * t_loss

    # --- Step 5: Bump quality (Mode 2) ---
    if bump_target is not None:
        b_loss = run_bump_trial(params, ring_params, cfg, bump_target, rng, connectivity=connectivity)
        total += bump_target.bump_loss_weight * b_loss

    return total, ring_means, ko_means


# ---------------------------------------------------------------------------
# Nevergrad parametrization for joint search space
# ---------------------------------------------------------------------------

def build_ring_parametrization(
    base_circuit: CircuitParams,
    circuit_bounds: dict[str, ParamBound],
    base_ring: RingParams,
    ring_bounds: dict[str, ParamBound],
    freeze: Optional[set[str]] = None,
) -> ng.p.Dict:
    """
    Build a Nevergrad parametrization over CircuitParams + RingParams.

    Circuit parameters are handled exactly as in build_nevergrad_parametrization.
    Ring parameters (w_pyr_pyr_inter, w_pv_global, sigma_pyr_deg) are appended
    with a 'ring__' prefix to avoid name collisions.
    """
    freeze = freeze or set()

    # Build circuit part using the existing helper
    circuit_ng = build_nevergrad_parametrization(base_circuit, circuit_bounds, freeze)
    params_dict: dict[str, Any] = dict(circuit_ng.value)
    # Rebuild as Scalar/Log parameters (not scalar values) from the existing parametrization
    params_dict = {}
    for f in fields(CircuitParams):
        name = f.name
        if name in freeze or name not in circuit_bounds:
            params_dict[name] = getattr(base_circuit, name)
        else:
            bound = circuit_bounds[name]
            raw = float(getattr(base_circuit, name))
            init = float(np.clip(raw, bound.lo, bound.hi))
            if bound.mode == "log" and bound.lo > 0:
                params_dict[name] = ng.p.Log(lower=bound.lo, upper=bound.hi, init=init)
            else:
                params_dict[name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi, init=init)

    # Append ring parameters with 'ring__' prefix
    ring_field_names = {f.name for f in fields(RingParams) if not f.name.startswith('_')}
    for name, bound in ring_bounds.items():
        if name not in ring_field_names:
            continue
        ng_name = f"ring__{name}"
        raw = float(getattr(base_ring, name))
        if name in freeze:
            params_dict[ng_name] = raw
            continue
        init = float(np.clip(raw, bound.lo, bound.hi))
        if bound.mode == "log" and bound.lo > 0:
            params_dict[ng_name] = ng.p.Log(lower=bound.lo, upper=bound.hi, init=init)
        else:
            params_dict[ng_name] = ng.p.Scalar(lower=bound.lo, upper=bound.hi, init=init)

    return ng.p.Dict(**params_dict)


def ring_params_from_ng_dict(ng_dict: dict[str, Any], base_ring: RingParams) -> RingParams:
    """Extract RingParams from a Nevergrad value dict (keys prefixed with 'ring__')."""
    ring_field_names = {f.name for f in fields(RingParams) if not f.name.startswith('_')}
    updates: dict[str, Any] = {}
    for ng_key, val in ng_dict.items():
        if ng_key.startswith("ring__"):
            name = ng_key[len("ring__"):]
            if name in ring_field_names:
                updates[name] = val
    return replace(base_ring, **updates)


def _save_ring_candidate(output_dir: str, candidate: RingCandidate) -> None:
    """Save both param files from a RingCandidate to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_params_json(str(Path(output_dir) / "best_circuit_params.json"), candidate.params)
    # Save ring params as a simple JSON (dataclass fields only)
    ring_dict = {
        f.name: getattr(candidate.ring_params, f.name)
        for f in fields(RingParams)
        if not f.name.startswith('_')
    }
    ring_path = Path(output_dir) / "best_ring_params.json"
    with open(ring_path, "w", encoding="utf-8") as fh:
        json.dump(ring_dict, fh, indent=2)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def nevergrad_optimize_ring(
    target: TargetRates,
    *,
    base_circuit: CircuitParams,
    circuit_bounds: dict[str, ParamBound],
    base_ring: RingParams,
    ring_bounds: dict[str, ParamBound],
    ring_cfg: RingFitConfig,
    bump_target: Optional[BumpTarget],
    n_samples: int,
    top_k: int,
    seed: Optional[int],
    optimizer: str = "de",
    freeze: Optional[set[str]] = None,
    early_stop_loss: Optional[float] = 1e-4,
    plateau_patience: int = 3000,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    save_output_dir: Optional[str] = None,
    turing_weight: float = 0.0,
    turing_margin: float = 0.1,
) -> list[RingCandidate]:
    """
    Joint optimization of CircuitParams + RingParams against ring-level targets.

    Mode 1 (bump_target=None):
        total_loss = ring_rate_loss + ko_loss/n_ko + jacobian_penalty
                   [+ turing_weight * turing_loss  if turing_weight > 0]

    Mode 2 (bump_target set):
        total_loss = ring_rate_loss + ko_loss/n_ko + jacobian_penalty
                   [+ turing_weight * turing_loss  if turing_weight > 0]
                   + bump_loss_weight * bump_loss

    Parameters:
        target: Target firing rates (from quiet wakefulness data)
        base_circuit: Starting point for CircuitParams
        circuit_bounds: Search bounds for CircuitParams
        base_ring: Starting point for RingParams
        ring_bounds: Search bounds for ring parameters (w_pyr_pyr_inter, w_pv_global, sigma_pyr_deg)
        ring_cfg: Ring-specific fit configuration
        bump_target: Bump quality constraint (None = Mode 1, set = Mode 2)
        n_samples: Number of optimization steps
        top_k: Keep top K candidates
        seed: Random seed
        optimizer: 'de', 'cma', 'chaining', or 'auto'
        freeze: Set of CircuitParams field names to freeze during optimization
        early_stop_loss: Stop if loss falls below this threshold
        plateau_patience: Stop if no improvement for this many steps
        log_file: Path to JSONL log file
        log_interval: Log every N steps
        save_output_dir: Directory to save best circuit + ring params during optimization
        turing_weight: Weight of the Turing instability penalty (0 = disabled)
        turing_margin: Safety margin above the Turing threshold (penalize if turing < 1 + margin)

    Returns:
        List of top-k RingCandidates sorted by loss (ascending)
    """
    rng = np.random.default_rng(seed)

    parametrization = build_ring_parametrization(
        base_circuit, circuit_bounds, base_ring, ring_bounds, freeze,
    )
    ng_optimizer = _build_optimizer(optimizer, parametrization, n_samples, num_workers=1)

    if seed is not None:
        ng_optimizer.parametrization.random_state = np.random.RandomState(seed)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        open(log_file, "w", encoding="utf-8").close()

    if save_output_dir:
        Path(save_output_dir).mkdir(parents=True, exist_ok=True)

    turing_str = f" + Turing penalty (w={turing_weight}, margin={turing_margin})" if turing_weight > 0.0 else ""
    mode_str = ("Mode 2 (rates + bump quality)" if bump_target is not None else "Mode 1 (rates only)") + turing_str
    print(f"Ring joint optimization — {mode_str}")
    print(f"Optimizer: {optimizer}")
    print(f"Ring trials per eval: {ring_cfg.n_trials_ring}")
    print(f"KO conditions on: {'ring' if ring_cfg.ko_on_ring else 'single-node'}")

    plateau_start_step = 0
    if optimizer == "chaining":
        de_budget = min(5000, n_samples - 1)
        plateau_start_step = de_budget + 1
        print(f"Chaining: DE for {de_budget} steps → Nelder-Mead for {n_samples - de_budget} steps")

    best: list[RingCandidate] = []
    steps_since_improvement = 0
    last_step = 0
    stopped_early = False

    pbar = tqdm(range(1, n_samples + 1), desc="Ring-Optimize", unit="step")
    for step in pbar:
        last_step = step

        x = ng_optimizer.ask()
        ng_dict = x.value

        p = params_from_ng_dict(ng_dict, base_circuit)
        rp = ring_params_from_ng_dict(ng_dict, base_ring)

        L, ring_means, ko_means = evaluate_ring_params(p, rp, target, ring_cfg, bump_target, rng, turing_weight, turing_margin)
        ng_optimizer.tell(x, L)

        prev_best_loss = best[0].loss if best else float("inf")
        cand = RingCandidate(loss=L, ring_means=ring_means, ko_means=ko_means, params=p, ring_params=rp)

        if len(best) < top_k:
            best.append(cand)
            best.sort(key=lambda c: c.loss)
        elif L < best[-1].loss:
            best[-1] = cand
            best.sort(key=lambda c: c.loss)

        if best[0].loss < prev_best_loss:
            steps_since_improvement = 0
            if save_output_dir and best:
                _save_ring_candidate(save_output_dir, best[0])
            if log_file and best:
                _log_ring_candidate(log_file, step, best[0], target)
        elif step >= plateau_start_step:
            steps_since_improvement += 1

        pbar.set_postfix(
            loss=f"{best[0].loss:.4g}" if best else "N/A",
            step=step,
            plateau=steps_since_improvement,
        )

        if log_file and step % log_interval == 0 and best:
            _log_ring_candidate(log_file, step, best[0], target)

        if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
            if log_file and best:
                _log_ring_candidate(log_file, step, best[0], target)
            stopped_early = True
            break

        if plateau_patience > 0 and steps_since_improvement >= plateau_patience:
            print(f"\nEarly stop: no improvement for {plateau_patience} steps.")
            if log_file and best:
                _log_ring_candidate(log_file, step, best[0], target)
            stopped_early = True
            break

    pbar.close()

    if log_file and best and (not stopped_early) and last_step % log_interval != 0:
        _log_ring_candidate(log_file, last_step, best[0], target)

    return best


def _log_ring_candidate(path: str, step: int, cand: RingCandidate, target: TargetRates) -> None:
    """Append a JSONL log entry for a ring candidate."""
    ring_means = cand.ring_means.tolist() if isinstance(cand.ring_means, np.ndarray) else list(cand.ring_means)
    entry = {
        "step": step,
        "loss": round(float(cand.loss), 6),
        "ring_means": {
            "pyr": round(ring_means[0], 4),
            "som": round(ring_means[1], 4),
            "pv":  round(ring_means[2], 4),
            "vip": round(ring_means[3], 4),
        },
        "ko_means": {
            "alpha7_ko": cand.ko_means.alpha7_ko.tolist() if cand.ko_means.alpha7_ko is not None else None,
            "alpha5_ko": cand.ko_means.alpha5_ko.tolist() if cand.ko_means.alpha5_ko is not None else None,
            "beta2_ko":  cand.ko_means.beta2_ko.tolist()  if cand.ko_means.beta2_ko  is not None else None,
        },
        "ring_params": {
            "w_pyr_pyr_inter": round(float(cand.ring_params.w_pyr_pyr_inter), 6),
            "w_pv_global":     round(float(cand.ring_params.w_pv_global), 6),
            "sigma_pyr_deg":   round(float(cand.ring_params.sigma_pyr_deg), 6),
            "n_nodes":         cand.ring_params.n_nodes,
        },
        "target": {
            "mean_r_pyr": target.mean_r_pyr,
            "mean_r_som": target.mean_r_som,
            "mean_r_pv":  target.mean_r_pv,
            "mean_r_vip": target.mean_r_vip,
            "alpha7_ko_pyr": target.alpha7_ko_pyr,
            "alpha5_ko_pyr": target.alpha5_ko_pyr,
            "beta2_ko_pyr":  target.beta2_ko_pyr,
        },
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
