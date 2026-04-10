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
import sys

import nevergrad as ng
import numpy as np
from tqdm import tqdm

from ..params import CircuitParams, ParamBound, default_bounds
from ..loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr, jacobian_connectivity_penalty, ach_ratio_penalty, transfer_function_slope
from ..optimization import (
    KOMeans,
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

    Wraps a FitConfig and adds ring-specific settings.
    """
    fit_cfg: FitConfig = None  # type: ignore[assignment]  # set post-init via __post_init__
    n_trials_ring: int = 5     # Number of ring trials per candidate (5 gives good averaging of stochastic noise)

    def __post_init__(self):
        if self.fit_cfg is None:
            object.__setattr__(self, 'fit_cfg', FitConfig())

    @classmethod
    def from_fit_cfg(cls, fit_cfg: FitConfig, n_trials_ring: int = 3) -> "RingFitConfig":
        """Build a RingFitConfig from an existing FitConfig."""
        return cls(fit_cfg=fit_cfg, n_trials_ring=n_trials_ring)


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
    ko_means: KOMeans        # KO condition results on ring
    params: CircuitParams
    ring_params: RingParams
    breakdown: Optional["RingLossBreakdown"] = None


@dataclass(frozen=True)
class RingLossBreakdown:
    """Breakdown of ring loss components."""
    ring_rate: float
    ko_penalty: float
    jacobian: float
    ack_ratio: float
    turing: float
    spatial_uniformity: float
    bump: float
    total: float

    def __str__(self) -> str:
        return (f"loss=[ring_rate={self.ring_rate:.3g}, ko={self.ko_penalty:.3g}, "
                f"jac={self.jacobian:.3g}, tur={self.turing:.3g}, "
                f"sp_unif={self.spatial_uniformity:.3g}, bump={self.bump:.3g}, "
                f"total={self.total:.3g}]")


# ---------------------------------------------------------------------------
# Ring simulation helpers
# ---------------------------------------------------------------------------

def run_ring_trials(
    params: CircuitParams,
    ring_params: RingParams,
    cfg: RingFitConfig,
    rng: np.random.Generator,
    connectivity: Optional[RingConnectivity] = None,
) -> tuple[bool, np.ndarray, float]:
    """
    Run n_trials_ring ring simulations at rest (no stimulus).

    Returns:
        (success, means, spatial_cv) where means has shape (4,) — firing rates averaged
        over nodes and time for [pyr, som, pv, vip], and spatial_cv is the mean
        coefficient of variation of PYR rates across nodes (std/mean, averaged over trials).
        Returns (False, zeros, 0.0) if any trial produces NaN or rates above max_rate.
    """
    fit_cfg = cfg.fit_cfg
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params)

    means_trials: list[np.ndarray] = []
    cv_trials: list[float] = []

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
            record_dt_ms=fit_cfg.record_dt_ms,
        )

        # shape (n_nodes, 4) → average over nodes → (4,)
        rates_per_node = mean_rates_ring(result, burn_in_ms=fit_cfg.burn_in_ms, window_ms=fit_cfg.window_ms)
        means = rates_per_node.mean(axis=0)  # (4,)

        if not np.all(np.isfinite(means)) or np.any(means > fit_cfg.max_rate):
            return False, np.zeros(4), 0.0

        means_trials.append(means)
        # Spatial coefficient of variation for PYR across nodes
        cv_trials.append(float(np.std(rates_per_node[:, 0]) / (means[0] + 1e-8)))

    means_avg = np.mean(np.stack(means_trials, axis=0), axis=0)
    mean_spatial_cv = float(np.mean(cv_trials))
    return True, means_avg, mean_spatial_cv


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
        record_dt_ms=fit_cfg.record_dt_ms,
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
# Turing bistability loss (simulation trace based)
# ---------------------------------------------------------------------------

# The trace-based loss targets two stable regimes in the ring:
#   - Rest near target rates with gain product safely below threshold.
#   - Cue-evoked bump around 40 Hz with gain floor for sustain and explicit anti-runaway terms.

def _phi_derivative_array(
    I: np.ndarray,
    *,
    theta: float,
    c: float,
    g: float,
) -> np.ndarray:
    """Vectorized Wong-Wang derivative dPhi/dI with stable handling near z=0."""
    u = c * (I - theta)
    z = g * u
    z_clip = np.clip(z, -60.0, 60.0)
    e = np.exp(-z_clip)
    denom = 1.0 - e
    deriv = np.where(
        np.abs(z) < 1e-8,
        c / 2.0,
        c * (1.0 - e * (1.0 + z_clip)) / np.maximum(denom * denom, 1e-16),
    )
    cutoff = -700.0 / max(g, 1e-9)
    deriv = np.where(u < cutoff, 0.0, deriv)
    deriv = np.where(z < -60.0, 0.0, deriv)
    return deriv


def turing_trace_bistability_loss(
    params: CircuitParams,
    ring_params: RingParams,
    cfg: RingFitConfig,
    target: TargetRates,
    *,
    connectivity: RingConnectivity,
    margin: float = 0.05,
    cue_amplitude: float = 0.4,
    cue_duration_ms: float = 250.0,
    cue_sigma_deg: float = 20.0,
    late_delay_ms: float = 500.0,
    bump_min_hz: float = 35.0,
    bump_max_hz: float = 45.0,
    topk_nodes: int = 5,
    runaway_gain_ceiling: float = 1.15,
    background_max_hz: float = 20.0,
) -> float:
    """Simulation-trace loss enforcing rest-vs-bump bistability in the ring.

    A deterministic cue simulation is run once, then gain traces are reconstructed
    from recorded rates and adaptation. The loss combines:
      - rest-rate matching and rest gain ceiling (no spontaneous bump)
      - bump-node late-delay rate band around 40 Hz
      - bump-node late-delay gain floor (self-sustain)
      - bump-node gain ceiling + non-bump rate ceiling (anti-runaway)
    """
    fit_cfg = cfg.fit_cfg
    cue_onset_ms = fit_cfg.burn_in_ms
    cue_current = max(0.0, cue_amplitude) * params.I0_pyr

    stim = RingStimulus(
        center_deg=0.0,
        amplitude=cue_current,
        sigma_deg=cue_sigma_deg,
        onset_ms=cue_onset_ms,
        duration_ms=cue_duration_ms,
    )
    T_ms = cue_onset_ms + cue_duration_ms + max(late_delay_ms, 200.0)

    result = simulate_ring(
        params,
        ring_params,
        T_ms=T_ms,
        dt_ms=fit_cfg.dt_ms,
        stimuli=[stim],
        seed=0,
        noise_type="none",
        tau_noise_ms=fit_cfg.tau_noise_ms,
        connectivity=connectivity,
        record_dt_ms=fit_cfg.record_dt_ms,
        record_adaptation=True,
    )

    if result.I_adapt_stored is None:
        return 1e6

    t_ms = result.t_ms
    r = result.r
    r_pyr = r[:, :, 0]
    r_som = r[:, :, 1]
    r_pv = r[:, :, 2]
    r_vip = r[:, :, 3]
    I_adapt = result.I_adapt_stored[:, :, 0]

    cue_offset_ms = cue_onset_ms + cue_duration_ms
    rest_start_ms = max(0.0, cue_onset_ms - max(fit_cfg.window_ms, 200.0))
    rest_mask = (t_ms >= rest_start_ms) & (t_ms < cue_onset_ms)
    late_start_ms = max(cue_offset_ms, T_ms - late_delay_ms)
    late_mask = t_ms >= late_start_ms

    if not np.any(rest_mask) or not np.any(late_mask):
        return 1e6

    # Reconstruct inter-node currents from recorded rates.
    I_pyr_inter = np.einsum("ij,tj->ti", connectivity.W_pyr_pyr, r_pyr)
    I_pv_inter = np.einsum("ij,tj->ti", connectivity.W_pv_pyr, r_pv)

    # Rebuild the cue current trace exactly as used in simulation.
    node_angles = ring_params.node_angles_rad
    dist = np.abs(node_angles - stim.center_rad)
    dist = np.minimum(dist, 2.0 * np.pi - dist)
    spatial = np.exp(-(dist * dist) / (2.0 * stim.sigma_rad * stim.sigma_rad))
    cue_time = ((t_ms >= cue_onset_ms) & (t_ms < cue_offset_ms)).astype(float)
    I_stim = cue_time[:, None] * (cue_current * spatial[None, :])

    ggaba = params.g_gaba()
    denom = 1.0 + ggaba * params.w_pe * r_pv
    I_pyr = (
        (params.w_ee * r_pyr) / np.maximum(denom, 1e-12)
        + I_pyr_inter
        - ggaba * I_pv_inter
        - ggaba * params.w_se * r_som
        - I_adapt
        + params.I_ext_pyr()
        + I_stim
    )
    I_pv = (
        params.w_ep * r_pyr
        - ggaba * params.w_pp * r_pv
        - ggaba * params.w_sp * r_som
        - params.w_vp * r_vip
        + params.I_ext_pv()
    )

    dphi_pyr = _phi_derivative_array(
        I_pyr,
        theta=params.Theta_pyr,
        c=params.alpha_pyr,
        g=params.g_exc,
    )
    dphi_pv = _phi_derivative_array(
        I_pv,
        theta=params.Theta_pv,
        c=params.alpha_pv,
        g=params.g_inh,
    )
    G_eff = dphi_pyr / (
        1.0
        + params.J_adapt_pyr * dphi_pyr
        + ggaba * params.w_pe * dphi_pv * params.w_ep * dphi_pyr
    )
    gain = G_eff * ring_params.w_pyr_pyr_inter

    # Pick bump-supporting nodes from late-delay mean PYR and keep them fixed.
    late_mean_pyr = r_pyr[late_mask].mean(axis=0)
    k = int(np.clip(topk_nodes, 1, ring_params.n_nodes))
    bump_nodes = np.argsort(late_mean_pyr)[-k:]
    bg_nodes = np.setdiff1d(np.arange(ring_params.n_nodes), bump_nodes)

    rest_gain = gain[rest_mask]
    rest_rate = r_pyr[rest_mask].mean()
    late_gain_bump = gain[late_mask][:, bump_nodes]
    late_rate_bump = r_pyr[late_mask][:, bump_nodes]

    target_rest = max(target.mean_r_pyr, 1e-3)
    loss_rest_rate = ((rest_rate - target_rest) / target_rest) ** 2
    loss_rest_gain = np.mean(np.maximum(0.0, rest_gain - (1.0 - margin)) ** 2)

    mean_bump_rate = float(np.mean(late_rate_bump))
    loss_bump_low = (max(0.0, bump_min_hz - mean_bump_rate) / max(bump_min_hz, 1e-3)) ** 2
    loss_bump_high = (max(0.0, mean_bump_rate - bump_max_hz) / max(bump_max_hz, 1e-3)) ** 2
    loss_bump_rate = loss_bump_low + loss_bump_high

    loss_sustain = np.mean(np.maximum(0.0, (1.0 + margin) - late_gain_bump) ** 2)
    loss_runaway_gain = np.mean(np.maximum(0.0, late_gain_bump - runaway_gain_ceiling) ** 2)

    if bg_nodes.size > 0:
        late_rate_bg = r_pyr[late_mask][:, bg_nodes]
        loss_bg_runaway = np.mean(np.maximum(0.0, late_rate_bg - background_max_hz) ** 2) / (background_max_hz ** 2)
    else:
        loss_bg_runaway = 0.0

    return float(
        loss_rest_rate
        + loss_rest_gain
        + loss_bump_rate
        + loss_sustain
        + 0.5 * loss_runaway_gain
        + 0.5 * loss_bg_runaway
    )


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
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_cue_amplitude: float = 0.4,
    turing_cue_duration_ms: float = 250.0,
    turing_cue_sigma_deg: float = 20.0,
    turing_late_delay_ms: float = 500.0,
    turing_bump_min_hz: float = 35.0,
    turing_bump_max_hz: float = 45.0,
    turing_topk_nodes: int = 5,
    turing_activate_below_ring_rate_loss: float = 1.0,
    spatial_uniformity_weight: float = 0.0,
    ach_ratio_weight: float = 2.0,
    turing_activated_sticky: bool = False,
) -> tuple[float, np.ndarray, KOMeans, "RingLossBreakdown"]:
    """
    Evaluate a (CircuitParams, RingParams) pair.

    Steps:
    1. Run ring at rest (n_trials_ring) → ring rate loss
    2. Run KO conditions on ring
    3. Jacobian connectivity penalty
    4. (optional) Trace-based Turing bistability penalty (turing_weight > 0)
       - Gate: activated when ring_rate_loss <= turing_activate_below_ring_rate_loss OR
         once turing_activated_sticky=True (stays on for rest of optimization)
    4b.(optional) Spatial uniformity penalty (spatial_uniformity_weight > 0):
       penalises high coefficient of variation of PYR rates across nodes at rest,
       discouraging spontaneous bump formation in the resting state.
    5. (Mode 2) Run bump trial → bump loss

    Parameters:
        turing_activated_sticky: If True, Turing loss is always evaluated (sticky gate).
            In the optimization loop, this is set to True once Turing activates, keeping it on
            even if ring_rate_loss rises back above the activation threshold.

    Returns:
        (total_loss, ring_means, ko_means, breakdown)
    """
    # Pre-compute connectivity once for all ring simulations in this evaluation
    connectivity = RingConnectivity.from_params(ring_params)

    # --- Step 1: Ring baseline ---
    ok, ring_means, spatial_cv = run_ring_trials(params, ring_params, cfg, rng, connectivity=connectivity)
    if not ok:
        bd = RingLossBreakdown(ring_rate=1e9, ko_penalty=0., jacobian=0., ack_ratio=0., turing=0., spatial_uniformity=0., bump=0., total=1e9)
        return 1e9, ring_means, KOMeans(), bd

    ring_rate_loss = loss_from_means(ring_means, target)

    # --- Step 2: KO conditions (only if at least one KO target is provided) ---
    ko_means = KOMeans()
    ko_penalty = 0.0
    has_ko_targets = (
        target.alpha7_ko_pyr is not None
        or target.alpha5_ko_pyr is not None
        or target.beta2_ko_pyr is not None
    )
    if has_ko_targets:
        ko_loss = 0.0
        n_ko = 0

        # Run each KO condition on the ring.
        ko_conditions = [
            ("alpha7_ko", replace(params, act_alpha7=0.0, g_alpha7=0.0)),
            ("alpha5_ko", replace(params, act_alpha5=0.0)),
            ("beta2_ko",  replace(params, act_beta2=0.0)),
        ]
        for ko_name, ko_params in ko_conditions:
            ko_ok, ko_m, _ = run_ring_trials(ko_params, ring_params, cfg, rng, connectivity=connectivity)
            if not ko_ok:
                bd = RingLossBreakdown(ring_rate=ring_rate_loss, ko_penalty=1e9, jacobian=0., ack_ratio=0., turing=0., spatial_uniformity=0., bump=0., total=1e9)
                return 1e9, ring_means, ko_means, bd
            if ko_name == "alpha7_ko":
                ko_means.alpha7_ko = ko_m
            elif ko_name == "alpha5_ko":
                ko_means.alpha5_ko = ko_m
            elif ko_name == "beta2_ko":
                ko_means.beta2_ko = ko_m

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

        ko_penalty = ko_loss / n_ko if n_ko > 0 else 0.0

    # --- Step 3: Jacobian penalty (evaluated at ring-averaged rates) ---
    jacobian_loss = 0.0
    if jacobian_weight > 0.0:
        jacobian_loss = jacobian_connectivity_penalty(params, ring_means) * jacobian_weight

    # --- Step 3b: ACh β2/α7 ratio penalty (Koukouli et al. 2025: β2 ~35× α7 on SOM) ---
    ach_loss = 0.0
    if ach_ratio_weight > 0.0:
        ach_loss = ach_ratio_penalty(params, weight=ach_ratio_weight)

    # --- Step 4: Turing bistability penalty (simulation trace, deterministic cue) ---
    # Gate Turing so rate fitting can settle first. Once activated, stay activated (sticky).
    turing_loss = 0.0
    turing_gate_ok = turing_activated_sticky or (ring_rate_loss <= max(0.0, turing_activate_below_ring_rate_loss))
    if turing_weight > 0.0 and turing_gate_ok:
        t_loss = turing_trace_bistability_loss(
            params,
            ring_params,
            cfg,
            target,
            connectivity=connectivity,
            margin=turing_margin,
            cue_amplitude=turing_cue_amplitude,
            cue_duration_ms=turing_cue_duration_ms,
            cue_sigma_deg=turing_cue_sigma_deg,
            late_delay_ms=turing_late_delay_ms,
            bump_min_hz=turing_bump_min_hz,
            bump_max_hz=turing_bump_max_hz,
            topk_nodes=turing_topk_nodes,
        )
        turing_loss = turing_weight * t_loss

    # --- Step 4b: Spatial uniformity penalty (simulation-based) ---
    # Penalises spatial CV of PYR rates at rest: high CV → spontaneous bump at rest.
    spatial_loss = 0.0
    if spatial_uniformity_weight > 0.0:
        spatial_loss = spatial_uniformity_weight * spatial_cv ** 2

    # --- Step 5: Bump quality (Mode 2) ---
    bump_loss = 0.0
    if bump_target is not None:
        b_loss = run_bump_trial(params, ring_params, cfg, bump_target, rng, connectivity=connectivity)
        bump_loss = bump_target.bump_loss_weight * b_loss

    total = ring_rate_loss + ko_penalty + jacobian_loss + ach_loss + turing_loss + spatial_loss + bump_loss

    breakdown = RingLossBreakdown(
        ring_rate=float(ring_rate_loss),
        ko_penalty=float(ko_penalty),
        jacobian=float(jacobian_loss),
        ack_ratio=float(ach_loss),
        turing=float(turing_loss),
        spatial_uniformity=float(spatial_loss),
        bump=float(bump_loss),
        total=float(total),
    )

    return total, ring_means, ko_means, breakdown


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
# Bounds saturation diagnostics
# ---------------------------------------------------------------------------

def _compute_saturation(value: float, lower: float, upper: float) -> float:
    """Compute saturation fraction: (value - lower) / (upper - lower) ∈ [0, 1]."""
    if upper <= lower:
        return 0.5
    sat = (value - lower) / (upper - lower)
    return np.clip(sat, 0.0, 1.0)


def _print_bounds_saturation(
    cand: RingCandidate,
    circuit_bounds: dict[str, ParamBound],
    ring_bounds: dict[str, ParamBound],
    step: int,
    periodic: bool = False,
) -> None:
    """
    Print bounds saturation diagnostics for a candidate.

    If periodic=False (improvement-triggered), prints two lines:
      Line 1: [BOUNDS] step=N loss=X.XXX | saturated: K/total (K_lo at lower, K_hi at upper)
      Line 2: [BOUNDS-SAT] param=value (lo/hi, sat=X.XX) | ... [only if K > 0]

    If periodic=True (every 50 steps), prints just Line 1 with a "[50s]" prefix.
    """
    params = cand.params
    ring_params = cand.ring_params

    # Collect all free parameters with their bounds
    param_info: list[tuple[str, float, float, float]] = []

    # Circuit parameters
    for f in fields(CircuitParams):
        name = f.name
        if name not in circuit_bounds:
            continue
        value = float(getattr(params, name))
        bound = circuit_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((name, value, bound.lo, bound.hi, sat))

    # Ring parameters
    for f in fields(RingParams):
        name = f.name
        if name.startswith('_') or name not in ring_bounds:
            continue
        value = float(getattr(ring_params, name))
        bound = ring_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((f"ring__{name}", value, bound.lo, bound.hi, sat))

    # Count saturated parameters
    saturated = [(name, val, lo, hi, sat) for name, val, lo, hi, sat in param_info
                 if sat < 0.05 or sat > 0.95]
    saturated_lo = sum(1 for _, _, _, _, sat in saturated if sat < 0.05)
    saturated_hi = sum(1 for _, _, _, _, sat in saturated if sat > 0.95)

    # Line 1: Summary
    total = len(param_info)
    n_sat = len(saturated)
    if periodic:
        summary = f"[BOUNDS-50s] step={step} loss={cand.loss:.3f} | saturated: {n_sat}/{total} ({saturated_lo} lo, {saturated_hi} hi)"
    else:
        summary = f"[BOUNDS] step={step} loss={cand.loss:.3f} | saturated: {n_sat}/{total} ({saturated_lo} lo, {saturated_hi} hi)"
    print(summary, file=sys.stderr, flush=True)

    # Line 2: Saturated parameters (only if periodic=False and any saturated)
    if not periodic and saturated:
        # Sort by extremeness (min sat first for lower, then max sat first for upper)
        saturated_sorted = sorted(saturated, key=lambda x: min(x[4], 1 - x[4]))
        sat_strs = []
        for name, val, lo, hi, sat in saturated_sorted:
            extreme_type = "lo" if sat < 0.05 else "hi"
            sat_strs.append(f"{name}={val:.3g} ({extreme_type}, sat={sat:.2f})")
        sat_line = "[BOUNDS-SAT] " + " | ".join(sat_strs)
        print(sat_line, file=sys.stderr, flush=True)


def _print_bounds_periodic_table(
    cand: RingCandidate,
    circuit_bounds: dict[str, ParamBound],
    ring_bounds: dict[str, ParamBound],
    step: int,
) -> None:
    """Print compact table of ALL parameters every 50 steps, sorted by saturation."""
    params = cand.params
    ring_params = cand.ring_params

    # Collect all free parameters with their bounds
    param_info: list[tuple[str, float, float, float, float]] = []

    # Circuit parameters
    for f in fields(CircuitParams):
        name = f.name
        if name not in circuit_bounds:
            continue
        value = float(getattr(params, name))
        bound = circuit_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((name, value, bound.lo, bound.hi, sat))

    # Ring parameters
    for f in fields(RingParams):
        name = f.name
        if name.startswith('_') or name not in ring_bounds:
            continue
        value = float(getattr(ring_params, name))
        bound = ring_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((f"ring__{name}", value, bound.lo, bound.hi, sat))

    # Sort by extremeness: min(sat, 1-sat) ascending
    param_info_sorted = sorted(param_info, key=lambda x: min(x[4], 1 - x[4]))

    # Print header
    print(f"[BOUNDS-50s] step={step}", file=sys.stderr, flush=True)
    print("[BOUNDS-PARAMS] param | value | lower | upper | sat", file=sys.stderr, flush=True)

    # Print each parameter
    for name, val, lo, hi, sat in param_info_sorted:
        marker = ""
        if sat < 0.05:
            marker = " <-- LOWER"
        elif sat > 0.95:
            marker = " <-- UPPER"
        line = f"  {name:<25} | {val:>10.3g} | {lo:>10.3g} | {hi:>10.3g} | {sat:>5.2f}{marker}"
        print(line, file=sys.stderr, flush=True)


def _print_turing_diagnostic(
    params: CircuitParams,
    ring_params: RingParams,
    r_ss: np.ndarray,
    margin: float = 0.05,
    cue_scale: float = 0.4,
    rate_loss: float | None = None,
) -> float:
    """Legacy helper retained for API compatibility after trace-loss migration."""
    rate_loss_str = f"L_rate={rate_loss:.4g}" if rate_loss is not None else "L_rate=N/A"
    print(
        "[TURING] analytical diagnostic is deprecated; "
        f"trace-based loss is active (margin={margin:.3g}, cue_amp={cue_scale:.3g}) {rate_loss_str}",
        file=sys.stderr,
        flush=True,
    )
    return 0.0


def _print_bounds_final_table(
    cand: RingCandidate,
    circuit_bounds: dict[str, ParamBound],
    ring_bounds: dict[str, ParamBound],
) -> None:
    """Print final table of ALL parameters sorted by saturation extremeness."""
    params = cand.params
    ring_params = cand.ring_params

    # Collect all free parameters with their bounds
    param_info: list[tuple[str, float, float, float, float]] = []

    # Circuit parameters
    for f in fields(CircuitParams):
        name = f.name
        if name not in circuit_bounds:
            continue
        value = float(getattr(params, name))
        bound = circuit_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((name, value, bound.lo, bound.hi, sat))

    # Ring parameters
    for f in fields(RingParams):
        name = f.name
        if name.startswith('_') or name not in ring_bounds:
            continue
        value = float(getattr(ring_params, name))
        bound = ring_bounds[name]
        sat = _compute_saturation(value, bound.lo, bound.hi)
        param_info.append((f"ring__{name}", value, bound.lo, bound.hi, sat))

    # Sort by extremeness: min(sat, 1-sat) ascending
    param_info_sorted = sorted(param_info, key=lambda x: min(x[4], 1 - x[4]))

    # Print header
    print("[BOUNDS-FINAL] param | value | lower | upper | sat", file=sys.stderr, flush=True)

    # Print each parameter with extremeness marker
    for name, val, lo, hi, sat in param_info_sorted:
        marker = ""
        if sat < 0.05:
            marker = " <-- LOWER"
        elif sat > 0.95:
            marker = " <-- UPPER"
        line = f"  {name:<25} | {val:>10.3g} | {lo:>10.3g} | {hi:>10.3g} | {sat:>5.2f}{marker}"
        print(line, file=sys.stderr, flush=True)

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
    plateau_patience: int = 500,
    log_file: Optional[str] = None,
    log_interval: int = 50,
    save_output_dir: Optional[str] = None,
    jacobian_weight: float = 1.0,
    turing_weight: float = 2.0,
    turing_margin: float = 0.05,
    turing_cue_amplitude: float = 0.4,
    turing_cue_duration_ms: float = 250.0,
    turing_cue_sigma_deg: float = 20.0,
    turing_late_delay_ms: float = 500.0,
    turing_bump_min_hz: float = 35.0,
    turing_bump_max_hz: float = 45.0,
    turing_topk_nodes: int = 5,
    turing_activate_below_ring_rate_loss: float = 1.0,
    spatial_uniformity_weight: float = 0.0,
    ach_ratio_weight: float = 2.0,
) -> list[RingCandidate]:
    """
    Joint optimization of CircuitParams + RingParams against ring-level targets.

    Mode 1 (bump_target=None):
        total_loss = ring_rate_loss + ko_loss/n_ko + jacobian_penalty
                   [+ turing_weight * turing_loss  if turing_weight > 0]
                   [+ spatial_uniformity_weight * spatial_cv²  if > 0]

    Mode 2 (bump_target set):
        total_loss = ring_rate_loss + ko_loss/n_ko + jacobian_penalty
                   [+ turing_weight * turing_loss  if turing_weight > 0]
                   [+ spatial_uniformity_weight * spatial_cv²  if > 0]
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
        turing_weight: Weight of the trace-based Turing bistability penalty (0 = disabled)
        turing_margin: Safety margin around the Turing threshold (default 0.05)
        turing_cue_amplitude: Additive cue amplitude factor on I0_pyr for deterministic Turing pass.
        turing_activate_below_ring_rate_loss: Activate Turing term only when ring_rate_loss
            is <= this threshold (default 1.0).
        spatial_uniformity_weight: Weight of the spatial uniformity penalty (0 = disabled).
            Penalises std(r_pyr_nodes)/mean(r_pyr_nodes) at rest to prevent spontaneous bump formation.
        ach_ratio_weight: Weight of the β2/α7 ACh current ratio penalty (default 2.0, 0 = disabled).
            Penalises solutions where I_beta2_som / I_alpha7_som deviates from 35 (Koukouli et al. 2025).

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

    turing_str = (
        f" + Turing penalty (w={turing_weight}, margin={turing_margin}, "
        f"active_when_rate_loss<={turing_activate_below_ring_rate_loss})"
        if turing_weight > 0.0
        else ""
    )
    mode_str = ("Mode 2 (legacy bump term enabled)" if bump_target is not None else "Mode 1 (rates only)") + turing_str
    print(f"Ring joint optimization — {mode_str}")
    if optimizer == "chaining":
        print(f"Optimizer: {optimizer} (DE → Nelder-Mead at step 5000)")
    else:
        print(f"Optimizer: {optimizer}")
    print(f"Ring trials per eval: {ring_cfg.n_trials_ring}")
    print("KO conditions on: ring")
    print(f"Plateau patience: {plateau_patience} steps (switching to Nelder-Mead at step 5000)")

    # For dynamic chaining: track if we've switched and current optimizer
    chaining_active = optimizer == "chaining"
    optimizer_switched = False
    current_ng_optimizer = ng_optimizer
    current_optimizer_name = "DE" if chaining_active else optimizer

    best: list[RingCandidate] = []
    steps_since_improvement = 0
    last_step = 0
    stopped_early = False
    turing_activated_sticky = False  # Sticky gate: once Turing activates, keep it on

    pbar = tqdm(range(1, n_samples + 1), desc="Ring-Optimize", unit="step", position=0, leave=True)
    try:
        for step in pbar:
            last_step = step

            x = current_ng_optimizer.ask()
            ng_dict = x.value

            p = params_from_ng_dict(ng_dict, base_circuit)
            rp = ring_params_from_ng_dict(ng_dict, base_ring)

            L, ring_means, ko_means, breakdown = evaluate_ring_params(
                p,
                rp,
                target,
                ring_cfg,
                bump_target,
                rng,
                jacobian_weight,
                turing_weight,
                turing_margin,
                turing_cue_amplitude,
                turing_cue_duration_ms,
                turing_cue_sigma_deg,
                turing_late_delay_ms,
                turing_bump_min_hz,
                turing_bump_max_hz,
                turing_topk_nodes,
                turing_activate_below_ring_rate_loss,
                spatial_uniformity_weight,
                ach_ratio_weight,
                turing_activated_sticky=turing_activated_sticky,
            )
            # Once Turing loss becomes non-zero, keep it activated (sticky gate)
            if turing_weight > 0.0 and breakdown.turing > 0.0:
                turing_activated_sticky = True
            current_ng_optimizer.tell(x, L)

            prev_best_loss = best[0].loss if best else float("inf")
            cand = RingCandidate(loss=L, ring_means=ring_means, ko_means=ko_means, params=p, ring_params=rp, breakdown=breakdown)

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
            else:
                steps_since_improvement += 1

            # Chaining: switch to Nelder-Mead after 5000 steps
            if chaining_active and not optimizer_switched and step == 5000:
                remaining_budget = n_samples - step
                if remaining_budget > 0:
                    current_ng_optimizer = ng.optimizers.NelderMead(
                        parametrization=parametrization, budget=remaining_budget, num_workers=1
                    )
                    optimizer_switched = True
                    current_optimizer_name = "NM"
                    print(f"\n→ Switched to Nelder-Mead at step {step} (loss={best[0].loss:.4g})")
                    steps_since_improvement = 0  # Reset plateau counter after switch

            # Build postfix with loss breakdown
            postfix = {"opt": current_optimizer_name, "plateau": steps_since_improvement}
            if best:
                bd = best[0].breakdown
                if bd:
                    postfix.update({
                        "rate": f"{bd.ring_rate:.2g}",
                        "jac": f"{bd.jacobian:.2g}",
                        "tur": f"{bd.turing:.2g}",
                        "pyr": f"{best[0].ring_means[0]:.2f}",
                    })
                else:
                    postfix["loss"] = f"{best[0].loss:.4g}"
            pbar.set_postfix(postfix)

            if log_file and step % log_interval == 0 and best:
                _log_ring_candidate(log_file, step, best[0], target)
                # Generate loss evolution plots every log_interval steps
                try:
                    _generate_loss_plots(log_file)
                except Exception:
                    pass

            if early_stop_loss is not None and best and best[0].loss <= early_stop_loss:
                if log_file and best:
                    _log_ring_candidate(log_file, step, best[0], target)
                    try:
                        _generate_loss_plots(log_file)
                    except Exception:
                        pass
                stopped_early = True
                break

            if plateau_patience > 0 and steps_since_improvement >= plateau_patience:
                if chaining_active and not optimizer_switched:
                    # During TwoPointsDE phase: switch to Nelder-Mead early instead of stopping
                    remaining_budget = n_samples - step
                    if remaining_budget > 0:
                        current_ng_optimizer = ng.optimizers.NelderMead(
                            parametrization=parametrization, budget=remaining_budget, num_workers=1
                        )
                        optimizer_switched = True
                        current_optimizer_name = "NM"
                        steps_since_improvement = 0
                        print(f"\n→ Plateau after {plateau_patience} steps — switched to Nelder-Mead at step {step} (loss={best[0].loss:.4g})")
                        continue
                print(f"\nEarly stop: no improvement for {plateau_patience} steps.")
                if log_file and best:
                    _log_ring_candidate(log_file, step, best[0], target)
                    try:
                        _generate_loss_plots(log_file)
                    except Exception:
                        pass
                stopped_early = True
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr, flush=True)
    finally:
        pbar.close()

    if log_file and best and (not stopped_early) and last_step % log_interval != 0:
        _log_ring_candidate(log_file, last_step, best[0], target)
        try:
            _generate_loss_plots(log_file)
        except Exception:
            pass

    # Print final bounds saturation table
    if best:
        print("\n", file=sys.stderr, flush=True)
        _print_bounds_final_table(best[0], circuit_bounds, ring_bounds)

    return best


def _generate_loss_plots(log_file: str) -> None:
    """Generate loss evolution plots from the current log file.
    
    Called periodically during ring optimization to update live visualizations.
    Saves plots to the same directory as the log file.
    """
    try:
        from ..loss_evolution_plot import plot_loss_evolution, plot_loss_evolution_ratios
        
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
    if cand.breakdown is not None:
        entry["breakdown"] = {
            "ring_rate": float(cand.breakdown.ring_rate),
            "ko_penalty": float(cand.breakdown.ko_penalty),
            "jacobian": float(cand.breakdown.jacobian),
            "ach_ratio": float(cand.breakdown.ack_ratio),
            "turing": float(cand.breakdown.turing),
            "spatial_uniformity": float(cand.breakdown.spatial_uniformity),
            "bump": float(cand.breakdown.bump),
            "total": float(cand.breakdown.total),
        }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
