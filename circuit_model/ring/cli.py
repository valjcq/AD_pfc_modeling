"""
Ring attractor CLI logic.

This module contains the ring-specific CLI functions (cmd_run, cmd_study)
and their helpers. These are invoked from circuit_model.cli via the
ring-run and ring-study subcommands.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_MP_CONTEXT = multiprocessing.get_context('spawn')
from dataclasses import replace
from typing import Optional

import numpy as np

from ..params import CircuitParams
from ..io import load_params_json, save_params_json, save_fit_summary_txt, build_fit_comparison, output_dir as _output_dir
from ..study import STUDY_CONDITIONS, CONDITION_ORDER, apply_condition as _study_apply_condition
from ..defaults import DEFAULT_WT_PARAMS_PATH, DEFAULT_APP_PARAMS_PATH, DEFAULT_WT_RING_PARAMS_PATH, DEFAULT_APP_RING_PARAMS_PATH

from .params import RingParams
from .stimulus import RingStimulus
from .simulation import simulate_ring
from .connectivity import RingConnectivity
from .constants import TRANSIENT_SKIP_TIME_MS
from .analysis import (
    compute_bump_metrics,  # noqa: F401 (used by ring-run)
    compute_metrics_at_delay_times,
    aggregate_metrics_across_trials,
    aggregate_single_metrics,
    population_vector_decode,
    compute_noise_floor,
    compute_oscillation_band_timecourse,
    summarize_oscillation_timecourse,
    compute_plv_timecourse,
)
from .plotting import (
    plot_ring_dashboard,
    animate_ring_snapshot_evolution,
    plot_ring_connectome,
    plot_connectivity_matrices,
    plot_bump_metrics_over_time,
    plot_population_activity,
    extract_comparison_data,
    plot_bump_metrics_comparison,
    plot_metrics_vs_delay,
    plot_metrics_vs_amplitude,
    plot_noise_floor_histogram,
    plot_calibration_heatmap,
    plot_calibration_timecourses,
    plot_noise_summary,
    plot_oscillation_band_heatmap,
    plot_oscillation_violin,
    plot_oscillation_multi_violin,
    plot_oscillation_amp_sweep_violin,
    plot_oscillation_amp_sweep_lines,
    plot_osc_distractor_timecourses,
    plot_osc_distractor_spectrograms,
    plot_osc_distractor_amp_sweep,
    plot_osc_phase_timecourses_grid,
    plot_osc_phase_sweep,
    plot_osc_phase_polar,
    plot_osc_phase_heatmap,
    plot_osc_phase_sweep_offsets,
    plot_osc_conditions_boxplot,
    plot_pre_cue_power_spectrum,
    plot_pre_cue_power_metric,
    plot_study_firing_rates_violin,
)

from .optimization import evaluate_ring_params, RingFitConfig


# ============================================================================
# JAX GPU DETECTION
# ============================================================================

def _resolve_workers(args) -> int:
    """Return worker count: requested, or half of available CPUs (min 1, max 16)."""
    requested = getattr(args, 'n_workers', None)
    if requested is None:
        n_cpu = os.cpu_count() or 4
        requested = max(1, min(n_cpu // 2, 16))
    return requested


# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

BURN_IN_MS = 10000.0
STIM_ONSET_MS = BURN_IN_MS + 500.0
STIM_DURATION_MS = 250.0
STIM_CENTER_DEG = 180.0
STIM_SIGMA_DEG = 18.0
RATE_CAP_HZ = 200.0
CAP_WARNING_FRACTION = 0.10

# 3D calibration sweep: bump state thresholds
CAL3D_SAT_THRESH_HZ = 90.0     # max PYR >= this -> saturated state
CAL3D_RESTING_MULT = 2.5       # bump lower bound = resting_hz * this
CAL3D_BUMP_MIN_HZ = 10.0       # minimum bump threshold regardless of resting
CAL3D_CUE_SAT_THRESH_HZ = 190.0  # cue peak >= this -> cue saturated

BUMP_DECAY_REF_OFFSET_MS: float = 400.0  # ms after cue offset used as normalization reference

DEFAULT_FIT_INIT_KWARGS = {
    "I0_pv": 0.35,
    "I0_pyr": 0.44,
    "I0_som": 0.35,
    "I0_vip": 0.33,
    "I_alpha5_vip": 0.0,
    "I_alpha7_pv": 0.0,
    "I_alpha7_som": 0.0,
    "I_beta2_som": 0.0,
    "J_adapt_pyr": 0.002,
    "J_adapt_som": 0.0,
    "Theta_pv": 0.2878,
    "Theta_pyr": 0.40323,
    "Theta_som": 0.2878,
    "Theta_vip": 0.2878,
    "act_alpha5": 1.0,
    "act_alpha7": 1.0,
    "act_beta2": 1.0,
    "alpha_pv": 615.0,
    "alpha_pyr": 310.0,
    "alpha_som": 615.0,
    "alpha_vip": 615.0,
    "g_alpha7": 0.0,
    "g_exc": 0.16,
    "g_gaba_base": 1.0,
    "g_inh": 0.087,
    "sigma_noise": 0.3,
    "sigma_s": 0.0,
    "tau_adapt_pyr": 600.0,
    "tau_adapt_som": 150.0,
    "tau_s": 20.0,
    "trans_duration_ms": 500.0,
    "trans_enabled": False,
    "trans_factor": 0.2,
    "trans_start_ms": 1000.0,
    "J_NMDA": 0.3,
    "w_ep": 0.002,
    "w_es": 0.002,
    "w_ev": 0.002,
    "w_pe": 0.05,
    "w_pp": 0.002,
    "w_se": 0.002,
    "w_sp": 0.002,
    "w_vp": 0.002,
    "w_vs": 0.002,
}


def _default_fit_init_params() -> CircuitParams:
    """Return hardcoded fit initialization parameters."""
    return CircuitParams(**DEFAULT_FIT_INIT_KWARGS)

# Set per-command when loading base params. The local apply_condition wrapper
# then uses this as app_params for *_APP conditions automatically.
_ACTIVE_APP_PARAMS: CircuitParams | None = None
_ACTIVE_RING_PARAMS: RingParams | None = None
_ACTIVE_APP_RING_PARAMS: RingParams | None = None
# Set to True when ring args were filled from WT defaults (not user-supplied).
_ring_args_from_defaults: bool = False

# Fallback ring param values used when no JSON file is found.
_RING_PARAMS_FALLBACK = {"w_pyr_pyr_inter": 8.0, "sigma_pyr_deg": 30.0, "w_pv_global": 10.0, "n_nodes": 128}


def _print_ring_init_summary(base_circuit: CircuitParams, base_ring: RingParams, ring_means: np.ndarray, init_loss: float) -> None:
    """Print effective ring optimization initialization and its predicted ring rates."""
    print("Initial condition (effective after --set/--no_adapt):")
    print(f"  Circuit I0: pyr={base_circuit.I0_pyr:.6g}, som={base_circuit.I0_som:.6g}, pv={base_circuit.I0_pv:.6g}, vip={base_circuit.I0_vip:.6g}")
    print(f"  Ring: n_nodes={base_ring.n_nodes}, w_pyr_pyr_inter={base_ring.w_pyr_pyr_inter:.6g}, w_pv_global={base_ring.w_pv_global:.6g}, sigma_pyr_deg={base_ring.sigma_pyr_deg:.6g}")
    print("Initial predicted ring rates (Hz):")
    print(f"  PYR={ring_means[0]:.4f}, SOM={ring_means[1]:.4f}, PV={ring_means[2]:.4f}, VIP={ring_means[3]:.4f}")
    print(f"  Initial loss={init_loss:.6g}")


def _load_ring_params_json(path: str) -> RingParams:
    """Load RingParams from a JSON file."""
    import json
    from dataclasses import fields as _fields, replace as _replace

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    allowed = {fld.name for fld in _fields(RingParams) if not fld.name.startswith("_")}
    # RingParams has required fields — build via replace from a dummy with placeholder values
    clean = {k: d[k] for k in d if k in allowed}
    # Construct with required args present; fill with loaded values
    base = RingParams(
        w_pyr_pyr_inter=clean.pop("w_pyr_pyr_inter"),
        w_pv_global=clean.pop("w_pv_global"),
    )
    return _replace(base, **clean) if clean else base


def _load_base_params_for_ring(
    params_json: str,
    args=None,
    condition_key: str | None = None,
) -> tuple[CircuitParams, str]:
    """Load base params for ring experiments and configure APP-family context.

    When *args* is provided and ring-connectivity args are ``None`` (i.e. not
    explicitly set on the CLI), the values are filled from the default ring
    params JSON (``DEFAULT_WT_RING_PARAMS_PATH``) or from hard-coded fallbacks.
    """
    global _ACTIVE_APP_PARAMS, _ACTIVE_RING_PARAMS, _ACTIVE_APP_RING_PARAMS, _ring_args_from_defaults

    if params_json:
        _ACTIVE_APP_PARAMS = None
        _ACTIVE_RING_PARAMS = None
        _ACTIVE_APP_RING_PARAMS = None
        result = load_params_json(params_json), f"Loaded parameters from: {params_json}"
    elif DEFAULT_WT_PARAMS_PATH.exists():
        base = load_params_json(str(DEFAULT_WT_PARAMS_PATH))
        _ACTIVE_APP_PARAMS = (
            load_params_json(str(DEFAULT_APP_PARAMS_PATH))
            if DEFAULT_APP_PARAMS_PATH.exists()
            else None
        )
        _ACTIVE_RING_PARAMS = (
            _load_ring_params_json(str(DEFAULT_WT_RING_PARAMS_PATH))
            if DEFAULT_WT_RING_PARAMS_PATH.exists()
            else None
        )
        _ACTIVE_APP_RING_PARAMS = (
            _load_ring_params_json(str(DEFAULT_APP_RING_PARAMS_PATH))
            if DEFAULT_APP_RING_PARAMS_PATH.exists()
            else None
        )
        if _ACTIVE_APP_PARAMS is None:
            msg = (
                f"Loaded default WT parameters from: {DEFAULT_WT_PARAMS_PATH} "
                f"(WT_APP defaults not found at {DEFAULT_APP_PARAMS_PATH})"
            )
        else:
            msg = (
                f"Loaded default WT/WT_APP parameters from: {DEFAULT_WT_PARAMS_PATH} "
                f"and {DEFAULT_APP_PARAMS_PATH}"
            )
        if _ACTIVE_RING_PARAMS is not None:
            msg += f"\nLoaded default ring parameters from: {DEFAULT_WT_RING_PARAMS_PATH}"
        if _ACTIVE_APP_RING_PARAMS is not None:
            msg += f"\nLoaded default WT_APP ring parameters from: {DEFAULT_APP_RING_PARAMS_PATH}"
        result = base, msg
    elif (
        condition_key in STUDY_CONDITIONS
        and STUDY_CONDITIONS[condition_key].is_app
        and DEFAULT_APP_PARAMS_PATH.exists()
    ):
        # If WT defaults are unavailable but the requested condition is APP-family,
        # use APP defaults directly instead of hardcoded fit-init values.
        base = load_params_json(str(DEFAULT_APP_PARAMS_PATH))
        _ACTIVE_APP_PARAMS = base
        _ACTIVE_RING_PARAMS = (
            _load_ring_params_json(str(DEFAULT_APP_RING_PARAMS_PATH))
            if DEFAULT_APP_RING_PARAMS_PATH.exists()
            else None
        )
        _ACTIVE_APP_RING_PARAMS = _ACTIVE_RING_PARAMS
        msg = (
            f"Loaded APP default parameters from: {DEFAULT_APP_PARAMS_PATH} "
            f"(WT defaults not found at {DEFAULT_WT_PARAMS_PATH})"
        )
        if _ACTIVE_RING_PARAMS is not None:
            msg += f"\nLoaded APP ring parameters from: {DEFAULT_APP_RING_PARAMS_PATH}"
        result = base, msg
    else:
        _ACTIVE_APP_PARAMS = None
        _ACTIVE_RING_PARAMS = None
        _ACTIVE_APP_RING_PARAMS = None
        result = _default_fit_init_params(), "Using hardcoded fit-init default parameters"

    # Patch CLI args that were left at None with values from the ring params JSON.
    if args is not None:
        _rp = _ACTIVE_RING_PARAMS
        fb = _RING_PARAMS_FALLBACK
        # True when the user did not explicitly provide connectivity args —
        # n_nodes is intentionally excluded (ring size ≠ connectivity profile).
        _ring_args_from_defaults = (
            args.w_pyr_pyr_inter is None
            and args.sigma_pyr_deg is None
            and args.w_pv_global is None
        )
        if args.w_pyr_pyr_inter is None:
            args.w_pyr_pyr_inter = [_rp.w_pyr_pyr_inter if _rp else fb["w_pyr_pyr_inter"]]
        if args.sigma_pyr_deg is None:
            args.sigma_pyr_deg = _rp.sigma_pyr_deg if _rp else fb["sigma_pyr_deg"]
        if args.w_pv_global is None:
            args.w_pv_global = _rp.w_pv_global if _rp else fb["w_pv_global"]
        if args.n_nodes is None:
            args.n_nodes = _rp.n_nodes if _rp else fb["n_nodes"]

    return result


def apply_condition(
    base_params: CircuitParams,
    condition,
    rng: np.random.Generator | None = None,
    app_params: CircuitParams | None = None,
) -> CircuitParams:
    """Ring-local condition application with automatic WT_APP family support."""
    chosen_app = app_params if app_params is not None else _ACTIVE_APP_PARAMS
    return _study_apply_condition(base_params, condition, rng=rng, app_params=chosen_app)


def _base_rp_for_cond(cond_key: str, default_rp: "RingParams") -> "RingParams":
    """Return the appropriate base RingParams for a condition.

    APP-family conditions use _ACTIVE_APP_RING_PARAMS when available; all
    others use the provided default_rp (typically built from CLI args / WT JSON).
    """
    if (
        _ACTIVE_APP_RING_PARAMS is not None
        and cond_key in STUDY_CONDITIONS
        and STUDY_CONDITIONS[cond_key].is_app
    ):
        return _ACTIVE_APP_RING_PARAMS
    return default_rp


def _has_distractor(args) -> bool:
    """Return True when both distractor_factor and distractor_offset_deg are set."""
    return (
        getattr(args, 'distractor_factor', None) is not None
        and getattr(args, 'distractor_offset_deg', None) is not None
    )


def _compute_delay_end_ms(args, stim_offset_ms: float) -> float:
    """Return the end-of-delay time (ms) used for response-transient placement.

    Without distractor: stim_offset + delay_ms
    With distractor:    stim_offset + delay1 + distractor_duration + delay2
    """
    if _has_distractor(args):
        dist_onset_ms = stim_offset_ms + args.delay_ms
        dist_offset_ms = dist_onset_ms + args.distractor_duration_ms
        return dist_offset_ms + args.delay2_ms
    return stim_offset_ms + args.delay_ms


def _build_common(args, amp_factor: float | None = None):
    """Build base params, ring params, T_ms, and stimuli from parsed args.

    The *amp_factor* (or ``args.amplitude``) is a **multiplier of
    I_ext_pyr**.  The actual peak current injected into the stimulus is
    ``amp_factor * base_params.I_ext_pyr()``.

    Returns:
        (base_params, ring_params, T_ms, stimuli, amp_factor, load_msg)
    """
    cond_key = getattr(args, "condition", None)
    # Save n_nodes before _load_base_params_for_ring patches it from defaults,
    # so we can tell whether the user explicitly passed --n_nodes.
    _explicit_n_nodes = args.n_nodes
    base_params, load_msg = _load_base_params_for_ring(
        args.params_json,
        args,
        condition_key=cond_key,
    )

    # Load ring params from JSON if provided, otherwise construct from args
    if args.ring_params_json:
        from ..io import load_ring_params_json
        ring_params = load_ring_params_json(args.ring_params_json)
        load_msg += f"\nLoaded RingParams from: {args.ring_params_json}"
        # CLI --n_nodes overrides the JSON value when explicitly provided.
        if _explicit_n_nodes is not None:
            ring_params = replace(ring_params, n_nodes=_explicit_n_nodes)
            load_msg += f"\nOverriding n_nodes from CLI: {_explicit_n_nodes}"
    else:
        ring_params = RingParams(
            n_nodes=args.n_nodes,
            w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
            sigma_pyr_deg=args.sigma_pyr_deg,
            w_pv_global=args.w_pv_global,
        )

    factor = amp_factor if amp_factor is not None else args.amplitude[0]
    actual_current = factor * base_params.I_ext_pyr()

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = _compute_delay_end_ms(args, stim_offset_ms)

    response_onset_ms = getattr(args, 'response_onset_ms', 0.0)
    response_duration_ms = getattr(args, 'response_duration_ms', 500.0)
    post_response_ms = getattr(args, 'post_response_ms', 3000.0)

    if response_onset_ms > 0:
        trans_start = delay_end_ms + response_onset_ms
        T_ms = trans_start + response_duration_ms + post_response_ms
    elif getattr(args, 'total_time_ms', None) is not None:
        if args.total_time_ms < delay_end_ms:
            print(f"Error: total_time_ms ({args.total_time_ms} ms) must be "
                  f">= delay end time ({delay_end_ms} ms)")
            sys.exit(1)
        T_ms = args.total_time_ms
    else:
        T_ms = delay_end_ms

    stimuli = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS, duration_ms=STIM_DURATION_MS,
        ),
    ]

    if _has_distractor(args):
        dist_onset_ms = stim_offset_ms + args.delay_ms
        dist_center_deg = (STIM_CENTER_DEG + args.distractor_offset_deg) % 360.0
        dist_current = args.distractor_factor * actual_current
        stimuli.append(
            RingStimulus(
                center_deg=dist_center_deg,
                amplitude=dist_current,
                sigma_deg=STIM_SIGMA_DEG,
                onset_ms=dist_onset_ms,
                duration_ms=args.distractor_duration_ms,
            )
        )

    return base_params, ring_params, T_ms, stimuli, factor, load_msg


def _apply_response_transient(params: CircuitParams, args, delay_end_ms: float) -> CircuitParams:
    """Apply response transient settings to CircuitParams if enabled."""
    response_onset_ms = getattr(args, 'response_onset_ms', 0.0)
    if response_onset_ms <= 0:
        return params
    response_duration_ms = getattr(args, 'response_duration_ms', 500.0)
    response_factor = getattr(args, 'response_factor', 0.5)
    trans_start = delay_end_ms + response_onset_ms
    return replace(params,
                   trans_enabled=True,
                   trans_start_ms=trans_start,
                   trans_duration_ms=response_duration_ms,
                   trans_factor=response_factor)


def _print_config(args, amp_factor: float, base_params: CircuitParams, T_ms: float,
                  ring_params: RingParams | None = None,
                  experiment_info: list[str] | None = None,
                  save_path: str | None = None):
    """Print a comprehensive configuration summary and optionally save it to a file."""
    import datetime
    sep = "═" * 66
    thin = "─" * 66

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"\n{sep}")
    emit(f"  Run started: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    emit(sep)

    # ── Circuit parameters ───────────────────────────────────────────────
    emit("  CIRCUIT PARAMETERS")

    emit("  ── Time constants (ms)")
    emit(f"       tau_s         = {base_params.tau_s:.4g}")
    emit(f"       tau_adapt_pyr = {base_params.tau_adapt_pyr:.4g}")

    emit("  ── Adaptation")
    emit(f"       J_adapt_pyr   = {base_params.J_adapt_pyr:.4g}")

    emit("  ── Noise")
    emit(f"       sigma_noise   = {base_params.sigma_noise:.4g}"
         f"  (noise current std = {base_params.sigma_noise * base_params.I_ext_pyr():.4g} nA)")

    emit("  ── GABA scaling")
    emit(f"       g_gaba_base   = {base_params.g_gaba_base:.4g}")
    emit(f"       g_alpha7      = {base_params.g_alpha7:.4g}")
    emit(f"       g_gaba (total)= {base_params.g_gaba():.4g}")

    emit("  ── Receptor activation")
    emit(f"       act_alpha7    = {base_params.act_alpha7:.4g}"
         f"   act_beta2 = {base_params.act_beta2:.4g}"
         f"   act_alpha5 = {base_params.act_alpha5:.4g}")

    emit("  ── Synaptic weights")
    emit(f"       J_NMDA={base_params.J_NMDA:<8.4g}  w_ep={base_params.w_ep:<8.4g}"
         f"  w_es={base_params.w_es:<8.4g}  w_ev={base_params.w_ev:.2e}")
    emit(f"       w_pe={base_params.w_pe:<8.4g}  w_pp={base_params.w_pp:.4g}")
    emit(f"       w_se={base_params.w_se:<8.4g}  w_sp={base_params.w_sp:.2e}")
    emit(f"       w_vp={base_params.w_vp:<8.4g}  w_vs={base_params.w_vs:.4g}")

    emit("  ── External currents")
    emit(f"       PYR: I0={base_params.I0_pyr:.4g}"
         f"  → I_ext_pyr={base_params.I_ext_pyr():.4g}")
    emit(f"       PV:  I0={base_params.I0_pv:.4g}"
         f"  + act_alpha7×{base_params.I_alpha7_pv:.4g}"
         f"  → I_ext_pv={base_params.I_ext_pv():.4g}")
    emit(f"       SOM: I0={base_params.I0_som:.4g}"
         f"  + act_alpha7×{base_params.I_alpha7_som:.4g}"
         f"  + act_beta2×{base_params.I_beta2_som:.4g}"
         f"  → I_ext_som={base_params.I_ext_som():.4g}")
    emit(f"       VIP: I0={base_params.I0_vip:.4g}"
         f"  + act_alpha5×{base_params.I_alpha5_vip:.4g}"
         f"  → I_ext_vip={base_params.I_ext_vip():.4g}")

    if base_params.trans_enabled:
        emit("  ── Transient current")
        emit(f"       trans_factor={base_params.trans_factor:.4g}"
             f"   start={base_params.trans_start_ms:.0f} ms"
             f"   duration={base_params.trans_duration_ms:.0f} ms")

    emit("  ── Transfer function")
    emit(f"       g_exc={base_params.g_exc:.4g} (PYR)   g_inh={base_params.g_inh:.4g} (SOM/PV/VIP)")
    emit(f"       PYR: Theta={base_params.Theta_pyr:.4g}  alpha={base_params.alpha_pyr:.4g}")
    emit(f"       PV:  Theta={base_params.Theta_pv:.4g}  alpha={base_params.alpha_pv:.4g}")
    emit(f"       SOM: Theta={base_params.Theta_som:.4g}  alpha={base_params.alpha_som:.4g}")
    emit(f"       VIP: Theta={base_params.Theta_vip:.4g}  alpha={base_params.alpha_vip:.4g}")

    # ── Ring network ─────────────────────────────────────────────────────
    if ring_params is not None:
        emit(thin)
        emit("  RING NETWORK")
        emit(f"       n_nodes         = {ring_params.n_nodes}")
        emit(f"       w_pyr_pyr_inter = {ring_params.w_pyr_pyr_inter:.4g}")
        emit(f"       sigma_pyr_deg   = {ring_params.sigma_pyr_deg:.4g} deg")
        emit(f"       w_pv_global     = {ring_params.w_pv_global:.4g}")

    # ── Stimulus ─────────────────────────────────────────────────────────
    emit(thin)
    emit("  STIMULUS")
    I_baseline = base_params.I_ext_pyr()
    actual_current = amp_factor * I_baseline
    emit(f"       amplitude     = {amp_factor:.4g}× I_ext_pyr = {actual_current:.4g}"
         f"   (I_ext_pyr baseline = {I_baseline:.4g})")
    emit(f"       sigma         = {STIM_SIGMA_DEG:.0f} deg"
         f"   duration = {STIM_DURATION_MS:.0f} ms"
         f"   onset = {STIM_ONSET_MS:.0f} ms")
    if T_ms > 0:
        emit(f"       total sim time= {T_ms:.0f} ms")

    response_onset = getattr(args, 'response_onset_ms', 0.0)
    if response_onset > 0:
        response_factor = getattr(args, 'response_factor', 0.5)
        response_duration = getattr(args, 'response_duration_ms', 500.0)
        emit(f"       response transient: +{response_factor:.0%} × I0"
             f"   onset={response_onset:.0f} ms after delay end"
             f"   duration={response_duration:.0f} ms")

    # ── Experiment-specific ───────────────────────────────────────────────
    if experiment_info:
        emit(thin)
        emit("  EXPERIMENT")
        for line in experiment_info:
            emit(f"       {line}")

    emit(sep)

    if save_path:
        with open(save_path, "w") as _f:
            _f.write("\n".join(lines) + "\n")
        print(f"  Config saved → {save_path}")


def _fmt(v: float) -> str:
    """Format float for labels/paths: drop trailing zeros, up to 4 decimals for small values."""
    if abs(v) < 0.1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _format_duration_human(seconds: float) -> str:
    """Format a duration in seconds as s/mm:ss/hh:mm:ss."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


def _estimate_mp4_times(
    time_range: tuple[float, float],
    frame_step_ms: float,
    fps: int,
) -> tuple[int, float, tuple[float, float]]:
    """Estimate frame count, video duration, and rough wall-time range for export."""
    t0, t1 = time_range
    dt = max(1e-9, float(frame_step_ms))
    frame_count = max(1, int(np.floor(max(0.0, t1 - t0) / dt)) + 1)
    video_seconds = frame_count / max(1, int(fps))
    wall_time_fast = frame_count / 15.0
    wall_time_slow = frame_count / 6.0
    return frame_count, video_seconds, (wall_time_fast, wall_time_slow)


def _start_mp4_progress(
    total_videos: int,
    frame_step_ms: float,
    fps: int,
    sample_time_range: tuple[float, float] | None = None,
):
    """Create MP4 tqdm and print a start message when only one video is exported."""
    from tqdm import tqdm

    pbar = tqdm(total=total_videos, desc="MP4 export", unit="video")
    if total_videos == 1:
        if sample_time_range is not None:
            n_frames, video_s, (wall_fast, wall_slow) = _estimate_mp4_times(
                sample_time_range, frame_step_ms=frame_step_ms, fps=fps,
            )
            pbar.set_postfix_str(
                f"1 video | ~{n_frames} frames | vid { _format_duration_human(video_s) } | "
                f"est { _format_duration_human(wall_fast) }–{ _format_duration_human(wall_slow) }"
            )
        else:
            pbar.set_postfix_str("1 video")
    return pbar


def _network_label(rp: RingParams) -> str:
    """Build a directory-safe label encoding n_nodes, connectivity weights, and Gaussian sigma.

    Example: 128_inhib_10_excit_7_sigma_30
    """
    return (
        f"{rp.n_nodes}_inhib_{_fmt(rp.w_pv_global)}"
        f"_excit_{_fmt(rp.w_pyr_pyr_inter)}"
        f"_sigma_{_fmt(rp.sigma_pyr_deg)}"
    )


def _calibration_network_label(rp: RingParams) -> str:
    """Label for calibration directories: inhibition + Gaussian sigma only.

    Excitation is excluded because calibration is the process that determines it.
    Example: 128_inhib_10_sigma_30
    """
    return f"{rp.n_nodes}_inhib_{_fmt(rp.w_pv_global)}_sigma_{_fmt(rp.sigma_pyr_deg)}"


def _balance_cue_location(target_deg: float, rp: RingParams) -> float:
    """Place cue at a location that balances left/right node counts when possible.

    For even node counts, use half-step locations (between two nodes).
    For odd node counts, snap to nearest node (already balanced by design).
    """
    n = int(rp.n_nodes)
    step = 360.0 / max(1, n)
    if n % 2 == 0:
        k = int(np.round((target_deg - 0.5 * step) / step))
        return (k * step + 0.5 * step) % 360.0
    k = int(np.round(target_deg / step))
    return (k * step) % 360.0


def _run_type_label(args) -> str:
    """Return a folder name encoding the experiment type for ring-run outputs.

    Possible values: cue, cue_transient, cue_distractor, cue_distractor_transient
    """
    has_dist = _has_distractor(args)
    has_trans = getattr(args, 'response_onset_ms', 0.0) > 0
    if has_dist and has_trans:
        return "cue_distractor_transient"
    if has_dist:
        return "cue_distractor"
    if has_trans:
        return "cue_transient"
    return "cue"


def _stim_label(amp_factor: float) -> str:
    """Short label for stimulus amplitude factor, used in plot titles."""
    return f"amp={_fmt(amp_factor)}×"


def _weights_label(rp: RingParams) -> str:
    """Short label for PYR and PV weights and sigma, used in plot titles."""
    return f"w_pyr={_fmt(rp.w_pyr_pyr_inter)}, w_pv={_fmt(rp.w_pv_global)}, σ={_fmt(rp.sigma_pyr_deg)}°"


def _parse_seed(value: str) -> int | None:
    """Parse --seed argument: integer or 'rdm' for a truly random seed."""
    if value == "rdm":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"seed must be an integer or 'rdm', got {value!r}"
        )


def _resolve_seed(args: argparse.Namespace) -> None:
    """If --seed rdm was given, generate a random seed, store it, and print it."""
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"Using random seed: {args.seed}")


def _snapshot_animation_quality_kwargs(args: argparse.Namespace) -> dict[str, int]:
    """Return animation quality settings from CLI flags."""
    if getattr(args, "quality_high", False):
        return {"dpi": 130, "av1_crf": 31, "av1_preset": 7}
    return {"dpi": 100, "av1_crf": 35, "av1_preset": 8}


def _resolve_per_cond_param(
    values: list[float],
    condition_keys: list[str],
    param_name: str,
) -> dict[str, float]:
    """Map per-condition parameter values.

    If a single value is provided it is broadcast to all conditions.
    If N values are provided they must match the N conditions in order.
    """
    if len(values) == 1:
        return {ck: values[0] for ck in condition_keys}
    if len(values) != len(condition_keys):
        print(
            f"Error: --{param_name} has {len(values)} values but "
            f"{len(condition_keys)} conditions ({', '.join(condition_keys)})."
        )
        sys.exit(1)
    return dict(zip(condition_keys, values))


def _build_cond_labels(
    condition_keys: list[str],
    cond_excit: dict[str, float],
    cond_amp: dict[str, float] | None = None,
) -> dict[str, str]:
    """Build legend labels annotated with per-condition parameters.

    When all conditions share the same excitation weight (and amplitude),
    no annotation is added and the result is identical to the default
    ``STUDY_CONDITIONS[ck].name`` labels.  When values differ, each label
    gets a short suffix, e.g. ``"WT APP (e=7.5)"``.
    """
    all_same_excit = len(set(cond_excit.values())) == 1
    all_same_amp = cond_amp is None or len(set(cond_amp.values())) == 1
    labels: dict[str, str] = {}
    for ck in condition_keys:
        base = STUDY_CONDITIONS[ck].name if ck in STUDY_CONDITIONS else ck
        extras: list[str] = []
        if not all_same_excit:
            extras.append(f"e={_fmt(cond_excit[ck])}")
        if not all_same_amp and cond_amp is not None:
            extras.append(f"a={_fmt(cond_amp[ck])}")
        labels[ck] = f"{base} ({', '.join(extras)})" if extras else base
    return labels


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by ring-run and ring-study."""
    parser.add_argument(
        "--params_json", type=str, default="",
        help="Load base parameters from JSON file",
    )
    parser.add_argument(
        "--ring_params_json", type=str, default="",
        help="Load ring parameters (w_pyr_pyr_inter, w_pv_global, sigma_pyr_deg, n_nodes) from JSON file "
             "(same format as --save_best_ring_json output from ring-optimize)",
    )
    parser.add_argument(
        "--seed", type=_parse_seed, default=442,
        help="Random seed (int) or 'rdm' for random seed",
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Do not display figures interactively",
    )

    parser.add_argument(
        "--n_nodes", type=int, default=None,
        help="Number of ring nodes (default: from ring params JSON or 128)",
    )
    parser.add_argument(
        "--w_pyr_pyr_inter", nargs="+", type=float, default=None,
        help="Inter-node PYR->PYR weight. "
             "Accepts one value (shared) or one per condition for per-condition excitation. "
             "Default: from ring params JSON or 8.0.",
    )
    parser.add_argument(
        "--sigma_pyr_deg", type=float, default=None,
        help="PYR ring connectivity width in degrees. Default: from ring params JSON or 30.0.",
    )
    parser.add_argument(
        "--w_pv_global", type=float, default=None,
        help="Global PV->PYR inhibition weight. Default: from ring params JSON or 10.0.",
    )

    parser.add_argument(
        "--amplitude", nargs="+", type=float, default=[10.0],
        help="Cue amplitude factor (multiplier of I_ext_pyr). "
             "Accepts one value (shared) or one per condition in ring-study.",
    )
    parser.add_argument(
        "--delay_ms", type=float, default=5000.0,
        help="Delay duration after cue offset in ms (default: 5000). "
             "When a distractor is enabled this becomes delay1 (cue offset → distractor onset).",
    )
    parser.add_argument(
        "--distractor_offset_deg", type=float, default=None,
        help="Distractor angular offset from cue in degrees. "
             "If set together with --distractor_factor, enables the distractor.",
    )
    parser.add_argument(
        "--distractor_factor", type=float, default=None,
        help="Distractor amplitude as a fraction of cue amplitude (e.g. 0.75). "
             "If set together with --distractor_offset_deg, enables the distractor.",
    )
    parser.add_argument(
        "--total_time_ms", type=float, default=None,
        help="Total simulation time override (must be >= cue+delay end)",
    )
    parser.add_argument(
        "--record_dt_ms", type=float, default=5.0,
        help="Recorded sampling step in ms (default: 5)",
    )

    parser.add_argument(
        "--response_onset_ms", type=float, default=0.0,
        help="Start a global response transient this many ms after delay end (0 disables)",
    )
    parser.add_argument(
        "--response_duration_ms", type=float, default=500.0,
        help="Response transient duration in ms (default: 500)",
    )
    parser.add_argument(
        "--response_factor", type=float, default=0.5,
        help="Response transient strength as fraction of I0 (default: 0.5)",
    )

    parser.add_argument(
        "--sigma_noise", type=float, default=None,
        help="Relative noise amplitude: std of current noise injected into PYR = "
             "sigma_noise × I_ext_pyr. Default: use value from params (typically 0.1). "
             "Set to 0 to disable noise.",
    )

    parser.add_argument(
        "--snapshot_anim_step_ms", type=float, default=2.0,
        help="Frame spacing for snapshot MP4 export in ms (default: 2)",
    )
    parser.add_argument(
        "--snapshot_anim_fps", type=int, default=30,
        help="FPS for snapshot MP4 export (default: 30)",
    )
    parser.add_argument(
        "--quality_high", action="store_true",
        help="Use higher-quality (slower) MP4 encoding settings",
    )
    parser.add_argument(
        "--no_snapshot_mp4", action="store_true",
        help="Skip snapshot MP4 exports",
    )


# ============================================================================
# STUDY: BURN-IN CACHE
# ============================================================================

def _compute_burnin_state(
    local_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    noise_type: str = "white",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a burn-in simulation and return the final state.

    Uses record_dt_ms=BURN_IN_MS so only the final snapshot is stored,
    reducing memory usage.

    Parameters:
        noise_type: "white" (default) to include noise and match ring-run
                    dynamics. Pass "none" only for explicitly noiseless
                    experiments (e.g. temporal_dissection).
        seed: RNG seed for the burn-in noise.
    """
    result = simulate_ring(
        local_params, ring_params, T_ms=BURN_IN_MS,
        stimuli=None, r0=None, I_adapt0=None,
        seed=seed, noise_type=noise_type,
        connectivity=connectivity,
        record_dt_ms=BURN_IN_MS,
    )
    r0 = result.r[-1].copy()
    I_adapt0 = result.I_adapt_final.copy()
    del result
    return r0, I_adapt0


# ============================================================================
# STUDY: CSV CACHING
# ============================================================================

_CSV_FIELDS = [
    'condition_key', 'amplitude', 'trial_idx', 'seed', 'eval_time_ms',
    'center_mean_deg', 'center_std_deg', 'amplitude_mean',
    'width_mean_deg', 'drift_rate_deg_per_s', 'diffusion_deg2_per_s',
    'error_from_cue_deg',
    'mean_rate_pyr_hz', 'mean_rate_som_hz', 'mean_rate_pv_hz', 'mean_rate_vip_hz',
    'cue_rate_pyr_hz', 'cue_rate_som_hz', 'cue_rate_pv_hz', 'cue_rate_vip_hz',
]

_METRIC_KEYS = [
    'center_mean_deg', 'center_std_deg', 'amplitude_mean',
    'width_mean_deg', 'drift_rate_deg_per_s', 'diffusion_deg2_per_s',
    'error_from_cue_deg',
    'mean_rate_pyr_hz', 'mean_rate_som_hz', 'mean_rate_pv_hz', 'mean_rate_vip_hz',
    'cue_rate_pyr_hz', 'cue_rate_som_hz', 'cue_rate_pv_hz', 'cue_rate_vip_hz',
]

_RATE_POPS = [
    ('mean_rate_pyr_hz', 'PYR', 'Hz'),
    ('mean_rate_som_hz', 'SOM', 'Hz'),
    ('mean_rate_pv_hz', 'PV', 'Hz'),
    ('mean_rate_vip_hz', 'VIP', 'Hz'),
]

_CUE_RATE_POPS = [
    ('cue_rate_pyr_hz', 'PYR', 'Hz'),
    ('cue_rate_som_hz', 'SOM', 'Hz'),
    ('cue_rate_pv_hz', 'PV', 'Hz'),
    ('cue_rate_vip_hz', 'VIP', 'Hz'),
]


def _load_cached_metrics(
    csv_path: str,
    expected_eval_times: list[float] | None = None,
) -> set[tuple[str, float, int]]:
    """Load CSV and return set of (cond_key, amplitude, trial_idx) already computed."""
    if not os.path.exists(csv_path):
        return set()

    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))

    from collections import defaultdict
    job_eval_times: dict[tuple, set[float]] = defaultdict(set)
    for row in rows:
        key = (row['condition_key'], float(row['amplitude']), int(row['trial_idx']))
        if row['eval_time_ms'] != 'full_delay':
            job_eval_times[key].add(float(row['eval_time_ms']))

    expected_set = set(expected_eval_times) if expected_eval_times else None

    completed: set[tuple[str, float, int]] = set()
    stale_keys: set[tuple] = set()

    for key, cached_times in job_eval_times.items():
        if expected_set is not None and cached_times != expected_set:
            stale_keys.add(key)
        else:
            completed.add(key)

    if stale_keys:
        kept = [
            row for row in rows
            if (row['condition_key'], float(row['amplitude']), int(row['trial_idx']))
            not in stale_keys
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(kept)
        n_removed = len(stale_keys)
        print(f"  Cache: {n_removed} job(s) had stale eval times -- will re-run")

    return completed


def _append_metrics_to_csv(csv_path: str, rows: list[dict]):
    """Append metric rows to CSV, creating header if file is new."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _job_result_to_csv_rows(res: dict) -> list[dict]:
    """Convert a worker result dict to CSV row dicts."""
    rows = []
    base = {
        'condition_key': res['cond_key'],
        'amplitude': res['amplitude'],
        'trial_idx': res['trial_idx'],
        'seed': res['seed'],
    }
    for m in res['delay_metrics']:
        row = {**base, 'eval_time_ms': m['eval_time_ms']}
        for k in _METRIC_KEYS:
            row[k] = m.get(k, np.nan)
        rows.append(row)
    m = res['full_delay_metrics']
    row = {**base, 'eval_time_ms': 'full_delay'}
    for k in _METRIC_KEYS:
        row[k] = m[k]
    rows.append(row)
    return rows


def _load_all_metrics(csv_path: str) -> list[dict]:
    """Load the entire CSV as a list of row dicts."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r') as f:
        return list(csv.DictReader(f))


# ============================================================================
# STUDY: PARALLEL WORKER
# ============================================================================

_ring_sim_args: Optional[dict] = None


def _ring_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    per_cond_rp: dict[str, RingParams],
    per_cond_conn: dict[str, RingConnectivity],
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    delay_eval_times: list[float],
    T_ms_full: float,
):
    """Initialize worker process with shared parameters."""
    global _ring_sim_args
    _ring_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'per_cond_conn': per_cond_conn,
        'burnin_states': burnin_states,
        'delay_eval_times': delay_eval_times,
        'T_ms_full': T_ms_full,
    }


def _ring_run_single(job: tuple) -> dict:
    """Run a single simulation job. Called by ProcessPoolExecutor."""
    global _ring_sim_args
    cfg = _ring_sim_args
    cond_key, amplitude, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['per_cond_rp'][cond_key]
    connectivity = cfg['per_cond_conn'][cond_key]
    T_ms_full = cfg['T_ms_full']

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args_d['delay_ms']
    response_onset_ms = args_d.get('response_onset_ms', 0.0)
    if response_onset_ms > 0:
        local_params = replace(
            local_params,
            trans_enabled=True,
            trans_start_ms=delay_end_ms + response_onset_ms,
            trans_duration_ms=args_d.get('response_duration_ms', 500.0),
            trans_factor=args_d.get('response_factor', 0.5),
        )

    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    # amplitude is a factor of I_ext_pyr — convert to actual current
    actual_current = amplitude * base_params.I_ext_pyr()

    T_ms_short = T_ms_full - BURN_IN_MS
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    if local_params.trans_enabled:
        local_params = replace(
            local_params,
            trans_start_ms=local_params.trans_start_ms - BURN_IN_MS,
        )

    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
        seed=seed, connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    result.t_ms += BURN_IN_MS
    result.stim_window = (STIM_ONSET_MS, STIM_ONSET_MS + STIM_DURATION_MS)
    if result.local_params.trans_enabled:
        result.local_params = replace(
            result.local_params,
            trans_start_ms=result.local_params.trans_start_ms + BURN_IN_MS,
        )

    delay_metrics = compute_metrics_at_delay_times(
        result, cfg['delay_eval_times'], window_ms=200.0,
    )
    full_delay_metrics = compute_bump_metrics(result)

    # Mean firing rate per population during delay period
    _t_start_rate = result.stim_window[1] + 100.0
    _rate_mask = (result.t_ms >= _t_start_rate) & (result.t_ms <= result.t_ms[-1])
    _cue_idx = int(np.argmin(np.abs(result.ring_params.node_angles_deg - STIM_CENTER_DEG)))
    if np.any(_rate_mask):
        _pop_means = result.r[_rate_mask, :, :].mean(axis=(0, 1))  # shape (4,)
        _cue_means = result.r[_rate_mask, _cue_idx, :].mean(axis=0)  # shape (4,)
        for _pi, _pn in enumerate(('pyr', 'som', 'pv', 'vip')):
            full_delay_metrics[f'mean_rate_{_pn}_hz'] = float(_pop_means[_pi])
            full_delay_metrics[f'cue_rate_{_pn}_hz'] = float(_cue_means[_pi])
    else:
        for _pn in ('pyr', 'som', 'pv', 'vip'):
            full_delay_metrics[f'mean_rate_{_pn}_hz'] = np.nan
            full_delay_metrics[f'cue_rate_{_pn}_hz'] = np.nan

    comparison_data = None
    if trial_idx == 0:
        time_range = (BURN_IN_MS, result.t_ms[-1])
        comparison_data = extract_comparison_data(
            result, population=0, time_range=time_range, t_offset=BURN_IN_MS,
        )

    del result

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'trial_idx': trial_idx,
        'seed': seed,
        'delay_metrics': delay_metrics,
        'full_delay_metrics': full_delay_metrics,
        'comparison_data': comparison_data,
    }


# ============================================================================
# STUDY: HELPERS
# ============================================================================

def _generate_trial_seeds(base_seed: int, n_trials: int) -> list[int]:
    """Generate deterministic per-trial seeds from a base seed."""
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(n_trials)]


def _generate_trial_seeds_range(base_seed: int, start_idx: int, count: int) -> list[int]:
    """Generate deterministic seeds for trial indices [start_idx, start_idx+count)."""
    if count <= 0:
        return []
    seeds = _generate_trial_seeds(base_seed, start_idx + count)
    return seeds[start_idx:start_idx + count]


def _compute_delay_eval_times(
    args, stim_offset_ms: float, T_ms: float,
) -> tuple[list[float], list[str]]:
    """Compute delay evaluation times and labels."""
    delay_step = getattr(args, 'delay_step_ms', None)
    if delay_step is None or delay_step <= 0:
        delay_step = 200.0  # default: every 200 ms

    offsets = []
    t = delay_step
    while t <= args.delay_ms:
        offsets.append(t)
        t += delay_step

    delay_eval_times = [stim_offset_ms + dt for dt in offsets
                        if stim_offset_ms + dt <= T_ms]
    delay_labels = [f"{dt/1000:.1f}s" for dt in offsets
                    if stim_offset_ms + dt <= T_ms]
    return delay_eval_times, delay_labels


def _args_to_dict(args: argparse.Namespace) -> dict:
    """Convert argparse Namespace to a plain dict for pickling."""
    return {
        'delay_ms': args.delay_ms,
        'response_onset_ms': getattr(args, 'response_onset_ms', 0.0),
        'response_duration_ms': getattr(args, 'response_duration_ms', 500.0),
        'response_factor': getattr(args, 'response_factor', 0.5),
        'record_dt_ms': getattr(args, 'record_dt_ms', 5.0),
    }


# ============================================================================
# BUMP DECAY STUDY: CACHE HELPER
# ============================================================================

def _bump_decay_cache_key(
    args,
    base_params: "CircuitParams",
    ring_params: "RingParams",
    condition_keys: list,
    amplitudes: list,
    w_inter_values: list,
) -> str:
    """Return a 16-char hex key uniquely identifying one set of bump-decay simulation inputs."""
    import dataclasses
    import hashlib
    import json

    def _to_json(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return str(obj)

    params = {
        'base_params':    _to_json(base_params),
        'ring_params':    _to_json(ring_params),
        'condition_keys': sorted(condition_keys),
        'amplitudes':     sorted(amplitudes),
        'w_inter_values': sorted(w_inter_values),
        'n_trials':       int(args.n_trials),
        'seed':           int(args.seed),
        'delay_ms':       float(args.delay_ms),
        'ref_offset_ms':  float(getattr(args, 'ref_offset_ms', BUMP_DECAY_REF_OFFSET_MS)),
        'window_ms':      float(getattr(args, 'window_ms', 500.0)),
        'record_dt_ms':   float(getattr(args, 'record_dt_ms', 5.0)),
    }
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ============================================================================
# OSCILLATION STUDY: CACHE HELPERS
# ============================================================================

def _osc_cache_key(
    args,
    base_params: "CircuitParams",
    ring_params: "RingParams",
    condition_keys: list,
    amplitudes: list,
) -> str:
    """Return a 16-char hex key uniquely identifying one set of simulation inputs."""
    import dataclasses
    import hashlib
    import json

    def _to_json(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return str(obj)

    params = {
        'base_params':       _to_json(base_params),
        'ring_params':       _to_json(ring_params),
        'condition_keys':    sorted(condition_keys),
        'amplitudes':        sorted(amplitudes),
        'n_trials':          int(args.n_trials),
        'seed':              int(args.seed),
        'delay_ms':          float(args.delay_ms),
        'osc_skip_ms':       float(args.osc_skip_ms),
        'min_freq_hz':       float(args.min_freq_hz),
        'max_freq_hz':       float(args.max_freq_hz),
        'tf_window_s':       float(args.tf_window_s),
        'tf_overlap':        float(args.tf_overlap),
        'sample_time_frac':  float(args.sample_time_frac),
        'response_onset_ms':    float(getattr(args, 'response_onset_ms', 0.0)),
        'response_duration_ms': float(getattr(args, 'response_duration_ms', 500.0)),
        'response_factor':      float(getattr(args, 'response_factor', 0.5)),
        'record_dt_ms':         float(getattr(args, 'record_dt_ms', 5.0)),
    }
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ============================================================================
# BUMP DECAY STUDY: PARALLEL WORKER
# ============================================================================

_bump_decay_sim_args: Optional[dict] = None


def _bump_decay_init_worker(
    base_params: "CircuitParams",
    per_cond_rp: dict,
    connectivity_map: dict,
    burnin_states: dict,
    delay_ms: float,
    ref_offset_ms: float,
    window_ms: float,
    record_dt_ms: float,
    T_ms_full: float,
) -> None:
    """Initialise worker process for bump-decay-study jobs."""
    global _bump_decay_sim_args
    _bump_decay_sim_args = {
        'base_params':      base_params,
        'per_cond_rp':      per_cond_rp,
        'connectivity_map': connectivity_map,
        'burnin_states':    burnin_states,
        'delay_ms':         delay_ms,
        'ref_offset_ms':    ref_offset_ms,
        'window_ms':        window_ms,
        'record_dt_ms':     record_dt_ms,
        'T_ms_full':        T_ms_full,
    }


def _bump_decay_run_single(job: tuple) -> dict:
    """Run one bump-decay trial; return amplitude timecourse relative to cue onset.

    job = (cond_key, amp_factor, w_inter, trial_idx, seed)
    """
    from .analysis import decode_bump_center

    global _bump_decay_sim_args
    cfg = _bump_decay_sim_args

    cond_key, amp_factor, w_inter, trial_idx, seed = job

    base_params   = cfg['base_params']
    ring_params   = cfg['per_cond_rp'][cond_key]
    T_ms_full     = cfg['T_ms_full']
    ref_offset_ms = cfg['ref_offset_ms']
    window_ms     = cfg['window_ms']

    # Look up precomputed connectivity for this w_inter
    connectivity = cfg['connectivity_map'][w_inter]

    # Build matching RingParams if w_inter differs from this condition's base
    if w_inter != ring_params.w_pyr_pyr_inter:
        ring_params = RingParams(
            n_nodes=ring_params.n_nodes,
            w_pyr_pyr_inter=w_inter,
            sigma_pyr_deg=ring_params.sigma_pyr_deg,
            w_pv_global=ring_params.w_pv_global,
        )

    condition    = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    r0, I_adapt0 = cfg['burnin_states'][cond_key]
    cue_current  = amp_factor * base_params.I_ext_pyr()

    T_ms_short        = T_ms_full - BURN_IN_MS
    stim_onset_short  = STIM_ONSET_MS - BURN_IN_MS  # = 500 ms inside short sim

    stimuli = [RingStimulus(
        center_deg=STIM_CENTER_DEG,
        amplitude=cue_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=stim_onset_short,
        duration_ms=STIM_DURATION_MS,
    )]

    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=T_ms_short,
        stimuli=stimuli,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=seed,
        connectivity=connectivity,
        record_dt_ms=cfg['record_dt_ms'],
    )
    result.t_ms = result.t_ms + BURN_IN_MS  # restore absolute time axis

    _, bump_amplitude = decode_bump_center(result, population=0)

    # Time relative to cue onset
    t_rel = result.t_ms - STIM_ONSET_MS  # 0 at cue onset

    # Reference: mean amplitude in the window_ms bin centred at STIM_DURATION_MS + ref_offset_ms
    ref_center_rel = STIM_DURATION_MS + ref_offset_ms
    half_win = window_ms / 2.0
    ref_mask = (t_rel >= ref_center_rel - half_win) & (t_rel < ref_center_rel + half_win)

    if ref_mask.any():
        ref_amplitude = float(np.mean(bump_amplitude[ref_mask]))
    else:
        # Fallback: use single nearest timestep
        idx = int(np.argmin(np.abs(t_rel - ref_center_rel)))
        ref_amplitude = float(bump_amplitude[idx])

    # Keep only from cue onset onward (t_rel >= 0)
    delay_mask = t_rel >= 0.0
    t_ms_out  = t_rel[delay_mask].tolist()
    amp_out   = bump_amplitude[delay_mask].tolist()

    del result

    return {
        'cond_key':             cond_key,
        'amplitude':            float(amp_factor),
        'w_inter':              float(w_inter),
        'trial_idx':            int(trial_idx),
        'seed':                 int(seed),
        't_ms':                 t_ms_out,
        'amplitude_timecourse': amp_out,
        'ref_amplitude':        ref_amplitude,
    }


# ============================================================================
# OSCILLATION STUDY: PARALLEL WORKER
# ============================================================================

_osc_sim_args: Optional[dict] = None


def _osc_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    per_cond_rp: dict[str, RingParams],
    per_cond_conn: dict[str, RingConnectivity],
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    T_ms_full: float,
):
    """Initialize worker process for oscillation-study jobs."""
    global _osc_sim_args
    _osc_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'per_cond_conn': per_cond_conn,
        'burnin_states': burnin_states,
        'T_ms_full': T_ms_full,
    }


def _osc_run_single(job: tuple) -> dict:
    """Run one cue-only trial and extract oscillation metrics."""
    global _osc_sim_args
    cfg = _osc_sim_args
    cond_key, amplitude, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['per_cond_rp'][cond_key]
    connectivity = cfg['per_cond_conn'][cond_key]
    T_ms_full = cfg['T_ms_full']

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args_d['delay_ms']
    response_onset_ms = args_d.get('response_onset_ms', 0.0)
    if response_onset_ms > 0:
        local_params = replace(
            local_params,
            trans_enabled=True,
            trans_start_ms=delay_end_ms + response_onset_ms,
            trans_duration_ms=args_d.get('response_duration_ms', 500.0),
            trans_factor=args_d.get('response_factor', 0.5),
        )

    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    cue_current = amplitude * base_params.I_ext_pyr()
    T_ms_short = T_ms_full - BURN_IN_MS
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=cue_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    if local_params.trans_enabled:
        local_params = replace(
            local_params,
            trans_start_ms=local_params.trans_start_ms - BURN_IN_MS,
        )

    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=T_ms_short,
        stimuli=stimuli_short,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=seed,
        connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    result.t_ms += BURN_IN_MS

    center_rad, amp_t = population_vector_decode(result.r[:, :, 0], ring_params.node_angles_rad)
    del center_rad

    delay_start = stim_offset_ms + args_d.get('osc_skip_ms', 200.0)
    delay_stop = stim_offset_ms + args_d['delay_ms']
    mask = (result.t_ms >= delay_start) & (result.t_ms <= delay_stop)
    t_delay_s = (result.t_ms[mask] - delay_start) / 1000.0
    amp_delay = amp_t[mask]
    cue_idx = int(np.argmin(np.abs(np.rad2deg(ring_params.node_angles_rad) - STIM_CENTER_DEG)))
    cue_rate_delay_hz = result.r[mask, cue_idx, 0]

    try:
        osc = compute_oscillation_band_timecourse(
            amp_delay,
            t_delay_s,
            min_freq_hz=args_d.get('min_freq_hz', 2.0),
            max_freq_hz=args_d.get('max_freq_hz', 12.0),
            window_s=args_d.get('tf_window_s', 1.0),
            overlap_frac=args_d.get('tf_overlap', 0.8),
        )
    except ValueError:
        osc = {
            'freqs_hz': np.array([], dtype=float),
            'times_s': np.array([], dtype=float),
            'power': np.zeros((0, 0), dtype=float),
            'dominant_freq_hz': np.array([], dtype=float),
            'dominant_power': np.array([], dtype=float),
        }

    sample_time_s = None
    sample_frac = args_d.get('sample_time_frac', 0.75)
    if len(osc['times_s']) > 0:
        t0 = float(osc['times_s'][0])
        t1 = float(osc['times_s'][-1])
        sample_time_s = t0 + float(np.clip(sample_frac, 0.0, 1.0)) * (t1 - t0)

    summary = summarize_oscillation_timecourse(
        osc['dominant_freq_hz'],
        osc['dominant_power'],
        osc['times_s'],
        sample_time_s=sample_time_s,
    )

    mean_cue_rate_hz = float(np.mean(cue_rate_delay_hz)) if len(cue_rate_delay_hz) > 0 else np.nan

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'trial_idx': trial_idx,
        'seed': seed,
        'summary': summary,
        'mean_cue_rate_hz': mean_cue_rate_hz,
        'times_s': osc['times_s'],
        'freqs_hz': osc['freqs_hz'],
        'power': osc['power'],
        'dominant_freq_hz': osc['dominant_freq_hz'],
        'dominant_power': osc['dominant_power'],
    }


# ============================================================================
# BUMP DECAY STUDY: MAIN COMMAND
# ============================================================================

def cmd_bump_decay_study(args: argparse.Namespace) -> None:
    """Bump decay / attractor stability study across conditions, amplitudes, and w_inter."""
    _resolve_seed(args)
    from collections import defaultdict
    import csv as _csv
    import pickle as _pickle
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    condition_keys = list(args.conditions) if args.conditions else ['WT', 'WT_APP']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'.\n"
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for config display

    amplitudes     = list(args.amplitudes) if args.amplitudes else [5.0, 10.0, 15.0, 20.0, 25.0]
    w_inter_values = list(args.w_inter_values) if args.w_inter_values else [args.w_pyr_pyr_inter[0]]
    n_trials       = int(args.n_trials)
    n_workers      = _resolve_workers(args)
    delay_ms       = float(args.delay_ms)
    ref_offset_ms  = float(getattr(args, 'ref_offset_ms', BUMP_DECAY_REF_OFFSET_MS))
    window_ms      = float(getattr(args, 'window_ms', 500.0))
    record_dt_ms   = float(getattr(args, 'record_dt_ms', 5.0))

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    T_ms_full      = stim_offset_ms + delay_ms

    # Build time bins (relative to cue onset, step = window_ms)
    max_t_rel    = STIM_DURATION_MS + delay_ms
    bin_edges    = np.arange(0.0, max_t_rel + window_ms, window_ms)
    bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins       = len(bin_centers)

    # Reference bin: the bin whose center is closest to STIM_DURATION_MS + ref_offset_ms
    ref_center_rel = STIM_DURATION_MS + ref_offset_ms  # e.g. 650 ms
    ref_bin_idx    = int(np.argmin(np.abs(bin_centers - ref_center_rel)))

    conn_label = _calibration_network_label(base_rp)
    out_dir = os.path.join(
        _output_dir("figs/ring/bump_decay", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    cond_labels = _build_cond_labels(condition_keys, cond_excit)

    _print_config(args, amplitudes[0], base_params, T_ms_full, ring_params,
                  experiment_info=[
                      f"Conditions:    {', '.join(condition_keys)}",
                      f"Amplitudes:    {', '.join(_fmt(a) for a in amplitudes)}× I_ext_pyr",
                      f"w_inter sweep: {', '.join(_fmt(w) for w in w_inter_values)}",
                      f"Delay:         {delay_ms:.0f} ms",
                      f"Trials:        {n_trials}   seed={args.seed}   workers={n_workers}",
                      f"Window:        {window_ms:.0f} ms   ref bin center={bin_centers[ref_bin_idx]:.0f} ms",
                  ],
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    # ── Burn-in states (one per condition, using per-condition ring_params) ───
    print("\nComputing burn-in states...")
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond_key in tqdm(condition_keys, desc="Burn-in", unit="cond"):
        local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
        burnin_states[cond_key] = _compute_burnin_state(
            local_params, per_cond_rp[cond_key], per_cond_conn[cond_key], seed=args.seed,
        )

    # ── Precompute connectivity map {w_inter: RingConnectivity} ─────────────
    connectivity_map: dict[float, RingConnectivity] = {}
    for w in w_inter_values:
        rp_w = RingParams(
            n_nodes=base_rp.n_nodes,
            w_pyr_pyr_inter=w,
            sigma_pyr_deg=base_rp.sigma_pyr_deg,
            w_pv_global=base_rp.w_pv_global,
        )
        connectivity_map[w] = RingConnectivity.from_params(rp_w)

    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    jobs = [
        (ck, amp, w, ti, trial_seeds[ti])
        for ck in condition_keys
        for amp in amplitudes
        for w in w_inter_values
        for ti in range(n_trials)
    ]

    # ── Cache lookup ──────────────────────────────────────────────────────────
    use_cache  = not getattr(args, 'no_cache', False)
    cache_key  = _bump_decay_cache_key(args, base_params, base_rp,
                                        condition_keys, amplitudes, w_inter_values)
    cache_file = os.path.join(out_dir, f'.bump_decay_cache_{cache_key}.pkl')

    all_results: list[dict] = []
    if use_cache and os.path.exists(cache_file):
        print(f"\nLoading cached results (key={cache_key})...")
        with open(cache_file, 'rb') as _cf:
            all_results = _pickle.load(_cf)
        print(f"  Loaded {len(all_results)} trials from cache.")
    else:
        init_args = (
            base_params, per_cond_rp, connectivity_map,
            burnin_states, delay_ms, ref_offset_ms,
            window_ms, record_dt_ms, T_ms_full,
        )
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                mp_context=_MP_CONTEXT,
                max_workers=n_workers,
                initializer=_bump_decay_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_bump_decay_run_single, job): job
                           for job in jobs}
                with tqdm(total=len(jobs), desc="Simulations",
                          unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _bump_decay_init_worker(*init_args)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                all_results.append(_bump_decay_run_single(job))

        with open(cache_file, 'wb') as _cf:
            _pickle.dump(all_results, _cf, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"\nSimulation results cached → {cache_file}")

    # ── Aggregate: bin into windows, normalize, collect distributions ────────
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r['cond_key'], r['amplitude'], r['w_inter'])].append(r)

    aggregated: dict[tuple, dict] = {}
    for key, trials in groups.items():
        per_trial_bins = []
        skipped        = 0

        for tr in trials:
            t_arr = np.array(tr['t_ms'])
            a_arr = np.array(tr['amplitude_timecourse'])

            # Bin the raw timecourse into window_ms bins
            binned = np.full(n_bins, np.nan)
            for bi in range(n_bins):
                mask = (t_arr >= bin_edges[bi]) & (t_arr < bin_edges[bi + 1])
                if mask.any():
                    binned[bi] = float(np.mean(a_arr[mask]))

            # Normalization: use the reference bin
            ref_amp = binned[ref_bin_idx] if np.isfinite(binned[ref_bin_idx]) else 0.0
            if ref_amp > 1e-10:
                per_trial_bins.append(binned / ref_amp)
            else:
                skipped += 1

        if skipped:
            ck, amp, w = key
            print(f"  [{ck} amp={amp:g} w={w:g}] {skipped}/{len(trials)} trials"
                  f" had ref_amplitude≈0 and were excluded.")
        if not per_trial_bins:
            continue

        bins_arr  = np.array(per_trial_bins)   # (n_valid, n_bins)
        mean_bins = np.nanmean(bins_arr, axis=0)
        n_valid   = bins_arr.shape[0]
        sem_bins  = (np.nanstd(bins_arr, axis=0, ddof=1) / np.sqrt(n_valid)
                     if n_valid > 1 else np.zeros(n_bins))

        # Per-bin trial distributions for boxplots
        per_bin_vals = [bins_arr[:, bi] for bi in range(n_bins)]

        # Scalar: mean of last bin
        end_val = float(np.nanmean(bins_arr[:, -1]))

        aggregated[key] = {
            'bin_centers':   bin_centers,
            'mean_bins':     mean_bins,
            'sem_bins':      sem_bins,
            'per_bin_vals':  per_bin_vals,
            'bins_arr':      bins_arr,       # (n_trials, n_bins) for amp sweep
            'end_val':       end_val,
            'n_trials':      n_valid,
        }

    # ── Save trial summary CSV ───────────────────────────────────────────────
    summary_csv = os.path.join(out_dir, "bump_decay_trials.csv")
    with open(summary_csv, 'w', newline='') as _f:
        writer = _csv.DictWriter(_f, fieldnames=[
            'condition', 'amplitude', 'w_inter', 'trial_idx', 'seed',
            'ref_amplitude', 'end_val_normalized',
        ])
        writer.writeheader()
        for r in sorted(all_results,
                        key=lambda x: (x['cond_key'], x['amplitude'],
                                       x['w_inter'], x['trial_idx'])):
            key = (r['cond_key'], r['amplitude'], r['w_inter'])
            agg = aggregated.get(key, {})
            writer.writerow({
                'condition':          r['cond_key'],
                'amplitude':          r['amplitude'],
                'w_inter':            r['w_inter'],
                'trial_idx':          r['trial_idx'],
                'seed':               r['seed'],
                'ref_amplitude':      r['ref_amplitude'],
                'end_val_normalized': agg.get('end_val', float('nan')),
            })
    print(f"\nTrial summary → {summary_csv}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    from .plotting import (
        plot_bump_decay_timecourse,
        plot_bump_decay_boxplot,
        plot_bump_decay_heatmap,
        plot_oscillation_amp_sweep_lines,
    )

    # Helper: build amp subdirectory path (includes w_label only when >1 w_inter)
    multi_w = len(w_inter_values) > 1

    def _amp_dir(amp, w):
        sub = f"amp{amp:g}"
        if multi_w:
            sub = os.path.join(sub, f"w{w:g}")
        return os.path.join(out_dir, sub)

    conn_lbl = _weights_label(base_rp)

    # ── Per-amplitude: timecourse overlay + boxplot over time ────────────────
    for amp in amplitudes:
        for w in w_inter_values:
            lines_data = {
                ck: aggregated[(ck, amp, w)]
                for ck in condition_keys
                if (ck, amp, w) in aggregated
            }
            if not lines_data:
                continue

            amp_dir = _amp_dir(amp, w)
            os.makedirs(amp_dir, exist_ok=True)
            w_suffix = f" | w_inter={w:g}" if multi_w else ""

            # Timecourse overlay (mean ± SEM, windowed)
            tc_path = os.path.join(amp_dir, "bump_decay_timecourse.png")
            fig = plot_bump_decay_timecourse(
                lines_data=lines_data,
                condition_keys=condition_keys,
                stim_duration_ms=STIM_DURATION_MS,
                ref_bin_center=bin_centers[ref_bin_idx],
                delay_ms=delay_ms,
                title=f"Bump decay | amp={amp:g}{w_suffix}",
                save_path=tc_path,
            )
            if not args.no_show:
                plt.show()
            plt.close(fig)

            # Boxplot over time
            bx_path = os.path.join(amp_dir, "bump_decay_boxplot.png")
            fig = plot_bump_decay_boxplot(
                lines_data=lines_data,
                condition_keys=condition_keys,
                stim_duration_ms=STIM_DURATION_MS,
                ref_bin_center=bin_centers[ref_bin_idx],
                title=f"Bump decay (boxplot) | amp={amp:g}{w_suffix}",
                save_path=bx_path,
            )
            if not args.no_show:
                plt.show()
            plt.close(fig)

    # ── Amplitude sweep summary (like oscillation_amp_sweep_variance.png) ────
    # X = amplitude, Y = end-of-delay normalized A_hat, one line per condition
    sweep_data: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    for ck in condition_keys:
        for amp in amplitudes:
            vals_list = [
                aggregated[(ck, amp, w)]['bins_arr'][:, -1]
                for w in w_inter_values
                if (ck, amp, w) in aggregated
            ]
            if vals_list:
                sweep_data[ck][amp] = np.concatenate(vals_list)

    fig_sweep = plot_oscillation_amp_sweep_lines(
        panels=[(
            "Mean norm. $\\hat{A}$ (last window)",
            "Normalised bump amplitude",
            sweep_data,
        )],
        amplitudes=amplitudes,
        cond_order=condition_keys,
        cond_labels=cond_labels,
        suptitle=f"Bump decay vs cue amplitude — {conn_lbl}",
        save_path=os.path.join(out_dir, "bump_decay_amp_sweep.png"),
    )
    if not args.no_show:
        plt.show()
    plt.close(fig_sweep)

    # ── 2D heatmap per condition (only when w_inter sweep > 1) ───────────────
    if multi_w:
        for cond_key in condition_keys:
            heatmap_data = {
                (amp, w): aggregated[(cond_key, amp, w)]['end_val']
                for amp in amplitudes
                for w in w_inter_values
                if (cond_key, amp, w) in aggregated
            }
            if not heatmap_data:
                continue
            hm_dir  = os.path.join(out_dir, cond_key)
            os.makedirs(hm_dir, exist_ok=True)
            hm_path = os.path.join(hm_dir, "bump_decay_heatmap.png")
            fig = plot_bump_decay_heatmap(
                heatmap_data=heatmap_data,
                amplitudes=amplitudes,
                w_inter_values=w_inter_values,
                condition=cond_key,
                save_path=hm_path,
            )
            if not args.no_show:
                plt.show()
            plt.close(fig)

    print("\nDone.")


def cmd_oscillation_study(args: argparse.Namespace) -> None:
    """Cue-only oscillation analysis across conditions and amplitudes."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as _scipy_stats

    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    if args.conditions is None:
        condition_keys = ['WT', 'WT_APP']
    else:
        condition_keys = args.conditions
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'.\n"
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for suptitle / config display

    amplitudes = list(args.amplitudes) if args.amplitudes else [args.amplitude[0]]
    n_trials = int(args.n_trials)
    n_workers = _resolve_workers(args)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    T_ms_full = stim_offset_ms + args.delay_ms

    conn_label = _calibration_network_label(base_rp)
    out_dir = os.path.join(
        _output_dir("figs/ring/oscillation", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    _print_config(args, amplitudes[0], base_params, T_ms_full, ring_params,
                  experiment_info=[
                      f"Conditions:  {', '.join(condition_keys)}",
                      f"Amplitudes:  {', '.join(_fmt(a) for a in amplitudes)}× I_ext_pyr",
                      f"Delay:       {args.delay_ms:.0f} ms",
                      f"Trials:      {n_trials}   seed={args.seed}   workers={n_workers}",
                      f"Freq band:   [{args.min_freq_hz:.1f}, {args.max_freq_hz:.1f}] Hz"
                      f"   window={args.tf_window_s:.3f} s   overlap={args.tf_overlap:.2f}",
                  ],
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    print("\nComputing burn-in states...")
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond_key in tqdm(condition_keys, desc="Burn-in", unit="cond"):
        local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
        burnin_states[cond_key] = _compute_burnin_state(
            local_params,
            per_cond_rp[cond_key],
            per_cond_conn[cond_key],
            seed=args.seed,
        )

    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    jobs = [
        (ck, amp, ti, trial_seeds[ti])
        for ck in condition_keys
        for amp in amplitudes
        for ti in range(n_trials)
    ]

    args_dict = {
        'delay_ms': args.delay_ms,
        'response_onset_ms': getattr(args, 'response_onset_ms', 0.0),
        'response_duration_ms': getattr(args, 'response_duration_ms', 500.0),
        'response_factor': getattr(args, 'response_factor', 0.5),
        'record_dt_ms': getattr(args, 'record_dt_ms', 5.0),
        'osc_skip_ms': args.osc_skip_ms,
        'min_freq_hz': args.min_freq_hz,
        'max_freq_hz': args.max_freq_hz,
        'tf_window_s': args.tf_window_s,
        'tf_overlap': args.tf_overlap,
        'sample_time_frac': args.sample_time_frac,
    }

    # ------------------------------------------------------------------
    # Cache: load or run
    # ------------------------------------------------------------------
    import pickle as _pickle
    use_cache = not getattr(args, 'no_cache', False)
    cache_key = _osc_cache_key(args, base_params, base_rp, condition_keys, amplitudes)
    cache_file = os.path.join(out_dir, f'.osc_cache_{cache_key}.pkl')

    all_results: list[dict] = []
    if use_cache and os.path.exists(cache_file):
        print(f"\nLoading cached simulation results (key={cache_key})...")
        with open(cache_file, 'rb') as _cf:
            all_results = _pickle.load(_cf)
        print(f"  Loaded {len(all_results)} trials from cache — skipping simulations.")
        print(f"  Pass --no_cache to force re-computation.")
    else:
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT,
                max_workers=n_workers,
                initializer=_osc_init_worker,
                initargs=(args_dict, base_params, per_cond_rp, per_cond_conn, burnin_states, T_ms_full),
            ) as executor:
                futures = {executor.submit(_osc_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _osc_init_worker(args_dict, base_params, per_cond_rp, per_cond_conn, burnin_states, T_ms_full)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                all_results.append(_osc_run_single(job))

        with open(cache_file, 'wb') as _cf:
            _pickle.dump(all_results, _cf, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"\nSimulation results cached → {cache_file}")

    # ------------------------------------------------------------------
    # Save trial-level summaries
    # ------------------------------------------------------------------
    summary_csv = os.path.join(out_dir, "oscillation_trial_summary.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'condition', 'amplitude', 'trial_idx', 'seed',
                'freq_median_hz', 'power_median',
                'freq_sample_hz', 'power_sample', 'sample_time_s',
                'mean_cue_rate_hz',
            ],
        )
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x['cond_key'], x['amplitude'], x['trial_idx'])):
            s = r['summary']
            writer.writerow({
                'condition': r['cond_key'],
                'amplitude': r['amplitude'],
                'trial_idx': r['trial_idx'],
                'seed': r['seed'],
                'freq_median_hz': s['freq_median_hz'],
                'power_median': s['power_median'],
                'freq_sample_hz': s['freq_sample_hz'],
                'power_sample': s['power_sample'],
                'sample_time_s': s['sample_time_s'],
                'mean_cue_rate_hz': r['mean_cue_rate_hz'],
            })

    traj_csv = os.path.join(out_dir, "oscillation_dominant_timecourse.csv")
    with open(traj_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'condition', 'amplitude', 'trial_idx',
                'time_s', 'dominant_freq_hz', 'dominant_power',
            ],
        )
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x['cond_key'], x['amplitude'], x['trial_idx'])):
            for tt, ff, pp in zip(r['times_s'], r['dominant_freq_hz'], r['dominant_power']):
                writer.writerow({
                    'condition': r['cond_key'],
                    'amplitude': r['amplitude'],
                    'trial_idx': r['trial_idx'],
                    'time_s': float(tt),
                    'dominant_freq_hz': float(ff) if np.isfinite(ff) else '',
                    'dominant_power': float(pp) if np.isfinite(pp) else '',
                })

    # ------------------------------------------------------------------
    # Aggregate and plot
    # ------------------------------------------------------------------
    def _arr(vals: list[float]) -> np.ndarray:
        if not vals:
            return np.array([], dtype=float)
        a = np.asarray(vals, dtype=float)
        return a[np.isfinite(a)]

    stats_rows: list[dict] = []

    # Accumulate per-(cond, amp) data for cross-amplitude sweep violin
    sweep_power_median: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_power_sample: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_power_var: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_power_dvar: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_power_autocorr: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_spec_concentration: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}
    sweep_spec_entropy: dict[str, dict[float, np.ndarray]] = {ck: {} for ck in condition_keys}

    # Store per-amplitude data for deferred violin plot generation (after FDR correction)
    amp_plot_data: dict = {}

    for amp in amplitudes:
        amp_dir = os.path.join(out_dir, f"amp{_fmt(amp)}")
        os.makedirs(amp_dir, exist_ok=True)

        by_cond_median_power: dict[str, np.ndarray] = {}
        by_cond_sample_power: dict[str, np.ndarray] = {}
        by_cond_power_var: dict[str, np.ndarray] = {}
        by_cond_power_dvar: dict[str, np.ndarray] = {}
        by_cond_power_autocorr: dict[str, np.ndarray] = {}
        by_cond_spec_concentration: dict[str, np.ndarray] = {}
        by_cond_spec_entropy: dict[str, np.ndarray] = {}
        by_cond_cue_rate: dict[str, np.ndarray] = {}
        by_cond_best_freq_hz: dict[str, float] = {}
        sample_time_after_cue_vals: list[float] = []
        metrics_over_delay: dict[str, list[dict]] = {}
        delay_labels: list[str] = []

        for ck in condition_keys:
            rows = [r for r in all_results if r['cond_key'] == ck and abs(r['amplitude'] - amp) < 1e-9]

            by_cond_median_power[ck] = _arr([r['summary']['power_median'] for r in rows])
            by_cond_sample_power[ck] = _arr([r['summary']['power_sample'] for r in rows])
            by_cond_cue_rate[ck] = _arr([r['mean_cue_rate_hz'] for r in rows])

            def _trial_power_var(r: dict) -> float:
                dp = np.asarray(r['dominant_power'], dtype=float)
                return float(np.nanvar(dp)) if np.any(np.isfinite(dp)) else np.nan

            def _trial_power_dvar(r: dict) -> float:
                dp = np.asarray(r['dominant_power'], dtype=float)
                finite_mask = np.isfinite(dp)
                if finite_mask.sum() < 3:
                    return np.nan
                x = np.where(finite_mask)[0].astype(float)
                y = dp[finite_mask]
                coeffs = np.polyfit(x, y, 1)
                residuals = y - np.polyval(coeffs, x)
                return float(np.var(residuals))

            def _trial_power_autocorr(r: dict) -> float:
                dp = np.asarray(r['dominant_power'], dtype=float)
                finite = dp[np.isfinite(dp)]
                if len(finite) < 3:
                    return np.nan
                x, y = finite[:-1], finite[1:]
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    return np.nan
                return float(np.corrcoef(x, y)[0, 1])

            def _trial_spec_concentration(r: dict) -> float:
                """Mean fraction of total band power at the dominant frequency."""
                pw = np.asarray(r['power'], dtype=float)  # (n_freqs, n_times)
                dp = np.asarray(r['dominant_power'], dtype=float)
                if pw.ndim != 2 or pw.shape[1] == 0:
                    return np.nan
                total = np.sum(pw, axis=0)  # (n_times,)
                valid = (total > 0) & np.isfinite(dp) & np.isfinite(total)
                if not np.any(valid):
                    return np.nan
                return float(np.mean(dp[valid] / total[valid]))

            def _trial_spec_entropy(r: dict) -> float:
                """Mean Shannon entropy of the frequency power distribution (nats)."""
                pw = np.asarray(r['power'], dtype=float)  # (n_freqs, n_times)
                if pw.ndim != 2 or pw.shape[0] < 2 or pw.shape[1] == 0:
                    return np.nan
                total = np.sum(pw, axis=0, keepdims=True)
                total = np.where(total > 0, total, np.nan)
                p_norm = pw / total  # (n_freqs, n_times)
                p_norm = np.clip(p_norm, 1e-30, None)
                entropy_per_t = -np.sum(p_norm * np.log(p_norm), axis=0)
                finite = entropy_per_t[np.isfinite(entropy_per_t)]
                return float(np.mean(finite)) if len(finite) > 0 else np.nan

            by_cond_power_var[ck] = _arr([_trial_power_var(r) for r in rows])
            by_cond_power_dvar[ck] = _arr([_trial_power_dvar(r) for r in rows])
            by_cond_power_autocorr[ck] = _arr([_trial_power_autocorr(r) for r in rows])
            by_cond_spec_concentration[ck] = _arr([_trial_spec_concentration(r) for r in rows])
            by_cond_spec_entropy[ck] = _arr([_trial_spec_entropy(r) for r in rows])

            sweep_power_median[ck][amp] = by_cond_median_power[ck]
            sweep_power_sample[ck][amp] = by_cond_sample_power[ck]
            sweep_power_var[ck][amp] = by_cond_power_var[ck]
            sweep_power_dvar[ck][amp] = by_cond_power_dvar[ck]
            sweep_power_autocorr[ck][amp] = by_cond_power_autocorr[ck]
            sweep_spec_concentration[ck][amp] = by_cond_spec_concentration[ck]
            sweep_spec_entropy[ck][amp] = by_cond_spec_entropy[ck]
            sample_time_after_cue_vals.extend([
                float(v) + args.osc_skip_ms / 1000.0
                for v in [r['summary']['sample_time_s'] for r in rows]
                if np.isfinite(v)
            ])

            # Mean heatmap per (condition, amplitude)
            powers = [r['power'] for r in rows if r['power'].size > 0]
            if powers:
                power_mean_hm = np.mean(np.stack(powers, axis=0), axis=0)
                f_axis = rows[0]['freqs_hz']
                t_axis_hm = rows[0]['times_s']

                # Power-weighted mean frequency across the delay period.
                power_by_freq = np.mean(power_mean_hm, axis=1)
                total_pw = float(np.sum(power_by_freq))
                if total_pw > 0 and len(f_axis) > 0:
                    by_cond_best_freq_hz[ck] = float(np.sum(f_axis * power_by_freq) / total_pw)

                fig_h = plot_oscillation_band_heatmap(
                    power_mean_hm,
                    f_axis,
                    t_axis_hm,
                    title=(f"{STUDY_CONDITIONS[ck].name} | amp={_fmt(amp)}x "
                           f"[{args.min_freq_hz:g}-{args.max_freq_hz:g} Hz]"),
                    save_path=os.path.join(amp_dir, f"heatmap_{ck}.png"),
                )
                plt.close(fig_h)

            # Time-resolved metrics — pad trials with NaN rather than truncating.
            rows_t = [r for r in rows if len(r['times_s']) > 0]
            if rows_t:
                max_len = max(len(r['times_s']) for r in rows_t)
                longest = max(rows_t, key=lambda r: len(r['times_s']))
                t_axis_delay = np.asarray(longest['times_s'], dtype=float)

                p_stack = np.full((len(rows_t), max_len), np.nan)
                f_stack = np.full((len(rows_t), max_len), np.nan)
                for j, r in enumerate(rows_t):
                    n = len(r['dominant_power'])
                    p_stack[j, :n] = r['dominant_power']
                    f_stack[j, :n] = r['dominant_freq_hz']

                cond_metrics: list[dict] = []
                for ti in range(max_len):
                    pvals = p_stack[:, ti]
                    fvals = f_stack[:, ti]
                    valid_p = pvals[np.isfinite(pvals)]
                    valid_f = fvals[np.isfinite(fvals)]
                    n_p = len(valid_p)
                    n_f = len(valid_f)
                    p_mean = float(np.mean(valid_p)) if n_p > 0 else np.nan
                    p_sd   = float(np.std(valid_p, ddof=1)) if n_p > 1 else 0.0
                    p_sem  = float(p_sd / np.sqrt(n_p)) if n_p > 1 else 0.0
                    f_mean = float(np.mean(valid_f)) if n_f > 0 else np.nan
                    f_sd   = float(np.std(valid_f, ddof=1)) if n_f > 1 else 0.0
                    f_sem  = float(f_sd / np.sqrt(n_f)) if n_f > 1 else 0.0
                    cond_metrics.append({
                        'power_sample_mean': p_mean,
                        'power_sample_sd': p_sd,
                        'power_sample_sem': p_sem,
                        'freq_sample_hz_mean': f_mean,
                        'freq_sample_hz_sd': f_sd,
                        'freq_sample_hz_sem': f_sem,
                    })

                metrics_over_delay[ck] = cond_metrics
                if len(t_axis_delay) > len(delay_labels):
                    delay_labels = [f"{t:.2f}s" for t in t_axis_delay]

        pick_parts = [
            f"{STUDY_CONDITIONS[ck].name}={by_cond_best_freq_hz[ck]:.2f} Hz"
            for ck in condition_keys if ck in by_cond_best_freq_hz
        ]
        pick_lbl = ", ".join(pick_parts) if pick_parts else "NA"
        sample_time_lbl = "NA"
        if sample_time_after_cue_vals:
            sample_time_lbl = f"{float(np.mean(sample_time_after_cue_vals)):.1f} s"

        # Store data for deferred violin generation (needs FDR-corrected q-values)
        amp_plot_data[amp] = {
            'amp_dir': amp_dir,
            'pick_lbl': pick_lbl,
            'sample_time_lbl': sample_time_lbl,
            'by_cond_median_power': dict(by_cond_median_power),
            'by_cond_sample_power': dict(by_cond_sample_power),
            'by_cond_cue_rate': dict(by_cond_cue_rate),
            'by_cond_power_var': dict(by_cond_power_var),
            'by_cond_power_dvar': dict(by_cond_power_dvar),
            'by_cond_power_autocorr': dict(by_cond_power_autocorr),
            'by_cond_spec_concentration': dict(by_cond_spec_concentration),
            'by_cond_spec_entropy': dict(by_cond_spec_entropy),
        }

        if metrics_over_delay and delay_labels:
            fig_t = plot_metrics_vs_delay(
                metrics_over_delay,
                delay_labels=delay_labels,
                metrics_to_plot=('power_sample', 'freq_sample_hz'),
                save_path=os.path.join(amp_dir, "oscillation_vs_time.png"),
                suptitle=(
                    f"Oscillation Metrics vs Time | amp={_fmt(amp)}x "
                    f"({n_trials} trials, +/-SEM) [{args.min_freq_hz:g}-{args.max_freq_hz:g} Hz]"
                ),
                error_band='sem',
                separate_app=False,
            )
            plt.close(fig_t)

        # Pairwise distribution tests for this amplitude
        for i, ca in enumerate(condition_keys):
            for j, cb in enumerate(condition_keys):
                if j <= i:
                    continue
                for metric_name, by_cond in [
                    ('power_median', by_cond_median_power),
                    ('power_sample', by_cond_sample_power),
                    ('power_var', by_cond_power_var),
                    ('power_dvar', by_cond_power_dvar),
                    ('power_autocorr', by_cond_power_autocorr),
                    ('spec_concentration', by_cond_spec_concentration),
                    ('spec_entropy', by_cond_spec_entropy),
                    ('mean_cue_rate_hz', by_cond_cue_rate),
                ]:
                    arr_a = by_cond.get(ca, np.array([]))
                    arr_b = by_cond.get(cb, np.array([]))
                    if len(arr_a) > 0 and len(arr_b) > 0:
                        u, p = _scipy_stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
                        stats_rows.append({
                            'amplitude': amp,
                            'metric': metric_name,
                            'cond_a': ca,
                            'cond_b': cb,
                            'n_a': len(arr_a),
                            'n_b': len(arr_b),
                            'u_stat': float(u),
                            'p_value': float(p),
                        })

    # FDR correction (Benjamini-Hochberg) across all tests
    if stats_rows:
        from scipy.stats import false_discovery_control as _fdr
        raw_pvals = np.array([r['p_value'] for r in stats_rows])
        q_vals = _fdr(raw_pvals, method='bh')
        for r, q in zip(stats_rows, q_vals):
            r['q_value'] = float(q)
    else:
        for r in stats_rows:
            r['q_value'] = np.nan

    stats_csv = os.path.join(out_dir, "oscillation_stats.csv")
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['amplitude', 'metric', 'cond_a', 'cond_b', 'n_a', 'n_b', 'u_stat', 'p_value', 'q_value'],
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    # ------------------------------------------------------------------
    # Per-amplitude grouped violin plots (deferred until after FDR)
    # ------------------------------------------------------------------
    def _amp_stat(amp, metric):
        """Return the first matching stats row for this amp+metric (cond_a vs cond_b)."""
        for r in stats_rows:
            if r['amplitude'] == amp and r['metric'] == metric:
                return {'cond_a': r['cond_a'], 'cond_b': r['cond_b'], 'q_value': r['q_value']}
        return None

    conn_lbl = _weights_label(base_rp)
    cond_labels = _build_cond_labels(condition_keys, cond_excit)
    for amp, pd_amp in amp_plot_data.items():
        amp_dir = pd_amp['amp_dir']
        pick_lbl = pd_amp['pick_lbl']
        sample_time_lbl = pd_amp['sample_time_lbl']

        fig_vp = plot_oscillation_multi_violin(
            panels=[
                (
                    "Median power\n(full delay)",
                    "Median dominant power",
                    pd_amp['by_cond_median_power'],
                ),
                (
                    f"Sampled power\n(t={sample_time_lbl} post-cue)",
                    "Sampled dominant power",
                    pd_amp['by_cond_sample_power'],
                ),
                (
                    "Cue-node rate\n(delay)",
                    "Mean firing rate (Hz)",
                    pd_amp['by_cond_cue_rate'],
                ),
            ],
            cond_order=condition_keys,
            cond_labels=cond_labels,
            suptitle=(
                f"Dominant power | amp={_fmt(amp)}x | {conn_lbl}"
                + (f" | f: {pick_lbl}" if pick_lbl != "NA" else "")
            ),
            stats_per_panel=[
                _amp_stat(amp, 'power_median'),
                _amp_stat(amp, 'power_sample'),
                _amp_stat(amp, 'mean_cue_rate_hz'),
            ],
            save_path=os.path.join(amp_dir, "violin_power.png"),
        )
        plt.close(fig_vp)

        fig_vs = plot_oscillation_multi_violin(
            panels=[
                (
                    "Total variance\n(delay)",
                    "Var(dominant power)",
                    pd_amp['by_cond_power_var'],
                ),
                (
                    "Detrended variance\n(delay)",
                    "Var(residuals)",
                    pd_amp['by_cond_power_dvar'],
                ),
                (
                    "Spectral concentration\n(delay)",
                    "Peak / total band power  [0–1]",
                    pd_amp['by_cond_spec_concentration'],
                ),
                (
                    "Spectral entropy\n(delay)",
                    "Shannon entropy (lower = sharper)",
                    pd_amp['by_cond_spec_entropy'],
                ),
            ],
            cond_order=condition_keys,
            cond_labels=cond_labels,
            suptitle=f"Oscillation stability & spectral focus | amp={_fmt(amp)}x | {conn_lbl}",
            stats_per_panel=[
                _amp_stat(amp, 'power_var'),
                _amp_stat(amp, 'power_dvar'),
                _amp_stat(amp, 'spec_concentration'),
                _amp_stat(amp, 'spec_entropy'),
            ],
            save_path=os.path.join(amp_dir, "violin_stability.png"),
        )
        plt.close(fig_vs)

    # ------------------------------------------------------------------
    # Cross-amplitude sweep: mean ± std line plots
    # ------------------------------------------------------------------
    if len(amplitudes) > 1:
        def _sweep_stats(metric):
            return [
                {'amp': r['amplitude'], 'q_value': r['q_value'],
                 'cond_a': r['cond_a'], 'cond_b': r['cond_b']}
                for r in stats_rows if r['metric'] == metric
            ]

        fig_sw1 = plot_oscillation_amp_sweep_lines(
            panels=[
                (
                    "Dominant power — full delay (median)",
                    "Median dominant power",
                    sweep_power_median,
                ),
                (
                    "Dominant power — 2 s post-cue (sample)",
                    "Sampled dominant power",
                    sweep_power_sample,
                ),
            ],
            amplitudes=amplitudes,
            cond_order=condition_keys,
            cond_labels=cond_labels,
            stats_per_panel=[_sweep_stats('power_median'), _sweep_stats('power_sample')],
            suptitle=f"Dominant power vs cue amplitude — {conn_lbl}",
            save_path=os.path.join(out_dir, "oscillation_amp_sweep_power.png"),
        )
        plt.close(fig_sw1)

        fig_sw2 = plot_oscillation_amp_sweep_lines(
            panels=[
                (
                    "Total variance over delay",
                    "Var(dominant power)",
                    sweep_power_var,
                ),
                (
                    "Detrended variance over delay",
                    "Var(residuals after linear detrend)",
                    sweep_power_dvar,
                ),
                (
                    "Spectral concentration",
                    "Peak / total band power  [0–1]",
                    sweep_spec_concentration,
                ),
                (
                    "Spectral entropy",
                    "Shannon entropy (lower = sharper)",
                    sweep_spec_entropy,
                ),
            ],
            amplitudes=amplitudes,
            cond_order=condition_keys,
            cond_labels=cond_labels,
            stats_per_panel=[
                _sweep_stats('power_var'),
                _sweep_stats('power_dvar'),
                _sweep_stats('spec_concentration'),
                _sweep_stats('spec_entropy'),
            ],
            suptitle=f"Oscillation stability & spectral focus\nvs cue amplitude — {conn_lbl}",
            save_path=os.path.join(out_dir, "oscillation_amp_sweep_variance.png"),
        )
        plt.close(fig_sw2)

    print("\nOscillation study complete.")
    print(f"  Trial summary CSV: {summary_csv}")
    print(f"  Timecourse CSV:    {traj_csv}")
    print(f"  Stats CSV:         {stats_csv}")
    print(f"  Figures:           {out_dir}")
    print(f"  Cache file:        {cache_file}  (key={cache_key})")


# ============================================================================
# OSCILLATION-DISTRACTOR STUDY: PARALLEL WORKER
# ============================================================================

_osc_dist_sim_args: Optional[dict] = None


def _osc_dist_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    per_cond_rp: dict[str, RingParams],
    per_cond_conn: dict[str, RingConnectivity],
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    T_ms_full: float,
):
    """Initialize worker process for oscillation-distractor-study jobs."""
    global _osc_dist_sim_args
    _osc_dist_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'per_cond_conn': per_cond_conn,
        'burnin_states': burnin_states,
        'T_ms_full': T_ms_full,
    }


def _osc_dist_run_single(job: tuple) -> dict:
    """Run one cue + optional-distractor trial and extract oscillation metrics at both nodes."""
    global _osc_dist_sim_args
    cfg = _osc_dist_sim_args
    cond_key, amplitude, distractor_factor, offset_deg, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['per_cond_rp'][cond_key]
    connectivity = cfg['per_cond_conn'][cond_key]

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    # ------------------------------------------------------------------
    # Timeline (all times in post-burnin coordinates: t=0 = start of sim
    # after burn-in, i.e. STIM_ONSET_MS - BURN_IN_MS = 500 ms)
    # ------------------------------------------------------------------
    pre_cue_ms = STIM_ONSET_MS - BURN_IN_MS          # 500 ms
    cue_offset_ms = pre_cue_ms + STIM_DURATION_MS    # 750 ms
    delay1_ms = float(args_d['delay1_ms'])
    dist_duration_ms = float(args_d['distractor_duration_ms'])
    delay2_ms = float(args_d['delay2_ms'])

    dist_onset_ms = cue_offset_ms + delay1_ms
    dist_offset_ms = dist_onset_ms + dist_duration_ms
    T_ms_short = dist_offset_ms + delay2_ms

    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    cue_current = amplitude * base_params.I_ext_pyr()
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=cue_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=pre_cue_ms,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    if offset_deg is not None:
        dist_center_deg = (STIM_CENTER_DEG + float(offset_deg)) % 360.0
        dist_current = distractor_factor * cue_current
        stimuli_short.append(
            RingStimulus(
                center_deg=dist_center_deg,
                amplitude=dist_current,
                sigma_deg=STIM_SIGMA_DEG,
                onset_ms=dist_onset_ms,
                duration_ms=dist_duration_ms,
            )
        )

    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=T_ms_short,
        stimuli=stimuli_short,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=seed,
        connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    # Shift time axis to absolute (post-burnin already, but match STIM_ONSET_MS reference)
    result.t_ms += BURN_IN_MS

    # ------------------------------------------------------------------
    # Identify node indices
    # ------------------------------------------------------------------
    angles_deg = np.rad2deg(ring_params.node_angles_rad)
    cue_idx = int(np.argmin(np.abs(angles_deg - STIM_CENTER_DEG)))
    if offset_deg is not None:
        dist_center_deg = (STIM_CENTER_DEG + float(offset_deg)) % 360.0
        # Account for wrap-around
        ang_diff = np.abs(angles_deg - dist_center_deg)
        ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
        dist_idx = int(np.argmin(ang_diff))
    else:
        dist_idx = (cue_idx + len(angles_deg) // 2) % len(angles_deg)  # antipodal node for no-distractor control

    # ------------------------------------------------------------------
    # Extract timecourses over full post-cue window
    # ------------------------------------------------------------------
    analysis_start_ms = STIM_ONSET_MS + STIM_DURATION_MS   # absolute
    mask_full = result.t_ms >= analysis_start_ms
    t_full_s = (result.t_ms[mask_full] - analysis_start_ms) / 1000.0  # s since cue offset
    cue_rate = result.r[mask_full, cue_idx, 0]
    dist_rate = result.r[mask_full, dist_idx, 0]

    dist_onset_rel_s = delay1_ms / 1000.0     # distractor onset in t_full_s coords
    dist_offset_rel_s = dist_onset_rel_s + dist_duration_ms / 1000.0

    min_freq = args_d.get('min_freq_hz', 2.0)
    max_freq = args_d.get('max_freq_hz', 12.0)
    win_s = args_d.get('tf_window_s', 1.0)
    overlap = args_d.get('tf_overlap', 0.8)

    _empty_osc = {
        'freqs_hz': np.array([], dtype=float),
        'times_s': np.array([], dtype=float),
        'power': np.zeros((0, 0), dtype=float),
        'dominant_freq_hz': np.array([], dtype=float),
        'dominant_power': np.array([], dtype=float),
    }

    try:
        osc_cue = compute_oscillation_band_timecourse(
            cue_rate, t_full_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except ValueError:
        osc_cue = _empty_osc.copy()

    try:
        osc_dist = compute_oscillation_band_timecourse(
            dist_rate, t_full_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except ValueError:
        osc_dist = _empty_osc.copy()

    try:
        plv_result = compute_plv_timecourse(
            cue_rate, dist_rate, t_full_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except Exception:
        plv_result = {'times_s': np.array([], dtype=float), 'plv': np.array([], dtype=float)}

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'distractor_factor': distractor_factor,
        'offset_deg': offset_deg,       # None = no-distractor control
        'trial_idx': trial_idx,
        'seed': seed,
        # Cue node STFT
        'cue_times_s': osc_cue['times_s'],
        'cue_freqs_hz': osc_cue['freqs_hz'],
        'cue_power': osc_cue['power'],
        'cue_dominant_freq_hz': osc_cue['dominant_freq_hz'],
        'cue_dominant_power': osc_cue['dominant_power'],
        # Distractor node STFT
        'dist_times_s': osc_dist['times_s'],
        'dist_freqs_hz': osc_dist['freqs_hz'],
        'dist_power': osc_dist['power'],
        'dist_dominant_freq_hz': osc_dist['dominant_freq_hz'],
        'dist_dominant_power': osc_dist['dominant_power'],
        # PLV
        'plv_times_s': plv_result['times_s'],
        'plv': plv_result['plv'],
        # Timeline references (in t_full_s coords = seconds since cue offset)
        'dist_onset_rel_s': dist_onset_rel_s,
        'dist_offset_rel_s': dist_offset_rel_s,
    }


def _osc_dist_cache_key(
    args: argparse.Namespace,
    base_params: CircuitParams,
    ring_params: RingParams,
    condition_keys: list[str],
    amplitudes: list[float],
) -> str:
    """Return a 16-char hex key for the oscillation-distractor study inputs."""
    import dataclasses
    import hashlib
    import json

    def _to_json(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return str(obj)

    params = {
        'base_params':           _to_json(base_params),
        'ring_params':           _to_json(ring_params),
        'condition_keys':        sorted(condition_keys),
        'amplitudes':            sorted(amplitudes),
        'distractor_factors':    sorted(getattr(args, 'distractor_factors', [1.0])),
        'offsets_deg':           sorted(getattr(args, 'offsets_deg', [90.0])),
        'n_trials':              int(args.n_trials),
        'seed':                  int(args.seed),
        'delay1_ms':             float(args.delay1_ms),
        'distractor_duration_ms': float(args.distractor_duration_ms),
        'delay2_ms':             float(args.delay2_ms),
        'min_freq_hz':           float(args.min_freq_hz),
        'max_freq_hz':           float(args.max_freq_hz),
        'tf_window_s':           float(args.tf_window_s),
        'tf_overlap':            float(args.tf_overlap),
        'record_dt_ms':          float(getattr(args, 'record_dt_ms', 5.0)),
    }
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cmd_osc_distractor_study(args: argparse.Namespace) -> None:
    """Oscillation-distractor study: STFT at cue/distractor nodes + PLV timecourses."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    if args.conditions is None:
        condition_keys = ['WT']
    else:
        condition_keys = args.conditions
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'.\nValid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for config display

    amplitudes = list(args.amplitudes) if args.amplitudes else [args.amplitude[0]]
    distractor_factors = list(args.distractor_factors)
    offsets_deg = list(args.offsets_deg)
    n_trials = int(args.n_trials)
    n_workers = _resolve_workers(args)

    conn_label = _calibration_network_label(base_rp)
    conn_lbl = _weights_label(base_rp)
    out_root = os.path.join(
        _output_dir("figs/ring/osc_distractor", args.params_json),
        conn_label,
    )
    os.makedirs(out_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache key — computed before burn-in so we can skip it on cache hit
    # ------------------------------------------------------------------
    import pickle as _pickle
    use_cache = not getattr(args, 'no_cache', False)
    cache_key = _osc_dist_cache_key(args, base_params, base_rp, condition_keys, amplitudes)
    cache_file = os.path.join(out_root, f'.osc_dist_cache_{cache_key}.pkl')

    cond_labels = _build_cond_labels(condition_keys, cond_excit)

    _print_config(args, amplitudes[0], base_params, 0.0, ring_params,
                  experiment_info=[
                      f"Conditions:        {', '.join(condition_keys)}",
                      f"Amplitudes:        {', '.join(_fmt(a) for a in amplitudes)}× I_ext_pyr",
                      f"Distractor factors:{', '.join(str(f) for f in distractor_factors)}",
                      f"Offsets (deg):     {', '.join(str(o) for o in offsets_deg)}",
                      f"Timing:            delay1={args.delay1_ms:.0f} ms"
                      f"   distractor={args.distractor_duration_ms:.0f} ms"
                      f"   delay2={args.delay2_ms:.0f} ms",
                      f"Trials:            {n_trials}   seed={args.seed}   workers={n_workers}",
                      f"Freq band:         [{args.min_freq_hz:.1f}, {args.max_freq_hz:.1f}] Hz"
                      f"   window={args.tf_window_s:.3f} s   overlap={args.tf_overlap:.2f}",
                      f"Cache key:         {cache_key}",
                  ],
                  save_path=os.path.join(out_root, "experiment_config.txt"))

    all_results: list[dict] = []
    if use_cache and os.path.exists(cache_file):
        print(f"\nLoading cached results (key={cache_key})...")
        with open(cache_file, 'rb') as _cf:
            all_results = _pickle.load(_cf)
        print(f"  Loaded {len(all_results)} trials from cache.")
        print(f"  Pass --no_cache to force re-computation.")
    else:
        # Burn-in and simulation — only run when no valid cache exists
        print("\nComputing burn-in states...")
        burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cond_key in tqdm(condition_keys, desc="Burn-in", unit="cond"):
            local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
            burnin_states[cond_key] = _compute_burnin_state(
                local_params,
                per_cond_rp[cond_key],
                per_cond_conn[cond_key],
                seed=args.seed,
            )

        trial_seeds = _generate_trial_seeds(args.seed, n_trials)

        # Build jobs: per (condition, amplitude, factor, offset_or_None, trial)
        jobs = []
        for ck in condition_keys:
            for amp in amplitudes:
                for factor in distractor_factors:
                    for off in offsets_deg:
                        for ti in range(n_trials):
                            jobs.append((ck, amp, factor, off, ti, trial_seeds[ti]))
                    # Control: no distractor
                    for ti in range(n_trials):
                        jobs.append((ck, amp, factor, None, ti, trial_seeds[ti]))

        args_dict = {
            'delay1_ms': args.delay1_ms,
            'distractor_duration_ms': args.distractor_duration_ms,
            'delay2_ms': args.delay2_ms,
            'min_freq_hz': args.min_freq_hz,
            'max_freq_hz': args.max_freq_hz,
            'tf_window_s': args.tf_window_s,
            'tf_overlap': args.tf_overlap,
            'record_dt_ms': getattr(args, 'record_dt_ms', 5.0),
        }

        stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
        cue_offset_post_burnin = stim_offset_ms - BURN_IN_MS
        T_ms_full = cue_offset_post_burnin + args.delay1_ms + args.distractor_duration_ms + args.delay2_ms

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT,
                max_workers=n_workers,
                initializer=_osc_dist_init_worker,
                initargs=(args_dict, base_params, per_cond_rp, per_cond_conn, burnin_states, T_ms_full),
            ) as executor:
                futures = {executor.submit(_osc_dist_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _osc_dist_init_worker(args_dict, base_params, per_cond_rp, per_cond_conn, burnin_states, T_ms_full)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                all_results.append(_osc_dist_run_single(job))

        with open(cache_file, 'wb') as _cf:
            _pickle.dump(all_results, _cf, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"\nSimulation results cached → {cache_file}")

    # ------------------------------------------------------------------
    # Trial-level CSV
    # ------------------------------------------------------------------
    trials_csv = os.path.join(out_root, "osc_distractor_trials.csv")
    with open(trials_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'condition', 'amplitude', 'distractor_factor', 'offset_deg',
            'trial_idx', 'seed',
            'cue_freq_median_hz', 'cue_power_median',
            'dist_freq_median_hz', 'dist_power_median',
            'plv_median_delay2',
        ])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (
            x['cond_key'], x['amplitude'], x['distractor_factor'],
            str(x['offset_deg']), x['trial_idx'],
        )):
            # PLV median in post-distractor window
            plv_t = np.asarray(r['plv_times_s'], dtype=float)
            plv_v = np.asarray(r['plv'], dtype=float)
            post_mask = plv_t > r['dist_offset_rel_s']
            plv_median_delay2 = float(np.nanmedian(plv_v[post_mask])) if np.any(post_mask) else np.nan

            # Cue/dist STFT summaries over full window
            def _median_or_nan(arr):
                a = np.asarray(arr, dtype=float)
                v = a[np.isfinite(a)]
                return float(np.median(v)) if len(v) > 0 else np.nan

            writer.writerow({
                'condition': r['cond_key'],
                'amplitude': r['amplitude'],
                'distractor_factor': r['distractor_factor'],
                'offset_deg': '' if r['offset_deg'] is None else r['offset_deg'],
                'trial_idx': r['trial_idx'],
                'seed': r['seed'],
                'cue_freq_median_hz': _median_or_nan(r['cue_dominant_freq_hz']),
                'cue_power_median':   _median_or_nan(r['cue_dominant_power']),
                'dist_freq_median_hz': _median_or_nan(r['dist_dominant_freq_hz']),
                'dist_power_median':   _median_or_nan(r['dist_dominant_power']),
                'plv_median_delay2':   plv_median_delay2,
            })

    # ------------------------------------------------------------------
    # Aggregate and plot per (condition, amplitude, distractor_factor)
    # ------------------------------------------------------------------
    def _stack_timecourse(rows, key):
        """Stack a timecourse key from a list of result dicts → (t_axis, mean, sd)."""
        valid = [r for r in rows if len(r.get(key, [])) > 0]
        if not valid:
            return np.array([]), np.array([]), np.array([])
        max_len = max(len(r[key]) for r in valid)
        longest = max(valid, key=lambda r: len(r[key]))
        t_axis = np.asarray(longest.get(key.replace('plv', 'plv_times').replace(
            'cue_dominant_power', 'cue_times').replace('dist_dominant_power', 'dist_times'
        ), []), dtype=float)
        # For PLV use plv_times_s; for cue/dist use their respective times
        if key == 'plv':
            t_rows = [r.get('plv_times_s', []) for r in valid]
        elif key.startswith('cue_'):
            t_rows = [r.get('cue_times_s', []) for r in valid]
        else:
            t_rows = [r.get('dist_times_s', []) for r in valid]
        # Use longest t as reference
        t_lens = [len(t) for t in t_rows]
        t_ref_idx = int(np.argmax(t_lens))
        t_axis = np.asarray(t_rows[t_ref_idx], dtype=float)
        n = len(t_axis)

        stack = np.full((len(valid), n), np.nan)
        for j, r in enumerate(valid):
            v = np.asarray(r[key], dtype=float)
            stack[j, :len(v)] = v

        with np.errstate(all='ignore'):
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=0)
        return t_axis, mean, sd

    amp_sweep_data: dict[float, dict[str, dict[float, dict]]] = {
        factor: {'full': {}, 'last500': {}} for factor in distractor_factors
    }  # {factor: {'full'|'last500': {offset_deg: {amp: plv_values}}}}

    for ck in condition_keys:
        cond_out = os.path.join(out_root, ck)
        os.makedirs(cond_out, exist_ok=True)

        for factor in distractor_factors:
            factor_label = f"factor{_fmt(factor)}"
            factor_out = os.path.join(cond_out, factor_label)
            os.makedirs(factor_out, exist_ok=True)

            for amp in amplitudes:
                amp_label = f"amp{_fmt(amp)}"

                # Build aggregated data_by_offset for timecourse plot
                data_by_offset: dict = {}

                # All offsets + control
                for off in offsets_deg + [None]:
                    rows = [
                        r for r in all_results
                        if r['cond_key'] == ck
                        and abs(r['amplitude'] - amp) < 1e-9
                        and abs(r['distractor_factor'] - factor) < 1e-9
                        and r['offset_deg'] == off
                    ]
                    if not rows:
                        continue

                    t_cue, cue_mean, cue_sd = _stack_timecourse(rows, 'cue_dominant_power')
                    t_dist, dist_mean, dist_sd = _stack_timecourse(rows, 'dist_dominant_power')
                    t_plv, plv_mean, plv_sd = _stack_timecourse(rows, 'plv')

                    # Use cue time axis as common reference (they share the same STFT grid)
                    dist_onset_rel_s = rows[0]['dist_onset_rel_s']
                    if len(t_cue) > 0:
                        t_rel = t_cue - dist_onset_rel_s
                    else:
                        t_rel = np.array([])

                    data_by_offset[off] = {
                        'cue_mean': cue_mean,
                        'cue_sd': cue_sd,
                        'dist_mean': dist_mean,
                        'dist_sd': dist_sd,
                        'plv_mean': plv_mean,
                        'plv_sd': plv_sd,
                        't_rel': t_rel,
                    }

                    # Amplitude sweep data: PLV median in delay2 per trial
                    if off is not None:
                        dist_offset_rel_s = rows[0]['dist_offset_rel_s']
                        plv_medians_full = []
                        plv_medians_last500 = []
                        for r in rows:
                            plv_t = np.asarray(r['plv_times_s'], dtype=float)
                            plv_v = np.asarray(r['plv'], dtype=float)
                            t_end = plv_t[-1] if len(plv_t) > 0 else dist_offset_rel_s
                            post_mask = plv_t > dist_offset_rel_s
                            last500_mask = plv_t >= (t_end - 0.5)
                            if np.any(post_mask):
                                plv_medians_full.append(float(np.nanmedian(plv_v[post_mask])))
                            if np.any(last500_mask):
                                plv_medians_last500.append(float(np.nanmedian(plv_v[last500_mask])))
                        off_float = float(off)
                        for window, medians in [('full', plv_medians_full), ('last500', plv_medians_last500)]:
                            if off_float not in amp_sweep_data[factor][window]:
                                amp_sweep_data[factor][window][off_float] = {}
                            amp_sweep_data[factor][window][off_float][amp] = np.array(medians)

                # Common t_rel axis: use the longest from non-None offsets
                t_rel_axis = np.array([])
                dist_offset_s = rows[0]['dist_offset_rel_s'] - rows[0]['dist_onset_rel_s'] if rows else 0.2
                for off, d in data_by_offset.items():
                    t = d.get('t_rel', np.array([]))
                    if len(t) > len(t_rel_axis):
                        t_rel_axis = t

                # Realign all entries to common axis
                for off, d in data_by_offset.items():
                    t = d.get('t_rel', np.array([]))
                    if len(t) < len(t_rel_axis):
                        pad = len(t_rel_axis) - len(t)
                        d['cue_mean'] = np.concatenate([d['cue_mean'], np.full(pad, np.nan)])
                        d['cue_sd'] = np.concatenate([d['cue_sd'], np.full(pad, np.nan)])
                        d['dist_mean'] = np.concatenate([d['dist_mean'], np.full(pad, np.nan)])
                        d['dist_sd'] = np.concatenate([d['dist_sd'], np.full(pad, np.nan)])
                        d['plv_mean'] = np.concatenate([d['plv_mean'], np.full(pad, np.nan)])
                        d['plv_sd'] = np.concatenate([d['plv_sd'], np.full(pad, np.nan)])

                amp_out = os.path.join(factor_out, amp_label)
                os.makedirs(amp_out, exist_ok=True)

                # 1. Timecourse figure
                fig_tc = plot_osc_distractor_timecourses(
                    t_rel_axis=t_rel_axis,
                    data_by_offset=data_by_offset,
                    dist_offset_s=dist_offset_s,
                    suptitle=(
                        f"Osc-Distractor | {ck} | {amp_label}× | {factor_label} | {conn_lbl}"
                    ),
                    save_path=os.path.join(amp_out, "osc_distractor_timecourses.png"),
                )
                plt.close(fig_tc)

                # 2. Spectrogram per offset
                for off in offsets_deg:
                    rows_off = [
                        r for r in all_results
                        if r['cond_key'] == ck
                        and abs(r['amplitude'] - amp) < 1e-9
                        and abs(r['distractor_factor'] - factor) < 1e-9
                        and r['offset_deg'] == off
                        and r['cue_power'].size > 0
                    ]
                    if not rows_off:
                        continue
                    ref = rows_off[0]
                    cue_powers = [r['cue_power'] for r in rows_off if r['cue_power'].size > 0]
                    dist_powers = [r['dist_power'] for r in rows_off if r['dist_power'].size > 0]
                    cue_pm = np.mean(np.stack(cue_powers), axis=0) if cue_powers else np.zeros((0, 0))
                    dist_pm = np.mean(np.stack(dist_powers), axis=0) if dist_powers else np.zeros((0, 0))

                    t_rel_sg = ref['cue_times_s'] - ref['dist_onset_rel_s']

                    # Mean dominant freq across trials
                    def _mean_freq(rows_f, key):
                        arrs = [np.asarray(r[key], dtype=float) for r in rows_f]
                        if not arrs:
                            return np.array([])
                        ml = max(len(a) for a in arrs)
                        st = np.full((len(arrs), ml), np.nan)
                        for j, a in enumerate(arrs):
                            st[j, :len(a)] = a
                        return np.nanmean(st, axis=0)

                    cue_df = _mean_freq(rows_off, 'cue_dominant_freq_hz')
                    dist_df = _mean_freq(rows_off, 'dist_dominant_freq_hz')

                    fig_sg = plot_osc_distractor_spectrograms(
                        cue_power_mean=cue_pm,
                        dist_power_mean=dist_pm,
                        freqs_hz=ref['cue_freqs_hz'],
                        times_rel_s=t_rel_sg,
                        cue_dominant_freq=cue_df,
                        dist_dominant_freq=dist_df,
                        dist_offset_s=dist_offset_s,
                        title=(
                            f"STFT | {ck} | {amp_label}× | offset={int(off)}° | {factor_label}"
                        ),
                        save_path=os.path.join(
                            amp_out, f"osc_distractor_spectrograms_offset{int(off)}.png"
                        ),
                    )
                    plt.close(fig_sg)

        # 3. Amplitude sweep (one per condition and factor)
        if len(amplitudes) > 1:
            for factor in distractor_factors:
                factor_label = f"factor{_fmt(factor)}"
                factor_out = os.path.join(out_root, ck, factor_label)

                factor_sweep = amp_sweep_data.get(factor, {'full': {}, 'last500': {}})
                panels = [
                    (
                        f"Full post-distractor delay | {factor_label}",
                        "Median PLV (full post-distractor)",
                        factor_sweep['full'],
                    ),
                    (
                        f"Last 500 ms of delay | {factor_label}",
                        "Median PLV (last 500 ms)",
                        factor_sweep['last500'],
                    ),
                ]
                fig_sw = plot_osc_distractor_amp_sweep(
                    panels=panels,
                    amplitudes=amplitudes,
                    offsets_deg=offsets_deg,
                    suptitle=f"PLV vs cue amplitude | {ck} | {factor_label} | {conn_lbl}",
                    save_path=os.path.join(factor_out, "osc_distractor_amp_sweep.png"),
                )
                plt.close(fig_sw)

    # ------------------------------------------------------------------
    # Cross-condition box plots (only when multiple conditions are compared)
    # ------------------------------------------------------------------
    if len(condition_keys) > 1:
        def _stack_full_osc(rows, val_key, time_key):
            valid = [r for r in rows if len(r.get(val_key, [])) > 0]
            if not valid:
                return np.array([]), np.zeros((0, 0))
            t_rows = [np.asarray(r.get(time_key, []), dtype=float) for r in valid]
            t_ref = max(t_rows, key=len)
            n = len(t_ref)
            stack = np.full((len(valid), n), np.nan)
            for j, r in enumerate(valid):
                v = np.asarray(r[val_key], dtype=float)
                stack[j, :len(v)] = v
            return t_ref, stack

        for factor in distractor_factors:
            factor_label = f"factor{_fmt(factor)}"
            for amp in amplitudes:
                amp_label = f"amp{_fmt(amp)}"
                amp_dir = os.path.join(out_root, condition_keys[0], factor_label, amp_label)
                for off in offsets_deg:
                    data_by_cond: dict = {}
                    t_ref_bp = np.array([])
                    dist_off_s_bp = 0.2
                    for ck in condition_keys:
                        rows_bp = [
                            r for r in all_results
                            if r['cond_key'] == ck
                            and abs(r['amplitude'] - amp) < 1e-9
                            and abs(r['distractor_factor'] - factor) < 1e-9
                            and r['offset_deg'] == off
                        ]
                        if not rows_bp:
                            continue
                        t_cue, stack_cue = _stack_full_osc(
                            rows_bp, 'cue_dominant_power', 'cue_times_s')
                        _, stack_dst = _stack_full_osc(
                            rows_bp, 'dist_dominant_power', 'dist_times_s')
                        _, stack_plv = _stack_full_osc(
                            rows_bp, 'plv', 'plv_times_s')
                        dist_onset_s = rows_bp[0].get('dist_onset_rel_s', 0.0)
                        dist_off_s_bp = (
                            rows_bp[0].get('dist_offset_rel_s', dist_onset_s + 0.2)
                            - dist_onset_s
                        )
                        t_cue_rel = t_cue - dist_onset_s
                        if len(t_cue_rel) >= len(t_ref_bp):
                            t_ref_bp = t_cue_rel
                        data_by_cond[ck] = {
                            'plv':        stack_plv,
                            'cue_power':  stack_cue,
                            'dist_power': stack_dst,
                        }
                    if len(data_by_cond) < 2:
                        continue
                    cmp_dir = os.path.join(out_root, "comparison", factor_label, amp_label)
                    os.makedirs(cmp_dir, exist_ok=True)
                    fig_bp = plot_osc_conditions_boxplot(
                        t_axis=t_ref_bp,
                        data_by_condition=data_by_cond,
                        dist_offset_s=dist_off_s_bp,
                        cond_labels=cond_labels,
                        suptitle=(
                            f"Condition comparison | {amp_label}× | {factor_label} | "
                            f"offset={int(off)}° | {conn_lbl}"
                        ),
                        save_path=os.path.join(
                            cmp_dir,
                            f"conditions_boxplot_offset{int(off)}.png",
                        ),
                    )
                    plt.close(fig_bp)

    print("\nOscillation-distractor study complete.")
    print(f"  Trial CSV:  {trials_csv}")
    print(f"  Figures:    {out_root}")
    print(f"  Cache file: {cache_file}  (key={cache_key})")


# ============================================================================
# PRE-CUE POWER STUDY: PARALLEL WORKERS
# ============================================================================

_pre_cue_power_sim_args: Optional[dict] = None


def _pre_cue_power_init_worker(
    base_params: "CircuitParams",
    per_cond_rp: dict,
    per_cond_conn: dict,
    burnin_states: dict,
    duration_ms: float,
    record_dt_ms: float,
    min_freq_hz: float,
    max_freq_hz: float,
    tf_window_s: float,
    tf_overlap: float,
) -> None:
    """Initialise worker process for pre-cue power study jobs."""
    global _pre_cue_power_sim_args
    _pre_cue_power_sim_args = {
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'per_cond_conn': per_cond_conn,
        'burnin_states': burnin_states,
        'duration_ms': duration_ms,
        'record_dt_ms': record_dt_ms,
        'min_freq_hz': min_freq_hz,
        'max_freq_hz': max_freq_hz,
        'tf_window_s': tf_window_s,
        'tf_overlap': tf_overlap,
    }


def _pre_cue_power_run_single(job: tuple) -> dict:
    """Run one noise-only trial from burn-in state and return mean PSD per frequency."""
    global _pre_cue_power_sim_args
    cfg = _pre_cue_power_sim_args
    cond_key, trial_idx, seed = job

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(cfg['base_params'], condition)
    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    rp = cfg['per_cond_rp'][cond_key]
    result = simulate_ring(
        local_params,
        rp,
        T_ms=cfg['duration_ms'],
        stimuli=None,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=seed,
        connectivity=cfg['per_cond_conn'][cond_key],
        record_dt_ms=cfg['record_dt_ms'],
    )

    t_s = result.t_ms / 1000.0
    _, amp_t = population_vector_decode(result.r[:, :, 0], rp.node_angles_rad)

    try:
        osc = compute_oscillation_band_timecourse(
            amp_t, t_s,
            min_freq_hz=cfg['min_freq_hz'],
            max_freq_hz=cfg['max_freq_hz'],
            window_s=cfg['tf_window_s'],
            overlap_frac=cfg['tf_overlap'],
        )
        if osc['power'].size > 0:
            mean_power = np.mean(osc['power'], axis=1)  # (n_freqs,): average over time
        else:
            mean_power = np.zeros(len(osc['freqs_hz']))
        freqs_hz = osc['freqs_hz']
    except ValueError:
        freqs_hz = np.array([], dtype=float)
        mean_power = np.array([], dtype=float)

    return {
        'cond_key': cond_key,
        'trial_idx': trial_idx,
        'seed': seed,
        'freqs_hz': freqs_hz,
        'mean_power': mean_power,
    }


# ============================================================================
# PRE-CUE POWER STUDY: SUBCOMMAND
# ============================================================================

def cmd_pre_cue_power_study(args: argparse.Namespace) -> None:
    """Pre-cue (noise-driven) power spectrum analysis across conditions.

    Runs noise-only simulations from the burn-in state, computes the mean
    power spectral density (PSD) across the specified frequency band, and
    compares spectral peakedness (1 − normalised entropy) between conditions.

    Outputs
    -------
    - pre_cue_power_spectrum.png  : mean PSD ± std per condition
    - pre_cue_power_metric.png    : boxplot of spectral peakedness per condition
    - pre_cue_power_trials.csv    : per-trial spectral peakedness values
    """
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    if args.conditions is None:
        condition_keys = ['WT']
    else:
        condition_keys = args.conditions
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'.\nValid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for config display

    n_trials = int(args.n_trials)
    n_workers = _resolve_workers(args)
    duration_ms = float(args.duration_ms)
    record_dt_ms = float(getattr(args, 'record_dt_ms', 5.0))

    conn_label = _calibration_network_label(base_rp)
    conn_lbl = _weights_label(base_rp)
    freq_label = f"{_fmt(args.min_freq_hz)}-{_fmt(args.max_freq_hz)}hz"
    out_root = os.path.join(
        _output_dir("figs/ring/pre_cue_power", args.params_json),
        conn_label,
        freq_label,
    )
    os.makedirs(out_root, exist_ok=True)

    cond_labels = _build_cond_labels(condition_keys, cond_excit)

    _print_config(args, args.amplitude[0], base_params, 0.0, ring_params,
                  experiment_info=[
                      f"Conditions:  {', '.join(condition_keys)}",
                      f"Duration:    {duration_ms:.0f} ms (noise-only per trial)",
                      f"Trials:      {n_trials}   seed={args.seed}   workers={n_workers}",
                      f"Freq band:   [{args.min_freq_hz:.1f}, {args.max_freq_hz:.1f}] Hz"
                      f"   window={args.tf_window_s:.2f} s   overlap={args.tf_overlap:.2f}",
                  ],
                  save_path=os.path.join(out_root, "experiment_config.txt"))

    print("\nComputing burn-in states...")
    burnin_states: dict = {}
    for ck in tqdm(condition_keys, desc="Burn-in", unit="cond"):
        local_params = apply_condition(base_params, STUDY_CONDITIONS[ck])
        burnin_states[ck] = _compute_burnin_state(
            local_params, per_cond_rp[ck], per_cond_conn[ck], seed=args.seed,
        )

    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    jobs = [
        (ck, ti, trial_seeds[ti])
        for ck in condition_keys
        for ti in range(n_trials)
    ]

    init_args = (
        base_params, per_cond_rp, per_cond_conn, burnin_states,
        duration_ms, record_dt_ms,
        args.min_freq_hz, args.max_freq_hz, args.tf_window_s, args.tf_overlap,
    )

    all_results: list[dict] = []
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(
            mp_context=_MP_CONTEXT,
            max_workers=n_workers,
            initializer=_pre_cue_power_init_worker,
            initargs=init_args,
        ) as executor:
            futures = {executor.submit(_pre_cue_power_run_single, job): job for job in jobs}
            with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                for future in as_completed(futures):
                    all_results.append(future.result())
                    pbar.update()
    else:
        _pre_cue_power_init_worker(*init_args)
        for job in tqdm(jobs, desc="Simulations", unit="sim"):
            all_results.append(_pre_cue_power_run_single(job))

    # ------------------------------------------------------------------
    # Spectral peakedness metric: 1 − normalised Shannon entropy of PSD
    # High = power concentrated at one frequency; Low = flat spectrum
    # ------------------------------------------------------------------
    def _spectral_peakedness(power: np.ndarray) -> float:
        p = np.abs(power)
        total = p.sum()
        if total < 1e-30 or len(p) < 2:
            return 0.0
        p_norm = p / total
        H = -float(np.sum(p_norm * np.log(p_norm + 1e-30)))
        H_max = float(np.log(len(p)))
        return 0.0 if H_max < 1e-30 else 1.0 - H / H_max

    # ------------------------------------------------------------------
    # Organise per condition
    # ------------------------------------------------------------------
    spectrum_data: dict = {}  # {cond_key: {'freqs_hz': ..., 'powers': (n_trials, n_freqs)}}
    metric_data: dict = {}    # {cond_key: np.ndarray of peakedness values}

    for ck in condition_keys:
        rows = [r for r in all_results if r['cond_key'] == ck and r['freqs_hz'].size > 0]
        if not rows:
            continue
        freqs_hz = rows[0]['freqs_hz']
        powers, metrics = [], []
        for r in rows:
            p = np.asarray(r['mean_power'], dtype=float)
            if p.shape == freqs_hz.shape:
                powers.append(p)
                metrics.append(_spectral_peakedness(p))
        if powers:
            spectrum_data[ck] = {
                'freqs_hz': freqs_hz,
                'powers': np.stack(powers, axis=0),  # (n_trials, n_freqs)
            }
            metric_data[ck] = np.array(metrics)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(out_root, "pre_cue_power_trials.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'trial_idx', 'seed', 'spectral_peakedness'])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x['cond_key'], x['trial_idx'])):
            ck = r['cond_key']
            metrics = metric_data.get(ck, np.array([]))
            # Find index of this trial in the per-condition results
            rows_ck = [x for x in all_results if x['cond_key'] == ck and x['freqs_hz'].size > 0]
            row_idx = next((i for i, x in enumerate(rows_ck) if x['trial_idx'] == r['trial_idx']), None)
            peak = float(metrics[row_idx]) if row_idx is not None and row_idx < len(metrics) else float('nan')
            writer.writerow({
                'condition': ck,
                'trial_idx': r['trial_idx'],
                'seed': r['seed'],
                'spectral_peakedness': peak,
            })

    # ------------------------------------------------------------------
    # Statistical tests (pairwise Mann-Whitney U)
    # ------------------------------------------------------------------
    from scipy.stats import mannwhitneyu as _mwu

    def _sig(p: float) -> str:
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    cond_keys_valid = [ck for ck in condition_keys if ck in metric_data]
    print("\nSpectral peakedness statistics:")
    for ci, ck1 in enumerate(cond_keys_valid):
        v1 = metric_data[ck1]
        print(f"  {ck1}: mean={v1.mean():.4f} ± {v1.std(ddof=1):.4f} (n={len(v1)})")
    for ci, ck1 in enumerate(cond_keys_valid):
        for ck2 in cond_keys_valid[ci + 1:]:
            v1, v2 = metric_data[ck1], metric_data[ck2]
            if len(v1) >= 2 and len(v2) >= 2:
                _, p = _mwu(v1, v2, alternative='two-sided')
                print(f"  Mann-Whitney U — {ck1} vs {ck2}: p={p:.4g} {_sig(p)}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if spectrum_data:
        fig_spec = plot_pre_cue_power_spectrum(
            data=spectrum_data,
            title=f"Pre-cue power spectrum | {conn_lbl}",
            save_path=os.path.join(out_root, "pre_cue_power_spectrum.png"),
        )
        plt.close(fig_spec)

    if metric_data:
        fig_met = plot_pre_cue_power_metric(
            data=metric_data,
            title=f"Pre-cue spectral peakedness | {conn_lbl}",
            save_path=os.path.join(out_root, "pre_cue_power_metric.png"),
        )
        plt.close(fig_met)

    print("\nPre-cue power study complete.")
    print(f"  CSV:    {csv_path}")
    print(f"  Figs:   {out_root}")

    if not args.no_show:
        plt.show()


# ============================================================================
# RUN SUBCOMMAND
# ============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Run one ring simulation for a single condition and generate figures."""
    _resolve_seed(args)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, ring_params, T_ms, stimuli, amp_factor, load_msg = _build_common(args)
    print(load_msg)

    if getattr(args, 'no_adapt', False):
        base_params = replace(base_params, J_adapt_pyr=0.0, J_adapt_som=0.0)
        print("--no_adapt: J_adapt_pyr=0, J_adapt_som=0")

    if getattr(args, 'sigma_noise', None) is not None:
        base_params = replace(base_params, sigma_noise=args.sigma_noise)
        print(f"Noise override: sigma_noise = {args.sigma_noise}")

    cond_key = getattr(args, "condition", "WT")
    if cond_key not in STUDY_CONDITIONS:
        print(
            f"Error: unknown condition '{cond_key}'.\n"
            f"Valid: {', '.join(STUDY_CONDITIONS.keys())}"
        )
        sys.exit(1)

    # Use APP ring params (connectivity) when no connectivity args were explicitly
    # provided by the user; preserve --n_nodes if explicitly set.
    if _ring_args_from_defaults:
        app_rp = _base_rp_for_cond(cond_key, ring_params)
        if app_rp is not ring_params:
            ring_params = replace(app_rp, n_nodes=args.n_nodes)

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = _compute_delay_end_ms(args, stim_offset_ms)
    local_params = _apply_response_transient(local_params, args, delay_end_ms)

    # ------------------------------------------------------------------
    # Output directory (needed before _print_config for save_path)
    # ------------------------------------------------------------------
    # Use explicit output_dir if provided, otherwise auto-generate
    if getattr(args, 'output_dir', None) and args.output_dir.strip():
        out_dir = args.output_dir
    else:
        _amp_label = f"amp{_fmt(amp_factor)}"
        if getattr(args, 'no_adapt', False):
            _amp_label += "_no_adapt"
        _conn_label = f"wpyr{_fmt(ring_params.w_pyr_pyr_inter)}_wpv{_fmt(ring_params.w_pv_global)}"
        out_dir_parts = [
            _output_dir("figs/ring/run", args.params_json),
            _run_type_label(args),
            _amp_label,
            _conn_label,
        ]
        if _has_distractor(args):
            out_dir_parts.append(
                f"offset{_fmt(args.distractor_offset_deg)}_factor{_fmt(args.distractor_factor)}"
            )
        out_dir_parts.append(cond_key)
        out_dir = os.path.join(*out_dir_parts)

    os.makedirs(out_dir, exist_ok=True)

    _run_info = [
        f"Condition: {cond_key}   seed={args.seed}",
        f"Delay:     {args.delay_ms:.0f} ms",
    ]
    if _has_distractor(args):
        _dist_extra = (f"   delay3={args.response_onset_ms:.0f} ms"
                       if getattr(args, 'response_onset_ms', 0.0) > 0 else "")
        _run_info.append(
            f"Distractor: offset={args.distractor_offset_deg:.1f} deg"
            f"   factor={args.distractor_factor:.2f}×cue"
            f"   duration={args.distractor_duration_ms:.0f} ms"
            f"   delay1={args.delay_ms:.0f} ms   delay2={args.delay2_ms:.0f} ms"
            + _dist_extra
        )
    _print_config(args, amp_factor, base_params, T_ms, ring_params=ring_params,
                  experiment_info=_run_info,
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    # ------------------------------------------------------------------
    # Distractor geometry (computed once, used for both plots and MP4)
    # ------------------------------------------------------------------
    dist_node: Optional[int] = None
    dist_center_deg: Optional[float] = None
    dist_window: Optional[tuple[float, float]] = None
    if _has_distractor(args):
        dist_onset_ms = stim_offset_ms + args.delay_ms
        dist_offset_ms = dist_onset_ms + args.distractor_duration_ms
        dist_window = (dist_onset_ms, dist_offset_ms)
        dist_center_deg = (STIM_CENTER_DEG + args.distractor_offset_deg) % 360.0
        angles_deg = np.rad2deg(ring_params.node_angles_rad)
        ang_diff = np.abs(angles_deg - dist_center_deg)
        ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
        dist_node = int(np.argmin(ang_diff))

    connectivity = RingConnectivity.from_params(ring_params)
    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=T_ms,
        stimuli=stimuli,
        seed=args.seed,
        connectivity=connectivity,
        record_dt_ms=args.record_dt_ms,
        record_adaptation=True,
    )

    suptitle = (
        f"{condition.label} -- {_stim_label(amp_factor)}, {_weights_label(ring_params)}"
    )
    if dist_center_deg is not None:
        suptitle += f" | distractor {dist_center_deg:.0f}°"
    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, result.t_ms[-1])

    fig_dash = plot_ring_dashboard(
        result,
        save_path=os.path.join(out_dir, "dashboard.png"),
        time_range=time_range,
        t_offset=t_offset,
        suptitle=suptitle,
        distractor_node=dist_node,
        distractor_angle_deg=dist_center_deg,
        distractor_window=dist_window,
    )
    plt.close(fig_dash)

    ax_metrics = plot_bump_metrics_over_time(
        result,
        time_range=time_range,
        t_offset=t_offset,
    )
    fig_metrics = ax_metrics[0].figure
    fig_metrics.suptitle(f"Bump metrics -- {suptitle}")
    fig_metrics.savefig(
        os.path.join(out_dir, "bump_metrics_over_time.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_metrics)

    fig_pop = plot_population_activity(
        result,
        t_offset=t_offset,
        save_path=os.path.join(out_dir, "population_activity.png"),
        distractor_node=dist_node,
        distractor_window=dist_window,
    )
    plt.close(fig_pop)

    ax_conn = plot_ring_connectome(
        ring_params,
        save_path=os.path.join(out_dir, "connectome.png"),
    )
    plt.close(ax_conn.figure)

    fig_mat = plot_connectivity_matrices(
        ring_params,
        save_path=os.path.join(out_dir, "connectivity_matrices.png"),
    )
    plt.close(fig_mat)

    from ..diagnostic import plot_turing_gain_timecourse
    fig_turing = plot_turing_gain_timecourse(
        result,
        local_params,
        ring_params,
        t_offset=t_offset,
        time_range=time_range,
        save_path=os.path.join(out_dir, "turing_gain_timecourse.png"),
        show=False,
    )
    plt.close(fig_turing)

    if not getattr(args, "no_snapshot_mp4", False):
        anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
        anim_path = os.path.join(out_dir, "snapshot_evolution.mp4")
        try:
            fig_anim, _ = animate_ring_snapshot_evolution(
                result,
                save_path=anim_path,
                time_range=time_range,
                t_offset=t_offset,
                frame_step_ms=args.snapshot_anim_step_ms,
                fps=args.snapshot_anim_fps,
                suptitle=f"{condition.label} -- snapshot evolution",
                show_asymmetry=dist_node is None,  # asymmetry panel only without distractor
                distractor_window=dist_window,
                distractor_angle_deg=dist_center_deg,
                **anim_quality_kwargs,
            )
            plt.close(fig_anim)
        except Exception as exc:
            import traceback
            print(f"Warning: snapshot animation export failed: {exc}")
            traceback.print_exc()

    print(f"\nFigures saved to {out_dir}/")

    # ── Numerical summary JSON ────────────────────────────────────────────────
    import json as _json
    from .analysis import compute_bump_metrics as _compute_bump_metrics
    from ..diagnostic import compute_turing_gain_timecourse

    _bump_metrics = _compute_bump_metrics(result)

    # Baseline PYR rate: mean across all nodes during burn-in (pre-cue)
    _pyr = result.r[:, :, 0]
    _burnin_mask = result.t_ms < result.stim_window[0]
    _baseline_pyr = float(np.mean(_pyr[_burnin_mask])) if np.any(_burnin_mask) else float("nan")

    # Peak PYR rate during cue
    _cue_mask = (result.t_ms >= result.stim_window[0]) & (result.t_ms <= result.stim_window[1])
    _peak_pyr_cue = float(np.max(_pyr[_cue_mask])) if np.any(_cue_mask) else float("nan")

    # Delay-period mean PYR at bump center node vs opposite node
    _delay_mask = result.t_ms > result.stim_window[1]
    _stim_node = int(np.argmin(
        np.abs(np.linspace(0, 360, result.r.shape[1], endpoint=False) - result.stim_angle_deg)
    ))
    _opp_node = (_stim_node + result.r.shape[1] // 2) % result.r.shape[1]
    _delay_pyr_center = float(np.mean(_pyr[_delay_mask, _stim_node])) if np.any(_delay_mask) else float("nan")
    _delay_pyr_opposite = float(np.mean(_pyr[_delay_mask, _opp_node])) if np.any(_delay_mask) else float("nan")

    # Turing gain at bump node during delay period
    _t_ms, _gain_mean, _gain_peak = compute_turing_gain_timecourse(result, local_params, ring_params)
    _delay_mask_turing = _t_ms > result.stim_window[1]
    _delay_turing_bump = float(np.mean(_gain_peak[_delay_mask_turing])) if np.any(_delay_mask_turing) else float("nan")
    _delay_turing_mean = float(np.mean(_gain_mean[_delay_mask_turing])) if np.any(_delay_mask_turing) else float("nan")

    _summary = {
        "params": {
            "w_pyr_pyr_inter": ring_params.w_pyr_pyr_inter,
            "w_pv_global": ring_params.w_pv_global,
            "sigma_pyr_deg": ring_params.sigma_pyr_deg,
            "n_nodes": ring_params.n_nodes,
            "amplitude": amp_factor,
            "condition": condition.name,
        },
        "steady_state": {
            "baseline_pyr_hz": round(_baseline_pyr, 3),
            "peak_pyr_cue_hz": round(_peak_pyr_cue, 3),
            "delay_pyr_center_hz": round(_delay_pyr_center, 3),
            "delay_pyr_opposite_hz": round(_delay_pyr_opposite, 3),
        },
        "turing_gain_delay": {
            "bump_node": round(_delay_turing_bump, 4) if not np.isnan(_delay_turing_bump) else None,
            "mean_background": round(_delay_turing_mean, 4) if not np.isnan(_delay_turing_mean) else None,
        },
        "bump_metrics": {k: (round(v, 4) if not np.isnan(v) else None)
                         for k, v in _bump_metrics.items()},
    }
    _summary_path = os.path.join(out_dir, "run_metrics.json")
    with open(_summary_path, "w") as _f:
        _json.dump(_summary, _f, indent=2)
    print(f"Metrics saved to {_summary_path}")

    # ── Print summary metrics ─────────────────────────────────────────────────
    print("\n" + "─" * 66)
    print("  KEY METRICS SUMMARY")
    print("─" * 66)
    print(f"  Amplitude:                    {amp_factor:.2f}×")
    print(f"  PYR firing rate (bump node):  {_summary['steady_state']['delay_pyr_center_hz']:.3f} Hz")
    print(f"  Turing gain (bump node):      {_summary['turing_gain_delay']['bump_node']}")
    print(f"  Turing gain (background):    {_summary['turing_gain_delay']['mean_background']}")
    print(f"  Bump amplitude:               {_summary['bump_metrics']['amplitude_mean']:.4f}")
    print(f"  Bump width:                   {_summary['bump_metrics']['width_mean_deg']:.2f}°")
    print("─" * 66 + "\n")

    if not args.no_show:
        plt.show()


# ============================================================================
# STUDY SUBCOMMAND
# ============================================================================

def cmd_study(args: argparse.Namespace) -> None:
    """Run multiple conditions and generate comparison plots."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    # Determine conditions first (needed for per-cond param resolution)
    if args.conditions is None:
        condition_keys = list(CONDITION_ORDER)
    else:
        if "all" in args.conditions:
            condition_keys = list(CONDITION_ORDER)
        else:
            condition_keys = args.conditions
            for k in condition_keys:
                if k not in STUDY_CONDITIONS:
                    print(f"Error: unknown condition '{k}'.\n"
                        f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                    sys.exit(1)

    # Per-condition excitation weight
    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp   = {ck: replace(base_rp, w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # for suptitle / _weights_label (shared params)

    # Per-condition or sweep amplitudes
    if args.amplitudes is not None:
        # Amplitude sweep applied uniformly to all conditions
        amplitudes = list(args.amplitudes)
        cond_amp: dict[str, float] | None = None
    else:
        # Per-condition (or shared) base amplitude
        cond_amp = _resolve_per_cond_param(args.amplitude, condition_keys, 'amplitude')
        amplitudes = [args.amplitude[0]]  # used for timing; one loop iteration
    n_trials = getattr(args, 'n_trials', 1)
    n_workers = _resolve_workers(args)
    no_cache = getattr(args, 'no_cache', False)
    error_band = getattr(args, 'error_band', 'sem')

    # Legend labels (annotated when per-cond params differ)
    cond_labels = _build_cond_labels(condition_keys, cond_excit, cond_amp)

    conn_label = _calibration_network_label(base_rp)
    out_dir = os.path.join(
        _output_dir("figs/ring/run", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "study_metrics.csv")

    # Compute T_ms using first amplitude (timing is same for all amplitudes)
    _, _, T_ms_full, _, _, _ = _build_common(args, amp_factor=args.amplitude[0])
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS

    _print_config(args, args.amplitude[0], base_params, T_ms_full, base_rp,
                  experiment_info=[
                      f"Conditions:  {', '.join(condition_keys)}",
                      f"Excit/cond:  {', '.join(f'{ck}={_fmt(cond_excit[ck])}' for ck in condition_keys)}",
                      f"Amp/cond:    {', '.join(f'{ck}={_fmt(cond_amp[ck])}' for ck in condition_keys) if cond_amp else ', '.join(_fmt(a) for a in amplitudes) + '× I_ext_pyr'}",
                      f"Delay:       {args.delay_ms:.0f} ms",
                      f"Trials:      {n_trials}   seed={args.seed}   workers={n_workers}",
                  ],
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    # --- Burn-in states (once per condition, using per-condition ring_params) ---
    print("\nComputing burn-in states...")
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond_key in tqdm(condition_keys, desc="Burn-in", unit="cond"):
        condition = STUDY_CONDITIONS[cond_key]
        local_params = apply_condition(base_params, condition)
        burnin_states[cond_key] = _compute_burnin_state(
            local_params, per_cond_rp[cond_key], per_cond_conn[cond_key], seed=args.seed,
        )

    # --- Trial seeds ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

    # --- Delay evaluation times ---
    delay_eval_times, delay_labels = _compute_delay_eval_times(
        args, stim_offset_ms, T_ms_full,
    )

    # --- CSV cache ---
    if no_cache and os.path.exists(csv_path):
        os.remove(csv_path)
        completed = set()
    else:
        completed = _load_cached_metrics(csv_path, expected_eval_times=delay_eval_times)

    # --- Build jobs ---
    jobs = []
    for cond_key in condition_keys:
        amps_for_cond = [cond_amp[cond_key]] if cond_amp else amplitudes
        for amp in amps_for_cond:
            for trial_idx, seed in enumerate(trial_seeds):
                if (cond_key, amp, trial_idx) not in completed:
                    jobs.append((cond_key, amp, trial_idx, seed))

    if cond_amp:
        total_jobs = len(condition_keys) * n_trials
    else:
        total_jobs = len(condition_keys) * len(amplitudes) * n_trials
    cached_jobs = total_jobs - len(jobs)
    print(f"\nJobs: {len(jobs)} to run, {cached_jobs} cached")

    # --- Run simulations ---
    all_results: list[dict] = []

    if jobs:
        args_dict = _args_to_dict(args)
        init_args = (
            args_dict, base_params, per_cond_rp, per_cond_conn,
            burnin_states, delay_eval_times, T_ms_full,
        )

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT, 
                max_workers=n_workers,
                initializer=_ring_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_ring_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        res = future.result()
                        all_results.append(res)
                        _append_metrics_to_csv(csv_path, _job_result_to_csv_rows(res))
                        pbar.update()
        else:
            # Sequential fallback
            _ring_init_worker(*init_args)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                res = _ring_run_single(job)
                all_results.append(res)
                _append_metrics_to_csv(csv_path, _job_result_to_csv_rows(res))

    # --- Load all cached data too (for aggregation) ---
    if cached_jobs > 0:
        csv_rows = _load_all_metrics(csv_path)
        from collections import defaultdict
        grouped: dict[tuple, dict] = defaultdict(lambda: {
            'delay_metrics': [], 'full_delay_metrics': None,
        })
        for row in csv_rows:
            key = (row['condition_key'], float(row['amplitude']), int(row['trial_idx']))
            if any(r['cond_key'] == key[0] and r['amplitude'] == key[1]
                   and r['trial_idx'] == key[2] for r in all_results):
                continue
            metrics = {k: float(row.get(k, 'nan')) for k in _METRIC_KEYS}
            if row['eval_time_ms'] == 'full_delay':
                grouped[key]['full_delay_metrics'] = metrics
            else:
                metrics['eval_time_ms'] = float(row['eval_time_ms'])
                grouped[key]['delay_metrics'].append(metrics)

        for (ck, amp, ti), data in grouped.items():
            if data['full_delay_metrics'] is not None:
                all_results.append({
                    'cond_key': ck, 'amplitude': amp, 'trial_idx': ti,
                    'seed': 0,
                    'delay_metrics': sorted(data['delay_metrics'],
                                            key=lambda m: m['eval_time_ms']),
                    'full_delay_metrics': data['full_delay_metrics'],
                    'comparison_data': None,
                })

    # --- Aggregate and plot ---
    all_delay_metrics_agg: dict[float, dict[str, dict]] = {}
    export_mp4 = not getattr(args, "no_snapshot_mp4", False)
    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
    mp4_pbar = None
    if export_mp4:
        total_videos = len(amplitudes) * len(condition_keys)
        mp4_pbar = _start_mp4_progress(
            total_videos=total_videos,
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
            sample_time_range=(BURN_IN_MS, T_ms_full),
        )
    mp4_pbar = None
    if export_mp4:
        total_videos = len(amplitudes) * len(condition_keys)
        mp4_pbar = _start_mp4_progress(
            total_videos=total_videos,
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
            sample_time_range=(BURN_IN_MS, T_ms_full),
        )

    try:
        for amp in amplitudes:
            amp_out = os.path.join(out_dir, f"amp{_fmt(amp)}")
            os.makedirs(amp_out, exist_ok=True)
            suptitle = f"{_stim_label(amp)}, {_weights_label(base_rp)}"

            metrics_over_delay_agg: dict[str, list[dict]] = {}
            delay_end_metrics_agg: dict[str, dict] = {}
            comparison_data: dict[str, dict] = {}

            for cond_key in condition_keys:
                ck_amp = cond_amp[cond_key] if cond_amp else amp
                trial_results = [
                    r for r in all_results
                    if r['cond_key'] == cond_key and abs(r['amplitude'] - ck_amp) < 1e-9
                ]
                if not trial_results:
                    continue

                trial_delay = [r['delay_metrics'] for r in trial_results]
                trial_full = [r['full_delay_metrics'] for r in trial_results]

                if n_trials > 1 and len(trial_results) > 1:
                    metrics_over_delay_agg[cond_key] = aggregate_metrics_across_trials(trial_delay)
                    delay_end_metrics_agg[cond_key] = aggregate_single_metrics(trial_full)
                else:
                    metrics_over_delay_agg[cond_key] = trial_delay[0]
                    delay_end_metrics_agg[cond_key] = trial_full[0]

                for r in trial_results:
                    if r['trial_idx'] == 0 and r.get('comparison_data') is not None:
                        comparison_data[cond_key] = r['comparison_data']
                        break

            if delay_eval_times and metrics_over_delay_agg and len(delay_labels) > 1:
                band_tag = f", {n_trials} trials, ±{error_band.upper()}" if n_trials > 1 else ""
                # Skip first point — bump is still forming at the earliest eval time
                plot_metrics_vs_delay(
                    {ck: v[1:] for ck, v in metrics_over_delay_agg.items()},
                    delay_labels=delay_labels[1:],
                    save_path=os.path.join(amp_out, f"metrics_vs_delay_{error_band}.png"),
                    suptitle=f"Bump Metrics During Delay  ({suptitle}{band_tag})",
                    error_band=error_band,
                    separate_app=False,  # all conditions on same plot for delay time course
                    cond_labels=cond_labels,
                )
                plt.close()

            if comparison_data:
                plot_bump_metrics_comparison(
                    comparison_data,
                    save_path=os.path.join(amp_out, "bump_metrics_comparison.png"),
                    suptitle=f"Bump Metrics Comparison  ({suptitle})",
                    cond_labels=cond_labels,
                )
                plt.close()

            # --- Per-amplitude firing rate violin plots ---
            _all_rate_pops = _RATE_POPS + _CUE_RATE_POPS
            rate_by_cond: dict[str, dict[str, np.ndarray]] = {}
            for cond_key in condition_keys:
                trial_full = [
                    r['full_delay_metrics'] for r in all_results
                    if r['cond_key'] == cond_key and r['amplitude'] == amp
                ]
                if trial_full:
                    rate_by_cond[cond_key] = {
                        mk: np.array([m.get(mk, np.nan) for m in trial_full])
                        for mk, *_ in _all_rate_pops
                    }

            if rate_by_cond:
                import scipy.stats as _scipy_stats_study
                rate_stats_rows: list[dict] = []
                for _i, _ca in enumerate(condition_keys):
                    for _j, _cb in enumerate(condition_keys):
                        if _j <= _i:
                            continue
                        for mk, *_ in _all_rate_pops:
                            arr_a = rate_by_cond.get(_ca, {}).get(mk, np.array([]))
                            arr_b = rate_by_cond.get(_cb, {}).get(mk, np.array([]))
                            a_v = arr_a[np.isfinite(arr_a)]
                            b_v = arr_b[np.isfinite(arr_b)]
                            if len(a_v) > 0 and len(b_v) > 0:
                                _u, _p = _scipy_stats_study.mannwhitneyu(
                                    a_v, b_v, alternative='two-sided'
                                )
                                rate_stats_rows.append({
                                    'metric': mk, 'cond_a': _ca, 'cond_b': _cb,
                                    'u_stat': float(_u), 'p_value': float(_p),
                                })
                if rate_stats_rows:
                    from scipy.stats import false_discovery_control as _fdr_study
                    _q_vals = _fdr_study(
                        [r['p_value'] for r in rate_stats_rows], method='bh'
                    )
                    for _r, _q in zip(rate_stats_rows, _q_vals):
                        _r['q_value'] = float(_q)

                for _pop_list, _fname, _title in [
                    (_RATE_POPS, "firing_rates_all_violin.png",
                     f"Population Firing Rates — All Nodes  ({suptitle})"),
                    (_CUE_RATE_POPS, "firing_rates_cue_violin.png",
                     f"Population Firing Rates — Cue Node  ({suptitle})"),
                ]:
                    _panels = [
                        (mk, lbl, 'Mean firing rate (Hz)', {
                            ck: rate_by_cond.get(ck, {}).get(mk, np.array([]))
                            for ck in condition_keys
                        })
                        for mk, lbl, _ in _pop_list
                    ]
                    plot_study_firing_rates_violin(
                        panels=_panels,
                        cond_order=condition_keys,
                        stats_rows=rate_stats_rows,
                        suptitle=_title,
                        save_path=os.path.join(amp_out, _fname),
                        cond_labels=cond_labels,
                    )
                    plt.close()

            # --- Per-amplitude interneuron/PYR ratio violin plots ---
            _RATIO_DEFS = [
                # (prefix, som_key, pv_key, vip_key, pyr_key, fname, title)
                ('mean', 'mean_rate_som_hz', 'mean_rate_pv_hz', 'mean_rate_vip_hz',
                 'mean_rate_pyr_hz', 'interneuron_ratios_all_violin.png',
                 f"Interneuron/PYR Firing Rate Ratio — All Nodes  ({suptitle})"),
                ('cue', 'cue_rate_som_hz', 'cue_rate_pv_hz', 'cue_rate_vip_hz',
                 'cue_rate_pyr_hz', 'interneuron_ratios_cue_violin.png',
                 f"Interneuron/PYR Firing Rate Ratio — Cue Node  ({suptitle})"),
            ]
            if rate_by_cond:
                for _prefix, _som_k, _pv_k, _vip_k, _pyr_k, _rfname, _rtitle in _RATIO_DEFS:
                    ratio_by_cond: dict[str, dict[str, np.ndarray]] = {}
                    for ck in condition_keys:
                        _d = rate_by_cond.get(ck, {})
                        _pyr = _d.get(_pyr_k, np.array([]))
                        _nonzero = _pyr != 0
                        ratio_by_cond[ck] = {
                            'som_pyr': np.where(_nonzero, _d.get(_som_k, np.full_like(_pyr, np.nan)) / _pyr, np.nan),
                            'pv_pyr':  np.where(_nonzero, _d.get(_pv_k,  np.full_like(_pyr, np.nan)) / _pyr, np.nan),
                            'vip_pyr': np.where(_nonzero, _d.get(_vip_k, np.full_like(_pyr, np.nan)) / _pyr, np.nan),
                        }

                    ratio_stats_rows: list[dict] = []
                    for _i, _ca in enumerate(condition_keys):
                        for _j, _cb in enumerate(condition_keys):
                            if _j <= _i:
                                continue
                            for _rk in ('som_pyr', 'pv_pyr', 'vip_pyr'):
                                arr_a = ratio_by_cond.get(_ca, {}).get(_rk, np.array([]))
                                arr_b = ratio_by_cond.get(_cb, {}).get(_rk, np.array([]))
                                a_v = arr_a[np.isfinite(arr_a)]
                                b_v = arr_b[np.isfinite(arr_b)]
                                if len(a_v) > 0 and len(b_v) > 0:
                                    _u, _p = _scipy_stats_study.mannwhitneyu(
                                        a_v, b_v, alternative='two-sided'
                                    )
                                    ratio_stats_rows.append({
                                        'metric': _rk, 'cond_a': _ca, 'cond_b': _cb,
                                        'u_stat': float(_u), 'p_value': float(_p),
                                    })
                    if ratio_stats_rows:
                        _q_r = _fdr_study([r['p_value'] for r in ratio_stats_rows], method='bh')
                        for _r, _q in zip(ratio_stats_rows, _q_r):
                            _r['q_value'] = float(_q)

                    ratio_panels = [
                        (_rk, lbl, 'Rate ratio (relative to PYR)', {
                            ck: ratio_by_cond.get(ck, {}).get(_rk, np.array([]))
                            for ck in condition_keys
                        })
                        for _rk, lbl in [('som_pyr', 'SOM/PYR'), ('pv_pyr', 'PV/PYR'), ('vip_pyr', 'VIP/PYR')]
                    ]
                    plot_study_firing_rates_violin(
                        panels=ratio_panels,
                        cond_order=condition_keys,
                        stats_rows=ratio_stats_rows,
                        suptitle=_rtitle,
                        save_path=os.path.join(amp_out, _rfname),
                    )
                    plt.close()

            if export_mp4:
                anim_dir = os.path.join(amp_out, "snapshot_evolution")
                os.makedirs(anim_dir, exist_ok=True)
                for cond_key in condition_keys:
                    mp4_pbar.set_postfix_str(f"amp={_fmt(amp)} cond={cond_key}")
                    condition = STUDY_CONDITIONS[cond_key]
                    local_params = apply_condition(base_params, condition)
                    delay_end_ms = stim_offset_ms + args.delay_ms
                    local_params = _apply_response_transient(local_params, args, delay_end_ms)
                    ck_amp_vis = cond_amp[cond_key] if cond_amp else amp
                    cue_current = ck_amp_vis * base_params.I_ext_pyr()
                    stimuli = [
                        RingStimulus(
                            center_deg=STIM_CENTER_DEG,
                            amplitude=cue_current,
                            sigma_deg=STIM_SIGMA_DEG,
                            onset_ms=STIM_ONSET_MS,
                            duration_ms=STIM_DURATION_MS,
                        )
                    ]
                    vis_seed = trial_seeds[0] if trial_seeds else args.seed
                    vis_result = simulate_ring(
                        local_params,
                        per_cond_rp[cond_key],
                        T_ms=T_ms_full,
                        stimuli=stimuli,
                        seed=vis_seed,
                        connectivity=per_cond_conn[cond_key],
                        record_dt_ms=args.record_dt_ms,
                    )
                    anim_path = os.path.join(anim_dir, f"{cond_key}.mp4")
                    fig_anim, _ = animate_ring_snapshot_evolution(
                        vis_result,
                        save_path=anim_path,
                        time_range=(BURN_IN_MS, T_ms_full),
                        t_offset=BURN_IN_MS,
                        frame_step_ms=args.snapshot_anim_step_ms,
                        fps=args.snapshot_anim_fps,
                        suptitle=f"{condition.label} — Snapshot Evolution ({suptitle})",
                        show_asymmetry=True,
                        **anim_quality_kwargs,
                    )
                    plt.close(fig_anim)
                    mp4_pbar.update(1)

            all_delay_metrics_agg[amp] = delay_end_metrics_agg
    finally:
        if mp4_pbar is not None:
            mp4_pbar.close()
        if mp4_pbar is not None:
            mp4_pbar.close()

    # Cross-amplitude comparison (full delay)
    if len(amplitudes) > 1:
        band_tag = f"  ({n_trials} trials, ±{error_band.upper()})" if n_trials > 1 else ""
        plot_metrics_vs_amplitude(
            all_delay_metrics_agg,
            amplitude_values=amplitudes,
            save_path=os.path.join(out_dir, f"metrics_vs_amplitude_{error_band}.png"),
            suptitle=f"Metrics vs Amplitude (full delay){band_tag}  [{_weights_label(base_rp)}]",
            error_band=error_band,
            separate_app=False,  # all conditions on same plot for amplitude comparison
            cond_labels=cond_labels,
        )
        plt.close()

        # Firing rate evolution over amplitude
        import scipy.stats as _scipy_stats_sweep
        _all_rate_pops_sweep = _RATE_POPS + _CUE_RATE_POPS
        rate_sweep: dict[str, dict[str, dict[float, np.ndarray]]] = {}
        for mk, *_ in _all_rate_pops_sweep:
            by_cond_amp: dict[str, dict[float, np.ndarray]] = {}
            for ck in condition_keys:
                by_cond_amp[ck] = {}
                for _amp in amplitudes:
                    trial_full = [
                        r['full_delay_metrics'] for r in all_results
                        if r['cond_key'] == ck and r['amplitude'] == _amp
                    ]
                    vals = np.array([m.get(mk, np.nan) for m in trial_full])
                    by_cond_amp[ck][_amp] = vals[np.isfinite(vals)]
            rate_sweep[mk] = by_cond_amp

        sweep_stats_rows: list[dict] = []
        for mk, *_ in _all_rate_pops_sweep:
            for _amp in amplitudes:
                for _i, _ca in enumerate(condition_keys):
                    for _j, _cb in enumerate(condition_keys):
                        if _j <= _i:
                            continue
                        arr_a = rate_sweep[mk].get(_ca, {}).get(_amp, np.array([]))
                        arr_b = rate_sweep[mk].get(_cb, {}).get(_amp, np.array([]))
                        if len(arr_a) > 0 and len(arr_b) > 0:
                            _u, _p = _scipy_stats_sweep.mannwhitneyu(
                                arr_a, arr_b, alternative='two-sided'
                            )
                            sweep_stats_rows.append({
                                'metric': mk, 'amp': _amp,
                                'cond_a': _ca, 'cond_b': _cb,
                                'p_value': float(_p),
                            })
        if sweep_stats_rows:
            from scipy.stats import false_discovery_control as _fdr_sweep
            _q_vals_sw = _fdr_sweep(
                [r['p_value'] for r in sweep_stats_rows], method='bh'
            )
            for _r, _q in zip(sweep_stats_rows, _q_vals_sw):
                _r['q_value'] = float(_q)

        for _pop_list, _fname, _title in [
            (_RATE_POPS, "firing_rates_all_vs_amplitude.png",
             f"Firing Rates vs Amplitude — All Nodes  [{_weights_label(ring_params)}]"),
            (_CUE_RATE_POPS, "firing_rates_cue_vs_amplitude.png",
             f"Firing Rates vs Amplitude — Cue Node  [{_weights_label(ring_params)}]"),
        ]:
            _sweep_panels = [
                (lbl, 'Mean firing rate (Hz)', rate_sweep[mk])
                for mk, lbl, _ in _pop_list
            ]
            _stats_per_panel = [
                [
                    {'amp': r['amp'], 'q_value': r['q_value'],
                     'cond_a': r['cond_a'], 'cond_b': r['cond_b']}
                    for r in sweep_stats_rows if r['metric'] == mk
                ]
                for mk, *_ in _pop_list
            ]
            plot_oscillation_amp_sweep_lines(
                panels=_sweep_panels,
                amplitudes=amplitudes,
                cond_order=condition_keys,
                stats_per_panel=_stats_per_panel,
                suptitle=_title,
                save_path=os.path.join(out_dir, _fname),
            )
            plt.close()

        # Interneuron/PYR ratio amplitude sweep
        _RATIO_SWEEP_DEFS = [
            ('mean_rate_som_hz', 'mean_rate_pv_hz', 'mean_rate_vip_hz', 'mean_rate_pyr_hz',
             'interneuron_ratios_all_vs_amplitude.png',
             f"Interneuron/PYR Ratio vs Amplitude — All Nodes  [{_weights_label(ring_params)}]"),
            ('cue_rate_som_hz', 'cue_rate_pv_hz', 'cue_rate_vip_hz', 'cue_rate_pyr_hz',
             'interneuron_ratios_cue_vs_amplitude.png',
             f"Interneuron/PYR Ratio vs Amplitude — Cue Node  [{_weights_label(ring_params)}]"),
        ]
        for _som_k, _pv_k, _vip_k, _pyr_k, _rfname, _rtitle in _RATIO_SWEEP_DEFS:
            ratio_sweep: dict[str, dict[str, dict[float, np.ndarray]]] = {}
            for _rk, _num_k in [('som_pyr', _som_k), ('pv_pyr', _pv_k), ('vip_pyr', _vip_k)]:
                by_cond_amp: dict[str, dict[float, np.ndarray]] = {}
                for ck in condition_keys:
                    by_cond_amp[ck] = {}
                    for _amp in amplitudes:
                        trial_full = [
                            r['full_delay_metrics'] for r in all_results
                            if r['cond_key'] == ck and r['amplitude'] == _amp
                        ]
                        _pyr = np.array([m.get(_pyr_k, np.nan) for m in trial_full])
                        _num = np.array([m.get(_num_k, np.nan) for m in trial_full])
                        with np.errstate(invalid='ignore', divide='ignore'):
                            _ratio = np.where(_pyr != 0, _num / _pyr, np.nan)
                        by_cond_amp[ck][_amp] = _ratio[np.isfinite(_ratio)]
                ratio_sweep[_rk] = by_cond_amp

            ratio_sweep_stats: list[dict] = []
            for _rk in ('som_pyr', 'pv_pyr', 'vip_pyr'):
                for _amp in amplitudes:
                    for _i, _ca in enumerate(condition_keys):
                        for _j, _cb in enumerate(condition_keys):
                            if _j <= _i:
                                continue
                            arr_a = ratio_sweep[_rk].get(_ca, {}).get(_amp, np.array([]))
                            arr_b = ratio_sweep[_rk].get(_cb, {}).get(_amp, np.array([]))
                            if len(arr_a) > 0 and len(arr_b) > 0:
                                _u, _p = _scipy_stats_sweep.mannwhitneyu(
                                    arr_a, arr_b, alternative='two-sided'
                                )
                                ratio_sweep_stats.append({
                                    'metric': _rk, 'amp': _amp,
                                    'cond_a': _ca, 'cond_b': _cb,
                                    'p_value': float(_p),
                                })
            if ratio_sweep_stats:
                _q_rs = _fdr_sweep([r['p_value'] for r in ratio_sweep_stats], method='bh')
                for _r, _q in zip(ratio_sweep_stats, _q_rs):
                    _r['q_value'] = float(_q)

            _ratio_panels = [
                (lbl, 'Rate ratio (relative to PYR)', ratio_sweep[_rk])
                for _rk, lbl in [('som_pyr', 'SOM/PYR'), ('pv_pyr', 'PV/PYR'), ('vip_pyr', 'VIP/PYR')]
            ]
            _ratio_stats_per_panel = [
                [
                    {'amp': r['amp'], 'q_value': r['q_value'],
                     'cond_a': r['cond_a'], 'cond_b': r['cond_b']}
                    for r in ratio_sweep_stats if r['metric'] == _rk
                ]
                for _rk in ('som_pyr', 'pv_pyr', 'vip_pyr')
            ]
            plot_oscillation_amp_sweep_lines(
                panels=_ratio_panels,
                amplitudes=amplitudes,
                cond_order=condition_keys,
                stats_per_panel=_ratio_stats_per_panel,
                suptitle=_rtitle,
                save_path=os.path.join(out_dir, _rfname),
            )
            plt.close()

    # Timed metrics-vs-amplitude plots (at different delay offsets)
    amp_eval_step_ms = getattr(args, 'amp_eval_step_ms', 500.0)
    if len(amplitudes) > 1 and amp_eval_step_ms > 0 and delay_eval_times:
        from collections import defaultdict as _defaultdict

        # Collect available eval times
        available_eval_times = set()
        for r in all_results:
            for m in r['delay_metrics']:
                available_eval_times.add(m['eval_time_ms'])
        available_eval_times = sorted(available_eval_times)

        # Select target offsets at the requested step
        target_offsets = []
        t = amp_eval_step_ms
        while t <= args.delay_ms:
            target_offsets.append(t)
            t += amp_eval_step_ms

        # Map each target to nearest available eval time
        selected = []  # list of (eval_time_abs, offset_ms, label)
        seen_eval_times = set()
        for offset in target_offsets:
            target_abs = stim_offset_ms + offset
            if available_eval_times:
                nearest = min(available_eval_times, key=lambda et: abs(et - target_abs))
                if nearest not in seen_eval_times:
                    seen_eval_times.add(nearest)
                    selected.append((nearest, offset, f"{offset/1000:.1f}s"))

        # Generate one plot per selected time point
        for eval_time, offset, label in selected:
            timed_metrics: dict[float, dict[str, dict]] = {}
            for amp in amplitudes:
                timed_metrics[amp] = {}
                for cond_key in condition_keys:
                    trial_results = [
                        r for r in all_results
                        if r['cond_key'] == cond_key and r['amplitude'] == amp
                    ]
                    if not trial_results:
                        continue
                    # Extract the matching eval_time metric from each trial
                    trial_at_time = []
                    for r in trial_results:
                        for m in r['delay_metrics']:
                            if m['eval_time_ms'] == eval_time:
                                trial_at_time.append(m)
                                break
                    if not trial_at_time:
                        continue
                    if n_trials > 1 and len(trial_at_time) > 1:
                        timed_metrics[amp][cond_key] = aggregate_single_metrics(trial_at_time)
                    else:
                        timed_metrics[amp][cond_key] = trial_at_time[0]

            plot_metrics_vs_amplitude(
                timed_metrics,
                amplitude_values=amplitudes,
                save_path=os.path.join(out_dir, f"metrics_vs_amplitude_at_{label}_{error_band}.png"),
                suptitle=f"Metrics vs Amplitude at delay = {label}{band_tag}  [{_weights_label(ring_params)}]",
                error_band=error_band,
                separate_app=False,  # all conditions on same plot for amplitude comparison
            )
            plt.close()

    # Connectome (once)
    plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))
    plt.close()

    print(f"\nFigures saved to {out_dir}/")
    print(f"Metrics cached in {csv_path}")


# ============================================================================
# DIFFUSION: PARALLEL WORKER
# ============================================================================

_diffusion_sim_args: Optional[dict] = None


def _diffusion_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    T_ms_full: float,
):
    """Initialize worker process for diffusion analysis."""
    global _diffusion_sim_args
    _diffusion_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
        'burnin_states': burnin_states,
        'T_ms_full': T_ms_full,
    }


def _diffusion_run_single(job: tuple) -> dict:
    """Run a single diffusion trial.  Returns decoded bump center trajectory."""
    global _diffusion_sim_args
    cfg = _diffusion_sim_args
    cond_key, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['ring_params']
    connectivity = cfg['connectivity']
    T_ms_full = cfg['T_ms_full']

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    amp_factor = args_d['amplitude']
    actual_current = amp_factor * base_params.I_ext_pyr()

    T_ms_short = T_ms_full - BURN_IN_MS
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
        seed=seed, connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    # Shift time back to absolute
    result.t_ms += BURN_IN_MS

    # Extract delay period trajectory
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_start_ms = stim_offset_ms + TRANSIENT_SKIP_TIME_MS
    delay_end_ms = stim_offset_ms + args_d['delay_ms']

    mask = (result.t_ms >= delay_start_ms) & (result.t_ms <= delay_end_ms)
    t_delay = result.t_ms[mask]
    activity_delay = result.r[mask, :, 0]  # PYR activity

    center_rad, amplitude = population_vector_decode(
        activity_delay, ring_params.node_angles_rad,
    )
    center_unwrapped = np.unwrap(center_rad)

    t_delay_s = (t_delay - t_delay[0]) / 1000.0  # seconds, starting from 0

    return {
        'cond_key': cond_key,
        'trial_idx': trial_idx,
        'center_unwrapped_rad': center_unwrapped,
        'amplitude': amplitude,
        't_delay_s': t_delay_s,
        # Snapshots of PYR population activity at start and end of delay
        'activity_start': activity_delay[0].copy(),   # shape (n_nodes,)
        'activity_end': activity_delay[-1].copy(),    # shape (n_nodes,)
    }


# ============================================================================
# DIFFUSION SUBCOMMAND
# ============================================================================

def cmd_diffusion(args: argparse.Namespace) -> None:
    """Run diffusion (MSD) analysis across conditions."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    if args.conditions is None:
            condition_keys = list(CONDITION_ORDER)
    else:
        if "all" in args.conditions:
            condition_keys = list(STUDY_CONDITIONS.keys())
        else:
            condition_keys = args.conditions
            for k in condition_keys:
                if k not in STUDY_CONDITIONS:
                    print(f"Error: unknown condition '{k}'.\n"
                        f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                    sys.exit(1)

    n_trials = args.n_trials
    n_workers = _resolve_workers(args)

    _, _, T_ms_full, _, amp_factor, _ = _build_common(args)

    conn_label = _network_label(ring_params)
    amp_label = f"amp{_fmt(amp_factor)}"
    out_dir = os.path.join(
        _output_dir("figs/ring/diffusion", args.params_json),
        conn_label,
        amp_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    _print_config(args, amp_factor, base_params, T_ms_full, ring_params,
                  experiment_info=[
                      f"Conditions:  {', '.join(condition_keys)}",
                      f"Delay:       {args.delay_ms:.0f} ms",
                      f"Trials:      {n_trials}   seed={args.seed}   workers={n_workers}",
                  ],
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    # --- Pre-compute connectivity and burn-in ---
    connectivity = RingConnectivity.from_params(ring_params)

    print("\nComputing burn-in states...")
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond_key in tqdm(condition_keys, desc="Burn-in", unit="cond"):
        condition = STUDY_CONDITIONS[cond_key]
        local_params = apply_condition(base_params, condition)
        burnin_states[cond_key] = _compute_burnin_state(
            local_params, ring_params, connectivity, seed=args.seed,
        )

    # --- Trial seeds ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

    # --- Output paths (defined early for cache check) ---
    summary_csv = os.path.join(out_dir, "diffusion_displacement_summary.csv")
    trials_csv = os.path.join(out_dir, "diffusion_displacement_trials.csv")
    amplitude_csv = os.path.join(out_dir, "diffusion_amplitude.csv")

    # --- Check for cached displacement data ---
    disp_data: dict[str, dict] = {}
    loaded_from_cache = False

    if os.path.exists(summary_csv) and os.path.exists(trials_csv):
        try:
            with open(summary_csv, newline='') as _f:
                summary_rows = list(csv.DictReader(_f))
            cond_keys_set = set(condition_keys)
            cached_conds = {r['condition_key'] for r in summary_rows}
            params_ok = cond_keys_set <= cached_conds and all(
                float(r['delay_ms']) == args.delay_ms
                and float(r['amplitude_factor']) == amp_factor
                and int(r['n_trials']) >= n_trials
                and int(r['seed']) == args.seed
                for r in summary_rows if r['condition_key'] in cond_keys_set
            )
            if params_ok:
                with open(trials_csv, newline='') as _f:
                    trial_rows = list(csv.DictReader(_f))
                trials_by_cond: dict[str, list] = {}
                for row in trial_rows:
                    trials_by_cond.setdefault(row['condition_key'], []).append(row)
                if cond_keys_set <= set(trials_by_cond.keys()):
                    print(f"\nLoading cached displacement data from {trials_csv}")
                    sr_by_cond = {r['condition_key']: r for r in summary_rows}
                    for ck in condition_keys:
                        sr = sr_by_cond[ck]
                        disps = np.array([
                            float(r['displacement_deg'])
                            for r in trials_by_cond[ck]
                            if r.get('valid', '1') == '1'
                        ])
                        cond_label = STUDY_CONDITIONS[ck].name
                        print(f"  {cond_label}: mean |shift| = "
                              f"{float(sr['abs_mean_deg']):.2f}°  "
                              f"(n={sr['n_valid']}/{sr['n_total']})")
                        disp_data[ck] = {
                            'displacements_deg': disps,
                            'mean_deg': float(sr['mean_deg']),
                            'std_deg': float(sr['std_deg']),
                            'abs_mean_deg': float(sr['abs_mean_deg']),
                            'n_valid': int(sr['n_valid']),
                            'n_total': int(sr['n_total']),
                            'amplitude_factor': float(sr['amplitude_factor']),
                            'stim_current': float(sr['amplitude_factor']) * base_params.I_ext_pyr(),
                            # Activity snapshots not available from cache
                            'snap_activity_start': None,
                            'snap_activity_end': None,
                            'snap_angles_deg': None,
                            'snap_displacement_deg': None,
                        }
                    loaded_from_cache = True

                    # Also load amplitude data if available
                    if os.path.exists(amplitude_csv):
                        try:
                            with open(amplitude_csv, newline='') as _fa:
                                amp_rows = list(csv.DictReader(_fa))
                            amp_by_cond: dict[str, list] = {}
                            for row in amp_rows:
                                amp_by_cond.setdefault(row['condition_key'], []).append(row)
                            if cond_keys_set <= set(amp_by_cond.keys()):
                                for ck in condition_keys:
                                    rows_a = sorted(
                                        amp_by_cond[ck], key=lambda r: float(r['t_s'])
                                    )
                                    nt_str = rows_a[0].get('noise_threshold', '')
                                    disp_data[ck]['amp_t_s'] = np.array(
                                        [float(r['t_s']) for r in rows_a]
                                    )
                                    disp_data[ck]['amp_mean'] = np.array(
                                        [float(r['amp_mean']) for r in rows_a]
                                    )
                                    disp_data[ck]['amp_sem'] = np.array(
                                        [float(r['amp_sem']) for r in rows_a]
                                    )
                                    disp_data[ck]['survival'] = np.array(
                                        [float(r['survival_frac']) for r in rows_a]
                                    )
                                    disp_data[ck]['noise_threshold'] = (
                                        float(nt_str) if nt_str else None
                                    )
                                print(f"  Loaded cached amplitude data from {amplitude_csv}")
                        except Exception as _ea:
                            print(f"  Amplitude cache read failed ({_ea}), skipping.")

                    # Re-run one sample trial per condition for ring snapshot visualization
                    print("  Re-running sample trials for ring snapshot visualization...")
                    rng_snapshot = np.random.default_rng(args.seed)
                    stim_offset_ms_local = STIM_ONSET_MS + STIM_DURATION_MS
                    for ck in condition_keys:
                        valid_rows = [
                            r for r in trials_by_cond.get(ck, [])
                            if r.get('valid', '1') == '1'
                        ]
                        if not valid_rows:
                            disp_data[ck]['sample_result'] = None
                            disp_data[ck]['sample_displacement_deg'] = None
                            continue
                        sample_row = valid_rows[int(rng_snapshot.integers(len(valid_rows)))]
                        sample_seed = trial_seeds[int(sample_row['trial_idx'])]
                        local_params = apply_condition(base_params, STUDY_CONDITIONS[ck])
                        r0, I_adapt0 = burnin_states[ck]
                        actual_current = amp_factor * base_params.I_ext_pyr()
                        T_ms_short = T_ms_full - BURN_IN_MS
                        stimuli_short = [
                            RingStimulus(
                                center_deg=STIM_CENTER_DEG, amplitude=actual_current,
                                sigma_deg=STIM_SIGMA_DEG,
                                onset_ms=STIM_ONSET_MS - BURN_IN_MS,
                                duration_ms=STIM_DURATION_MS,
                            ),
                        ]
                        sample_result = simulate_ring(
                            local_params, ring_params, T_ms=T_ms_short,
                            stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
                            seed=sample_seed, connectivity=connectivity,
                            record_dt_ms=5.0,
                        )
                        sample_result.t_ms += BURN_IN_MS
                        disp_data[ck]['sample_result'] = sample_result
                        disp_data[ck]['sample_displacement_deg'] = float(sample_row['displacement_deg'])
                        disp_data[ck]['delay_start_ms'] = stim_offset_ms_local + TRANSIENT_SKIP_TIME_MS
                        disp_data[ck]['delay_end_ms'] = stim_offset_ms_local + args.delay_ms
        except Exception as _e:
            print(f"  Cache read failed ({_e}), rerunning simulations.")
            disp_data = {}

    if not loaded_from_cache:
        # --- Build jobs ---
        jobs = []
        for cond_key in condition_keys:
            for trial_idx, seed in enumerate(trial_seeds):
                jobs.append((cond_key, trial_idx, seed))

        # --- Run simulations ---
        args_dict = {
            **_args_to_dict(args),
            'amplitude': amp_factor,
        }
        init_args = (
            args_dict, base_params, ring_params, connectivity,
            burnin_states, T_ms_full,
        )

        all_results: list[dict] = []
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT, 
                max_workers=n_workers,
                initializer=_diffusion_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_diffusion_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Diffusion trials", unit="trial", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _diffusion_init_worker(*init_args)
            for job in tqdm(jobs, desc="Diffusion trials", unit="trial"):
                all_results.append(_diffusion_run_single(job))

        # --- Auto-detect noise threshold from calibration ---
        cal_conn_label = _calibration_network_label(ring_params)
        cal_csv = os.path.join(
            _output_dir("figs/ring/calibration", args.params_json),
            cal_conn_label, "calibration_summary.csv",
        )
        noise_thresholds: dict[str, Optional[float]] = {}
        for ck in condition_keys:
            noise_thresholds[ck] = _lookup_noise_threshold(
                cal_csv, ck, amp_factor, ring_params.w_pyr_pyr_inter,
            )
        has_threshold = any(v is not None for v in noise_thresholds.values())
        if has_threshold:
            print(f"\nNoise thresholds from calibration ({cal_csv}):")
            for ck, nt in noise_thresholds.items():
                label = STUDY_CONDITIONS[ck].name
                if nt is not None:
                    exact_match = _lookup_noise_threshold_exact(
                        cal_csv, ck, amp_factor, ring_params.w_pyr_pyr_inter,
                    ) is not None
                    tag = "" if exact_match else " (shared — no exact match)"
                    print(f"  {label}: {nt:.4f}{tag}")
                else:
                    print(f"  {label}: not found — melt check disabled")
        else:
            print(f"\nNo calibration data found at {cal_csv}; bump-melt check disabled.")

        # --- Final displacement analysis per condition ---
        print("\nFinal displacement analysis:")
        angles_deg = np.degrees(ring_params.node_angles_rad)
        rng_snapshot = np.random.default_rng(args.seed + 314159)

        for cond_key in condition_keys:
            trials = [r for r in all_results if r['cond_key'] == cond_key]
            t_s = trials[0]['t_delay_s']
            noise_threshold = noise_thresholds.get(cond_key)
            cond_label = STUDY_CONDITIONS[cond_key].name

            # Amplitude stats over the full delay period
            amplitudes_arr = np.array([r['amplitude'] for r in trials])
            amp_mean = np.mean(amplitudes_arr, axis=0)
            amp_sem = (
                np.std(amplitudes_arr, axis=0, ddof=1) / np.sqrt(len(trials))
                if len(trials) > 1 else np.zeros(len(t_s))
            )
            survival = (
                np.mean(amplitudes_arr >= noise_threshold, axis=0)
                if noise_threshold is not None else np.ones(len(t_s))
            )

            # Per-trial: compute final displacement from cue.
            # Strategy:
            #   - Reference position: the known stimulus location (STIM_CENTER_DEG),
            #     converted to radians.  Using the fixed cue location avoids any
            #     bias introduced by the transient at the start of bump formation.
            #   - End window: last 500 ms of the delay (~5 oscillation cycles).
            #     Within that window, take the displacement with the *minimum*
            #     absolute value — i.e., the moment the bump was closest to the
            #     cue during the end window.  This estimates the DC shift of the
            #     attractor (oscillation amplitude cancels out at zero-crossings).
            dt_s = float(t_s[1] - t_s[0]) if len(t_s) > 1 else 1e-3
            end_window_frames = max(1, int(round(0.500 / dt_s)))    # 500 ms
            center_start = float(np.radians(STIM_CENTER_DEG))

            trial_displacements: list[float] = []
            trial_valid: list[bool] = []
            trial_indices: list[int] = []
            valid_trials_data: list[dict] = []  # for selecting one random valid trial

            for r in trials:
                center = r['center_unwrapped_rad']
                amp_end = float(r['amplitude'][-1])

                # Bump present at end of delay?
                bump_present = (
                    noise_threshold is None or amp_end >= noise_threshold
                )

                if len(center) >= 2 and bump_present:
                    # Displacement at every frame in the end window
                    w_end = min(end_window_frames, len(center))
                    disp_series = center[-w_end:] - center_start
                    # Wrap to [-π, π]
                    disp_series = (disp_series + np.pi) % (2 * np.pi) - np.pi
                    # Frame where bump was closest to cue
                    min_idx = int(np.argmin(np.abs(disp_series)))
                    disp_rad = float(disp_series[min_idx])
                    disp_deg = float(np.degrees(disp_rad))
                    trial_displacements.append(disp_deg)
                    trial_valid.append(True)
                    valid_trials_data.append({
                        'disp_deg': disp_deg,
                        'activity_start': r['activity_start'],
                        'activity_end': r['activity_end'],
                        'trial_idx': r['trial_idx'],
                    })
                else:
                    trial_displacements.append(0.0)
                    trial_valid.append(False)

                trial_indices.append(r['trial_idx'])

            disps = np.array(trial_displacements)
            valid_mask = np.array(trial_valid)
            valid_disps = disps[valid_mask]
            n_valid = int(np.sum(valid_mask))
            n_melted = len(trials) - n_valid

            mean_d = float(np.mean(valid_disps)) if n_valid > 0 else np.nan
            std_d  = float(np.std(valid_disps, ddof=1)) if n_valid > 1 else np.nan
            abs_mean = float(np.mean(np.abs(valid_disps))) if n_valid > 0 else np.nan

            # Select one random valid trial (bump present at end) for visualization
            sample_result = None
            if valid_trials_data:
                sample = valid_trials_data[int(rng_snapshot.integers(len(valid_trials_data)))]
                snap_disp = sample['disp_deg']
                print(f"  {cond_label}: mean shift = {mean_d:+.2f}°, "
                      f"mean |shift| = {abs_mean:.2f}°, "
                      f"std = {std_d:.2f}°  "
                      f"(n={n_valid}/{len(trials)}, "
                    f"random sample = {snap_disp:+.1f}°)")

                # Rerun the sampled trial with full recording for heatmap
                sample_seed = trial_seeds[sample['trial_idx']]
                print(f"    Rerunning random sample trial (seed={sample_seed}) for visualization...")
                local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
                r0, I_adapt0 = burnin_states[cond_key]
                actual_current = amp_factor * base_params.I_ext_pyr()
                T_ms_short = T_ms_full - BURN_IN_MS
                stimuli_short = [
                    RingStimulus(
                        center_deg=STIM_CENTER_DEG, amplitude=actual_current,
                        sigma_deg=STIM_SIGMA_DEG,
                        onset_ms=STIM_ONSET_MS - BURN_IN_MS,
                        duration_ms=STIM_DURATION_MS,
                    ),
                ]
                sample_result = simulate_ring(
                    local_params, ring_params, T_ms=T_ms_short,
                    stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
                    seed=sample_seed, connectivity=connectivity,
                    record_dt_ms=5.0,  # 5 ms resolution — enough for heatmap
                )
                sample_result.t_ms += BURN_IN_MS
            else:
                snap_disp = None
                print(f"  {cond_label}: WARNING — no valid trials (all melted at end)")

            if noise_threshold is not None and n_melted > 0:
                print(f"    ({n_melted} trial(s) had no bump at end of delay — excluded)")

            stim_offset_ms_local = STIM_ONSET_MS + STIM_DURATION_MS
            disp_data[cond_key] = {
                'displacements_deg': valid_disps,
                'mean_deg': mean_d,
                'std_deg': std_d,
                'abs_mean_deg': abs_mean,
                'n_valid': n_valid,
                'n_total': len(trials),
                'amplitude_factor': amp_factor,
                'stim_current': amp_factor * base_params.I_ext_pyr(),
                'noise_threshold': noise_threshold,
                'amp_t_s': t_s,
                'amp_mean': amp_mean,
                'amp_sem': amp_sem,
                'survival': survival,
                # Full simulation result for one random valid trial
                'sample_result': sample_result,
                'sample_displacement_deg': snap_disp,
                'delay_start_ms': stim_offset_ms_local + TRANSIENT_SKIP_TIME_MS,
                'delay_end_ms': stim_offset_ms_local + args.delay_ms,
                # Per-trial lists for CSV
                '_all_displacements': trial_displacements,
                '_all_valid': trial_valid,
                '_all_indices': trial_indices,
            }

    # --- Save CSVs (skipped when loaded from cache) ---
    if not loaded_from_cache:
        # 1. Summary CSV: one row per condition
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'mean_deg', 'std_deg', 'abs_mean_deg',
                'n_valid', 'n_total', 'delay_ms', 'amplitude_factor', 'seed', 'n_trials',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                d = disp_data[cond_key]
                writer.writerow({
                    'condition_key': cond_key,
                    'mean_deg': d['mean_deg'],
                    'std_deg': d['std_deg'],
                    'abs_mean_deg': d['abs_mean_deg'],
                    'n_valid': d['n_valid'],
                    'n_total': d['n_total'],
                    'delay_ms': args.delay_ms,
                    'amplitude_factor': amp_factor,
                    'seed': args.seed,
                    'n_trials': n_trials,
                })

        # 2. Per-trial displacement CSV
        with open(trials_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'trial_idx', 'displacement_deg', 'valid',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                d = disp_data[cond_key]
                for ti, disp, valid in zip(
                    d['_all_indices'], d['_all_displacements'], d['_all_valid']
                ):
                    writer.writerow({
                        'condition_key': cond_key,
                        'trial_idx': ti,
                        'displacement_deg': disp,
                        'valid': int(valid),
                    })

        # 3. Amplitude CSV
        with open(amplitude_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 't_s', 'amp_mean', 'amp_sem',
                'survival_frac', 'noise_threshold',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                d = disp_data[cond_key]
                if 'amp_t_s' not in d:
                    continue
                nt = d.get('noise_threshold')
                for i in range(len(d['amp_t_s'])):
                    writer.writerow({
                        'condition_key': cond_key,
                        't_s': d['amp_t_s'][i],
                        'amp_mean': d['amp_mean'][i],
                        'amp_sem': d['amp_sem'][i],
                        'survival_frac': d['survival'][i],
                        'noise_threshold': nt if nt is not None else '',
                    })

        print(f"\nCSVs saved to {out_dir}/")
        print(f"  diffusion_displacement_summary.csv  (per-condition stats)")
        print(f"  diffusion_displacement_trials.csv   (per-trial displacements)")
        print(f"  diffusion_amplitude.csv             (amplitude over time)")

    # --- Plots ---
    from .plotting import plot_displacement_distribution, plot_diffusion_ring_snapshot

    band_tag = f"  ({n_trials} trials)" if n_trials > 1 else ""

    # 1. Displacement distribution plot
    disp_save = os.path.join(out_dir, "diffusion_displacement.png")
    plot_displacement_distribution(
        disp_data,
        save_path=disp_save,
        suptitle=f"Final Bump Displacement from Cue{band_tag}  [{_weights_label(ring_params)}]",
    )
    plt.close()
    print(f"Figure saved to {disp_save}")

    # 2. Ring activity during delay (one random sample per condition)
    has_snaps = any(
        d.get('sample_result') is not None for d in disp_data.values()
    )
    if has_snaps:
        snap_save = os.path.join(out_dir, "diffusion_ring_snapshot.png")
        plot_diffusion_ring_snapshot(
            disp_data,
            save_path=snap_save,
            suptitle=f"Ring Activity During Delay Across Conditions{band_tag}  [{_weights_label(ring_params)}]",
        )
        plt.close()
        print(f"Figure saved to {snap_save}")


# ============================================================================
# DRIFT FIELD: PARALLEL WORKER
# ============================================================================

_drift_sim_args: Optional[dict] = None


def _drift_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    T_ms_full: float,
):
    """Initialize worker process for drift field analysis."""
    global _drift_sim_args
    _drift_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
        'burnin_states': burnin_states,
        'T_ms_full': T_ms_full,
    }


def _drift_run_single(job: tuple) -> dict:
    """Run a single distractor trial.  Returns pre/post bump positions."""
    global _drift_sim_args
    cfg = _drift_sim_args
    cond_key, offset_deg, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['ring_params']
    connectivity = cfg['connectivity']
    T_ms_full = cfg['T_ms_full']

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    r0, I_adapt0 = cfg['burnin_states'][cond_key]

    amp_factor = args_d['amplitude']
    actual_current = amp_factor * base_params.I_ext_pyr()

    distractor_onset_ms = args_d['distractor_onset_ms']
    distractor_duration_ms = args_d['distractor_duration_ms']
    distractor_amp_factor = args_d['distractor_amplitude']
    distractor_current = distractor_amp_factor * base_params.I_ext_pyr()

    # Distractor location = cue + offset
    distractor_location_deg = (STIM_CENTER_DEG + offset_deg) % 360.0

    T_ms_short = T_ms_full - BURN_IN_MS

    # Cue stimulus (shifted for burn-in removal)
    cue_stim = RingStimulus(
        center_deg=STIM_CENTER_DEG, amplitude=actual_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=STIM_ONSET_MS - BURN_IN_MS,
        duration_ms=STIM_DURATION_MS,
    )

    # Distractor stimulus (onset is relative to simulation start, after burn-in removal)
    dist_onset_abs = STIM_ONSET_MS + STIM_DURATION_MS + distractor_onset_ms
    dist_stim = RingStimulus(
        center_deg=distractor_location_deg,
        amplitude=distractor_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=dist_onset_abs - BURN_IN_MS,
        duration_ms=distractor_duration_ms,
    )

    stimuli_short = [cue_stim, dist_stim]

    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
        seed=seed, connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    result.t_ms += BURN_IN_MS

    # Measure bump position just before distractor and shortly after
    pre_dist_t = dist_onset_abs - 50  # 50ms before distractor
    post_dist_t = dist_onset_abs + distractor_duration_ms + TRANSIENT_SKIP_TIME_MS

    # Pre-distractor position
    pre_idx = np.argmin(np.abs(result.t_ms - pre_dist_t))
    pre_activity = result.r[pre_idx, :, 0]
    pre_center_rad, pre_amp = population_vector_decode(
        pre_activity, ring_params.node_angles_rad,
    )

    # Post-distractor position
    post_idx = np.argmin(np.abs(result.t_ms - post_dist_t))
    post_activity = result.r[post_idx, :, 0]
    post_center_rad, post_amp = population_vector_decode(
        post_activity, ring_params.node_angles_rad,
    )

    # Signed displacement (positive = toward distractor)
    from .connectivity import angular_distance
    raw_disp = post_center_rad - pre_center_rad
    # Wrap to [-pi, pi]
    displacement_rad = (raw_disp + np.pi) % (2 * np.pi) - np.pi

    return {
        'cond_key': cond_key,
        'offset_deg': offset_deg,
        'trial_idx': trial_idx,
        'displacement_rad': float(displacement_rad),
        'pre_amp': float(pre_amp),
        'post_amp': float(post_amp),
    }


# ============================================================================
# DRIFT FIELD SUBCOMMAND
# ============================================================================

def _unique_path(path: str) -> str:
    """Return a non-colliding path by appending _N when needed."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        candidate = f"{base}_{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def _is_calibrate_cached(
    cond_dir: str,
    cond_key: str,
    amplitudes: list[float],
    w_inter_values: list[float],
    n_trials: int,
) -> bool:
    """Check whether calibration summary already has all requested grid points."""
    csv_path = os.path.join(cond_dir, "calibration_summary.csv")
    if not os.path.exists(csv_path):
        return False
    needed = {(float(a), float(w)) for a in amplitudes for w in w_inter_values}
    found: set[tuple[float, float]] = set()
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                a = float(row.get("amplitude", "nan"))
                w = float(row.get("w_inter", "nan"))
                tr = int(float(row.get("n_trials", 0)))
                if (a, w) in needed and tr >= n_trials:
                    found.add((a, w))
    except Exception:
        return False
    return found == needed


def _load_baseline_trial_counts(
    cond_dir: str,
    cond_key: str,
) -> tuple[dict[tuple[str, float], int], bool]:
    """Return cached baseline trial counts and whether trial metadata is present."""
    csv_path = os.path.join(cond_dir, "baseline_A_hat.csv")
    if not os.path.exists(csv_path):
        return {}, False
    counts: dict[tuple[str, float], int] = {}
    has_trial_idx = False
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "trial_idx" in rows[0]:
        has_trial_idx = True
    for row in rows:
        ck = row.get("condition", cond_key)
        if ck != cond_key:
            continue
        try:
            w = float(row["w_inter"])
        except Exception:
            continue
        key = (ck, w)
        if has_trial_idx:
            counts[key] = counts.get(key, 0) + 1
        else:
            counts[key] = max(counts.get(key, 0), 1)
    return counts, has_trial_idx


def _load_calibrate_baseline(
    cond_dir: str,
    cond_key: str,
    w_inter_values: list[float],
    noise_percentile: float,
) -> tuple[
    dict[tuple[str, float], float],
    dict[tuple[str, float], np.ndarray],
    set[float],
    dict[tuple[str, float], float],
]:
    """Load baseline amplitudes and thresholds for one condition."""
    csv_path = os.path.join(cond_dir, "baseline_A_hat.csv")
    if not os.path.exists(csv_path):
        return {}, {}, set(), {}

    allowed_w = {float(w) for w in w_inter_values}
    samples: dict[tuple[str, float], list[float]] = {}
    cap_hits: dict[tuple[str, float], int] = {}
    cap_counts: dict[tuple[str, float], int] = {}
    thresholds: dict[tuple[str, float], float] = {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        ck = row.get("condition", cond_key)
        if ck != cond_key:
            continue
        try:
            w = float(row["w_inter"])
            if w not in allowed_w:
                continue
            a_hat = float(row["A_hat"])
        except Exception:
            continue
        key = (ck, w)
        samples.setdefault(key, []).append(a_hat)
        cap_raw = str(row.get("cap_hit", "")).strip().lower()
        if cap_raw != "":
            cap_bool: bool | None = None
            if cap_raw in {"1", "true", "t", "yes", "y"}:
                cap_bool = True
            elif cap_raw in {"0", "false", "f", "no", "n"}:
                cap_bool = False
            else:
                try:
                    cap_bool = float(cap_raw) > 0.0
                except Exception:
                    cap_bool = None
            if cap_bool is not None:
                cap_counts[key] = cap_counts.get(key, 0) + 1
                if cap_bool:
                    cap_hits[key] = cap_hits.get(key, 0) + 1
        if row.get("noise_threshold", "") != "":
            try:
                thresholds[key] = float(row["noise_threshold"])
            except Exception:
                pass

    baseline = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
    for key, vals in baseline.items():
        if key not in thresholds:
            thresholds[key] = compute_noise_floor(vals, percentile=noise_percentile)
    cap_hit_fractions = {
        key: float(cap_hits.get(key, 0)) / float(count)
        for key, count in cap_counts.items()
        if count > 0
    }

    saturated = {w for (ck, w), th in thresholds.items() if ck == cond_key and th <= 1e-6}
    return thresholds, baseline, saturated, cap_hit_fractions


_noise_floor_sim_args: dict = {}


def _noise_floor_init_worker(
    base_params: "CircuitParams",
    ring_params_base: "RingParams",
    delay_ms: float,
    record_dt_ms: float,
) -> None:
    global _noise_floor_sim_args
    _noise_floor_sim_args = {
        'base_params': base_params,
        'ring_params_base': ring_params_base,
        'delay_ms': delay_ms,
        'record_dt_ms': record_dt_ms,
    }

    # Warm up Numba once per worker so the first real trial does not pay
    # compilation overhead in the middle of progress reporting.
    try:
        from .simulation import RING_NUMBA_AVAILABLE

        if RING_NUMBA_AVAILABLE:
            rp_warm = RingParams(
                n_nodes=8,
                w_pyr_pyr_inter=float(ring_params_base.w_pyr_pyr_inter),
                sigma_pyr_deg=float(ring_params_base.sigma_pyr_deg),
                w_pv_global=float(ring_params_base.w_pv_global),
            )
            conn_warm = RingConnectivity.from_params(rp_warm)
            _ = simulate_ring(
                base_params,
                rp_warm,
                T_ms=0.2,
                dt_ms=0.1,
                stimuli=None,
                seed=0,
                connectivity=conn_warm,
                noise_type="white",
                record_dt_ms=0.1,
            )
    except Exception:
        # Warmup is an optimization only; execution continues without it.
        pass


def _noise_floor_run_single(job: tuple) -> dict:
    """Run one no-stimulus baseline trial. Called by ProcessPoolExecutor."""
    global _noise_floor_sim_args
    cfg = _noise_floor_sim_args
    cond_key, cond_idx, w, trial_idx, trial_seed, noise_percentile = job
    del cond_idx

    rp = replace(cfg['ring_params_base'], w_pyr_pyr_inter=float(w))
    conn = RingConnectivity.from_params(rp)
    local_params = apply_condition(cfg['base_params'], STUDY_CONDITIONS[cond_key])

    result = simulate_ring(
        local_params,
        rp,
        T_ms=max(BURN_IN_MS, float(cfg['delay_ms'])),
        stimuli=None,
        connectivity=conn,
        seed=trial_seed,
        record_dt_ms=max(10.0, float(cfg['record_dt_ms'])),
    )
    cap_hit = bool(np.any(result.r >= (RATE_CAP_HZ - 1e-9)))
    _, a_hat = population_vector_decode(result.r[-1, :, 0], rp.node_angles_rad)
    return {
        "condition": cond_key,
        "w_inter": f"{float(w):.8g}",
        "trial_idx": str(trial_idx),
        "seed": str(trial_seed),
        "A_hat": f"{float(a_hat):.10g}",
        "cap_hit": "1" if cap_hit else "0",
        "noise_percentile": f"{float(noise_percentile):.8g}",
        "noise_threshold": "",
    }


def _run_noise_floor_for_conditions(
    conditions_to_run: list[str],
    w_inter_values: list[float],
    ring_params_base: RingParams,
    base_params: CircuitParams,
    n_baseline: int,
    noise_percentile: float,
    out_dir: str,
    n_workers: int,
    batch_chunk_size: int,
    seed: int,
    delay_ms: float,
    record_dt_ms: float,
    w_inter_values_by_condition: dict[str, list[float]] | None = None,
    trials_to_add_by_key: dict[tuple[str, float], int] | None = None,
    trial_start_idx_by_key: dict[tuple[str, float], int] | None = None,
    preserve_existing_cache: bool = True,
) -> tuple[
    dict[tuple[str, float], float],
    dict[tuple[str, float], np.ndarray],
    dict[tuple[str, float], float],
]:
    """Compute baseline no-stimulus amplitudes and thresholds for conditions."""
    from tqdm import tqdm

    all_thresholds: dict[tuple[str, float], float] = {}
    all_baseline: dict[tuple[str, float], np.ndarray] = {}
    all_cap_hit_fractions: dict[tuple[str, float], float] = {}

    # Build all jobs and load existing rows per condition.
    del batch_chunk_size
    existing_rows_by_cond: dict[str, list[dict]] = {}
    jobs: list[tuple] = []
    for cond_idx, ck in enumerate(conditions_to_run):
        cond_dir = os.path.join(out_dir, ck)
        os.makedirs(cond_dir, exist_ok=True)
        csv_path = os.path.join(cond_dir, "baseline_A_hat.csv")

        existing_rows: list[dict] = []
        if preserve_existing_cache and os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                existing_rows = list(csv.DictReader(f))
        existing_rows_by_cond[ck] = existing_rows

        target_ws = (
            w_inter_values_by_condition.get(ck, w_inter_values)
            if w_inter_values_by_condition is not None
            else w_inter_values
        )

        for w in target_ws:
            key = (ck, float(w))
            n_add = (
                int(trials_to_add_by_key.get(key, n_baseline))
                if trials_to_add_by_key is not None else n_baseline
            )
            start_idx = (
                int(trial_start_idx_by_key.get(key, 0))
                if trial_start_idx_by_key is not None else 0
            )
            for i in range(n_add):
                trial_idx = start_idx + i
                trial_seed = int(seed + cond_idx * 100000 + int(round(w * 1000)) * 10 + trial_idx)
                jobs.append((ck, cond_idx, float(w), trial_idx, trial_seed, noise_percentile))

    # Run simulations (parallel or sequential).
    init_args = (base_params, ring_params_base, delay_ms, record_dt_ms)
    new_rows_by_cond: dict[str, list[dict]] = {ck: [] for ck in conditions_to_run}

    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(
            mp_context=_MP_CONTEXT,
            max_workers=n_workers,
            initializer=_noise_floor_init_worker,
            initargs=init_args,
        ) as executor:
            with tqdm(total=len(jobs), desc="Noise floor", unit="trial", smoothing=0) as pbar:
                job_iter = iter(jobs)
                max_in_flight = max(1, n_workers * 4)
                in_flight: dict = {}

                for _ in range(min(max_in_flight, len(jobs))):
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        break
                    fut = executor.submit(_noise_floor_run_single, job)
                    in_flight[fut] = job

                while in_flight:
                    for future in as_completed(list(in_flight.keys()), timeout=None):
                        in_flight.pop(future, None)
                        row = future.result()
                        new_rows_by_cond[row["condition"]].append(row)
                        pbar.update(1)

                        try:
                            job = next(job_iter)
                            fut = executor.submit(_noise_floor_run_single, job)
                            in_flight[fut] = job
                        except StopIteration:
                            pass
                        break
    else:
        _noise_floor_init_worker(*init_args)
        for job in tqdm(jobs, desc="Noise floor", unit="trial"):
            row = _noise_floor_run_single(job)
            new_rows_by_cond[row["condition"]].append(row)

    # Per-condition: aggregate, compute thresholds, write CSV.
    for ck in conditions_to_run:
        cond_dir = os.path.join(out_dir, ck)
        csv_path = os.path.join(cond_dir, "baseline_A_hat.csv")
        rows = existing_rows_by_cond[ck] + new_rows_by_cond[ck]

        vals_by_w: dict[float, list[float]] = {}
        cap_by_w: dict[float, list[float]] = {}
        for row in rows:
            if row.get("condition", ck) != ck:
                continue
            try:
                w = float(row["w_inter"])
                vals_by_w.setdefault(w, []).append(float(row["A_hat"]))
            except Exception:
                continue
            cap_raw = str(row.get("cap_hit", "")).strip().lower()
            if cap_raw != "":
                cap_bool: bool | None = None
                if cap_raw in {"1", "true", "t", "yes", "y"}:
                    cap_bool = True
                elif cap_raw in {"0", "false", "f", "no", "n"}:
                    cap_bool = False
                else:
                    try:
                        cap_bool = float(cap_raw) > 0.0
                    except Exception:
                        cap_bool = None
                if cap_bool is not None:
                    cap_by_w.setdefault(w, []).append(1.0 if cap_bool else 0.0)

        thresholds_by_w = {
            w: compute_noise_floor(np.asarray(vals, dtype=float), percentile=noise_percentile)
            for w, vals in vals_by_w.items()
        }

        for row in rows:
            try:
                w = float(row["w_inter"])
                row["noise_threshold"] = f"{float(thresholds_by_w[w]):.10g}"
            except Exception:
                pass

        rows.sort(key=lambda r: (r.get("condition", ""), float(r.get("w_inter", 0.0)), int(float(r.get("trial_idx", 0)))))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "condition",
                    "w_inter",
                    "trial_idx",
                    "seed",
                    "A_hat",
                    "cap_hit",
                    "noise_percentile",
                    "noise_threshold",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        for w, vals in vals_by_w.items():
            key = (ck, w)
            all_baseline[key] = np.asarray(vals, dtype=float)
            all_thresholds[key] = float(thresholds_by_w[w])
        for w, cap_vals in cap_by_w.items():
            if cap_vals:
                all_cap_hit_fractions[(ck, w)] = float(np.mean(np.asarray(cap_vals, dtype=float)))

    return all_thresholds, all_baseline, all_cap_hit_fractions


def _compute_calibrate_metrics(
    result,
    cond_key: str,
    amplitude: float,
    w_inter: float,
    trial_idx: int,
    seed: int,
    eval_times_ms: list[float],
    delay_ms: float,
) -> dict:
    """Compute per-trial calibration metrics from a ring simulation result."""
    del delay_ms

    t = np.asarray(result.t_ms)
    a_hat_tc: list[float] = []
    for et in eval_times_ms:
        idx = int(np.argmin(np.abs(t - float(et))))
        _, a_hat = population_vector_decode(result.r[idx, :, 0], result.ring_params.node_angles_rad)
        a_hat_tc.append(float(a_hat))

    center_final_rad, a_hat_final = population_vector_decode(
        result.r[-1, :, 0], result.ring_params.node_angles_rad,
    )
    center_final_deg = float(np.degrees(center_final_rad) % 360.0)
    err_deg = float((center_final_deg - STIM_CENTER_DEG + 180.0) % 360.0 - 180.0)
    peak_pyr_rate = float(np.max(result.r[:, :, 0]))

    return {
        "cond_key": cond_key,
        "amplitude": float(amplitude),
        "w_inter": float(w_inter),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "A_hat_final": float(a_hat_final),
        "A_hat_timecourse": a_hat_tc,
        "peak_pyr_rate": peak_pyr_rate,
        "center_final_deg": center_final_deg,
        "error_from_cue_deg": abs(err_deg),
    }


_calibrate_sim_args: Optional[dict] = None


def _calibrate_init_worker(
    local_params_by_cond: dict[str, CircuitParams],
    ring_params_base: RingParams,
    connectivity_cache: dict[float, RingConnectivity],
    burnin_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]],
    eval_times_ms: list[float],
    delay_ms: float,
    record_dt_ms: float,
    T_ms_short: float,
    base_I_ext_pyr: float,
) -> None:
    """Initialize worker state for calibration trial-level jobs."""
    global _calibrate_sim_args
    _calibrate_sim_args = {
        "local_params_by_cond": local_params_by_cond,
        "ring_params_base": ring_params_base,
        "connectivity_cache": connectivity_cache,
        "burnin_cache": burnin_cache,
        "eval_times_ms": eval_times_ms,
        "delay_ms": delay_ms,
        "record_dt_ms": record_dt_ms,
        "T_ms_short": T_ms_short,
        "base_I_ext_pyr": base_I_ext_pyr,
    }


def _calibrate_run_single(job: tuple) -> dict:
    """Run one calibration trial for one (condition, amplitude, w_inter)."""
    global _calibrate_sim_args
    cfg = _calibrate_sim_args

    cond_key, amp, w, trial_idx, seed_trial = job
    local_params = cfg["local_params_by_cond"][cond_key]
    rp = replace(cfg["ring_params_base"], w_pyr_pyr_inter=float(w))
    conn = cfg["connectivity_cache"][float(w)]
    r0, I_adapt0 = cfg["burnin_cache"][(cond_key, float(w))]

    actual_current = float(amp) * float(cfg["base_I_ext_pyr"])
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    res = simulate_ring(
        local_params,
        rp,
        T_ms=float(cfg["T_ms_short"]),
        stimuli=stimuli_short,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=int(seed_trial),
        noise_type="white",
        connectivity=conn,
        record_dt_ms=float(cfg["record_dt_ms"]),
    )
    res.t_ms += BURN_IN_MS
    return _compute_calibrate_metrics(
        res,
        cond_key,
        float(amp),
        float(w),
        int(trial_idx),
        int(seed_trial),
        cfg["eval_times_ms"],
        float(cfg["delay_ms"]),
    )


def _load_calibrate_grid_results(cond_dir: str, cond_key: str) -> list[dict]:
    """Load cached per-trial calibration results from CSV."""
    csv_path = os.path.join(cond_dir, "calibration_results.csv")
    if not os.path.exists(csv_path):
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("condition_key", cond_key) != cond_key:
                continue
            tc_raw = row.get("a_hat_timecourse", "").strip()
            tc = [float(x) for x in tc_raw.split()] if tc_raw else []
            rows.append(
                {
                    "cond_key": cond_key,
                    "amplitude": float(row["amplitude"]),
                    "w_inter": float(row["w_inter"]),
                    "trial_idx": int(float(row["trial_idx"])),
                    "seed": int(float(row.get("seed", 0))),
                    "A_hat_final": float(row.get("A_hat_final", "nan")),
                    "A_hat_timecourse": tc,
                    "peak_pyr_rate": float(row.get("peak_pyr_rate", "nan")),
                    "center_final_deg": float(row.get("center_final_deg", "nan")),
                    "error_from_cue_deg": float(row.get("error_from_cue_deg", "nan")),
                }
            )
    return rows


# ============================================================================
# 3D CALIBRATION SWEEP HELPERS
# ============================================================================

def _compute_delay_state_fracs(
    pyr: np.ndarray,
    t_ms: np.ndarray,
    resting_hz: float,
    stim_onset_ms: float,
    stim_offset_ms: float,
) -> dict:
    """Classify each delay timepoint as resting / bump / saturated.

    The bump lower bound is resting_hz * CAL3D_RESTING_MULT, clamped to
    at least resting_hz + 5 Hz and CAL3D_BUMP_MIN_HZ absolute.

    Parameters:
        pyr: (n_times, n_nodes) PYR firing rates from simulation.
        t_ms: time axis (ms), zero-based from the short simulation.
        resting_hz: Mean PYR rate from the burn-in (pre-cue baseline).
        stim_onset_ms, stim_offset_ms: cue window in short-sim time.

    Returns:
        dict with keys: cue_peak_hz, cue_saturated, delay_rest_frac,
        delay_bump_frac, delay_sat_frac, delay_mean_peak_hz, bump_lo_hz.
    """
    bump_lo = max(
        resting_hz * CAL3D_RESTING_MULT,
        resting_hz + 5.0,
        CAL3D_BUMP_MIN_HZ,
    )

    # --- Cue ---
    cue_mask = (t_ms >= stim_onset_ms) & (t_ms <= stim_offset_ms)
    if cue_mask.any():
        cue_peak = float(pyr[cue_mask].max())
    else:
        cue_peak = float("nan")
    cue_sat = cue_peak >= CAL3D_CUE_SAT_THRESH_HZ

    # --- Delay ---
    delay_mask = t_ms > stim_offset_ms
    if not delay_mask.any():
        return {
            "cue_peak_hz": cue_peak, "cue_saturated": cue_sat,
            "delay_rest_frac": float("nan"), "delay_bump_frac": float("nan"),
            "delay_sat_frac": float("nan"), "delay_mean_peak_hz": float("nan"),
            "bump_lo_hz": bump_lo,
        }

    # max PYR across all nodes at each delay time step
    peak_t = pyr[delay_mask].max(axis=1)
    rest_m = peak_t < bump_lo
    bump_m = (peak_t >= bump_lo) & (peak_t < CAL3D_SAT_THRESH_HZ)
    sat_m = peak_t >= CAL3D_SAT_THRESH_HZ

    return {
        "cue_peak_hz": cue_peak,
        "cue_saturated": bool(cue_sat),
        "delay_rest_frac": float(rest_m.mean()),
        "delay_bump_frac": float(bump_m.mean()),
        "delay_sat_frac": float(sat_m.mean()),
        "delay_mean_peak_hz": float(peak_t.mean()),
        "bump_lo_hz": float(bump_lo),
    }


_cal3d_worker_args: Optional[dict] = None


def _cal3d_init_worker(
    local_params_by_cond: "dict[str, CircuitParams]",
    ring_params_by_key: "dict[tuple[float, float], RingParams]",
    conn_by_key: "dict[tuple[float, float], RingConnectivity]",
    burnin_by_key: "dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]]",
    resting_by_key: "dict[tuple[str, float, float], float]",
    base_I_ext: float,
    stim_onset_ms: float,
    stim_offset_ms: float,
    T_ms_short: float,
    record_dt_ms: float,
) -> None:
    """Initialise per-worker state for the 3D calibration sweep."""
    global _cal3d_worker_args
    _cal3d_worker_args = {
        "local_params": local_params_by_cond,
        "ring_params": ring_params_by_key,
        "conn": conn_by_key,
        "burnin": burnin_by_key,
        "resting": resting_by_key,
        "base_I_ext": base_I_ext,
        "stim_onset_ms": stim_onset_ms,
        "stim_offset_ms": stim_offset_ms,
        "T_ms_short": T_ms_short,
        "record_dt_ms": record_dt_ms,
    }


def _cal3d_run_single(job: tuple) -> dict:
    """Run one trial of the 3D calibration sweep."""
    global _cal3d_worker_args
    cfg = _cal3d_worker_args
    ck, w_pv, w_pyr, amp, trial_idx, seed = job

    rp = cfg["ring_params"][(w_pv, w_pyr)]
    conn = cfg["conn"][(w_pv, w_pyr)]
    r0, Ia0 = cfg["burnin"][(ck, w_pv, w_pyr)]
    resting_hz = cfg["resting"][(ck, w_pv, w_pyr)]
    local_params = cfg["local_params"][ck]

    actual_current = float(amp) * float(cfg["base_I_ext"])
    stimuli = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=float(cfg["stim_onset_ms"]),
            duration_ms=STIM_DURATION_MS,
        )
    ]

    res = simulate_ring(
        local_params, rp,
        T_ms=float(cfg["T_ms_short"]),
        stimuli=stimuli,
        r0=r0,
        I_adapt0=Ia0,
        seed=int(seed),
        noise_type="white",
        connectivity=conn,
        record_dt_ms=float(cfg["record_dt_ms"]),
    )

    pyr = res.r[:, :, 0]  # (n_times, n_nodes)
    state = _compute_delay_state_fracs(
        pyr, res.t_ms,
        resting_hz=resting_hz,
        stim_onset_ms=float(cfg["stim_onset_ms"]),
        stim_offset_ms=float(cfg["stim_offset_ms"]),
    )

    return {
        "cond_key": ck,
        "w_pv": float(w_pv),
        "w_pyr": float(w_pyr),
        "amp": float(amp),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "resting_hz": float(resting_hz),
        **state,
    }


def _cal3d_is_cached(csv_path: str, w_pv_values: list, w_pyr_values: list,
                     amplitude_values: list, n_trials: int) -> bool:
    """Return True if the 3D sweep CSV has all requested grid points."""
    if not os.path.exists(csv_path):
        return False
    needed = {(float(wv), float(wy), float(a))
              for wv in w_pv_values for wy in w_pyr_values for a in amplitude_values}
    found: set[tuple[float, float, float]] = set()
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (float(row["w_pv"]), float(row["w_pyr"]), float(row["amp"]))
                if key in needed and int(float(row.get("n_trials", 0))) >= n_trials:
                    found.add(key)
    except Exception:
        return False
    return found == needed


def _cal3d_load_cached(csv_path: str) -> list[dict]:
    """Load per-trial 3D sweep results from CSV."""
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "cond_key": row.get("cond_key", "WT"),
                "w_pv": float(row["w_pv"]),
                "w_pyr": float(row["w_pyr"]),
                "amp": float(row["amp"]),
                "trial_idx": int(float(row.get("trial_idx", 0))),
                "seed": int(float(row.get("seed", 0))),
                "resting_hz": float(row.get("resting_hz", "nan")),
                "cue_peak_hz": float(row.get("cue_peak_hz", "nan")),
                "cue_saturated": row.get("cue_saturated", "False").lower() in ("1", "true"),
                "delay_rest_frac": float(row.get("delay_rest_frac", "nan")),
                "delay_bump_frac": float(row.get("delay_bump_frac", "nan")),
                "delay_sat_frac": float(row.get("delay_sat_frac", "nan")),
                "delay_mean_peak_hz": float(row.get("delay_mean_peak_hz", "nan")),
                "bump_lo_hz": float(row.get("bump_lo_hz", "nan")),
            })
    return rows


ASYM_SETTLING_MS: float = 1000.0
ASYM_PRE_CUE_WINDOW_MS: float = 200.0

_asym_sim_args: Optional[dict] = None


def _asym_init_worker(
    base_params: CircuitParams,
    per_cond_rp: dict,
    per_cond_conn: dict,
    amplitude: float,
    delay_ms: float,
    record_dt_ms: float,
    random_cue_location: bool,
    balance_cue: bool,
    correct_asymmetry: bool,
) -> None:
    """Initialise worker process for asymmetry trials."""
    global _asym_sim_args
    _asym_sim_args = {
        "base_params": base_params,
        "per_cond_rp": per_cond_rp,
        "per_cond_conn": per_cond_conn,
        "amplitude": amplitude,
        "delay_ms": delay_ms,
        "record_dt_ms": record_dt_ms,
        "random_cue_location": random_cue_location,
        "balance_cue": balance_cue,
        "correct_asymmetry": correct_asymmetry,
    }


def _asym_run_single(job: tuple) -> dict:
    """Run one asymmetry trial and return summary metrics."""
    from .analysis import compute_bump_asymmetry, decode_bump_center, compute_asymmetry_temporal_metrics

    global _asym_sim_args
    cfg = _asym_sim_args

    cond_key, trial_idx, seed = job
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(cfg["base_params"], condition)
    rp = cfg["per_cond_rp"][cond_key]

    if cfg["random_cue_location"]:
        rng = np.random.default_rng(int(seed) ^ 0xA51A51)
        cue_deg = float(rng.uniform(0.0, 360.0))
    elif cfg["balance_cue"]:
        cue_deg = _balance_cue_location(STIM_CENTER_DEG, rp)
    else:
        cue_deg = STIM_CENTER_DEG

    stim_onset = ASYM_SETTLING_MS
    stim_offset = stim_onset + STIM_DURATION_MS
    T_ms = stim_offset + cfg["delay_ms"]
    cue_current = cfg["amplitude"] * cfg["base_params"].I_ext_pyr()

    stimuli = [
        RingStimulus(
            center_deg=cue_deg,
            amplitude=cue_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=stim_onset,
            duration_ms=STIM_DURATION_MS,
        )
    ]

    result = simulate_ring(
        local_params,
        rp,
        T_ms=T_ms,
        stimuli=stimuli,
        seed=seed,
        connectivity=cfg["per_cond_conn"][cond_key],
        record_dt_ms=cfg["record_dt_ms"],
        record_adaptation=False,
    )

    asym = compute_bump_asymmetry(result)
    _, amp_trace = decode_bump_center(result, population=0)

    pre_mask = (result.t_ms >= (stim_onset - ASYM_PRE_CUE_WINDOW_MS)) & (result.t_ms < stim_onset)
    delay_start = stim_offset + TRANSIENT_SKIP_TIME_MS
    delay_mask = (result.t_ms >= delay_start) & (result.t_ms <= T_ms)

    def _window_asym(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        a = asym[mask]
        if not cfg["correct_asymmetry"]:
            return float(np.mean(a))
        amp_w = amp_trace[mask]
        denom = float(np.sum(amp_w))
        if denom <= 1e-10:
            return 0.0
        return float(np.sum(a * amp_w) / denom)

    pre_cue_asym = _window_asym(pre_mask)
    last_pre_vals = asym[pre_mask]
    last_pre_cue_asym = float(last_pre_vals[-1]) if len(last_pre_vals) > 0 else float("nan")
    delay_asym = _window_asym(delay_mask)

    m_delay = compute_asymmetry_temporal_metrics(asym[delay_mask], result.t_ms[delay_mask])
    m_pre = compute_asymmetry_temporal_metrics(asym[pre_mask], result.t_ms[pre_mask])

    return {
        "cond_key": cond_key,
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "cue_deg": float(cue_deg),
        "pre_cue_asym": float(pre_cue_asym),
        "last_pre_cue_asym": float(last_pre_cue_asym),
        "delay_asym": float(delay_asym),
        "mean_abs_asym": float(m_delay.get("mean_abs_asym", np.nan)),
        "asym_std": float(m_delay.get("asym_std", np.nan)),
        "mean_abs_asym_precue": float(m_pre.get("mean_abs_asym", np.nan)),
        "asym_std_precue": float(m_pre.get("asym_std", np.nan)),
    }

def cmd_calibrate(args: argparse.Namespace) -> None:
    """3D parameter calibration sweep: w_pv_global × w_pyr_pyr_inter × amplitude.

    For each grid point, runs n_trials short ring simulations from a
    pre-computed burn-in state and classifies the delay period into three
    states based on the maximum PYR firing rate across all nodes:
      - RESTING   : max_PYR < resting × 2.5  (or < 10 Hz)
      - BUMP      : resting_threshold ≤ max_PYR < 90 Hz
      - SATURATED : max_PYR ≥ 90 Hz

    The key quality metric is delay_bump_frac (fraction of delay time in
    bump state). A true bump requires high bump_frac AND low sat_frac.
    """
    from .plotting import plot_3d_sweep_slice, plot_3d_sweep_summary
    _resolve_seed(args)

    import json as _json
    from tqdm import tqdm

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    # --- Grid ---
    w_pv_values: list[float] = list(getattr(args, 'w_pv_values', None) or [args.w_pv_global])
    # w_pyr values
    _w_inter_explicit = getattr(args, 'w_inter_values', None)
    _w_min = getattr(args, 'w_inter_min', None)
    _w_max = getattr(args, 'w_inter_max', None)
    _n_inter = getattr(args, 'n_inter', None)
    if _w_inter_explicit is not None:
        w_pyr_values: list[float] = list(_w_inter_explicit)
    elif _w_min is not None and _w_max is not None and _n_inter is not None:
        w_pyr_values = list(np.linspace(_w_min, _w_max, _n_inter))
    else:
        w_pyr_values = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]

    amplitudes: list[float] = list(args.amplitudes)
    n_trials: int = int(args.n_trials)
    n_workers: int = max(1, _resolve_workers(args))
    no_cache: bool = getattr(args, 'no_cache', False)
    record_dt_ms: float = getattr(args, 'record_dt_ms', 5.0)
    sigma_deg: float = float(args.sigma_pyr_deg)

    # --- Conditions ---
    if args.conditions is None:
        condition_keys = ["WT"]
    elif "all" in args.conditions:
        condition_keys = list(STUDY_CONDITIONS.keys())
    else:
        condition_keys = list(args.conditions)
        for k in condition_keys:
            if k not in STUDY_CONDITIONS:
                print(f"Error: unknown condition '{k}'.\n"
                      f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                sys.exit(1)

    # --- Output directory (sigma-only label since both w_pv and w_pyr are swept) ---
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=w_pyr_values[0],
        sigma_pyr_deg=sigma_deg,
        w_pv_global=w_pv_values[0],
    )
    sigma_label = f"{args.n_nodes}_sigma_{_fmt(sigma_deg)}"
    out_dir = os.path.join(
        _output_dir("figs/ring/calibration", args.params_json),
        sigma_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Timing
    stim_onset_ms_short = STIM_ONSET_MS - BURN_IN_MS   # 500 ms in short sim
    stim_offset_ms_short = stim_onset_ms_short + STIM_DURATION_MS
    delay_ms = float(args.delay_ms)
    T_ms_short = stim_offset_ms_short + delay_ms
    T_ms_full = BURN_IN_MS + T_ms_short

    n_grid = len(w_pv_values) * len(w_pyr_values) * len(amplitudes)
    n_burnin = len(condition_keys) * len(w_pv_values) * len(w_pyr_values)
    n_total_trials = n_grid * len(condition_keys) * n_trials

    print(f"\n3D Calibration sweep configuration:")
    print(f"  Conditions:       {', '.join(condition_keys)}")
    print(f"  w_pv_global:      {w_pv_values}")
    print(f"  w_pyr_pyr_inter:  {[f'{v:.4g}' for v in w_pyr_values]}")
    print(f"  amplitudes:       {amplitudes}")
    print(f"  Grid points:      {len(w_pv_values)} × {len(w_pyr_values)} × {len(amplitudes)} = {n_grid}")
    print(f"  Trials/point:     {n_trials}")
    print(f"  Total trials:     {n_total_trials} × {len(condition_keys)} cond")
    print(f"  delay_ms:         {delay_ms:.0f}")
    print(f"  Workers:          {n_workers}")
    print(f"  Sat threshold:    {CAL3D_SAT_THRESH_HZ} Hz")
    print(f"  Cue sat thresh:   {CAL3D_CUE_SAT_THRESH_HZ} Hz")
    print(f"  Output dir:       {out_dir}")

    # --- Cache check ---
    all_trial_results: list[dict] = []
    for ck in condition_keys:
        cond_csv = os.path.join(out_dir, f"cal3d_trials_{ck}.csv")
        if not no_cache and _cal3d_is_cached(cond_csv, w_pv_values, w_pyr_values,
                                              amplitudes, n_trials):
            print(f"\n  Cache hit: {ck} — loading {cond_csv}")
            all_trial_results.extend(_cal3d_load_cached(cond_csv))
        else:
            # Need to simulate this condition
            pass

    cached_conds = {r["cond_key"] for r in all_trial_results}
    conds_to_run = [ck for ck in condition_keys if ck not in cached_conds]

    if conds_to_run:
        # ----------------------------------------------------------------
        # Phase 1: Pre-compute burn-in states
        # ----------------------------------------------------------------
        print(f"\n--- Phase 1: Burn-in states ({n_burnin} per condition) ---")
        burnin_cache: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]] = {}
        resting_cache: dict[tuple[str, float, float], float] = {}
        connectivity_cache: dict[tuple[float, float], RingConnectivity] = {}
        ring_params_cache: dict[tuple[float, float], RingParams] = {}

        # Pre-build connectivity and ring params (reused across conditions)
        print("  Building connectivity matrices...")
        for w_pv in w_pv_values:
            for w_pyr in w_pyr_values:
                rp = RingParams(
                    n_nodes=args.n_nodes,
                    w_pyr_pyr_inter=float(w_pyr),
                    sigma_pyr_deg=sigma_deg,
                    w_pv_global=float(w_pv),
                )
                ring_params_cache[(w_pv, w_pyr)] = rp
                connectivity_cache[(w_pv, w_pyr)] = RingConnectivity.from_params(rp)

        burnin_total = len(conds_to_run) * len(w_pv_values) * len(w_pyr_values)
        import time as _time
        t0_burnin = _time.time()
        with tqdm(total=burnin_total, desc="Burn-in", unit="sim", smoothing=0) as pbar:
            for ck in conds_to_run:
                local_params = apply_condition(base_params, STUDY_CONDITIONS[ck])
                for w_pv in w_pv_values:
                    for w_pyr in w_pyr_values:
                        rp = ring_params_cache[(w_pv, w_pyr)]
                        conn = connectivity_cache[(w_pv, w_pyr)]
                        res = simulate_ring(
                            local_params, rp,
                            T_ms=BURN_IN_MS,
                            noise_type="white",
                            record_dt_ms=BURN_IN_MS,  # only final snapshot
                            connectivity=conn,
                            seed=args.seed,
                        )
                        burnin_cache[(ck, w_pv, w_pyr)] = (
                            res.r[-1].copy(), res.I_adapt_final.copy()
                        )
                        resting_cache[(ck, w_pv, w_pyr)] = float(np.mean(res.r[-1, :, 0]))
                        pbar.update()
        t_burnin = _time.time() - t0_burnin
        print(f"  Burn-in complete in {t_burnin:.1f}s "
              f"({t_burnin / burnin_total:.2f}s/sim)")

        # ----------------------------------------------------------------
        # Phase 2: Grid trials
        # ----------------------------------------------------------------
        print(f"\n--- Phase 2: Grid trials ---")
        trial_seeds = _generate_trial_seeds(args.seed, n_trials)
        local_params_by_cond = {
            ck: apply_condition(base_params, STUDY_CONDITIONS[ck])
            for ck in conds_to_run
        }

        jobs = [
            (ck, w_pv, w_pyr, amp, ti, int(trial_seeds[ti]))
            for ck in conds_to_run
            for w_pv in w_pv_values
            for w_pyr in w_pyr_values
            for amp in amplitudes
            for ti in range(n_trials)
        ]
        print(f"  {len(jobs)} trials across {len(conds_to_run)} conditions")

        init_args = (
            local_params_by_cond,
            ring_params_cache,
            connectivity_cache,
            burnin_cache,
            resting_cache,
            base_params.I_ext_pyr(),
            stim_onset_ms_short,
            stim_offset_ms_short,
            T_ms_short,
            record_dt_ms,
        )

        new_results: list[dict] = []
        t0_grid = _time.time()
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                mp_context=_MP_CONTEXT,
                max_workers=n_workers,
                initializer=_cal3d_init_worker,
                initargs=init_args,
            ) as executor:
                with tqdm(total=len(jobs), desc="Trials", unit="trial",
                          smoothing=0) as pbar:
                    job_iter = iter(jobs)
                    max_in_flight = max(1, n_workers * 4)
                    in_flight: dict = {}
                    for _ in range(min(max_in_flight, len(jobs))):
                        try:
                            job = next(job_iter)
                        except StopIteration:
                            break
                        fut = executor.submit(_cal3d_run_single, job)
                        in_flight[fut] = job
                    while in_flight:
                        for future in as_completed(list(in_flight.keys()), timeout=None):
                            in_flight.pop(future, None)
                            new_results.append(future.result())
                            pbar.update()
                            try:
                                job = next(job_iter)
                                fut = executor.submit(_cal3d_run_single, job)
                                in_flight[fut] = job
                            except StopIteration:
                                pass
                            break
        else:
            _cal3d_init_worker(*init_args)
            for job in tqdm(jobs, desc="Trials", unit="trial", smoothing=0):
                new_results.append(_cal3d_run_single(job))

        t_grid = _time.time() - t0_grid
        print(f"  Grid complete in {t_grid:.1f}s "
              f"({t_grid / max(len(jobs), 1):.2f}s/trial)")

        # ----------------------------------------------------------------
        # Save per-trial CSV per condition
        # ----------------------------------------------------------------
        for ck in conds_to_run:
            cond_results = [r for r in new_results if r["cond_key"] == ck]
            cond_csv = os.path.join(out_dir, f"cal3d_trials_{ck}.csv")
            fieldnames = [
                "cond_key", "w_pv", "w_pyr", "amp", "trial_idx", "seed",
                "resting_hz", "cue_peak_hz", "cue_saturated",
                "delay_rest_frac", "delay_bump_frac", "delay_sat_frac",
                "delay_mean_peak_hz", "bump_lo_hz",
            ]
            with open(cond_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in cond_results:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
            print(f"  Saved {len(cond_results)} trials → {cond_csv}")

        all_trial_results.extend(new_results)

    # ----------------------------------------------------------------
    # Phase 3: Aggregate per grid point
    # ----------------------------------------------------------------
    print("\n--- Phase 3: Aggregating results ---")
    # Structure: (ck, w_pyr, w_pv, amp) -> {metrics}
    agg: dict[tuple, dict] = {}
    for ck in condition_keys:
        for w_pyr in w_pyr_values:
            for w_pv in w_pv_values:
                for amp in amplitudes:
                    trials = [r for r in all_trial_results
                              if r["cond_key"] == ck
                              and abs(r["w_pyr"] - w_pyr) < 1e-9
                              and abs(r["w_pv"] - w_pv) < 1e-9
                              and abs(r["amp"] - amp) < 1e-9]
                    if not trials:
                        continue
                    def _mean(key):
                        vals = [r[key] for r in trials if not np.isnan(float(r.get(key, float("nan"))))]
                        return float(np.mean(vals)) if vals else float("nan")
                    cue_sats = [r["cue_saturated"] for r in trials]
                    agg[(ck, w_pyr, w_pv, amp)] = {
                        "cond_key": ck,
                        "w_pyr": w_pyr,
                        "w_pv": w_pv,
                        "amp": amp,
                        "n_trials": len(trials),
                        "resting_hz": _mean("resting_hz"),
                        "cue_peak_hz_mean": _mean("cue_peak_hz"),
                        "cue_sat_frac": float(np.mean(cue_sats)) if cue_sats else float("nan"),
                        "delay_rest_frac_mean": _mean("delay_rest_frac"),
                        "delay_bump_frac_mean": _mean("delay_bump_frac"),
                        "delay_sat_frac_mean": _mean("delay_sat_frac"),
                        "delay_mean_peak_hz_mean": _mean("delay_mean_peak_hz"),
                        "bump_lo_hz": _mean("bump_lo_hz"),
                        # Quality score: bump fraction penalized by saturation
                        "quality_score": _mean("delay_bump_frac") * (1.0 - _mean("delay_sat_frac"))
                            if not np.isnan(_mean("delay_bump_frac")) else float("nan"),
                    }

    # Save aggregated summary CSV (one row per grid point per condition)
    summary_csv = os.path.join(out_dir, "cal3d_summary.csv")
    agg_fieldnames = [
        "cond_key", "w_pyr", "w_pv", "amp", "n_trials",
        "resting_hz", "cue_peak_hz_mean", "cue_sat_frac",
        "delay_rest_frac_mean", "delay_bump_frac_mean", "delay_sat_frac_mean",
        "delay_mean_peak_hz_mean", "bump_lo_hz", "quality_score",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for d in sorted(agg.values(), key=lambda x: (x["cond_key"], x["w_pyr"], x["w_pv"], x["amp"])):
            writer.writerow({k: d.get(k, "") for k in agg_fieldnames})
    print(f"  Summary CSV: {summary_csv}")

    # Save JSON (easy to load programmatically)
    json_out = {
        "sweep_config": {
            "w_pv_values": w_pv_values,
            "w_pyr_values": w_pyr_values,
            "amplitude_values": amplitudes,
            "conditions": condition_keys,
            "n_trials": n_trials,
            "delay_ms": delay_ms,
            "sigma_deg": sigma_deg,
            "sat_thresh_hz": CAL3D_SAT_THRESH_HZ,
            "cue_sat_thresh_hz": CAL3D_CUE_SAT_THRESH_HZ,
            "resting_mult": CAL3D_RESTING_MULT,
        },
        "results": {
            f"{ck}_wpyr{w_pyr:.5g}_wpv{w_pv:.4g}_amp{amp:.3g}": d
            for (ck, w_pyr, w_pv, amp), d in agg.items()
        },
    }
    json_path = os.path.join(out_dir, "cal3d_summary.json")
    with open(json_path, "w") as f:
        _json.dump(json_out, f, indent=2)
    print(f"  Summary JSON: {json_path}")

    # ----------------------------------------------------------------
    # Phase 4: Figures
    # ----------------------------------------------------------------
    print("\n--- Phase 4: Generating figures ---")
    for ck in condition_keys:
        cond_label = STUDY_CONDITIONS[ck].name
        cond_dir = os.path.join(out_dir, ck)
        os.makedirs(cond_dir, exist_ok=True)

        # Per w_pyr slice
        for w_pyr in w_pyr_values:
            slice_data = {
                (w_pv, amp): agg.get((ck, w_pyr, w_pv, amp), {})
                for w_pv in w_pv_values
                for amp in amplitudes
            }
            wpyr_str = f"{w_pyr:.5g}".replace(".", "_")
            slice_dir = os.path.join(cond_dir, f"wpyr_{wpyr_str}")
            os.makedirs(slice_dir, exist_ok=True)
            fig = plot_3d_sweep_slice(
                slice_data,
                w_pyr=w_pyr,
                w_pv_values=w_pv_values,
                amplitude_values=amplitudes,
                sigma_deg=sigma_deg,
                condition_key=cond_label,
                n_trials=n_trials,
                save_path=os.path.join(slice_dir, "slice_heatmaps.png"),
            )
            plt.close(fig)
            print(f"  Slice figure: {slice_dir}/slice_heatmaps.png")

        # Summary across all w_pyr
        all_data_for_cond = {
            (w_pyr, w_pv, amp): agg.get((ck, w_pyr, w_pv, amp), {})
            for w_pyr in w_pyr_values
            for w_pv in w_pv_values
            for amp in amplitudes
        }
        fig = plot_3d_sweep_summary(
            all_data_for_cond,
            w_pyr_values=w_pyr_values,
            w_pv_values=w_pv_values,
            amplitude_values=amplitudes,
            sigma_deg=sigma_deg,
            condition_key=cond_label,
            n_trials=n_trials,
            save_path=os.path.join(cond_dir, "summary_best_bump.png"),
        )
        plt.close(fig)
        print(f"  Summary figure: {cond_dir}/summary_best_bump.png")

    # Print top results
    print("\n--- Top parameter sets by quality score ---")
    top = sorted(
        [d for d in agg.values() if not np.isnan(d.get("quality_score", float("nan")))],
        key=lambda d: d["quality_score"],
        reverse=True,
    )[:10]
    if top:
        print(f"  {'cond':6} {'w_pv':8} {'w_pyr':8} {'amp':6} "
              f"{'rest':6} {'bump%':6} {'sat%':6} {'score':6}")
        print("  " + "-" * 62)
        for d in top:
            print(f"  {d['cond_key']:6} {d['w_pv']:.4f}  {d['w_pyr']:.5g}  "
                  f"{d['amp']:.3f}  "
                  f"{d['resting_hz']:5.1f}  "
                  f"{d['delay_bump_frac_mean']:5.2f}  "
                  f"{d['delay_sat_frac_mean']:5.2f}  "
                  f"{d['quality_score']:5.2f}")

    print(f"\nDone. Output: {out_dir}")

    if not args.no_show:
        plt.show()


# ============================================================================
# NOISE FLOOR SUBCOMMAND
# ============================================================================

def cmd_noise_floor(args: argparse.Namespace) -> None:
    """Run noise floor estimation from no-stimulus baseline trials."""
    _resolve_seed(args)
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    ring_params_base = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    if args.conditions is None:
        condition_keys = ["WT"]
    else:
        if "all" in args.conditions:
            condition_keys = list(STUDY_CONDITIONS.keys())
        else:
            condition_keys = args.conditions
            for k in condition_keys:
                if k not in STUDY_CONDITIONS:
                    print(f"Error: unknown condition '{k}'.\n"
                          f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                    sys.exit(1)

    w_inter_values = args.w_inter_values
    n_baseline = args.n_baseline
    noise_percentile = args.noise_percentile
    replot_only = getattr(args, 'replot_only', False)
    no_cache = getattr(args, 'no_cache', False)
    batch_chunk_size = getattr(args, 'batch_chunk_size', 50)
    del batch_chunk_size
    n_workers = _resolve_workers(args)

    conn_label = _calibration_network_label(ring_params_base)
    out_dir = os.path.join(
        _output_dir("figs/ring/calibration", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    if replot_only:
        print("\nReplot-only mode: loading cached baseline CSVs")
        all_cond_noise_data: dict[str, dict] = {}
        all_cond_cap_hit_data: dict[str, dict[float, float]] = {}

        for ck in condition_keys:
            cond_label = STUDY_CONDITIONS[ck].name
            cond_dir = os.path.join(out_dir, ck)

            cached_nt, cached_base, saturated_w, cached_cap_frac = _load_calibrate_baseline(
                cond_dir, ck, w_inter_values, noise_percentile,
            )
            missing_w = [w for w in w_inter_values
                         if (ck, w) not in cached_nt and w not in saturated_w]
            if missing_w:
                print(f"  Incomplete noise thresholds for {cond_label}: "
                      f"missing w_inter={', '.join(_fmt(w) for w in missing_w)}")
                continue

            thresholds_for_plot = {w: cached_nt[(ck, w)]
                                   for w in w_inter_values if (ck, w) in cached_nt}
            baseline_for_plot = {
                w: cached_base.get((ck, w), np.array([]))
                for w in w_inter_values if (ck, w) in cached_nt
            }
            if any(len(v) > 0 for v in baseline_for_plot.values()):
                plot_noise_floor_histogram(
                    baseline_for_plot, thresholds_for_plot,
                    save_path=os.path.join(cond_dir, "noise_floor.png"),
                    suptitle=f"Noise Floor ({cond_label}, {n_baseline} trials, p{noise_percentile:.0f})",
                    skipped_w_values=sorted(saturated_w) if saturated_w else None,
                )
                plt.close()
                print(f"  Replotted per-condition noise floor: {cond_label}")
            else:
                print(f"  Skipping noise floor histogram for {cond_label} "
                      f"(baseline_A_hat.csv unavailable; only summary thresholds found)")

            all_cond_noise_data[ck] = thresholds_for_plot
            all_cond_cap_hit_data[ck] = {
                w: cached_cap_frac[(ck, w)]
                for w in w_inter_values
                if (ck, w) in cached_cap_frac
            }

        if all_cond_noise_data:
            n_cond_label = f"{len(all_cond_noise_data)} condition{'s' if len(all_cond_noise_data) > 1 else ''}"
            plot_noise_summary(
                all_cond_noise_data,
                cap_hit_fraction_data=all_cond_cap_hit_data,
                cap_warning_threshold=CAP_WARNING_FRACTION,
                cap_rate_hz=RATE_CAP_HZ,
                save_path=os.path.join(out_dir, "noise_summary.png"),
                suptitle=f"Noise Floor ({n_cond_label}, {n_baseline} baseline trials, p{noise_percentile:.0f})",
            )
            plt.close()
            print(f"Saved cross-condition noise summary: {os.path.join(out_dir, 'noise_summary.png')}")
        else:
            print("No noise plots were regenerated (missing or incomplete cache).")
        return

    # --- Cache check (per condition × w_inter) ---
    conditions_to_run: list[str] = []
    condition_missing_w: dict[str, list[float]] = {}
    condition_cached_w: dict[str, list[float]] = {}
    condition_cached_trials: dict[str, dict[float, int]] = {}
    cached_noise_thresholds: dict[tuple[str, float], float] = {}
    cached_baseline_data: dict[tuple[str, float], np.ndarray] = {}
    cached_cap_hit_fractions: dict[tuple[str, float], float] = {}
    trials_to_add_by_key: dict[tuple[str, float], int] = {}
    trial_start_idx_by_key: dict[tuple[str, float], int] = {}
    legacy_cache_conditions: list[str] = []

    if not no_cache:
        for ck in condition_keys:
            cond_dir_check = os.path.join(out_dir, ck)
            cached_nt, cached_base, _, cached_cap_frac = _load_calibrate_baseline(
                cond_dir_check, ck, w_inter_values, noise_percentile,
            )
            cached_noise_thresholds.update(cached_nt)
            cached_baseline_data.update(cached_base)
            cached_cap_hit_fractions.update(cached_cap_frac)

            trial_counts, has_trial_metadata = _load_baseline_trial_counts(cond_dir_check, ck)
            if cached_nt and not has_trial_metadata:
                legacy_cache_conditions.append(ck)

            cached_ws: list[float] = []
            missing_ws: list[float] = []
            per_cond_counts: dict[float, int] = {}

            for w in w_inter_values:
                key = (ck, w)
                # If trial metadata is unavailable (legacy cache), treat as 0 cached
                # to force a one-time rebuild with explicit trial indexing.
                cached_trials = int(trial_counts.get(key, 0)) if has_trial_metadata else 0
                if key not in cached_nt:
                    cached_trials = 0
                per_cond_counts[w] = cached_trials

                if cached_trials >= n_baseline:
                    cached_ws.append(w)
                else:
                    missing_ws.append(w)
                    trials_to_add_by_key[key] = n_baseline - cached_trials
                    trial_start_idx_by_key[key] = cached_trials

            condition_cached_w[ck] = cached_ws
            condition_missing_w[ck] = missing_ws
            condition_cached_trials[ck] = per_cond_counts
            if missing_ws:
                conditions_to_run.append(ck)
    else:
        conditions_to_run = list(condition_keys)
        condition_missing_w = {ck: list(w_inter_values) for ck in condition_keys}
        condition_cached_w = {ck: [] for ck in condition_keys}
        condition_cached_trials = {ck: {w: 0 for w in w_inter_values} for ck in condition_keys}
        trials_to_add_by_key = {(ck, w): n_baseline for ck in condition_keys for w in w_inter_values}
        trial_start_idx_by_key = {(ck, w): 0 for ck in condition_keys for w in w_inter_values}

    print(f"\nNoise floor configuration:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    if legacy_cache_conditions:
        print("  Legacy cache detected (no trial_idx metadata): "
              f"{', '.join(sorted(legacy_cache_conditions))} — rebuilding trial-indexed cache")
    if not no_cache:
        fully_cached = [ck for ck in condition_keys if len(condition_missing_w.get(ck, [])) == 0]
        partially_cached = [
            ck for ck in condition_keys
            if len(condition_cached_w.get(ck, [])) > 0 and len(condition_missing_w.get(ck, [])) > 0
        ]
        if fully_cached:
            print(f"  Cache hit (full): {', '.join(sorted(fully_cached))} — skipping simulation")
        for ck in partially_cached:
            cond_label = STUDY_CONDITIONS[ck].name
            missing_fmt = ', '.join(_fmt(w) for w in condition_missing_w[ck])
            print(f"  Cache hit (partial): {cond_label} — simulating missing w_inter: {missing_fmt}")
    if conditions_to_run:
        print(f"  To simulate: {', '.join(conditions_to_run)}")
    print(f"  w_inter values: {', '.join(_fmt(w) for w in w_inter_values)}")
    print(f"  Baseline trials per w_inter: {n_baseline}")
    requested_trials_total = len(condition_keys) * len(w_inter_values) * n_baseline
    trials_cached = 0
    for ck in condition_keys:
        for w in w_inter_values:
            trials_cached += min(int(condition_cached_trials.get(ck, {}).get(w, 0)), n_baseline)
    trials_to_run = requested_trials_total - trials_cached
    print(f"  Baseline trials: {trials_to_run} to run, {trials_cached} cached")
    print(f"  Noise percentile: p{noise_percentile:.0f}")
    print(f"  Delay = {args.delay_ms:.0f} ms")
    print(f"  Workers: {n_workers}")
    if conditions_to_run:
        total_sims = sum(trials_to_add_by_key.get((ck, w), 0)
                         for ck in condition_keys for w in w_inter_values)
        print(f"  Total simulations: {total_sims}")

    # Containers
    baseline_A_hat_data: dict[tuple[str, float], np.ndarray] = dict(cached_baseline_data)
    noise_thresholds: dict[tuple[str, float], float] = dict(cached_noise_thresholds)
    cap_hit_fraction_data: dict[tuple[str, float], float] = dict(cached_cap_hit_fractions)

    if conditions_to_run:
        new_nt, new_base, new_cap_frac = _run_noise_floor_for_conditions(
            conditions_to_run=conditions_to_run,
            w_inter_values=w_inter_values,
            ring_params_base=ring_params_base,
            base_params=base_params,
            n_baseline=n_baseline,
            noise_percentile=noise_percentile,
            out_dir=out_dir,
            n_workers=n_workers,
            batch_chunk_size=1,
            seed=args.seed,
            delay_ms=args.delay_ms,
            record_dt_ms=getattr(args, 'record_dt_ms', 5.0),
            w_inter_values_by_condition=condition_missing_w,
            trials_to_add_by_key=trials_to_add_by_key,
            trial_start_idx_by_key=trial_start_idx_by_key,
            preserve_existing_cache=not no_cache,
        )
        noise_thresholds.update(new_nt)
        baseline_A_hat_data.update(new_base)
        cap_hit_fraction_data.update(new_cap_frac)

    # Report cached baselines
    if not no_cache:
        for ck in condition_keys:
            cond_label = STUDY_CONDITIONS[ck].name
            for w in condition_cached_w.get(ck, []):
                key = (ck, w)
                nt = noise_thresholds.get(key, float('nan'))
                n_samples = len(baseline_A_hat_data.get(key, []))
                print(f"  {cond_label}, w={w:.2f}: threshold = {nt:.4f} "
                      f"(p{noise_percentile:.0f}, n={n_samples}) [cached]")

    # --- Plots ---
    all_cond_noise_data: dict[str, dict] = {}
    all_cond_cap_hit_data: dict[str, dict[float, float]] = {}
    for ck in condition_keys:
        cond_label = STUDY_CONDITIONS[ck].name
        cond_dir = os.path.join(out_dir, ck)
        os.makedirs(cond_dir, exist_ok=True)

        saturated_w_cond = [w for w in w_inter_values if (ck, w) not in noise_thresholds]
        baseline_for_plot = {w: baseline_A_hat_data.get((ck, w), np.array([]))
                             for w in w_inter_values if (ck, w) in noise_thresholds}
        thresholds_for_plot = {w: noise_thresholds[(ck, w)]
                               for w in w_inter_values if (ck, w) in noise_thresholds}

        if any(len(v) > 0 for v in baseline_for_plot.values()):
            plot_noise_floor_histogram(
                baseline_for_plot, thresholds_for_plot,
                save_path=os.path.join(cond_dir, "noise_floor.png"),
                suptitle=f"Noise Floor ({cond_label}, {n_baseline} trials, p{noise_percentile:.0f})",
                skipped_w_values=saturated_w_cond if saturated_w_cond else None,
            )
            plt.close()
        else:
            print(f"  Skipping noise floor histogram for {cond_label} (re-run to generate)")

        all_cond_noise_data[ck] = thresholds_for_plot
        all_cond_cap_hit_data[ck] = {
            w: cap_hit_fraction_data[(ck, w)]
            for w in w_inter_values
            if (ck, w) in cap_hit_fraction_data
        }

    n_cond_label = f"{len(condition_keys)} condition{'s' if len(condition_keys) > 1 else ''}"
    plot_noise_summary(
        all_cond_noise_data,
        cap_hit_fraction_data=all_cond_cap_hit_data,
        cap_warning_threshold=CAP_WARNING_FRACTION,
        cap_rate_hz=RATE_CAP_HZ,
        save_path=os.path.join(out_dir, "noise_summary.png"),
        suptitle=f"Noise Floor ({n_cond_label}, {n_baseline} baseline trials, p{noise_percentile:.0f})",
    )
    plt.close()
    print(f"\nNoise floor estimation complete.")
    print(f"Results saved to: {out_dir}")


# ============================================================================
# NOISE THRESHOLD LOOKUP HELPERS
# ============================================================================

def _lookup_noise_threshold(
    csv_path: str,
    cond_key: str,
    amplitude: float,
    w_inter: float,
) -> Optional[float]:
    """Read noise_threshold from a calibration_summary.csv for matching parameters.

    First tries to match on condition_key + amplitude + w_inter.  If no
    condition-specific row is found, falls back to any row matching amplitude
    and w_inter (the noise floor is primarily a network-parameter property,
    not a condition property, so cross-condition reuse is a reasonable proxy).
    Returns None if the file is missing, unreadable, or has no matching row.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        fallback: Optional[float] = None
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                amp_match = abs(float(row['amplitude']) - amplitude) < 1e-4
                w_match = abs(float(row['w_inter']) - w_inter) < 1e-4
                if amp_match and w_match:
                    if row.get('condition_key', '').strip() == cond_key:
                        return float(row['noise_threshold'])
                    if fallback is None:
                        fallback = float(row['noise_threshold'])
        return fallback  # None if no amp/w match at all
    except Exception:
        pass
    return None


def _lookup_noise_threshold_exact(
    csv_path: str,
    cond_key: str,
    amplitude: float,
    w_inter: float,
) -> Optional[float]:
    """Like _lookup_noise_threshold but only returns a condition-specific match."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if (row.get('condition_key', '').strip() == cond_key
                        and abs(float(row['amplitude']) - amplitude) < 1e-4
                        and abs(float(row['w_inter']) - w_inter) < 1e-4):
                    return float(row['noise_threshold'])
    except Exception:
        pass
    return None


def cmd_asymmetry(args: argparse.Namespace) -> None:
    """Run L/R asymmetry analysis across conditions.

    Each trial starts from zero initial conditions and runs its own independent
    noisy burn-in (ASYM_SETTLING_MS) with a unique seed, so pre-cue spontaneous
    states are fully uncorrelated across trials.  The pre-cue and delay
    asymmetry are measured per trial and visualised as:

      asymmetry_distribution.png  – violin/strip of pre-cue & delay asymmetry
      asymmetry_correlation.png   – scatter: pre-cue vs delay asymmetry
      asymmetry_summary.png       – mean, balance, and magnitude bar charts
      worst_case/{cond}/          – dashboard + bump metrics + animation for
                                    the trial with the largest |delay asymmetry|
      asymmetry_trials.csv        – raw per-trial data
    """
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    condition_keys = args.conditions if args.conditions else ['WT', 'WT_APP', 'a7_KO_APP']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'. "
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for config display

    amp = args.amplitude[0]
    n_trials = args.n_trials
    n_workers = _resolve_workers(args)
    random_cue_location: bool = getattr(args, 'random_cue_location', False)
    balance_cue: bool = not getattr(args, 'no_cue_balance', False)
    correct_asymmetry: bool = getattr(args, 'correct_asymmetry', True)

    conn_label = _calibration_network_label(base_rp)
    asym_mode_label = "corrected" if correct_asymmetry else "uncorrected"
    amp_label = f"amp{amp:g}_{asym_mode_label}"
    out_dir = os.path.join(
        _output_dir("figs/ring/asymmetry", args.params_json),
        conn_label,
        amp_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- Even-N warning and cue placement diagnostics ---
    N = ring_params.n_nodes
    even_n = (N % 2 == 0)
    if even_n and not random_cue_location:
        if balance_cue:
            _effective_cue = _balance_cue_location(STIM_CENTER_DEG, ring_params)
            _balance_note = (
                f"  [N={N} is even] Cue placed at {_effective_cue:.4f}° "
                f"(half-step between nodes) to balance left/right counts."
            )
        else:
            _bias = -1.0 / (N - 1)
            _balance_note = (
                f"  WARNING: N={N} is even and --no_cue_balance is set. "
                f"Cue at {STIM_CENTER_DEG:.1f}° falls exactly on a node → "
                f"structural pre-cue bias ≈ {_bias:.4f} (left has one extra node)."
            )
    else:
        _balance_note = None

    if random_cue_location:
        cue_label = "random [0°, 360°)  (no balance correction needed)"
        _cue_title = "cue@random"
    elif balance_cue:
        _eff = _balance_cue_location(STIM_CENTER_DEG, ring_params)
        _strategy = "between nodes" if even_n else "on nearest node"
        cue_label = f"{_eff:.4f}° (balanced, {_strategy})"
        _cue_title = f"cue@{_eff:.2f}° (balanced)"
    else:
        cue_label = f"{STIM_CENTER_DEG:.1f}° (raw, no balance)"
        _cue_title = f"cue@{STIM_CENTER_DEG:.0f}° (unbalanced)"

    _asym_correction = ("on (weighted: Σ[A(t)·Amp(t)] / Σ[Amp(t)])"
                        if correct_asymmetry else "off (raw mean of A(t))")
    _asym_info = [
        f"Conditions:          {', '.join(condition_keys)}",
        f"Trials:              {n_trials}   seed={args.seed}   workers={n_workers}",
        f"Timing:              burn-in={ASYM_SETTLING_MS:.0f} ms"
        f"   pre-cue window={ASYM_PRE_CUE_WINDOW_MS:.0f} ms"
        f"   delay={args.delay_ms:.0f} ms",
        f"Cue location:        {cue_label}",
        f"Asymmetry correction:{_asym_correction}",
    ]
    if _balance_note:
        _asym_info.append(_balance_note.strip())
    _print_config(args, amp, base_params, 0.0, ring_params, experiment_info=_asym_info,
                  save_path=os.path.join(out_dir, "experiment_config.txt"))

    # --- CSV cache: load existing trials if parameters match ---
    csv_path = os.path.join(out_dir, "asymmetry_trials.csv")
    all_results: list[dict] = []
    cached_indices: dict[str, set] = {ck: set() for ck in condition_keys}

    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='') as _f:
                cached_rows = list(csv.DictReader(_f))
            if cached_rows and 'delay_ms' in cached_rows[0]:
                # Validate simulation params match
                params_ok = all(
                    abs(float(r.get('delay_ms', 0)) - args.delay_ms) < 1e-6
                    and abs(float(r.get('amplitude', 0)) - amp) < 1e-9
                    for r in cached_rows
                )
                # Validate cue mode: check random_cue and balance_cue flags match
                if params_ok and 'random_cue' in cached_rows[0]:
                    cached_random = bool(int(cached_rows[0].get('random_cue', 0)))
                    cached_balance = bool(int(cached_rows[0].get('balance_cue', 1)))
                    if cached_random != random_cue_location or cached_balance != balance_cue:
                        params_ok = False
                if params_ok and 'correct_asymmetry' in cached_rows[0]:
                    cached_correct = bool(int(cached_rows[0].get('correct_asymmetry', 1)))
                    if cached_correct != correct_asymmetry:
                        params_ok = False
                elif params_ok:
                    # Backward compatibility: legacy CSVs may not include
                    # 'correct_asymmetry'. Infer mode from folder suffix.
                    # - amp*_uncorrected -> raw asymmetry cache
                    # - amp*_corrected   -> corrected asymmetry cache
                    # - no suffix        -> legacy raw cache
                    amp_dir_name = os.path.basename(out_dir)
                    if amp_dir_name.endswith("_uncorrected"):
                        cached_correct = False
                    elif amp_dir_name.endswith("_corrected"):
                        cached_correct = True
                    else:
                        cached_correct = False
                    if cached_correct != correct_asymmetry:
                        params_ok = False
            else:
                params_ok = False  # old format — no validation columns
            if params_ok:
                for r in cached_rows:
                    ck = r['condition']
                    if ck not in condition_keys:
                        continue
                    all_results.append({
                        'cond_key': ck,
                        'trial_idx': int(r['trial_idx']),
                        'seed': int(r['seed']),
                        'cue_deg': float(r.get('cue_deg', STIM_CENTER_DEG)),
                        'pre_cue_asym': float(r['pre_cue_asym']),
                        'last_pre_cue_asym': float(r['last_pre_cue_asym']) if r.get('last_pre_cue_asym', '') != '' else float('nan'),
                        'delay_asym': float(r['delay_asym']),
                        'mean_abs_asym': float(r['mean_abs_asym']) if r.get('mean_abs_asym', '') != '' else float('nan'),
                        'asym_std': float(r['asym_std']) if r.get('asym_std', '') != '' else float('nan'),
                        'mean_abs_asym_precue': float(r['mean_abs_asym_precue']) if r.get('mean_abs_asym_precue', '') != '' else float('nan'),
                        'asym_std_precue': float(r['asym_std_precue']) if r.get('asym_std_precue', '') != '' else float('nan'),
                    })
                    cached_indices[ck].add(int(r['trial_idx']))
                n_cached = sum(len(v) for v in cached_indices.values())
                if n_cached > 0:
                    print(f"\nLoaded {n_cached} cached trial(s) from {csv_path}")
                    for ck in condition_keys:
                        print(f"  {ck}: {len(cached_indices[ck])} / {n_trials}")
            else:
                print("\nCache parameters mismatch — rerunning all trials.")
        except Exception as _e:
            print(f"\nCache read failed ({_e}) — rerunning all trials.")
            all_results = []
            cached_indices = {ck: set() for ck in condition_keys}

    # --- Build remaining trial jobs (skip already cached) ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    jobs = [
        (cond_key, trial_idx, seed)
        for cond_key in condition_keys
        for trial_idx, seed in enumerate(trial_seeds)
        if trial_idx not in cached_indices[cond_key]
    ]

    # --- Run new trials (parallel or sequential) ---
    new_results: list[dict] = []
    if jobs:
        init_args = (
            base_params, per_cond_rp, per_cond_conn,
            amp, args.delay_ms, args.record_dt_ms,
            random_cue_location, balance_cue, correct_asymmetry,
        )
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT, 
                max_workers=n_workers,
                initializer=_asym_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_asym_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        new_results.append(future.result())
                        pbar.update()
        else:
            _asym_init_worker(*init_args)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                new_results.append(_asym_run_single(job))
        all_results.extend(new_results)
    else:
        print("\nAll trials already cached — skipping simulations.")

    # --- Organise by condition ---
    data_by_condition: dict = {}
    worst_by_condition: dict = {}

    for cond_key in condition_keys:
        trials = sorted(
            [r for r in all_results if r['cond_key'] == cond_key],
            key=lambda r: r['trial_idx'],
        )
        pre_cue = np.array([t['pre_cue_asym'] for t in trials])
        delay = np.array([t['delay_asym'] for t in trials])
        data_by_condition[cond_key] = {
            'pre_cue': pre_cue,
            'last_pre_cue': np.array([t.get('last_pre_cue_asym', float('nan')) for t in trials]),
            'delay': delay,
            'mean_abs_asym': np.array([t.get('mean_abs_asym', float('nan')) for t in trials]),
            'asym_std': np.array([t.get('asym_std', float('nan')) for t in trials]),
            'mean_abs_asym_precue': np.array([t.get('mean_abs_asym_precue', float('nan')) for t in trials]),
            'asym_std_precue': np.array([t.get('asym_std_precue', float('nan')) for t in trials]),
        }

        worst_idx = int(np.argmax(np.abs(delay)))
        worst_by_condition[cond_key] = trials[worst_idx]

    # --- Save / update CSV (only when new trials were run) ---
    if new_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition', 'trial_idx', 'seed', 'cue_deg',
                'pre_cue_asym', 'last_pre_cue_asym', 'delay_asym', 'delay_ms', 'amplitude',
                'random_cue', 'balance_cue', 'correct_asymmetry',
                'mean_abs_asym', 'asym_std',
                'mean_abs_asym_precue', 'asym_std_precue',
            ])
            writer.writeheader()
            for r in sorted(all_results, key=lambda r: (r['cond_key'], r['trial_idx'])):
                writer.writerow({
                    'condition': r['cond_key'],
                    'trial_idx': r['trial_idx'],
                    'seed': r['seed'],
                    'cue_deg': r.get('cue_deg', STIM_CENTER_DEG),
                    'pre_cue_asym': r['pre_cue_asym'],
                    'last_pre_cue_asym': r.get('last_pre_cue_asym', float('nan')),
                    'delay_asym': r['delay_asym'],
                    'delay_ms': args.delay_ms,
                    'amplitude': amp,
                    'random_cue': int(random_cue_location),
                    'balance_cue': int(balance_cue),
                    'correct_asymmetry': int(correct_asymmetry),
                    'mean_abs_asym': r.get('mean_abs_asym', float('nan')),
                    'asym_std': r.get('asym_std', float('nan')),
                    'mean_abs_asym_precue': r.get('mean_abs_asym_precue', float('nan')),
                    'asym_std_precue': r.get('asym_std_precue', float('nan')),
                })
        print(f"\nTrial data → {csv_path}")

    # --- Statistical tests: delay asymmetry vs. 0 ---
    from scipy import stats as _scipy_stats

    def _sig_label(p) -> str:
        if p is None or np.isnan(p):
            return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    # --- One-sample tests vs 0 for both pre-cue and delay ---
    stats_by_condition: dict[str, dict] = {}
    hdr = f"  {'Condition':<14}  {'n':>4}  {'mean':>8}  {'t':>7}  {'p(t)':>8}  {'W':>8}  {'p(W)':>8}"
    for period_key, period_label in [('pre_cue', 'Pre-cue'), ('delay', 'Delay')]:
        print(f"\nStatistical tests — {period_label} asymmetry vs. 0 (one-sample):")
        print(hdr)
        print("  " + "-" * 68)
        for cond_key in condition_keys:
            vals = data_by_condition[cond_key][period_key]
            n = len(vals)
            mean = float(np.mean(vals))
            t_stat, p_t = _scipy_stats.ttest_1samp(vals, 0.0)
            if n >= 10:
                w_stat, p_w = _scipy_stats.wilcoxon(vals, alternative='two-sided')
            else:
                w_stat, p_w = np.nan, np.nan
            stars_t = _sig_label(p_t)
            stars_w = _sig_label(p_w if not np.isnan(p_w) else None)
            p_w_str = f"{p_w:.4f} {stars_w:<3}" if not np.isnan(p_w) else "    n/a   "
            print(f"  {cond_key:<14}  {n:>4}  {mean:>+8.4f}  {t_stat:>+7.3f}  "
                  f"{p_t:.4f} {stars_t:<3}  {w_stat:>8.1f}  {p_w_str}")
            if cond_key not in stats_by_condition:
                stats_by_condition[cond_key] = {}
            stats_by_condition[cond_key][period_key] = {
                'n': n, 'mean': mean,
                't_stat': float(t_stat), 'p_t': float(p_t),
                'w_stat': float(w_stat) if not np.isnan(w_stat) else None,
                'p_w': float(p_w) if not np.isnan(p_w) else None,
            }
        print("  (* p<0.05  ** p<0.01  *** p<0.001)")

    # --- Pairwise tests: asymmetry magnitude between conditions, both periods + new metrics ---
    pairwise_stats: list[dict] = []
    if len(condition_keys) >= 2:
        # Signed-magnitude tests for pre-cue / delay (existing behaviour: compare |scalar|)
        for period_key, period_label in [('delay', 'Delay'), ('pre_cue', 'Pre-cue')]:
            print(f"\nStatistical tests — pairwise |asymmetry| {period_label} (Mann-Whitney U):")
            print(f"  {'Cond A':<14}  {'Cond B':<14}  {'n_A':>4}  {'n_B':>4}  {'U':>8}  {'p(U)':>10}")
            print("  " + "-" * 70)
            for i, ck_a in enumerate(condition_keys):
                for j, ck_b in enumerate(condition_keys):
                    if j <= i:
                        continue
                    abs_a = np.abs(data_by_condition[ck_a][period_key])
                    abs_b = np.abs(data_by_condition[ck_b][period_key])
                    u_stat, p_u = _scipy_stats.mannwhitneyu(abs_a, abs_b, alternative='two-sided')
                    stars = _sig_label(p_u)
                    print(f"  {ck_a:<14}  {ck_b:<14}  {len(abs_a):>4}  {len(abs_b):>4}  "
                          f"{u_stat:>8.1f}  {p_u:.4f} {stars:<3}")
                    pairwise_stats.append({
                        'period': period_key,
                        'cond_a': ck_a, 'cond_b': ck_b,
                        'n_a': len(abs_a), 'n_b': len(abs_b),
                        'u_stat': float(u_stat), 'p_u': float(p_u),
                    })
            print("  (* p<0.05  ** p<0.01  *** p<0.001)")

        # Pairwise tests for the temporal metrics (delay and pre-cue)
        for metric_key, metric_label in [
            ('mean_abs_asym', 'Mean|A(t)| — Delay'),
            ('asym_std', 'Std(A(t)) — Delay'),
            ('mean_abs_asym_precue', 'Mean|A(t)| — Pre-cue'),
            ('asym_std_precue', 'Std(A(t)) — Pre-cue'),
        ]:
            vals_by_cond = {ck: data_by_condition[ck].get(metric_key, np.array([]))
                            for ck in condition_keys}
            # Skip if all NaN (old CSV without these columns)
            if all(np.all(np.isnan(v)) for v in vals_by_cond.values()):
                continue
            print(f"\nStatistical tests — pairwise {metric_label} (Mann-Whitney U):")
            print(f"  {'Cond A':<14}  {'Cond B':<14}  {'n_A':>4}  {'n_B':>4}  {'U':>8}  {'p(U)':>10}")
            print("  " + "-" * 70)
            for i, ck_a in enumerate(condition_keys):
                for j, ck_b in enumerate(condition_keys):
                    if j <= i:
                        continue
                    va = vals_by_cond[ck_a]
                    vb = vals_by_cond[ck_b]
                    va = va[~np.isnan(va)]
                    vb = vb[~np.isnan(vb)]
                    if len(va) < 2 or len(vb) < 2:
                        continue
                    u_stat, p_u = _scipy_stats.mannwhitneyu(va, vb, alternative='two-sided')
                    stars = _sig_label(p_u)
                    print(f"  {ck_a:<14}  {ck_b:<14}  {len(va):>4}  {len(vb):>4}  "
                          f"{u_stat:>8.1f}  {p_u:.4f} {stars:<3}")
                    pairwise_stats.append({
                        'period': metric_key,
                        'cond_a': ck_a, 'cond_b': ck_b,
                        'n_a': len(va), 'n_b': len(vb),
                        'u_stat': float(u_stat), 'p_u': float(p_u),
                    })
            print("  (* p<0.05  ** p<0.01  *** p<0.001)")

    # --- Save text statistics report ---
    def _fmt_onesample(s):
        p_t_str = f"{s['p_t']:.4f} {_sig_label(s['p_t']):<4}"
        if s['w_stat'] is not None:
            return (f"{s['n']:>4}  {s['mean']:>+8.4f}  {s['t_stat']:>+7.3f}  "
                    f"{p_t_str}  {s['w_stat']:>8.1f}  {s['p_w']:.4f} {_sig_label(s['p_w']):<4}")
        return (f"{s['n']:>4}  {s['mean']:>+8.4f}  {s['t_stat']:>+7.3f}  "
                f"{p_t_str}  {'n/a':>8}  {'n/a':<9}")

    stats_txt_path = os.path.join(out_dir, "asymmetry_stats.txt")
    with open(stats_txt_path, 'w') as _f:
        _f.write(
            f"Asymmetry Statistical Report — amp {amp:g}× "
            f"({'corrected' if correct_asymmetry else 'raw'})\n"
        )
        _f.write("=" * 60 + "\n\n")
        col_hdr = f"  {'Condition':<14}  {'n':>4}  {'mean':>8}  {'t':>7}  {'p(t)':>10}  {'W':>8}  {'p(W)':>10}\n"
        sep = "  " + "-" * 74 + "\n"
        for period_key, period_label in [('pre_cue', 'Pre-cue'), ('delay', 'Delay')]:
            _f.write(f"One-sample tests — {period_label} asymmetry vs. 0\n")
            _f.write(col_hdr)
            _f.write(sep)
            for ck in condition_keys:
                s = stats_by_condition[ck][period_key]
                _f.write(f"  {ck:<14}  {_fmt_onesample(s)}\n")
            _f.write("  (* p<0.05  ** p<0.01  *** p<0.001)\n\n")
        if pairwise_stats:
            for period_key, period_label in [('delay', 'Delay'), ('pre_cue', 'Pre-cue')]:
                _f.write(f"Pairwise tests — |asymmetry| {period_label} (Mann-Whitney U)\n")
                _f.write(f"  {'Cond A':<14}  {'Cond B':<14}  {'n_A':>4}  {'n_B':>4}  {'U':>8}  {'p(U)':>10}\n")
                _f.write("  " + "-" * 70 + "\n")
                for pw in pairwise_stats:
                    if pw['period'] != period_key:
                        continue
                    p_str = f"{pw['p_u']:.4f} {_sig_label(pw['p_u']):<4}"
                    _f.write(f"  {pw['cond_a']:<14}  {pw['cond_b']:<14}  "
                             f"{pw['n_a']:>4}  {pw['n_b']:>4}  {pw['u_stat']:>8.1f}  {p_str}\n")
                _f.write("  (* p<0.05  ** p<0.01  *** p<0.001)\n\n")
    print(f"Statistical report saved to {stats_txt_path}")

    # --- Summary figures ---
    from .plotting import (
        plot_asymmetry_distribution,
        plot_asymmetry_correlation,
        plot_asymmetry_summary,
        plot_bump_metrics_over_time,
        plot_ring_dashboard,
        animate_ring_snapshot_evolution,
    )

    corr_label = "asymmetry corrected" if correct_asymmetry else "asymmetry raw"
    title_suffix = f" — amp {amp:g}×, {_cue_title}, {corr_label}"

    plot_asymmetry_distribution(
        data_by_condition, condition_keys,
        save_path=os.path.join(out_dir, "asymmetry_distribution.png"),
        title_suffix=title_suffix,
        stats_by_condition=stats_by_condition,
    )
    plt.close()

    plot_asymmetry_correlation(
        data_by_condition, condition_keys,
        save_path=os.path.join(out_dir, "asymmetry_correlation.png"),
        title_suffix=title_suffix,
    )
    plt.close()

    plot_asymmetry_summary(
        data_by_condition, condition_keys,
        save_path=os.path.join(out_dir, "asymmetry_summary.png"),
        title_suffix=title_suffix,
        stats_by_condition=stats_by_condition,
        pairwise_stats=pairwise_stats,
    )
    plt.close()

    print("Summary figures saved.")

    # --- Worst-case visualisations (per condition) ---
    stim_onset = ASYM_SETTLING_MS
    stim_offset = stim_onset + STIM_DURATION_MS
    T_ms = stim_offset + args.delay_ms
    actual_current = amp * base_params.I_ext_pyr()

    # Display time: t=0 = cue onset; show 500 ms pre-cue through end of delay
    t_offset_disp = ASYM_SETTLING_MS
    time_range = (ASYM_SETTLING_MS - ASYM_PRE_CUE_WINDOW_MS, T_ms)

    export_mp4 = not getattr(args, "no_snapshot_mp4", False)
    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
    mp4_pbar = None
    if export_mp4:
        total_videos = len(condition_keys)
        mp4_pbar = _start_mp4_progress(
            total_videos=total_videos,
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
            sample_time_range=time_range,
        )
    mp4_pbar = None
    if export_mp4:
        total_videos = len(condition_keys)
        mp4_pbar = _start_mp4_progress(
            total_videos=total_videos,
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
            sample_time_range=time_range,
        )

    for cond_key in condition_keys:
        worst = worst_by_condition[cond_key]
        cond_dir = os.path.join(out_dir, "worst_case", cond_key)
        os.makedirs(cond_dir, exist_ok=True)

        worst_cue_deg = worst.get('cue_deg', STIM_CENTER_DEG)
        print(f"\nWorst-case ({cond_key}): trial {worst['trial_idx']}, "
              f"seed {worst['seed']}, cue@{worst_cue_deg:.1f}°, "
              f"delay_asym = {worst['delay_asym']:+.3f}")

        # Re-run worst trial with full recording (same seed → same independent burn-in)
        local_params_wc = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
        stimuli_worst = [RingStimulus(
            center_deg=worst_cue_deg, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=stim_onset, duration_ms=STIM_DURATION_MS,
        )]

        result_worst = simulate_ring(
            local_params_wc, per_cond_rp[cond_key], T_ms=T_ms,
            stimuli=stimuli_worst, seed=worst['seed'],
            connectivity=per_cond_conn[cond_key],
            record_dt_ms=args.record_dt_ms,
            record_adaptation=True,
        )

        side = "right" if worst['delay_asym'] > 0 else "left"
        suptitle = (
            f"{STUDY_CONDITIONS[cond_key].name} — worst-case trial "
            f"(amp {amp:g}×, {_cue_title}, {corr_label}, "
            f"delay asym = {worst['delay_asym']:+.3f}, {side}ward)"
        )

        # Dashboard
        plot_ring_dashboard(
            result_worst,
            save_path=os.path.join(cond_dir, "dashboard.png"),
            time_range=time_range, t_offset=t_offset_disp,
            suptitle=suptitle,
        )
        plt.close()

        # Bump metrics over time (includes asymmetry panel)
        plot_bump_metrics_over_time(
            result_worst, time_range=time_range, t_offset=t_offset_disp,
        )
        plt.suptitle(suptitle, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(cond_dir, "bump_metrics.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Snapshot evolution animation
        if export_mp4:
            anim_path = os.path.join(cond_dir, "snapshot_evolution.mp4")
            mp4_pbar.set_postfix_str(f"cond={cond_key}")
            try:
                fig_anim, _ = animate_ring_snapshot_evolution(
                    result_worst,
                    save_path=anim_path,
                    time_range=time_range,
                    t_offset=t_offset_disp,
                    frame_step_ms=args.snapshot_anim_step_ms,
                    fps=args.snapshot_anim_fps,
                    suptitle=f"{STUDY_CONDITIONS[cond_key].name} — worst-case",
                    show_asymmetry=True,
                    **anim_quality_kwargs,
                )
                plt.close(fig_anim)
                mp4_pbar.update(1)
            except Exception as exc:
                print(f"  Warning: animation failed: {exc}")
        if export_mp4:
            anim_path = os.path.join(cond_dir, "snapshot_evolution.mp4")
            mp4_pbar.set_postfix_str(f"cond={cond_key}")
            try:
                fig_anim, _ = animate_ring_snapshot_evolution(
                    result_worst,
                    save_path=anim_path,
                    time_range=time_range,
                    t_offset=t_offset_disp,
                    frame_step_ms=args.snapshot_anim_step_ms,
                    fps=args.snapshot_anim_fps,
                    suptitle=f"{STUDY_CONDITIONS[cond_key].name} — worst-case",
                    show_asymmetry=True,
                    **anim_quality_kwargs,
                )
                plt.close(fig_anim)
                mp4_pbar.update(1)
            except Exception as exc:
                print(f"  Warning: animation failed: {exc}")

        del result_worst

    if mp4_pbar is not None:
        mp4_pbar.close()
    if mp4_pbar is not None:
        mp4_pbar.close()

    print(f"\nAll outputs saved to {out_dir}/")
    print(f"\nFigure saved to {out_dir}/temporal_dissection.png")


# ============================================================================
# BURN-IN STABILITY: PARALLEL WORKER
# ============================================================================

_burnin_stability_sim_args: Optional[dict] = None


def _burnin_stability_init_worker(
    base_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    burnin_ms: float,
    period_ms: float,
    n_periods: int,
    ref_deg: float,
    record_dt_ms: float,
) -> None:
    """Initialise worker process for burn-in stability trials."""
    global _burnin_stability_sim_args
    _burnin_stability_sim_args = {
        'base_params': base_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
        'burnin_ms': burnin_ms,
        'period_ms': period_ms,
        'n_periods': n_periods,
        'ref_deg': ref_deg,
        'record_dt_ms': record_dt_ms,
    }


def _burnin_stability_run_single(job: tuple) -> list[dict]:
    """Run one burn-in stability trial: noisy spontaneous activity from zero IC.

    Returns a list of per-window metric dicts (one entry per 1000ms window).
    """
    global _burnin_stability_sim_args
    from .analysis import compute_bump_asymmetry, population_vector_decode

    cfg = _burnin_stability_sim_args
    cond_key, trial_idx, seed = job

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(cfg['base_params'], condition)

    result = simulate_ring(
        local_params, cfg['ring_params'],
        T_ms=cfg['burnin_ms'],
        stimuli=None, r0=None, I_adapt0=None,
        seed=seed, noise_type='white',
        connectivity=cfg['connectivity'],
        record_dt_ms=cfg['record_dt_ms'],
    )

    # Set fixed reference angle for asymmetry (no stimulus → manual reference)
    result.stim_angle_deg = cfg['ref_deg']

    asym = compute_bump_asymmetry(result, population=0)  # (n_steps,)
    angles_rad = np.deg2rad(cfg['ring_params'].node_angles_deg)

    rows = []
    for w in range(cfg['n_periods']):
        t_start = w * cfg['period_ms']
        t_end = (w + 1) * cfg['period_ms']
        mask = (result.t_ms >= t_start) & (result.t_ms < t_end)
        if not mask.any():
            continue
        r_window = result.r[mask, :, 0]  # PYR population: (T_w, n_nodes)
        _, amp = population_vector_decode(r_window, angles_rad)  # (T_w,)
        asym_w = asym[mask]
        rows.append({
            'cond_key': cond_key,
            'trial_idx': trial_idx,
            'seed': seed,
            'window_idx': w,
            'window_start_ms': t_start,
            'window_end_ms': t_end,
            'amp_mean': float(amp.mean()),
            'abs_asym_mean': float(np.abs(asym_w).mean()),
        })

    del result
    return rows


# ============================================================================
# BURN-IN STABILITY SUBCOMMAND
# ============================================================================

def cmd_burnin_stability(args: argparse.Namespace) -> None:
    """Assess whether the burn-in period reaches stationarity.

    Runs n_trials independent noisy simulations from zero initial conditions
    for burnin_ms.  Divides each run into windows of period_ms and computes
    per-window mean amplitude and mean |A(t)| (asymmetry relative to a fixed
    reference angle, default 0°).  A Kruskal-Wallis test across windows checks
    whether the network has reached stationarity.

    Outputs:
        burnin_stability_trials.csv   – per-trial, per-window raw metrics
        burnin_stability_summary.csv  – Kruskal-Wallis H and p per condition/metric
        burnin_stability_{cond}.png   – box plots per window (one per condition)
    """
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    if getattr(args, 'sigma_noise', None) is not None:
        base_params = replace(base_params, sigma_noise=args.sigma_noise)
        print(f"Noise override: sigma_noise = {args.sigma_noise}")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )

    burnin_ms: float = args.burnin_ms
    period_ms: float = args.period_ms
    n_periods: int = int(round(burnin_ms / period_ms))
    ref_deg: float = args.ref_deg
    n_trials: int = args.n_trials
    n_workers = _resolve_workers(args)
    record_dt_ms: float = getattr(args, 'record_dt_ms', 1.0)

    condition_keys = args.conditions if args.conditions else ['WT']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'. "
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/burnin_stability", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nBurn-in stability experiment:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Trials: {n_trials},  workers: {n_workers}")
    print(f"  Burn-in: {burnin_ms:.0f} ms  →  {n_periods} windows of {period_ms:.0f} ms")
    print(f"  Asymmetry reference: {ref_deg:.1f}°")

    connectivity = RingConnectivity.from_params(ring_params)

    # --- CSV cache: load existing rows if parameters match ---
    csv_path = os.path.join(out_dir, "burnin_stability_trials.csv")
    all_rows: list[dict] = []
    cached_trial_ids: dict[str, set] = {ck: set() for ck in condition_keys}

    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='') as _f:
                cached = list(csv.DictReader(_f))
            if cached and 'burnin_ms' in cached[0]:
                params_ok = all(
                    abs(float(r.get('burnin_ms', 0)) - burnin_ms) < 1e-6
                    and abs(float(r.get('period_ms', 0)) - period_ms) < 1e-6
                    and abs(float(r.get('ref_deg', 0)) - ref_deg) < 1e-6
                    for r in cached
                )
                if params_ok:
                    for r in cached:
                        ck = r['condition']
                        if ck not in condition_keys:
                            continue
                        all_rows.append({
                            'cond_key': ck,
                            'trial_idx': int(r['trial_idx']),
                            'seed': int(r['seed']),
                            'window_idx': int(r['window_idx']),
                            'window_start_ms': float(r['window_start_ms']),
                            'window_end_ms': float(r['window_end_ms']),
                            'amp_mean': float(r['amp_mean']),
                            'abs_asym_mean': float(r['abs_asym_mean']),
                        })
                        cached_trial_ids[ck].add(int(r['trial_idx']))
                    n_cached = sum(len(v) for v in cached_trial_ids.values())
                    if n_cached > 0:
                        print(f"\nLoaded {n_cached} cached trial(s) from {csv_path}")
                        for ck in condition_keys:
                            print(f"  {ck}: {len(cached_trial_ids[ck])} / {n_trials}")
                else:
                    print("\nCache parameter mismatch — rerunning all trials.")
            else:
                print("\nOld cache format — rerunning all trials.")
        except Exception as _e:
            print(f"\nCache read failed ({_e}) — rerunning all trials.")
            all_rows = []
            cached_trial_ids = {ck: set() for ck in condition_keys}

    # --- Build remaining trial jobs (skip already cached) ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    jobs = [
        (cond_key, trial_idx, seed)
        for cond_key in condition_keys
        for trial_idx, seed in enumerate(trial_seeds)
        if trial_idx not in cached_trial_ids[cond_key]
    ]

    # --- Run new trials (parallel or sequential) ---
    new_rows: list[dict] = []
    if jobs:
        init_args = (
            base_params, ring_params, connectivity,
            burnin_ms, period_ms, n_periods, ref_deg, record_dt_ms,
        )
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(mp_context=_MP_CONTEXT, 
                max_workers=n_workers,
                initializer=_burnin_stability_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {
                    executor.submit(_burnin_stability_run_single, job): job
                    for job in jobs
                }
                with tqdm(total=len(jobs), desc="Simulations", unit="sim", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        new_rows.extend(future.result())
                        pbar.update()
        else:
            _burnin_stability_init_worker(*init_args)
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                new_rows.extend(_burnin_stability_run_single(job))
        all_rows.extend(new_rows)
    else:
        print("\nAll trials already cached — skipping simulations.")

    # --- Save / update CSV ---
    if new_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition', 'trial_idx', 'seed',
                'window_idx', 'window_start_ms', 'window_end_ms',
                'amp_mean', 'abs_asym_mean',
                'burnin_ms', 'period_ms', 'ref_deg',
            ])
            writer.writeheader()
            for r in sorted(all_rows, key=lambda r: (r['cond_key'], r['trial_idx'], r['window_idx'])):
                writer.writerow({
                    'condition': r['cond_key'],
                    'trial_idx': r['trial_idx'],
                    'seed': r['seed'],
                    'window_idx': r['window_idx'],
                    'window_start_ms': r['window_start_ms'],
                    'window_end_ms': r['window_end_ms'],
                    'amp_mean': r['amp_mean'],
                    'abs_asym_mean': r['abs_asym_mean'],
                    'burnin_ms': burnin_ms,
                    'period_ms': period_ms,
                    'ref_deg': ref_deg,
                })
        print(f"\nTrial data → {csv_path}")

    # --- Statistical tests: Kruskal-Wallis + pairwise Mann-Whitney U ---
    from scipy.stats import kruskal as _kruskal, mannwhitneyu as _mwu

    def _sig_label(p: float) -> str:
        if np.isnan(p): return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    summary_rows: list[dict] = []

    for cond_key in condition_keys:
        cond_rows = [r for r in all_rows if r['cond_key'] == cond_key]

        # Build (n_trials × n_periods) arrays
        amp_matrix = np.full((n_trials, n_periods), np.nan)
        asym_matrix = np.full((n_trials, n_periods), np.nan)
        for r in cond_rows:
            ti, wi = r['trial_idx'], r['window_idx']
            if ti < n_trials and wi < n_periods:
                amp_matrix[ti, wi] = r['amp_mean']
                asym_matrix[ti, wi] = r['abs_asym_mean']

        # Kruskal-Wallis: each group = one window across all trials
        amp_groups = [amp_matrix[:, w][~np.isnan(amp_matrix[:, w])] for w in range(n_periods)]
        asym_groups = [asym_matrix[:, w][~np.isnan(asym_matrix[:, w])] for w in range(n_periods)]

        valid_amp = [g for g in amp_groups if len(g) > 0]
        valid_asym = [g for g in asym_groups if len(g) > 0]

        if len(valid_amp) >= 2:
            h_amp, p_amp = _kruskal(*valid_amp)
        else:
            h_amp, p_amp = np.nan, np.nan

        if len(valid_asym) >= 2:
            h_asym, p_asym = _kruskal(*valid_asym)
        else:
            h_asym, p_asym = np.nan, np.nan

        print(f"\nKruskal-Wallis across windows — {cond_key}:")
        print(f"  Amplitude:   H={h_amp:.3f},  p={p_amp:.4f} {_sig_label(p_amp)}")
        print(f"  |Asymmetry|: H={h_asym:.3f},  p={p_asym:.4f} {_sig_label(p_asym)}")
        print("  (n.s. = windows are statistically indistinguishable → stationarity reached)")

        # Pairwise Mann-Whitney U: adjacent windows only
        print(f"\n  Pairwise Mann-Whitney U (adjacent windows) — {cond_key}:")
        print(f"  {'Window A':>12}  {'Window B':>12}  {'U':>8}  {'p':>8}  sig")
        pairwise_mwu: list[dict] = []
        for w in range(n_periods - 1):
            ga = amp_groups[w]
            gb = amp_groups[w + 1]
            ga_asym = asym_groups[w]
            gb_asym = asym_groups[w + 1]
            if len(ga) >= 2 and len(gb) >= 2:
                u_amp, p_mwu_amp = _mwu(ga, gb, alternative='two-sided')
            else:
                u_amp, p_mwu_amp = np.nan, np.nan
            if len(ga_asym) >= 2 and len(gb_asym) >= 2:
                u_asym, p_mwu_asym = _mwu(ga_asym, gb_asym, alternative='two-sided')
            else:
                u_asym, p_mwu_asym = np.nan, np.nan
            w_start_a = int(w * period_ms)
            w_start_b = int((w + 1) * period_ms)
            print(f"  {w_start_a:>5}–{int((w+1)*period_ms):>5} ms  "
                  f"{w_start_b:>5}–{int((w+2)*period_ms):>5} ms  "
                  f"amp: U={u_amp:.0f} p={p_mwu_amp:.4f} {_sig_label(p_mwu_amp)}  "
                  f"|asym|: U={u_asym:.0f} p={p_mwu_asym:.4f} {_sig_label(p_mwu_asym)}")
            pairwise_mwu.append({
                'window_a': w, 'window_b': w + 1,
                'p_amp': float(p_mwu_amp),
                'p_asym': float(p_mwu_asym),
            })

        summary_rows.append({'condition': cond_key, 'metric': 'amplitude',
                              'H': h_amp, 'p': p_amp})
        summary_rows.append({'condition': cond_key, 'metric': 'abs_asymmetry',
                              'H': h_asym, 'p': p_asym})

        # --- Plot per condition ---
        from .plotting import plot_burnin_stability
        plot_path = os.path.join(out_dir, f"burnin_stability_{cond_key}.png")
        fig = plot_burnin_stability(
            amp_matrix=amp_matrix,
            asym_matrix=asym_matrix,
            period_ms=period_ms,
            cond_key=cond_key,
            p_amp=p_amp,
            p_asym=p_asym,
            pairwise_mwu=pairwise_mwu,
        )
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot → {plot_path}")

    # --- Save summary CSV ---
    summary_path = os.path.join(out_dir, "burnin_stability_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'metric', 'H', 'p'])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"\nSummary → {summary_path}")
    print(f"\nAll outputs saved to {out_dir}/")

    if not args.no_show:
        plt.show()


# ============================================================================
# ASYMMETRY × AMPLITUDE SWEEP: PARALLEL WORKER
# ============================================================================

#: Short secondary burn-in run from the shared state, giving per-trial
#: pre-cue variation without repeating the expensive long burn-in.
ASYM_AMP_SWEEP_SECONDARY_BURNIN_MS: float = 1000.0

_asym_amp_sweep_sim_args: Optional[dict] = None


def _asym_amp_sweep_init_worker(
    base_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    delay_ms: float,
    record_dt_ms: float,
    balance_cue: bool,
    correct_asymmetry: bool,
    shared_r0: dict,
    shared_Ia: dict,
) -> None:
    """Initialise worker process for asymmetry–amplitude-sweep trials."""
    global _asym_amp_sweep_sim_args
    _asym_amp_sweep_sim_args = {
        'base_params':    base_params,
        'ring_params':    ring_params,
        'connectivity':   connectivity,
        'delay_ms':       delay_ms,
        'record_dt_ms':   record_dt_ms,
        'balance_cue':    balance_cue,
        'correct_asymmetry': correct_asymmetry,
        'shared_r0':      shared_r0,
        'shared_Ia':      shared_Ia,
    }


def _asym_amp_sweep_run_single(job: tuple) -> dict:
    """Run one amplitude-sweep trial: secondary burn-in → cue → delay.

    The secondary burn-in starts from the shared condition state
    (pre-computed outside the pool), giving each trial a distinct
    but cheap pre-cue state without re-running the full long burn-in.

    job = (cond_key, trial_idx, seed, amplitude)
    """
    global _asym_amp_sweep_sim_args
    from .analysis import (
        compute_bump_asymmetry,
        decode_bump_center,
        compute_asymmetry_temporal_metrics,
    )

    cfg = _asym_amp_sweep_sim_args
    cond_key, trial_idx, seed, amplitude = job

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(cfg['base_params'], condition)

    # ── Secondary burn-in from shared state ──────────────────────────────────
    sec_result = simulate_ring(
        local_params, cfg['ring_params'],
        T_ms=ASYM_AMP_SWEEP_SECONDARY_BURNIN_MS,
        stimuli=None,
        r0=cfg['shared_r0'][cond_key],
        I_adapt0=cfg['shared_Ia'][cond_key],
        seed=seed,
        connectivity=cfg['connectivity'],
        record_dt_ms=ASYM_AMP_SWEEP_SECONDARY_BURNIN_MS,
    )
    r0_trial = sec_result.r[-1].copy()
    Ia_trial  = sec_result.I_adapt_final.copy()
    del sec_result

    # ── Cue + delay ──────────────────────────────────────────────────────────
    stim_onset  = 0.0
    stim_offset = STIM_DURATION_MS
    T_ms        = stim_offset + cfg['delay_ms']
    actual_current = amplitude * cfg['base_params'].I_ext_pyr()

    if cfg['balance_cue']:
        center_deg = _balance_cue_location(STIM_CENTER_DEG, cfg['ring_params'])
    else:
        center_deg = STIM_CENTER_DEG

    stimuli = [RingStimulus(
        center_deg=center_deg, amplitude=actual_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=stim_onset, duration_ms=STIM_DURATION_MS,
    )]

    # Derive a distinct seed for the cue-delay noise so secondary burn-in and
    # stimulus noise are independent random streams.
    cue_seed = int(seed) ^ 0xC0FFEE42

    result = simulate_ring(
        local_params, cfg['ring_params'], T_ms=T_ms,
        stimuli=stimuli, r0=r0_trial, I_adapt0=Ia_trial,
        seed=cue_seed,
        connectivity=cfg['connectivity'],
        record_dt_ms=cfg['record_dt_ms'],
    )

    asym = compute_bump_asymmetry(result)
    _, bump_amplitude = decode_bump_center(result, population=0)

    def _window_metric(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        asym_w = asym[mask]
        if not cfg['correct_asymmetry']:
            return float(asym_w.mean())
        amp_w = bump_amplitude[mask]
        denom = float(amp_w.sum())
        if denom <= 1e-10:
            return 0.0
        return float((asym_w * amp_w).sum() / denom)

    # Pre-cue window: last ASYM_PRE_CUE_WINDOW_MS of secondary burn-in
    # (recorded time runs from 0 to T_ms with stim onset at 0)
    # Since we start from the secondary state (no burn-in recorded), there is
    # no pre-cue window to show — report NaN for compatibility.
    pre_cue_asym      = float('nan')
    last_pre_cue_asym = float('nan')

    # Delay: after stim offset + transient skip
    delay_start = stim_offset + TRANSIENT_SKIP_TIME_MS
    delay_mask  = (result.t_ms >= delay_start) & (result.t_ms <= T_ms)
    delay_asym  = _window_metric(delay_mask)

    temporal = compute_asymmetry_temporal_metrics(asym[delay_mask], result.t_ms[delay_mask])

    del result

    return {
        'cond_key':             cond_key,
        'trial_idx':            trial_idx,
        'seed':                 seed,
        'amplitude':            amplitude,
        'cue_deg':              center_deg,
        'pre_cue_asym':         pre_cue_asym,
        'last_pre_cue_asym':    last_pre_cue_asym,
        'delay_asym':           delay_asym,
        'correct_asymmetry':    bool(cfg['correct_asymmetry']),
        'mean_abs_asym':        temporal['mean_abs_asym'],
        'asym_std':             temporal['asym_std'],
        'mean_abs_asym_precue': float('nan'),
        'asym_std_precue':      float('nan'),
    }


# ============================================================================
# OSC-PHASE-DISTRACTOR: PARALLEL WORKERS
# ============================================================================

_osc_phase_dist_sim_args: Optional[dict] = None


def _osc_phase_dist_init_worker(
    args_dict: dict,
    base_params,
    per_cond_rp: dict,
    per_cond_conn: dict,
    pre_dist_states: dict,
) -> None:
    """Initialize worker for the phase-timing distractor experiment.

    pre_dist_states : {cond_key: {amplitude: {phase_pi: (r0, I_adapt0)}}}
    """
    global _osc_phase_dist_sim_args
    _osc_phase_dist_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'per_cond_conn': per_cond_conn,
        'pre_dist_states': pre_dist_states,
    }


def _osc_phase_dist_run_single(job: tuple) -> dict:
    """Run one distractor trial starting from the pre-computed pre-distractor state.

    job = (cond_key, amplitude, distractor_factor, offset_deg, phase_pi, trial_idx, seed)
    """
    global _osc_phase_dist_sim_args
    cfg = _osc_phase_dist_sim_args
    cond_key, amplitude, distractor_factor, offset_deg, phase_pi, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    ring_params = cfg['per_cond_rp'][cond_key]
    connectivity = cfg['per_cond_conn'][cond_key]

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    # Retrieve the pre-distractor state for this (cond, amp, phase_pi).
    # The state is the network snapshot at distractor onset (after burn-in + cue + delay1).
    r0, I_adapt0 = cfg['pre_dist_states'][cond_key][amplitude][phase_pi]

    dist_duration_ms = float(args_d['distractor_duration_ms'])
    delay2_ms = float(args_d['delay2_ms'])
    T_ms = dist_duration_ms + delay2_ms   # t = 0 corresponds to distractor onset

    cue_current = amplitude * base_params.I_ext_pyr()

    stimuli = []
    if offset_deg is not None:
        dist_center_deg = (STIM_CENTER_DEG + float(offset_deg)) % 360.0
        dist_current = distractor_factor * cue_current
        stimuli.append(RingStimulus(
            center_deg=dist_center_deg,
            amplitude=dist_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=0.0,
            duration_ms=dist_duration_ms,
        ))

    result = simulate_ring(
        local_params, ring_params,
        T_ms=T_ms,
        stimuli=stimuli if stimuli else None,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=seed,
        connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    # t_s: time in seconds relative to distractor onset (starts at 0)
    t_s = result.t_ms / 1000.0

    # Node indices
    angles_deg = np.rad2deg(ring_params.node_angles_rad)
    cue_idx = int(np.argmin(np.abs(angles_deg - STIM_CENTER_DEG)))
    if offset_deg is not None:
        dist_center_deg = (STIM_CENTER_DEG + float(offset_deg)) % 360.0
        ang_diff = np.abs(angles_deg - dist_center_deg)
        ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
        dist_idx = int(np.argmin(ang_diff))
    else:
        dist_idx = (cue_idx + len(angles_deg) // 2) % len(angles_deg)

    cue_rate = result.r[:, cue_idx, 0]
    dist_rate = result.r[:, dist_idx, 0]
    dist_offset_rel_s = dist_duration_ms / 1000.0

    min_freq = args_d.get('min_freq_hz', 2.0)
    max_freq = args_d.get('max_freq_hz', 12.0)
    win_s = args_d.get('tf_window_s', 1.0)
    overlap = args_d.get('tf_overlap', 0.8)

    _empty_osc = {
        'freqs_hz': np.array([], dtype=float),
        'times_s': np.array([], dtype=float),
        'power': np.zeros((0, 0), dtype=float),
        'dominant_freq_hz': np.array([], dtype=float),
        'dominant_power': np.array([], dtype=float),
    }

    try:
        osc_cue = compute_oscillation_band_timecourse(
            cue_rate, t_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except ValueError:
        osc_cue = _empty_osc.copy()

    try:
        osc_dist = compute_oscillation_band_timecourse(
            dist_rate, t_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except ValueError:
        osc_dist = _empty_osc.copy()

    try:
        plv_result = compute_plv_timecourse(
            cue_rate, dist_rate, t_s,
            min_freq_hz=min_freq, max_freq_hz=max_freq,
            window_s=win_s, overlap_frac=overlap,
        )
    except Exception:
        plv_result = {'times_s': np.array([], dtype=float), 'plv': np.array([], dtype=float)}

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'distractor_factor': distractor_factor,
        'offset_deg': offset_deg,
        'phase_pi': phase_pi,
        'trial_idx': trial_idx,
        'seed': seed,
        # Cue node STFT
        'cue_times_s': osc_cue['times_s'],
        'cue_freqs_hz': osc_cue['freqs_hz'],
        'cue_power': osc_cue['power'],
        'cue_dominant_freq_hz': osc_cue['dominant_freq_hz'],
        'cue_dominant_power': osc_cue['dominant_power'],
        # Distractor node STFT
        'dist_times_s': osc_dist['times_s'],
        'dist_freqs_hz': osc_dist['freqs_hz'],
        'dist_power': osc_dist['power'],
        'dist_dominant_freq_hz': osc_dist['dominant_freq_hz'],
        'dist_dominant_power': osc_dist['dominant_power'],
        # PLV
        'plv_times_s': plv_result['times_s'],
        'plv': plv_result['plv'],
        # Timeline references (relative to distractor onset = t 0)
        'dist_offset_rel_s': dist_offset_rel_s,
    }


def _osc_phase_dist_cache_key(
    args,
    base_params,
    ring_params,
    condition_keys: list,
    amplitudes: list,
    all_phase_pis: list,
) -> str:
    import hashlib, json
    data = {
        'experiment': 'osc_phase_distractor_v1',
        'n_nodes': ring_params.n_nodes,
        'w_pyr_pyr_inter': ring_params.w_pyr_pyr_inter,
        'sigma_pyr_deg': ring_params.sigma_pyr_deg,
        'w_pv_global': ring_params.w_pv_global,
        'conditions': sorted(condition_keys),
        'amplitudes': sorted(amplitudes),
        'distractor_factors': sorted(args.distractor_factors),
        'offsets_deg': sorted(args.offsets_deg),
        'delay1_base_ms': args.delay1_base_ms,
        'phase_pis': [round(p, 6) for p in sorted(all_phase_pis)],
        'distractor_duration_ms': args.distractor_duration_ms,
        'delay2_ms': args.delay2_ms,
        'n_trials': args.n_trials,
        'seed': args.seed,
        'min_freq_hz': args.min_freq_hz,
        'max_freq_hz': args.max_freq_hz,
        'tf_window_s': args.tf_window_s,
        'tf_overlap': args.tf_overlap,
    }
    s = json.dumps(data, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def cmd_osc_distractor_phase_study(args) -> None:
    """Phase-dependent distractor study.

    Keeps the burn-in and cue trajectory IDENTICAL across all phase values
    (same seed, same starting state).  For each phase_pi value, the distractor
    is applied at a different point in the ongoing oscillation cycle and we
    measure how PLV and oscillatory power over the post-distractor delay depend
    on that phase.

    Outputs per (condition, amplitude, distractor_factor, offset_deg):
        - 2×2 grid of PLV timecourses for 4 representative phases
        - 2×2 grid of cue/distractor power timecourses for 4 representative phases
        - 3-row sweep: PLV / cue power / dist power vs. continuous phase
        - Polar version of the sweep
        - Phase × time heatmaps for each metric
    """
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pickle as _pickle

    base_params, load_msg = _load_base_params_for_ring(args.params_json, args)
    print(load_msg)

    condition_keys = args.conditions if args.conditions else ['WT']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'.")
            sys.exit(1)

    cond_excit = _resolve_per_cond_param(args.w_pyr_pyr_inter, condition_keys, 'w_pyr_pyr_inter')
    base_rp = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter[0],
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )
    per_cond_rp = {ck: replace(_base_rp_for_cond(ck, base_rp), w_pyr_pyr_inter=cond_excit[ck]) for ck in condition_keys}
    per_cond_conn = {ck: RingConnectivity.from_params(per_cond_rp[ck]) for ck in condition_keys}
    ring_params = base_rp  # alias for config display

    amplitudes = list(args.amplitudes) if args.amplitudes else [args.amplitude[0]]
    distractor_factors = list(args.distractor_factors)
    offsets_deg = list(args.offsets_deg)
    n_trials = int(args.n_trials)
    n_workers = _resolve_workers(args)
    delay1_base_ms = float(args.delay1_base_ms)
    n_phase = int(args.n_phase_sweep)
    osc_freq_fallback = float(args.osc_freq_hz)

    conn_label = _calibration_network_label(base_rp)
    conn_lbl = _weights_label(base_rp)
    out_root = os.path.join(
        _output_dir("figs/ring/osc_phase_distractor", args.params_json),
        conn_label,
    )
    os.makedirs(out_root, exist_ok=True)

    cond_labels = _build_cond_labels(condition_keys, cond_excit)

    # ------------------------------------------------------------------
    # Step 1 — Estimate oscillation frequency from a reference simulation
    # ------------------------------------------------------------------
    print("\nEstimating oscillation frequency from reference simulation...")
    ref_cond_key = condition_keys[0]
    ref_local_params = apply_condition(base_params, STUDY_CONDITIONS[ref_cond_key])
    r0_bi_ref, Ia_bi_ref = _compute_burnin_state(
        ref_local_params, per_cond_rp[ref_cond_key], per_cond_conn[ref_cond_key], seed=args.seed,
    )

    pre_cue_ms = STIM_ONSET_MS - BURN_IN_MS        # 500 ms
    cue_current_ref = amplitudes[0] * base_params.I_ext_pyr()
    ref_delay_ms = max(3000.0, delay1_base_ms + 4.0 * (1000.0 / osc_freq_fallback))
    T_ref_ms = pre_cue_ms + STIM_DURATION_MS + ref_delay_ms

    ref_result = simulate_ring(
        ref_local_params, per_cond_rp[ref_cond_key],
        T_ms=T_ref_ms,
        stimuli=[RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=cue_current_ref,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=pre_cue_ms,
            duration_ms=STIM_DURATION_MS,
        )],
        r0=r0_bi_ref,
        I_adapt0=Ia_bi_ref,
        seed=args.seed,
        connectivity=per_cond_conn[ref_cond_key],
        record_dt_ms=5.0,
    )

    angles_deg = np.rad2deg(per_cond_rp[ref_cond_key].node_angles_rad)
    cue_idx_ref = int(np.argmin(np.abs(angles_deg - STIM_CENTER_DEG)))
    cue_offset_abs = pre_cue_ms + STIM_DURATION_MS
    mask_post = ref_result.t_ms >= cue_offset_abs
    t_post_s = (ref_result.t_ms[mask_post] - cue_offset_abs) / 1000.0
    cue_rate_ref = ref_result.r[mask_post, cue_idx_ref, 0]
    del ref_result

    f_osc = osc_freq_fallback
    try:
        osc_ref = compute_oscillation_band_timecourse(
            cue_rate_ref, t_post_s,
            min_freq_hz=args.min_freq_hz,
            max_freq_hz=args.max_freq_hz,
            window_s=args.tf_window_s,
            overlap_frac=args.tf_overlap,
        )
        estimated = float(np.nanmedian(osc_ref['dominant_freq_hz']))
        if np.isfinite(estimated) and estimated > 0:
            f_osc = estimated
    except Exception:
        pass

    T_osc_ms = 1000.0 / f_osc
    print(f"  Oscillation frequency: {f_osc:.2f} Hz  (period {T_osc_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # Step 2 — Build phase_pi grid
    # ------------------------------------------------------------------
    # Evenly-spaced sweep values [0, 2) and fixed discrete values for 4-panel figure
    sweep_pis = np.linspace(0.0, 2.0, n_phase, endpoint=False).tolist()
    discrete_pis = [0.0, 0.5, 1.0, 1.5]
    all_phase_pis = sorted(set([round(p, 8) for p in sweep_pis + discrete_pis]))

    def phase_pi_to_delay1(phi):
        return delay1_base_ms + phi * T_osc_ms / 2.0

    print(f"  Phase sweep: {n_phase} steps over [0, 2π)")
    print(f"  delay1 range: [{phase_pi_to_delay1(0):.0f}, "
          f"{phase_pi_to_delay1(max(all_phase_pis)):.0f}] ms")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    use_cache = not getattr(args, 'no_cache', False)
    cache_key = _osc_phase_dist_cache_key(
        args, base_params, base_rp, condition_keys, amplitudes, all_phase_pis,
    )
    cache_file = os.path.join(out_root, f'.osc_phase_cache_{cache_key}.pkl')

    all_results: list[dict] = []
    if use_cache and os.path.exists(cache_file):
        print(f"\nLoading cached results (key={cache_key})...")
        with open(cache_file, 'rb') as _cf:
            all_results = _pickle.load(_cf)
        print(f"  Loaded {len(all_results)} trials from cache.")
    else:
        # ------------------------------------------------------------------
        # Step 3 — Burn-in for all conditions
        # ------------------------------------------------------------------
        print("\nComputing burn-in states...")
        burnin_states: dict = {}
        for ck in tqdm(condition_keys, desc="Burn-in", unit="cond"):
            lp = apply_condition(base_params, STUDY_CONDITIONS[ck])
            burnin_states[ck] = _compute_burnin_state(
                lp, per_cond_rp[ck], per_cond_conn[ck], seed=args.seed,
            )

        # ------------------------------------------------------------------
        # Step 4 — Pre-distractor simulations
        # Each (cond, amplitude, phase_pi) → deterministic state at distractor onset.
        # All simulations use args.seed so the cue trajectory is IDENTICAL.
        # ------------------------------------------------------------------
        print("\nComputing pre-distractor states...")
        pre_dist_states: dict = {}
        total_pre = len(condition_keys) * len(amplitudes) * len(all_phase_pis)
        with tqdm(total=total_pre, desc="Pre-distractor sims", unit="sim") as pbar:
            for ck in condition_keys:
                pre_dist_states[ck] = {}
                lp = apply_condition(base_params, STUDY_CONDITIONS[ck])
                r0_bi, Ia_bi = burnin_states[ck]
                for amp in amplitudes:
                    pre_dist_states[ck][amp] = {}
                    cue_current_pre = amp * base_params.I_ext_pyr()
                    for phi_pi in all_phase_pis:
                        d1 = phase_pi_to_delay1(phi_pi)
                        T_pre = pre_cue_ms + STIM_DURATION_MS + d1
                        pre_res = simulate_ring(
                            lp, per_cond_rp[ck],
                            T_ms=T_pre,
                            stimuli=[RingStimulus(
                                center_deg=STIM_CENTER_DEG,
                                amplitude=cue_current_pre,
                                sigma_deg=STIM_SIGMA_DEG,
                                onset_ms=pre_cue_ms,
                                duration_ms=STIM_DURATION_MS,
                            )],
                            r0=r0_bi,
                            I_adapt0=Ia_bi,
                            seed=args.seed,   # FIXED seed → deterministic trajectory
                            connectivity=per_cond_conn[ck],
                            record_dt_ms=5.0,
                        )
                        pre_dist_states[ck][amp][phi_pi] = (
                            pre_res.r[-1].copy(),
                            pre_res.I_adapt_final.copy(),
                        )
                        del pre_res
                        pbar.update()

        # ------------------------------------------------------------------
        # Step 5 — Distractor trials
        # ------------------------------------------------------------------
        trial_seeds = _generate_trial_seeds(args.seed, n_trials)

        jobs = []
        for ck in condition_keys:
            for amp in amplitudes:
                for factor in distractor_factors:
                    for off in offsets_deg:
                        for phi_pi in all_phase_pis:
                            for ti in range(n_trials):
                                jobs.append((ck, amp, factor, off,
                                             phi_pi, ti, trial_seeds[ti]))
                    # No-distractor control at base phase (phase_pi=0)
                    ctrl_phi = all_phase_pis[0]
                    for ti in range(n_trials):
                        jobs.append((ck, amp, factor, None,
                                     ctrl_phi, ti, trial_seeds[ti]))

        args_dict = {
            'distractor_duration_ms': args.distractor_duration_ms,
            'delay2_ms': args.delay2_ms,
            'min_freq_hz': args.min_freq_hz,
            'max_freq_hz': args.max_freq_hz,
            'tf_window_s': args.tf_window_s,
            'tf_overlap': args.tf_overlap,
            'record_dt_ms': getattr(args, 'record_dt_ms', 5.0),
        }

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                mp_context=_MP_CONTEXT,
                max_workers=n_workers,
                initializer=_osc_phase_dist_init_worker,
                initargs=(args_dict, base_params, per_cond_rp,
                          per_cond_conn, pre_dist_states),
            ) as executor:
                futures = {
                    executor.submit(_osc_phase_dist_run_single, job): job
                    for job in jobs
                }
                with tqdm(total=len(jobs), desc="Simulations", unit="sim",
                          smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _osc_phase_dist_init_worker(
                args_dict, base_params, per_cond_rp, per_cond_conn, pre_dist_states,
            )
            for job in tqdm(jobs, desc="Simulations", unit="sim"):
                all_results.append(_osc_phase_dist_run_single(job))

        with open(cache_file, 'wb') as _cf:
            _pickle.dump(all_results, _cf, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"\nResults cached → {cache_file}")

    # ------------------------------------------------------------------
    # Helper: stack timecourses from a list of result dicts
    # ------------------------------------------------------------------
    def _stack_tc(rows, val_key, time_key):
        valid = [r for r in rows if len(r.get(val_key, [])) > 0]
        if not valid:
            return np.array([]), np.array([]), np.array([])
        t_rows = [np.asarray(r.get(time_key, []), dtype=float) for r in valid]
        t_ref = max(t_rows, key=len)
        n = len(t_ref)
        stack = np.full((len(valid), n), np.nan)
        for j, r in enumerate(valid):
            v = np.asarray(r[val_key], dtype=float)
            stack[j, :len(v)] = v
        with np.errstate(all='ignore'):
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=0)
        return t_ref, mean, sd

    def _stack_tc_full(rows, val_key, time_key):
        """Stack timecourse from result dicts → (t_ref, ndarray[n_trials, n_times])."""
        valid = [r for r in rows if len(r.get(val_key, [])) > 0]
        if not valid:
            return np.array([]), np.zeros((0, 0))
        t_rows = [np.asarray(r.get(time_key, []), dtype=float) for r in valid]
        t_ref = max(t_rows, key=len)
        n = len(t_ref)
        stack = np.full((len(valid), n), np.nan)
        for j, r in enumerate(valid):
            v = np.asarray(r[val_key], dtype=float)
            stack[j, :len(v)] = v
        return t_ref, stack

    # ------------------------------------------------------------------
    # Aggregate and plot
    # ------------------------------------------------------------------
    dist_offset_s = args.distractor_duration_ms / 1000.0

    # CSV
    csv_path = os.path.join(out_root, "osc_phase_trials.csv")
    with open(csv_path, 'w', newline='') as _csvf:
        writer = csv.DictWriter(_csvf, fieldnames=[
            'condition', 'amplitude', 'distractor_factor', 'offset_deg',
            'phase_pi', 'trial_idx', 'seed',
            'plv_mean_delay2', 'cue_power_mean_delay2', 'dist_power_mean_delay2',
        ])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (
            x['cond_key'], x['amplitude'], x['distractor_factor'],
            str(x['offset_deg']), x['phase_pi'], x['trial_idx'],
        )):
            plv_t = np.asarray(r['plv_times_s'], dtype=float)
            plv_v = np.asarray(r['plv'], dtype=float)
            cue_t = np.asarray(r['cue_times_s'], dtype=float)
            cue_v = np.asarray(r['cue_dominant_power'], dtype=float)
            dist_t = np.asarray(r['dist_times_s'], dtype=float)
            dist_v = np.asarray(r['dist_dominant_power'], dtype=float)
            post_mask_plv = plv_t > r['dist_offset_rel_s']
            post_mask_cue = cue_t > r['dist_offset_rel_s']
            post_mask_dist = dist_t > r['dist_offset_rel_s']

            def _safe_mean(v, m):
                return float(np.nanmean(v[m])) if np.any(m) else np.nan

            writer.writerow({
                'condition': r['cond_key'],
                'amplitude': r['amplitude'],
                'distractor_factor': r['distractor_factor'],
                'offset_deg': '' if r['offset_deg'] is None else r['offset_deg'],
                'phase_pi': r['phase_pi'],
                'trial_idx': r['trial_idx'],
                'seed': r['seed'],
                'plv_mean_delay2': _safe_mean(plv_v, post_mask_plv),
                'cue_power_mean_delay2': _safe_mean(cue_v, post_mask_cue),
                'dist_power_mean_delay2': _safe_mean(dist_v, post_mask_dist),
            })

    # --plot_conditions restricts the cross-condition comparison plots only;
    # per-condition figures are always generated for all simulated conditions.
    requested_plot = getattr(args, 'plot_conditions', None)
    if requested_plot:
        plot_cks = [ck for ck in requested_plot if ck in condition_keys]
        unknown = [ck for ck in requested_plot if ck not in condition_keys]
        if unknown:
            print(f"Warning: --plot_conditions ignored unknown keys (not in simulated set): {unknown}")
        if not plot_cks:
            print("Warning: --plot_conditions produced an empty set; falling back to all conditions.")
            plot_cks = condition_keys
    else:
        plot_cks = condition_keys

    for ck in condition_keys:
        cond_out = os.path.join(out_root, ck)
        os.makedirs(cond_out, exist_ok=True)

        for factor in distractor_factors:
            factor_label = f"factor{_fmt(factor)}"
            factor_out = os.path.join(cond_out, factor_label)
            os.makedirs(factor_out, exist_ok=True)

            for amp in amplitudes:
                amp_label = f"amp{_fmt(amp)}"
                amp_out = os.path.join(factor_out, amp_label)
                os.makedirs(amp_out, exist_ok=True)

                # Accumulated across offsets for the summary figure
                sweep_data_by_offset: dict = {}
                ctrl_values_by_offset: dict = {}

                for off in offsets_deg:
                    off_label = f"offset{int(off)}"
                    off_out = os.path.join(amp_out, off_label)
                    os.makedirs(off_out, exist_ok=True)

                    # --------------------------------------------------
                    # Collect timecourse data per phase_pi
                    # --------------------------------------------------
                    data_by_phase: dict = {}
                    t_rel_axis = np.array([])
                    sweep_means: dict = {m: [] for m in ('plv', 'cue_power', 'dist_power')}
                    sweep_sds: dict = {m: [] for m in ('plv', 'cue_power', 'dist_power')}

                    for phi_pi in all_phase_pis:
                        rows = [
                            r for r in all_results
                            if r['cond_key'] == ck
                            and abs(r['amplitude'] - amp) < 1e-9
                            and abs(r['distractor_factor'] - factor) < 1e-9
                            and r['offset_deg'] == off
                            and abs(r['phase_pi'] - phi_pi) < 1e-9
                        ]
                        if not rows:
                            for m in sweep_means:
                                sweep_means[m].append(np.nan)
                                sweep_sds[m].append(np.nan)
                            data_by_phase[phi_pi] = {}
                            continue

                        t_plv, plv_m, plv_s = _stack_tc(rows, 'plv', 'plv_times_s')
                        t_cue, cue_m, cue_s = _stack_tc(rows, 'cue_dominant_power', 'cue_times_s')
                        t_dst, dst_m, dst_s = _stack_tc(rows, 'dist_dominant_power', 'dist_times_s')

                        # Reference time axis (shared STFT grid)
                        t_ref = t_cue if len(t_cue) >= len(t_plv) else t_plv

                        if len(t_ref) > len(t_rel_axis):
                            t_rel_axis = t_ref

                        data_by_phase[phi_pi] = {
                            'plv_mean': plv_m, 'plv_sd': plv_s,
                            'cue_mean': cue_m, 'cue_sd': cue_s,
                            'dist_mean': dst_m, 'dist_sd': dst_s,
                        }

                        # Summary: mean over delay2 per trial
                        for metric_key, val_key, time_key in [
                            ('plv', 'plv', 'plv_times_s'),
                            ('cue_power', 'cue_dominant_power', 'cue_times_s'),
                            ('dist_power', 'dist_dominant_power', 'dist_times_s'),
                        ]:
                            vals_per_trial = []
                            for r in rows:
                                tv = np.asarray(r[time_key], dtype=float)
                                vv = np.asarray(r[val_key], dtype=float)
                                m = tv > r['dist_offset_rel_s']
                                if np.any(m):
                                    vals_per_trial.append(float(np.nanmean(vv[m])))
                            if vals_per_trial:
                                arr = np.array(vals_per_trial)
                                sweep_means[metric_key].append(float(np.nanmean(arr)))
                                sweep_sds[metric_key].append(
                                    float(np.nanstd(arr, ddof=min(1, len(arr) - 1)))
                                )
                            else:
                                sweep_means[metric_key].append(np.nan)
                                sweep_sds[metric_key].append(np.nan)

                    # No-distractor control
                    ctrl_phi = all_phase_pis[0]
                    ctrl_rows = [
                        r for r in all_results
                        if r['cond_key'] == ck
                        and abs(r['amplitude'] - amp) < 1e-9
                        and abs(r['distractor_factor'] - factor) < 1e-9
                        and r['offset_deg'] is None
                        and abs(r['phase_pi'] - ctrl_phi) < 1e-9
                    ]
                    ctrl_data: dict = {}
                    ctrl_values: dict = {}
                    if ctrl_rows:
                        _, ctrl_plv_m, ctrl_plv_s = _stack_tc(ctrl_rows, 'plv', 'plv_times_s')
                        _, ctrl_cue_m, ctrl_cue_s = _stack_tc(ctrl_rows, 'cue_dominant_power', 'cue_times_s')
                        _, ctrl_dst_m, ctrl_dst_s = _stack_tc(ctrl_rows, 'dist_dominant_power', 'dist_times_s')
                        ctrl_data = {
                            'plv_mean': ctrl_plv_m, 'plv_sd': ctrl_plv_s,
                            'cue_mean': ctrl_cue_m, 'cue_sd': ctrl_cue_s,
                            'dist_mean': ctrl_dst_m, 'dist_sd': ctrl_dst_s,
                        }
                        for mkey, val_key, time_key in [
                            ('plv', 'plv', 'plv_times_s'),
                            ('cue_power', 'cue_dominant_power', 'cue_times_s'),
                            ('dist_power', 'dist_dominant_power', 'dist_times_s'),
                        ]:
                            vals = []
                            for r in ctrl_rows:
                                tv = np.asarray(r[time_key], dtype=float)
                                vv = np.asarray(r[val_key], dtype=float)
                                m = tv > r['dist_offset_rel_s']
                                if np.any(m):
                                    vals.append(float(np.nanmean(vv[m])))
                            ctrl_values[mkey] = float(np.nanmean(vals)) if vals else np.nan

                    sweep_data_fig = {
                        m: {
                            'mean': np.array(sweep_means[m]),
                            'sd': np.array(sweep_sds[m]),
                        }
                        for m in sweep_means
                    }
                    # Accumulate for cross-offset summary
                    sweep_data_by_offset[off] = sweep_data_fig
                    ctrl_values_by_offset[off] = ctrl_values

                    base_title = (
                        f"{ck} | {amp_label}× | {factor_label} | "
                        f"offset={int(off)}° | {conn_lbl}"
                    )

                    # 1. PLV timecourse overlay (all 4 phases on one plot)
                    fig1 = plot_osc_phase_timecourses_grid(
                        t_rel_axis=t_rel_axis,
                        data_by_phase=data_by_phase,
                        dist_offset_s=dist_offset_s,
                        metric='plv',
                        discrete_phases=discrete_pis,
                        ctrl_data=ctrl_data if ctrl_data else None,
                        suptitle=f"PLV timecourses by phase | {base_title}",
                        save_path=os.path.join(off_out, "phase_plv_overlay.png"),
                    )
                    plt.close(fig1)

                    # 2. Cue power timecourse overlay
                    fig2 = plot_osc_phase_timecourses_grid(
                        t_rel_axis=t_rel_axis,
                        data_by_phase=data_by_phase,
                        dist_offset_s=dist_offset_s,
                        metric='cue_power',
                        discrete_phases=discrete_pis,
                        ctrl_data=ctrl_data if ctrl_data else None,
                        suptitle=f"Cue node power by phase | {base_title}",
                        save_path=os.path.join(off_out, "phase_cue_power_overlay.png"),
                    )
                    plt.close(fig2)

                    # 3. Distractor power timecourse overlay
                    fig3 = plot_osc_phase_timecourses_grid(
                        t_rel_axis=t_rel_axis,
                        data_by_phase=data_by_phase,
                        dist_offset_s=dist_offset_s,
                        metric='dist_power',
                        discrete_phases=discrete_pis,
                        ctrl_data=ctrl_data if ctrl_data else None,
                        suptitle=f"Distractor node power by phase | {base_title}",
                        save_path=os.path.join(off_out, "phase_dist_power_overlay.png"),
                    )
                    plt.close(fig3)

                    # 4. Continuous sweep (PLV + cue power + dist power)
                    fig4 = plot_osc_phase_sweep(
                        phase_pis=all_phase_pis,
                        sweep_data=sweep_data_fig,
                        ctrl_values=ctrl_values if ctrl_values else None,
                        suptitle=f"Delay₂ metric vs. distractor phase | {base_title}",
                        save_path=os.path.join(off_out, "phase_sweep.png"),
                    )
                    plt.close(fig4)

                    # 5. Polar sweep
                    fig5 = plot_osc_phase_polar(
                        phase_pis=all_phase_pis,
                        sweep_data=sweep_data_fig,
                        ctrl_values=ctrl_values if ctrl_values else None,
                        suptitle=f"Polar: delay₂ metric vs. phase | {base_title}",
                        save_path=os.path.join(off_out, "phase_polar.png"),
                    )
                    plt.close(fig5)

                    # 6–8. Phase × time heatmaps
                    n_phases = len(all_phase_pis)
                    n_times = len(t_rel_axis)

                    for metric_label, mean_key in [
                        ('plv', 'plv_mean'),
                        ('cue_power', 'cue_mean'),
                        ('dist_power', 'dist_mean'),
                    ]:
                        heat = np.full((n_phases, max(n_times, 1)), np.nan)
                        for pi, phi_pi in enumerate(all_phase_pis):
                            d = data_by_phase.get(phi_pi, {})
                            v = np.asarray(d.get(mean_key, []), dtype=float)
                            heat[pi, :len(v)] = v
                        fig6 = plot_osc_phase_heatmap(
                            phase_pis=all_phase_pis,
                            t_rel_axis=t_rel_axis,
                            heatmap_data=heat,
                            dist_offset_s=dist_offset_s,
                            metric=metric_label,
                            suptitle=f"Phase × time heatmap ({metric_label}) | {base_title}",
                            save_path=os.path.join(off_out, f"phase_heatmap_{metric_label}.png"),
                        )
                        plt.close(fig6)

                    # 9. Phases-as-conditions box plot (discrete phases compared at each time bin)
                    import colorsys as _colorsys
                    _phase_colors_hex = {
                        phi: '#{:02x}{:02x}{:02x}'.format(
                            *[int(x * 255) for x in
                              _colorsys.hsv_to_rgb(phi / 2.0, 0.80, 0.82)]
                        )
                        for phi in discrete_pis
                    }

                    def _phi_lbl(phi):
                        if phi == 0.0:   return '0'
                        if phi == 0.5:   return 'π/2'
                        if phi == 1.0:   return 'π'
                        if phi == 1.5:   return '3π/2'
                        return f'{phi}π'

                    data_by_phase_cond: dict = {}
                    t_ref_phase_bp = np.array([])
                    for phi_pi in discrete_pis:
                        rows_phi = [
                            r for r in all_results
                            if r['cond_key'] == ck
                            and abs(r['amplitude'] - amp) < 1e-9
                            and abs(r['distractor_factor'] - factor) < 1e-9
                            and r['offset_deg'] == off
                            and abs(r['phase_pi'] - phi_pi) < 1e-9
                        ]
                        if not rows_phi:
                            continue
                        t_cue, stack_cue = _stack_tc_full(
                            rows_phi, 'cue_dominant_power', 'cue_times_s')
                        _, stack_dst = _stack_tc_full(
                            rows_phi, 'dist_dominant_power', 'dist_times_s')
                        _, stack_plv = _stack_tc_full(
                            rows_phi, 'plv', 'plv_times_s')
                        dist_onset_s_ref = rows_phi[0].get('dist_onset_rel_s', 0.0)
                        t_cue_rel = t_cue - dist_onset_s_ref
                        if len(t_cue_rel) >= len(t_ref_phase_bp):
                            t_ref_phase_bp = t_cue_rel
                        lbl = _phi_lbl(phi_pi)
                        data_by_phase_cond[lbl] = {
                            'plv':        stack_plv,
                            'cue_power':  stack_cue,
                            'dist_power': stack_dst,
                        }
                    if len(data_by_phase_cond) >= 2:
                        phase_colors_by_lbl = {
                            _phi_lbl(phi): _phase_colors_hex[phi]
                            for phi in discrete_pis if _phi_lbl(phi) in data_by_phase_cond
                        }
                        fig_pbp = plot_osc_conditions_boxplot(
                            t_axis=t_ref_phase_bp,
                            data_by_condition=data_by_phase_cond,
                            dist_offset_s=dist_offset_s,
                            condition_colors=phase_colors_by_lbl,
                            suptitle=(
                                f"Phase comparison | {ck} | {amp_label}× | {factor_label} | "
                                f"offset={int(off)}° | {conn_lbl}"
                            ),
                            save_path=os.path.join(off_out, "phases_boxplot.png"),
                        )
                        plt.close(fig_pbp)

                # 10. Summary: all offsets on the same phase-sweep plot
                if len(offsets_deg) > 0 and sweep_data_by_offset:
                    summary_title = (
                        f"{ck} | {amp_label}× | {factor_label} | all offsets | {conn_lbl}"
                    )
                    fig_sum = plot_osc_phase_sweep_offsets(
                        phase_pis=all_phase_pis,
                        sweep_data_by_offset=sweep_data_by_offset,
                        ctrl_values_by_offset=ctrl_values_by_offset if ctrl_values_by_offset else None,
                        suptitle=f"Phase sweep — all offsets | {summary_title}",
                        save_path=os.path.join(amp_out, "phase_sweep_all_offsets.png"),
                    )
                    plt.close(fig_sum)

    # ------------------------------------------------------------------
    # Cross-condition box plots (only when multiple conditions are compared)
    # ------------------------------------------------------------------
    if len(plot_cks) > 1:
        def _phi_fname(phi_pi):
            if phi_pi == 0.0:   return '0'
            if phi_pi == 0.5:   return 'pi_over_2'
            if phi_pi == 1.0:   return 'pi'
            if phi_pi == 1.5:   return '3pi_over_2'
            return str(phi_pi).replace('.', '_') + 'pi'

        def _phi_title(phi_pi):
            if phi_pi == 0.0:   return '0'
            if phi_pi == 0.5:   return 'π/2'
            if phi_pi == 1.0:   return 'π'
            if phi_pi == 1.5:   return '3π/2'
            return f'{phi_pi}π'

        for factor in distractor_factors:
            factor_label = f"factor{_fmt(factor)}"
            for amp in amplitudes:
                amp_label = f"amp{_fmt(amp)}"
                for off in offsets_deg:
                    off_label = f"offset{int(off)}"
                    for phi_pi in discrete_pis:
                        data_by_cond: dict = {}
                        t_ref_bp = np.array([])
                        for ck in plot_cks:
                            rows_bp = [
                                r for r in all_results
                                if r['cond_key'] == ck
                                and abs(r['amplitude'] - amp) < 1e-9
                                and abs(r['distractor_factor'] - factor) < 1e-9
                                and r['offset_deg'] == off
                                and abs(r['phase_pi'] - phi_pi) < 1e-9
                            ]
                            if not rows_bp:
                                continue
                            t_cue, stack_cue = _stack_tc_full(
                                rows_bp, 'cue_dominant_power', 'cue_times_s')
                            _, stack_dst = _stack_tc_full(
                                rows_bp, 'dist_dominant_power', 'dist_times_s')
                            _, stack_plv = _stack_tc_full(
                                rows_bp, 'plv', 'plv_times_s')
                            dist_onset_s = rows_bp[0].get('dist_onset_rel_s', 0.0)
                            t_cue_rel = t_cue - dist_onset_s
                            if len(t_cue_rel) >= len(t_ref_bp):
                                t_ref_bp = t_cue_rel
                            data_by_cond[ck] = {
                                'plv':        stack_plv,
                                'cue_power':  stack_cue,
                                'dist_power': stack_dst,
                            }
                        if len(data_by_cond) < 2:
                            continue
                        cmp_dir = os.path.join(
                            out_root, "comparison", factor_label,
                            amp_label, off_label,
                        )
                        os.makedirs(cmp_dir, exist_ok=True)
                        fig_bp = plot_osc_conditions_boxplot(
                            t_axis=t_ref_bp,
                            data_by_condition=data_by_cond,
                            dist_offset_s=dist_offset_s,
                            suptitle=(
                                f"Condition comparison | {amp_label}× | {factor_label} | "
                                f"offset={int(off)}° | phase={_phi_title(phi_pi)} | {conn_lbl}"
                            ),
                            save_path=os.path.join(
                                cmp_dir,
                                f"conditions_boxplot_phase{_phi_fname(phi_pi)}.png",
                            ),
                        )
                        plt.close(fig_bp)

    print("\nPhase-distractor study complete.")
    print(f"  CSV:        {csv_path}")
    print(f"  Figures:    {out_root}")
    print(f"  Cache:      {cache_file}  (key={cache_key})")
    print(f"  Osc. freq:  {f_osc:.2f} Hz  (period {T_osc_ms:.1f} ms)")


# ============================================================================
# ASYMMETRY × AMPLITUDE SWEEP SUBCOMMAND
# ============================================================================


# ============================================================================
# RING-OPTIMIZE: JOINT CIRCUIT + RING PARAMETER OPTIMIZATION
# ============================================================================

def add_ring_optimize_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ring-optimize subcommand."""
    # --- Target firing rates (required) ---
    parser.add_argument("--target_pyr", type=float, required=True,
                        help="Target mean PYR firing rate (Hz)")
    parser.add_argument("--target_som", type=float, required=True,
                        help="Target mean SOM firing rate (Hz)")
    parser.add_argument("--target_pv", type=float, required=True,
                        help="Target mean PV firing rate (Hz)")
    parser.add_argument("--target_vip", type=float, required=True,
                        help="Target mean VIP firing rate (Hz)")

    # --- Optional knockout targets ---
    parser.add_argument("--target_alpha7_ko_pyr", type=float, default=None,
                        help="Target PYR rate under alpha7 knockout (Hz)")
    parser.add_argument("--target_alpha5_ko_pyr", type=float, default=None,
                        help="Target PYR rate under alpha5 knockout (Hz)")
    parser.add_argument("--target_beta2_ko_pyr", type=float, default=None,
                        help="Target PYR rate under beta2 knockout (Hz)")

    # --- Starting point for circuit params ---
    parser.add_argument("--params_json", type=str, default="",
                        help="Load initial CircuitParams from JSON file "
                             "(default: project WT default if available)")

    # --- Ring network settings (fixed during optimization) ---
    parser.add_argument("--n_nodes", type=int, default=64,
                        help="Number of ring nodes (fixed during optimization, default: 64)")

    # --- Starting point for ring params ---
    parser.add_argument("--ring_params_json", type=str, default="",
                        help="Load initial RingParams from JSON file (same format as --save_best_ring_json output)")

    # --- Ring parameter search bounds (optional overrides) ---
    parser.add_argument("--w_pyr_pyr_inter_lo", type=float, default=0.0005,
                        help="Lower bound for w_pyr_pyr_inter (default: 0.0005)")
    parser.add_argument("--w_pyr_pyr_inter_hi", type=float, default=0.05,
                        help="Upper bound for w_pyr_pyr_inter (default: 0.05)")
    parser.add_argument("--w_pv_global_lo", type=float, default=0.0005,
                        help="Lower bound for w_pv_global (default: 0.0005)")
    parser.add_argument("--w_pv_global_hi", type=float, default=0.1,
                        help="Upper bound for w_pv_global (default: 0.1)")
    parser.add_argument("--sigma_pyr_deg_lo", type=float, default=5.0,
                        help="Lower bound for sigma_pyr_deg (default: 5.0)")
    parser.add_argument("--sigma_pyr_deg_hi", type=float, default=60.0,
                        help="Upper bound for sigma_pyr_deg (default: 60.0)")

    # --- Optimization settings ---
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of optimization steps (default: 5000)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Keep top K candidates (default: 10)")
    parser.add_argument("--optimizer", type=str, default="de",
                        choices=["de", "cma", "chaining", "auto"],
                        help="Optimizer: de=TwoPointsDE, cma=CMA-ES, "
                             "chaining=DE->Nelder-Mead, auto=NGOpt (default: de)")
    parser.add_argument("--early_stop_loss", type=float, default=1e-4,
                        help="Stop early if loss falls below this value (default: 1e-4)")
    parser.add_argument("--plateau_patience", type=int, default=500,
                        help="Stop if no improvement for this many steps (0=disable, default: 500)")
    parser.add_argument("--de_fraction", type=float, default=0.25,
                        help="Fraction of budget for DE phase in chaining mode (default: 0.25). "
                             "Computed as de_steps = max(500, min(int(budget * de_fraction), 10000))")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--freeze", type=str, default="",
                        help="Comma-separated CircuitParams field names to freeze during optimization")
    parser.add_argument("--set", dest="set_params", type=str, default="",
                        help="Override CircuitParams values before optimizing: 'name=val,name=val' "
                             "(e.g. --set tau_s=20,g=1). Useful combined with --freeze.")

    # --- Ring fit config ---
    parser.add_argument("--n_trials_ring", type=int, default=5,
                        help="Ring simulations per candidate evaluation (default: 5 for noise averaging)")

    # --- Simulation time settings ---
    parser.add_argument("--T_ms", type=float, default=2500.0,
                        help="Ring simulation duration (ms, default: 2500)")
    parser.add_argument("--dt_ms", type=float, default=0.1,
                        help="Integration time step (ms, default: 0.1)")
    parser.add_argument("--burn_in_ms", type=float, default=1200.0,
                        help="Burn-in period to skip transients (ms, default: 1200 — reduced for noise)")
    parser.add_argument("--window_ms", type=float, default=500.0,
                        help="Rate averaging window (ms, default: 500)")
    parser.add_argument("--record_dt_ms", type=float, default=2.0,
                        help="Recording interval (ms, default: 2.0 — matches FitConfig)")
    parser.add_argument("--noise_type", choices=["none", "white", "ou"], default="white",
                        help="Noise type during optimization (default: white, matching ring-run)")
    parser.add_argument("--tau_noise_ms", type=float, default=5.0,
                        help="OU noise time constant (ms, default: 5.0)")
    parser.add_argument("--max_rate", type=float, default=200.0,
                        help="Maximum allowed firing rate for stability check (default: 200)")

    # --- KO penalty settings ---
    parser.add_argument("--ko_min_effect_penalty", type=float, default=5.0,
                        help="Penalty weight for weak KO effect (default: 5.0)")
    parser.add_argument("--ko_wrong_direction_penalty", type=float, default=10.0,
                        help="Penalty weight for wrong-direction KO effect (default: 10.0)")

    # --- Deprecated Mode 2: bump quality constraint ---
    parser.add_argument("--bump_mode", action="store_true",
                        help="Deprecated. Bump constraints are now integrated in the "
                            "trace-based Turing loss. Flag kept for backward compatibility.")
    parser.add_argument("--min_bump_amplitude", type=float, default=0.3,
                        help="Minimum acceptable bump amplitude [0,1] for Mode 2 (default: 0.3)")
    parser.add_argument("--bump_loss_weight", type=float, default=2.0,
                        help="Weight of bump quality loss relative to rate loss (default: 2.0)")
    parser.add_argument("--bump_stim_amplitude", type=float, default=5.0,
                        help="Peak current of test stimulus for bump evaluation (default: 5.0)")
    parser.add_argument("--bump_stim_sigma_deg", type=float, default=20.0,
                        help="Gaussian width of test stimulus in degrees (default: 20.0)")
    parser.add_argument("--bump_stim_duration_ms", type=float, default=250.0,
                        help="Test stimulus duration in ms (default: 250.0)")
    parser.add_argument("--bump_eval_window_ms", type=float, default=500.0,
                        help="Post-stimulus window to evaluate bump amplitude (ms, default: 500.0)")

    # --- Adaptation ---
    parser.add_argument("--no_adapt", action="store_true",
                        help="Disable spike-frequency adaptation: set J_adapt_pyr=0 and J_adapt_som=0 "
                             "and freeze them. Equivalent to --set J_adapt_pyr=0,J_adapt_som=0 "
                             "--freeze J_adapt_pyr,J_adapt_som.")

    # --- Turing instability penalty ---
    parser.add_argument("--turing_weight", type=float, default=2.0,
                        help="Weight of simulation-trace Turing bistability loss (default: 2.0). "
                            "Enforces rest stability + bump sustain around 40 Hz + anti-runaway.")
    parser.add_argument("--turing_margin", type=float, default=0.05,
                        help="Safety margin around the Turing threshold (default: 0.05)")
    parser.add_argument("--turing_cue_amplitude", type=float, default=0.4,
                        help="Cue amplitude as factor of I0_pyr for the deterministic Turing pass "
                            "(default: 0.4, additive PYR-only cue).")
    parser.add_argument("--turing_cue_duration_ms", type=float, default=250.0,
                        help="Cue duration (ms) used by the deterministic Turing pass (default: 250)")
    parser.add_argument("--turing_cue_sigma_deg", type=float, default=20.0,
                        help="Cue spatial width (deg) used by the deterministic Turing pass (default: 20)")
    parser.add_argument("--turing_late_delay_ms", type=float, default=500.0,
                        help="Late-delay window length (ms) used for bump/sustain checks (default: 500)")
    parser.add_argument("--turing_bump_min_hz", type=float, default=35.0,
                        help="Minimum late-delay bump-node PYR rate (Hz, default: 35)")
    parser.add_argument("--turing_bump_max_hz", type=float, default=45.0,
                        help="Maximum late-delay bump-node PYR rate (Hz, default: 45)")
    parser.add_argument("--turing_topk_nodes", type=int, default=5,
                        help="Number of top PYR nodes used as bump support set (default: 5)")
    parser.add_argument("--turing_activate_below_ring_rate_loss", type=float, default=1.0,
                        help="Activate Turing loss only when ring firing-rate loss is <= this threshold "
                            "(default: 1.0)")
    parser.add_argument("--spatial_uniformity_weight", type=float, default=0.0,
                        help="Weight of spatial uniformity penalty (default: 0 = disabled). "
                             "Penalises std(r_pyr_nodes)/mean(r_pyr_nodes) at rest to prevent "
                             "spontaneous bump formation in the resting state.")
    parser.add_argument("--skip-jacobian", action="store_true",
                        help="Skip the Jacobian connectivity penalty during optimization.")
    parser.add_argument("--jacobian_weight", type=float, default=1.0,
                        help="Weight of the Jacobian connectivity penalty (default: 1.0, 0 = disabled). "
                             "Controls the strength of connectivity constraints during optimization.")
    parser.add_argument("--ach_ratio_weight", type=float, default=2.0,
                        help="Weight of β2/α7 ACh current ratio penalty (default: 2.0, 0 = disabled). "
                             "Penalises solutions where I_beta2_som / I_alpha7_som deviates from 35 "
                             "(Koukouli et al. 2025: β2-type currents ~35× stronger than α7 at 1.77 μM ACh).")

    # --- I/O settings ---
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save best circuit + ring params as best_circuit_params.json / "
                             "best_ring_params.json. Ignored when --save_best_circuit_json / "
                             "--save_best_ring_json are provided.")
    parser.add_argument("--save_best_circuit_json", type=str, default="",
                        help="Explicit path for best CircuitParams JSON (overrides output_dir for circuit file)")
    parser.add_argument("--save_best_ring_json", type=str, default="",
                        help="Explicit path for best RingParams JSON (overrides output_dir for ring file)")
    parser.add_argument("--log_file", type=str, default="ring_optim_log.jsonl",
                        help="JSONL log file path (default: ring_optim_log.jsonl)")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N steps (default: 50)")


def cmd_ring_optimize(args: argparse.Namespace) -> None:
    """Run joint ring + circuit parameter optimization."""
    from dataclasses import fields, replace as _replace
    from ..params import default_bounds, ParamBound
    from ..loss import TargetRates, FitConfig
    from ..io import load_params_json, load_ring_params_json, save_params_json
    from .params import RingParams, default_ring_bounds
    from .optimization import BumpTarget, nevergrad_optimize_ring, _save_ring_candidate

    # --- Load base circuit parameters ---
    if args.params_json:
        base_circuit = load_params_json(args.params_json)
        print(f"Loaded CircuitParams from: {args.params_json}")
    elif DEFAULT_WT_PARAMS_PATH.exists():
        base_circuit = load_params_json(str(DEFAULT_WT_PARAMS_PATH))
        print(f"Loaded default CircuitParams from: {DEFAULT_WT_PARAMS_PATH}")
    else:
        base_circuit = _default_fit_init_params()
        print("Using hardcoded fit-init CircuitParams")

    # --- Apply --set overrides to base circuit params ---
    set_overrides: dict[str, float] = {}
    if args.set_params:
        from ..params import CircuitParams as _CircuitParams
        def _parse_set(s):
            out = {}
            for pair in s.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    out[k.strip()] = float(v.strip())
            return out
        set_overrides = _parse_set(args.set_params)
        allowed = {f.name for f in fields(_CircuitParams)}
        clean = {k: v for k, v in set_overrides.items() if k in allowed}
        if clean:
            base_circuit = _replace(base_circuit, **clean)
            print(f"Overrides applied: {', '.join(f'{k}={v}' for k, v in clean.items())}")

    # trans_factor is not part of the ring resting-rate objective; fix it before optimization.
    if "trans_factor" in set_overrides:
        fixed_trans_factor = float(set_overrides["trans_factor"])
        print(f"trans_factor fixed from --set: {fixed_trans_factor}")
    else:
        fixed_trans_factor = 0.2
        print(f"trans_factor fixed for ring-optimize: {fixed_trans_factor} (default)")
    base_circuit = _replace(base_circuit, trans_factor=fixed_trans_factor)

    # --- --no_adapt: zero and freeze adaptation strengths ---
    if args.no_adapt:
        from dataclasses import replace as _replace
        base_circuit = _replace(base_circuit, J_adapt_pyr=0.0, J_adapt_som=0.0)
        print("--no_adapt: J_adapt_pyr=0, J_adapt_som=0 (frozen)")

    # --- Build target rates ---
    target = TargetRates(
        mean_r_pyr=args.target_pyr,
        mean_r_som=args.target_som,
        mean_r_pv=args.target_pv,
        mean_r_vip=args.target_vip,
        alpha7_ko_pyr=args.target_alpha7_ko_pyr,
        alpha5_ko_pyr=args.target_alpha5_ko_pyr,
        beta2_ko_pyr=args.target_beta2_ko_pyr,
    )

    # --- Build initial ring parameters ---
    if args.ring_params_json:
        base_ring = load_ring_params_json(args.ring_params_json)
        print(f"Loaded RingParams from: {args.ring_params_json}")
    else:
        raise ValueError("--ring_params_json is required")

    # --- Build bounds ---
    circuit_bounds = default_bounds(base_circuit)
    circuit_bounds.pop("trans_factor", None)
    ring_bounds = {
        "w_pyr_pyr_inter": ParamBound(args.w_pyr_pyr_inter_lo, args.w_pyr_pyr_inter_hi, mode="lin"),
        "w_pv_global":     ParamBound(args.w_pv_global_lo,     args.w_pv_global_hi,     mode="lin"),
        "sigma_pyr_deg":   ParamBound(args.sigma_pyr_deg_lo,   args.sigma_pyr_deg_hi,   mode="lin"),
    }

    # --- Parse frozen parameters ---
    freeze: set[str] = {s.strip() for s in args.freeze.split(",") if s.strip()} if args.freeze else set()
    freeze.add("trans_factor")
    freeze.add("tau_adapt_pyr")  # Freeze at biological value (600 ms, Storm 1989)
    if args.no_adapt:
        freeze |= {"J_adapt_pyr", "J_adapt_som"}

    # --- Build fit configs ---
    fit_cfg = FitConfig(
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        burn_in_ms=args.burn_in_ms,
        window_ms=args.window_ms,
        record_dt_ms=args.record_dt_ms,
        n_trials=args.n_trials_ring,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        max_rate=args.max_rate,
        ko_min_effect_penalty=args.ko_min_effect_penalty,
        ko_wrong_direction_penalty=args.ko_wrong_direction_penalty,
    )
    ring_cfg = RingFitConfig(
        fit_cfg=fit_cfg,
        n_trials_ring=args.n_trials_ring,
    )

    # --- Deprecated Mode 2: bump quality constraint ---
    bump_target = None
    if args.bump_mode:
        print("[DEPRECATED] --bump_mode is ignored. Trace-based Turing loss now includes bump constraints.")

    # --- Print summary ---
    print("\nOptimization targets:")
    print(f"  PYR={target.mean_r_pyr} Hz  SOM={target.mean_r_som} Hz  "
          f"PV={target.mean_r_pv} Hz  VIP={target.mean_r_vip} Hz")
    if target.alpha7_ko_pyr is not None:
        print(f"  alpha7 KO PYR: {target.alpha7_ko_pyr} Hz")
    if target.alpha5_ko_pyr is not None:
        print(f"  alpha5 KO PYR: {target.alpha5_ko_pyr} Hz")
    if target.beta2_ko_pyr is not None:
        print(f"  beta2 KO PYR: {target.beta2_ko_pyr} Hz")
    print(f"\nRing: n_nodes={base_ring.n_nodes}, "
          f"w_pyr_pyr_inter={base_ring.w_pyr_pyr_inter}, "
          f"w_pv_global={base_ring.w_pv_global}, "
          f"sigma_pyr_deg={base_ring.sigma_pyr_deg}")
    print("Mode: rates + trace-based Turing bistability")
    print("KO conditions on: ring")
    print(f"Output dir: {args.output_dir}")
    print()

    init_rng = np.random.default_rng(args.seed if args.seed is not None else 0)
    jacobian_weight = 0.0 if args.skip_jacobian else args.jacobian_weight
    init_loss, init_ring_means, _, _ = evaluate_ring_params(
        base_circuit,
        base_ring,
        target,
        ring_cfg,
        bump_target,
        init_rng,
        jacobian_weight=jacobian_weight,
        turing_weight=args.turing_weight,
        turing_margin=args.turing_margin,
        turing_cue_amplitude=args.turing_cue_amplitude,
        turing_cue_duration_ms=args.turing_cue_duration_ms,
        turing_cue_sigma_deg=args.turing_cue_sigma_deg,
        turing_late_delay_ms=args.turing_late_delay_ms,
        turing_bump_min_hz=args.turing_bump_min_hz,
        turing_bump_max_hz=args.turing_bump_max_hz,
        turing_topk_nodes=args.turing_topk_nodes,
        turing_activate_below_ring_rate_loss=args.turing_activate_below_ring_rate_loss,
        spatial_uniformity_weight=args.spatial_uniformity_weight,
        ach_ratio_weight=args.ach_ratio_weight,
    )
    _print_ring_init_summary(base_circuit, base_ring, init_ring_means, init_loss)

    optimizer = args.optimizer
    print()

    # --- Run optimization ---
    best = nevergrad_optimize_ring(
        target,
        base_circuit=base_circuit,
        circuit_bounds=circuit_bounds,
        base_ring=base_ring,
        ring_bounds=ring_bounds,
        ring_cfg=ring_cfg,
        bump_target=bump_target,
        n_samples=args.n_samples,
        top_k=args.top_k,
        seed=args.seed,
        optimizer=optimizer,
        freeze=freeze,
        early_stop_loss=args.early_stop_loss,
        plateau_patience=args.plateau_patience,
        de_fraction=args.de_fraction,
        log_file=args.log_file or None,
        log_interval=args.log_interval,
        save_output_dir=args.output_dir or None,
        jacobian_weight=jacobian_weight,
        turing_weight=args.turing_weight,
        turing_margin=args.turing_margin,
        turing_cue_amplitude=args.turing_cue_amplitude,
        turing_cue_duration_ms=args.turing_cue_duration_ms,
        turing_cue_sigma_deg=args.turing_cue_sigma_deg,
        turing_late_delay_ms=args.turing_late_delay_ms,
        turing_bump_min_hz=args.turing_bump_min_hz,
        turing_bump_max_hz=args.turing_bump_max_hz,
        turing_topk_nodes=args.turing_topk_nodes,
        turing_activate_below_ring_rate_loss=args.turing_activate_below_ring_rate_loss,
        spatial_uniformity_weight=args.spatial_uniformity_weight,
        ach_ratio_weight=args.ach_ratio_weight,
    )

    if not best:
        raise RuntimeError("Optimization returned no candidates.")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    for i, c in enumerate(best, start=1):
        pyr, som, pv, vip = c.ring_means.tolist()
        rp = c.ring_params
        print(
            f"rank {i:02d}: loss={c.loss:.3e} "
            f"means=[pyr={pyr:.4g}, som={som:.4g}, pv={pv:.4g}, vip={vip:.4g}] "
            f"ring=[w_pyr={rp.w_pyr_pyr_inter:.4g}, w_pv={rp.w_pv_global:.4g}, "
            f"sigma={rp.sigma_pyr_deg:.4g}]"
        )

    # --- Jacobian + fit summary ---
    from ..jacobian import compute_jacobian
    r_ss = best[0].ring_means
    J = compute_jacobian(best[0].params, r_ss)
    fit_meta = build_fit_comparison(best[0].ring_means, best[0].ko_means, target, best[0].loss, jacobian=J)
    fit_meta["ring_params"] = {
        "w_pyr_pyr_inter": round(float(best[0].ring_params.w_pyr_pyr_inter), 6),
        "w_pv_global":     round(float(best[0].ring_params.w_pv_global), 6),
        "sigma_pyr_deg":   round(float(best[0].ring_params.sigma_pyr_deg), 6),
        "n_nodes":         int(best[0].ring_params.n_nodes),
    }

    # --- Save final results ---
    import json as _json
    from pathlib import Path as _Path
    from dataclasses import fields as _fields

    _out = args.output_dir or "."
    circuit_path = args.save_best_circuit_json or str(_Path(_out) / "best_circuit_params.json")
    ring_path    = args.save_best_ring_json    or str(_Path(_out) / "best_ring_params.json")

    _Path(circuit_path).parent.mkdir(parents=True, exist_ok=True)
    _Path(ring_path).parent.mkdir(parents=True, exist_ok=True)

    save_params_json(circuit_path, best[0].params, fit_meta=fit_meta)
    save_fit_summary_txt(circuit_path, fit_meta, params=best[0].params)

    ring_dict = {f.name: getattr(best[0].ring_params, f.name) for f in _fields(RingParams) if not f.name.startswith('_')}
    with open(ring_path, "w", encoding="utf-8") as _fh:
        _json.dump(ring_dict, _fh, indent=2)

    print(f"\nBest parameters saved:")
    print(f"  circuit: {circuit_path}")
    print(f"  ring:    {ring_path}")
    print(f"  summary: {str(_Path(circuit_path).with_suffix('.txt'))}")

    # Force clean exit to avoid hanging threads
    import sys as _sys
    _sys.exit(0)

