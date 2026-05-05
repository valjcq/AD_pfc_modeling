"""
Ring attractor CLI logic.

This module contains the ring-specific CLI functions (cmd_run, cmd_study)
and their helpers. These are invoked from circuit_model.cli via the
ring-run and ring-study subcommands.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    plot_amp_sweep_lines,
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
_RING_PARAMS_FALLBACK = {
    "n_nodes":       RingParams.__dataclass_fields__["n_nodes"].default,
    "sigma_pyr_deg": RingParams.__dataclass_fields__["sigma_pyr_deg"].default,
    "sigma_som_deg": RingParams.__dataclass_fields__["sigma_som_deg"].default,
    "som_pattern":   RingParams.__dataclass_fields__["som_pattern"].default,
}


def _print_ring_init_summary(base_circuit: CircuitParams, base_ring: RingParams, ring_means: np.ndarray, init_loss: float) -> None:
    """Print effective ring optimization initialization and its predicted ring rates."""
    print("Initial condition (effective after --set/--no_adapt):")
    print(f"  Circuit I0: pyr={base_circuit.I0_pyr:.6g}, som={base_circuit.I0_som:.6g}, pv={base_circuit.I0_pv:.6g}, vip={base_circuit.I0_vip:.6g}")
    print(f"  Ring: n_nodes={base_ring.n_nodes}, sigma_pyr_deg={base_ring.sigma_pyr_deg:.6g}, sigma_som_deg={base_ring.sigma_som_deg:.6g}")
    print("Initial predicted ring rates (Hz):")
    print(f"  PYR={ring_means[0]:.4f}, SOM={ring_means[1]:.4f}, PV={ring_means[2]:.4f}, VIP={ring_means[3]:.4f}")
    print(f"  Initial loss={init_loss:.6g}")


def _load_ring_params_json(path: str) -> RingParams:
    """Load RingParams from a JSON file."""
    import json
    from dataclasses import fields as _fields

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    allowed = {fld.name for fld in _fields(RingParams) if not fld.name.startswith("_")}
    # Silently discard legacy fields that no longer exist in RingParams.
    _legacy = {"w_pyr_pyr_inter", "w_pv_global"}
    clean = {k: d[k] for k in d if k in allowed and k not in _legacy}
    return RingParams(**clean)


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
            args.sigma_pyr_deg is None
            and args.sigma_som_deg is None
        )
        if args.sigma_pyr_deg is None:
            args.sigma_pyr_deg = _rp.sigma_pyr_deg if _rp else fb["sigma_pyr_deg"]
        if args.sigma_som_deg is None:
            args.sigma_som_deg = _rp.sigma_som_deg if _rp else fb["sigma_som_deg"]
        if args.n_nodes is None:
            args.n_nodes = _rp.n_nodes if _rp else fb["n_nodes"]
        if getattr(args, "som_pattern", None) is None:
            args.som_pattern = _rp.som_pattern if _rp else fb["som_pattern"]

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
    # Save values before _load_base_params_for_ring patches them from defaults,
    # so we can tell whether the user explicitly passed these args.
    _explicit_n_nodes = args.n_nodes
    _explicit_som_pattern = getattr(args, "som_pattern", None)
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
        if _explicit_som_pattern is not None:
            ring_params = replace(ring_params, som_pattern=_explicit_som_pattern)
            load_msg += f"\nOverriding som_pattern from CLI: {_explicit_som_pattern}"
    else:
        ring_params = RingParams(
            n_nodes=args.n_nodes,
            sigma_pyr_deg=args.sigma_pyr_deg,
            sigma_som_deg=args.sigma_som_deg,
            som_pattern=args.som_pattern,
        )

    factor = amp_factor if amp_factor is not None else args.amplitude[0]
    actual_current = factor * base_params.I_ext_pyr()

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = _compute_delay_end_ms(args, stim_offset_ms)

    response_onset_ms = getattr(args, 'response_onset_ms', None)
    response_duration_ms = getattr(args, 'response_duration_ms', 500.0)
    post_response_ms = getattr(args, 'post_response_ms', 3000.0)

    if response_onset_ms is not None and response_onset_ms >= 0:
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
    response_onset_ms = getattr(args, 'response_onset_ms', None)
    # response_onset_ms < 0 or None means disabled; >= 0 means enabled at that offset
    if response_onset_ms is None or response_onset_ms < 0:
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
        emit(f"       n_nodes       = {ring_params.n_nodes}")
        emit(f"       sigma_pyr_deg = {ring_params.sigma_pyr_deg:.4g} deg")
        emit(f"       sigma_som_deg = {ring_params.sigma_som_deg:.4g} deg")

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

    response_onset = getattr(args, 'response_onset_ms', None)
    if response_onset is not None and response_onset >= 0:
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
    """Build a directory-safe label encoding n_nodes and Gaussian sigmas.

    Example: 64_sigma_pyr_15_sigma_som_15
    """
    return (
        f"{rp.n_nodes}_sigma_pyr_{_fmt(rp.sigma_pyr_deg)}"
        f"_sigma_som_{_fmt(rp.sigma_som_deg)}"
    )


def _calibration_network_label(rp: RingParams) -> str:
    """Label for calibration directories: n_nodes + Gaussian sigmas.

    Example: 64_sigma_pyr_15_sigma_som_15
    """
    return f"{rp.n_nodes}_sigma_pyr_{_fmt(rp.sigma_pyr_deg)}_sigma_som_{_fmt(rp.sigma_som_deg)}"


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

    Possible values: cue, cue_distractor
    (response/transient is not encoded in the folder name)
    """
    has_dist = _has_distractor(args)
    return "cue_distractor" if has_dist else "cue"


def _stim_label(amp_factor: float) -> str:
    """Short label for stimulus amplitude factor, used in plot titles."""
    return f"amp={_fmt(amp_factor)}×"


def _weights_label(rp: RingParams) -> str:
    """Short label for PYR and SOM sigmas, used in plot titles."""
    return f"σ_pyr={_fmt(rp.sigma_pyr_deg)}°, σ_som={_fmt(rp.sigma_som_deg)}°"


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
        help="Load ring parameters (sigma_pyr_deg, sigma_som_deg, n_nodes) from JSON file "
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
        help="Number of ring nodes (default: from ring params JSON or 64)",
    )
    parser.add_argument(
        "--sigma_pyr_deg", type=float, default=None,
        help="PYR lateral ring connectivity width in degrees. Default: from ring params JSON or 15.0.",
    )
    parser.add_argument(
        "--sigma_som_deg", type=float, default=None,
        help="SOM lateral ring connectivity width in degrees. Default: from ring params JSON or 15.0.",
    )
    parser.add_argument(
        "--som_pattern", type=str, default=None,
        choices=["gaussian", "uniform", "none"],
        help="SOM→PYR connectivity pattern: 'gaussian' (annular surround, default), 'uniform' (all-to-all, zero diagonal), or 'none' (local only, no inter-node connections).",
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
        "--response_onset_ms", type=float, default=None,
        help="Start a global response transient this many ms after delay end (negative value disables, 0 = immediately)",
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
        "--post_response_ms", type=float, default=3000.0,
        help="Recording duration after response transient ends (default: 3000 ms = 3 seconds)",
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
    response_onset_ms = args_d.get('response_onset_ms', None)
    if response_onset_ms is not None and response_onset_ms >= 0:
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


_bump_decay_sim_args: Optional[dict] = None


def _bump_decay_init_worker(
    base_params: CircuitParams,
    per_cond_rp: dict[str, RingParams],
    connectivity_map: dict[str, RingConnectivity],
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    delay_ms: float,
    ref_offset_ms: float,
    window_ms: float,
    record_dt_ms: float,
    T_ms_full: float,
) -> None:
    """Initialize worker state for bump-decay simulations."""
    global _bump_decay_sim_args
    _bump_decay_sim_args = {
        'base_params': base_params,
        'per_cond_rp': per_cond_rp,
        'connectivity_map': connectivity_map,
        'burnin_states': burnin_states,
        'delay_ms': float(delay_ms),
        'ref_offset_ms': float(ref_offset_ms),
        'window_ms': float(window_ms),
        'record_dt_ms': float(record_dt_ms),
        'T_ms_full': float(T_ms_full),
    }


def _bump_decay_run_single(job: tuple) -> dict:
    """Run one bump-decay trial and return a compact timecourse payload."""
    global _bump_decay_sim_args
    cfg = _bump_decay_sim_args
    if cfg is None:
        raise RuntimeError("_bump_decay_run_single called before worker init")

    cond_key, amplitude, w_inter, trial_idx, seed = job

    base_params: CircuitParams = cfg['base_params']
    per_cond_rp: dict[str, RingParams] = cfg['per_cond_rp']
    connectivity_map: dict[str, RingConnectivity] = cfg['connectivity_map']
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]] = cfg['burnin_states']
    ref_offset_ms: float = cfg['ref_offset_ms']
    record_dt_ms: float = cfg['record_dt_ms']
    T_ms_full: float = cfg['T_ms_full']

    local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
    ring_params = per_cond_rp[cond_key]
    connectivity = connectivity_map[cond_key]

    r0, I_adapt0 = burnin_states[cond_key]

    actual_current = float(amplitude) * float(base_params.I_ext_pyr())
    T_ms_short = T_ms_full - BURN_IN_MS
    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG,
            amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=T_ms_short,
        stimuli=stimuli_short,
        r0=r0,
        I_adapt0=I_adapt0,
        seed=int(seed),
        noise_type='white',
        connectivity=connectivity,
        record_dt_ms=record_dt_ms,
    )

    result.t_ms += BURN_IN_MS

    t_abs_ms  = np.asarray(result.t_ms, dtype=float)
    t_rel_ms  = t_abs_ms - STIM_ONSET_MS
    _, amp_tc = population_vector_decode(
        result.r[:, :, 0],
        result.ring_params.node_angles_rad,
    )
    amp_tc = np.asarray(amp_tc, dtype=float)

    ref_t_rel = STIM_DURATION_MS + ref_offset_ms
    ref_idx = int(np.argmin(np.abs(t_rel_ms - ref_t_rel)))
    ref_amplitude = float(amp_tc[ref_idx]) if amp_tc.size else 0.0

    # ── Per-phase node-saturation diagnostics ────────────────────────────────
    # Thresholds: sat > 120 Hz, bump 30-120 Hz, rest <= 30 Hz
    _SAT_HZ      = 120.0
    _BUMP_HZ     = 30.0
    _WIN_MS      = 500.0   # sliding window size for post-cue analysis
    pyr_act      = result.r[:, :, 0]   # (n_steps, n_nodes)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    pre_mask  = t_abs_ms < STIM_ONSET_MS
    cue_mask  = (t_abs_ms >= STIM_ONSET_MS) & (t_abs_ms < stim_offset_ms)

    def _phase_fracs(mask):
        if not np.any(mask):
            return np.nan, np.nan, np.nan
        mean_node = pyr_act[mask].mean(axis=0)          # (n_nodes,)
        frac_sat  = float(np.mean(mean_node > _SAT_HZ))
        frac_bump = float(np.mean((mean_node > _BUMP_HZ) & (mean_node <= _SAT_HZ)))
        frac_rest = float(np.mean(mean_node <= _BUMP_HZ))
        return frac_sat, frac_bump, frac_rest

    pre_sat, _, _            = _phase_fracs(pre_mask)
    cue_sat, _, _            = _phase_fracs(cue_mask)

    # Windowed post-cue analysis: consecutive 500ms bins from cue offset
    t_post_rel = t_abs_ms - stim_offset_ms          # 0 at cue offset
    delay_total = float(t_post_rel[t_post_rel >= 0].max()) if np.any(t_post_rel >= 0) else 0.0
    win_edges   = np.arange(0.0, delay_total + _WIN_MS, _WIN_MS)
    win_centers = (win_edges[:-1] + win_edges[1:]) / 2.0
    n_wins      = len(win_centers)

    win_frac_sat  = []
    win_frac_bump = []
    win_frac_rest = []
    for i in range(n_wins):
        wmask = (t_post_rel >= win_edges[i]) & (t_post_rel < win_edges[i + 1])
        fs, fb, fr = _phase_fracs(wmask)
        win_frac_sat.append(fs)
        win_frac_bump.append(fb)
        win_frac_rest.append(fr)

    # Also keep scalar summaries: full post, and "late" (skip first window = transient)
    post_mask   = t_post_rel >= 0.0
    post_sat, post_bump, post_rest       = _phase_fracs(post_mask)
    late_mask   = t_post_rel >= _WIN_MS
    late_sat, late_bump, late_rest       = _phase_fracs(late_mask)

    return {
        'cond_key': cond_key,
        'amplitude': float(amplitude),
        'w_inter': float(w_inter),
        'trial_idx': int(trial_idx),
        'seed': int(seed),
        't_ms': t_rel_ms.tolist(),
        'amplitude_timecourse': amp_tc.tolist(),
        'ref_amplitude': ref_amplitude,
        # saturation diagnostics
        'pre_frac_sat':   pre_sat,
        'cue_frac_sat':   cue_sat,
        # full post-cue averages
        'post_frac_sat':  post_sat,
        'post_frac_bump': post_bump,
        'post_frac_rest': post_rest,
        # late post-cue (skip first 500ms transient)
        'late_frac_sat':  late_sat,
        'late_frac_bump': late_bump,
        'late_frac_rest': late_rest,
        # windowed: list of fracs per 500ms bin from cue offset
        'win_centers_ms':  win_centers.tolist(),
        'win_frac_sat':    win_frac_sat,
        'win_frac_bump':   win_frac_bump,
        'win_frac_rest':   win_frac_rest,
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
        # Bump when trial output format changes so old caches are invalidated
        '_trial_fmt_version': 3,  # v3: windowed post-cue saturation fracs
    }
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _extract_scalar(val):
    """Extract scalar from value (handles list or scalar input)."""
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else float("nan")
    return float(val)


def _write_run_metrics(result, args, ring_params, out_dir: str, delay_end_ms: float) -> None:
    """Compute and save run_metrics.json into out_dir.

    Always includes delay-period bump metrics.  When a response transient was
    applied, also includes post-transient state metrics computed over the last
    POST_WINDOW_MS of the simulation:

      post_mean_pyr_hz     – mean PYR rate averaged over all nodes
      post_var_pyr_hz      – variance of per-node mean PYR rates (spatial)
      post_frac_above_30hz – fraction of nodes with mean rate > 30 Hz
      post_frac_above_50hz – fraction of nodes with mean rate > 50 Hz
      post_state           – "SILENT" / "BUMP" / "HIGH"

    State heuristic:
      SILENT  post_frac_above_30hz < 0.15
      HIGH    post_frac_above_30hz > 0.60
      BUMP    otherwise  (localised activity preserved)
    """
    POST_WINDOW_MS = 1000.0  # look at last 1 s of simulation

    from .analysis import compute_bump_metrics

    # ── delay-period metrics ─────────────────────────────────────────────────
    stim_end_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_mask = (result.t_ms >= stim_end_ms + 100.0) & (result.t_ms <= delay_end_ms)
    bump_m = compute_bump_metrics(result)

    if np.any(delay_mask):
        delay_rates = result.r[delay_mask, :, 0]          # (T, n_nodes)
        baseline_pyr = float(result.r[result.t_ms < STIM_ONSET_MS, :, 0].mean())
        peak_pyr     = float(result.r[result.t_ms <= stim_end_ms, :, 0].max())
        delay_center = float(delay_rates.mean(axis=0).max())
        delay_opp    = float(delay_rates.mean(axis=0).min())
    else:
        baseline_pyr = delay_center = delay_opp = peak_pyr = float("nan")

    metrics: dict = {
        "params": {
            "sigma_pyr_deg": ring_params.sigma_pyr_deg,
            "sigma_som_deg": ring_params.sigma_som_deg,
            "n_nodes":       ring_params.n_nodes,
            "amplitude":     _extract_scalar(getattr(args, "amplitude", float("nan"))),
            "condition":     getattr(args, "condition", "WT"),
        },
        "steady_state": {
            "baseline_pyr_hz":      round(baseline_pyr, 3),
            "peak_pyr_cue_hz":      round(min(peak_pyr, 200.0), 3),
            "delay_pyr_center_hz":  round(delay_center, 3),
            "delay_pyr_opposite_hz": round(delay_opp, 3),
        },
        "bump_metrics": {k: round(float(v), 4) for k, v in bump_m.items()},
    }

    # ── post-transient metrics ───────────────────────────────────────────────
    response_onset = getattr(args, "response_onset_ms", None)
    if response_onset is not None and response_onset >= 0:
        trans_end_ms = (delay_end_ms + response_onset
                        + getattr(args, "response_duration_ms", 500.0))
        post_start   = max(result.t_ms[-1] - POST_WINDOW_MS, trans_end_ms + 50.0)
        post_mask    = result.t_ms >= post_start

        if np.any(post_mask):
            post_rates   = result.r[post_mask, :, 0]          # (T, n_nodes)
            node_means   = post_rates.mean(axis=0)             # (n_nodes,)
            mean_hz      = float(node_means.mean())
            var_hz       = float(node_means.var())
            frac_30      = float((node_means > 30.0).mean())
            frac_50      = float((node_means > 50.0).mean())

            if frac_30 < 0.15:
                state = "SILENT"
            elif frac_30 > 0.60:
                state = "HIGH"
            else:
                state = "BUMP"
        else:
            mean_hz = var_hz = frac_30 = frac_50 = float("nan")
            state = "UNKNOWN"

        metrics["post_transient"] = {
            "post_mean_pyr_hz":     round(mean_hz, 3),
            "post_var_pyr_hz":      round(var_hz, 3),
            "post_frac_above_30hz": round(frac_30, 4),
            "post_frac_above_50hz": round(frac_50, 4),
            "post_state":           state,
        }

    metrics_path = os.path.join(out_dir, "run_metrics.json")
    with open(metrics_path, "w") as _f:
        json.dump(metrics, _f, indent=2)


def cmd_run(args: argparse.Namespace) -> None:
    """Run one ring simulation for a single condition and generate figures."""
    _resolve_seed(args)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, ring_params, T_ms, stimuli, amp_factor, load_msg = _build_common(args)

    cond_key = getattr(args, "condition", "WT")
    if cond_key not in STUDY_CONDITIONS:
        print(
            f"Error: unknown condition '{cond_key}'.\n"
            f"Valid: {', '.join(STUDY_CONDITIONS.keys())}"
        )
        sys.exit(1)

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)
    if getattr(args, "no_adapt", False):
        local_params = replace(local_params, J_adapt_pyr=0.0, J_adapt_som=0.0)
    if args.sigma_noise is not None:
        local_params = replace(local_params, sigma_noise=float(args.sigma_noise))

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = _compute_delay_end_ms(args, stim_offset_ms)
    local_params = _apply_response_transient(local_params, args, delay_end_ms)

    connectivity = RingConnectivity.from_params(ring_params, local_params)
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

    sigma_tag = f"sigma{local_params.sigma_noise:.3g}"
    if getattr(args, "output_dir", ""):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            _output_dir("figs/ring/run", args.params_json),
            sigma_tag,
            f"amp_{_fmt(amp_factor)}",
            _run_type_label(args),
            _network_label(ring_params),
            cond_key,
        )
    os.makedirs(out_dir, exist_ok=True)

    # ── Save run_metrics.json ────────────────────────────────────────────────
    _write_run_metrics(result, args, ring_params, out_dir, delay_end_ms)

    suptitle = (
        f"{condition.label} -- {_stim_label(amp_factor)}, {_weights_label(ring_params)}, σ={local_params.sigma_noise:.3g}"
    )
    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, result.t_ms[-1])

    fig_dash = plot_ring_dashboard(
        result,
        save_path=os.path.join(out_dir, "dashboard.png"),
        time_range=time_range,
        t_offset=t_offset,
        suptitle=suptitle,
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
    )
    plt.close(fig_pop)

    ax_conn = plot_ring_connectome(
        ring_params,
        local_params=base_params,
        save_path=os.path.join(out_dir, "connectome.png"),
    )
    plt.close(ax_conn.figure)

    fig_mat = plot_connectivity_matrices(
        ring_params,
        local_params=base_params,
        save_path=os.path.join(out_dir, "connectivity_matrices.png"),
    )
    plt.close(fig_mat)

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
                show_asymmetry=True,
                **anim_quality_kwargs,
            )
            plt.close(fig_anim)
        except Exception as exc:
            print(f"Warning: snapshot animation export failed: {exc}")

    if not args.no_show:
        plt.show()


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

    base_rp = RingParams(
        n_nodes=args.n_nodes,
        sigma_pyr_deg=args.sigma_pyr_deg,
        sigma_som_deg=args.sigma_som_deg,
    )
    per_cond_rp = {ck: _base_rp_for_cond(ck, base_rp) for ck in condition_keys}
    per_cond_conn = {
        ck: RingConnectivity.from_params(
            per_cond_rp[ck],
            apply_condition(base_params, STUDY_CONDITIONS[ck]),
        )
        for ck in condition_keys
    }
    ring_params = base_rp  # alias for config display
    cond_excit = {ck: 0.0 for ck in condition_keys}  # placeholder for label building

    amplitudes     = list(args.amplitudes) if args.amplitudes else [5.0, 10.0, 15.0, 20.0, 25.0]
    w_inter_values = [0.0]  # no longer swept; kept for downstream grouping compatibility
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
    user_out_dir = getattr(args, 'output_dir', None)
    if user_out_dir:
        out_dir = str(user_out_dir)
    else:
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

    # ── Precompute connectivity map {cond_key: RingConnectivity} ────────────
    connectivity_map: dict[str, RingConnectivity] = {
        ck: per_cond_conn[ck] for ck in condition_keys
    }

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
    # Determine window column names from first result that has windowed data
    _sample_wins = next(
        (r['win_centers_ms'] for r in all_results if 'win_centers_ms' in r), []
    )
    win_col_sat  = [f"win{i:02d}_frac_sat"  for i in range(len(_sample_wins))]
    win_col_bump = [f"win{i:02d}_frac_bump" for i in range(len(_sample_wins))]
    win_col_rest = [f"win{i:02d}_frac_rest" for i in range(len(_sample_wins))]

    summary_csv = os.path.join(out_dir, "bump_decay_trials.csv")
    with open(summary_csv, 'w', newline='') as _f:
        fieldnames = [
            'condition', 'amplitude', 'w_inter', 'trial_idx', 'seed',
            'ref_amplitude', 'end_val_normalized',
            'pre_frac_sat', 'cue_frac_sat',
            'post_frac_sat', 'post_frac_bump', 'post_frac_rest',
            'late_frac_sat', 'late_frac_bump', 'late_frac_rest',
        ] + win_col_sat + win_col_bump + win_col_rest
        writer = _csv.DictWriter(_f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results,
                        key=lambda x: (x['cond_key'], x['amplitude'],
                                       x['w_inter'], x['trial_idx'])):
            key = (r['cond_key'], r['amplitude'], r['w_inter'])
            agg = aggregated.get(key, {})
            row = {
                'condition':          r['cond_key'],
                'amplitude':          r['amplitude'],
                'w_inter':            r['w_inter'],
                'trial_idx':          r['trial_idx'],
                'seed':               r['seed'],
                'ref_amplitude':      r['ref_amplitude'],
                'end_val_normalized': agg.get('end_val', float('nan')),
                'pre_frac_sat':       r.get('pre_frac_sat',  float('nan')),
                'cue_frac_sat':       r.get('cue_frac_sat',  float('nan')),
                'post_frac_sat':      r.get('post_frac_sat', float('nan')),
                'post_frac_bump':     r.get('post_frac_bump', float('nan')),
                'post_frac_rest':     r.get('post_frac_rest', float('nan')),
                'late_frac_sat':      r.get('late_frac_sat', float('nan')),
                'late_frac_bump':     r.get('late_frac_bump', float('nan')),
                'late_frac_rest':     r.get('late_frac_rest', float('nan')),
            }
            # Per-window columns
            for i, col in enumerate(win_col_sat):
                row[col] = r['win_frac_sat'][i]  if 'win_frac_sat'  in r and i < len(r['win_frac_sat'])  else float('nan')
            for i, col in enumerate(win_col_bump):
                row[col] = r['win_frac_bump'][i] if 'win_frac_bump' in r and i < len(r['win_frac_bump']) else float('nan')
            for i, col in enumerate(win_col_rest):
                row[col] = r['win_frac_rest'][i] if 'win_frac_rest' in r and i < len(r['win_frac_rest']) else float('nan')
            writer.writerow(row)
    print(f"\nTrial summary → {summary_csv}")

    # ── Plotting ─────────────────────────────────────────────────────────────
    from .plotting import (
        plot_bump_decay_timecourse,
        plot_bump_decay_boxplot,
        plot_bump_decay_heatmap,
        plot_amp_sweep_lines,
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

    fig_sweep = plot_amp_sweep_lines(
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


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Compatibility wrapper for the ring-calibrate command.

    The legacy 3D calibration implementation was removed from this module.
    We keep the public command available by running the bump-decay backend directly.
    """
    if getattr(args, 'conditions', None) is None:
        # Preserve historical ring-calibrate default.
        args.conditions = ['WT']

    base_output_dir = getattr(args, 'output_dir', None)
    print("[ring-calibrate] Compatibility mode: using bump-decay backend.")

    cmd_bump_decay_study(args)

    # Combined heatmap generation (legacy w_pv sweep) removed — no longer applicable.


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
    parser.add_argument("--sigma_pyr_deg_lo", type=float, default=5.0,
                        help="Lower bound for sigma_pyr_deg (default: 5.0)")
    parser.add_argument("--sigma_pyr_deg_hi", type=float, default=40.0,
                        help="Upper bound for sigma_pyr_deg (default: 40.0)")
    parser.add_argument("--sigma_som_deg_lo", type=float, default=5.0,
                        help="Lower bound for sigma_som_deg (default: 5.0)")
    parser.add_argument("--sigma_som_deg_hi", type=float, default=40.0,
                        help="Upper bound for sigma_som_deg (default: 40.0)")

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
        "sigma_pyr_deg": ParamBound(args.sigma_pyr_deg_lo, args.sigma_pyr_deg_hi, mode="lin"),
        "sigma_som_deg": ParamBound(args.sigma_som_deg_lo, args.sigma_som_deg_hi, mode="lin"),
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
          f"sigma_pyr_deg={base_ring.sigma_pyr_deg}, "
          f"sigma_som_deg={base_ring.sigma_som_deg}")
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
            f"ring=[sigma_pyr={rp.sigma_pyr_deg:.4g}, sigma_som={rp.sigma_som_deg:.4g}]"
        )

    # --- Jacobian + fit summary ---
    from ..jacobian import compute_jacobian
    r_ss = best[0].ring_means
    J = compute_jacobian(best[0].params, r_ss)
    fit_meta = build_fit_comparison(best[0].ring_means, best[0].ko_means, target, best[0].loss, jacobian=J)
    fit_meta["ring_params"] = {
        "sigma_pyr_deg": round(float(best[0].ring_params.sigma_pyr_deg), 6),
        "sigma_som_deg": round(float(best[0].ring_params.sigma_som_deg), 6),
        "n_nodes":       int(best[0].ring_params.n_nodes),
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

