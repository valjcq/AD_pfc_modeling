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
from dataclasses import replace
from typing import Optional

from joblib import Parallel, delayed

import numpy as np

from ..params import CircuitParams
from ..io import load_params_json, output_dir as _output_dir
from ..study import STUDY_CONDITIONS, CONDITION_ORDER, apply_condition

from .params import RingParams
from .stimulus import RingStimulus
from .simulation import simulate_ring, simulate_ring_batch
from .connectivity import RingConnectivity
from .constants import TRANSIENT_SKIP_TIME_MS
from .analysis import (
    compute_bump_metrics,  # noqa: F401 (used by ring-run)
    compute_metrics_at_delay_times,
    aggregate_metrics_across_trials,
    aggregate_single_metrics,
    population_vector_decode,

    compute_drift_field,
    compute_noise_floor,
    detect_saturated_w_values,
    SATURATION_A_HAT_THRESHOLD,

    # New: battery-of-experiments functions
    compute_bump_survival_time,
    compute_lesion_metrics,
    extract_tau_sweep_metrics,
    run_phase_plane_sweep,
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
    plot_drift_field,
    plot_noise_floor_histogram,
    plot_calibration_heatmap,
    plot_calibration_timecourses,
    plot_noise_summary,
    plot_distractor_sweep_heatmaps,
    plot_distractor_sweep_timecourses,
    plot_distractor_sweep_activity_grid,
    plot_distractor_sweep_node_differences,
    plot_distractor_sweep_node_timecourses,

    # New: battery-of-experiments plot functions
    plot_lesion_study,
    plot_tau_adapt_sweep,
    plot_phase_plane,
    plot_temporal_dissection,
)


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


def _build_common(args, amp_factor: float | None = None):
    """Build base params, ring params, T_ms, and stimuli from parsed args.

    The *amp_factor* (or ``args.amplitude``) is a **multiplier of
    I_ext_pyr**.  The actual peak current injected into the stimulus is
    ``amp_factor * base_params.I_ext_pyr()``.

    Returns:
        (base_params, ring_params, T_ms, stimuli, amp_factor)
    """
    if args.params_json:
        base_params = load_params_json(args.params_json)
    else:
        base_params = CircuitParams()

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    factor = amp_factor if amp_factor is not None else args.amplitude
    actual_current = factor * base_params.I_ext_pyr()

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms

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

    return base_params, ring_params, T_ms, stimuli, factor


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
                  ring_params: RingParams | None = None):
    """Print configuration summary."""
    I_baseline = base_params.I_ext_pyr()
    actual_current = amp_factor * I_baseline
    print(f"Stimulus: {amp_factor:.1f}× I_ext_pyr  "
          f"(= {actual_current:.2f}, baseline = {I_baseline:.2f})")
    print(f"          Gaussian sigma={STIM_SIGMA_DEG:.0f} deg, "
          f"duration={STIM_DURATION_MS:.0f} ms")

    if ring_params is not None:
        print(f"Connectivity: Gaussian profile, w_inter = {ring_params.w_pyr_pyr_inter:.2f}, "
              f"sigma = {ring_params.sigma_pyr_deg:.1f} deg")
        print(f"Inhibition:   Uniform PV, w_pv = {ring_params.w_pv_global:.2f}")

    response_onset = getattr(args, 'response_onset_ms', 0.0)
    if response_onset > 0:
        response_factor = getattr(args, 'response_factor', 0.5)
        response_duration = getattr(args, 'response_duration_ms', 500.0)
        print(f"Response transient: +{response_factor:.0%} of I0 to all populations, "
              f"{response_onset:.0f} ms after delay end, duration={response_duration:.0f} ms")


def _fmt(v: float) -> str:
    """Format float for labels/paths: drop trailing zeros, keep at most 2 decimals."""
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
    """Build a directory-safe label encoding n_nodes and connectivity weights.

    Example: 128_inhib_10_excit_7
    """
    return f"{rp.n_nodes}_inhib_{_fmt(rp.w_pv_global)}_excit_{_fmt(rp.w_pyr_pyr_inter)}"


def _stim_label(amp_factor: float) -> str:
    """Short label for stimulus amplitude factor, used in plot titles."""
    return f"amp={_fmt(amp_factor)}×"


def _weights_label(rp: RingParams) -> str:
    """Short label for PYR and PV weights, used in plot titles."""
    return f"w_pyr={_fmt(rp.w_pyr_pyr_inter)}, w_pv={_fmt(rp.w_pv_global)}"


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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by ring-run and ring-study."""
    parser.add_argument("--params_json", type=str, default="",
                        help="Load local circuit parameters from JSON file")
    parser.add_argument("--n_nodes", type=int, default=128,
                        help="Number of nodes on the ring (default: 128)")
    parser.add_argument("--amplitude", type=float,
                        help="Stimulus amplitude as factor of I_ext_pyr baseline "
                             "(e.g. 30 = 30× baseline current)")
    parser.add_argument("--delay_ms", type=float, default=5000.0,
                        help="Delay period duration in ms (default: 5000)")
    parser.add_argument("--seed", type=_parse_seed, default=42,
                        help="Random seed for reproducibility (default: 42). "
                             "Use 'rdm' for a truly random seed.")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't display plots (useful for batch processing)")
    parser.add_argument("--response_onset_ms", type=float, default=0.0,
                        help="Response transient onset after delay end (ms). "
                             "0 = disabled (default: 0)")
    parser.add_argument("--response_duration_ms", type=float, default=500.0,
                        help="Duration of response transient (ms, default: 500)")
    parser.add_argument("--response_factor", type=float, default=0.5,
                        help="Response transient amplitude as fraction of I0 "
                             "(default: 0.5 = +50%% of baseline to all populations)")
    parser.add_argument("--post_response_ms", type=float, default=3000.0,
                        help="Simulation time after response transient ends (ms, default: 3000)")
    parser.add_argument("--total_time_ms", type=float, default=None,
                        help="Total simulation time in ms (overrides automatic timing if set)")
    parser.add_argument("--record_dt_ms", type=float, default=5.0,
                        help="Recording time step in ms (default: 5.0). "
                             "Only every record_dt_ms the state is stored.")
    # Connectivity parameters
    parser.add_argument("--sigma_pyr_deg", type=float, default=30.0,
                        help="PYR→PYR connectivity width in degrees (default: 30.0)")
    parser.add_argument("--w_pyr_pyr_inter", type=float, required=True,
                        help="Total PYR→PYR coupling strength")
    parser.add_argument("--w_pv_global", type=float, required=True,
                        help="Total PV→PYR global inhibition strength")
    # Noise parameters
    parser.add_argument("--sigma_noise", type=float, default=None,
                        help="Noise amplitude sigma_s (overrides params_json value). "
                             "Default uses the value in CircuitParams (~5.89).")
    parser.add_argument("--snapshot_anim_fps", type=int, default=30,
                        help="FPS for snapshot evolution animation (default: 30)")
    parser.add_argument("--snapshot_anim_step_ms", type=float, default=2.0,
                        help="Time step between animation frames in ms (default: 2.0 — 60ms sim = 1s video at 30fps)")
    parser.add_argument("--quality_high", action="store_true",
                        help="Use moderately higher-quality animation rendering (higher DPI + AV1 quality; ~up to 2× slower encoding)")


# ============================================================================
# RUN SUBCOMMAND
# ============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single condition and plot results."""
    _resolve_seed(args)
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, ring_params, T_ms, stimuli, amp = _build_common(args)

    if not args.params_json:
        print("Using default parameters")
    else:
        print(f"Loaded parameters from: {args.params_json}")

    if args.sigma_noise is not None:
        from dataclasses import replace as _replace
        base_params = _replace(base_params, sigma_s=args.sigma_noise)
        print(f"Noise amplitude overridden: sigma_s = {args.sigma_noise}")

    _print_config(args, amp, base_params, T_ms, ring_params)

    cond_key = args.condition
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    local_params = _apply_response_transient(local_params, args, delay_end_ms)

    amp_dir = f"amp{_fmt(amp)}"
    conn_label = _network_label(ring_params)
    conn_dir = os.path.join(_output_dir("figs/ring/run", args.params_json), conn_label)
    out_dir = os.path.join(conn_dir, amp_dir, cond_key)
    os.makedirs(out_dir, exist_ok=True)

    connectivity_path = os.path.join(conn_dir, "connectivity.png")
    if not os.path.exists(connectivity_path):
        plot_connectivity_matrices(ring_params, save_path=connectivity_path)
        plt.close()

    print(f"\nSimulating: {condition.label} ({cond_key})")
    print(f"  T = {T_ms:.0f} ms, delay = {args.delay_ms:.0f} ms")
    result = simulate_ring(local_params, ring_params, T_ms=T_ms,
                           stimuli=stimuli, seed=args.seed,
                           record_dt_ms=args.record_dt_ms,
                           record_adaptation=True)

    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, T_ms)
    suptitle = _stim_label(amp)
    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)

    plot_ring_dashboard(result, save_path=os.path.join(out_dir, "dashboard.png"),
                        time_range=time_range, t_offset=t_offset, suptitle=suptitle)
    plt.close()

    anim_path = os.path.join(out_dir, "snapshot_evolution.mp4")
    mp4_pbar = _start_mp4_progress(
        total_videos=1,
        frame_step_ms=args.snapshot_anim_step_ms,
        fps=args.snapshot_anim_fps,
        sample_time_range=time_range,
    )
    try:
        mp4_pbar.set_postfix_str(f"cond={cond_key}")
        fig_anim, _ = animate_ring_snapshot_evolution(
            result,
            save_path=anim_path,
            time_range=time_range,
            t_offset=t_offset,
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
            suptitle=f"{condition.label} — Snapshot Evolution ({suptitle})",
            show_asymmetry=True,
            **anim_quality_kwargs,
        )
        plt.close(fig_anim)
        mp4_pbar.update(1)
    finally:
        mp4_pbar.close()
    print(f"Saved animation: {anim_path}")

    plot_bump_metrics_over_time(result, time_range=time_range, t_offset=t_offset)
    plt.suptitle(f"Bump Metrics Over Time  ({suptitle})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bump_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plot_population_activity(result, t_offset=t_offset,
                              save_path=os.path.join(out_dir, "population_activity.png"))
    plt.close()

    plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))
    plt.close()

    print(f"\nFigures saved to {out_dir}/")


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
]

_METRIC_KEYS = [
    'center_mean_deg', 'center_std_deg', 'amplitude_mean',
    'width_mean_deg', 'drift_rate_deg_per_s', 'diffusion_deg2_per_s',
    'error_from_cue_deg',
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
            row[k] = m[k]
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
    ring_params: RingParams,
    connectivity: RingConnectivity,
    burnin_states: dict[str, tuple[np.ndarray, np.ndarray]],
    delay_eval_times: list[float],
    T_ms_full: float,
):
    """Initialize worker process with shared parameters."""
    global _ring_sim_args
    _ring_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
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
    ring_params = cfg['ring_params']
    connectivity = cfg['connectivity']
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
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    # Determine conditions
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

    if args.amplitudes is None:
        amplitudes = [args.amplitude]
    elif isinstance(args.amplitudes, (list, tuple)):
        amplitudes = list(args.amplitudes)
    else:
        amplitudes = [float(args.amplitudes)]
    n_trials = getattr(args, 'n_trials', 1)
    n_workers = _resolve_workers(args)
    no_cache = getattr(args, 'no_cache', False)
    error_band = getattr(args, 'error_band', 'sem')

    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/run", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "study_metrics.csv")

    # Compute T_ms using first amplitude (timing is same for all amplitudes)
    _, _, T_ms_full, _, _ = _build_common(args, amp_factor=amplitudes[0])
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS

    _print_config(args, amplitudes[0], base_params, T_ms_full, ring_params)

    print(f"\nStudy configuration:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Amplitudes (× I_ext_pyr): {', '.join(_fmt(a) for a in amplitudes)}")
    print(f"  Delay = {args.delay_ms:.0f} ms, trials = {n_trials}, workers = {n_workers}")

    # --- Pre-compute connectivity (once) ---
    connectivity = RingConnectivity.from_params(ring_params)

    # --- Burn-in states (once per condition) ---
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
        for amp in amplitudes:
            for trial_idx, seed in enumerate(trial_seeds):
                if (cond_key, amp, trial_idx) not in completed:
                    jobs.append((cond_key, amp, trial_idx, seed))

    total_jobs = len(condition_keys) * len(amplitudes) * n_trials
    cached_jobs = total_jobs - len(jobs)
    print(f"\nJobs: {len(jobs)} to run, {cached_jobs} cached")

    # --- Run simulations ---
    all_results: list[dict] = []

    if jobs:
        args_dict = _args_to_dict(args)
        init_args = (
            args_dict, base_params, ring_params, connectivity,
            burnin_states, delay_eval_times, T_ms_full,
        )

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
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
            metrics = {k: float(row[k]) for k in _METRIC_KEYS}
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
    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
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
            suptitle = f"{_stim_label(amp)}, {_weights_label(ring_params)}"

            metrics_over_delay_agg: dict[str, list[dict]] = {}
            delay_end_metrics_agg: dict[str, dict] = {}
            comparison_data: dict[str, dict] = {}

            for cond_key in condition_keys:
                trial_results = [
                    r for r in all_results
                    if r['cond_key'] == cond_key and r['amplitude'] == amp
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

            if delay_eval_times and metrics_over_delay_agg:
                band_tag = f", {n_trials} trials, ±{error_band.upper()}" if n_trials > 1 else ""
                # Skip first point — bump is still forming at the earliest eval time
                plot_metrics_vs_delay(
                    {ck: v[1:] for ck, v in metrics_over_delay_agg.items()},
                    delay_labels=delay_labels[1:],
                    save_path=os.path.join(amp_out, f"metrics_vs_delay_{error_band}.png"),
                    suptitle=f"Bump Metrics During Delay  ({suptitle}{band_tag})",
                    error_band=error_band,
                    separate_app=False,  # all conditions on same plot for delay time course
                )
                plt.close()

            if comparison_data:
                plot_bump_metrics_comparison(
                    comparison_data,
                    save_path=os.path.join(amp_out, "bump_metrics_comparison.png"),
                    suptitle=f"Bump Metrics Comparison  ({suptitle})",
                )
                plt.close()

            anim_dir = os.path.join(amp_out, "snapshot_evolution")
            os.makedirs(anim_dir, exist_ok=True)
            for cond_key in condition_keys:
                mp4_pbar.set_postfix_str(f"amp={_fmt(amp)} cond={cond_key}")
                condition = STUDY_CONDITIONS[cond_key]
                local_params = apply_condition(base_params, condition)
                delay_end_ms = stim_offset_ms + args.delay_ms
                local_params = _apply_response_transient(local_params, args, delay_end_ms)
                cue_current = amp * base_params.I_ext_pyr()
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
                    ring_params,
                    T_ms=T_ms_full,
                    stimuli=stimuli,
                    seed=vis_seed,
                    connectivity=connectivity,
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
        mp4_pbar.close()

    # Cross-amplitude comparison (full delay)
    if len(amplitudes) > 1:
        band_tag = f"  ({n_trials} trials, ±{error_band.upper()})" if n_trials > 1 else ""
        plot_metrics_vs_amplitude(
            all_delay_metrics_agg,
            amplitude_values=amplitudes,
            save_path=os.path.join(out_dir, f"metrics_vs_amplitude_{error_band}.png"),
            suptitle=f"Metrics vs Amplitude (full delay){band_tag}  [{_weights_label(ring_params)}]",
            error_band=error_band,
            separate_app=False,  # all conditions on same plot for amplitude comparison
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
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
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

    _, _, T_ms_full, _, amp_factor = _build_common(args)

    conn_label = _network_label(ring_params)
    amp_label = f"amp{_fmt(amp_factor)}"
    out_dir = os.path.join(
        _output_dir("figs/ring/diffusion", args.params_json),
        conn_label,
        amp_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    _print_config(args, amp_factor, base_params, T_ms_full, ring_params)

    print(f"\nDiffusion analysis:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Trials per condition: {n_trials}")
    print(f"  Delay = {args.delay_ms:.0f} ms")

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
            with ProcessPoolExecutor(
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

def cmd_drift_field(args: argparse.Namespace) -> None:
    """Run distractor drift field analysis across conditions."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
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
    distractor_step = args.distractor_steps
    n_workers = _resolve_workers(args)

    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/drift_field", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Build offsets
    offsets_deg = np.arange(0, 180.0 + distractor_step / 2, distractor_step)

    # We need enough time for cue + distractor + post-distractor measurement
    distractor_onset_ms = args.distractor_onset_ms
    distractor_duration_ms = args.distractor_duration_ms

    # Override delay to ensure simulation is long enough
    min_delay = distractor_onset_ms + distractor_duration_ms + 200  # 200ms buffer
    effective_delay = max(args.delay_ms, min_delay)

    _, _, T_ms_full, _, amp_factor = _build_common(args)
    # Recompute T_ms_full with effective delay
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    T_ms_full = stim_offset_ms + effective_delay + BURN_IN_MS
    # Actually T_ms_full should include burn-in already from _build_common, let's be explicit
    T_ms_full = BURN_IN_MS + (STIM_ONSET_MS - BURN_IN_MS) + STIM_DURATION_MS + effective_delay

    _print_config(args, amp_factor, base_params, T_ms_full, ring_params)

    print(f"\nDrift field analysis:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Trials per offset: {n_trials}")
    print(f"  Offsets: {offsets_deg[0]:.0f}° to {offsets_deg[-1]:.0f}° "
          f"in {distractor_step:.0f}° steps ({len(offsets_deg)} offsets)")
    print(f"  Distractor: amp={args.distractor_amplitude:.1f}× I_ext_pyr, "
          f"onset={distractor_onset_ms:.0f}ms after stim, "
          f"duration={distractor_duration_ms:.0f}ms")
    total = len(condition_keys) * len(offsets_deg) * n_trials
    print(f"  Total simulations: {total}")

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
    raw_csv = os.path.join(out_dir, "drift_field_trials.csv")
    summary_csv = os.path.join(out_dir, "drift_field_summary.csv")

    # --- Check for cached trial data ---
    from collections import defaultdict
    all_results: list[dict] = []
    loaded_from_cache = False

    if os.path.exists(raw_csv) and os.path.exists(summary_csv):
        try:
            with open(summary_csv, newline='') as _f:
                summary_rows = list(csv.DictReader(_f))

            params_ok = summary_rows and all(
                float(r['distractor_amplitude_factor']) == args.distractor_amplitude
                and float(r['distractor_duration_ms']) == distractor_duration_ms
                and float(r['distractor_onset_ms']) == distractor_onset_ms
                for r in summary_rows
            )

            if params_ok:
                with open(raw_csv, newline='') as _f:
                    trial_rows = list(csv.DictReader(_f))

                offsets_set = {float(o) for o in offsets_deg}
                counts: dict[tuple, int] = defaultdict(int)
                for row in trial_rows:
                    counts[(row['condition_key'], float(row['offset_deg']))] += 1

                cache_valid = all(
                    counts[(ck, off)] >= n_trials
                    for ck in condition_keys
                    for off in offsets_set
                )

                if cache_valid:
                    print(f"\nLoading cached trial data from {raw_csv}")
                    seen: dict[tuple, int] = defaultdict(int)
                    for row in trial_rows:
                        ck = row['condition_key']
                        off = float(row['offset_deg'])
                        if ck not in condition_keys or off not in offsets_set:
                            continue
                        if seen[(ck, off)] >= n_trials:
                            continue
                        all_results.append({
                            'cond_key': ck,
                            'offset_deg': off,
                            'trial_idx': int(row['trial_idx']),
                            'displacement_rad': float(row['displacement_rad']),
                            'pre_amp': float(row['pre_amp']),
                            'post_amp': float(row['post_amp']),
                        })
                        seen[(ck, off)] += 1
                    loaded_from_cache = True
                    print(f"  Loaded {len(all_results)} cached trials.")
        except Exception as _e:
            print(f"  Cache read failed ({_e}), rerunning simulations.")

    if not loaded_from_cache:
        # --- Build jobs ---
        jobs = []
        for cond_key in condition_keys:
            for offset in offsets_deg:
                for trial_idx, seed in enumerate(trial_seeds):
                    jobs.append((cond_key, float(offset), trial_idx, seed))

        # --- Run simulations ---
        args_dict = {
            **_args_to_dict(args),
            'amplitude': amp_factor,
            'distractor_onset_ms': distractor_onset_ms,
            'distractor_duration_ms': distractor_duration_ms,
            'distractor_amplitude': args.distractor_amplitude,
        }
        init_args = (
            args_dict, base_params, ring_params, connectivity,
            burnin_states, T_ms_full,
        )

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_drift_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_drift_run_single, job): job for job in jobs}
                with tqdm(total=len(jobs), desc="Drift field trials", unit="trial", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _drift_init_worker(*init_args)
            for job in tqdm(jobs, desc="Drift field trials", unit="trial"):
                all_results.append(_drift_run_single(job))

    # --- Compute drift field per condition ---
    distractor_duration_s = distractor_duration_ms / 1000.0
    drift_data: dict[str, dict] = {}

    for cond_key in condition_keys:
        displacement_per_offset: dict[float, list[float]] = {}
        for offset in offsets_deg:
            trials = [r for r in all_results
                      if r['cond_key'] == cond_key and r['offset_deg'] == offset]
            displacement_per_offset[float(offset)] = [r['displacement_rad'] for r in trials]

        offsets_out, A_hat, A_hat_sem, A_hat_sd = compute_drift_field(
            displacement_per_offset, distractor_duration_s,
        )

        drift_data[cond_key] = {
            'offsets_deg': offsets_out,
            'A_hat': A_hat,
            'A_hat_sem': A_hat_sem,
            'A_hat_sd': A_hat_sd,
        }

        cond_label = STUDY_CONDITIONS[cond_key].name
        peak_idx = np.argmax(np.abs(A_hat))
        print(f"  {cond_label}: peak A_hat = {A_hat[peak_idx]:.4e} rad/s "
              f"at Δφ = {offsets_out[peak_idx]:.0f}°")

    # --- Save CSVs (skipped when loaded from cache) ---
    if not loaded_from_cache:
        # 1. Per-trial raw data
        with open(raw_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'offset_deg', 'trial_idx', 'seed',
                'displacement_rad', 'pre_amp', 'post_amp',
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    'condition_key': r['cond_key'],
                    'offset_deg': r['offset_deg'],
                    'trial_idx': r['trial_idx'],
                    'seed': trial_seeds[r['trial_idx']],
                    'displacement_rad': r['displacement_rad'],
                    'pre_amp': r['pre_amp'],
                    'post_amp': r['post_amp'],
                })

        # 2. Aggregated drift field summary
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'offset_deg', 'A_hat_rad_per_s',
                'A_hat_sem', 'A_hat_sd', 'n_trials',
                'distractor_amplitude_factor', 'distractor_duration_ms',
                'distractor_onset_ms',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                d = drift_data[cond_key]
                for i in range(len(d['offsets_deg'])):
                    writer.writerow({
                        'condition_key': cond_key,
                        'offset_deg': d['offsets_deg'][i],
                        'A_hat_rad_per_s': d['A_hat'][i],
                        'A_hat_sem': d['A_hat_sem'][i],
                        'A_hat_sd': d['A_hat_sd'][i],
                        'n_trials': n_trials,
                        'distractor_amplitude_factor': args.distractor_amplitude,
                        'distractor_duration_ms': distractor_duration_ms,
                        'distractor_onset_ms': distractor_onset_ms,
                    })

        print(f"\nCSVs saved to {out_dir}/")
        print(f"  drift_field_trials.csv  (per-trial raw displacements)")
        print(f"  drift_field_summary.csv  (A_hat per condition × offset)")

    # --- Plot ---
    error_band = getattr(args, 'error_band', 'sem')
    save_path = os.path.join(out_dir, f"drift_field_{error_band}.png")
    band_tag = f"  ({n_trials} trials, ±{error_band.upper()})" if n_trials > 1 else ""
    plot_drift_field(
        drift_data,
        save_path=save_path,
        suptitle=f"Distractor Drift Field{band_tag}",
        error_band=error_band,
    )
    plt.close()

    print(f"Drift field figure saved to {save_path}")


# ============================================================================
# CALIBRATE: HELPERS
# ============================================================================

def _calibration_network_label(rp: RingParams) -> str:
    """Build a directory-safe label for calibration output.

    Omits the excit weight since that parameter is swept during calibration.
    Example: 128_inhib_10
    """
    return f"{rp.n_nodes}_inhib_{_fmt(rp.w_pv_global)}"


def _unique_path(path: str) -> str:
    """Return path if it does not exist, else append _1, _2, ... before extension."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ============================================================================
# CALIBRATE: PARALLEL WORKER
# ============================================================================

_calibrate_sim_args: Optional[dict] = None


def _calibrate_init_worker(
    args_dict: dict,
    base_params: CircuitParams,
    ring_params_base: RingParams,
    connectivity_cache: dict[float, RingConnectivity],
    burnin_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]],
    T_ms_full: float,
    eval_times_ms: list[float],
):
    """Initialize worker process for calibration."""
    global _calibrate_sim_args
    _calibrate_sim_args = {
        'args_dict': args_dict,
        'base_params': base_params,
        'ring_params_base': ring_params_base,
        'connectivity_cache': connectivity_cache,
        'burnin_cache': burnin_cache,
        'T_ms_full': T_ms_full,
        'eval_times_ms': eval_times_ms,
    }


def _calibrate_run_single(job: tuple) -> dict:
    """Run a single calibration trial.

    Job format: (cond_key, amplitude, w_inter, trial_idx, seed)
    amplitude=0 means no-stimulus baseline trial.
    """
    global _calibrate_sim_args
    cfg = _calibrate_sim_args
    cond_key, amplitude, w_inter, trial_idx, seed = job

    args_d = cfg['args_dict']
    base_params = cfg['base_params']
    rp_base = cfg['ring_params_base']
    connectivity = cfg['connectivity_cache'][w_inter]
    T_ms_full = cfg['T_ms_full']

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    r0, I_adapt0 = cfg['burnin_cache'][(cond_key, w_inter)]

    ring_params = replace(rp_base, w_pyr_pyr_inter=w_inter)

    T_ms_short = T_ms_full - BURN_IN_MS

    # Build stimuli (empty if baseline)
    if amplitude > 0:
        actual_current = amplitude * base_params.I_ext_pyr()
        stimuli_short = [
            RingStimulus(
                center_deg=STIM_CENTER_DEG, amplitude=actual_current,
                sigma_deg=STIM_SIGMA_DEG,
                onset_ms=STIM_ONSET_MS - BURN_IN_MS,
                duration_ms=STIM_DURATION_MS,
            ),
        ]
    else:
        stimuli_short = None

    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
        seed=seed, connectivity=connectivity,
        record_dt_ms=args_d.get('record_dt_ms', 5.0),
    )

    result.t_ms += BURN_IN_MS

    # Decode bump metrics at evaluation times
    eval_times = cfg['eval_times_ms']

    A_hat_at_times = []
    for eval_t in eval_times:
        idx = np.argmin(np.abs(result.t_ms - eval_t))
        activity = result.r[idx, :, 0]  # PYR
        _, amp_val = population_vector_decode(activity, ring_params.node_angles_rad)
        A_hat_at_times.append(float(amp_val))

    # Final delay metrics
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args_d['delay_ms']
    final_idx = np.argmin(np.abs(result.t_ms - delay_end_ms))
    final_activity = result.r[final_idx, :, 0]
    center_rad_final, A_hat_final = population_vector_decode(
        final_activity, ring_params.node_angles_rad,
    )
    center_final_deg = float(center_rad_final) * 180 / np.pi
    from .analysis import angular_distance_deg
    error_deg = angular_distance_deg(center_final_deg, STIM_CENTER_DEG) if amplitude > 0 else np.nan

    # Peak PYR rate during delay
    delay_mask = (
        (result.t_ms >= stim_offset_ms + TRANSIENT_SKIP_TIME_MS)
        & (result.t_ms <= delay_end_ms)
    )
    if np.any(delay_mask):
        peak_pyr = float(np.max(result.r[delay_mask, :, 0]))
    else:
        peak_pyr = 0.0

    del result

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'w_inter': w_inter,
        'trial_idx': trial_idx,
        'seed': seed,
        'A_hat_timecourse': A_hat_at_times,
        'A_hat_final': float(A_hat_final),
        'peak_pyr_rate': peak_pyr,
        'center_final_deg': center_final_deg,
        'error_from_cue_deg': float(error_deg),
    }


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
    """Compute calibration metrics from a RingSimulationResult.

    Assumes result.t_ms already has BURN_IN_MS offset applied.
    """
    from .analysis import angular_distance_deg
    ring_params = result.ring_params
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + delay_ms

    A_hat_at_times = []
    for eval_t in eval_times_ms:
        idx = np.argmin(np.abs(result.t_ms - eval_t))
        activity = result.r[idx, :, 0]
        _, amp_val = population_vector_decode(activity, ring_params.node_angles_rad)
        A_hat_at_times.append(float(amp_val))

    final_idx = np.argmin(np.abs(result.t_ms - delay_end_ms))
    final_activity = result.r[final_idx, :, 0]
    center_rad_final, A_hat_final = population_vector_decode(
        final_activity, ring_params.node_angles_rad,
    )
    center_final_deg = float(center_rad_final) * 180 / np.pi
    error_deg = angular_distance_deg(center_final_deg, STIM_CENTER_DEG) if amplitude > 0 else np.nan

    delay_mask = (
        (result.t_ms >= stim_offset_ms + TRANSIENT_SKIP_TIME_MS)
        & (result.t_ms <= delay_end_ms)
    )
    peak_pyr = float(np.max(result.r[delay_mask, :, 0])) if np.any(delay_mask) else 0.0

    return {
        'cond_key': cond_key,
        'amplitude': amplitude,
        'w_inter': w_inter,
        'trial_idx': trial_idx,
        'seed': seed,
        'A_hat_timecourse': A_hat_at_times,
        'A_hat_final': float(A_hat_final),
        'peak_pyr_rate': peak_pyr,
        'center_final_deg': center_final_deg,
        'error_from_cue_deg': float(error_deg),
    }


# ============================================================================
# CALIBRATE CSV CACHE HELPERS
# ============================================================================

def _is_baseline_cached(
    cond_dir: str,
    cond_key: str,
    w_inter_values: list[float],
) -> bool:
    """Return True if baseline_A_hat.csv exists and covers all required w_inter values."""
    baseline_csv = os.path.join(cond_dir, "baseline_A_hat.csv")
    if not os.path.exists(baseline_csv):
        return False
    try:
        w_present: set[float] = set()
        with open(baseline_csv) as f:
            for row in csv.DictReader(f):
                if row.get('condition_key', '') == cond_key:
                    w_present.add(float(row['w_inter']))
        return set(w_inter_values) <= w_present
    except Exception:
        return False


def _is_calibrate_cached(
    cond_dir: str,
    cond_key: str,
    amplitudes: list[float],
    w_inter_values: list[float],
    n_trials: int,
) -> bool:
    """Return True if calibration CSV files exist and cover all expected trials.

    Compatible with both old CSVs (no a_hat_timecourse / baseline_A_hat.csv)
    and new ones.  Only requires calibration_results.csv + calibration_summary.csv.
    """
    results_csv = os.path.join(cond_dir, "calibration_results.csv")
    summary_csv = os.path.join(cond_dir, "calibration_summary.csv")
    if not (os.path.exists(results_csv) and os.path.exists(summary_csv)):
        return False
    try:
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        present = {
            (float(r['amplitude']), float(r['w_inter']), int(r['trial_idx']))
            for r in rows if r.get('condition_key', '') == cond_key
        }
        needed = {
            (amp, w, ti)
            for amp in amplitudes
            for w in w_inter_values
            for ti in range(n_trials)
        }
        if not needed <= present:
            return False
        # Verify summary has noise_threshold for all w_inter values
        with open(summary_csv) as f:
            srows = list(csv.DictReader(f))
        w_in_summary = {
            float(r['w_inter']) for r in srows
            if r.get('condition_key', '') == cond_key
            and r.get('noise_threshold', '') != ''
        }
        return set(w_inter_values) <= w_in_summary
    except Exception:
        return False


def _load_calibrate_grid_results(cond_dir: str, cond_key: str) -> list[dict]:
    """Load per-trial grid results from calibration_results.csv.

    The a_hat_timecourse column is optional (absent in old CSV format).
    """
    results_csv = os.path.join(cond_dir, "calibration_results.csv")
    results = []
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get('condition_key', '') != cond_key:
                continue
            tc_str = row.get('a_hat_timecourse', '')
            tc = [float(v) for v in tc_str.split()] if tc_str else []
            results.append({
                'cond_key': row['condition_key'],
                'amplitude': float(row['amplitude']),
                'w_inter': float(row['w_inter']),
                'trial_idx': int(row['trial_idx']),
                'seed': int(row['seed']),
                'A_hat_final': float(row['A_hat_final']),
                'A_hat_timecourse': tc,
                'peak_pyr_rate': float(row['peak_pyr_rate']),
                'center_final_deg': float(row['center_final_deg']),
                'error_from_cue_deg': float(row['error_from_cue_deg']),
            })
    return results


def _load_calibrate_baseline(
    cond_dir: str,
    cond_key: str,
    w_inter_values: list[float],
    noise_percentile: float,
) -> tuple[dict, dict, set]:
    """Load noise thresholds and baseline A_hat distributions.

    Primary source: baseline_A_hat.csv (full distribution, written by new runs).
    Fallback:       calibration_summary.csv noise_threshold column (old runs).

    Returns:
        noise_thresholds:   {(cond_key, w): float}
        baseline_A_hat_data: {(cond_key, w): np.ndarray}  (empty array if only summary available)
        saturated_w_values: set of w_inter values excluded due to node saturation

    w_inter values where saturation is detected (nodes hit firing-rate ceiling,
    producing near-zero A_hat artefacts) are excluded from both dicts.
    """
    noise_thresholds: dict = {}
    baseline_A_hat_data: dict = {}
    saturated_w: set[float] = set()

    baseline_csv = os.path.join(cond_dir, "baseline_A_hat.csv")
    if os.path.exists(baseline_csv):
        data: dict = {}
        with open(baseline_csv) as f:
            for row in csv.DictReader(f):
                if row.get('condition_key', '') != cond_key:
                    continue
                key = (cond_key, float(row['w_inter']))
                data.setdefault(key, []).append(float(row['a_hat_value']))
        for key, vals in data.items():
            all_A = np.array(vals)
            baseline_A_hat_data[key] = all_A
            noise_thresholds[key] = compute_noise_floor(all_A, noise_percentile)

        # Detect and remove saturated w_inter values
        flat_by_w = {w: baseline_A_hat_data[(cond_key, w)]
                     for (ck, w) in baseline_A_hat_data if ck == cond_key}
        saturated_w = detect_saturated_w_values(flat_by_w)
        if saturated_w:
            print(f"  [saturation] Excluding w_inter values where all nodes hit "
                  f"firing-rate ceiling: {sorted(saturated_w)}")
            for w in saturated_w:
                baseline_A_hat_data.pop((cond_key, w), None)
                noise_thresholds.pop((cond_key, w), None)

        non_saturated = {w for (_, w) in noise_thresholds}
        if set(w_inter_values) - saturated_w <= non_saturated:
            return noise_thresholds, baseline_A_hat_data, saturated_w

    # Fallback: read pre-computed noise_threshold from calibration_summary.csv
    summary_csv = os.path.join(cond_dir, "calibration_summary.csv")
    if not os.path.exists(summary_csv):
        return noise_thresholds, baseline_A_hat_data, saturated_w
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if row.get('condition_key', '') != cond_key:
                continue
            w = float(row['w_inter'])
            key = (cond_key, w)
            if key not in noise_thresholds and row.get('noise_threshold', ''):
                noise_thresholds[key] = float(row['noise_threshold'])
                baseline_A_hat_data[key] = np.array([])  # no distribution available
    return noise_thresholds, baseline_A_hat_data, saturated_w


def _load_baseline_trial_counts(
    cond_dir: str,
    cond_key: str,
) -> tuple[dict[tuple[str, float], int], bool]:
    """Load cached baseline trial counts from baseline_A_hat.csv.

    Returns:
        counts: {(cond_key, w_inter): n_unique_trials}
        has_trial_metadata: True when trial_idx metadata is available.
    """
    baseline_csv = os.path.join(cond_dir, "baseline_A_hat.csv")
    if not os.path.exists(baseline_csv):
        return {}, False

    try:
        with open(baseline_csv) as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if 'trial_idx' not in fieldnames:
                return {}, False

            trial_sets: dict[tuple[str, float], set[int]] = {}
            for row in reader:
                if row.get('condition_key', '') != cond_key:
                    continue
                trial_str = row.get('trial_idx', '')
                if trial_str == '':
                    continue
                key = (cond_key, float(row['w_inter']))
                trial_sets.setdefault(key, set()).add(int(trial_str))

        counts = {key: len(v) for key, v in trial_sets.items()}
        return counts, True
    except Exception:
        return {}, False


# ============================================================================
# NOISE FLOOR HELPER
# ============================================================================

def _run_noise_floor_for_conditions(
    conditions_to_run: list[str],
    w_inter_values: list[float],
    ring_params_base: RingParams,
    base_params,
    n_baseline: int,
    noise_percentile: float,
    out_dir: str,
    n_workers: int,
    batch_chunk_size: int,
    seed: int,
    delay_ms: float,
    record_dt_ms: float = 5.0,
    w_inter_values_by_condition: Optional[dict[str, list[float]]] = None,
    trials_to_add_by_key: Optional[dict[tuple[str, float], int]] = None,
    trial_start_idx_by_key: Optional[dict[tuple[str, float], int]] = None,
    preserve_existing_cache: bool = True,
) -> tuple[dict, dict]:
    """Run no-stimulus baseline simulations and estimate noise floor.

    Returns (noise_thresholds, baseline_A_hat_data) dicts keyed by (cond_key, w_inter).
    Saves baseline_A_hat.csv per condition under out_dir/<cond_key>/.
    """
    from tqdm import tqdm

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + delay_ms
    T_ms_full = delay_end_ms
    T_ms_short = T_ms_full - BURN_IN_MS

    # Evaluation times during delay (every 200 ms after stim offset)
    eval_step_ms = 200.0
    eval_times_ms: list[float] = []
    t = stim_offset_ms + eval_step_ms
    while t <= delay_end_ms:
        eval_times_ms.append(t)
        t += eval_step_ms

    baseline_A_hat_data: dict[tuple[str, float], np.ndarray] = {}
    noise_thresholds: dict[tuple[str, float], float] = {}

    # Resolve per-condition w_inter values to run
    if w_inter_values_by_condition is None:
        w_values_by_cond = {ck: list(w_inter_values) for ck in conditions_to_run}
    else:
        w_values_by_cond = {
            ck: list(w_inter_values_by_condition.get(ck, []))
            for ck in conditions_to_run
        }

    # Build per-(condition, w_inter) trial plan
    run_plan: dict[tuple[str, float], tuple[int, int]] = {}
    for ck in conditions_to_run:
        for w in w_values_by_cond.get(ck, []):
            key = (ck, w)
            n_to_add = (
                int(trials_to_add_by_key.get(key, n_baseline))
                if trials_to_add_by_key is not None
                else int(n_baseline)
            )
            start_idx = (
                int(trial_start_idx_by_key.get(key, 0))
                if trial_start_idx_by_key is not None
                else 0
            )
            if n_to_add > 0:
                run_plan[key] = (start_idx, n_to_add)

    unique_w_inter_values = sorted({w for (_, w) in run_plan.keys()})
    if not unique_w_inter_values:
        return noise_thresholds, baseline_A_hat_data

    # Pre-compute connectivity
    print("\nBuilding connectivity matrices...")
    connectivity_cache: dict[float, RingConnectivity] = {}
    for w in tqdm(unique_w_inter_values, desc="Connectivity", unit="w"):
        rp = replace(ring_params_base, w_pyr_pyr_inter=w)
        connectivity_cache[w] = RingConnectivity.from_params(rp)

    # Pre-compute burn-in states
    print("Computing burn-in states...")
    burnin_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for w in tqdm(unique_w_inter_values, desc="Burn-in", unit="w_inter"):
        rp = replace(ring_params_base, w_pyr_pyr_inter=w)
        conn = connectivity_cache[w]
        conds_for_w = [ck for ck in conditions_to_run if (ck, w) in run_plan]
        params_list = [apply_condition(base_params, STUDY_CONDITIONS[ck])
                       for ck in conds_for_w]
        batch_results = simulate_ring_batch(
            params_list, rp, T_ms=BURN_IN_MS,
            noise_type="white", record_dt_ms=1000.0,
            connectivity=conn,
        )
        for ck, res in zip(conds_for_w, batch_results):
            burnin_cache[(ck, w)] = (res.r[-1].copy(), res.I_adapt_final.copy())

    # Baseline simulations
    print(f"\n--- Noise floor estimation ---")
    n_baseline_groups = len(run_plan)
    total_baseline_sims = sum(n_add for (_, n_add) in run_plan.values())
    print(f"  {n_baseline} trials × {n_baseline_groups} groups "
          f"= {total_baseline_sims} total baseline sims")

    baseline_results: list[dict] = []

    def _baseline_group(ck, w, start_idx, n_trials_local):
        condition = STUDY_CONDITIONS[ck]
        local_params = apply_condition(base_params, condition)
        rp = replace(ring_params_base, w_pyr_pyr_inter=w)
        conn = connectivity_cache[w]
        r0, I_adapt0 = burnin_cache[(ck, w)]
        baseline_seeds = _generate_trial_seeds_range(seed, start_idx, n_trials_local)
        group = []
        for chunk_start in range(0, n_trials_local, batch_chunk_size):
            chunk_end = min(chunk_start + batch_chunk_size, n_trials_local)
            chunk_seeds = baseline_seeds[chunk_start:chunk_end]
            chunk_n = len(chunk_seeds)
            batch_results = simulate_ring_batch(
                [local_params] * chunk_n, rp, T_ms=T_ms_short,
                stimuli=None,
                r0=r0, I_adapt0=I_adapt0,
                seeds=list(chunk_seeds),
                noise_type='white',
                connectivity=conn,
                record_dt_ms=record_dt_ms,
            )
            for ti_chunk, res in enumerate(batch_results):
                ti = chunk_start + ti_chunk
                trial_idx_abs = start_idx + ti
                res.t_ms += BURN_IN_MS
                group.append(_compute_calibrate_metrics(
                    res, ck, 0.0, w, trial_idx_abs, int(baseline_seeds[ti]),
                    eval_times_ms, delay_ms,
                ))
        return group

    baseline_groups = [
        (ck, w, start_idx, n_to_add)
        for (ck, w), (start_idx, n_to_add) in run_plan.items()
    ]
    gen = Parallel(n_jobs=n_workers, backend='loky', return_as='generator')(
        delayed(_baseline_group)(ck, w, start_idx, n_to_add)
        for ck, w, start_idx, n_to_add in baseline_groups
    )
    for group in tqdm(gen, total=len(baseline_groups),
                      desc=f"Baseline (n={n_baseline},chunk={batch_chunk_size})", unit="group"):
        baseline_results.extend(group)

    # Compute noise floor and save baseline A_hat CSVs
    for ck in conditions_to_run:
        cond_dir_save = os.path.join(out_dir, ck)
        os.makedirs(cond_dir_save, exist_ok=True)
        baseline_csv = os.path.join(cond_dir_save, "baseline_A_hat.csv")

        existing_rows: list[dict[str, str | float | int]] = []
        cond_w_values = w_values_by_cond.get(ck, [])
        if preserve_existing_cache and os.path.exists(baseline_csv):
            try:
                with open(baseline_csv) as f:
                    for row in csv.DictReader(f):
                        if row.get('condition_key', '') != ck:
                            continue
                        existing_rows.append({
                            'condition_key': ck,
                            'w_inter': float(row['w_inter']),
                            'trial_idx': row.get('trial_idx', ''),
                            'seed': row.get('seed', ''),
                            'a_hat_value': float(row['a_hat_value']),
                        })
            except Exception:
                existing_rows = []

        with open(baseline_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['condition_key', 'w_inter', 'trial_idx', 'seed', 'a_hat_value'],
            )
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)

            for w in cond_w_values:
                trials = [r for r in baseline_results
                          if r['cond_key'] == ck and r['w_inter'] == w]
                for r in trials:
                    values = [r['A_hat_final'], *r['A_hat_timecourse']]
                    for v in values:
                        writer.writerow({
                            'condition_key': ck,
                            'w_inter': w,
                            'trial_idx': int(r['trial_idx']),
                            'seed': int(r['seed']),
                            'a_hat_value': float(v),
                        })

        # Recompute thresholds from merged cache (existing + newly appended data)
        merged_nt, merged_base, _ = _load_calibrate_baseline(
            cond_dir_save, ck, w_inter_values, noise_percentile,
        )
        for w in cond_w_values:
            key = (ck, w)
            if key in merged_nt:
                baseline_A_hat_data[key] = merged_base.get(key, np.array([]))
                noise_thresholds[key] = merged_nt[key]
                cond_label = STUDY_CONDITIONS[ck].name
                n_samples = len(baseline_A_hat_data[key])
                print(f"  {cond_label}, w={w:.2f}: threshold = {noise_thresholds[key]:.4f} "
                      f"(p{noise_percentile:.0f}, n={n_samples})")

    return noise_thresholds, baseline_A_hat_data


# ============================================================================
# CALIBRATE SUBCOMMAND
# ============================================================================

def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run 2D parameter calibration (amplitude x w_inter)."""
    _resolve_seed(args)
    import json
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Setup ---
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params_base = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
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

    amplitudes = args.amplitudes
    w_inter_values = args.w_inter_values
    n_trials = args.n_trials
    noise_percentile = args.noise_percentile
    no_cache = getattr(args, 'no_cache', False)
    batch_chunk_size = getattr(args, 'batch_chunk_size', 50)
    n_workers = _resolve_workers(args)

    # Auto-cap workers based on estimated peak RAM per worker.
    # Each worker allocates r_all = (chunk, n_recorded, n_nodes, 4) float64
    # plus ~500 MB of Python overhead and working arrays.
    record_dt_ms_est = getattr(args, 'record_dt_ms', 5.0)
    T_ms_short_est = (STIM_ONSET_MS + STIM_DURATION_MS + args.delay_ms) - BURN_IN_MS
    n_recorded_est = int(np.ceil(T_ms_short_est / record_dt_ms_est)) + 1
    bytes_r_all = batch_chunk_size * n_recorded_est * ring_params_base.n_nodes * 4 * 8
    bytes_overhead = 600 * 1024 * 1024  # ~600 MB Python + numpy + working arrays
    bytes_per_worker = bytes_r_all + bytes_overhead
    try:
        mem_available = 0
        with open('/proc/meminfo') as _mf:
            for _line in _mf:
                if _line.startswith('MemAvailable:'):
                    mem_available = int(_line.split()[1]) * 1024
                    break
        if mem_available == 0:
            mem_available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 2
    except Exception:
        mem_available = 4 * 1024 ** 3
    safe_workers = max(1, int(mem_available * 0.5 / bytes_per_worker))
    if safe_workers < n_workers:
        print(f"  RAM-aware worker cap: {safe_workers} workers "
              f"(~{bytes_per_worker / 1e9:.1f} GB/worker, "
              f"{mem_available / 1e9:.1f} GB available)")
        n_workers = safe_workers

    conn_label = _calibration_network_label(ring_params_base)
    out_dir = os.path.join(
        _output_dir("figs/ring/calibration", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Timing
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    T_ms_full = delay_end_ms

    # Evaluation times during delay (every 200ms after stim offset)
    eval_step_ms = 200.0
    eval_times_ms = []
    t = stim_offset_ms + eval_step_ms
    while t <= delay_end_ms:
        eval_times_ms.append(t)
        t += eval_step_ms
    eval_times_s = np.array([(et - stim_offset_ms) / 1000.0 for et in eval_times_ms])

    # --- Cache check: which conditions already have complete CSV results? ---
    cached_conditions: set[str] = set()
    if not no_cache:
        for ck in condition_keys:
            cond_dir_check = os.path.join(out_dir, ck)
            if _is_calibrate_cached(cond_dir_check, ck, amplitudes, w_inter_values, n_trials):
                cached_conditions.add(ck)

    conditions_to_run = [ck for ck in condition_keys if ck not in cached_conditions]

    print(f"\nCalibration configuration:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    if cached_conditions:
        print(f"  Cache hit:  {', '.join(sorted(cached_conditions))} — skipping simulation")
    if conditions_to_run:
        print(f"  To simulate: {', '.join(conditions_to_run)}")
    print(f"  Amplitudes (x I_ext_pyr): {', '.join(_fmt(a) for a in amplitudes)}")
    print(f"  w_inter values: {', '.join(_fmt(w) for w in w_inter_values)}")
    print(f"  Grid points: {len(amplitudes)} x {len(w_inter_values)} = {len(amplitudes) * len(w_inter_values)}")
    print(f"  Trials per grid point: {n_trials}")
    print(f"  Delay = {args.delay_ms:.0f} ms")
    print(f"  Workers: {n_workers}, batch chunk size: {batch_chunk_size}")

    if conditions_to_run:
        total_sims = len(conditions_to_run) * len(amplitudes) * len(w_inter_values) * n_trials
        print(f"  Total grid simulations: {total_sims}")

    # Shared containers — populated by simulation OR loaded from cache below
    baseline_A_hat_data: dict[tuple[str, float], np.ndarray] = {}
    noise_thresholds: dict[tuple[str, float], float] = {}
    grid_results: list[dict] = []

    if conditions_to_run:
        # --- Pre-compute connectivity for each w_inter ---
        print("\nBuilding connectivity matrices...")
        connectivity_cache: dict[float, RingConnectivity] = {}
        for w in tqdm(w_inter_values, desc="Connectivity", unit="w"):
            rp = replace(ring_params_base, w_pyr_pyr_inter=w)
            connectivity_cache[w] = RingConnectivity.from_params(rp)

        # --- Pre-compute burn-in for each (condition, w_inter) using GPU batching ---
        print("Computing burn-in states...")
        burnin_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
        for w in tqdm(w_inter_values, desc="Burn-in", unit="w_inter"):
            rp = replace(ring_params_base, w_pyr_pyr_inter=w)
            conn = connectivity_cache[w]
            params_list = [apply_condition(base_params, STUDY_CONDITIONS[ck])
                           for ck in conditions_to_run]
            batch_results = simulate_ring_batch(
                params_list, rp, T_ms=BURN_IN_MS,
                noise_type="white", record_dt_ms=1000.0,
                connectivity=conn,
            )
            for ck, res in zip(conditions_to_run, batch_results):
                burnin_cache[(ck, w)] = (res.r[-1].copy(), res.I_adapt_final.copy())

        # --- Trial seeds ---
        trial_seeds = _generate_trial_seeds(args.seed, n_trials)
        record_dt_ms = getattr(args, 'record_dt_ms', 5.0)
        T_ms_short = T_ms_full - BURN_IN_MS

        # --- Noise floor: auto-trigger ring-noise-floor if baseline is missing/incomplete ---
        baseline_n_trials_target = 100
        conditions_missing_baseline: list[str] = []
        condition_missing_w: dict[str, list[float]] = {}
        trials_to_add_by_key: dict[tuple[str, float], int] = {}
        trial_start_idx_by_key: dict[tuple[str, float], int] = {}

        # Preload cached baselines for all conditions to run
        for ck in conditions_to_run:
            cond_dir_load = os.path.join(out_dir, ck)
            cached_nt, cached_base, _ = _load_calibrate_baseline(
                cond_dir_load, ck, w_inter_values, noise_percentile)
            noise_thresholds.update(cached_nt)
            baseline_A_hat_data.update(cached_base)

        for ck in conditions_to_run:
            cond_dir_check = os.path.join(out_dir, ck)
            trial_counts, has_trial_metadata = _load_baseline_trial_counts(cond_dir_check, ck)
            missing_ws: list[float] = []

            for w in w_inter_values:
                key = (ck, w)
                if key not in noise_thresholds:
                    missing_ws.append(w)
                    trials_to_add_by_key[key] = baseline_n_trials_target
                    trial_start_idx_by_key[key] = 0
                    continue

                if has_trial_metadata:
                    cached_trials = int(trial_counts.get(key, 0))
                    if cached_trials < baseline_n_trials_target:
                        missing_ws.append(w)
                        trials_to_add_by_key[key] = baseline_n_trials_target - cached_trials
                        trial_start_idx_by_key[key] = cached_trials

            if missing_ws:
                conditions_missing_baseline.append(ck)
                condition_missing_w[ck] = missing_ws

        if conditions_missing_baseline:
            missing_desc = [
                f"{ck} (missing w_inter: {', '.join(_fmt(w) for w in condition_missing_w.get(ck, []))})"
                for ck in conditions_missing_baseline
            ]
            print(
                f"\nNoise floor cache incomplete for: "
                f"{'; '.join(missing_desc)}\n"
                f"  Auto-running ring-noise-floor with default parameters "
                f"(n_baseline={baseline_n_trials_target}, noise_percentile={noise_percentile}).\n"
                f"  Run 'ring-noise-floor' separately to customise these."
            )
            new_nt, new_base = _run_noise_floor_for_conditions(
                conditions_to_run=conditions_missing_baseline,
                w_inter_values=w_inter_values,
                ring_params_base=ring_params_base,
                base_params=base_params,
                n_baseline=baseline_n_trials_target,
                noise_percentile=noise_percentile,
                out_dir=out_dir,
                n_workers=n_workers,
                batch_chunk_size=batch_chunk_size,
                seed=args.seed,
                delay_ms=args.delay_ms,
                record_dt_ms=record_dt_ms,
                w_inter_values_by_condition=condition_missing_w,
                trials_to_add_by_key=trials_to_add_by_key,
                trial_start_idx_by_key=trial_start_idx_by_key,
                preserve_existing_cache=True,
            )
            noise_thresholds.update(new_nt)
            baseline_A_hat_data.update(new_base)
        else:
            print("  All noise floor baselines cached — skipping noise floor simulation")

        # Report cached / loaded baselines
        for ck in conditions_to_run:
            cond_label = STUDY_CONDITIONS[ck].name
            for w in w_inter_values:
                key = (ck, w)
                nt = noise_thresholds.get(key, float('nan'))
                n_samples = len(baseline_A_hat_data.get(key, []))
                print(f"  {cond_label}, w={w:.2f}: threshold = {nt:.4f} "
                      f"(p{noise_percentile:.0f}, n={n_samples}) [baseline cached]")

        # --- Phase 2: Grid exploration ---
        print("\n--- Phase 2: Grid exploration ---")
        n_grid_groups = len(conditions_to_run) * len(amplitudes) * len(w_inter_values)
        print(f"  {n_trials} trials × {n_grid_groups} groups = {n_trials * n_grid_groups} total grid sims")

        grid_seeds = trial_seeds[:n_trials]
        stim_onset_rel = STIM_ONSET_MS - BURN_IN_MS

        def _grid_group(ck, amp, w):
            condition = STUDY_CONDITIONS[ck]
            local_params = apply_condition(base_params, condition)
            actual_current = amp * base_params.I_ext_pyr()
            stimuli_short = [
                RingStimulus(
                    center_deg=STIM_CENTER_DEG, amplitude=actual_current,
                    sigma_deg=STIM_SIGMA_DEG,
                    onset_ms=stim_onset_rel, duration_ms=STIM_DURATION_MS,
                ),
            ]
            rp = replace(ring_params_base, w_pyr_pyr_inter=w)
            conn = connectivity_cache[w]
            r0, I_adapt0 = burnin_cache[(ck, w)]
            group = []
            for chunk_start in range(0, n_trials, batch_chunk_size):
                chunk_end = min(chunk_start + batch_chunk_size, n_trials)
                chunk_seeds = grid_seeds[chunk_start:chunk_end]
                chunk_n = len(chunk_seeds)
                batch_results = simulate_ring_batch(
                    [local_params] * chunk_n, rp, T_ms=T_ms_short,
                    stimuli=stimuli_short,
                    r0=r0, I_adapt0=I_adapt0,
                    seeds=list(chunk_seeds),
                    noise_type='white',
                    connectivity=conn,
                    record_dt_ms=record_dt_ms,
                )
                for ti_chunk, res in enumerate(batch_results):
                    ti = chunk_start + ti_chunk
                    res.t_ms += BURN_IN_MS
                    group.append(_compute_calibrate_metrics(
                        res, ck, amp, w, ti, int(grid_seeds[ti]),
                        eval_times_ms, args.delay_ms,
                    ))
            return group

        grid_groups = [
            (ck, amp, w)
            for ck in conditions_to_run
            for amp in amplitudes
            for w in w_inter_values
        ]
        gen = Parallel(n_jobs=n_workers, backend='loky', return_as='generator')(
            delayed(_grid_group)(ck, amp, w) for ck, amp, w in grid_groups
        )
        for group in tqdm(gen, total=len(grid_groups),
                          desc=f"Grid (n={n_trials},chunk={batch_chunk_size})", unit="group"):
            grid_results.extend(group)

    # --- Load cached conditions ---
    if cached_conditions:
        print(f"\nLoading cached results for: {', '.join(sorted(cached_conditions))}")
    for ck in cached_conditions:
        cond_dir_load = os.path.join(out_dir, ck)
        cached_rows = _load_calibrate_grid_results(cond_dir_load, ck)
        grid_results.extend(cached_rows)
        cached_nt, cached_base, _ = _load_calibrate_baseline(
            cond_dir_load, ck, w_inter_values, noise_percentile)
        noise_thresholds.update(cached_nt)
        baseline_A_hat_data.update(cached_base)
        cond_label = STUDY_CONDITIONS[ck].name
        for w in w_inter_values:
            key = (ck, w)
            nt = noise_thresholds.get(key, float('nan'))
            n_samples = len(baseline_A_hat_data.get(key, []))
            src = "full dist" if n_samples > 0 else "summary CSV"
            print(f"  {cond_label}, w={w:.2f}: threshold = {nt:.4f} "
                  f"(p{noise_percentile:.0f}, {src}) [cached]")

    # --- Aggregate per condition ---
    error_band = getattr(args, 'error_band', 'sem')

    # Collected across conditions for cross-condition summary plots
    all_cond_noise_data: dict[str, dict] = {}

    for ck in condition_keys:
        cond_label = STUDY_CONDITIONS[ck].name
        print(f"\n=== Results for {cond_label} ===")

        # Per-condition output subdirectory
        cond_dir = os.path.join(out_dir, ck)
        os.makedirs(cond_dir, exist_ok=True)

        grid_data: dict[tuple[float, float], dict] = {}
        timecourse_data: dict[tuple[float, float], dict] = {}

        for amp in amplitudes:
            for w in w_inter_values:
                trials = [r for r in grid_results
                          if r['cond_key'] == ck and r['amplitude'] == amp
                          and r['w_inter'] == w]
                if not trials:
                    continue

                threshold = noise_thresholds.get((ck, w), 0.0)
                A_hat_finals = np.array([r['A_hat_final'] for r in trials])
                success_rate = float(np.mean(A_hat_finals > threshold))
                mean_A_hat = float(np.mean(A_hat_finals))
                peak_rates = np.array([r['peak_pyr_rate'] for r in trials])
                errors = np.array([r['error_from_cue_deg'] for r in trials])

                grid_data[(amp, w)] = {
                    'success_rate': success_rate,
                    'mean_A_hat': mean_A_hat,
                    'peak_pyr_rate': float(np.mean(peak_rates)),
                    'mean_error_deg': float(np.nanmean(errors)),
                    'n_trials': len(trials),
                }

                # Timecourse data is only available when a_hat_timecourse is in the CSV
                n_t = len(trials)
                if trials[0]['A_hat_timecourse']:  # non-empty: new CSV format
                    tc_array = np.array([r['A_hat_timecourse'] for r in trials])
                    timecourse_data[(amp, w)] = {
                        'A_hat_mean': np.mean(tc_array, axis=0),
                        'A_hat_sem': np.std(tc_array, axis=0, ddof=1) / np.sqrt(n_t)
                        if n_t > 1 else np.zeros(tc_array.shape[1]),
                        'A_hat_sd': np.std(tc_array, axis=0, ddof=1)
                        if n_t > 1 else np.zeros(tc_array.shape[1]),
                        'success_rate': grid_data[(amp, w)]['success_rate'],
                    }

        # Collect for cross-condition summary
        all_cond_noise_data[ck] = {w: noise_thresholds[(ck, w)]
                                   for w in w_inter_values if (ck, w) in noise_thresholds}

        # --- Save CSVs (in per-condition subdir; skip for cached conditions) ---
        if ck not in cached_conditions:
            trial_csv = os.path.join(cond_dir, "calibration_results.csv")
            with open(trial_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'condition_key', 'amplitude', 'w_inter', 'trial_idx', 'seed',
                    'A_hat_final', 'a_hat_timecourse',
                    'peak_pyr_rate', 'center_final_deg', 'error_from_cue_deg',
                ])
                writer.writeheader()
                for r in grid_results:
                    if r['cond_key'] != ck:
                        continue
                    writer.writerow({
                        'condition_key': r['cond_key'],
                        'amplitude': r['amplitude'],
                        'w_inter': r['w_inter'],
                        'trial_idx': r['trial_idx'],
                        'seed': r['seed'],
                        'A_hat_final': r['A_hat_final'],
                        'a_hat_timecourse': ' '.join(f'{v:.6f}' for v in r['A_hat_timecourse']),
                        'peak_pyr_rate': r['peak_pyr_rate'],
                        'center_final_deg': r['center_final_deg'],
                        'error_from_cue_deg': r['error_from_cue_deg'],
                    })

        summary_csv = os.path.join(cond_dir, "calibration_summary.csv")
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'amplitude', 'w_inter', 'success_rate',
                'mean_A_hat', 'peak_pyr_rate', 'mean_error_deg',
                'noise_threshold', 'n_trials',
            ])
            writer.writeheader()
            for (amp, w), d in sorted(grid_data.items()):
                writer.writerow({
                    'condition_key': ck,
                    'amplitude': amp,
                    'w_inter': w,
                    'success_rate': d['success_rate'],
                    'mean_A_hat': d['mean_A_hat'],
                    'peak_pyr_rate': d['peak_pyr_rate'],
                    'mean_error_deg': d['mean_error_deg'],
                    'noise_threshold': noise_thresholds.get((ck, w), 0.0),
                    'n_trials': d['n_trials'],
                })

        # --- Per-condition plots (in cond_dir) ---
        saturated_w_cond = [w for w in w_inter_values if (ck, w) not in noise_thresholds]
        baseline_for_plot = {w: baseline_A_hat_data.get((ck, w), np.array([]))
                             for w in w_inter_values if (ck, w) in noise_thresholds}
        thresholds_for_plot = {w: noise_thresholds[(ck, w)]
                               for w in w_inter_values if (ck, w) in noise_thresholds}
        # Noise floor histogram requires the full A_hat distribution (not available from old CSVs)
        if any(len(v) > 0 for v in baseline_for_plot.values()):
            baseline_n_samples = int(sum(len(v) for v in baseline_for_plot.values()))
            plot_noise_floor_histogram(
                baseline_for_plot, thresholds_for_plot,
                save_path=_unique_path(os.path.join(cond_dir, "noise_floor.png")),
                suptitle=f"Noise Floor ({cond_label}, n={baseline_n_samples} samples, p{noise_percentile:.0f})",
                skipped_w_values=saturated_w_cond if saturated_w_cond else None,
            )
            plt.close()
        else:
            print(f"  Skipping noise floor histogram for {cond_label} (re-run to generate)")

        plot_calibration_heatmap(
            grid_data, "success_rate", amplitudes, w_inter_values,
            cmap="RdYlGn", vmin=0, vmax=1,
            save_path=_unique_path(os.path.join(cond_dir, "heatmap_success_rate.png")),
            suptitle=f"Success Rate ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        plot_calibration_heatmap(
            grid_data, "mean_A_hat", amplitudes, w_inter_values,
            cmap="viridis", vmin=0, vmax=1,
            save_path=_unique_path(os.path.join(cond_dir, "heatmap_A_hat.png")),
            suptitle=f"Mean A_hat ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        plot_calibration_heatmap(
            grid_data, "peak_pyr_rate", amplitudes, w_inter_values,
            cmap="hot",
            save_path=_unique_path(os.path.join(cond_dir, "heatmap_peak_pyr.png")),
            suptitle=f"Peak PYR Rate ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        if timecourse_data:
            tc_keys = sorted(
                k for k in timecourse_data
                if grid_data.get(k, {}).get('success_rate', 0.0) >= 0.9
            )
            tc_subset = {k: timecourse_data[k] for k in tc_keys}
            band_tag = f"+/-{error_band.upper()}" if n_trials > 1 else ""
            plot_calibration_timecourses(
                tc_subset, eval_times_s, error_band=error_band,
                save_path=_unique_path(os.path.join(cond_dir, f"timecourses_{error_band}.png")),
                suptitle=f"A_hat Time Courses — success ≥ 90% ({cond_label}, {n_trials} trials, {band_tag})",
            )
            plt.close()
        else:
            print(f"  Skipping timecourse plot for {cond_label} (re-run to generate)")

    # --- Cross-condition summary plots (in parent out_dir) ---
    n_cond_label = f"{len(condition_keys)} condition{'s' if len(condition_keys) > 1 else ''}"
    plot_noise_summary(
        all_cond_noise_data,
        save_path=_unique_path(os.path.join(out_dir, "noise_summary.png")),
        suptitle=f"Noise Floor ({n_cond_label}, p{noise_percentile:.0f})",
    )
    plt.close()


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
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params_base = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
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

        for ck in condition_keys:
            cond_label = STUDY_CONDITIONS[ck].name
            cond_dir = os.path.join(out_dir, ck)

            cached_nt, cached_base, saturated_w = _load_calibrate_baseline(
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

        if all_cond_noise_data:
            n_cond_label = f"{len(all_cond_noise_data)} condition{'s' if len(all_cond_noise_data) > 1 else ''}"
            plot_noise_summary(
                all_cond_noise_data,
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
    trials_to_add_by_key: dict[tuple[str, float], int] = {}
    trial_start_idx_by_key: dict[tuple[str, float], int] = {}
    legacy_cache_conditions: list[str] = []

    if not no_cache:
        for ck in condition_keys:
            cond_dir_check = os.path.join(out_dir, ck)
            cached_nt, cached_base, _ = _load_calibrate_baseline(
                cond_dir_check, ck, w_inter_values, noise_percentile,
            )
            cached_noise_thresholds.update(cached_nt)
            cached_baseline_data.update(cached_base)

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
    print(f"  Workers: {n_workers}, batch chunk size: {batch_chunk_size}")
    if conditions_to_run:
        total_sims = sum(trials_to_add_by_key.get((ck, w), 0)
                         for ck in condition_keys for w in w_inter_values)
        print(f"  Total simulations: {total_sims}")

    # Containers
    baseline_A_hat_data: dict[tuple[str, float], np.ndarray] = dict(cached_baseline_data)
    noise_thresholds: dict[tuple[str, float], float] = dict(cached_noise_thresholds)

    if conditions_to_run:
        new_nt, new_base = _run_noise_floor_for_conditions(
            conditions_to_run=conditions_to_run,
            w_inter_values=w_inter_values,
            ring_params_base=ring_params_base,
            base_params=base_params,
            n_baseline=n_baseline,
            noise_percentile=noise_percentile,
            out_dir=out_dir,
            n_workers=n_workers,
            batch_chunk_size=batch_chunk_size,
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

    n_cond_label = f"{len(condition_keys)} condition{'s' if len(condition_keys) > 1 else ''}"
    plot_noise_summary(
        all_cond_noise_data,
        save_path=os.path.join(out_dir, "noise_summary.png"),
        suptitle=f"Noise Floor ({n_cond_label}, {n_baseline} baseline trials, p{noise_percentile:.0f})",
    )
    plt.close()
    print(f"\nNoise floor estimation complete.")
    print(f"Results saved to: {out_dir}")


# ============================================================================
# DISTRACTOR SWEEP: HELPERS
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


# ============================================================================
# DISTRACTOR SWEEP: PARALLEL WORKER
# ============================================================================

_distractor_sweep_sim_args: Optional[dict] = None


def _distractor_sweep_init_worker(
    local_params: "CircuitParams",
    ring_params: "RingParams",
    connectivity: "RingConnectivity",
    burnin_state: tuple,
    T_ms_short: float,
    cue_amp_factor: float,
    delay1_ms: float,
    distractor_duration_ms: float,
    delay2_ms: float,
    collapse_threshold: float,
    record_dt_ms: float,
):
    """Initialize worker process for the 2-D distractor sweep."""
    global _distractor_sweep_sim_args
    _distractor_sweep_sim_args = {
        'local_params': local_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
        'burnin_state': burnin_state,
        'T_ms_short': T_ms_short,
        'cue_amp_factor': cue_amp_factor,
        'delay1_ms': delay1_ms,
        'distractor_duration_ms': distractor_duration_ms,
        'delay2_ms': delay2_ms,
        'collapse_threshold': collapse_threshold,
        'record_dt_ms': record_dt_ms,
    }


def _distractor_sweep_run_single(job: tuple) -> dict:
    """Run one (offset_deg, amp_factor, trial_idx, seed) trial.

    Returns pre/post bump position, displacement, and amplitudes.
    """
    global _distractor_sweep_sim_args
    cfg = _distractor_sweep_sim_args
    offset_deg, amp_factor, trial_idx, seed = job

    local_params = cfg['local_params']
    ring_params = cfg['ring_params']
    connectivity = cfg['connectivity']
    r0, I_adapt0 = cfg['burnin_state']
    T_ms_short = cfg['T_ms_short']
    cue_amp_factor = cfg['cue_amp_factor']
    delay1_ms = cfg['delay1_ms']
    distractor_duration_ms = cfg['distractor_duration_ms']
    record_dt_ms = cfg['record_dt_ms']

    cue_current = cue_amp_factor * local_params.I_ext_pyr()
    distractor_current = amp_factor * cue_current

    # Timing relative to post-burn-in simulation start
    cue_onset_rel = STIM_ONSET_MS - BURN_IN_MS
    dist_onset_rel = cue_onset_rel + STIM_DURATION_MS + delay1_ms

    distractor_location_deg = (STIM_CENTER_DEG + offset_deg) % 360.0

    cue_stim = RingStimulus(
        center_deg=STIM_CENTER_DEG,
        amplitude=cue_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=cue_onset_rel,
        duration_ms=STIM_DURATION_MS,
    )
    dist_stim = RingStimulus(
        center_deg=distractor_location_deg,
        amplitude=distractor_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=dist_onset_rel,
        duration_ms=distractor_duration_ms,
    )

    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=[cue_stim, dist_stim],
        r0=r0, I_adapt0=I_adapt0,
        seed=seed, connectivity=connectivity,
        record_dt_ms=record_dt_ms,
    )

    # Shift t_ms to absolute time (add burn-in back)
    result.t_ms += BURN_IN_MS

    # Absolute time of distractor onset/offset
    dist_onset_abs = STIM_ONSET_MS + STIM_DURATION_MS + delay1_ms
    dist_offset_abs = dist_onset_abs + distractor_duration_ms

    # Measurement windows: 50 ms before onset, transient-skip time after offset
    pre_t = dist_onset_abs - 50.0
    post_t = dist_offset_abs + TRANSIENT_SKIP_TIME_MS

    pre_idx = int(np.argmin(np.abs(result.t_ms - pre_t)))
    post_idx = int(np.argmin(np.abs(result.t_ms - post_t)))

    pre_center_rad, pre_amp = population_vector_decode(
        result.r[pre_idx, :, 0], ring_params.node_angles_rad,
    )
    post_center_rad, post_amp = population_vector_decode(
        result.r[post_idx, :, 0], ring_params.node_angles_rad,
    )

    # Signed angular displacement (positive = toward distractor)
    raw_disp = post_center_rad - pre_center_rad
    displacement_rad = (raw_disp + np.pi) % (2 * np.pi) - np.pi
    displacement_deg = float(np.degrees(displacement_rad))

    return {
        'offset_deg': float(offset_deg),
        'amp_factor': float(amp_factor),
        'trial_idx': int(trial_idx),
        'displacement_deg': displacement_deg,
        'pre_amp': float(pre_amp),
        'post_amp': float(post_amp),
    }


# ============================================================================
# DISTRACTOR SWEEP: MAIN COMMAND
# ============================================================================

def cmd_distractor_sweep(args: argparse.Namespace) -> None:
    """Run 2-D distractor parameter sweep (Δφ × distractor amplitude)."""
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Parameters ---
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    cond_key = args.condition
    if cond_key not in STUDY_CONDITIONS and cond_key != ["all"]:
        print(f"Error: unknown condition '{cond_key}'.")
        import sys; sys.exit(1)
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    offsets_deg = sorted(args.offsets_deg)
    amp_factors = sorted(args.amp_factors)
    n_trials = args.n_trials
    delay1_ms = args.delay1_ms
    delay2_ms = args.delay2_ms
    distractor_duration_ms = args.distractor_duration_ms
    cue_amp_factor = args.amplitude

    # --- Collapse threshold: prefer calibration noise floor over hardcoded default ---
    if args.collapse_threshold is not None:
        collapse_threshold = args.collapse_threshold
        print(f"Collapse threshold: {collapse_threshold:.4f} (manual override)")
    else:
        import warnings
        cal_conn_label = _calibration_network_label(ring_params)
        cal_csv = os.path.join(
            _output_dir("figs/ring/calibration", args.params_json),
            cal_conn_label, "calibration_summary.csv",
        )
        threshold = _lookup_noise_threshold(
            cal_csv, cond_key, cue_amp_factor, ring_params.w_pyr_pyr_inter,
        )
        if threshold is not None:
            collapse_threshold = threshold
            print(f"Collapse threshold: {collapse_threshold:.4f} "
                  f"(calibration noise floor from {cal_csv})")
        else:
            collapse_threshold = 0.2
            warnings.warn(
                f"No calibration noise threshold found for condition='{cond_key}', "
                f"amplitude={cue_amp_factor}, w_inter={ring_params.w_pyr_pyr_inter} "
                f"in {cal_csv}. "
                f"Falling back to 0.2. Run ring-calibrate first to set an "
                f"empirical noise floor.",
                stacklevel=2,
            )
    record_dt_ms = getattr(args, 'record_dt_ms', 5.0)
    n_workers = _resolve_workers(args)

    # Total simulation time (post burn-in)
    # cue_onset_rel + cue_dur + delay1 + dist_dur + delay2 + 200 ms buffer
    cue_onset_rel = STIM_ONSET_MS - BURN_IN_MS
    T_ms_short = (cue_onset_rel + STIM_DURATION_MS
                  + delay1_ms + distractor_duration_ms + delay2_ms + 200.0)

    # Absolute timing for measurements and figures
    dist_onset_abs = STIM_ONSET_MS + STIM_DURATION_MS + delay1_ms
    dist_offset_abs = dist_onset_abs + distractor_duration_ms
    cue_onset_abs = STIM_ONSET_MS
    cue_offset_abs = STIM_ONSET_MS + STIM_DURATION_MS

    # Output directory
    conn_label = _network_label(ring_params)
    amp_label = f"amp{_fmt(cue_amp_factor)}"
    out_dir = os.path.join(
        _output_dir("figs/ring/distractor_sweep", args.params_json),
        conn_label,
        cond_key,
        amp_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDistractor 2-D sweep:")
    print(f"  Condition: {condition.name}")
    print(f"  Δφ offsets: {offsets_deg}")
    print(f"  Amplitude factors: {amp_factors}")
    print(f"  Protocol: cue {STIM_DURATION_MS:.0f}ms → delay1 {delay1_ms:.0f}ms "
          f"→ distractor {distractor_duration_ms:.0f}ms → delay2 {delay2_ms:.0f}ms")
    print(f"  Trials per cell: {n_trials}")
    total = len(offsets_deg) * len(amp_factors) * n_trials
    print(f"  Total simulations: {total}")

    # --- Burn-in ---
    connectivity = RingConnectivity.from_params(ring_params)
    print("\nComputing burn-in state...")
    burnin_state = _compute_burnin_state(
        local_params, ring_params, connectivity, seed=args.seed,
    )

    # --- Trial seeds ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

    # Shared worker init args (used for both full simulations and
    # representative-cell timecourse reruns)
    init_args = (
        local_params, ring_params, connectivity, burnin_state,
        T_ms_short, cue_amp_factor, delay1_ms,
        distractor_duration_ms, delay2_ms,
        collapse_threshold, record_dt_ms,
    )

    # --- Output paths (defined early for cache check) ---
    raw_csv = os.path.join(out_dir, "distractor_sweep_trials.csv")
    summary_csv = os.path.join(out_dir, "distractor_sweep_summary.csv")

    # --- Check for cached trial data ---
    from collections import defaultdict
    all_results: list[dict] = []
    loaded_from_cache = False

    if os.path.exists(raw_csv) and os.path.exists(summary_csv):
        try:
            with open(summary_csv, newline='') as _f:
                summary_rows = list(csv.DictReader(_f))

            params_ok = summary_rows and all(
                r.get('condition_key') == cond_key
                and float(r.get('distractor_duration_ms', 'nan')) == distractor_duration_ms
                and float(r.get('delay1_ms', 'nan')) == delay1_ms
                and float(r.get('delay2_ms', 'nan')) == delay2_ms
                and float(r.get('cue_amp_factor', 'nan')) == cue_amp_factor
                for r in summary_rows
            )

            if params_ok:
                with open(raw_csv, newline='') as _f:
                    trial_rows = list(csv.DictReader(_f))

                offsets_set = {float(o) for o in offsets_deg}
                amps_set = {float(a) for a in amp_factors}
                counts: dict[tuple, int] = defaultdict(int)
                for row in trial_rows:
                    counts[(float(row['offset_deg']), float(row['amp_factor']))] += 1

                cache_valid = all(
                    counts[(off, amp)] >= n_trials
                    for off in offsets_set
                    for amp in amps_set
                )

                if cache_valid:
                    print(f"\nLoading cached trial data from {raw_csv}")
                    seen: dict[tuple, int] = defaultdict(int)
                    for row in trial_rows:
                        off = float(row['offset_deg'])
                        amp = float(row['amp_factor'])
                        if off not in offsets_set or amp not in amps_set:
                            continue
                        if seen[(off, amp)] >= n_trials:
                            continue
                        all_results.append({
                            'offset_deg': off,
                            'amp_factor': amp,
                            'trial_idx': int(row['trial_idx']),
                            'displacement_deg': float(row['displacement_deg']),
                            'pre_amp': float(row['pre_amp']),
                            'post_amp': float(row['post_amp']),
                        })
                        seen[(off, amp)] += 1
                    loaded_from_cache = True
                    print(f"  Loaded {len(all_results)} cached trials.")
        except Exception as _e:
            print(f"  Cache read failed ({_e}), rerunning simulations.")

    if not loaded_from_cache and n_trials > 1:
        # --- Build and execute jobs (multi-trial sweep) ---
        jobs = [
            (off, amp, trial_idx, trial_seeds[trial_idx])
            for off in offsets_deg
            for amp in amp_factors
            for trial_idx in range(n_trials)
        ]

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_distractor_sweep_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_distractor_sweep_run_single, job): job
                           for job in jobs}
                with tqdm(total=len(jobs), desc="Distractor sweep trials", unit="trial", smoothing=0) as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _distractor_sweep_init_worker(*init_args)
            for job in tqdm(jobs, desc="Distractor sweep trials", unit="trial"):
                all_results.append(_distractor_sweep_run_single(job))

    # --- Figure 3: Timecourses — 90° offset at low / mid / high amplitude ---
    tc_offset = 90.0
    tc_amps = [amp_factors[0], amp_factors[len(amp_factors) // 2], amp_factors[-1]]
    # Fall back to closest available offset if 90° not in the sweep
    if tc_offset not in offsets_deg:
        tc_offset = min(offsets_deg, key=lambda o: abs(o - 90.0))
    selected_cells = [(tc_offset, a) for a in tc_amps if a in amp_factors]

    # --- Figures 4+: Activity grid and node timecourses — all offsets × all amplitudes ---
    activity_cells = [(off, amp) for amp in amp_factors for off in offsets_deg]

    # Union of all cells needed (avoid duplicate simulations)
    all_cells_needed = list(dict.fromkeys(selected_cells + activity_cells))

    print("\nRunning detailed timecourse simulations...")
    _distractor_sweep_init_worker(*init_args)  # Re-init in main process
    tc_map: dict = {}
    for (off, amp) in tqdm(all_cells_needed, desc="Timecourse runs"):
        # Run full simulation with fine recording to get the r array
        cue_current = cue_amp_factor * base_params.I_ext_pyr()
        distractor_current = amp * cue_current
        cue_onset_rel = STIM_ONSET_MS - BURN_IN_MS
        dist_onset_rel = cue_onset_rel + STIM_DURATION_MS + delay1_ms
        distractor_location_deg = (STIM_CENTER_DEG + off) % 360.0

        cue_stim = RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=cue_current,
            sigma_deg=STIM_SIGMA_DEG, onset_ms=cue_onset_rel,
            duration_ms=STIM_DURATION_MS,
        )
        dist_stim = RingStimulus(
            center_deg=distractor_location_deg, amplitude=distractor_current,
            sigma_deg=STIM_SIGMA_DEG, onset_ms=dist_onset_rel,
            duration_ms=distractor_duration_ms,
        )
        full_result = simulate_ring(
            local_params, ring_params, T_ms=T_ms_short,
            stimuli=[cue_stim, dist_stim],
            r0=burnin_state[0], I_adapt0=burnin_state[1],
            seed=trial_seeds[0], connectivity=connectivity,
            record_dt_ms=record_dt_ms,
        )
        full_result.t_ms += BURN_IN_MS

        # Decode bump position at every timestep (PYR population)
        z = np.exp(1j * ring_params.node_angles_rad)
        pyr_activity = full_result.r[:, :, 0]  # (n_steps, n_nodes)
        total_act = pyr_activity.sum(axis=1) + 1e-12
        weighted_z = pyr_activity @ z
        norm_z = weighted_z / total_act
        center_rad = np.angle(norm_z)
        center_deg_arr = np.degrees(center_rad) % 360.0
        amplitude_arr = np.abs(norm_z)

        tc_map[(off, amp)] = {
            'offset_deg': off,
            'amp_factor': amp,
            't_ms': full_result.t_ms,
            'center_deg': center_deg_arr,
            'amplitude': amplitude_arr,
            'full_result': full_result,
        }

        # When n_trials==1 and not from cache, derive summary stats here
        # (avoids running a separate sweep pass)
        if n_trials == 1 and not loaded_from_cache:
            pre_t = dist_onset_abs - 50.0
            post_t = dist_onset_abs + distractor_duration_ms + TRANSIENT_SKIP_TIME_MS
            pre_idx = int(np.argmin(np.abs(full_result.t_ms - pre_t)))
            post_idx = int(np.argmin(np.abs(full_result.t_ms - post_t)))
            pre_center_rad, pre_amp_val = population_vector_decode(
                full_result.r[pre_idx, :, 0], ring_params.node_angles_rad)
            post_center_rad, post_amp_val = population_vector_decode(
                full_result.r[post_idx, :, 0], ring_params.node_angles_rad)
            raw_disp = post_center_rad - pre_center_rad
            displacement_rad = (raw_disp + np.pi) % (2 * np.pi) - np.pi
            all_results.append({
                'offset_deg': float(off),
                'amp_factor': float(amp),
                'trial_idx': 0,
                'displacement_deg': float(np.degrees(displacement_rad)),
                'pre_amp': float(pre_amp_val),
                'post_amp': float(post_amp_val),
            })

    # --- Aggregate per (offset_deg, amp_factor) cell ---
    grid_summary: dict[tuple, dict] = {}
    for off in offsets_deg:
        for amp in amp_factors:
            trials = [r for r in all_results
                      if r['offset_deg'] == off and r['amp_factor'] == amp]
            displacements = np.array([r['displacement_deg'] for r in trials])
            post_amps = np.array([r['post_amp'] for r in trials])
            n = len(displacements)
            drift_mean = float(np.mean(displacements))
            drift_sd = float(np.std(displacements, ddof=1)) if n > 1 else 0.0
            drift_sem = drift_sd / np.sqrt(n) if n > 1 else 0.0
            collapse_prob = float(np.mean(post_amps < collapse_threshold))
            grid_summary[(off, amp)] = {
                'drift_mean_deg': drift_mean,
                'drift_sd_deg': drift_sd,
                'drift_sem_deg': drift_sem,
                'collapse_prob': collapse_prob,
                'pre_amp_mean': float(np.mean([r['pre_amp'] for r in trials])),
                'post_amp_mean': float(np.mean(post_amps)),
                'n_trials': n,
            }

    # --- Save CSVs (skipped when loaded from cache) ---
    if not loaded_from_cache:
        with open(raw_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'offset_deg', 'amp_factor', 'trial_idx',
                'displacement_deg', 'pre_amp', 'post_amp',
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'offset_deg', 'amp_factor', 'n_trials',
                'drift_mean_deg', 'drift_sd_deg', 'drift_sem_deg',
                'collapse_prob', 'pre_amp_mean', 'post_amp_mean',
                'distractor_duration_ms', 'delay1_ms', 'delay2_ms', 'cue_amp_factor',
            ])
            writer.writeheader()
            for (off, amp), d in sorted(grid_summary.items()):
                writer.writerow({
                    'condition_key': cond_key,
                    'offset_deg': off,
                    'amp_factor': amp,
                    **d,
                    'distractor_duration_ms': distractor_duration_ms,
                    'delay1_ms': delay1_ms,
                    'delay2_ms': delay2_ms,
                    'cue_amp_factor': cue_amp_factor,
                })

        print(f"\nCSVs saved to {out_dir}/")

    # --- Figure 1 & 2: Heatmaps (only meaningful with multiple trials) ---
    if n_trials > 1:
        fig1, fig2 = plot_distractor_sweep_heatmaps(
            grid_summary, offsets_deg, amp_factors,
            collapse_threshold=collapse_threshold,
            save_dir=out_dir,
            suptitle_prefix=f"{condition.name} — ",
        )
        plt.close(fig1)
        plt.close(fig2)
        print(f"  distractor_sweep_drift.png")
        print(f"  distractor_sweep_collapse.png")

    tc_data = [tc_map[cell] for cell in selected_cells]

    fig3 = plot_distractor_sweep_timecourses(
        tc_data,
        cue_onset_ms=cue_onset_abs,
        cue_offset_ms=cue_offset_abs,
        dist_onset_ms=dist_onset_abs,
        dist_offset_ms=dist_offset_abs,
        save_path=os.path.join(out_dir, "distractor_sweep_timecourses.png"),
        suptitle=f"{condition.name} — Bump Trajectories",
    )
    plt.close(fig3)
    print(f"  distractor_sweep_timecourses.png")

    # Group by amplitude: one figure per amplitude level
    from collections import defaultdict
    amp_groups: dict = defaultdict(list)
    for (off, amp) in activity_cells:
        if (off, amp) in tc_map:
            amp_groups[amp].append(tc_map[(off, amp)])

    print("Saving per-amplitude activity grids and node timecourses...")
    # Subfolders for organised output
    dir_grid    = os.path.join(out_dir, "activity_grid")
    dir_raw     = os.path.join(out_dir, "node_activity")
    dir_cd      = os.path.join(out_dir, "node_diff_cue_dist")
    dir_de      = os.path.join(out_dir, "node_diff_dist_end")
    dir_ce      = os.path.join(out_dir, "node_diff_cue_end")
    for d in (dir_grid, dir_raw, dir_cd, dir_de, dir_ce):
        os.makedirs(d, exist_ok=True)

    for amp in sorted(amp_groups.keys()):
        amp_str = _fmt(amp)
        group_data = amp_groups[amp]

        fig4 = plot_distractor_sweep_activity_grid(
            group_data,
            cue_onset_ms=cue_onset_abs,
            cue_offset_ms=cue_offset_abs,
            dist_onset_ms=dist_onset_abs,
            dist_offset_ms=dist_offset_abs,
            burn_in_ms=BURN_IN_MS,
            save_path=os.path.join(dir_grid, f"activity_grid_amp{amp_str}.png"),
            suptitle=f"{condition.name} — PYR Activity (distractor {amp:.2g}× cue)",
        )
        plt.close(fig4)
        print(f"  activity_grid/activity_grid_amp{amp_str}.png")

        fig_raw = plot_distractor_sweep_node_timecourses(
            group_data,
            cue_onset_ms=cue_onset_abs,
            cue_offset_ms=cue_offset_abs,
            dist_onset_ms=dist_onset_abs,
            dist_offset_ms=dist_offset_abs,
            burn_in_ms=BURN_IN_MS,
            cue_center_deg=STIM_CENTER_DEG,
            save_path=os.path.join(dir_raw, f"node_activity_amp{amp_str}.png"),
            suptitle=f"{condition.name} — Node Activity (distractor {amp:.2g}× cue)",
        )
        plt.close(fig_raw)
        print(f"  node_activity/node_activity_amp{amp_str}.png")

        diff_paths = (
            os.path.join(dir_cd, f"cue_dist_amp{amp_str}.png"),
            os.path.join(dir_de, f"dist_end_amp{amp_str}.png"),
            os.path.join(dir_ce, f"cue_end_amp{amp_str}.png"),
        )
        fig_cd_f, fig_de_f, fig_ce_f = plot_distractor_sweep_node_differences(
            group_data,
            cue_onset_ms=cue_onset_abs,
            cue_offset_ms=cue_offset_abs,
            dist_onset_ms=dist_onset_abs,
            dist_offset_ms=dist_offset_abs,
            burn_in_ms=BURN_IN_MS,
            cue_center_deg=STIM_CENTER_DEG,
            save_paths=diff_paths,
            suptitle_prefix=f"{condition.name} — ",
        )
        for fig5 in (fig_cd_f, fig_de_f, fig_ce_f):
            plt.close(fig5)
        print(f"  node_diff_cue_dist/cue_dist_amp{amp_str}.png")
        print(f"  node_diff_dist_end/dist_end_amp{amp_str}.png")
        print(f"  node_diff_cue_end/cue_end_amp{amp_str}.png")

    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
    amp_target = 1.0
    if amp_factors:
        amp_anim = min(amp_factors, key=lambda a: abs(a - amp_target))
        if abs(amp_anim - amp_target) > 1e-12:
            print(f"Requested amp×1 not in sweep; using closest available {amp_anim:.3g}×")
        anim_dir = os.path.join(out_dir, "snapshot_evolution_amp1")
        os.makedirs(anim_dir, exist_ok=True)
        print("Saving snapshot evolution animations for each offset at distractor amp×1...")
        mp4_jobs: list[tuple[float, dict]] = []
        for off in offsets_deg:
            entry = tc_map.get((off, amp_anim))
            if entry is not None:
                mp4_jobs.append((off, entry))

        mp4_pbar = _start_mp4_progress(
            total_videos=len(mp4_jobs),
            frame_step_ms=args.snapshot_anim_step_ms,
            fps=args.snapshot_anim_fps,
        )
        try:
            for off, entry in mp4_jobs:
                mp4_pbar.set_postfix_str(f"offset={_fmt(off)}deg")
                anim_path = os.path.join(anim_dir, f"offset_{_fmt(off)}deg.mp4")
                fig_anim, _ = animate_ring_snapshot_evolution(
                    entry['full_result'],
                    save_path=anim_path,
                    time_range=(BURN_IN_MS, entry['full_result'].t_ms[-1]),
                    t_offset=BURN_IN_MS,
                    frame_step_ms=args.snapshot_anim_step_ms,
                    fps=args.snapshot_anim_fps,
                    cue_window=(cue_onset_abs, cue_offset_abs),
                    cue_angle_deg=STIM_CENTER_DEG,
                    distractor_window=(dist_onset_abs, dist_offset_abs),
                    distractor_angle_deg=(STIM_CENTER_DEG + off) % 360.0,
                    suptitle=(
                        f"{condition.name} — Snapshot Evolution "
                        f"(Δφ={off:.0f}°, distractor {amp_anim:.2g}× cue)"
                    ),
                    show_asymmetry=True,
                    **anim_quality_kwargs,
                )
                plt.close(fig_anim)
                mp4_pbar.update(1)
        finally:
            mp4_pbar.close()

    print(f"\nAll outputs saved to {out_dir}/")


# ============================================================================
# LESION STUDY SUBCOMMAND
# ============================================================================

def _apply_knockdown(
    base_params: CircuitParams,
    ring_params: RingParams,
    population: str,
    scale: float,
) -> tuple:
    """Apply a gain knockdown to one population.

    Parameters:
        base_params: CircuitParams to potentially modify.
        ring_params: RingParams to potentially modify.
        population: 'PYR_recurrence' | 'PV' | 'SOM' | 'VIP'
        scale: 1.0 = no knockdown, 0.0 = complete knockdown.

    Returns:
        (modified_local_params, modified_ring_params, rebuild_connectivity)
        rebuild_connectivity is True when the weight matrix must be recomputed.
    """
    if population == 'PYR_recurrence':
        new_rp = replace(ring_params, w_pyr_pyr_inter=ring_params.w_pyr_pyr_inter * scale)
        return base_params, new_rp, True
    elif population == 'PV':
        new_rp = replace(ring_params, w_pv_global=ring_params.w_pv_global * scale)
        return base_params, new_rp, True
    elif population == 'SOM':
        new_bp = replace(base_params, w_se=base_params.w_se * scale)
        return new_bp, ring_params, False
    elif population == 'VIP':
        new_bp = replace(base_params, w_vs=base_params.w_vs * scale)
        return new_bp, ring_params, False
    else:
        raise ValueError(f"Unknown population: {population!r}. "
                         f"Valid: 'PYR_recurrence', 'PV', 'SOM', 'VIP'")


def cmd_lesion(args: argparse.Namespace) -> None:
    """Systematic lesion study sweeping knockdown of each population."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Build base configuration
    if args.params_json:
        base_params = load_params_json(args.params_json)
    else:
        base_params = CircuitParams()

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    amp_factor = args.amplitude
    actual_current = amp_factor * base_params.I_ext_pyr()
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    T_ms_full = delay_end_ms
    T_ms_short = T_ms_full - BURN_IN_MS

    noise_floor = args.noise_floor
    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/lesion", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "lesion_results.csv")

    populations = args.populations
    knockdown_levels = args.knockdown_levels
    n_trials = args.n_trials
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    n_workers = _resolve_workers(args)

    print(f"Lesion study: {populations}")
    print(f"  Knockdown levels: {knockdown_levels}%")
    print(f"  N trials: {n_trials}, workers: {n_workers}")

    # Pre-compute burn-in state once with WT condition (no knockdown)
    condition_wt = STUDY_CONDITIONS['WT']
    local_params_wt = apply_condition(base_params, condition_wt)
    connectivity_base = RingConnectivity.from_params(ring_params)

    print("\nComputing burn-in state (WT)...")
    r0_bi, I_adapt0_bi = _compute_burnin_state(local_params_wt, ring_params, connectivity_base,
                                                seed=args.seed)

    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    # Load cached results if they exist
    if getattr(args, 'no_cache', False) and os.path.exists(csv_path):
        os.remove(csv_path)

    # Build and run jobs
    print("\nRunning simulations...")

    # We collect (population, knockdown_pct, trial_idx, result_dict) entries
    # in a flat CSV for caching
    _LESION_FIELDS = [
        'population', 'knockdown_pct', 'trial_idx', 'seed',
        'formation_ok', 'survival_time_ms',
    ]

    # Load cache
    cached_rows = []
    completed_jobs = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                pop = row['population']
                kd = float(row['knockdown_pct'])
                ti = int(row['trial_idx'])
                if pop in populations and kd in knockdown_levels:
                    completed_jobs.add((pop, kd, ti))
                    cached_rows.append(row)

    # Build pending jobs
    jobs_to_run = []
    for pop in populations:
        for kd_pct in knockdown_levels:
            for trial_idx in range(n_trials):
                if (pop, float(kd_pct), trial_idx) not in completed_jobs:
                    seed = trial_seeds[trial_idx]
                    jobs_to_run.append((pop, float(kd_pct), trial_idx, seed))

    print(f"  {len(jobs_to_run)} to run, {len(completed_jobs)} cached")

    new_rows = []

    def _run_lesion_job(job):
        pop, kd_pct, trial_idx, seed = job
        scale = 1.0 - kd_pct / 100.0
        lp, rp, rebuild = _apply_knockdown(local_params_wt, ring_params, pop, scale)
        conn = RingConnectivity.from_params(rp) if rebuild else connectivity_base
        result = simulate_ring(
            lp, rp, T_ms=T_ms_short,
            stimuli=stimuli_short, r0=r0_bi, I_adapt0=I_adapt0_bi,
            seed=seed, connectivity=conn,
            record_dt_ms=getattr(args, 'record_dt_ms', 5.0),
        )
        result.t_ms += BURN_IN_MS
        result.stim_window = (STIM_ONSET_MS, stim_offset_ms)

        # Formation check
        check_end = stim_offset_ms + 300.0
        mask = (result.t_ms >= stim_offset_ms) & (result.t_ms <= check_end)
        if np.any(mask):
            act = result.r[mask, :, 0]
            _, amp_cue = population_vector_decode(act, result.ring_params.node_angles_rad)
            formation_ok = bool(np.any(amp_cue > noise_floor))
        else:
            formation_ok = False

        # Survival time
        st = compute_bump_survival_time(result, noise_floor)
        if st is None:
            st = float(result.t_ms[-1] - stim_offset_ms)

        return {
            'population': pop,
            'knockdown_pct': kd_pct,
            'trial_idx': trial_idx,
            'seed': seed,
            'formation_ok': int(formation_ok),
            'survival_time_ms': st,
        }

    if jobs_to_run:
        if n_workers > 1 and len(jobs_to_run) > 4:
            results_list = Parallel(n_jobs=n_workers, prefer='processes', verbose=0)(
                delayed(_run_lesion_job)(job)
                for job in tqdm(jobs_to_run, desc="Lesion jobs")
            )
        else:
            results_list = [
                _run_lesion_job(job)
                for job in tqdm(jobs_to_run, desc="Lesion jobs")
            ]
        new_rows = results_list

    # Save new rows to CSV
    if new_rows:
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_LESION_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)

    # Aggregate results
    all_rows = cached_rows + new_rows
    lesion_data = {}
    for pop in populations:
        lesion_data[pop] = {}
        for kd_pct in knockdown_levels:
            # Collect all trials for this (pop, kd_pct)
            matching = [
                r for r in all_rows
                if r['population'] == pop and float(r['knockdown_pct']) == float(kd_pct)
            ]
            formation_arr = np.array([int(r['formation_ok']) for r in matching])
            survival_arr = np.array([float(r['survival_time_ms']) for r in matching])
            formed_mask = formation_arr.astype(bool)
            n_formed = int(np.sum(formed_mask))
            n_total = len(matching)

            if n_formed > 0:
                sv = survival_arr[formed_mask]
                mean_sv = float(np.mean(sv))
                sem_sv = float(np.std(sv, ddof=1) / np.sqrt(n_formed)) if n_formed > 1 else 0.0
            else:
                mean_sv = np.nan
                sem_sv = np.nan

            lesion_data[pop][kd_pct] = {
                'formation_rate': n_formed / n_total if n_total > 0 else 0.0,
                'survival_time_mean_ms': mean_sv,
                'survival_time_sem_ms': sem_sv,
                'n_formed': n_formed,
                'n_trials': n_total,
            }

    # Plot
    fig = plot_lesion_study(
        lesion_data, noise_floor,
        save_path=os.path.join(out_dir, "lesion_figure.png"),
        suptitle=f"Population Lesion Study  (amp={_fmt(amp_factor)}×, N={n_trials})",
    )
    plt.close(fig)
    print(f"\nFigures saved to {out_dir}/")


# ============================================================================
# TAU-ADAPT SWEEP SUBCOMMAND
# ============================================================================

def cmd_tau_sweep(args: argparse.Namespace) -> None:
    """Sweep tau_adapt_pyr values and measure bump survival, diffusion, oscillation."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    if args.params_json:
        base_params = load_params_json(args.params_json)
    else:
        base_params = CircuitParams()

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    amp_factor = args.amplitude
    actual_current = amp_factor * base_params.I_ext_pyr()
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    T_ms_full = delay_end_ms
    T_ms_short = T_ms_full - BURN_IN_MS

    noise_floor = args.noise_floor
    tau_values = args.tau_values
    n_trials = args.n_trials
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)
    n_workers = _resolve_workers(args)

    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/tau_sweep", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tau_sweep_results.csv")

    # Pre-compute burn-in state (WT)
    connectivity = RingConnectivity.from_params(ring_params)
    condition_wt = STUDY_CONDITIONS['WT']
    local_params_wt = apply_condition(base_params, condition_wt)

    print("\nComputing burn-in state (WT)...")
    r0_bi, I_adapt0_bi = _compute_burnin_state(local_params_wt, ring_params, connectivity,
                                                seed=args.seed)

    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    if getattr(args, 'no_cache', False) and os.path.exists(csv_path):
        os.remove(csv_path)

    _TAU_FIELDS = ['tau_ms', 'trial_idx', 'seed', 'survival_time_ms']

    # Load cache
    completed_tau_jobs = set()
    cached_tau_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                tau = float(row['tau_ms'])
                ti = int(row['trial_idx'])
                if tau in [float(t) for t in tau_values]:
                    completed_tau_jobs.add((tau, ti))
                    cached_tau_rows.append(row)

    jobs_to_run = []
    for tau_ms in tau_values:
        for trial_idx in range(n_trials):
            if (float(tau_ms), trial_idx) not in completed_tau_jobs:
                jobs_to_run.append((float(tau_ms), trial_idx, trial_seeds[trial_idx]))

    print(f"Tau sweep: {tau_values} ms, {n_trials} trials each")
    print(f"  {len(jobs_to_run)} to run, {len(completed_tau_jobs)} cached")

    def _run_tau_job(job):
        tau_ms, trial_idx, seed = job
        lp = replace(local_params_wt, tau_adapt_pyr=tau_ms)
        result = simulate_ring(
            lp, ring_params, T_ms=T_ms_short,
            stimuli=stimuli_short, r0=r0_bi, I_adapt0=I_adapt0_bi,
            seed=seed, connectivity=connectivity,
            record_dt_ms=getattr(args, 'record_dt_ms', 5.0),
        )
        result.t_ms += BURN_IN_MS
        result.stim_window = (STIM_ONSET_MS, stim_offset_ms)
        st = compute_bump_survival_time(result, noise_floor)
        if st is None:
            st = float(result.t_ms[-1] - stim_offset_ms)
        return {
            'tau_ms': tau_ms, 'trial_idx': trial_idx, 'seed': seed,
            'survival_time_ms': st,
            '_result': result,  # keep for MSD/osc analysis (not saved to CSV)
        }

    new_tau_rows = []
    results_with_data = []
    if jobs_to_run:
        if n_workers > 1 and len(jobs_to_run) > 4:
            job_results = Parallel(n_jobs=n_workers, prefer='processes', verbose=0)(
                delayed(_run_tau_job)(job)
                for job in tqdm(jobs_to_run, desc="Tau sweep jobs")
            )
        else:
            job_results = [
                _run_tau_job(job)
                for job in tqdm(jobs_to_run, desc="Tau sweep jobs")
            ]
        for jr in job_results:
            new_tau_rows.append({k: jr[k] for k in _TAU_FIELDS})
            results_with_data.append(jr)

    if new_tau_rows:
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_TAU_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_tau_rows)

    # Rebuild results_by_tau from fresh simulations (for MSD/osc analysis)
    results_by_tau = {}
    for jr in results_with_data:
        tau = jr['tau_ms']
        if tau not in results_by_tau:
            results_by_tau[tau] = []
        results_by_tau[tau].append(jr['_result'])

    # For cached taus with no fresh results, run a small batch to get center trajectories
    # (survival times are cached, but we still need trajectories for MSD/osc)
    for tau_ms in tau_values:
        tau_f = float(tau_ms)
        if tau_f not in results_by_tau:
            # Re-run n_trials fresh (seeds from cache)
            results_by_tau[tau_f] = []
            lp = replace(local_params_wt, tau_adapt_pyr=tau_f)
            for ti in range(min(n_trials, 10)):  # cap at 10 for efficiency
                result = simulate_ring(
                    lp, ring_params, T_ms=T_ms_short,
                    stimuli=stimuli_short, r0=r0_bi, I_adapt0=I_adapt0_bi,
                    seed=trial_seeds[ti], connectivity=connectivity,
                    record_dt_ms=getattr(args, 'record_dt_ms', 5.0),
                )
                result.t_ms += BURN_IN_MS
                result.stim_window = (STIM_ONSET_MS, stim_offset_ms)
                results_by_tau[tau_f].append(result)

    # Extract metrics
    sweep_metrics = extract_tau_sweep_metrics(
        results_by_tau, noise_floor,
        stim_offset_ms=stim_offset_ms,
        delay_ms=args.delay_ms,
        osc_skip_initial_ms=args.osc_skip_initial_ms,
    )

    # Override survival time with cached values (more complete)
    all_tau_rows = cached_tau_rows + new_tau_rows
    for tau_ms in tau_values:
        tau_f = float(tau_ms)
        matching = [r for r in all_tau_rows if float(r['tau_ms']) == tau_f]
        if matching:
            sv_arr = np.array([float(r['survival_time_ms']) for r in matching])
            sweep_metrics[tau_f]['survival_time_mean_ms'] = float(np.nanmean(sv_arr))
            n = len(sv_arr)
            sd = float(np.nanstd(sv_arr, ddof=1)) if n > 1 else 0.0
            sweep_metrics[tau_f]['survival_time_sem_ms'] = sd / np.sqrt(n) if n > 1 else 0.0

    fig = plot_tau_adapt_sweep(
        sweep_metrics, tau_values,
        save_path=os.path.join(out_dir, "tau_sweep_figure.png"),
        suptitle=f"τ_adapt Sweep  (amp={_fmt(amp_factor)}×, N={n_trials})",
    )
    plt.close(fig)
    print(f"\nFigures saved to {out_dir}/")


# ============================================================================
# PHASE PLANE SUBCOMMAND
# ============================================================================

def cmd_phase_plane(args: argparse.Namespace) -> None:
    """Phase plane bifurcation analysis for each condition."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.params_json:
        base_params = load_params_json(args.params_json)
    else:
        base_params = CircuitParams()

    # For connectivity label (used in output path)
    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )
    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/phase_plane", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    condition_keys = args.conditions if args.conditions else ['WT']
    delta_I_values = np.linspace(args.delta_I_min, args.delta_I_max, args.delta_I_steps)
    amp_factor = args.amplitude

    # Estimate operating points
    operating_points = {
        'spontaneous': 0.0,
        'cue': amp_factor * base_params.I_ext_pyr(),
    }

    print(f"Phase plane analysis for conditions: {condition_keys}")
    print(f"  ΔI sweep: [{args.delta_I_min:.1f}, {args.delta_I_max:.1f}] pA "
          f"in {args.delta_I_steps} steps")

    phase_data = {}
    for cond_key in condition_keys:
        if cond_key not in STUDY_CONDITIONS:
            print(f"  Warning: unknown condition {cond_key!r}, skipping")
            continue
        condition = STUDY_CONDITIONS[cond_key]
        local_params = apply_condition(base_params, condition)
        print(f"  Running {cond_key}...", end='', flush=True)
        data = run_phase_plane_sweep(
            local_params, delta_I_values,
            T_ms=args.step_ms,
            settle_ms=args.settle_ms,
            dt_ms=0.1,
            bistable_threshold=args.bistable_threshold,
        )
        phase_data[cond_key] = data
        print(" done")

        # Save per-condition CSV
        import csv as _csv
        csv_path = os.path.join(out_dir, f"{cond_key}_phase_plane.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = _csv.writer(f)
            writer.writerow(['delta_I', 'up_pyr', 'up_som', 'up_pv', 'up_vip',
                             'down_pyr', 'down_som', 'down_pv', 'down_vip', 'bistable'])
            for i, dI in enumerate(data['delta_I']):
                up = data['up_rates'][i]
                dn = data['down_rates'][i]
                writer.writerow([dI] + list(up) + list(dn) + [int(data['bistable_mask'][i])])

    fig = plot_phase_plane(
        phase_data, condition_keys,
        operating_points=operating_points,
        save_path=os.path.join(out_dir, "phase_plane_grid.png"),
        suptitle=f"Phase Plane Analysis  (amp={_fmt(amp_factor)}×)",
    )
    plt.close(fig)
    print(f"\nFigures saved to {out_dir}/")


# ============================================================================
# TEMPORAL DISSECTION SUBCOMMAND
# ============================================================================

def cmd_temporal_dissection(args: argparse.Namespace) -> None:
    """Single clean trial temporal dissection across nodes and populations."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.params_json:
        base_params = load_params_json(args.params_json)
    else:
        base_params = CircuitParams()

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    cond_key = getattr(args, 'condition', 'WT')
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)
    # Zero noise for clean single trial
    local_params = replace(local_params, sigma_s=0.0)

    amp_factor = args.amplitude
    actual_current = amp_factor * local_params.I_ext_pyr()
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    T_ms_full = delay_end_ms
    T_ms_short = T_ms_full - BURN_IN_MS
    noise_floor = args.noise_floor

    conn_label = _network_label(ring_params)
    out_dir = os.path.join(
        _output_dir("figs/ring/temporal_dissection", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Burn-in: explicitly noiseless — this command is a "clean single trial" dissection.
    connectivity = RingConnectivity.from_params(ring_params)
    print("Computing burn-in state (no noise)...")
    r0, I_adapt0 = _compute_burnin_state(local_params, ring_params, connectivity,
                                          noise_type="none")

    stimuli_short = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=actual_current,
            sigma_deg=STIM_SIGMA_DEG,
            onset_ms=STIM_ONSET_MS - BURN_IN_MS,
            duration_ms=STIM_DURATION_MS,
        ),
    ]

    print("Running single clean trial...")
    result = simulate_ring(
        local_params, ring_params, T_ms=T_ms_short,
        stimuli=stimuli_short, r0=r0, I_adapt0=I_adapt0,
        seed=args.seed, connectivity=connectivity,
        noise_type='none',
        record_dt_ms=getattr(args, 'record_dt_ms', 5.0),
        record_adaptation=True,
    )
    result.t_ms += BURN_IN_MS
    result.stim_window = (STIM_ONSET_MS, stim_offset_ms)

    time_range = (BURN_IN_MS, T_ms_full)

    fig = plot_temporal_dissection(
        result,
        t_offset=BURN_IN_MS,
        time_range=time_range,
        noise_floor=noise_floor,
        save_path=os.path.join(out_dir, "temporal_dissection.png"),
        suptitle=(f"Temporal Dissection — {cond_key}  "
                  f"(amp={_fmt(amp_factor)}×, no noise)"),
    )
    plt.close(fig)


# ============================================================================
# ASYMMETRY: CONSTANTS & PARALLEL WORKER
# ============================================================================

#: Per-trial noisy burn-in duration.  Each trial starts from the zero state
#: and runs this much noisy spontaneous activity with its own unique seed,
#: producing fully independent pre-cue states across trials.
ASYM_SETTLING_MS: float = 6000.0

#: Window before cue onset used to compute the pre-cue asymmetry value.
ASYM_PRE_CUE_WINDOW_MS: float = 500.0

_asym_sim_args: Optional[dict] = None


def _balance_cue_location(center_deg: float, ring_params: RingParams) -> float:
    """Place cue to guarantee equal left/right node counts in the asymmetry index.

    Even N: placing the cue exactly on a node causes the antipodal node (offset
    exactly -180°) to fall in the left mask, giving left=N/2, right=N/2-1.
    Fix: shift by half a node-step so the cue sits strictly between two nodes.
    Then no node has offset 0 or ±180°, and left=right=N/2.

    Odd N: the antipodal position (cue ± 180°) is never on a node, so snapping
    to the nearest node already gives left=right=(N-1)/2 — no imbalance.
    """
    nearest_idx = ring_params.angle_to_node(center_deg)
    if ring_params.n_nodes % 2 == 0:
        return (ring_params.node_to_angle_deg(nearest_idx)
                + ring_params.angular_spacing_deg / 2) % 360.0
    else:
        return ring_params.node_to_angle_deg(nearest_idx)


def _asym_init_worker(
    base_params: CircuitParams,
    ring_params: RingParams,
    connectivity: RingConnectivity,
    amplitude: float,
    delay_ms: float,
    record_dt_ms: float,
    random_cue_location: bool = False,
    balance_cue: bool = True,
    correct_asymmetry: bool = True,
) -> None:
    """Initialise worker process for asymmetry trials."""
    global _asym_sim_args
    _asym_sim_args = {
        'base_params': base_params,
        'ring_params': ring_params,
        'connectivity': connectivity,
        'amplitude': amplitude,
        'delay_ms': delay_ms,
        'record_dt_ms': record_dt_ms,
        'random_cue_location': random_cue_location,
        'balance_cue': balance_cue,
        'correct_asymmetry': correct_asymmetry,
    }


def _asym_run_single(job: tuple) -> dict:
    """Run one asymmetry trial: noisy burn-in → cue → delay.

    Each trial starts from zero initial conditions and runs ASYM_SETTLING_MS
    of noisy spontaneous activity driven by a unique seed.  This produces
    fully independent pre-cue states across trials.  Pre-cue asymmetry is
    measured from the last ASYM_PRE_CUE_WINDOW_MS of the burn-in period;
    delay asymmetry from the full delay window after cue offset.
    """
    global _asym_sim_args
    from .analysis import compute_bump_asymmetry, decode_bump_center, compute_asymmetry_temporal_metrics

    cfg = _asym_sim_args
    cond_key, trial_idx, seed = job

    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(cfg['base_params'], condition)

    stim_onset = ASYM_SETTLING_MS
    stim_offset = stim_onset + STIM_DURATION_MS
    T_ms = stim_offset + cfg['delay_ms']
    actual_current = cfg['amplitude'] * cfg['base_params'].I_ext_pyr()

    # --- Determine cue location for this trial ---
    random_cue_location: bool = cfg.get('random_cue_location', False)
    balance_cue: bool = cfg.get('balance_cue', True)

    if random_cue_location:
        # Continuous random angle → left and right counts are inherently equal;
        # no balance correction needed.
        cue_rng = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)
        center_deg = float(cue_rng.uniform(0.0, 360.0))
    elif balance_cue:
        center_deg = _balance_cue_location(STIM_CENTER_DEG, cfg['ring_params'])
    else:
        center_deg = STIM_CENTER_DEG

    stimuli = [RingStimulus(
        center_deg=center_deg, amplitude=actual_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=stim_onset, duration_ms=STIM_DURATION_MS,
    )]

    result = simulate_ring(
        local_params, cfg['ring_params'], T_ms=T_ms,
        stimuli=stimuli, seed=seed,
        connectivity=cfg['connectivity'],
        record_dt_ms=cfg['record_dt_ms'],
    )

    asym = compute_bump_asymmetry(result)
    _, bump_amplitude = decode_bump_center(result, population=0)

    def _window_metric(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        asym_w = asym[mask]
        if not cfg.get('correct_asymmetry', True):
            return float(asym_w.mean())
        amp_w = bump_amplitude[mask]
        denom = float(amp_w.sum())
        if denom <= 1e-10:
            return 0.0
        return float((asym_w * amp_w).sum() / denom)

    # Pre-cue: last ASYM_PRE_CUE_WINDOW_MS before cue onset
    pre_mask = (
        (result.t_ms >= stim_onset - ASYM_PRE_CUE_WINDOW_MS)
        & (result.t_ms < stim_onset)
    )
    pre_cue_asym = _window_metric(pre_mask)
    # Instantaneous A(t) at the single time step just before cue onset
    last_pre_cue_asym = float(asym[pre_mask][-1]) if pre_mask.any() else 0.0

    # Delay: after stim offset + transient skip
    delay_start = stim_offset + TRANSIENT_SKIP_TIME_MS
    delay_mask = (result.t_ms >= delay_start) & (result.t_ms <= T_ms)
    delay_asym = _window_metric(delay_mask)

    # Temporal metrics on the raw (uncorrected) asymmetry timecourse
    temporal = compute_asymmetry_temporal_metrics(asym[delay_mask], result.t_ms[delay_mask])
    temporal_precue = compute_asymmetry_temporal_metrics(asym[pre_mask], result.t_ms[pre_mask])

    del result

    return {
        'cond_key': cond_key,
        'trial_idx': trial_idx,
        'seed': seed,
        'cue_deg': center_deg,
        'pre_cue_asym': pre_cue_asym,
        'last_pre_cue_asym': last_pre_cue_asym,
        'delay_asym': delay_asym,
        'correct_asymmetry': bool(cfg.get('correct_asymmetry', True)),
        'mean_abs_asym': temporal['mean_abs_asym'],
        'asym_std': temporal['asym_std'],
        'mean_abs_asym_precue': temporal_precue['mean_abs_asym'],
        'asym_std_precue': temporal_precue['asym_std'],
    }


# ============================================================================
# ASYMMETRY SUBCOMMAND
# ============================================================================

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
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,

    )

    condition_keys = args.conditions if args.conditions else ['WT', 'WT_APP', 'a7_KO_APP']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'. "
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    amp = args.amplitude
    n_trials = args.n_trials
    n_workers = _resolve_workers(args)
    random_cue_location: bool = getattr(args, 'random_cue_location', False)
    balance_cue: bool = not getattr(args, 'no_cue_balance', False)
    correct_asymmetry: bool = getattr(args, 'correct_asymmetry', True)

    conn_label = _network_label(ring_params)
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

    _print_config(args, amp, base_params, 0.0, ring_params)
    print(f"\nAsymmetry experiment:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Trials: {n_trials},  workers: {n_workers}")
    print(f"  Per-trial burn-in: {ASYM_SETTLING_MS:.0f} ms,  "
          f"pre-cue window: {ASYM_PRE_CUE_WINDOW_MS:.0f} ms,  "
          f"delay: {args.delay_ms:.0f} ms")
    print(f"  Cue location: {cue_label}")
    print(
        "  Asymmetry correction: "
        + (
            "on (weighted: Σ[A(t)·Amp(t)] / Σ[Amp(t)])"
            if correct_asymmetry else
            "off (raw mean of A(t))"
        )
    )
    if _balance_note:
        print(_balance_note)

    # --- Connectivity ---
    connectivity = RingConnectivity.from_params(ring_params)

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
            base_params, ring_params, connectivity,
            amp, args.delay_ms, args.record_dt_ms,
            random_cue_location, balance_cue, correct_asymmetry,
        )
        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
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

    anim_quality_kwargs = _snapshot_animation_quality_kwargs(args)
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
            local_params_wc, ring_params, T_ms=T_ms,
            stimuli=stimuli_worst, seed=worst['seed'],
            connectivity=connectivity,
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
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    if getattr(args, 'sigma_noise', None) is not None:
        base_params = replace(base_params, sigma_s=args.sigma_noise)
        print(f"Noise amplitude overridden: sigma_s = {args.sigma_noise}")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
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
            with ProcessPoolExecutor(
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
# ASYMMETRY × AMPLITUDE SWEEP SUBCOMMAND
# ============================================================================

def cmd_asymmetry_amp_sweep(args: argparse.Namespace) -> None:
    """Sweep cue amplitude and measure how delay asymmetry evolves.

    For each amplitude in ``--amplitudes``:
      * Uses the **same cache directory** as ``ring-asymmetry`` for that
        amplitude and correction mode, so data from either command is
        interchangeable.
      * Runs any missing trials (secondary burn-in from a shared per-condition
        state, then cue + delay) and writes them to the per-amplitude
        ``asymmetry_trials.csv``.

    After all amplitudes are processed a cross-amplitude summary figure
    (``asymmetry_amp_sweep.png``) and violin figure
    (``asymmetry_amp_sweep_violin.png``) are saved one level above the
    per-amplitude directories.

    Outputs per amplitude (shared with ring-asymmetry):
        {conn_label}/amp{X}_{mode}/asymmetry_trials.csv
    Cross-amplitude outputs:
        {conn_label}/asymmetry_amp_sweep.png
        {conn_label}/asymmetry_amp_sweep_violin.png
        {conn_label}/asymmetry_amp_sweep.csv   (aggregated per-amp summary)
    """
    _resolve_seed(args)
    from tqdm import tqdm
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Setup ─────────────────────────────────────────────────────────────────
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )

    condition_keys: list[str] = args.conditions if args.conditions else ['WT', 'WT_APP']
    for k in condition_keys:
        if k not in STUDY_CONDITIONS:
            print(f"Error: unknown condition '{k}'. "
                  f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
            sys.exit(1)

    amp_values: list[float] = sorted(set(args.amplitudes))
    n_trials   = args.n_trials
    n_workers  = _resolve_workers(args)
    balance_cue: bool = not getattr(args, 'no_cue_balance', False)
    correct_asymmetry: bool = getattr(args, 'correct_asymmetry', True)
    delay_ms: float = args.delay_ms
    record_dt_ms: float = getattr(args, 'record_dt_ms', 5.0)

    asym_mode_label = "corrected" if correct_asymmetry else "uncorrected"
    conn_label      = _network_label(ring_params)

    # Top-level output dir (cross-amplitude figures go here)
    sweep_out_dir = os.path.join(
        _output_dir("figs/ring/asymmetry", args.params_json),
        conn_label,
    )
    os.makedirs(sweep_out_dir, exist_ok=True)

    _print_config(args, amp_values[0], base_params, 0.0, ring_params)
    print(f"\nAsymmetry × amplitude sweep:")
    print(f"  Conditions : {', '.join(condition_keys)}")
    print(f"  Amplitudes : {amp_values}")
    print(f"  Trials     : {n_trials}   workers: {n_workers}")
    print(f"  Shared burn-in   : {ASYM_SETTLING_MS:.0f} ms  (once per condition)")
    print(f"  Secondary burn-in: {ASYM_AMP_SWEEP_SECONDARY_BURNIN_MS:.0f} ms  (per trial)")
    print(f"  Delay            : {delay_ms:.0f} ms")
    print(f"  Asymmetry correction: {'on' if correct_asymmetry else 'off'}")

    connectivity = RingConnectivity.from_params(ring_params)

    # ── Shared burn-in per condition ──────────────────────────────────────────
    print(f"\nComputing shared burn-in states ({ASYM_SETTLING_MS:.0f} ms per condition) …")
    shared_r0: dict[str, np.ndarray] = {}
    shared_Ia: dict[str, np.ndarray] = {}
    burnin_rng = np.random.default_rng(args.seed)

    for cond_key in condition_keys:
        local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
        burnin_seed  = int(burnin_rng.integers(0, 2**31 - 1))
        r0, Ia = _compute_burnin_state(
            local_params, ring_params, connectivity, seed=burnin_seed
        )
        shared_r0[cond_key] = r0
        shared_Ia[cond_key] = Ia
        print(f"  {cond_key}  (seed={burnin_seed})")

    # Trial seeds — deterministic, same for all amplitudes so each trial_idx
    # always maps to the same secondary-burn-in + noise realization.
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

    # ── Per-amplitude loop ────────────────────────────────────────────────────
    # Collect all trial results indexed by amplitude for the summary figure.
    sweep_data: dict[str, dict[float, dict]] = {ck: {} for ck in condition_keys}

    for amp in amp_values:
        amp_label = f"amp{amp:g}_{asym_mode_label}"
        amp_out_dir = os.path.join(sweep_out_dir, amp_label)
        os.makedirs(amp_out_dir, exist_ok=True)
        csv_path = os.path.join(amp_out_dir, "asymmetry_trials.csv")

        print(f"\n── Amplitude {amp:g}× ─────────────────────────────────────────────")

        # Load existing cache (same format / validation as ring-asymmetry)
        all_results: list[dict] = []
        cached_indices: dict[str, set] = {ck: set() for ck in condition_keys}

        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline='') as _f:
                    cached_rows = list(csv.DictReader(_f))
                if cached_rows and 'delay_ms' in cached_rows[0]:
                    params_ok = all(
                        abs(float(r.get('delay_ms',    0)) - delay_ms) < 1e-6
                        and abs(float(r.get('amplitude', 0)) - amp)    < 1e-9
                        for r in cached_rows
                    )
                    if params_ok and 'correct_asymmetry' in cached_rows[0]:
                        cached_correct = bool(int(cached_rows[0].get('correct_asymmetry', 1)))
                        if cached_correct != correct_asymmetry:
                            params_ok = False
                    if params_ok:
                        for r in cached_rows:
                            ck = r['condition']
                            if ck not in condition_keys:
                                continue
                            all_results.append({
                                'cond_key':             ck,
                                'trial_idx':            int(r['trial_idx']),
                                'seed':                 int(r['seed']),
                                'amplitude':            float(r['amplitude']),
                                'cue_deg':              float(r.get('cue_deg', STIM_CENTER_DEG)),
                                'pre_cue_asym':         float(r.get('pre_cue_asym', 'nan') or 'nan'),
                                'last_pre_cue_asym':    float(r.get('last_pre_cue_asym', 'nan') or 'nan'),
                                'delay_asym':           float(r['delay_asym']),
                                'correct_asymmetry':    bool(int(r.get('correct_asymmetry', 1))),
                                'mean_abs_asym':        float(r.get('mean_abs_asym', 'nan') or 'nan'),
                                'asym_std':             float(r.get('asym_std', 'nan') or 'nan'),
                                'mean_abs_asym_precue': float(r.get('mean_abs_asym_precue', 'nan') or 'nan'),
                                'asym_std_precue':      float(r.get('asym_std_precue', 'nan') or 'nan'),
                            })
                            cached_indices[ck].add(int(r['trial_idx']))
                        n_cached = sum(len(v) for v in cached_indices.values())
                        if n_cached > 0:
                            print(f"  Loaded {n_cached} cached trial(s) from {csv_path}")
                            for ck in condition_keys:
                                print(f"    {ck}: {len(cached_indices[ck])} / {n_trials}")
                    else:
                        print("  Cache parameter mismatch — rerunning all trials.")
                else:
                    print("  Old cache format — rerunning all trials.")
            except Exception as _e:
                print(f"  Cache read failed ({_e}) — rerunning all trials.")

        # Build job list (skip cached trials)
        jobs = [
            (cond_key, trial_idx, seed, amp)
            for cond_key in condition_keys
            for trial_idx, seed in enumerate(trial_seeds)
            if trial_idx not in cached_indices[cond_key]
        ]

        # Run new trials
        new_results: list[dict] = []
        if jobs:
            init_args = (
                base_params, ring_params, connectivity,
                delay_ms, record_dt_ms,
                balance_cue, correct_asymmetry,
                shared_r0, shared_Ia,
            )
            if n_workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_asym_amp_sweep_init_worker,
                    initargs=init_args,
                ) as executor:
                    futures = {
                        executor.submit(_asym_amp_sweep_run_single, job): job
                        for job in jobs
                    }
                    with tqdm(
                        total=len(jobs), desc=f"  amp={amp:g}", unit="sim", smoothing=0
                    ) as pbar:
                        for future in as_completed(futures):
                            new_results.append(future.result())
                            pbar.update()
            else:
                _asym_amp_sweep_init_worker(*init_args)
                for job in tqdm(jobs, desc=f"  amp={amp:g}", unit="sim"):
                    new_results.append(_asym_amp_sweep_run_single(job))

            all_results.extend(new_results)
        else:
            print("  All trials cached — skipping simulations.")

        # Save / update CSV (same format as ring-asymmetry)
        if new_results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'condition', 'trial_idx', 'seed', 'cue_deg',
                    'pre_cue_asym', 'last_pre_cue_asym', 'delay_asym',
                    'delay_ms', 'amplitude',
                    'random_cue', 'balance_cue', 'correct_asymmetry',
                    'mean_abs_asym', 'asym_std',
                    'mean_abs_asym_precue', 'asym_std_precue',
                ])
                writer.writeheader()
                for r in sorted(all_results, key=lambda r: (r['cond_key'], r['trial_idx'])):
                    writer.writerow({
                        'condition':            r['cond_key'],
                        'trial_idx':            r['trial_idx'],
                        'seed':                 r['seed'],
                        'cue_deg':              r.get('cue_deg', STIM_CENTER_DEG),
                        'pre_cue_asym':         r.get('pre_cue_asym', float('nan')),
                        'last_pre_cue_asym':    r.get('last_pre_cue_asym', float('nan')),
                        'delay_asym':           r['delay_asym'],
                        'delay_ms':             delay_ms,
                        'amplitude':            amp,
                        'random_cue':           0,
                        'balance_cue':          int(balance_cue),
                        'correct_asymmetry':    int(correct_asymmetry),
                        'mean_abs_asym':        r.get('mean_abs_asym', float('nan')),
                        'asym_std':             r.get('asym_std', float('nan')),
                        'mean_abs_asym_precue': r.get('mean_abs_asym_precue', float('nan')),
                        'asym_std_precue':      r.get('asym_std_precue', float('nan')),
                    })
            print(f"  Trial data → {csv_path}")

        # Aggregate per condition for summary figure
        for cond_key in condition_keys:
            trials = [r for r in all_results if r['cond_key'] == cond_key]
            sweep_data[cond_key][amp] = {
                'mean_abs_asym': [t['mean_abs_asym'] for t in trials
                                  if not np.isnan(t['mean_abs_asym'])],
                'asym_std':      [t['asym_std']      for t in trials
                                  if not np.isnan(t['asym_std'])],
                'delay_asym':    [t['delay_asym']    for t in trials],
            }

    # ── Summary statistics ────────────────────────────────────────────────────
    from scipy import stats as _scipy_stats

    def _sig(p) -> str:
        if p is None or np.isnan(p): return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    print("\n" + "=" * 68)
    print("Linear regression — Mean |A(t)| vs amplitude")
    print("=" * 68)
    from .plotting import _ols_fit
    for cond_key in condition_keys:
        amps_sorted = sorted(amp_values)
        means = np.array([
            np.mean(sweep_data[cond_key][a]['mean_abs_asym'])
            if sweep_data[cond_key].get(a, {}).get('mean_abs_asym') else np.nan
            for a in amps_sorted
        ])
        slope, intercept, r2 = _ols_fit(np.array(amps_sorted), means)
        print(f"  {cond_key:<12}  slope={slope:+.6f}  intercept={intercept:.4f}  R²={r2:.4f}")

    # Pairwise Mann-Whitney U at each amplitude
    if len(condition_keys) >= 2:
        print(f"\n{'Amplitude':>10}  {'Pair':<28}  {'U':>8}  {'p(U)':>10}  sig")
        print("-" * 65)
        for amp in amp_values:
            for i, ck_a in enumerate(condition_keys):
                for j, ck_b in enumerate(condition_keys):
                    if j <= i:
                        continue
                    va = np.array(sweep_data[ck_a].get(amp, {}).get('mean_abs_asym', []))
                    vb = np.array(sweep_data[ck_b].get(amp, {}).get('mean_abs_asym', []))
                    va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
                    if len(va) < 2 or len(vb) < 2:
                        continue
                    u_stat, p_u = _scipy_stats.mannwhitneyu(va, vb, alternative='two-sided')
                    print(f"  {amp:>8g}  {ck_a:<12} vs {ck_b:<12}  "
                          f"{u_stat:>8.0f}  {p_u:.4f}      {_sig(p_u)}")

    # ── Save cross-amplitude summary CSV ──────────────────────────────────────
    sweep_csv_path = os.path.join(sweep_out_dir, "asymmetry_amp_sweep.csv")
    with open(sweep_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'condition', 'amplitude',
            'n', 'mean_abs_asym_mean', 'mean_abs_asym_sem',
            'asym_std_mean', 'asym_std_sem',
        ])
        writer.writeheader()
        for amp in amp_values:
            for cond_key in condition_keys:
                d = sweep_data[cond_key].get(amp, {})
                vals_m = np.asarray(d.get('mean_abs_asym', []), float)
                vals_s = np.asarray(d.get('asym_std',      []), float)
                vals_m, vals_s = vals_m[~np.isnan(vals_m)], vals_s[~np.isnan(vals_s)]
                n = len(vals_m)
                writer.writerow({
                    'condition':         cond_key,
                    'amplitude':         amp,
                    'n':                 n,
                    'mean_abs_asym_mean': float(vals_m.mean()) if n else float('nan'),
                    'mean_abs_asym_sem':  float(vals_m.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
                    'asym_std_mean':      float(vals_s.mean()) if n else float('nan'),
                    'asym_std_sem':       float(vals_s.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
                })
    print(f"\nSummary CSV → {sweep_csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    from .plotting import plot_asymmetry_amp_sweep, plot_asymmetry_amp_sweep_violin

    title_suffix = (
        f"N={ring_params.n_nodes}, "
        f"w_inter={ring_params.w_pyr_pyr_inter}, "
        f"σ={ring_params.sigma_pyr_deg}°, "
        f"w_pv={ring_params.w_pv_global}, "
        f"{asym_mode_label}"
    )

    fig_line = plot_asymmetry_amp_sweep(
        data=sweep_data,
        amp_values=amp_values,
        condition_order=condition_keys,
        save_path=os.path.join(sweep_out_dir, "asymmetry_amp_sweep.png"),
        title_suffix=title_suffix,
    )
    plt.close(fig_line)

    fig_vio = plot_asymmetry_amp_sweep_violin(
        data=sweep_data,
        amp_values=amp_values,
        condition_order=condition_keys,
        save_path=os.path.join(sweep_out_dir, "asymmetry_amp_sweep_violin.png"),
        title_suffix=title_suffix,
    )
    plt.close(fig_vio)

    print(f"\nAll outputs saved to {sweep_out_dir}/")

    if not args.no_show:
        plt.show()
