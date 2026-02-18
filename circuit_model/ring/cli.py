"""
Ring attractor CLI logic.

This module contains the ring-specific CLI functions (cmd_run, cmd_study)
and their helpers. These are invoked from circuit_model.cli via the
ring-run and ring-study subcommands.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Optional

import numpy as np

from ..params import CircuitParams
from ..io import load_params_json, output_dir as _output_dir
from ..study import STUDY_CONDITIONS, CONDITION_ORDER, apply_condition

from .params import RingParams
from .stimulus import RingStimulus
from .simulation import simulate_ring
from .connectivity import RingConnectivity
from .analysis import (
    compute_bump_metrics,
    compute_metrics_at_delay_times,
    aggregate_metrics_across_trials,
    aggregate_single_metrics,
    population_vector_decode,
    compute_msd_curve,
    fit_diffusion_coefficient,
    compute_drift_field,
    compute_noise_floor,
)
from .plotting import (
    plot_ring_dashboard,
    plot_ring_connectome,
    plot_bump_metrics_over_time,
    extract_comparison_data,
    plot_bump_metrics_comparison,
    plot_metrics_vs_delay,
    plot_metrics_vs_amplitude,
    plot_msd_curves,
    plot_drift_field,
    plot_noise_floor_histogram,
    plot_calibration_heatmap,
    plot_calibration_timecourses,
    plot_calibration_scatter,
    plot_distractor_sweep_heatmaps,
    plot_distractor_sweep_timecourses,
    plot_distractor_sweep_activity_grid,
)


# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

BURN_IN_MS = 10000.0
STIM_ONSET_MS = BURN_IN_MS + 500.0
STIM_DURATION_MS = 250.0
STIM_CENTER_DEG = 180.0
STIM_SIGMA_DEG = 20.0


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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
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
        if ring_params.pyr_profile_type == "compte":
            print(f"Connectivity: Compte profile, J+ = {ring_params.J_plus:.2f}, "
                  f"sigma = {ring_params.sigma_pyr_deg:.1f} deg")
        else:
            print(f"Connectivity: Gaussian profile, w_inter = {ring_params.w_pyr_pyr_inter:.2f}, "
                  f"sigma = {ring_params.sigma_pyr_deg:.1f} deg")
        if ring_params.pv_global_type == "gaussian":
            print(f"Inhibition:   Gaussian PV profile, sigma = {ring_params.sigma_pv_deg:.1f} deg, "
                  f"w_pv = {ring_params.w_pv_global:.2f}")
        else:
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


def _connectivity_label(rp: RingParams) -> str:
    """Build a directory-safe label encoding connectivity parameters.

    Examples:
        gauss_w4_s30-pv_unif_2.0
        compte_J1.6_s30-pv_gauss_0.3_s180
    """

    # Excitatory profile
    if rp.pyr_profile_type == "compte":
        exc = f"compte_J{_fmt(rp.J_plus)}_s{_fmt(rp.sigma_pyr_deg)}"
    else:
        exc = f"gauss_w{_fmt(rp.w_pyr_pyr_inter)}_s{_fmt(rp.sigma_pyr_deg)}"

    # Inhibitory profile
    if rp.pv_global_type == "gaussian":
        inh = f"pv_gauss_{_fmt(rp.w_pv_global)}_s{_fmt(rp.sigma_pv_deg)}"
    else:
        inh = f"pv_unif_{_fmt(rp.w_pv_global)}"

    return f"{exc}-{inh}"


def _stim_label(amp_factor: float) -> str:
    """Short label for stimulus amplitude factor, used in plot titles."""
    return f"amp={_fmt(amp_factor)}×"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by ring-run and ring-study."""
    parser.add_argument("--params_json", type=str, default="",
                        help="Load local circuit parameters from JSON file")
    parser.add_argument("--n_nodes", type=int, default=128,
                        help="Number of nodes on the ring (default: 128)")
    parser.add_argument("--amplitude", type=float, default=15.0,
                        help="Stimulus amplitude as factor of I_ext_pyr baseline "
                             "(default: 15, i.e. 15× baseline current)")
    parser.add_argument("--delay_ms", type=float, default=3000.0,
                        help="Delay period duration in ms (default: 3000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
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
    parser.add_argument("--record_dt_ms", type=float, default=1.0,
                        help="Recording time step in ms (default: 1.0). "
                             "Only every record_dt_ms the state is stored.")
    # Connectivity parameters
    parser.add_argument("--pyr_profile", type=str, default="gaussian",
                        choices=["gaussian", "compte"],
                        help="PYR→PYR connectivity profile (default: gaussian). "
                             "'compte' uses Compte et al. (2000) with surround inhibition.")
    parser.add_argument("--J_plus", type=float, default=1.6,
                        help="Compte profile J+ parameter (local excitation). "
                             "Only used with --pyr_profile compte. (default: 1.6)")
    parser.add_argument("--sigma_pyr_deg", type=float, default=30.0,
                        help="PYR→PYR connectivity width in degrees (default: 30.0)")
    parser.add_argument("--w_pyr_pyr_inter", type=float, default=4.0,
                        help="Total PYR→PYR coupling for Gaussian profile (default: 4.0). "
                             "Not used with --pyr_profile compte.")
    parser.add_argument("--w_pv_global", type=float, default=2.0,
                        help="Total PV→PYR global inhibition strength (default: 2.0)")
    parser.add_argument("--pv_profile", type=str, default="uniform",
                        choices=["uniform", "gaussian"],
                        help="PV→PYR connectivity profile (default: uniform). "
                             "'gaussian' uses a spatially tuned inhibition profile.")
    parser.add_argument("--sigma_pv_deg", type=float, default=180.0,
                        help="PV→PYR connectivity width in degrees (default: 180.0). "
                             "Only used with --pv_profile gaussian.")
    # Noise parameters
    parser.add_argument("--sigma_noise", type=float, default=None,
                        help="Noise amplitude sigma_s (overrides params_json value). "
                             "Default uses the value in CircuitParams (~5.89).")


# ============================================================================
# RUN SUBCOMMAND
# ============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single condition and plot results."""
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
    conn_label = _connectivity_label(ring_params)
    out_dir = os.path.join(
        _output_dir(f"figs/ring/{ring_params.n_nodes}", args.params_json),
        conn_label, amp_dir, cond_key,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSimulating: {condition.label} ({cond_key})")
    print(f"  T = {T_ms:.0f} ms, delay = {args.delay_ms:.0f} ms")
    result = simulate_ring(local_params, ring_params, T_ms=T_ms,
                           stimuli=stimuli, seed=args.seed,
                           record_dt_ms=args.record_dt_ms)

    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, T_ms)
    suptitle = _stim_label(amp)

    plot_ring_dashboard(result, save_path=os.path.join(out_dir, "dashboard.png"),
                        time_range=time_range, t_offset=t_offset, suptitle=suptitle)
    plt.close()

    plot_bump_metrics_over_time(result, time_range=time_range, t_offset=t_offset)
    plt.suptitle(f"Bump Metrics Over Time  ({suptitle})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bump_metrics.png"), dpi=150, bbox_inches="tight")
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
) -> tuple[np.ndarray, np.ndarray]:
    """Run a deterministic burn-in simulation and return the final state."""
    result = simulate_ring(
        local_params, ring_params, T_ms=BURN_IN_MS,
        stimuli=None, r0=None, I_adapt0=None,
        seed=None, noise_type="none",
        connectivity=connectivity,
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
        record_dt_ms=args_d.get('record_dt_ms', 1.0),
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
        'record_dt_ms': getattr(args, 'record_dt_ms', 1.0),
    }


# ============================================================================
# STUDY SUBCOMMAND
# ============================================================================

def cmd_study(args: argparse.Namespace) -> None:
    """Run multiple conditions and generate comparison plots."""
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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
    )

    # Determine conditions
    if args.conditions is None:
        condition_keys = list(CONDITION_ORDER)
    else:
        condition_keys = args.conditions
        for k in condition_keys:
            if k not in STUDY_CONDITIONS:
                print(f"Error: unknown condition '{k}'.\n"
                      f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                sys.exit(1)

    amplitudes = args.amplitudes if args.amplitudes else [args.amplitude]
    n_trials = getattr(args, 'n_trials', 1)
    n_workers = getattr(args, 'n_workers', None)
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 4)
    no_cache = getattr(args, 'no_cache', False)
    error_band = getattr(args, 'error_band', 'sem')

    conn_label = _connectivity_label(ring_params)
    out_dir = os.path.join(
        _output_dir(f"figs/ring/{ring_params.n_nodes}", args.params_json),
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
            local_params, ring_params, connectivity,
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
                with tqdm(total=len(jobs), desc="Simulations", unit="sim") as pbar:
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

    for amp in amplitudes:
        amp_out = os.path.join(out_dir, f"amp{_fmt(amp)}")
        os.makedirs(amp_out, exist_ok=True)
        suptitle = _stim_label(amp)

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
            plot_metrics_vs_delay(
                metrics_over_delay_agg, delay_labels=delay_labels,
                save_path=os.path.join(amp_out, f"metrics_vs_delay_{error_band}.png"),
                suptitle=f"Bump Metrics During Delay  ({suptitle}{band_tag})",
                error_band=error_band,
            )
            plt.close()

        if comparison_data:
            plot_bump_metrics_comparison(
                comparison_data,
                save_path=os.path.join(amp_out, "bump_metrics_comparison.png"),
                suptitle=f"Bump Metrics Comparison  ({suptitle})",
            )
            plt.close()

        all_delay_metrics_agg[amp] = delay_end_metrics_agg

    # Cross-amplitude comparison (full delay)
    if len(amplitudes) > 1:
        band_tag = f"  ({n_trials} trials, ±{error_band.upper()})" if n_trials > 1 else ""
        plot_metrics_vs_amplitude(
            all_delay_metrics_agg,
            amplitude_values=amplitudes,
            save_path=os.path.join(out_dir, f"metrics_vs_amplitude_{error_band}.png"),
            suptitle=f"Metrics vs Amplitude (full delay){band_tag}",
            error_band=error_band,
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
                suptitle=f"Metrics vs Amplitude at delay = {label}{band_tag}",
                error_band=error_band,
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
        record_dt_ms=args_d.get('record_dt_ms', 1.0),
    )

    # Shift time back to absolute
    result.t_ms += BURN_IN_MS

    # Extract delay period trajectory
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_start_ms = stim_offset_ms + 100  # 100ms after stim to let transient settle
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
        't_delay_s': t_delay_s,
    }


# ============================================================================
# DIFFUSION SUBCOMMAND
# ============================================================================

def cmd_diffusion(args: argparse.Namespace) -> None:
    """Run diffusion (MSD) analysis across conditions."""
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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
    )

    if args.conditions is None:
        condition_keys = list(CONDITION_ORDER)
    else:
        condition_keys = args.conditions
        for k in condition_keys:
            if k not in STUDY_CONDITIONS:
                print(f"Error: unknown condition '{k}'.\n"
                      f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                sys.exit(1)

    n_trials = args.n_trials
    n_workers = getattr(args, 'n_workers', None)
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 4)

    conn_label = _connectivity_label(ring_params)
    out_dir = os.path.join(
        _output_dir(f"figs/diffusion/{ring_params.n_nodes}", args.params_json),
        conn_label,
    )
    os.makedirs(out_dir, exist_ok=True)

    _, _, T_ms_full, _, amp_factor = _build_common(args)
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
            local_params, ring_params, connectivity,
        )

    # --- Trial seeds ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

    # --- Output paths (defined early for cache check) ---
    summary_csv = os.path.join(out_dir, "diffusion_summary.csv")
    curve_csv = os.path.join(out_dir, "diffusion_msd_curves.csv")

    # --- Check for cached MSD data (aggregate-level cache) ---
    msd_data: dict[str, dict] = {}
    loaded_from_cache = False

    if os.path.exists(summary_csv) and os.path.exists(curve_csv):
        try:
            with open(summary_csv, newline='') as _f:
                summary_rows = list(csv.DictReader(_f))

            cond_keys_set = set(condition_keys)
            cached_conds = {r['condition_key'] for r in summary_rows}
            params_ok = cond_keys_set <= cached_conds and all(
                float(r['delay_ms']) == args.delay_ms
                and float(r['amplitude_factor']) == amp_factor
                and int(r['n_trials']) >= n_trials
                for r in summary_rows
                if r['condition_key'] in cond_keys_set
            )

            if params_ok:
                with open(curve_csv, newline='') as _f:
                    curve_rows = list(csv.DictReader(_f))

                curves_by_cond: dict[str, list] = {}
                for row in curve_rows:
                    curves_by_cond.setdefault(row['condition_key'], []).append(row)

                if cond_keys_set <= set(curves_by_cond.keys()):
                    print(f"\nLoading cached MSD data from {curve_csv}")
                    summary_by_cond = {r['condition_key']: r for r in summary_rows}
                    for ck in condition_keys:
                        rows = curves_by_cond[ck]
                        sr = summary_by_cond[ck]
                        msd_data[ck] = {
                            'lag_times': np.array([float(r['lag_s']) for r in rows]),
                            'msd_mean':  np.array([float(r['msd_mean']) for r in rows]),
                            'msd_sem':   np.array([float(r['msd_sem']) for r in rows]),
                            'msd_sd':    np.array([float(r['msd_sd']) for r in rows]),
                            'fit_line':  np.array([float(r['fit_line']) for r in rows]),
                            'B_hat':     float(sr['B_hat_rad2_per_s']),
                            'r_squared': float(sr['r_squared']),
                        }
                        cond_label = STUDY_CONDITIONS[ck].name
                        print(f"  {cond_label}: B_hat = {msd_data[ck]['B_hat']:.4e} rad²/s"
                              f"  (R² = {msd_data[ck]['r_squared']:.3f})")
                    loaded_from_cache = True
        except Exception as _e:
            print(f"  Cache read failed ({_e}), rerunning simulations.")
            msd_data = {}

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
                with tqdm(total=len(jobs), desc="Diffusion trials", unit="trial") as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _diffusion_init_worker(*init_args)
            for job in tqdm(jobs, desc="Diffusion trials", unit="trial"):
                all_results.append(_diffusion_run_single(job))

        # --- Compute MSD per condition ---
        fit_range_s = (0.1, min(args.delay_ms / 1000.0 * 0.4, 2.0))

        for cond_key in condition_keys:
            trials = [r for r in all_results if r['cond_key'] == cond_key]
            centers = [r['center_unwrapped_rad'] for r in trials]
            t_s = trials[0]['t_delay_s']

            lag_times, msd_mean, msd_sem, msd_sd = compute_msd_curve(centers, t_s)
            B_hat, fit_line, r_sq = fit_diffusion_coefficient(lag_times, msd_mean,
                                                               fit_range=fit_range_s)

            msd_data[cond_key] = {
                'lag_times': lag_times,
                'msd_mean': msd_mean,
                'msd_sem': msd_sem,
                'msd_sd': msd_sd,
                'fit_line': fit_line,
                'B_hat': B_hat,
                'r_squared': r_sq,
            }

            cond_label = STUDY_CONDITIONS[cond_key].name
            print(f"  {cond_label}: B_hat = {B_hat:.4e} rad²/s  (R² = {r_sq:.3f})")

    # --- Save CSVs (skipped when loaded from cache) ---
    if not loaded_from_cache:
        # 1. Summary CSV: one row per condition
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'B_hat_rad2_per_s', 'r_squared', 'n_trials',
                'delay_ms', 'amplitude_factor',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                writer.writerow({
                    'condition_key': cond_key,
                    'B_hat_rad2_per_s': msd_data[cond_key]['B_hat'],
                    'r_squared': msd_data[cond_key]['r_squared'],
                    'n_trials': n_trials,
                    'delay_ms': args.delay_ms,
                    'amplitude_factor': amp_factor,
                })

        # 2. MSD curve CSV: per-condition MSD vs lag
        with open(curve_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'lag_s', 'msd_mean', 'msd_sem', 'msd_sd',
                'fit_line',
            ])
            writer.writeheader()
            for cond_key in condition_keys:
                d = msd_data[cond_key]
                for i in range(len(d['lag_times'])):
                    writer.writerow({
                        'condition_key': cond_key,
                        'lag_s': d['lag_times'][i],
                        'msd_mean': d['msd_mean'][i],
                        'msd_sem': d['msd_sem'][i],
                        'msd_sd': d['msd_sd'][i],
                        'fit_line': d['fit_line'][i],
                    })

        print(f"\nCSVs saved to {out_dir}/")
        print(f"  diffusion_summary.csv  (B_hat per condition)")
        print(f"  diffusion_msd_curves.csv  (MSD vs lag per condition)")

    # --- Plot ---
    error_band = getattr(args, 'error_band', 'sem')
    save_path = os.path.join(out_dir, f"diffusion_msd_{error_band}.png")
    band_tag = f"  ({n_trials} trials, ±{error_band.upper()})" if n_trials > 1 else ""
    plot_msd_curves(
        msd_data,
        save_path=save_path,
        suptitle=f"Diffusion Analysis (MSD){band_tag}",
        error_band=error_band,
    )
    plt.close()

    print(f"Figure saved to {save_path}")


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
        record_dt_ms=args_d.get('record_dt_ms', 1.0),
    )

    result.t_ms += BURN_IN_MS

    # Measure bump position just before distractor and shortly after
    pre_dist_t = dist_onset_abs - 50  # 50ms before distractor
    post_dist_t = dist_onset_abs + distractor_duration_ms + 100  # 100ms after

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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
    )

    if args.conditions is None:
        condition_keys = list(CONDITION_ORDER)
    else:
        condition_keys = args.conditions
        for k in condition_keys:
            if k not in STUDY_CONDITIONS:
                print(f"Error: unknown condition '{k}'.\n"
                      f"Valid: {', '.join(STUDY_CONDITIONS.keys())}")
                sys.exit(1)

    n_trials = args.n_trials
    distractor_step = args.distractor_steps
    n_workers = getattr(args, 'n_workers', None)
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 4)

    conn_label = _connectivity_label(ring_params)
    out_dir = os.path.join(
        _output_dir(f"figs/drift_field/{ring_params.n_nodes}", args.params_json),
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
            local_params, ring_params, connectivity,
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
                with tqdm(total=len(jobs), desc="Drift field trials", unit="trial") as pbar:
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

def _calibration_conn_label(rp: RingParams) -> str:
    """Build a directory-safe label for calibration output.

    Like _connectivity_label but omits the w_pyr_pyr_inter component since
    that parameter is swept during calibration.
    """
    if rp.pyr_profile_type == "compte":
        exc = f"compte_s{_fmt(rp.sigma_pyr_deg)}"
    else:
        exc = f"gauss_s{_fmt(rp.sigma_pyr_deg)}"

    if rp.pv_global_type == "gaussian":
        inh = f"pv_gauss_{_fmt(rp.w_pv_global)}_s{_fmt(rp.sigma_pv_deg)}"
    else:
        inh = f"pv_unif_{_fmt(rp.w_pv_global)}"

    return f"{exc}-{inh}"


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
        record_dt_ms=args_d.get('record_dt_ms', 1.0),
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
    delay_mask = (result.t_ms >= stim_offset_ms + 100) & (result.t_ms <= delay_end_ms)
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


# ============================================================================
# CALIBRATE SUBCOMMAND
# ============================================================================

def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run 2D parameter calibration (amplitude x w_inter)."""
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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
    )

    if args.conditions is None:
        condition_keys = ["WT"]
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
    n_baseline = args.n_baseline
    noise_percentile = args.noise_percentile
    n_workers = getattr(args, 'n_workers', None)
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 4)

    conn_label = _calibration_conn_label(ring_params_base)
    out_dir = os.path.join(
        _output_dir(f"figs/calibration/{ring_params_base.n_nodes}", args.params_json),
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

    print(f"\nCalibration configuration:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  Amplitudes (x I_ext_pyr): {', '.join(_fmt(a) for a in amplitudes)}")
    print(f"  w_inter values: {', '.join(_fmt(w) for w in w_inter_values)}")
    print(f"  Grid points: {len(amplitudes)} x {len(w_inter_values)} = {len(amplitudes) * len(w_inter_values)}")
    print(f"  Trials per grid point: {n_trials}")
    print(f"  Baseline trials per w_inter: {n_baseline}")
    print(f"  Delay = {args.delay_ms:.0f} ms, workers = {n_workers}")

    total_sims = (
        len(condition_keys) * len(w_inter_values) * n_baseline
        + len(condition_keys) * len(amplitudes) * len(w_inter_values) * n_trials
    )
    print(f"  Total simulations: {total_sims}")

    # --- Pre-compute connectivity for each w_inter ---
    print("\nBuilding connectivity matrices...")
    connectivity_cache: dict[float, RingConnectivity] = {}
    for w in tqdm(w_inter_values, desc="Connectivity", unit="w"):
        rp = replace(ring_params_base, w_pyr_pyr_inter=w)
        connectivity_cache[w] = RingConnectivity.from_params(rp)

    # --- Pre-compute burn-in for each (condition, w_inter) ---
    print("Computing burn-in states...")
    burnin_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    burnin_jobs = [(ck, w) for ck in condition_keys for w in w_inter_values]
    for ck, w in tqdm(burnin_jobs, desc="Burn-in", unit="state"):
        condition = STUDY_CONDITIONS[ck]
        local_params = apply_condition(base_params, condition)
        rp = replace(ring_params_base, w_pyr_pyr_inter=w)
        burnin_cache[(ck, w)] = _compute_burnin_state(
            local_params, rp, connectivity_cache[w],
        )

    # --- Trial seeds ---
    max_trials = max(n_trials, n_baseline)
    trial_seeds = _generate_trial_seeds(args.seed, max_trials)

    # --- Phase 1: Baseline (no-stimulus) trials ---
    print("\n--- Phase 1: Noise floor estimation ---")
    baseline_jobs = []
    for ck in condition_keys:
        for w in w_inter_values:
            for ti in range(n_baseline):
                baseline_jobs.append((ck, 0.0, w, ti, trial_seeds[ti]))

    args_dict = {
        **_args_to_dict(args),
        'amplitude': 0.0,
    }
    init_args = (
        args_dict, base_params, ring_params_base,
        connectivity_cache, burnin_cache, T_ms_full, eval_times_ms,
    )

    baseline_results: list[dict] = []
    if n_workers > 1 and len(baseline_jobs) > 1:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_calibrate_init_worker,
            initargs=init_args,
        ) as executor:
            futures = {executor.submit(_calibrate_run_single, job): job
                       for job in baseline_jobs}
            with tqdm(total=len(baseline_jobs), desc="Baseline", unit="trial") as pbar:
                for future in as_completed(futures):
                    baseline_results.append(future.result())
                    pbar.update()
    else:
        _calibrate_init_worker(*init_args)
        for job in tqdm(baseline_jobs, desc="Baseline", unit="trial"):
            baseline_results.append(_calibrate_run_single(job))

    # Compute noise floor per (condition, w_inter)
    noise_thresholds: dict[tuple[str, float], float] = {}
    baseline_A_hat_data: dict[tuple[str, float], np.ndarray] = {}
    for ck in condition_keys:
        for w in w_inter_values:
            trials = [r for r in baseline_results
                      if r['cond_key'] == ck and r['w_inter'] == w]
            all_A = []
            for r in trials:
                all_A.append(r['A_hat_final'])
                all_A.extend(r['A_hat_timecourse'])
            all_A = np.array(all_A)
            baseline_A_hat_data[(ck, w)] = all_A
            noise_thresholds[(ck, w)] = compute_noise_floor(all_A, noise_percentile)
            cond_label = STUDY_CONDITIONS[ck].name
            print(f"  {cond_label}, w={w:.2f}: threshold = {noise_thresholds[(ck, w)]:.4f} "
                  f"(p{noise_percentile:.0f}, n={len(all_A)})")

    # --- Phase 2: Grid exploration ---
    print("\n--- Phase 2: Grid exploration ---")
    grid_jobs = []
    for ck in condition_keys:
        for amp in amplitudes:
            for w in w_inter_values:
                for ti in range(n_trials):
                    grid_jobs.append((ck, amp, w, ti, trial_seeds[ti]))

    args_dict_grid = {
        **_args_to_dict(args),
        'amplitude': 0.0,
    }
    init_args_grid = (
        args_dict_grid, base_params, ring_params_base,
        connectivity_cache, burnin_cache, T_ms_full, eval_times_ms,
    )

    grid_results: list[dict] = []
    if n_workers > 1 and len(grid_jobs) > 1:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_calibrate_init_worker,
            initargs=init_args_grid,
        ) as executor:
            futures = {executor.submit(_calibrate_run_single, job): job
                       for job in grid_jobs}
            with tqdm(total=len(grid_jobs), desc="Grid trials", unit="trial") as pbar:
                for future in as_completed(futures):
                    grid_results.append(future.result())
                    pbar.update()
    else:
        _calibrate_init_worker(*init_args_grid)
        for job in tqdm(grid_jobs, desc="Grid trials", unit="trial"):
            grid_results.append(_calibrate_run_single(job))

    # --- Aggregate per condition ---
    error_band = getattr(args, 'error_band', 'sem')

    for ck in condition_keys:
        cond_label = STUDY_CONDITIONS[ck].name
        print(f"\n=== Results for {cond_label} ===")

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

                tc_array = np.array([r['A_hat_timecourse'] for r in trials])
                n_t = len(trials)
                timecourse_data[(amp, w)] = {
                    'A_hat_mean': np.mean(tc_array, axis=0),
                    'A_hat_sem': np.std(tc_array, axis=0, ddof=1) / np.sqrt(n_t)
                    if n_t > 1 else np.zeros(tc_array.shape[1]),
                    'A_hat_sd': np.std(tc_array, axis=0, ddof=1)
                    if n_t > 1 else np.zeros(tc_array.shape[1]),
                }

        # --- Save CSVs ---
        cond_prefix = f"{ck}_" if len(condition_keys) > 1 else ""

        trial_csv = os.path.join(out_dir, f"{cond_prefix}calibration_results.csv")
        with open(trial_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'condition_key', 'amplitude', 'w_inter', 'trial_idx', 'seed',
                'A_hat_final', 'peak_pyr_rate', 'center_final_deg',
                'error_from_cue_deg',
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
                    'peak_pyr_rate': r['peak_pyr_rate'],
                    'center_final_deg': r['center_final_deg'],
                    'error_from_cue_deg': r['error_from_cue_deg'],
                })

        summary_csv = os.path.join(out_dir, f"{cond_prefix}calibration_summary.csv")
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

        # --- Plots ---
        baseline_for_plot = {w: baseline_A_hat_data[(ck, w)] for w in w_inter_values}
        thresholds_for_plot = {w: noise_thresholds[(ck, w)] for w in w_inter_values}
        plot_noise_floor_histogram(
            baseline_for_plot, thresholds_for_plot,
            save_path=os.path.join(out_dir, f"{cond_prefix}noise_floor.png"),
            suptitle=f"Noise Floor ({cond_label}, {n_baseline} trials, p{noise_percentile:.0f})",
        )
        plt.close()

        plot_calibration_heatmap(
            grid_data, "success_rate", amplitudes, w_inter_values,
            cmap="RdYlGn", vmin=0, vmax=1,
            save_path=os.path.join(out_dir, f"{cond_prefix}heatmap_success_rate.png"),
            suptitle=f"Success Rate ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        plot_calibration_heatmap(
            grid_data, "mean_A_hat", amplitudes, w_inter_values,
            cmap="viridis", vmin=0, vmax=1,
            save_path=os.path.join(out_dir, f"{cond_prefix}heatmap_A_hat.png"),
            suptitle=f"Mean A_hat ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        plot_calibration_heatmap(
            grid_data, "peak_pyr_rate", amplitudes, w_inter_values,
            cmap="hot",
            save_path=os.path.join(out_dir, f"{cond_prefix}heatmap_peak_pyr.png"),
            suptitle=f"Peak PYR Rate ({cond_label}, {n_trials} trials)",
        )
        plt.close()

        tc_keys = sorted(timecourse_data.keys())
        if len(tc_keys) > 8:
            step = max(1, len(tc_keys) // 8)
            tc_keys = tc_keys[::step][:8]
        tc_subset = {k: timecourse_data[k] for k in tc_keys}
        band_tag = f"+/-{error_band.upper()}" if n_trials > 1 else ""
        plot_calibration_timecourses(
            tc_subset, eval_times_s, error_band=error_band,
            save_path=os.path.join(out_dir, f"{cond_prefix}timecourses_{error_band}.png"),
            suptitle=f"A_hat Time Courses ({cond_label}, {n_trials} trials, {band_tag})",
        )
        plt.close()

        plot_calibration_scatter(
            grid_data, save_path=os.path.join(out_dir, f"{cond_prefix}scatter_summary.png"),
            suptitle=f"Calibration Summary ({cond_label})",
        )
        plt.close()


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

    Matches on condition_key, amplitude, and w_inter (within 1e-4 tolerance).
    Returns None if the file is missing, unreadable, or has no matching row.
    """
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
    base_params: "CircuitParams",
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
        'base_params': base_params,
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

    base_params = cfg['base_params']
    ring_params = cfg['ring_params']
    connectivity = cfg['connectivity']
    r0, I_adapt0 = cfg['burnin_state']
    T_ms_short = cfg['T_ms_short']
    cue_amp_factor = cfg['cue_amp_factor']
    delay1_ms = cfg['delay1_ms']
    distractor_duration_ms = cfg['distractor_duration_ms']
    record_dt_ms = cfg['record_dt_ms']

    cue_current = cue_amp_factor * base_params.I_ext_pyr()
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
        base_params, ring_params, T_ms=T_ms_short,
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

    # Measurement windows: 50 ms before onset, 100 ms after offset
    pre_t = dist_onset_abs - 50.0
    post_t = dist_offset_abs + 100.0

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
        pv_global_type=args.pv_profile,
        sigma_pv_deg=args.sigma_pv_deg,
        pyr_profile_type=args.pyr_profile,
        J_plus=args.J_plus,
    )

    cond_key = args.condition
    if cond_key not in STUDY_CONDITIONS:
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
        cal_conn_label = _calibration_conn_label(ring_params)
        cal_csv = os.path.join(
            _output_dir(f"figs/calibration/{ring_params.n_nodes}", args.params_json),
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
    record_dt_ms = getattr(args, 'record_dt_ms', 1.0)
    n_workers = getattr(args, 'n_workers', None) or min(4, os.cpu_count() or 4)

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
    conn_label = _connectivity_label(ring_params)
    out_dir = os.path.join(
        _output_dir(f"figs/distractor_sweep/{ring_params.n_nodes}", args.params_json),
        conn_label,
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
    burnin_state = _compute_burnin_state(local_params, ring_params, connectivity)

    # --- Trial seeds ---
    trial_seeds = _generate_trial_seeds(args.seed, n_trials)

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

    if not loaded_from_cache:
        # --- Build and execute jobs ---
        jobs = [
            (off, amp, trial_idx, trial_seeds[trial_idx])
            for off in offsets_deg
            for amp in amp_factors
            for trial_idx in range(n_trials)
        ]

        init_args = (
            base_params, ring_params, connectivity, burnin_state,
            T_ms_short, cue_amp_factor, delay1_ms,
            distractor_duration_ms, delay2_ms,
            collapse_threshold, record_dt_ms,
        )

        if n_workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_distractor_sweep_init_worker,
                initargs=init_args,
            ) as executor:
                futures = {executor.submit(_distractor_sweep_run_single, job): job
                           for job in jobs}
                with tqdm(total=len(jobs), desc="Distractor sweep trials", unit="trial") as pbar:
                    for future in as_completed(futures):
                        all_results.append(future.result())
                        pbar.update()
        else:
            _distractor_sweep_init_worker(*init_args)
            for job in tqdm(jobs, desc="Distractor sweep trials", unit="trial"):
                all_results.append(_distractor_sweep_run_single(job))

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
            print(f"  Δφ={off:.0f}°, amp={amp:.2g}×: "
                  f"drift={drift_mean:.1f}°±{drift_sem:.1f}°, "
                  f"collapse={collapse_prob:.0%}")

    # --- Save CSVs (skipped when loaded from cache) ---
    if not loaded_from_cache:
        # Per-trial raw
        with open(raw_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'offset_deg', 'amp_factor', 'trial_idx',
                'displacement_deg', 'pre_amp', 'post_amp',
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        # Summary grid (includes metadata columns for cache validation)
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

    # --- Figure 1 & 2: Heatmaps ---
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

    # --- Figure 3: Timecourses for representative cells ---
    # Pick up to 6 representative (offset, amp) pairs: corners + centre
    candidates = [
        (offsets_deg[0],  amp_factors[0]),
        (offsets_deg[0],  amp_factors[-1]),
        (offsets_deg[len(offsets_deg) // 2], amp_factors[len(amp_factors) // 2]),
        (offsets_deg[-1], amp_factors[0]),
        (offsets_deg[-1], amp_factors[len(amp_factors) // 2]),
        (offsets_deg[-1], amp_factors[-1]),
    ]
    # Deduplicate while preserving order
    seen: set = set()
    selected_cells = []
    for cell in candidates:
        if cell not in seen:
            seen.add(cell)
            selected_cells.append(cell)

    # --- Figure 4: Activity grid — all non-zero offsets at amp ≈ 0.75× ---
    activity_amp = min(amp_factors, key=lambda a: abs(a - 0.75))
    activity_cells = [(off, activity_amp) for off in offsets_deg if off != 0.0]

    # Union of all cells needed (avoid duplicate simulations)
    all_cells_needed = list(dict.fromkeys(selected_cells + activity_cells))

    print("\nRunning detailed timecourse simulations for representative cells...")
    _distractor_sweep_init_worker(*init_args)  # Re-init in main process
    tc_map: dict = {}
    for (off, amp) in tqdm(all_cells_needed, desc="Timecourse runs"):
        tc_job = (off, amp, 0, trial_seeds[0])
        # Run with fine recording to get full trajectory
        _distractor_sweep_sim_args['record_dt_ms'] = record_dt_ms

        tc_result_dict = _distractor_sweep_run_single(tc_job)

        # Re-run to get the full result object (we need r for every timestep)
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

    tc_data = [tc_map[cell] for cell in selected_cells]
    activity_tc_data = [tc_map[cell] for cell in activity_cells]

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

    fig4 = plot_distractor_sweep_activity_grid(
        activity_tc_data,
        cue_onset_ms=cue_onset_abs,
        cue_offset_ms=cue_offset_abs,
        dist_onset_ms=dist_onset_abs,
        dist_offset_ms=dist_offset_abs,
        burn_in_ms=BURN_IN_MS,
        save_path=os.path.join(out_dir, "activity_grid.png"),
        suptitle=f"{condition.name} — PYR Activity",
    )
    plt.close(fig4)
    print(f"  activity_grid.png")
    print(f"\nAll outputs saved to {out_dir}/")
