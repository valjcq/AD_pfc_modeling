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
)
from .plotting import (
    plot_ring_dashboard,
    plot_ring_connectome,
    plot_bump_metrics_over_time,
    extract_comparison_data,
    plot_bump_metrics_comparison,
    plot_metrics_vs_delay,
    plot_metrics_vs_amplitude,
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

    response_onset = getattr(args, 'response_onset_ms', 0.0)
    if response_onset > 0:
        response_factor = getattr(args, 'response_factor', 0.5)
        response_duration = getattr(args, 'response_duration_ms', 500.0)
        print(f"Response transient: +{response_factor:.0%} of I0 to all populations, "
              f"{response_onset:.0f} ms after delay end, duration={response_duration:.0f} ms")


def _connectivity_label(rp: RingParams) -> str:
    """Build a directory-safe label encoding connectivity parameters.

    Examples:
        gauss_w3.96_s10-pv_unif_2.0
        compte_J1.6_s30-pv_gauss_0.3_s180
    """
    def _fmt(v: float) -> str:
        """Format float: drop trailing zeros, keep at most 2 decimals."""
        return f"{v:.2f}".rstrip("0").rstrip(".")

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
    return f"amp={amp_factor:.0f}×"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by ring-run and ring-study."""
    parser.add_argument("--params_json", type=str, default="",
                        help="Load local circuit parameters from JSON file")
    parser.add_argument("--n_nodes", type=int, default=128,
                        help="Number of nodes on the ring (default: 128)")
    parser.add_argument("--amplitude", type=float, default=20.0,
                        help="Stimulus amplitude as factor of I_ext_pyr baseline "
                             "(default: 20, i.e. 20× baseline current)")
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
    parser.add_argument("--sigma_pyr_deg", type=float, default=10.0,
                        help="PYR→PYR connectivity width in degrees (default: 10.0)")
    parser.add_argument("--w_pyr_pyr_inter", type=float, default=3.96,
                        help="Total PYR→PYR coupling for Gaussian profile (default: 3.96). "
                             "Not used with --pyr_profile compte.")
    parser.add_argument("--w_pv_global", type=float, default=2.0,
                        help="Total PV→PYR global inhibition strength (default: 2.0)")


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
    _print_config(args, amp, base_params, T_ms, ring_params)

    cond_key = args.condition
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_end_ms = stim_offset_ms + args.delay_ms
    local_params = _apply_response_transient(local_params, args, delay_end_ms)

    amp_dir = f"amp{amp:.0f}"
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
    print(f"  Amplitudes (× I_ext_pyr): {', '.join(f'{a:.0f}' for a in amplitudes)}")
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
        amp_out = os.path.join(out_dir, f"amp{amp:.0f}")
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
            n_trials_label = f", {n_trials} trials" if n_trials > 1 else ""
            plot_metrics_vs_delay(
                metrics_over_delay_agg, delay_labels=delay_labels,
                save_path=os.path.join(amp_out, "metrics_vs_delay.png"),
                suptitle=f"Bump Metrics During Delay  ({suptitle}{n_trials_label})",
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
        n_trials_label = f"  ({n_trials} trials)" if n_trials > 1 else ""
        plot_metrics_vs_amplitude(
            all_delay_metrics_agg,
            amplitude_values=amplitudes,
            save_path=os.path.join(out_dir, "metrics_vs_amplitude.png"),
            suptitle=f"Metrics vs Amplitude (full delay){n_trials_label}",
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
                save_path=os.path.join(out_dir, f"metrics_vs_amplitude_at_{label}.png"),
                suptitle=f"Metrics vs Amplitude at delay = {label}{n_trials_label}",
            )
            plt.close()

    # Connectome (once)
    plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))
    plt.close()

    print(f"\nFigures saved to {out_dir}/")
    print(f"Metrics cached in {csv_path}")
