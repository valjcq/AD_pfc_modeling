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

