"""
Command-line interface for the ring attractor simulation.

This module provides two main commands:
- run: Run a single condition and plot results
- study: Run multiple conditions and generate comparison plots

Usage:
    python ring_cli.py run [options]
    python ring_cli.py study [options]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from circuit_model import CircuitParams, load_params_json
from circuit_model.study import STUDY_CONDITIONS, CONDITION_ORDER, apply_condition
from ring_attractor import (
    RingParams, RingStimulus, RingSimulationResult, simulate_ring,
    plot_ring_dashboard, plot_ring_connectome, plot_bump_metrics_over_time,
    compute_metrics_at_delay_times,
    plot_bump_metrics_comparison, plot_metrics_vs_delay,
)


# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

BURN_IN_MS = 10000.0
STIM_ONSET_MS = BURN_IN_MS + 500.0
STIM_DURATION_MS = 250.0
STIM_CENTER_DEG = 180.0
STIM_AMPLITUDE = 150


def _output_dir(base_dir: str, params_json: str) -> str:
    """Derive output directory from params file."""
    if params_json:
        stem = Path(params_json).stem
    else:
        stem = "default"
    out = os.path.join(base_dir, stem)
    os.makedirs(out, exist_ok=True)
    return out


def _build_common(args) -> tuple[CircuitParams, RingParams, float, list[RingStimulus]]:
    """Build base params, ring params, T_ms, and stimuli from parsed args."""
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    ring_params = RingParams(
        n_nodes=128, w_pyr_pyr_inter=0.5,
        sigma_pyr_deg=10.0, w_pv_global=2,
    )

    T_ms = STIM_ONSET_MS + STIM_DURATION_MS + args.delay_ms

    stimuli = [
        RingStimulus(
            center_deg=STIM_CENTER_DEG, amplitude=STIM_AMPLITUDE,
            onset_ms=STIM_ONSET_MS, duration_ms=STIM_DURATION_MS,
        ),
    ]

    return base_params, ring_params, T_ms, stimuli


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by run and study."""
    parser.add_argument("--params_json", type=str, default="",
                        help="Load local circuit parameters from JSON file")
    parser.add_argument("--delay_ms", type=float, default=3000.0,
                        help="Delay period duration in ms (default: 3000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't display plots (useful for batch processing)")


# ============================================================================
# RUN SUBCOMMAND
# ============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single condition and plot results."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, ring_params, T_ms, stimuli = _build_common(args)

    # Apply condition
    cond_key = args.condition
    condition = STUDY_CONDITIONS[cond_key]
    local_params = apply_condition(base_params, condition)

    out_dir = os.path.join(_output_dir("figs/ring", args.params_json), cond_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSimulating: {condition.label} ({cond_key})")
    print(f"  T = {T_ms:.0f} ms, delay = {args.delay_ms:.0f} ms")
    result = simulate_ring(local_params, ring_params, T_ms=T_ms,
                           stimuli=stimuli, seed=args.seed)

    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, T_ms)

    # Dashboard
    plot_ring_dashboard(result, save_path=os.path.join(out_dir, "dashboard.png"),
                        time_range=time_range, t_offset=t_offset)
    plt.close()

    # Bump metrics
    plot_bump_metrics_over_time(result, time_range=time_range, t_offset=t_offset)
    plt.savefig(os.path.join(out_dir, "bump_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Connectome
    plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))
    plt.close()

    print(f"\nFigures saved to {out_dir}/")


# ============================================================================
# STUDY SUBCOMMAND
# ============================================================================

def cmd_study(args: argparse.Namespace) -> None:
    """Run multiple conditions and generate comparison plots."""
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_params, ring_params, T_ms, stimuli = _build_common(args)
    out_dir = _output_dir("figs/ring", args.params_json)

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

    print(f"\nStudy configuration:")
    print(f"  Conditions: {', '.join(condition_keys)}")
    print(f"  T = {T_ms:.0f} ms, delay = {args.delay_ms:.0f} ms")
    print()

    # Run simulations
    results: dict[str, RingSimulationResult] = {}
    for cond_key in condition_keys:
        condition = STUDY_CONDITIONS[cond_key]
        local_params = apply_condition(base_params, condition)

        print(f"  Simulating: {condition.label} ({cond_key})...")
        result = simulate_ring(local_params, ring_params, T_ms=T_ms,
                               stimuli=stimuli, seed=args.seed)
        results[cond_key] = result

    t_offset = BURN_IN_MS
    time_range = (BURN_IN_MS, T_ms)

    # Per-condition dashboards
    for cond_key, result in results.items():
        cond_out = os.path.join(out_dir, cond_key)
        os.makedirs(cond_out, exist_ok=True)

        plot_ring_dashboard(result, save_path=os.path.join(cond_out, "dashboard.png"),
                            time_range=time_range, t_offset=t_offset)
        plt.close()

        plot_bump_metrics_over_time(result, time_range=time_range, t_offset=t_offset)
        plt.savefig(os.path.join(cond_out, "bump_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Comparison: overlaid bump metrics
    plot_bump_metrics_comparison(
        results, time_range=time_range, t_offset=t_offset,
        save_path=os.path.join(out_dir, "bump_metrics_comparison.png"),
    )
    plt.close()

    # Comparison: metrics vs delay time
    stim_offset_ms = STIM_ONSET_MS + STIM_DURATION_MS
    delay_eval_offsets = [1000.0, 2000.0, 3000.0]
    delay_eval_times = [stim_offset_ms + dt for dt in delay_eval_offsets
                        if stim_offset_ms + dt <= T_ms]
    delay_labels = [f"{dt/1000:.0f}s" for dt in delay_eval_offsets
                    if stim_offset_ms + dt <= T_ms]

    if delay_eval_times:
        metrics_over_delay = {}
        for cond_key, result in results.items():
            metrics_over_delay[cond_key] = compute_metrics_at_delay_times(
                result, delay_eval_times, window_ms=200.0,
            )

        plot_metrics_vs_delay(
            metrics_over_delay, delay_labels=delay_labels,
            save_path=os.path.join(out_dir, "metrics_vs_delay.png"),
        )
        plt.close()

    # Connectome (once)
    plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))
    plt.close()

    print(f"\nFigures saved to {out_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ring Attractor Working Memory Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single simulation (WT by default)
    python ring_cli.py run

    # Run with APP condition
    python ring_cli.py run --condition WT_APP

    # Run with alpha5 KO and custom params
    python ring_cli.py run --condition a5_KO --params_json params/code.json

    # Study: compare all 8 conditions
    python ring_cli.py study

    # Study: compare specific conditions
    python ring_cli.py study --conditions WT WT_APP a5_KO a5_KO_APP

    # Study with longer delay
    python ring_cli.py study --conditions WT WT_APP --delay_ms 5000
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- RUN subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run simulation for a single condition",
        description="Run ring attractor simulation with a single experimental "
                    "condition and visualize results.",
    )
    add_common_args(run_parser)
    run_parser.add_argument(
        "--condition", type=str, default="WT",
        choices=list(STUDY_CONDITIONS.keys()),
        help="Experimental condition (default: WT)",
    )

    # --- STUDY subcommand ---
    study_parser = subparsers.add_parser(
        "study",
        help="Run multiple conditions and compare bump metrics",
        description="Run ring attractor simulation across multiple experimental "
                    "conditions and generate comparison plots.",
    )
    add_common_args(study_parser)
    study_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to simulate (default: all 8). "
             f"Valid: {', '.join(STUDY_CONDITIONS.keys())}",
    )

    # Parse
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        print("\nNo command specified. Use 'run' or 'study'.")
        sys.exit(1)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "study":
        cmd_study(args)


if __name__ == "__main__":
    main()
