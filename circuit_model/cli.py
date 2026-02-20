"""
Command-line interface for the circuit model.

This module provides two main commands:
- run: Run a simulation with given parameters and plot results
- optimize: Run Nevergrad optimization to find parameters matching target rates

Usage:
    python -m circuit_model run [options]
    python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

from .params import CircuitParams, ParamBound, default_bounds
from .loss import TargetRates, FitConfig
from .io import load_params_json, format_params_as_code, output_dir as _output_dir
from .optimization import nevergrad_optimize
from .simulation import simulate_circuit


def parse_freeze_list(s: str) -> set[str]:
    """Parse comma-separated list of parameter names to freeze."""
    return {x.strip() for x in s.split(",") if x.strip()}


def parse_set_params(s: str) -> dict[str, float]:
    """Parse 'name=value,name=value' into a dict of overrides."""
    overrides: dict[str, float] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --set format: '{item}' (expected name=value)")
        name, val = item.split("=", 1)
        overrides[name.strip()] = float(val.strip())
    return overrides


def print_parameter_status(
    bounds: dict[str, ParamBound],
    freeze: set[str],
    base: CircuitParams,
) -> None:
    """Print which parameters are free vs frozen during optimization."""
    all_params = {f.name for f in fields(CircuitParams)}
    in_bounds = set(bounds.keys())
    frozen = freeze | (all_params - in_bounds)
    free = in_bounds - freeze

    print("\n" + "=" * 70)
    print("PARAMETER STATUS")
    print("=" * 70)

    # Group parameters by category
    categories = {
        "Time constants": ["tau_s", "tau_adapt_pyr", "tau_adapt_som"],
        "Adaptation": ["J_adapt_pyr", "J_adapt_som"],
        "Noise & GABA": ["sigma_s", "g_gaba_base", "g_alpha7"],
        "Weights (excitatory)": ["w_ee", "w_ep", "w_es", "w_ev"],
        "Weights (inhibitory)": ["w_pe", "w_pp", "w_ps", "w_se", "w_sp", "w_vp", "w_vs", "w_vv"],
        "External currents": ["I0_pyr", "I0_pv", "I_alpha7_pv", "I0_som", "I_alpha7_som", "I_beta2_som", "I0_vip", "I_alpha5_vip"],
        "Transient": ["trans_factor"],
        "Transfer function": ["Theta_pyr", "alpha_pyr", "Theta_pv", "alpha_pv", "Theta_som", "alpha_som", "Theta_vip", "alpha_vip", "g_e", "g_i"],
        "Receptor activation": ["act_alpha7", "act_beta2", "act_alpha5"],
    }

    for cat_name, param_names in categories.items():
        print(f"\n{cat_name}:")
        for name in param_names:
            if name not in all_params:
                continue
            value = getattr(base, name)
            if name in free:
                bound = bounds.get(name)
                if bound:
                    mode_str = "log" if bound.mode == "log" else "lin"
                    print(f"  [FREE]   {name:<20} = {value:<12.6g}  ({bound.lo:.2g} - {bound.hi:.2g}, {mode_str})")
                else:
                    print(f"  [FREE]   {name:<20} = {value:<12.6g}")
            else:
                print(f"  [FROZEN] {name:<20} = {value:<12.6g}")

    print("\n" + "-" * 70)
    print(f"Total: {len(free)} free parameters, {len(frozen)} frozen parameters")
    print("=" * 70 + "\n")


def add_simulation_args(parser: argparse.ArgumentParser) -> None:
    """Add common simulation arguments to a parser."""
    parser.add_argument("--T_ms", type=float, default=2500.0,
                        help="Simulation duration (ms)")
    parser.add_argument("--dt_ms", type=float, default=0.1,
                        help="Integration time step (ms)")
    parser.add_argument("--noise_type", choices=["none", "white", "ou"], default="none",
                        help="Noise type: none, white, or ou (Ornstein-Uhlenbeck)")
    parser.add_argument("--tau_noise_ms", type=float, default=5.0,
                        help="OU noise time constant (ms)")
    parser.add_argument("--seed", type=int, default=442,  # Chosen for reproducibility
                        help="Random seed for reproducibility")
    parser.add_argument("--params_json", type=str, default="",
                        help="Load parameters from JSON file")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a simulation and plot the results."""
    from dataclasses import replace
    from .plotting import plot_simulation_dashboard, print_simulation_summary

    # Load or create parameters
    if args.params_json:
        params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        params = CircuitParams()
        print("Using default parameters")

    # Apply transient settings if enabled
    use_transient = args.enable_transient
    if use_transient:
        params = replace(
            params,
            trans_enabled=True,
            trans_start_ms=args.trans_start_ms,
            trans_duration_ms=args.trans_duration_ms,
            trans_factor=args.trans_factor,
        )

    # Print key parameter values
    print("\nKey parameters:")
    print(f"  tau_s = {params.tau_s:.2f} ms")
    print(f"  sigma_s = {params.sigma_s:.2f} (noise)")
    print(f"  g_gaba = {params.g_gaba():.2f} (GABA scaling)")

    if use_transient:
        trans_end = params.trans_start_ms + params.trans_duration_ms
        print(f"\nTransient current (applied to all populations):")
        print(f"  trans_factor = {params.trans_factor:.2f} (fraction of I0)")
        print(f"  Window: {params.trans_start_ms:.1f} - {trans_end:.1f} ms")

    # Run simulation
    print(f"\nRunning simulation: T={args.T_ms} ms, dt={args.dt_ms} ms, noise={args.noise_type}")

    result = simulate_circuit(
        params,
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        seed=args.seed,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        use_transient=use_transient,
    )

    # Print summary
    burn_in = args.burn_in_ms if hasattr(args, "burn_in_ms") else args.T_ms * 0.5
    print_simulation_summary(result, burn_in_ms=burn_in)

    # Plot
    time_range = None
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            time_range = (float(parts[0]), float(parts[1]))

    title = f"Circuit Model Simulation (noise={args.noise_type})"
    if use_transient:
        title += f" [Transient: {params.trans_start_ms:.0f}-{params.trans_start_ms + params.trans_duration_ms:.0f} ms]"

    # Determine save path
    if args.save_plot:
        save_path = args.save_plot
    else:
        out_dir = _output_dir("figs/runs", args.params_json)
        save_path = os.path.join(out_dir, f"circuit_simulation_{args.noise_type}.png")

    plot_simulation_dashboard(
        result,
        title=title,
        time_range=time_range,
        save_path=save_path,
        show=not args.no_show,
        unit=args.unit,
    )


def cmd_study(args: argparse.Namespace) -> None:
    """Run batch study across experimental conditions and generate box plots."""
    from .study import (
        STUDY_CONDITIONS,
        StudyConfig,
        run_study,
        plot_study_boxplots,
    )

    # Load base parameters
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        base_params = CircuitParams()
        print("Using default parameters")

    # Override noise amplitude if provided
    if args.sigma_noise is not None:
        from dataclasses import replace
        base_params = replace(base_params, sigma_s=args.sigma_noise)
        print(f"Noise amplitude overridden: sigma_s = {args.sigma_noise}")

    # Build config
    cfg = StudyConfig(
        n_runs=args.n_runs,
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        burn_in_ms=args.burn_in_ms,
        window_ms=args.window_ms,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        n_workers=args.n_workers,
        fixed_receptor_values=args.fixed_receptor_values,
    )

    # Print study info
    print(f"\nStudy configuration:")
    print(f"  Conditions: {len(STUDY_CONDITIONS)}")
    print(f"  Runs per condition: {cfg.n_runs}")
    print(f"  Total simulations: {len(STUDY_CONDITIONS) * cfg.n_runs}")
    if cfg.noise_type == "none":
        noise_detail = "none"
    elif cfg.noise_type == "white":
        noise_detail = f"white, sigma_s={base_params.sigma_s:.4f}"
    else:  # ou
        noise_detail = f"ou, sigma_s={base_params.sigma_s:.4f}, tau_noise={cfg.tau_noise_ms}ms"
    print(f"  Simulation: T={cfg.T_ms}ms, dt={cfg.dt_ms}ms, noise={noise_detail}")
    print(f"  Receptor activation: {'fixed mean values' if cfg.fixed_receptor_values else 'sampled from distributions'}")
    print(f"  Statistics: burn_in={cfg.burn_in_ms}ms, window={cfg.window_ms}ms")
    print()

    # Run study
    seed = args.seed if args.seed is not None else 0
    results = run_study(base_params, cfg, base_seed=seed, verbose=True)

    # Determine save path
    if args.save_plot:
        save_path = args.save_plot
    else:
        out_dir = _output_dir("figs/boxplot", args.params_json)
        save_path = os.path.join(out_dir, f"study_boxplots_{cfg.noise_type}.png")

    # Generate box plot
    print("\nGenerating box plot...")
    plot_study_boxplots(
        results,
        title=f"Firing Rate Distribution ({cfg.n_runs} runs per condition)",
        save_path=save_path,
        show=not args.no_show,
        unit=args.unit,
    )


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run parameter optimization."""
    # Build target rates
    target = TargetRates(
        mean_r_pyr=args.target_pyr,
        mean_r_som=args.target_som,
        mean_r_pv=args.target_pv,
        mean_r_vip=args.target_vip,
        alpha7_ko_pyr=args.target_alpha7_ko_pyr,
        alpha5_ko_pyr=args.target_alpha5_ko_pyr,
        beta2_ko_pyr=args.target_beta2_ko_pyr,
    )

    # Load or create base parameters
    if args.params_json:
        base = load_params_json(args.params_json)
        print(f"Loaded base parameters from: {args.params_json}")
    else:
        base = CircuitParams()
        print("Using default base parameters")

    # Apply --set overrides (e.g. --set w_vv=0,w_sp=0)
    if args.set_params:
        from dataclasses import replace
        overrides = parse_set_params(args.set_params)
        allowed = {f.name for f in fields(CircuitParams)}
        for name in overrides:
            if name not in allowed:
                print(f"Warning: '{name}' is not a valid parameter, skipping.")
        clean = {k: v for k, v in overrides.items() if k in allowed}
        if clean:
            base = replace(base, **clean)
            print(f"Overrides applied: {', '.join(f'{k}={v}' for k, v in clean.items())}")

    bounds = default_bounds(base)
    freeze = parse_freeze_list(args.freeze)

    # Print parameter status
    if args.show_params:
        print_parameter_status(bounds, freeze, base)

    # Build fit config
    fit_cfg = FitConfig(
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        burn_in_ms=args.burn_in_ms,
        window_ms=args.window_ms,
        n_trials=args.n_trials,
        init_rate_scale=args.init_rate_scale,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        max_rate=args.max_rate,
        ko_min_effect_penalty=args.ko_min_effect_penalty,
        ko_wrong_direction_penalty=args.ko_wrong_direction_penalty,
    )

    # Print targets
    unit = args.unit
    print("\nOptimization targets:")
    print(f"  PYR: {target.mean_r_pyr} {unit}")
    print(f"  SOM: {target.mean_r_som} {unit}")
    print(f"  PV:  {target.mean_r_pv} {unit}")
    print(f"  VIP: {target.mean_r_vip} {unit}")
    if target.alpha7_ko_pyr is not None:
        print(f"  alpha7 KO PYR: {target.alpha7_ko_pyr} {unit}")
    if target.alpha5_ko_pyr is not None:
        print(f"  alpha5 KO PYR: {target.alpha5_ko_pyr} {unit}")
    if target.beta2_ko_pyr is not None:
        print(f"  beta2 KO PYR: {target.beta2_ko_pyr} {unit}")
    print()

    # Run optimization
    best = nevergrad_optimize(
        target,
        base=base,
        bounds=bounds,
        fit_cfg=fit_cfg,
        n_samples=args.n_samples,
        top_k=args.top_k,
        seed=args.seed if args.seed is not None else 0,
        freeze=freeze,
        early_stop_loss=args.early_stop_loss,
        log_file=args.log_file or None,
        log_interval=args.log_interval,
        n_workers=args.n_workers,
        save_best_json=args.save_best_json or None,
    )

    if not best:
        raise RuntimeError("Optimization returned no candidates.")

    # Print results
    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    for i, c in enumerate(best, start=1):
        pyr, som, pv, vip = c.means.tolist()
        ko_str = ""
        if c.ko_means.alpha7_ko is not None:
            ko_str += f" a7KO_pyr={c.ko_means.alpha7_ko[0]:.4g}"
        if c.ko_means.alpha5_ko is not None:
            ko_str += f" a5KO_pyr={c.ko_means.alpha5_ko[0]:.4g}"
        if c.ko_means.beta2_ko is not None:
            ko_str += f" b2KO_pyr={c.ko_means.beta2_ko[0]:.4g}"
        print(
            f"rank {i:02d}: loss={c.loss:.3e} "
            f"means=[pyr={pyr:.4g}, som={som:.4g}, pv={pv:.4g}, vip={vip:.4g}]"
            f"{ko_str}"
        )

    print("\nBest parameter set:\n")
    print(format_params_as_code(best[0].params))

    if args.save_best_json:
        print(f"\nBest params saved to: {args.save_best_json}")


def main() -> None:
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="PFC Circuit Model: 4-population rate model with parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run simulation with default parameters and plot
    python -m circuit_model run

    # Run with custom parameters
    python -m circuit_model run --params_json my_params.json --T_ms 5000

    # Run with noise
    python -m circuit_model run --noise_type ou --tau_noise_ms 10

    # Optimize parameters to match target rates
    python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8

    # Optimize with frozen parameters
    python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8 \\
        --freeze "tau_s,g_gaba_base" --show_params

    # Run batch study across conditions
    python -m circuit_model study --n_runs 100 --noise_type white --tau_noise_ms 5

    # Ring attractor: single condition
    python -m circuit_model ring-run --condition WT --amplitude 3

    # Ring attractor: compare conditions
    python -m circuit_model ring-study --conditions WT WT_APP --n_trials 10

    # Ring attractor: multi-amplitude study
    python -m circuit_model ring-study --amplitudes 8 10 15 20 --conditions WT WT_APP
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # =========================================================================
    # RUN subcommand
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run simulation and plot results",
        description="Run a simulation with given parameters and visualize the results."
    )
    add_simulation_args(run_parser)
    run_parser.add_argument("--burn_in_ms", type=float, default=500.0,
                            help="Burn-in period for statistics (ms)")
    run_parser.add_argument("--time_range", type=str, default="",
                            help="Time range to plot: 'start,end' in ms (e.g., '1000,2000')")
    run_parser.add_argument("--save_plot", type=str, default="",
                            help="Save plot to file (e.g., 'output.png')")
    run_parser.add_argument("--no_show", action="store_true",
                            help="Don't display the plot (useful for batch processing)")

    # Transient current options
    run_parser.add_argument("--enable_transient", action="store_true",
                            help="Enable time-dependent transient current (applied only during transient window)")
    run_parser.add_argument("--trans_start_ms", type=float, default=1000.0,
                            help="Time when transient starts (ms), default=1000")
    run_parser.add_argument("--trans_duration_ms", type=float, default=500.0,
                            help="Duration of transient pulse (ms), default=500")
    run_parser.add_argument("--trans_factor", type=float, default=0.2,
                            help="Transient as fraction of each population's I0, default=0.2")
    run_parser.add_argument("--unit", type=str, default="transients/min",
                            choices=["transients/min", "Hz"],
                            help="Rate unit for display and plots (default: transients/min)")

    # =========================================================================
    # OPTIMIZE subcommand
    # =========================================================================
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Optimize parameters to match target rates",
        description="Run Nevergrad optimization to find parameters matching target firing rates."
    )

    # Unit selection
    opt_parser.add_argument("--unit", type=str, default="transients/min",
                            choices=["transients/min", "Hz"],
                            help="Rate unit for display (default: transients/min)")

    # Target firing rates (required)
    opt_parser.add_argument("--target_pyr", type=float, required=True,
                            help="Target mean firing rate for PYR")
    opt_parser.add_argument("--target_som", type=float, required=True,
                            help="Target mean firing rate for SOM")
    opt_parser.add_argument("--target_pv", type=float, required=True,
                            help="Target mean firing rate for PV")
    opt_parser.add_argument("--target_vip", type=float, required=True,
                            help="Target mean firing rate for VIP")

    # Optional knockout targets
    opt_parser.add_argument("--target_alpha7_ko_pyr", type=float, default=None,
                            help="Target PYR rate under alpha7 knockout")
    opt_parser.add_argument("--target_alpha5_ko_pyr", type=float, default=None,
                            help="Target PYR rate under alpha5 knockout")
    opt_parser.add_argument("--target_beta2_ko_pyr", type=float, default=None,
                            help="Target PYR rate under beta2 knockout")

    # Optimization settings
    opt_parser.add_argument("--n_samples", type=int, default=5000,
                            help="Number of optimization samples")
    opt_parser.add_argument("--top_k", type=int, default=10,
                            help="Keep top K candidates")
    opt_parser.add_argument("--early_stop_loss", type=float, default=1e-4,
                            help="Stop if loss falls below this value")

    # Simulation settings
    add_simulation_args(opt_parser)
    opt_parser.add_argument("--burn_in_ms", type=float, default=1800.0,
                            help="Burn-in period (ms)")
    opt_parser.add_argument("--window_ms", type=float, default=500.0,
                            help="Averaging window (ms)")
    opt_parser.add_argument("--n_trials", type=int, default=8,
                            help="Trials per parameter set")
    opt_parser.add_argument("--init_rate_scale", type=float, default=0.2,
                            help="Scale for random initial conditions")
    opt_parser.add_argument("--max_rate", type=float, default=200.0,
                            help="Maximum allowed rate (stability check)")

    # KO penalty settings
    opt_parser.add_argument("--ko_min_effect_penalty", type=float, default=5.0,
                            help="Penalty weight for weak KO effect")
    opt_parser.add_argument("--ko_wrong_direction_penalty", type=float, default=10.0,
                            help="Penalty weight for wrong direction KO effect")

    # Parameter control
    opt_parser.add_argument("--freeze", type=str, default="",
                            help="Comma-separated parameter names to freeze")
    opt_parser.add_argument("--set", dest="set_params", type=str, default="",
                            help="Override parameter values: 'name=val,name=val' (e.g. --set w_vv=0,w_sp=0)")
    opt_parser.add_argument("--show_params", action="store_true",
                            help="Show which parameters are free vs frozen")

    # I/O settings
    opt_parser.add_argument("--save_best_json", type=str, default="best_params.json",
                            help="Save best parameters to JSON file")
    opt_parser.add_argument("--log_file", type=str, default="results_log.jsonl",
                            help="Log results to JSONL file")
    opt_parser.add_argument("--log_interval", type=int, default=50,
                            help="Log every N steps")
    opt_parser.add_argument("--n_workers", type=int, default=None,
                            help="Parallel workers (auto if None)")

    # =========================================================================
    # STUDY subcommand
    # =========================================================================
    study_parser = subparsers.add_parser(
        "study",
        help="Run batch study across experimental conditions",
        description="Run simulations across 8 conditions (WT, APP, KO variants) "
                    "and generate box plots of firing rate distributions."
    )

    # Study-specific arguments
    study_parser.add_argument("--n_runs", type=int, default=50,
                              help="Number of simulations per condition (default: 50)")
    study_parser.add_argument("--save_plot", type=str, default="",
                              help="Save box plot to file (e.g., 'study_results.png')")
    study_parser.add_argument("--no_show", action="store_true",
                              help="Don't display the plot")

    # Simulation parameters
    study_parser.add_argument("--T_ms", type=float, default=2500.0,
                              help="Simulation duration (ms)")
    study_parser.add_argument("--dt_ms", type=float, default=0.1,
                              help="Integration time step (ms)")
    study_parser.add_argument("--noise_type", choices=["none", "white", "ou"], default="white",
                              help="Noise type (default: white)")
    study_parser.add_argument("--sigma_noise", type=float, default=None,
                              help="Noise amplitude sigma_s (overrides params_json value)")
    study_parser.add_argument("--tau_noise_ms", type=float, default=5.0,
                              help="OU noise time constant (ms)")
    study_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed for reproducibility")
    study_parser.add_argument("--params_json", type=str, default="",
                              help="Load base parameters from JSON file")

    # Receptor activation mode
    study_parser.add_argument("--fixed_receptor_values", action="store_true",
                              help="Use fixed mean receptor values instead of sampling "
                                   "from distributions (default: sample from distributions)")

    # Statistics parameters
    study_parser.add_argument("--burn_in_ms", type=float, default=1800.0,
                              help="Burn-in period for statistics (ms)")
    study_parser.add_argument("--window_ms", type=float, default=500.0,
                              help="Averaging window (ms)")

    # Parallel processing
    study_parser.add_argument("--n_workers", type=int, default=None,
                              help="Parallel workers (auto if None)")

    # Display options
    study_parser.add_argument("--unit", type=str, default="transients/min",
                              choices=["transients/min", "Hz"],
                              help="Rate unit for display (default: transients/min)")

    # =========================================================================
    # RING-RUN subcommand
    # =========================================================================
    ring_run_parser = subparsers.add_parser(
        "ring-run",
        help="Run ring attractor simulation for a single condition",
        description="Run ring attractor simulation with a single experimental "
                    "condition and visualize results.",
    )
    from .ring.cli import add_common_args as _add_ring_common
    _add_ring_common(ring_run_parser)
    ring_run_parser.add_argument(
        "--condition", type=str, default="WT",
        help="Experimental condition (default: WT). "
             "Valid: WT, WT_APP, a5_KO, a5_KO_APP, a7_KO, a7_KO_APP, b2_KO, b2_KO_APP",
    )

    # =========================================================================
    # RING-STUDY subcommand
    # =========================================================================
    ring_study_parser = subparsers.add_parser(
        "ring-study",
        help="Run ring attractor study across conditions",
        description="Run ring attractor simulation across multiple experimental "
                    "conditions and generate comparison plots.",
    )
    _add_ring_common(ring_study_parser)
    ring_study_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to simulate (default: all 8). "
             "Valid: WT, WT_APP, a5_KO, a5_KO_APP, a7_KO, a7_KO_APP, b2_KO, b2_KO_APP",
    )
    ring_study_parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=None,
        help="Stimulus amplitude factors (multiples of I_ext_pyr). "
             "E.g. --amplitudes 8 10 15 20 means 8×, 10×, 15×, 20× baseline.",
    )
    ring_study_parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of trials per condition x amplitude (default: 100)",
    )
    ring_study_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    ring_study_parser.add_argument(
        "--delay_step_ms", type=float, default=None,
        help="Delay evaluation step size in ms (default: use [1s,2s,3s])",
    )
    ring_study_parser.add_argument(
        "--no_cache", action="store_true",
        help="Ignore existing CSV cache and recompute all conditions",
    )
    ring_study_parser.add_argument(
        "--amp_eval_step_ms", type=float, default=500.0,
        help="Step for timed metrics-vs-amplitude plots (ms). "
             "0 = disabled. (default: 500)",
    )
    ring_study_parser.add_argument(
        "--error_band", type=str, default="sem", choices=["sem", "sd"],
        help="Error band type for plots: 'sem' (default) or 'sd'.",
    )

    # =========================================================================
    # RING-DIFFUSION subcommand
    # =========================================================================
    ring_diff_parser = subparsers.add_parser(
        "ring-diffusion",
        help="Run MSD diffusion analysis on the ring attractor",
        description="Compute mean squared displacement (MSD) of bump center "
                    "during delay periods across conditions, and extract the "
                    "diffusion strength B_hat (Seeholzer et al. 2019).",
    )
    _add_ring_common(ring_diff_parser)
    ring_diff_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to simulate (default: all 8).",
    )
    ring_diff_parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of trials per condition (default: 50)",
    )
    ring_diff_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    ring_diff_parser.add_argument(
        "--error_band", type=str, default="sem", choices=["sem", "sd"],
        help="Error band type for plots: 'sem' (default) or 'sd'.",
    )
    ring_diff_parser.add_argument(
        "--filter_cutoff_hz", type=float, default=None,
        help="Low-pass filter cutoff (Hz) applied to bump center trajectory before MSD "
             "computation. If not set, the cutoff is auto-detected from the bump amplitude "
             "oscillation spectrum (0.4 × dominant oscillation frequency). "
             "Set to 0 to disable filtering entirely.",
    )

    # =========================================================================
    # RING-DRIFT-FIELD subcommand
    # =========================================================================
    ring_drift_parser = subparsers.add_parser(
        "ring-drift-field",
        help="Run distractor drift field analysis on the ring attractor",
        description="Sweep distractor angular offsets and measure bump "
                    "displacement to estimate the drift field A_hat(Δφ) "
                    "(Seeholzer et al. 2019).",
    )
    _add_ring_common(ring_drift_parser)
    ring_drift_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to simulate (default: all 8).",
    )
    ring_drift_parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of trials per condition per offset (default: 50)",
    )
    ring_drift_parser.add_argument(
        "--distractor_steps", type=float, default=10.0,
        help="Angular step size for distractor sweep in degrees (default: 10)",
    )
    ring_drift_parser.add_argument(
        "--distractor_amplitude", type=float, default=15.0,
        help="Distractor stimulus amplitude as factor of I_ext_pyr baseline "
             "(default: 15.0, i.e. 15× baseline current)",
    )
    ring_drift_parser.add_argument(
        "--distractor_duration_ms", type=float, default=200.0,
        help="Distractor duration in ms (default: 200)",
    )
    ring_drift_parser.add_argument(
        "--distractor_onset_ms", type=float, default=1500.0,
        help="Distractor onset after stimulus offset in ms (default: 1500)",
    )
    ring_drift_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    ring_drift_parser.add_argument(
        "--error_band", type=str, default="sem", choices=["sem", "sd"],
        help="Error band type for plots: 'sem' (default) or 'sd'.",
    )

    # =========================================================================
    # RING-DISTRACTOR-SWEEP subcommand
    # =========================================================================
    ring_ds_parser = subparsers.add_parser(
        "ring-distractor-sweep",
        help="2-D distractor sweep (Δφ × amplitude) on the ring attractor",
        description="Sweep a 2-D grid of distractor angular offset × distractor "
                    "amplitude, measuring bump drift and collapse probability. "
                    "Protocol: cue → delay1 → distractor → delay2.",
    )
    _add_ring_common(ring_ds_parser)
    ring_ds_parser.add_argument(
        "--condition", type=str, default="WT",
        help="Experimental condition (default: WT).",
    )
    ring_ds_parser.add_argument(
        "--offsets_deg", type=float, nargs="+",
        default=[0, 5, 10, 15, 20, 30, 40, 60, 80, 100, 130, 150, 180],
        help="Distractor angular offsets from cue in degrees (default: 0 5 10 15 20 30 40 60 80 100 130 150 180)",
    )
    ring_ds_parser.add_argument(
        "--amp_factors", type=float, nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Distractor amplitude factors relative to cue (default: 0.5 0.75 1.0 1.25 1.5)",
    )
    ring_ds_parser.add_argument(
        "--n_trials", type=int, default=10,
        help="Number of trials per grid cell (default: 10)",
    )
    ring_ds_parser.add_argument(
        "--delay1_ms", type=float, default=1000.0,
        help="Delay period before distractor in ms (default: 1000)",
    )
    ring_ds_parser.add_argument(
        "--delay2_ms", type=float, default=1000.0,
        help="Delay period after distractor in ms (default: 1000)",
    )
    ring_ds_parser.add_argument(
        "--distractor_duration_ms", type=float, default=250.0,
        help="Distractor duration in ms (default: 250)",
    )
    ring_ds_parser.add_argument(
        "--collapse_threshold", type=float, default=None,
        help="Pop-vector amplitude Â below which bump is declared collapsed. "
             "Auto-detected from calibration_summary.csv when not specified "
             "(run ring-calibrate first). Falls back to 0.2 with a warning if "
             "no calibration data is found.",
    )
    ring_ds_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )

    # =========================================================================
    # RING-CALIBRATE subcommand
    # =========================================================================
    ring_cal_parser = subparsers.add_parser(
        "ring-calibrate",
        help="Run 2D parameter calibration (amplitude x w_inter) for the ring attractor",
        description="Sweep a 2D grid of (stimulus_amplitude, w_pyr_pyr_inter) to find "
                    "parameter combinations that produce a stable memory bump. Estimates "
                    "a noise floor from no-stimulus baseline trials and outputs diagnostic "
                    "figures and a JSON summary with recommended parameters.",
    )
    _add_ring_common(ring_cal_parser)
    ring_cal_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to calibrate (default: WT only).",
    )
    ring_cal_parser.add_argument(
        "--amplitudes", type=float, nargs="+",
        default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help="Stimulus amplitude factors to sweep (default: 5 10 15 20 25 30)",
    )
    ring_cal_parser.add_argument(
        "--w_inter_values", type=float, nargs="+",
        default=[2.0, 3.0, 4, 5.0, 6.0],
        help="w_pyr_pyr_inter values to sweep (default: 2.0 3.0 4 5.0 6.0)",
    )
    ring_cal_parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of trials per grid point (default: 50)",
    )
    ring_cal_parser.add_argument(
        "--n_baseline", type=int, default=100,
        help="Number of no-stimulus baseline trials per w_inter for noise floor (default: 100)",
    )
    ring_cal_parser.add_argument(
        "--noise_percentile", type=float, default=95.0,
        help="Percentile of baseline A_hat used as noise floor threshold (default: 95)",
    )
    ring_cal_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    ring_cal_parser.add_argument(
        "--error_band", type=str, default="sem", choices=["sem", "sd"],
        help="Error band type for plots: 'sem' (default) or 'sd'.",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nNo command specified. Use 'run', 'optimize', 'study', "
              "'ring-run', 'ring-study', 'ring-diffusion', 'ring-drift-field', "
              "'ring-distractor-sweep', or 'ring-calibrate'.")
        sys.exit(1)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "study":
        cmd_study(args)
    elif args.command == "ring-run":
        from .ring.cli import cmd_run as cmd_ring_run
        cmd_ring_run(args)
    elif args.command == "ring-study":
        from .ring.cli import cmd_study as cmd_ring_study
        cmd_ring_study(args)
    elif args.command == "ring-diffusion":
        from .ring.cli import cmd_diffusion as cmd_ring_diffusion
        cmd_ring_diffusion(args)
    elif args.command == "ring-drift-field":
        from .ring.cli import cmd_drift_field as cmd_ring_drift_field
        cmd_ring_drift_field(args)
    elif args.command == "ring-distractor-sweep":
        from .ring.cli import cmd_distractor_sweep as cmd_ring_distractor_sweep
        cmd_ring_distractor_sweep(args)
    elif args.command == "ring-calibrate":
        from .ring.cli import cmd_calibrate as cmd_ring_calibrate
        cmd_ring_calibrate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
