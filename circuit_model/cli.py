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

import numpy as np

from .params import CircuitParams, ParamBound, default_bounds
from .loss import TargetRates, FitConfig
from .io import load_params_json, save_params_json, save_fit_summary_txt, format_params_as_code, build_fit_comparison, output_dir as _output_dir
from .optimization import nevergrad_optimize, evaluate_params, KOMeans, LossBreakdown, optimize_drug_activations, STAGE2_FREE_FIELDS
from .loss import DrugTarget
from .simulation import simulate_circuit
from .jacobian import print_sanity_check, compute_jacobian
from .defaults import DEFAULT_WT_PARAMS_PATH, DEFAULT_APP_PARAMS_PATH

# Hardcoded fallback initialization used when params/fit_init.json is unavailable.
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
    "act_alpha7_pv": 1.0,
    "act_alpha7_som": 1.0,
    "act_alpha7_ndnf": 1.0,
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


def _load_params_with_optional_condition(
    *,
    params_json: str,
    condition_key: str | None,
    context: str,
) -> tuple[CircuitParams, str]:
    """Load CircuitParams, optionally applying a study condition preset."""
    if condition_key:
        from .study import STUDY_CONDITIONS, apply_condition

        cond = STUDY_CONDITIONS[condition_key]

        if params_json:
            base = load_params_json(params_json)
            params = apply_condition(base, cond, app_params=None)
            msg = f"Loaded parameters from: {params_json} + applied condition: {condition_key}"
            return params, msg

        if DEFAULT_WT_PARAMS_PATH.exists():
            base = load_params_json(str(DEFAULT_WT_PARAMS_PATH))
            app_params = None
            if cond.is_app and DEFAULT_APP_PARAMS_PATH.exists():
                app_params = load_params_json(str(DEFAULT_APP_PARAMS_PATH))
            params = apply_condition(base, cond, app_params=app_params)
            if cond.is_app and app_params is None:
                msg = (
                    f"Loaded WT defaults from: {DEFAULT_WT_PARAMS_PATH} + applied condition: {condition_key} "
                    f"(WT_APP defaults not found at {DEFAULT_APP_PARAMS_PATH})"
                )
            else:
                msg = f"Loaded default project condition: {condition_key}"
            return params, msg

        params = apply_condition(_default_fit_init_params(), cond, app_params=None)
        msg = f"Using hardcoded fit-init defaults + applied condition: {condition_key}"
        return params, msg

    if params_json:
        return load_params_json(params_json), f"Loaded parameters from: {params_json}"

    if DEFAULT_WT_PARAMS_PATH.exists():
        return load_params_json(str(DEFAULT_WT_PARAMS_PATH)), f"Loaded default project parameters from: {DEFAULT_WT_PARAMS_PATH}"

    if context == "plot-transfer":
        return _default_fit_init_params(), "Using hardcoded fit-init default parameters"

    return _default_fit_init_params(), "Using hardcoded fit-init default parameters"


def print_comparison_table(
    means: np.ndarray,
    ko_means: KOMeans,
    target: TargetRates,
    loss: float,
) -> None:
    """Print actual vs target comparison table for all conditions and populations."""
    pops = ["PYR", "SOM", "PV ", "VIP", "NDNF"]
    tgt_arr = target.as_array()

    print("\n" + "=" * 62)
    print("  FITTING COMPARISON  (actual vs target)")
    print(f"  Total loss: {loss:.4g}")
    print("=" * 62)
    print(f"  {'Condition':<14}  {'Pop':<4}  {'Actual':>8}  {'Target':>8}  {'Error':>7}")
    print("  " + "-" * 55)

    for i, pop in enumerate(pops):
        actual = float(means[i])
        tgt = float(tgt_arr[i])
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print(f"  {'base':<14}  {pop:<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")

    if target.alpha7_ko_pyr is not None and ko_means.alpha7_ko is not None:
        actual = float(ko_means.alpha7_ko[0])
        tgt = target.alpha7_ko_pyr
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print("  " + "-" * 55)
        print(f"  {'alpha7_ko':<14}  {'PYR':<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")

    if target.alpha5_ko_pyr is not None and ko_means.alpha5_ko is not None:
        actual = float(ko_means.alpha5_ko[0])
        tgt = target.alpha5_ko_pyr
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print(f"  {'alpha5_ko':<14}  {'PYR':<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")

    if target.beta2_ko_pyr is not None and ko_means.beta2_ko is not None:
        actual = float(ko_means.beta2_ko[0])
        tgt = target.beta2_ko_pyr
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print(f"  {'beta2_ko':<14}  {'PYR':<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")

    if target.alpha7_ndnf_ko_ndnf is not None and ko_means.alpha7_ndnf_ko is not None:
        actual = float(ko_means.alpha7_ndnf_ko[4])
        tgt = target.alpha7_ndnf_ko_ndnf
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print(f"  {'a7_ndnf_ko':<14}  {'NDNF':<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")
    if target.alpha7_pv_ko_pv is not None and ko_means.alpha7_pv_ko is not None:
        actual = float(ko_means.alpha7_pv_ko[2])
        tgt = target.alpha7_pv_ko_pv
        err = 100.0 * (actual - tgt) / max(abs(tgt), 1e-6)
        print(f"  {'a7_pv_ko':<14}  {'PV':<4}  {actual:8.3f}  {tgt:8.3f}  {err:+6.1f}%")

    print("=" * 62 + "\n")


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
        "Time constants": ["tau_s", "tau_adapt_pyr"],
        "Adaptation": ["J_adapt_pyr"],
        "Noise & GABA": ["sigma_noise", "g_gaba_base", "g_alpha7"],
        "Weights (excitatory)": ["J_NMDA", "w_ep", "w_es", "w_ev"],
        "Weights (inhibitory)": ["w_pe", "w_pp", "w_se", "w_sp", "w_vp", "w_vs",
                                  "w_sn", "w_ne", "w_np", "w_nv"],
        "External currents": ["I0_pyr", "I0_pv", "I_alpha7_pv", "I0_som", "I_alpha7_som", "I_beta2_som",
                                "I0_vip", "I_alpha5_vip", "I0_ndnf", "I_alpha7_ndnf", "I_beta2_ndnf"],
        "Transient": ["trans_factor"],
        "Transfer function": ["Theta_pyr", "alpha_pyr", "Theta_pv", "alpha_pv", "Theta_som", "alpha_som",
                                "Theta_vip", "alpha_vip", "Theta_ndnf", "alpha_ndnf", "g"],
        "Receptor activation": ["act_alpha7_pv", "act_alpha7_som", "act_alpha7_ndnf",
                                  "act_beta2", "act_alpha5"],
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


def _print_opt_init_summary(params: CircuitParams, means: np.ndarray, breakdown: "LossBreakdown") -> None:  # type: ignore
    """Print effective optimization initialization and its predicted rates."""
    print("Initial condition (effective after --set/--no_adapt):")
    print(f"  I0: pyr={params.I0_pyr:.6g}, som={params.I0_som:.6g}, pv={params.I0_pv:.6g}, vip={params.I0_vip:.6g}")
    print(f"  W:  J_NMDA={params.J_NMDA:.6g}, w_ep={params.w_ep:.6g}, w_es={params.w_es:.6g}, w_ev={params.w_ev:.6g}")
    print(f"      w_pe={params.w_pe:.6g}, w_pp={params.w_pp:.6g}, w_se={params.w_se:.6g}, w_sp={params.w_sp:.6g}, w_vp={params.w_vp:.6g}, w_vs={params.w_vs:.6g}")
    print(f"      w_sn={params.w_sn:.6g}, w_ne={params.w_ne:.6g}, w_np={params.w_np:.6g}, w_nv={params.w_nv:.6g}")
    print(f"  Transfer: tau_s={params.tau_s:.6g}, alpha_pyr={params.alpha_pyr:.6g}, alpha_som={params.alpha_som:.6g}, alpha_pv={params.alpha_pv:.6g}, alpha_vip={params.alpha_vip:.6g}")
    print(f"            Theta_pyr={params.Theta_pyr:.6g}, Theta_som={params.Theta_som:.6g}, Theta_pv={params.Theta_pv:.6g}, Theta_vip={params.Theta_vip:.6g}")
    print("Initial predicted rates (Hz):")
    print(f"  PYR={means[0]:.4f}, SOM={means[1]:.4f}, PV={means[2]:.4f}, VIP={means[3]:.4f}, NDNF={means[4]:.4f}")
    print(f"  Initial {breakdown}")


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
    parser.add_argument(
        "--condition",
        type=str,
        default="",
        choices=["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP", "APP_sim"],
        help=(
            "Apply an experimental condition preset. If --params_json is not provided, "
            "the command auto-loads default project WT/WT_APP fitted files when available."
        ),
    )


def cmd_plot_transfer(args: argparse.Namespace) -> None:
    """Plot transfer functions for all 4 populations."""
    from dataclasses import replace
    from .plotting import plot_transfer_functions

    condition_key = args.condition if getattr(args, "condition", "") else None
    params, load_msg = _load_params_with_optional_condition(
        params_json=args.params_json,
        condition_key=condition_key,
        context="plot-transfer",
    )
    print(load_msg)

    if args.set_params:
        overrides = parse_set_params(args.set_params)
        allowed = {f.name for f in fields(CircuitParams)}
        clean = {k: v for k, v in overrides.items() if k in allowed}
        params = replace(params, **clean)

    if args.save_plot:
        save_path = args.save_plot
    elif args.params_json:
        from pathlib import Path as _Path
        stem = _Path(args.params_json).stem
        save_path = f"figs/optim/transfer_functions_{stem}.png"
    elif condition_key:
        save_path = f"figs/optim/transfer_functions_{condition_key}.png"
    else:
        save_path = "figs/optim/transfer_functions.png"

    plot_transfer_functions(
        params,
        I_range=(args.I_min, args.I_max),
        save_path=save_path,
        show=not args.no_show,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run a simulation and plot the results."""
    from dataclasses import replace
    from .plotting import plot_simulation_dashboard, print_simulation_summary

    condition_key = args.condition if getattr(args, "condition", "") else None
    params, load_msg = _load_params_with_optional_condition(
        params_json=args.params_json,
        condition_key=condition_key,
        context="run",
    )
    print(load_msg)

    # Apply sigma_noise override if provided
    if getattr(args, "sigma_noise", None) is not None:
        from dataclasses import replace as _replace
        params = _replace(params, sigma_noise=args.sigma_noise)
        print(f"sigma_noise overridden to: {args.sigma_noise}")

    # Apply transient settings if enabled
    use_transient = args.enable_transient or getattr(args, "enable_trans2", False)
    if args.enable_transient:
        params = replace(
            params,
            trans_enabled=True,
            trans_start_ms=args.trans_start_ms,
            trans_duration_ms=args.trans_duration_ms,
            trans_factor=args.trans_factor,
        )
    if getattr(args, "enable_trans2", False):
        params = replace(
            params,
            trans2_enabled=True,
            trans2_start_ms=args.trans2_start_ms,
            trans2_duration_ms=args.trans2_duration_ms,
            trans2_factor=args.trans2_factor,
        )

    # Print key parameter values
    print("\nKey parameters:")
    print(f"  tau_s = {params.tau_s:.2f} ms")
    print(f"  sigma_noise = {params.sigma_noise:.4f} (noise ratio, effective {params.sigma_noise * params.I_ext_pyr():.4f} nA)")
    print(f"  g_gaba = {params.g_gaba():.2f} (GABA scaling)")

    if args.enable_transient:
        trans_end = params.trans_start_ms + params.trans_duration_ms
        print(f"\nTransient 1 (excitatory push → high state):")
        print(f"  trans_factor = {params.trans_factor:.2f} (fraction of PYR I0)")
        print(f"  Window: {params.trans_start_ms:.1f} - {trans_end:.1f} ms")
    if getattr(args, "enable_trans2", False):
        trans2_end = params.trans2_start_ms + params.trans2_duration_ms
        print(f"\nTransient 2 (return to resting state):")
        print(f"  trans2_factor = {params.trans2_factor:.2f} (fraction of I0)")
        print(f"  Window: {params.trans2_start_ms:.1f} - {trans2_end:.1f} ms")

    # Parse initial rates
    r0 = None
    if getattr(args, "r0", ""):
        parts = [float(x) for x in args.r0.split(",")]
        if len(parts) != 4:
            raise ValueError("--r0 must have exactly 4 values: pyr,som,pv,vip")
        r0 = np.array(parts, dtype=float)
        print(f"  r0 = PYR={r0[0]:.1f}, SOM={r0[1]:.1f}, PV={r0[2]:.1f}, VIP={r0[3]:.1f} Hz")

    # Run simulation
    print(f"\nRunning simulation: T={args.T_ms} ms, dt={args.dt_ms} ms, noise={args.noise_type}")

    result = simulate_circuit(
        params,
        T_ms=args.T_ms,
        dt_ms=args.dt_ms,
        r0=r0,
        seed=args.seed,
        noise_type=args.noise_type,
        tau_noise_ms=args.tau_noise_ms,
        use_transient=use_transient,
    )

    # Print summary
    burn_in = args.burn_in_ms if hasattr(args, "burn_in_ms") else args.T_ms * 0.5
    print_simulation_summary(result, burn_in_ms=burn_in, params=params)

    # Plot
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            time_range = (float(parts[0]), float(parts[1]))
        else:
            time_range = None
    else:
        # Skip burn-in by default to avoid the initial transient spike
        time_range = (burn_in, result.t_ms[-1])

    sigma_str = f"σ={params.sigma_noise:.3g}" if args.noise_type != "none" else ""
    title = f"Circuit Model Simulation (noise={args.noise_type}{', ' + sigma_str if sigma_str else ''})"
    if args.enable_transient:
        title += f" [T1: {params.trans_start_ms:.0f}-{params.trans_start_ms + params.trans_duration_ms:.0f} ms, ×{params.trans_factor:+.2f}]"
    if getattr(args, "enable_trans2", False):
        title += f" [T2: {params.trans2_start_ms:.0f}-{params.trans2_start_ms + params.trans2_duration_ms:.0f} ms, ×{params.trans2_factor:+.2f}]"

    # Determine save path
    noise_tag = f"{args.noise_type}_sigma{params.sigma_noise:.3g}" if args.noise_type != "none" else "none"
    if args.save_plot:
        save_path = args.save_plot
    else:
        out_dir = _output_dir("figs/single_node/runs")
        if condition_key:
            fname = f"circuit_simulation_{noise_tag}_{condition_key}.png"
        elif args.params_json:
            stem = Path(args.params_json).stem
            fname = f"circuit_simulation_{noise_tag}_{stem}.png"
        else:
            fname = f"circuit_simulation_{noise_tag}.png"
        save_path = os.path.join(out_dir, fname)

    plot_simulation_dashboard(
        result,
        title=title,
        time_range=time_range,
        save_path=save_path,
        show=not args.no_show,
        unit=args.unit,
        smooth_ms=getattr(args, "smooth_ms", 0.0),
    )

def cmd_study(args: argparse.Namespace) -> None:
    """Run batch study across experimental conditions and generate box plots."""
    from .study import (
        STUDY_CONDITIONS,
        StudyConfig,
        run_study,
        plot_study_boxplots,
    )

    # Load base parameters (prefer project default fit when params_json is omitted)
    if args.params_json:
        base_params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    elif DEFAULT_WT_PARAMS_PATH.exists():
        base_params = load_params_json(str(DEFAULT_WT_PARAMS_PATH))
        print(f"Loaded default project WT parameters from: {DEFAULT_WT_PARAMS_PATH}")
    else:
        base_params = _default_fit_init_params()
        print("Using hardcoded fit-init default parameters")

    # Load APP parameters (dual-params mode). Prefer explicit app_params_json,
    # otherwise use project default WT_APP fit when available.
    app_params = None
    if args.app_params_json:
        app_params = load_params_json(args.app_params_json)
        print(f"Loaded APP parameters from: {args.app_params_json}")
        print("  -> Dual-params mode: APP conditions use the APP fit directly.")
    elif DEFAULT_APP_PARAMS_PATH.exists():
        app_params = load_params_json(str(DEFAULT_APP_PARAMS_PATH))
        print(f"Loaded default project WT_APP parameters from: {DEFAULT_APP_PARAMS_PATH}")
        print("  -> Dual-params mode: APP conditions use the WT_APP fit directly.")

    # Override noise amplitude if provided
    if args.sigma_noise is not None:
        from dataclasses import replace
        base_params = replace(base_params, sigma_noise=args.sigma_noise)
        print(f"Noise amplitude overridden: sigma_noise = {args.sigma_noise}")

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
        noise_detail = f"white, sigma_noise={base_params.sigma_noise:.4f}"
    else:  # ou
        noise_detail = f"ou, sigma_noise={base_params.sigma_noise:.4f}, tau_noise={cfg.tau_noise_ms}ms"
    print(f"  Simulation: T={cfg.T_ms}ms, dt={cfg.dt_ms}ms, noise={noise_detail}")
    print(f"  Receptor activation: {'fixed mean values' if cfg.fixed_receptor_values else 'sampled from distributions'}")
    print(f"  Statistics: burn_in={cfg.burn_in_ms}ms, window={cfg.window_ms}ms")
    print()

    # Run study
    seed = args.seed if args.seed is not None else 0
    results = run_study(base_params, cfg, base_seed=seed, verbose=True, app_params=app_params)

    # Determine save path
    if args.save_plot:
        save_path = args.save_plot
    else:
        out_dir = _output_dir("figs/single_node/boxplot")
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


def cmd_optimize_receptors(args: argparse.Namespace) -> None:
    """Stage 2: per-drug fit of receptor activations.

    Loads a Stage-1 best_params.json (via --params_json) and, for each drug
    in --drugs, runs an independent nevergrad fit varying only the per-cell
    α7 activations, β2, and α5. All other circuit parameters stay frozen.
    """
    import json
    if not args.params_json:
        raise SystemExit("--stage receptors requires --params_json (a Stage-1 fit).")
    base = load_params_json(args.params_json)
    print(f"Stage 2: loaded base from {args.params_json}")

    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    drug_target_args = {
        "MLA":      (args.target_mla_ndnf,      args.target_mla_pv),
        "PNU":      (args.target_pnu_ndnf,      args.target_pnu_pv),
        "nicotine": (args.target_nicotine_ndnf, args.target_nicotine_pv),
    }
    drug_targets: list[DrugTarget] = []
    for d in drugs:
        if d not in drug_target_args:
            raise SystemExit(f"Unknown drug '{d}'. Known: {list(drug_target_args)}.")
        ndnf_hz, pv_hz = drug_target_args[d]
        if ndnf_hz is None or pv_hz is None:
            raise SystemExit(
                f"Drug '{d}' selected but its targets are not set "
                f"(need --target_{d.lower()}_ndnf and --target_{d.lower()}_pv)."
            )
        drug_targets.append(DrugTarget(drug=d, population="NDNF", target_hz=ndnf_hz))
        drug_targets.append(DrugTarget(drug=d, population="PV",   target_hz=pv_hz))

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
    )

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.save_best_json:
        out_dir = Path(args.save_best_json).parent
    else:
        out_dir = Path("stage2_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or str(out_dir / "stage2_log.jsonl")
    if Path(log_file).exists():
        Path(log_file).unlink()

    print(f"Drugs to fit: {drugs}")
    print(f"Total measurements: {len(drug_targets)} ({len(drugs)} drugs × 2 cell types)")
    print(f"Output dir: {out_dir}")
    print(f"NOTE: with only 2 measurements per drug and 5 free activations, "
          f"this fit is under-constrained — multiple activation tuples can produce "
          f"the same NDNF + PV rates. Bounds [0, 5] keep solutions physiological.")

    results = optimize_drug_activations(
        base, drug_targets, fit_cfg,
        n_samples=args.n_samples,
        optimizer=args.optimizer,
        seed=args.seed,
        log_file=log_file,
        log_interval=args.log_interval,
    )

    # Write summary JSON
    summary_path = out_dir / "stage2_results.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nStage 2 results saved to: {summary_path}")

    # Print pretty summary
    print("\n" + "=" * 70)
    print("STAGE 2 RESULTS")
    print("=" * 70)
    for drug, info in results.items():
        print(f"\n  {drug:>10s}  loss={info['loss']:.4g}")
        acts = info["activations"]
        for k in STAGE2_FREE_FIELDS:
            if k in acts:
                print(f"      {k:<18s} = {acts[k]:.4f}")
        for tgt in info["targets"]:
            err = 100.0 * (tgt["predicted_hz"] - tgt["target_hz"]) / max(abs(tgt["target_hz"]), 1e-6)
            print(f"      {tgt['population']:<5s} actual={tgt['predicted_hz']:8.3f}  target={tgt['target_hz']:8.3f}  err={err:+6.1f}%")
    print("=" * 70 + "\n")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run parameter optimization."""
    if args.stage == "receptors":
        cmd_optimize_receptors(args)
        return

    if not getattr(args, "resume", False):
        missing = [f"--target_{k}" for k, v in [
            ("pyr", args.target_pyr), ("som", args.target_som),
            ("pv", args.target_pv), ("vip", args.target_vip),
            ("ndnf", args.target_ndnf),
        ] if v is None]
        if missing:
            raise SystemExit(f"error: the following arguments are required: {', '.join(missing)}\n"
                             "(or use --resume to load targets from a previous log)")

    # Handle --resume: load best params and targets from log, continue from last logged step
    step_offset = 0
    append_log = False
    if getattr(args, "resume", False):
        resume_json = args.save_best_json or "best_params.json"
        log_path = args.log_file or "results_log.jsonl"
        if not os.path.exists(resume_json):
            raise FileNotFoundError(f"--resume: could not find '{resume_json}'")
        base = load_params_json(resume_json)
        print(f"Resuming from: {resume_json}")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"--resume: could not find log file '{log_path}'")
        import json as _json
        last_step = 0
        last_entry = None
        with open(log_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    last_entry = _json.loads(_line)
                    last_step = last_entry.get("step", last_step)
        if last_entry is None:
            raise ValueError(f"--resume: log file '{log_path}' has no entries")
        step_offset = last_step
        append_log = True
        print(f"Appending to log '{log_path}' from step {last_step}")
        t = last_entry["target"]
        target = TargetRates(
            mean_r_pyr=t["mean_r_pyr"],
            mean_r_som=t["mean_r_som"],
            mean_r_pv=t["mean_r_pv"],
            mean_r_vip=t["mean_r_vip"],
            mean_r_ndnf=t.get("mean_r_ndnf", 0.0),
            alpha7_ko_pyr=t.get("alpha7_ko_pyr"),
            alpha5_ko_pyr=t.get("alpha5_ko_pyr"),
            beta2_ko_pyr=t.get("beta2_ko_pyr"),
            alpha7_ndnf_ko_ndnf=t.get("alpha7_ndnf_ko_ndnf"),
            alpha7_pv_ko_pv=t.get("alpha7_pv_ko_pv"),
        )
        print(f"Targets loaded from log: pyr={target.mean_r_pyr}, som={target.mean_r_som}, "
              f"pv={target.mean_r_pv}, vip={target.mean_r_vip}, ndnf={target.mean_r_ndnf}")
    else:
        # Build target rates from CLI args
        target = TargetRates(
            mean_r_pyr=args.target_pyr,
            mean_r_som=args.target_som,
            mean_r_pv=args.target_pv,
            mean_r_vip=args.target_vip,
            mean_r_ndnf=args.target_ndnf if args.target_ndnf is not None else 0.0,
            alpha7_ko_pyr=args.target_alpha7_ko_pyr,
            alpha5_ko_pyr=args.target_alpha5_ko_pyr,
            beta2_ko_pyr=args.target_beta2_ko_pyr,
            alpha7_ndnf_ko_ndnf=args.target_alpha7_ndnf_ko_ndnf,
            alpha7_pv_ko_pv=args.target_alpha7_pv_ko_pv,
        )

    # Load or create base parameters (only if not already loaded via --resume)
    if not getattr(args, "resume", False):
        condition_key = args.condition if getattr(args, "condition", "") else None
        base, load_msg = _load_params_with_optional_condition(
            params_json=args.params_json,
            condition_key=condition_key,
            context="optimize",
        )
        print(load_msg)

    # Apply --set overrides (e.g. --set w_sp=0)
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

    # --no_adapt: zero and freeze adaptation strengths
    if args.no_adapt:
        from dataclasses import replace
        base = replace(base, J_adapt_pyr=0.0, J_adapt_som=0.0)
        print("--no_adapt: J_adapt_pyr=0, J_adapt_som=0 (frozen)")

    n_samples = args.n_samples

    bounds = default_bounds(base, w_hi=getattr(args, "w_hi", None))
    freeze = parse_freeze_list(args.freeze)
    if args.no_adapt:
        freeze |= {"J_adapt_pyr", "J_adapt_som"}
    # Stage 1: receptor activations are NOT free; only weights/currents are fit.
    freeze |= set(STAGE2_FREE_FIELDS)

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
    )

    # Print targets
    unit = args.unit
    print("\nOptimization targets:")
    print(f"  PYR:  {target.mean_r_pyr} {unit}")
    print(f"  SOM:  {target.mean_r_som} {unit}")
    print(f"  PV:   {target.mean_r_pv} {unit}")
    print(f"  VIP:  {target.mean_r_vip} {unit}")
    print(f"  NDNF: {target.mean_r_ndnf} {unit}")
    if target.alpha7_ko_pyr is not None:
        print(f"  alpha7 KO PYR: {target.alpha7_ko_pyr} {unit}")
    if target.alpha5_ko_pyr is not None:
        print(f"  alpha5 KO PYR: {target.alpha5_ko_pyr} {unit}")
    if target.beta2_ko_pyr is not None:
        print(f"  beta2 KO PYR:  {target.beta2_ko_pyr} {unit}")
    if target.alpha7_ndnf_ko_ndnf is not None:
        print(f"  NDNF-selective a7 KO (NDNF): {target.alpha7_ndnf_ko_ndnf} {unit}")
    if target.alpha7_pv_ko_pv is not None:
        print(f"  PV-selective   a7 KO (PV):   {target.alpha7_pv_ko_pv} {unit}")
    print()

    # --output_dir is the canonical "put everything here" flag.
    # Explicit --save_best_json / --log_file still override it.
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # If save_best_json is at its default ("best_params.json"), put it in out_dir.
        if args.save_best_json == "best_params.json":
            save_best_json_to_use = str(out_dir / "best_params.json")
        else:
            save_best_json_to_use = args.save_best_json
        # Same logic for log file
        log_file_to_use = args.log_file or str(out_dir / "log.jsonl")
    else:
        # No --output_dir: anchor on save_best_json's parent, fall back to cwd.
        out_dir = Path(args.save_best_json).parent if args.save_best_json else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_best_json_to_use = args.save_best_json
        log_file_to_use = args.log_file or str(out_dir / "log.jsonl")

    # Auto-pick a finer log interval when the default is in effect
    log_interval_to_use = args.log_interval if args.log_interval != 500 else 50
    if args.log_file:
        # User set --log_file explicitly → respect their --log_interval too
        log_interval_to_use = args.log_interval

    init_seed = args.seed if args.seed is not None else 0
    init_rng = np.random.default_rng(init_seed)
    init_loss, init_means, _, init_breakdown = evaluate_params(
        base,
        target,
        fit_cfg,
        rng=init_rng,
        weight_base=args.weight_base,
        weight_global_ko=args.weight_global_ko,
        weight_selective_ko=args.weight_selective_ko,
        weight_drug=args.weight_drug,
    )
    _print_opt_init_summary(base, init_means, init_breakdown)
    print()

    if log_file_to_use:
        print(f"Logging optimization progress to: {log_file_to_use}")
        print(f"Log interval: every {log_interval_to_use} steps\n")

    # Run optimization
    best = nevergrad_optimize(
        target,
        base=base,
        bounds=bounds,
        fit_cfg=fit_cfg,
        n_samples=n_samples,
        top_k=args.top_k,
        seed=args.seed if args.seed is not None else 0,
        optimizer=args.optimizer,
        freeze=freeze,
        log_file=log_file_to_use,
        log_interval=log_interval_to_use,
        save_best_json=save_best_json_to_use or None,
        step_offset=step_offset,
        append_log=append_log,
        weight_base=args.weight_base,
        weight_global_ko=args.weight_global_ko,
        weight_selective_ko=args.weight_selective_ko,
        weight_drug=args.weight_drug,
    )

    if not best:
        print("No optimization candidates available (run may have been interrupted very early).")
        return

    # Print results
    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    for i, c in enumerate(best, start=1):
        pyr, som, pv, vip, ndnf = c.means.tolist()
        ko_str = ""
        if c.ko_means.alpha7_ko is not None:
            ko_str += f" a7KO_pyr={c.ko_means.alpha7_ko[0]:.4g}"
        if c.ko_means.alpha5_ko is not None:
            ko_str += f" a5KO_pyr={c.ko_means.alpha5_ko[0]:.4g}"
        if c.ko_means.beta2_ko is not None:
            ko_str += f" b2KO_pyr={c.ko_means.beta2_ko[0]:.4g}"
        if c.ko_means.alpha7_ndnf_ko is not None:
            ko_str += f" a7_ndnf_KO_ndnf={c.ko_means.alpha7_ndnf_ko[4]:.4g}"
        if c.ko_means.alpha7_pv_ko is not None:
            ko_str += f" a7_pv_KO_pv={c.ko_means.alpha7_pv_ko[2]:.4g}"
        print(
            f"rank {i:02d}: loss={c.loss:.3e} "
            f"means=[pyr={pyr:.4g}, som={som:.4g}, pv={pv:.4g}, vip={vip:.4g}, ndnf={ndnf:.4g}]"
            f"{ko_str}"
        )

    print("\nBest parameter set:\n")
    print(format_params_as_code(best[0].params))

    # Jacobian sanity check at the best fitted steady state
    r_ss = best[0].means  # ndarray [pyr, som, pv, vip, ndnf]
    J = compute_jacobian(best[0].params, r_ss)
    print_sanity_check(best[0].params, r_ss)

    # Comparison table: actual vs target for all conditions and populations
    fit_meta = build_fit_comparison(best[0].means, best[0].ko_means, target, best[0].loss, jacobian=J)
    print_comparison_table(best[0].means, best[0].ko_means, target, best[0].loss)

    # Final save with metadata (overrides the incremental params-only saves done during optimization)
    if save_best_json_to_use:
        save_params_json(save_best_json_to_use, best[0].params, fit_meta=fit_meta)
        save_fit_summary_txt(save_best_json_to_use, fit_meta, params=best[0].params)
        print(f"Best params saved to: {save_best_json_to_use}")

    # Generate loss evolution plots
    if log_file_to_use:
        try:
            from .loss_evolution_plot import plot_loss_evolution, plot_loss_evolution_ratios
            log_dir = Path(log_file_to_use).parent
            plot_loss_evolution(log_file_to_use, output_dir=str(log_dir))
            plot_loss_evolution_ratios(log_file_to_use, output_dir=str(log_dir))
        except Exception as e:
            print(f"Warning: could not generate loss evolution plots: {e}")


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
    run_parser.add_argument("--sigma_noise", type=float, default=None,
                            help="Override sigma_noise from params (noise ratio, e.g. 0.1)")
    run_parser.add_argument("--burn_in_ms", type=float, default=500.0,
                            help="Burn-in period for statistics (ms)")
    run_parser.add_argument("--r0", type=str, default="",
                            help="Initial firing rates as 'pyr,som,pv,vip' in Hz (e.g. '80,2,3,7')")
    run_parser.add_argument("--time_range", type=str, default="",
                            help="Time range to plot: 'start,end' in ms (e.g., '1000,2000')")
    run_parser.add_argument("--save_plot", type=str, default="",
                            help="Save plot to file (e.g., 'output.png')")
    run_parser.add_argument("--save_metrics", type=str, default="",
                            help="Save pre/post-transient state metrics to JSON (e.g., 'run_metrics.json')")
    run_parser.add_argument("--no_show", action="store_true",
                            help="Don't display the plot (useful for batch processing)")

    # Transient current options (first transient: push to high state)
    run_parser.add_argument("--enable_transient", action="store_true",
                            help="Enable first transient current (applied only during transient window)")
    run_parser.add_argument("--trans_start_ms", type=float, default=1000.0,
                            help="Time when first transient starts (ms), default=1000")
    run_parser.add_argument("--trans_duration_ms", type=float, default=500.0,
                            help="Duration of first transient pulse (ms), default=500")
    run_parser.add_argument("--trans_factor", type=float, default=0.2,
                            help="First transient as fraction of PYR's I0 (PYR-only), default=0.2")
    # Second transient (e.g., inhibitory pulse to return to resting state)
    run_parser.add_argument("--enable_trans2", action="store_true",
                            help="Enable second transient current (e.g., negative factor to return to low state)")
    run_parser.add_argument("--trans2_start_ms", type=float, default=3000.0,
                            help="Time when second transient starts (ms), default=3000")
    run_parser.add_argument("--trans2_duration_ms", type=float, default=500.0,
                            help="Duration of second transient pulse (ms), default=500")
    run_parser.add_argument("--trans2_factor", type=float, default=-0.3,
                            help="Second transient as fraction of each population's I0, default=-0.3")
    run_parser.add_argument("--smooth_ms", type=float, default=0.0,
                            help="Boxcar smoothing window for firing rate plots in ms (0 = no smoothing, default: 0)")
    run_parser.add_argument("--unit", type=str, default="Hz",
                            choices=["Hz"],
                            help="Rate unit for display and plots (default: Hz)")

    # =========================================================================
    # OPTIMIZE subcommand
    # =========================================================================
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Optimize parameters to match target rates",
        description="Run Nevergrad optimization to find parameters matching target firing rates."
    )

    # Unit selection
    opt_parser.add_argument("--unit", type=str, default="Hz",
                            choices=["Hz"],
                            help="Rate unit for display (default: Hz)")

    # Target firing rates (required)
    opt_parser.add_argument("--target_pyr", type=float, default=None,
                            help="Target mean firing rate for PYR (not needed with --resume)")
    opt_parser.add_argument("--target_som", type=float, default=None,
                            help="Target mean firing rate for SOM (not needed with --resume)")
    opt_parser.add_argument("--target_pv", type=float, default=None,
                            help="Target mean firing rate for PV (not needed with --resume)")
    opt_parser.add_argument("--target_vip", type=float, default=None,
                            help="Target mean firing rate for VIP (not needed with --resume)")
    opt_parser.add_argument("--target_ndnf", type=float, default=None,
                            help="Target mean firing rate for NDNF (not needed with --resume)")

    # Optional knockout targets
    opt_parser.add_argument("--target_alpha7_ko_pyr", type=float, default=None,
                            help="Target PYR rate under global alpha7 knockout")
    opt_parser.add_argument("--target_alpha5_ko_pyr", type=float, default=None,
                            help="Target PYR rate under alpha5 knockout")
    opt_parser.add_argument("--target_beta2_ko_pyr", type=float, default=None,
                            help="Target PYR rate under beta2 knockout")
    opt_parser.add_argument("--target_alpha7_ndnf_ko_ndnf", type=float, default=None,
                            help="Target NDNF rate under NDNF-selective alpha7 KO "
                                 "(measured on NDNF itself, e.g. flx/flx baseline)")
    opt_parser.add_argument("--target_alpha7_pv_ko_pv", type=float, default=None,
                            help="Target PV rate under PV-selective alpha7 KO "
                                 "(measured on PV itself, e.g. a7flx/flx baseline)")

    # Optimization settings
    opt_parser.add_argument("--n_samples", type=int, default=5000,
                            help="Number of optimization samples")
    opt_parser.add_argument("--top_k", type=int, default=10,
                            help="Keep top K candidates")
    opt_parser.add_argument(
        "--optimizer", type=str, default="de",
        choices=["de", "twopointde", "cma", "chaining", "auto"],
        help=(
            "Optimizer to use (default: de). "
            "de=TwoPointsDE (robust global search); "
            "cma=CMA-ES (fast local convergence, learns parameter correlations); "
            "chaining=TwoPointsDE then Nelder-Mead (global then local refine, matches reference paper); "
            "auto=NGOpt (Nevergrad selects algorithm automatically)."
        ),
    )

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

    # Per-bucket loss weights (loss is Σ ((actual-target)/target)² across all measurements)
    opt_parser.add_argument("--weight_base", type=float, default=1.0,
                            help="Weight on baseline firing-rate matching (default: 1.0)")
    opt_parser.add_argument("--weight_global_ko", type=float, default=1.0,
                            help="Weight on global α7/α5/β2 KO PYR matching (default: 1.0)")
    opt_parser.add_argument("--weight_selective_ko", type=float, default=1.0,
                            help="Weight on cell-type-selective α7 KO matching (default: 1.0)")
    opt_parser.add_argument("--weight_drug", type=float, default=1.0,
                            help="Weight on drug-condition matching, Stage 2 (default: 1.0)")

    # Parameter control
    opt_parser.add_argument("--freeze", type=str, default="",
                            help="Comma-separated parameter names to freeze")
    opt_parser.add_argument("--set", dest="set_params", type=str, default="",
                            help="Override parameter values: 'name=val,name=val' (e.g. --set w_sp=0)")
    opt_parser.add_argument("--show_params", action="store_true",
                            help="Show which parameters are free vs frozen")

    # Adaptation
    opt_parser.add_argument("--no_adapt", action="store_true",
                            help="Disable spike-frequency adaptation: set J_adapt_pyr=0 and J_adapt_som=0 "
                                 "and freeze them.")

    # Two-stage flow
    opt_parser.add_argument("--stage", type=str, default="weights",
                            choices=["weights", "receptors"],
                            help="Optimization stage. 'weights' (default): fit synaptic weights + currents, "
                                 "receptor activations frozen at 1. 'receptors': fit per-drug receptor "
                                 "activations only (per-drug independent fits); requires --params_json.")
    # Stage-2 drug targets (in Hz). Pass --drugs to choose which drugs to fit.
    opt_parser.add_argument("--drugs", type=str, default="MLA,PNU,nicotine",
                            help="Comma-separated drug names to fit in --stage receptors (default: MLA,PNU,nicotine)")
    opt_parser.add_argument("--target_mla_ndnf",      type=float, default=None, help="MLA   NDNF target (Hz)")
    opt_parser.add_argument("--target_mla_pv",        type=float, default=None, help="MLA   PV   target (Hz)")
    opt_parser.add_argument("--target_pnu_ndnf",      type=float, default=None, help="PNU   NDNF target (Hz)")
    opt_parser.add_argument("--target_pnu_pv",        type=float, default=None, help="PNU   PV   target (Hz)")
    opt_parser.add_argument("--target_nicotine_ndnf", type=float, default=None, help="Nicotine NDNF target (Hz)")
    opt_parser.add_argument("--target_nicotine_pv",   type=float, default=None, help="Nicotine PV   target (Hz)")

    opt_parser.add_argument("--w_hi", type=float, default=None,
                            help="Upper bound for synaptic weights (nA/Hz). Default: 0.01")

    # I/O settings
    opt_parser.add_argument("--output_dir", type=str, default="",
                            help="Directory where all run outputs are written "
                                 "(best_params.json, best_params.txt, log.jsonl, loss-evolution plots). "
                                 "Explicit --save_best_json or --log_file still override the path.")
    opt_parser.add_argument("--save_best_json", type=str, default="best_params.json",
                            help="Save best parameters to JSON file. If --output_dir is set and this is "
                                 "left at the default, the file goes inside --output_dir.")
    opt_parser.add_argument("--log_file", type=str, default=None,
                            help="Log results to JSONL file (default: {output_dir}/log.jsonl)")
    opt_parser.add_argument("--log_interval", type=int, default=500,
                            help="Log every N steps")
    opt_parser.add_argument("--resume", action="store_true",
                            help="Resume from best_params.json, appending to existing log")

    # ==================================================================
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
                              help="Noise ratio sigma_noise (overrides params_json value)")
    study_parser.add_argument("--tau_noise_ms", type=float, default=5.0,
                              help="OU noise time constant (ms)")
    study_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed for reproducibility")
    study_parser.add_argument("--params_json", type=str, default="",
                              help="Load base (WT) parameters from JSON file")
    study_parser.add_argument("--app_params_json", type=str, default="",
                              help="Load APP parameters from JSON file. When provided, "
                                   "APP conditions use this genuine fit instead of "
                                   "simulating desensitization via activation sampling. "
                                   "KO conditions still set activation to 0.")

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
    study_parser.add_argument("--unit", type=str, default="Hz",
                              choices=["Hz"],
                              help="Rate unit for display (default: Hz)")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nNo command specified. Use 'run', 'optimize', 'study', "
              "or 'plot-transfer'.")
        sys.exit(1)
    elif args.command == "plot-transfer":
        cmd_plot_transfer(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "study":
        cmd_study(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
