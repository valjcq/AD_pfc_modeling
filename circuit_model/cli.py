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
from .optimization import nevergrad_optimize, evaluate_params, KOMeans, LossBreakdown
from .simulation import simulate_circuit
from .jacobian import print_sanity_check, compute_jacobian
from .defaults import DEFAULT_WT_PARAMS_PATH, DEFAULT_APP_PARAMS_PATH, DEFAULT_WT_RING_PARAMS_PATH, DEFAULT_APP_RING_PARAMS_PATH
from .random_search import RandomBistableSearchConfig, run_random_bistable_search

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
    pops = ["PYR", "SOM", "PV ", "VIP"]
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
        "Weights (inhibitory)": ["w_pe", "w_pp", "w_se", "w_sp", "w_vp", "w_vs"],
        "External currents": ["I0_pyr", "I0_pv", "I_alpha7_pv", "I0_som", "I_alpha7_som", "I_beta2_som", "I0_vip", "I_alpha5_vip"],
        "Transient": ["trans_factor"],
        "Transfer function": ["Theta_pyr", "alpha_pyr", "Theta_pv", "alpha_pv", "Theta_som", "alpha_som", "Theta_vip", "alpha_vip", "g"],
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


def _print_opt_init_summary(params: CircuitParams, means: np.ndarray, breakdown: "LossBreakdown") -> None:  # type: ignore
    """Print effective optimization initialization and its predicted rates."""
    print("Initial condition (effective after --set/--no_adapt):")
    print(f"  I0: pyr={params.I0_pyr:.6g}, som={params.I0_som:.6g}, pv={params.I0_pv:.6g}, vip={params.I0_vip:.6g}")
    print(f"  W:  J_NMDA={params.J_NMDA:.6g}, w_ep={params.w_ep:.6g}, w_es={params.w_es:.6g}, w_ev={params.w_ev:.6g}")
    print(f"      w_pe={params.w_pe:.6g}, w_pp={params.w_pp:.6g}, w_se={params.w_se:.6g}, w_sp={params.w_sp:.6g}, w_vp={params.w_vp:.6g}, w_vs={params.w_vs:.6g}")
    print(f"  Transfer: tau_s={params.tau_s:.6g}, alpha_pyr={params.alpha_pyr:.6g}, alpha_som={params.alpha_som:.6g}, alpha_pv={params.alpha_pv:.6g}, alpha_vip={params.alpha_vip:.6g}")
    print(f"            Theta_pyr={params.Theta_pyr:.6g}, Theta_som={params.Theta_som:.6g}, Theta_pv={params.Theta_pv:.6g}, Theta_vip={params.Theta_vip:.6g}")
    print("Initial predicted rates (Hz):")
    print(f"  PYR={means[0]:.4f}, SOM={means[1]:.4f}, PV={means[2]:.4f}, VIP={means[3]:.4f}")
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


def cmd_diagnostic(args: argparse.Namespace) -> None:
    """Plot Turing gain product and transfer functions (analytical, no simulation)."""
    import json
    from .io import load_params_json
    from .ring.params import RingParams
    from .diagnostic import plot_turing_gain_product, plot_transfer_functions_diagnostic

    # Load circuit parameters
    if args.params_json:
        circuit_params = load_params_json(args.params_json)
        print(f"Loaded circuit parameters from: {args.params_json}")
    elif DEFAULT_WT_PARAMS_PATH.exists():
        circuit_params = load_params_json(str(DEFAULT_WT_PARAMS_PATH))
        print(f"Loaded default circuit parameters from: {DEFAULT_WT_PARAMS_PATH}")
    else:
        print("ERROR: --params_json is required or default params file not found at", DEFAULT_WT_PARAMS_PATH)
        sys.exit(1)

    # Load ring parameters
    if args.ring_params_json:
        with open(args.ring_params_json) as f:
            ring_dict = json.load(f)
        ring_params = RingParams(**ring_dict)
        print(f"Loaded ring parameters from: {args.ring_params_json}")
    elif DEFAULT_WT_RING_PARAMS_PATH.exists():
        with open(DEFAULT_WT_RING_PARAMS_PATH) as f:
            ring_dict = json.load(f)
        ring_params = RingParams(**ring_dict)
        print(f"Loaded default ring parameters from: {DEFAULT_WT_RING_PARAMS_PATH}")
    else:
        print("ERROR: --ring_params_json is required or default ring params file not found at", DEFAULT_WT_RING_PARAMS_PATH)
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Turing gain product
    save_path_1 = out_dir / "turing_gain_product.png"
    print(f"\nGenerating Turing gain product plot...")
    plot_turing_gain_product(
        circuit_params,
        ring_params,
        target_pyr=args.target_pyr,
        turing_bump_hz=args.turing_bump_hz,
        turing_cue_hz=args.turing_cue_hz,
        save_path=str(save_path_1),
        show=not args.no_show,
    )

    # Plot 2: Transfer functions
    save_path_2 = out_dir / "transfer_functions.png"
    print(f"Generating transfer function plots...")
    plot_transfer_functions_diagnostic(
        circuit_params,
        target_pyr=args.target_pyr,
        turing_bump_hz=args.turing_bump_hz,
        turing_cue_hz=args.turing_cue_hz,
        save_path=str(save_path_2),
        show=not args.no_show,
    )

    print(f"\nDiagnostic plots saved to: {out_dir}/")


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

    if getattr(args, "save_metrics", ""):
        _save_run_metrics_bistable(result, params, args, args.save_metrics)


def _save_run_metrics_bistable(result, params, args, metrics_path: str) -> None:
    """Compute and save pre/post-transient state metrics as JSON.

    Time windows
    ------------
    pre-transient  : [100 ms, trans_start - 100 ms)   — settled baseline
    during          : [trans_start, trans_end]          — peak only
    post-transient  : [trans_end + 300 ms, T_ms]        — settled post state
    """
    import json
    from pathlib import Path as _Path

    t      = result.t_ms
    r_pyr  = result.r[:, 0]

    trans_start = params.trans_start_ms if params.trans_enabled else float("nan")
    trans_end   = (trans_start + params.trans_duration_ms) if params.trans_enabled else float("nan")
    SETTLE_MS   = 300.0

    if params.trans_enabled and not np.isnan(trans_start):
        pre_mask   = (t >= 100.0) & (t < trans_start - 100.0)
        trans_mask = (t >= trans_start) & (t <= trans_end)
        post_mask  = t >= (trans_end + SETTLE_MS)
    else:
        pre_mask   = t >= 100.0
        trans_mask = np.zeros(len(t), dtype=bool)
        post_mask  = t >= (t[-1] * 0.5)

    pre_pyr   = float(r_pyr[pre_mask].mean())   if np.any(pre_mask)   else float("nan")
    trans_peak = float(r_pyr[trans_mask].max())  if np.any(trans_mask) else float("nan")
    post_pyr  = float(r_pyr[post_mask].mean())  if np.any(post_mask)  else float("nan")

    actual_sigma = 0.0 if getattr(args, "noise_type", "none") == "none" else float(params.sigma_noise)

    metrics = {
        "params": {
            "amplitude":        round(float(params.trans_factor), 4),
            "sigma_noise":      round(actual_sigma, 6),
            "trans_start_ms":   float(trans_start),
            "trans_duration_ms": float(params.trans_duration_ms) if params.trans_enabled else float("nan"),
            "T_ms":             float(t[-1]),
        },
        "steady_state": {
            "pre_trans_pyr_hz":  round(pre_pyr,   3),
            "trans_peak_pyr_hz": round(trans_peak, 3),
            "post_trans_pyr_hz": round(post_pyr,  3),
        },
    }

    out = _Path(metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)


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


def cmd_random_bistable_search(args: argparse.Namespace) -> None:
    """Run random parameter sampling and log only bistable hits."""
    from dataclasses import replace
    from .bistable_loss import BistableConfig

    condition_key = args.condition if getattr(args, "condition", "") else None
    base, load_msg = _load_params_with_optional_condition(
        params_json=args.params_json,
        condition_key=condition_key,
        context="random-bistable-search",
    )
    print(load_msg)

    if args.set_params:
        overrides = parse_set_params(args.set_params)
        allowed = {f.name for f in fields(CircuitParams)}
        clean = {k: v for k, v in overrides.items() if k in allowed}
        if clean:
            base = replace(base, **clean)
            print(f"Overrides applied: {', '.join(f'{k}={v}' for k, v in clean.items())}")

    freeze = parse_freeze_list(args.freeze)
    if args.no_adapt:
        base = replace(base, J_adapt_pyr=0.0, J_adapt_som=0.0)
        freeze |= {"J_adapt_pyr", "J_adapt_som"}
        print("--no_adapt: J_adapt_pyr=0, J_adapt_som=0 (frozen)")

    bounds = default_bounds(base, w_hi=getattr(args, "w_hi", None))

    if args.show_params:
        print_parameter_status(bounds, freeze, base)

    bistable_cfg = BistableConfig(
        r_low_target=args.r_low_hz,
        delta_r_min=args.delta_r_min,
        r_pv_target=args.r_pv_low_target,
        r_som_target=args.r_som_low_target,
        r_vip_target=args.r_vip_low_target,
        r_pyr_high_target=args.r_pyr_high_target,
        r_som_high_target=args.r_som_high_target,
        r_pv_high_target=args.r_pv_high_target,
        r_vip_high_target=args.r_vip_high_target,
    )

    search_cfg = RandomBistableSearchConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        show_every=args.show_every,
        output_jsonl=args.output_jsonl,
        summary_txt=args.summary_txt,
        append=args.append,
        max_hits=args.max_hits,
        sim_T_ms=args.T_ms,
        sim_dt_ms=args.dt_ms,
        sim_burn_in_ms=args.burn_in_ms,
        sim_window_ms=args.window_ms,
        sim_noise_type=args.noise_type,
        sim_tau_noise_ms=args.tau_noise_ms,
    )

    print("\nRandom bistable search:")
    print(f"  samples:    {search_cfg.n_samples}")
    print(f"  seed:       {search_cfg.seed}")
    print(f"  freeze:     {len(freeze)} params")
    print(f"  output:     {search_cfg.output_jsonl}")
    print("  simulation: "
          f"T={search_cfg.sim_T_ms} ms, dt={search_cfg.sim_dt_ms} ms, "
          f"burn_in={search_cfg.sim_burn_in_ms} ms, window={search_cfg.sim_window_ms} ms, "
          f"noise={search_cfg.sim_noise_type}")
    print()

    summary = run_random_bistable_search(
        base=base,
        bounds=bounds,
        freeze=freeze,
        bistable_cfg=bistable_cfg,
        search_cfg=search_cfg,
    )

    print("\nSearch complete:")
    print(f"  bistable hits:      {summary['bistable_hits']}")
    print(f"  evaluation errors:  {summary['evaluation_errors']}")
    print(f"  hit rate:           {summary['hit_rate_pct']:.6f}%")
    print(f"  hits JSONL:         {summary['hits_jsonl']}")
    print(f"  summary TXT:        {summary['summary_txt']}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run parameter optimization."""
    mode = getattr(args, "mode", "standard")

    if not getattr(args, "resume", False) and mode in ("standard", "bistable"):
        missing = [f"--target_{k}" for k, v in [
            ("pyr", args.target_pyr), ("som", args.target_som),
            ("pv", args.target_pv), ("vip", args.target_vip),
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
            alpha7_ko_pyr=t.get("alpha7_ko_pyr"),
            alpha5_ko_pyr=t.get("alpha5_ko_pyr"),
            beta2_ko_pyr=t.get("beta2_ko_pyr"),
        )
        print(f"Targets loaded from log: pyr={target.mean_r_pyr}, som={target.mean_r_som}, "
              f"pv={target.mean_r_pv}, vip={target.mean_r_vip}")
    else:
        # Build target rates from CLI args (used in both standard and bistable modes)
        target = TargetRates(
            mean_r_pyr=args.target_pyr,
            mean_r_som=args.target_som,
            mean_r_pv=args.target_pv,
            mean_r_vip=args.target_vip,
            alpha7_ko_pyr=args.target_alpha7_ko_pyr,
            alpha5_ko_pyr=args.target_alpha5_ko_pyr,
            beta2_ko_pyr=args.target_beta2_ko_pyr,
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

    # Resolve n_samples (budget is an alias for n_samples in bistable mode)
    n_samples = args.budget if getattr(args, "budget", None) is not None else args.n_samples

    bounds = default_bounds(base, w_hi=getattr(args, "w_hi", None))
    freeze = parse_freeze_list(args.freeze)
    if args.no_adapt:
        freeze |= {"J_adapt_pyr", "J_adapt_som"}

    # Build bistable config if in bistable mode
    bistable_cfg = None
    if mode == "bistable":
        from .bistable_loss import BistableConfig
        # If --r_low_hz not explicitly set, use --target_pyr
        # (the resting PYR rate is both the low FP location and the rate target)
        r_low_hz = args.r_low_hz if args.r_low_hz is not None else args.target_pyr
        bistable_cfg = BistableConfig(
            r_low_target=r_low_hz,
            delta_r_min=args.delta_r_min,
            r_pv_target=args.target_pv,
            r_som_target=args.target_som,
            r_vip_target=args.target_vip,
            r_pyr_high_target=args.r_pyr_high_target,
            r_som_high_target=args.r_som_high_target,
            r_pv_high_target=args.r_pv_high_target,
            r_vip_high_target=args.r_vip_high_target,
            w_bistab=args.w_bistab,
            w_rate=args.w_rate_bistab,
            w_rate_high=args.w_rate_high,
            w_margin=args.w_margin,
            w_jacobian=args.jacobian_weight,
            nullcline_peak_max=args.nullcline_peak_max,
            w_peak=args.w_peak,
            condition=getattr(args, "condition", "WT"),
        )

    if mode == "bistable":
        if args.turing_weight != 0.0 or args.ach_ratio_weight != 0.0:
            print("WARNING (bistable mode): --turing_weight and --ach_ratio_weight have no effect.")

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
    if mode == "bistable":
        print("\nOptimization targets (bistable mode):")
        print(f"  LOW state  — PYR: {target.mean_r_pyr} {unit}  SOM: {target.mean_r_som} {unit}  PV: {target.mean_r_pv} {unit}  VIP: {target.mean_r_vip} {unit}")
        print(f"  HIGH state — PYR: {args.r_pyr_high_target} {unit}  SOM: {args.r_som_high_target} {unit}  PV: {args.r_pv_high_target} {unit}  VIP: {args.r_vip_high_target} {unit}")
    else:
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

    jacobian_weight = 0.0 if args.skip_jacobian else args.jacobian_weight

    # Determine output directory for logging
    out_dir_for_logs = args.output_dir or (Path(args.save_best_json).parent if args.save_best_json else ".")

    # If output_dir is specified and save_best_json is the default, put best_params.json in output_dir
    save_best_json_to_use = args.save_best_json
    if args.output_dir and args.save_best_json == "best_params.json":
        # User didn't explicitly set --save_best_json, so put it in output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_best_json_to_use = str(Path(args.output_dir) / "best_params.json")

    # Automatically set up log file if output_dir is specified and log_file is not
    log_file_to_use = args.log_file
    if not log_file_to_use and out_dir_for_logs:
        log_dir = Path(out_dir_for_logs)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_to_use = str(log_dir / "log.jsonl")
        # Also set a better log_interval default if not explicitly set by user
        log_interval_to_use = args.log_interval if args.log_interval != 500 else 50
    else:
        log_interval_to_use = args.log_interval

    if mode == "standard":
        init_seed = args.seed if args.seed is not None else 0
        init_rng = np.random.default_rng(init_seed)
        init_loss, init_means, _, init_breakdown = evaluate_params(
            base,
            target,
            fit_cfg,
            rng=init_rng,
            squared_loss=args.squared_loss,
            jacobian_weight=jacobian_weight,
            turing_weight=args.turing_weight,
            turing_margin=args.turing_margin,
            turing_w_inter_ref=args.turing_w_inter_ref,
            turing_cue_scale=args.turing_cue_scale,
            ach_ratio_weight=args.ach_ratio_weight,
        )
        _print_opt_init_summary(base, init_means, init_breakdown)
        print()
    else:
        # Bistable mode: print initial bistability loss
        from .bistable_loss import bistable_loss as _bistable_loss
        init_loss, init_components = _bistable_loss(base, bistable_cfg, return_components=True)
        print("\nInitial bistability loss components:")
        print(f"  L_bistab:    {init_components.get('L_bistab', 0.0):.4g}")
        print(f"  L_rate:      {init_components.get('L_rate', 0.0):.4g}")
        print(f"  L_rate_high: {init_components.get('L_rate_high', 0.0):.4g}")
        print(f"  L_margin:    {init_components.get('L_margin', 0.0):.4g}")
        print(f"  L_jac:       {init_components.get('L_jac', 0.0):.4g}")
        print(f"  L_total:     {init_components.get('L_total', 0.0):.4g}")
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
        squared_loss=args.squared_loss,
        jacobian_weight=jacobian_weight,
        turing_weight=args.turing_weight,
        turing_margin=args.turing_margin,
        turing_w_inter_ref=args.turing_w_inter_ref,
        turing_cue_scale=args.turing_cue_scale,
        ach_ratio_weight=args.ach_ratio_weight,
        bistable_cfg=bistable_cfg,
    )

    if not best:
        print("No optimization candidates available (run may have been interrupted very early).")
        return

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

    if mode == "standard":
        print("\nBest parameter set:\n")
        print(format_params_as_code(best[0].params))

        # Jacobian sanity check at the best fitted steady state
        r_ss = best[0].means  # ndarray [pyr, som, pv, vip]
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
    else:
        # Bistable mode: save bistable-specific outputs
        from .bistable_loss import bistable_loss as _bistable_loss, save_bistable_summary

        # Determine output directory
        out_dir = args.output_dir or (Path(args.save_best_json).parent if args.save_best_json else ".")
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Save the full command used to launch this run
        cmd_path = os.path.join(out_dir, "command.txt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write("python -m circuit_model " + " ".join(sys.argv[1:]) + "\n")

        # --- Save best (rank 1) at the top level ---
        _, components = _bistable_loss(best[0].params, bistable_cfg, return_components=True)
        bistable_json = os.path.join(out_dir, "bistable_params.json")
        save_params_json(bistable_json, best[0].params)
        save_bistable_summary(out_dir, best[0].params, components, bistable_cfg)

        # --- Save all top-K candidates in top10/ sub-folder ---
        top_dir = os.path.join(out_dir, "top10")
        os.makedirs(top_dir, exist_ok=True)

        all_components = []
        for rank, cand in enumerate(best, start=1):
            _, comp = _bistable_loss(cand.params, bistable_cfg, return_components=True)
            all_components.append(comp)
            rank_dir = os.path.join(top_dir, f"rank{rank:02d}")
            os.makedirs(rank_dir, exist_ok=True)
            save_params_json(os.path.join(rank_dir, "bistable_params.json"), cand.params)
            save_bistable_summary(rank_dir, cand.params, comp, bistable_cfg)

        # --- Write a consolidated leaderboard table ---
        leaderboard_path = os.path.join(top_dir, "leaderboard.txt")
        with open(leaderboard_path, "w", encoding="utf-8") as f:
            header = (
                f"{'Rank':>4}  {'L_total':>9}  {'L_bistab':>9}  {'L_rate':>9}  "
                f"{'L_rate_hi':>9}  {'L_margin':>9}  "
                f"{'r_low':>7}  {'r_high':>7}  "
                f"{'SOM_hi':>7}  {'PV_hi':>7}  {'VIP_hi':>7}  {'regime':>10}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for rank, (cand, comp) in enumerate(zip(best, all_components), start=1):
                is_bistable = comp.get("n_stable", 0) >= 2
                regime = "BISTABLE" if is_bistable else "monostable"
                r_high = comp.get("r_high_fp")
                som_hi = comp.get("r_som_high_fp")
                pv_hi  = comp.get("r_pv_high_fp")
                vip_hi = comp.get("r_vip_high_fp")
                row = (
                    f"{rank:>4}  {comp.get('L_total', 0.0):>9.4f}  "
                    f"{comp.get('L_bistab', 0.0):>9.4f}  {comp.get('L_rate', 0.0):>9.4f}  "
                    f"{comp.get('L_rate_high', 0.0):>9.4f}  {comp.get('L_margin', 0.0):>9.4f}  "
                    f"{comp.get('r_low_fp', 0.0):>7.2f}  "
                    f"{r_high if r_high is not None else float('nan'):>7.2f}  "
                    f"{som_hi if som_hi is not None else float('nan'):>7.2f}  "
                    f"{pv_hi  if pv_hi  is not None else float('nan'):>7.2f}  "
                    f"{vip_hi if vip_hi is not None else float('nan'):>7.2f}  "
                    f"{regime:>10}"
                )
                f.write(row + "\n")

        print("\nBest bistable parameters and summary saved:")
        print(f"  Command: {cmd_path}")
        print(f"  Params:  {bistable_json}")
        print(f"  Summary: {os.path.join(out_dir, 'bistable_summary.txt')}")
        print(f"\nTop-{len(best)} leaderboard saved to: {leaderboard_path}")
        print(f"  (individual rank folders: {top_dir}/rank01/ … rank{len(best):02d}/)")
        print(f"\n  Final bistability regime: {'BISTABLE' if components.get('n_stable', 0) >= 2 else 'MONOSTABLE'}")
    
    # Generate loss evolution plots
    if log_file_to_use:
        try:
            from .loss_evolution_plot import plot_loss_evolution, plot_loss_evolution_ratios
            log_dir = Path(log_file_to_use).parent
            plot_loss_evolution(log_file_to_use, output_dir=str(log_dir))
            plot_loss_evolution_ratios(log_file_to_use, output_dir=str(log_dir))
        except Exception as e:
            print(f"Warning: could not generate loss evolution plots: {e}")

    # Generate nullcline plot for best bistable params
    if mode == "bistable" and bistable_json:
        try:
            import subprocess
            _script = Path(__file__).parent.parent / "scripts" / "nullcline_analysis.py"
            subprocess.run(
                [sys.executable, str(_script), "--no_show", "--params_json", bistable_json],
                check=False,
            )
        except Exception as e:
            print(f"Warning: could not generate nullcline plot: {e}")


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

    # Random parameter search for bistable regimes
    python -m circuit_model random-bistable-search --n_samples 100000 --show_every 2000

    # Ring attractor: single condition
    python -m circuit_model ring-run --condition WT --amplitude 3

    # Ring attractor: bump-decay study across conditions
    python -m circuit_model ring-bump-decay-study --conditions WT WT_APP --amplitudes 10 20 30

    # Ring attractor: joint circuit + ring optimization
    python -m circuit_model ring-optimize --target_pyr 8 --target_som 5 --target_pv 3 --target_vip 2
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

    # Optional knockout targets
    opt_parser.add_argument("--target_alpha7_ko_pyr", type=float, default=None,
                            help="Target PYR rate under alpha7 knockout")
    opt_parser.add_argument("--target_alpha5_ko_pyr", type=float, default=None,
                            help="Target PYR rate under alpha5 knockout")
    opt_parser.add_argument("--target_beta2_ko_pyr", type=float, default=None,
                            help="Target PYR rate under beta2 knockout")

    # Optimization settings
    opt_parser.add_argument("--squared_loss", action=argparse.BooleanOptionalAction, default=True,
                            help="Use MSPE (squared percentage error) loss — default on. Pass --no_squared_loss to revert to MAPE.")
    opt_parser.add_argument("--n_samples", type=int, default=5000,
                            help="Number of optimization samples")
    opt_parser.add_argument("--top_k", type=int, default=10,
                            help="Keep top K candidates")
    opt_parser.add_argument(
        "--optimizer", type=str, default="de",
        choices=["de", "cma", "chaining", "auto"],
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

    # KO penalty settings
    opt_parser.add_argument("--ko_min_effect_penalty", type=float, default=5.0,
                            help="Penalty weight for weak KO effect")
    opt_parser.add_argument("--ko_wrong_direction_penalty", type=float, default=10.0,
                            help="Penalty weight for wrong direction KO effect")

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

    # Turing instability penalty
    opt_parser.add_argument("--turing_weight", type=float, default=2.0,
                            help="Weight of two-sided Turing bistability penalty (default: 2.0). "
                                 "Penalises rest-state gain above 1-margin AND cue-state gain below 1+margin.")
    opt_parser.add_argument("--turing_margin", type=float, default=0.05,
                            help="Safety margin around the Turing threshold (default: 0.05)")
    opt_parser.add_argument("--turing_w_inter_ref", type=float, default=10.0,
                            help="Reference inter-node weight used in the Turing condition for single-node "
                                 "optimization (default: 10.0). Has no effect if --turing_weight is 0.")
    opt_parser.add_argument("--turing_cue_scale", type=float, default=0.4,
                            help="Multiplier applied to I0_pyr to approximate the cue operating point "
                                 "(default: 0.4)")
    opt_parser.add_argument("--skip-jacobian", action="store_true",
                            help="Skip the Jacobian connectivity penalty during optimization.")
    opt_parser.add_argument("--jacobian_weight", type=float, default=1.0,
                            help="Weight of the Jacobian connectivity penalty (default: 1.0, 0 = disabled). "
                                 "Controls the strength of connectivity constraints during optimization.")
    opt_parser.add_argument("--ach_ratio_weight", type=float, default=2.0,
                            help="Weight of β2/α7 ACh current ratio penalty (default: 2.0, 0 = disabled). "
                                 "Penalises solutions where I_beta2_som / I_alpha7_som deviates from 35 "
                                 "(Koukouli et al. 2025: β2-type currents ~35× stronger than α7 at 1.77 μM ACh).")

    # Bistable mode arguments
    opt_parser.add_argument("--mode", type=str, default="standard",
                            choices=["standard", "bistable"],
                            help="Optimization mode: standard (fit to rates) or bistable (find bistable nullcline). "
                                 "Default: standard")
    opt_parser.add_argument("--output_dir", type=str, default="",
                            help="Output directory for bistable mode results (default: use parent of --save_best_json)")
    opt_parser.add_argument("--budget", type=int, default=None,
                            help="Nevergrad budget (alias for --n_samples, convenience for bistable mode)")
    opt_parser.add_argument("--r_low_hz", type=float, default=None,
                            help="Target low fixed point PYR rate for bistable mode (Hz, default: uses --target_pyr)")
    opt_parser.add_argument("--delta_r_min", type=float, default=15.0,
                            help="Minimum gap between low and high fixed points (Hz, default: 15.0)")
    # High-state rate targets (Rooy 2021 defaults)
    opt_parser.add_argument("--r_pyr_high_target", type=float, default=60.2,
                            help="Target PYR rate at the high fixed point (Hz, default: 60.2 — Rooy 2021)")
    opt_parser.add_argument("--r_som_high_target", type=float, default=35.2,
                            help="Target SOM rate at the high fixed point (Hz, default: 35.2 — Rooy 2021)")
    opt_parser.add_argument("--r_pv_high_target", type=float, default=35.3,
                            help="Target PV rate at the high fixed point (Hz, default: 35.3 — Rooy 2021)")
    opt_parser.add_argument("--r_vip_high_target", type=float, default=68.8,
                            help="Target VIP rate at the high fixed point (Hz, default: 68.8 — Rooy 2021)")
    # Loss weights
    opt_parser.add_argument("--w_hi", type=float, default=None,
                            help="Upper bound for synaptic weights (nA/Hz). Default: 0.02")
    opt_parser.add_argument("--w_bistab", type=float, default=5.0,
                            help="Weight of bistability sign pattern loss (default: 5.0)")
    opt_parser.add_argument("--w_rate_bistab", type=float, default=1.0,
                            help="Weight of low-FP rate matching loss in bistable mode (default: 1.0)")
    opt_parser.add_argument("--w_rate_high", type=float, default=1.5,
                            help="Weight of high-FP rate matching loss (default: 1.5)")
    opt_parser.add_argument("--w_margin", type=float, default=2.0,
                            help="Weight of fixed point separation margin loss (default: 2.0)")
    opt_parser.add_argument("--nullcline_peak_max", type=float, default=80.0,
                            help="Maximum allowed nullcline peak Φ (Hz). Values above this are penalised. "
                                 "Default 80 Hz. Increase if bistability is hard to achieve.")
    opt_parser.add_argument("--w_peak", type=float, default=0.0,
                            help="Weight of nullcline peak penalty (default: 0.0 = off)")

    # I/O settings
    opt_parser.add_argument("--save_best_json", type=str, default="best_params.json",
                            help="Save best parameters to JSON file")
    opt_parser.add_argument("--log_file", type=str, default=None,
                            help="Log results to JSONL file (default: auto-generated in figs/optim/)")
    opt_parser.add_argument("--log_interval", type=int, default=500,
                            help="Log every N steps")
    opt_parser.add_argument("--resume", action="store_true",
                            help="Resume from best_params.json, appending to existing log")

    # =========================================================================
    # PLOT-TRANSFER subcommand
    # =========================================================================
    tf_parser = subparsers.add_parser(
        "plot-transfer",
        help="Plot transfer functions for all 4 populations",
        description="Plot Phi(I) = u / (1 - exp(-g*u)) for each population on a single axis."
    )
    tf_parser.add_argument("--params_json", type=str, default="",
                           help="Load parameters from JSON file (default: use built-in defaults)")
    tf_parser.add_argument(
        "--condition",
        type=str,
        default="",
        choices=["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP", "APP_sim"],
        help=(
            "Apply an experimental condition preset. If --params_json is not provided, "
            "the command auto-loads default project WT/WT_APP fitted files when available."
        ),
    )
    tf_parser.add_argument("--set", dest="set_params", type=str, default="",
                           help="Override parameter values: 'name=val,name=val'")
    tf_parser.add_argument("--I_min", type=float, default=-5.0,
                           help="Minimum input current to plot (default: -5)")
    tf_parser.add_argument("--I_max", type=float, default=7.0,
                           help="Maximum input current to plot (default: 7)")
    tf_parser.add_argument("--save_plot", type=str, default="",
                           help="Save plot to file (e.g., 'transfer_functions.png')")
    tf_parser.add_argument("--no_show", action="store_true",
                           help="Don't display the plot")

    # =========================================================================
    # DIAGNOSTIC subcommand
    # =========================================================================
    diag_parser = subparsers.add_parser(
        "diagnostic",
        help="Analytical diagnostic plots (Turing gain product + transfer functions)",
        description=(
            "Generate analytical (no-simulation) diagnostic plots:\n"
            "  1. Turing gain product vs PYR firing rate\n"
            "  2. Transfer functions for all 4 populations with operating point markers\n"
            "\n"
            "If --params_json and --ring_params_json are omitted, defaults are loaded from:\n"
            "  - params/new/ring_firing_rate/WT_1mo_article_ko.json\n"
            "  - params/new/ring_firing_rate/WT_1mo_article_ko_ring.json"
        ),
    )
    diag_parser.add_argument("--params_json", type=str, default="",
                            help="Path to circuit parameters JSON file (default: auto-load if available)")
    diag_parser.add_argument("--ring_params_json", type=str, default="",
                            help="Path to ring parameters JSON file (default: auto-load if available)")
    diag_parser.add_argument("--target_pyr", type=float, default=8.0,
                            help="Rest PYR firing rate for operating point marker (Hz, default: 8.0)")
    diag_parser.add_argument("--turing_bump_hz", type=float, default=40.0,
                            help="PYR firing rate for the bump operating point marker (Hz, default: 40.0)")
    diag_parser.add_argument("--turing_cue_hz", type=float, default=60.0,
                            help="PYR firing rate for the cue operating point marker (Hz, default: 60.0)")
    diag_parser.add_argument("--out_dir", type=str, default="figs/diagnostic",
                            help="Output directory for figures (default: figs/diagnostic)")
    diag_parser.add_argument("--no_show", action="store_true",
                            help="Don't display the plots")

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

    # =========================================================================
    # RANDOM-BISTABLE-SEARCH subcommand
    # =========================================================================
    random_parser = subparsers.add_parser(
        "random-bistable-search",
        help="Randomly sample parameter sets and record bistable hits",
        description=(
            "Sample parameters from default bounds (respecting --freeze), evaluate bistability "
            "with nullcline tools, and for bistable hits simulate low/high states and log their rates."
        ),
    )
    random_parser.add_argument("--n_samples", type=int, default=100000,
                               help="Number of random parameter sets to evaluate (default: 100000)")
    random_parser.add_argument("--seed", type=int, default=0,
                               help="Random seed for reproducibility (default: 0)")
    random_parser.add_argument("--show_every", type=int, default=1000,
                               help="Progress print interval in number of evaluated samples (default: 1000)")
    random_parser.add_argument("--max_hits", type=int, default=None,
                               help="Optional early stop after this many bistable hits")
    random_parser.add_argument("--output_jsonl", type=str, default="figs/optim/random_bistable_hits.jsonl",
                               help="Output JSONL file for bistable hits")
    random_parser.add_argument("--summary_txt", type=str, default="figs/optim/random_bistable_summary.txt",
                               help="Output summary text file")
    random_parser.add_argument("--append", action="store_true",
                               help="Append hits to --output_jsonl instead of overwriting")

    random_parser.add_argument("--params_json", type=str, default="",
                               help="Load base parameters from JSON file")
    random_parser.add_argument(
        "--condition",
        type=str,
        default="",
        choices=["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP", "APP_sim"],
        help=(
            "Apply an experimental condition preset. If --params_json is not provided, "
            "the command auto-loads default project WT/WT_APP fitted files when available."
        ),
    )
    random_parser.add_argument("--set", dest="set_params", type=str, default="",
                               help="Override parameter values: 'name=val,name=val'")
    random_parser.add_argument("--freeze", type=str, default="",
                               help="Comma-separated parameter names to freeze to base values")
    random_parser.add_argument("--w_hi", type=float, default=None,
                               help="Upper bound for synaptic weights (nA/Hz). Default: 0.01")
    random_parser.add_argument("--show_params", action="store_true",
                               help="Show which parameters are free vs frozen")
    random_parser.add_argument("--no_adapt", action="store_true",
                               help="Set and freeze J_adapt_pyr=0 and J_adapt_som=0")

    random_parser.add_argument("--r_low_hz", type=float, default=8.0,
                               help="Target low-state PYR fixed point for bistability check (Hz)")
    random_parser.add_argument("--r_som_low_target", type=float, default=5.0,
                               help="Target low-state SOM fixed point (Hz)")
    random_parser.add_argument("--r_pv_low_target", type=float, default=3.0,
                               help="Target low-state PV fixed point (Hz)")
    random_parser.add_argument("--r_vip_low_target", type=float, default=2.0,
                               help="Target low-state VIP fixed point (Hz)")
    random_parser.add_argument("--delta_r_min", type=float, default=15.0,
                               help="Minimum high-vs-low PYR fixed-point separation (Hz)")
    random_parser.add_argument("--r_pyr_high_target", type=float, default=60.2,
                               help="Target high-state PYR fixed point (Hz)")
    random_parser.add_argument("--r_som_high_target", type=float, default=35.2,
                               help="Target high-state SOM fixed point (Hz)")
    random_parser.add_argument("--r_pv_high_target", type=float, default=35.3,
                               help="Target high-state PV fixed point (Hz)")
    random_parser.add_argument("--r_vip_high_target", type=float, default=68.8,
                               help="Target high-state VIP fixed point (Hz)")

    random_parser.add_argument("--T_ms", type=float, default=2500.0,
                               help="Simulation duration for low/high state validation (ms)")
    random_parser.add_argument("--dt_ms", type=float, default=0.1,
                               help="Simulation integration step for state validation (ms)")
    random_parser.add_argument("--burn_in_ms", type=float, default=1800.0,
                               help="Burn-in period when computing low/high state means (ms)")
    random_parser.add_argument("--window_ms", type=float, default=500.0,
                               help="Averaging window for low/high state means (ms)")
    random_parser.add_argument("--noise_type", choices=["none", "white", "ou"], default="none",
                               help="Noise model for low/high state simulation checks")
    random_parser.add_argument("--tau_noise_ms", type=float, default=5.0,
                               help="OU noise time constant (ms) for low/high state validation")
    random_parser.add_argument("--n_workers", type=int, default=10,
                               help="Number of parallel workers for multiprocessing (default: 10). "
                                    "Set to 1 for serial execution.")

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
    ring_run_parser.add_argument(
        "--distractor_duration_ms", type=float, default=250.0,
        help="Distractor stimulus duration in ms (default: 250). "
             "Only used when --distractor_factor and --distractor_offset_deg are set.",
    )
    ring_run_parser.add_argument(
        "--delay2_ms", type=float, default=5000.0,
        help="Delay after distractor offset in ms (default: 5000). "
             "Only used when the distractor is enabled.",
    )
    ring_run_parser.add_argument(
        "--no_adapt", action="store_true",
        help="Disable spike-frequency adaptation: set J_adapt_pyr=0 and J_adapt_som=0.",
    )
    ring_run_parser.add_argument(
        "--output_dir", type=str, default="",
        help="Explicit output directory for results (default: auto-generated path based on params)",
    )

    # =========================================================================
    # RING-CALIBRATE subcommand
    # =========================================================================
    ring_cal_parser = subparsers.add_parser(
        "ring-calibrate",
        help="3D parameter sweep (w_pv_global × w_pyr_pyr_inter × amplitude) for the ring attractor",
        description="Sweep a 3D grid of (w_pv_global, w_pyr_pyr_inter, stimulus_amplitude) "
                    "to find parameter combinations that produce a stable, localised memory bump. "
                    "Classifies delay period into resting/bump/saturated states and reports "
                    "the fraction of delay time in each state. "
                    "Burn-in is computed once per (cond, w_pv, w_pyr) and reused across amplitudes.",
    )
    _add_ring_common(ring_cal_parser)
    # w_pyr_pyr_inter is swept via --w_inter_values; make the base value optional
    for _action in ring_cal_parser._actions:
        if _action.dest == "w_pyr_pyr_inter":
            _action.required = False
            _action.default = [0.0]
            break
    ring_cal_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to calibrate (default: WT only). Use 'all' for all conditions.",
    )
    ring_cal_parser.add_argument(
        "--amplitudes", type=float, nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80],
        help="Stimulus amplitudes to sweep (default: 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.80)",
    )
    ring_cal_parser.add_argument(
        "--w_inter_values", type=float, nargs="+",
        default=None,
        help="w_pyr_pyr_inter values to sweep (explicit list). "
             "Mutually exclusive with --w_inter_min/--w_inter_max/--n_inter. "
             "Default when none specified: 0.002 0.003 0.004 0.005 0.006 0.008 0.010",
    )
    ring_cal_parser.add_argument(
        "--w_inter_min", type=float, default=None,
        help="Minimum w_pyr_pyr_inter for linspace sweep (requires --w_inter_max and --n_inter).",
    )
    ring_cal_parser.add_argument(
        "--w_inter_max", type=float, default=None,
        help="Maximum w_pyr_pyr_inter for linspace sweep (requires --w_inter_min and --n_inter).",
    )
    ring_cal_parser.add_argument(
        "--n_inter", type=int, default=None,
        help="Number of w_pyr_pyr_inter steps for linspace sweep (requires --w_inter_min and --w_inter_max).",
    )
    ring_cal_parser.add_argument(
        "--w_pv_values", type=float, nargs="+", default=None,
        help="w_pv_global values to sweep (default: uses --w_pv_global single value). "
             "Provide multiple values for a full 3D sweep across w_pv dimension.",
    )
    ring_cal_parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Number of trials per grid point (default: 20)",
    )
    ring_cal_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    ring_cal_parser.add_argument(
        "--no_cache", action="store_true",
        help="Ignore existing CSV cache and recompute all conditions from scratch.",
    )
    ring_cal_parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results (default: auto-generated in figs/ring/calibration)",
    )

    # =========================================================================
    # RING-BUMP-DECAY-STUDY subcommand
    # =========================================================================
    ring_bump_decay_parser = subparsers.add_parser(
        "ring-bump-decay-study",
        help="Assess bump decay vs. self-sustained attractor across conditions",
        description=(
            "Run cue-only ring simulations across conditions, amplitude factors, "
            "and optionally excitatory coupling values (--w_inter_values). "
            "Normalises each trial's bump amplitude timecourse by its mean value "
            "at a reference time (default: 400 ms after cue offset). "
            "Produces (1) normalised timecourse plots (mean ± SEM per condition) "
            "and (2) 2D heatmaps of mean normalised amplitude in the last 200 ms "
            "of delay, sweeping amplitude × w_inter."
        ),
    )
    _add_ring_common(ring_bump_decay_parser)
    ring_bump_decay_parser.set_defaults(delay_ms=10000.0)
    ring_bump_decay_parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to simulate (default: WT WT_APP). "
             "Valid: WT, WT_APP, a5_KO, a5_KO_APP, a7_KO, a7_KO_APP, b2_KO, b2_KO_APP",
    )
    ring_bump_decay_parser.add_argument(
        "--amplitudes", type=float, nargs="+",
        default=[5.0, 10.0, 15.0, 20.0, 25.0],
        help="Cue amplitude factors (× I_ext_pyr, default: 5 10 15 20 25).",
    )
    ring_bump_decay_parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Trials per condition × amplitude × w_inter (default: 50)",
    )
    ring_bump_decay_parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Parallel workers (default: auto)",
    )
    ring_bump_decay_parser.add_argument(
        "--w_inter_values", type=float, nargs="+", default=None,
        help="w_pyr_pyr_inter values to sweep for the 2D heatmap. "
             "If omitted, only the base --w_pyr_pyr_inter value is used "
             "(no heatmap produced).",
    )
    ring_bump_decay_parser.add_argument(
        "--ref_offset_ms", type=float, default=400.0,
        help="Time after cue offset (ms) used as the normalization reference "
             "(default: 400 ms).",
    )
    ring_bump_decay_parser.add_argument(
        "--window_ms", type=float, default=500.0,
           help="Width (ms) of time windows used for averaging the delay trajectory "
               "(default: 500 ms). Used both for the normalization reference "
               "and for all timecourse bins.",
    )
    ring_bump_decay_parser.add_argument(
        "--no_cache", action="store_true",
        help="Ignore existing pickle cache and recompute all simulations.",
    )

    # =========================================================================
    # RING-OPTIMIZE subcommand
    # =========================================================================
    ring_opt_parser = subparsers.add_parser(
        "ring-optimize",
        help="Joint optimization of circuit + ring parameters against ring-level firing rate targets",
        description=(
            "Optimize CircuitParams and RingParams simultaneously so the ring network "
            "at rest (no stimulus) reproduces the target firing rates from quiet wakefulness data. "
            "The objective includes optional trace-based Turing bistability constraints "
            "(rest stability + bump sustain around 40 Hz + anti-runaway). "
            "Legacy --bump_mode is deprecated and ignored."
        ),
    )
    from .ring.cli import add_ring_optimize_args as _add_ring_optimize_args
    _add_ring_optimize_args(ring_opt_parser)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nNo command specified. Use 'run', 'optimize', 'study', 'diagnostic', "
              "'plot-transfer', 'random-bistable-search', "
              "'ring-run', 'ring-calibrate', 'ring-bump-decay-study', 'ring-optimize'.")
        sys.exit(1)
    elif args.command == "diagnostic":
        cmd_diagnostic(args)
    elif args.command == "plot-transfer":
        cmd_plot_transfer(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "study":
        cmd_study(args)
    elif args.command == "random-bistable-search":
        cmd_random_bistable_search(args)
    elif args.command == "ring-run":
        from .ring.cli import cmd_run as cmd_ring_run
        cmd_ring_run(args)
    elif args.command == "ring-calibrate":
        from .ring.cli import cmd_calibrate as cmd_ring_calibrate
        cmd_ring_calibrate(args)
    elif args.command == "ring-bump-decay-study":
        from .ring.cli import cmd_bump_decay_study as _cmd
        _cmd(args)
    elif args.command == "ring-optimize":
        from .ring.cli import cmd_ring_optimize as _cmd
        _cmd(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
