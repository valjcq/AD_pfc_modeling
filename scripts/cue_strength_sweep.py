#!/usr/bin/env python3
"""
Cue strength sweep: firing rates vs. transient input amplitude.

Runs single-node simulations with a brief cue (transient current) at varying
strengths and reports population firing rates before, during, and after the cue.

Goal: check whether PYR can reach ~60 Hz during/after the cue and whether
any elevated state persists once the cue is removed (bistability check).

Usage:
    python scripts/cue_strength_sweep.py --params_json figs/optim/bistable_fixed/best_params.json
    python scripts/cue_strength_sweep.py --params_json figs/optim/bistable_fixed/best_params.json \
        --factor_min 0 --factor_max 4 --n_steps 20 --no_show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.io import load_params_json
from circuit_model.simulation import simulate_circuit

POPULATIONS = ["PYR", "SOM", "PV", "VIP"]
COLORS = {"pre": "#888888", "during": "#e07b39", "post": "#3a82b5"}


def mean_in_window(result, t_start, t_end):
    """Mean firing rates in a time window [t_start, t_end] ms."""
    mask = (result.t_ms >= t_start) & (result.t_ms <= t_end)
    if not mask.any():
        return np.zeros(4)
    return result.r[mask].mean(axis=0)


def run_sweep(params_json, factor_min, factor_max, n_steps,
              T_ms, trans_start_ms, trans_duration_ms,
              noise_type, seed):

    params_base = load_params_json(params_json)

    factors = np.linspace(factor_min, factor_max, n_steps)

    # Time windows
    pre_window   = (500.0, trans_start_ms - 1.0)          # settled rest before cue
    during_window = (trans_start_ms, trans_start_ms + trans_duration_ms)
    post_window  = (trans_start_ms + trans_duration_ms + 200.0, T_ms)  # 200 ms buffer

    # Results: shape (n_steps, 3 windows, 4 populations)
    rates = np.zeros((n_steps, 3, 4))

    print(f"\nParams: {params_json}")
    print(f"Transient: start={trans_start_ms} ms, duration={trans_duration_ms} ms")
    print(f"Windows — pre: {pre_window}, during: {during_window}, post: {post_window}")
    print(f"Sweeping {n_steps} factors from {factor_min:.2f} to {factor_max:.2f}\n")

    for i, factor in enumerate(factors):
        from dataclasses import replace
        p = replace(
            params_base,
            trans_enabled=True,
            trans_start_ms=trans_start_ms,
            trans_duration_ms=trans_duration_ms,
            trans_factor=factor,
        )

        result = simulate_circuit(
            p,
            T_ms=T_ms,
            dt_ms=0.1,
            seed=seed,
            noise_type=noise_type,
            use_transient=True,
        )

        rates[i, 0] = mean_in_window(result, *pre_window)
        rates[i, 1] = mean_in_window(result, *during_window)
        rates[i, 2] = mean_in_window(result, *post_window)

    return factors, rates


def print_table(factors, rates):
    pops = POPULATIONS
    windows = ["pre", "during", "post"]
    header = f"{'factor':>8} | " + " | ".join(
        f"{w:^28}" for w in windows
    )
    subheader = " " * 9 + "| " + " | ".join(
        "  ".join(f"{p:>6}" for p in pops) for _ in windows
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(subheader)
    print(sep)
    for i, factor in enumerate(factors):
        row = f"{factor:>8.3f} | "
        row += " | ".join(
            "  ".join(f"{rates[i, w, p]:>6.1f}" for p in range(4))
            for w in range(3)
        )
        print(row)
    print(sep)


def plot_sweep(factors, rates, params_json, save_path, no_show):
    windows = ["pre", "during", "post"]
    labels  = {"pre": "Pre-cue (rest)", "during": "During cue", "post": "Post-cue"}
    markers = {"pre": "o", "during": "s", "post": "^"}

    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(
        f"Cue strength sweep\n{Path(params_json).name}",
        fontsize=11, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    for pi, (ax, pop) in enumerate(zip(axes, POPULATIONS)):
        for wi, w in enumerate(windows):
            ax.plot(
                factors, rates[:, wi, pi],
                color=COLORS[w], marker=markers[w], markersize=4,
                linewidth=1.5, label=labels[w]
            )
        ax.set_title(pop, fontweight="bold")
        ax.set_xlabel("trans_factor (× I₀)")
        ax.set_ylabel("Mean rate (Hz)")
        ax.grid(True, alpha=0.3)
        if pi == 0:
            ax.legend(fontsize=8)
            ax.axhline(60, color="red", linestyle="--", linewidth=0.8, label="60 Hz target")
            ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params_json", required=True,
                        help="Path to CircuitParams JSON file")
    parser.add_argument("--factor_min", type=float, default=0.0,
                        help="Minimum trans_factor (default: 0)")
    parser.add_argument("--factor_max", type=float, default=4.0,
                        help="Maximum trans_factor (default: 4)")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Number of factor values to sweep (default: 20)")
    parser.add_argument("--T_ms", type=float, default=4000.0,
                        help="Total simulation duration in ms (default: 4000)")
    parser.add_argument("--trans_start_ms", type=float, default=1000.0,
                        help="Cue onset time in ms (default: 1000)")
    parser.add_argument("--trans_duration_ms", type=float, default=200.0,
                        help="Cue duration in ms (default: 200)")
    parser.add_argument("--noise_type", choices=["none", "white", "ou"], default="white",
                        help="Noise type (default: white)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save_plot", type=str, default="",
                        help="Save figure to this path")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not display the figure")
    args = parser.parse_args()

    factors, rates = run_sweep(
        args.params_json,
        args.factor_min, args.factor_max, args.n_steps,
        args.T_ms, args.trans_start_ms, args.trans_duration_ms,
        args.noise_type, args.seed,
    )

    print_table(factors, rates)

    save_path = args.save_plot or str(
        Path(args.params_json).parent / f"cue_sweep_{Path(args.params_json).stem}.png"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_sweep(factors, rates, args.params_json, save_path, args.no_show)


if __name__ == "__main__":
    main()
