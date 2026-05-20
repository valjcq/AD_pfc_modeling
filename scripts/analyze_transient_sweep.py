#!/usr/bin/env python3
"""
Analyse 2D transient sweep and produce heatmaps.

Creates 2D heatmaps (Duration × Amplitude) for each polarity showing:
  • Spatial variance of PYR rate (high → localised bump preserved)
  • Fraction of nodes firing > 30 Hz (low → silent, high → all active)

Prints best SILENT state conditions (lowest variance, lowest activity).

Usage:
    python3 scripts/analyze_transient_sweep.py [--base-dir DIR] [--no-show]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default="figs/ring/run/transient_sweep")
    p.add_argument("--no-show",  action="store_true")
    p.add_argument("--save",     default="", help="Save heatmaps to this path (optional)")
    return p.parse_args()


def load_2d_sweep(sweep_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Load 2D grid data (Duration × Amplitude) from sweep directory.

    Dynamically discovers and loads all runs, handling missing combinations gracefully.
    Issues warnings if coverage is incomplete.

    Returns:
        (var_matrix, frac30_matrix, state_matrix, durations, amplitudes)
    """
    if not sweep_dir.exists():
        return None, None, None, [], []

    # Discover all runs and extract (duration, amplitude) pairs
    runs = {}
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        # Parse directory name: "XXms_YY.ZZ"
        parts = run_dir.name.split('_')
        if len(parts) != 2:
            continue
        try:
            dur_ms = int(parts[0].replace('ms', ''))
            amp = float(parts[1])
            mf = run_dir / "run_metrics.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                m = json.load(f)
            pt = m.get("post_transient")
            if pt is None:
                continue
            runs[(dur_ms, amp)] = pt
        except (ValueError, KeyError):
            continue

    if not runs:
        return None, None, None, [], []

    # Extract unique durations and amplitudes, sorted
    durations = sorted(set(k[0] for k in runs.keys()))
    amplitudes = sorted(set(k[1] for k in runs.keys()))

    # Build matrices: rows=durations, cols=amplitudes
    n_dur, n_amp = len(durations), len(amplitudes)
    var_matrix = np.full((n_dur, n_amp), np.nan)
    frac30_matrix = np.full((n_dur, n_amp), np.nan)
    state_matrix = np.empty((n_dur, n_amp), dtype=object)

    n_missing = 0
    for i, dur in enumerate(durations):
        for j, amp in enumerate(amplitudes):
            if (dur, amp) in runs:
                r = runs[(dur, amp)]
                var_matrix[i, j] = r.get("post_var_pyr_hz", np.nan)
                frac30_matrix[i, j] = r.get("post_frac_above_30hz", np.nan)
                state_matrix[i, j] = r.get("post_state", "UNKNOWN")
            else:
                state_matrix[i, j] = "MISSING"
                n_missing += 1

    # Check coverage and warn if incomplete
    coverage = (n_dur * n_amp - n_missing) / (n_dur * n_amp) * 100 if n_dur * n_amp > 0 else 100
    if coverage < 95:
        print(f"⚠️  WARNING ({sweep_dir.name}): Only {coverage:.1f}% coverage ({n_dur*n_amp - n_missing}/{n_dur*n_amp} cells)")
        print(f"    {n_missing} cells missing → white in heatmap")
        print(f"    Run: .venv/bin/python3 scripts/run_transient_sweep.py --workers 10")

    return var_matrix, frac30_matrix, state_matrix, durations, amplitudes


def find_best_silent(var_matrix, frac30_matrix, state_matrix, durations, amplitudes):
    """Find conditions with best SILENT state (lowest variance & activity)."""
    best = []
    for i, dur in enumerate(durations):
        for j, amp in enumerate(amplitudes):
            if state_matrix[i, j] == "SILENT":
                best.append({
                    "duration_ms": dur,
                    "amplitude": amp,
                    "variance": var_matrix[i, j],
                    "frac_30hz": frac30_matrix[i, j],
                })

    # Sort by variance (primary), then frac_30hz (secondary)
    best.sort(key=lambda x: (x["variance"], x["frac_30hz"]))
    return best[:5] if best else []


def plot_2d_heatmap(matrix, durations, amplitudes, title, cmap, ax, cbar_label):
    """Plot a single 2D heatmap."""
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(amplitudes)))
    ax.set_yticks(np.arange(len(durations)))
    ax.set_xticklabels([f"{a:.2f}" for a in amplitudes], rotation=45, fontsize=8)
    ax.set_yticklabels([f"{d}" for d in durations], fontsize=8)
    ax.set_xlabel("Amplitude (fraction of I₀)", fontsize=9)
    ax.set_ylabel("Duration (ms)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    return im


def main():
    args = parse_args()
    base = Path(args.base_dir)

    if not base.exists():
        print(f"Error: {base} not found. Run the sweep first.")
        return

    # Load negative and positive sweeps
    neg_var, neg_frac30, neg_state, neg_dur, neg_amp = load_2d_sweep(base / "negative")
    pos_var, pos_frac30, pos_state, pos_dur, pos_amp = load_2d_sweep(base / "positive")

    if neg_var is None and pos_var is None:
        print("No completed runs found (no run_metrics.json with post_transient section).")
        return

    # Create figure with 2×2 subplots (NEG var, NEG frac30, POS var, POS frac30)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Transient Sweep — 2D Heatmaps (Duration × Amplitude)", fontsize=13,
                 fontweight="bold")

    # Negative heatmaps
    if neg_var is not None:
        plot_2d_heatmap(neg_var, neg_dur, neg_amp, "NEG — Spatial Variance (Hz²)",
                       "viridis", axes[0, 0], "Variance")
        plot_2d_heatmap(neg_frac30, neg_dur, neg_amp, "NEG — Fraction nodes > 30 Hz",
                       "RdYlGn_r", axes[0, 1], "Fraction")
    else:
        axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 1].transAxes)

    # Positive heatmaps
    if pos_var is not None:
        plot_2d_heatmap(pos_var, pos_dur, pos_amp, "POS — Spatial Variance (Hz²)",
                       "viridis", axes[1, 0], "Variance")
        plot_2d_heatmap(pos_frac30, pos_dur, pos_amp, "POS — Fraction nodes > 30 Hz",
                       "RdYlGn_r", axes[1, 1], "Fraction")
    else:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 1].transAxes)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")

    if not args.no_show:
        plt.show()

    # Print best SILENT state conditions
    print("\n" + "=" * 70)
    print("BEST SILENT STATE CONDITIONS (lowest variance, lowest activity)")
    print("=" * 70)

    if neg_var is not None:
        best_neg = find_best_silent(neg_var, neg_frac30, neg_state, neg_dur, neg_amp)
        print("\nNEGATIVE transient — Top 5 best SILENT conditions:")
        if best_neg:
            print(f"{'Rank':<5} {'Duration (ms)':<15} {'Amplitude':<12} {'Variance':<15} {'Frac>30Hz':<15}")
            print("-" * 70)
            for i, cond in enumerate(best_neg, 1):
                print(f"{i:<5} {cond['duration_ms']:<15} {cond['amplitude']:<12.4f} "
                      f"{cond['variance']:<15.4f} {cond['frac_30hz']:<15.4f}")
        else:
            print("No SILENT state found")

    if pos_var is not None:
        best_pos = find_best_silent(pos_var, pos_frac30, pos_state, pos_dur, pos_amp)
        print("\nPOSITIVE transient — Top 5 best SILENT conditions:")
        if best_pos:
            print(f"{'Rank':<5} {'Duration (ms)':<15} {'Amplitude':<12} {'Variance':<15} {'Frac>30Hz':<15}")
            print("-" * 70)
            for i, cond in enumerate(best_pos, 1):
                print(f"{i:<5} {cond['duration_ms']:<15} {cond['amplitude']:<12.4f} "
                      f"{cond['variance']:<15.4f} {cond['frac_30hz']:<15.4f}")
        else:
            print("No SILENT state found")

    print("=" * 70)


if __name__ == "__main__":
    main()
