#!/usr/bin/env python3
"""
Analyse w_pyr_pyr_inter × amplitude sweep for low_fr parameter set.

Reads bump_decay_trials.csv from each w_pv_*/  sub-directory produced by
run_low_fr_wInter_sweep.py (ring-calibrate backend) and produces:

  1. summary.json   — aggregated per-(condition, w_pv, w_inter, amplitude) metrics
  2. threshold_curve.png — minimum active amplitude vs w_inter for each w_pv
  3. heatmap_<cond>_wpv_<val>.png — 2D heatmap per (condition, w_pv)

Active criterion
----------------
  A (w_inter, amplitude) point is considered "active" when the mean
  ref_amplitude across trials exceeds ACTIVE_REF_AMP_THRESH.  This avoids
  the NaN issue with end_val_normalized (which is NaN when the reference
  amplitude ≈ 0 because the network stayed silent).

  end_val_mean is also reported when available (non-NaN fraction ≥ 0.5).

Usage
-----
  python3 scripts/analyze_low_fr_wInter_sweep.py \\
      --sweep_dir figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep \\
      --no_show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Constants ──────────────────────────────────────────────────────────────
ACTIVE_REF_AMP_THRESH = 0.05   # Hz: mean ref_amplitude above this → "active"
END_VAL_MIN_FRAC      = 0.5    # require ≥ 50 % non-NaN trials to report end_val


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sweep_dir",
        default="figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep",
        help="Root directory produced by run_low_fr_wInter_sweep.py",
    )
    p.add_argument("--no_show", action="store_true",
                   help="Do not call plt.show() (batch / headless mode)")
    p.add_argument(
        "--active_thresh", type=float, default=ACTIVE_REF_AMP_THRESH,
        help=f"ref_amplitude threshold for 'active' label (default: {ACTIVE_REF_AMP_THRESH})",
    )
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────

def load_sweep(sweep_dir: Path) -> dict[float, pd.DataFrame]:
    """Return {w_pv: DataFrame} from all w_pv_* sub-directories."""
    data: dict[float, pd.DataFrame] = {}
    wpv_dirs = sorted(sweep_dir.glob("w_pv_*"))
    if not wpv_dirs:
        raise FileNotFoundError(
            f"No w_pv_* sub-directories found in {sweep_dir}.\n"
            "Run run_low_fr_wInter_sweep.py first."
        )
    for d in wpv_dirs:
        csv = d / "bump_decay_trials.csv"
        if not csv.exists():
            print(f"  [warn] {csv} not found, skipping")
            continue
        wpv = float(d.name.replace("w_pv_", ""))
        df  = pd.read_csv(csv)
        data[wpv] = df
        print(f"  Loaded {len(df)} rows from {csv.relative_to(sweep_dir.parent.parent)}")
    return data


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame, wpv: float, active_thresh: float) -> list[dict]:
    """Aggregate per-trial rows → per-(cond, w_inter, amplitude) summary dicts."""
    records = []
    for (cond, amp, w_inter), grp in df.groupby(
        ["condition", "amplitude", "w_inter"], sort=True
    ):
        ref_amps = grp["ref_amplitude"].to_numpy(dtype=float)
        end_vals = grp["end_val_normalized"].to_numpy(dtype=float)

        valid_ev  = end_vals[~np.isnan(end_vals)]
        ev_frac   = len(valid_ev) / max(len(end_vals), 1)
        is_active = float(np.mean(ref_amps)) > active_thresh

        rec = {
            "condition":         str(cond),
            "w_pv":              float(wpv),
            "w_inter":           float(w_inter),
            "amplitude":         float(amp),
            "n_trials":          int(len(grp)),
            # Primary activity metric
            "ref_amplitude_mean": float(np.mean(ref_amps)),
            "ref_amplitude_std":  float(np.std(ref_amps)),
            "ref_amplitude_max":  float(np.max(ref_amps)),
            # Sustained bump metric (NaN when network is silent)
            "end_val_mean":      float(np.mean(valid_ev)) if ev_frac >= END_VAL_MIN_FRAC else None,
            "end_val_std":       float(np.std(valid_ev))  if ev_frac >= END_VAL_MIN_FRAC else None,
            "end_val_frac_valid": round(ev_frac, 3),
            # Active flag
            "active":            bool(is_active),
        }
        records.append(rec)
    return records


# ── Threshold extraction ───────────────────────────────────────────────────

def extract_thresholds(
    records: list[dict],
    condition: str = "WT",
) -> dict[float, dict[float, float]]:
    """
    Return nested dict:  {w_pv: {w_inter: min_active_amplitude}}

    If no active point exists for a (w_pv, w_inter) pair, the value is None.
    """
    from collections import defaultdict
    # Structure: active[w_pv][w_inter] = list of amplitudes that are active
    active: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["condition"] == condition and r["active"]:
            active[r["w_pv"]][r["w_inter"]].append(r["amplitude"])

    thresholds: dict[float, dict[float, float | None]] = {}
    for wpv, wdict in active.items():
        thresholds[wpv] = {}
        # Collect all w_inter values seen for this w_pv
        all_winters = sorted({r["w_inter"] for r in records if r["w_pv"] == wpv})
        for w in all_winters:
            amps = wdict.get(w, [])
            thresholds[wpv][w] = float(min(amps)) if amps else None

    return thresholds


# ── Plotting ───────────────────────────────────────────────────────────────

CMAP_NAME = "viridis"


def _log_norm(data_flat):
    """Return LogNorm(vmin, vmax) for data, ignoring zeros/NaNs."""
    pos = data_flat[np.isfinite(data_flat) & (data_flat > 0)]
    if len(pos) == 0:
        return mcolors.LogNorm(vmin=1e-10, vmax=1.0)
    return mcolors.LogNorm(vmin=float(np.min(pos)), vmax=float(np.max(pos)))


def plot_heatmap(
    records: list[dict],
    condition: str,
    wpv: float,
    save_path: Path,
    no_show: bool,
) -> plt.Figure:
    """2D heatmap: w_inter (x) × amplitude (y), colour = ref_amplitude_mean."""
    sub = [r for r in records if r["condition"] == condition and r["w_pv"] == wpv]
    if not sub:
        return None

    w_vals  = sorted({r["w_inter"]   for r in sub})
    a_vals  = sorted({r["amplitude"] for r in sub})
    nx, ny  = len(w_vals), len(a_vals)
    wi_idx  = {w: i for i, w in enumerate(w_vals)}
    am_idx  = {a: i for i, a in enumerate(a_vals)}

    ref_grid = np.full((ny, nx), np.nan)
    ev_grid  = np.full((ny, nx), np.nan)
    for r in sub:
        xi = wi_idx[r["w_inter"]]
        yi = am_idx[r["amplitude"]]
        ref_grid[yi, xi] = r["ref_amplitude_mean"]
        if r["end_val_mean"] is not None:
            ev_grid[yi, xi] = r["end_val_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Low_fr  |  w_pv = {wpv:.4g}  |  condition = {condition}\n"
        "Hypothesis: higher w_pyr_pyr_inter lowers amplitude threshold",
        fontsize=11,
    )

    # ── Panel 1: ref_amplitude (log scale) ──
    ax = axes[0]
    norm = _log_norm(ref_grid.ravel())
    im = ax.imshow(
        ref_grid, aspect="auto", origin="lower",
        norm=norm, cmap=CMAP_NAME,
    )
    ax.set_xticks(range(nx))
    ax.set_xticklabels([f"{w:.3g}" for w in w_vals], rotation=45, ha="right")
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{a:.3g}" for a in a_vals])
    ax.set_xlabel("w_pyr_pyr_inter")
    ax.set_ylabel("Amplitude (× I_ext_pyr)")
    ax.set_title("ref_amplitude (mean, log scale)")
    fig.colorbar(im, ax=ax, label="ref_amplitude (Hz)")

    # Draw threshold boundary
    for yi, a in enumerate(a_vals):
        for xi, w in enumerate(w_vals):
            if ref_grid[yi, xi] is not None and not np.isnan(ref_grid[yi, xi]):
                if ref_grid[yi, xi] > ACTIVE_REF_AMP_THRESH:
                    ax.add_patch(plt.Rectangle(
                        (xi - 0.5, yi - 0.5), 1, 1,
                        fill=False, edgecolor="red", linewidth=1.5,
                    ))

    # ── Panel 2: end_val_mean ──
    ax = axes[1]
    im2 = ax.imshow(
        np.where(np.isnan(ev_grid), -0.05, ev_grid),
        aspect="auto", origin="lower",
        vmin=0.0, vmax=1.0, cmap="RdYlGn",
    )
    ax.set_xticks(range(nx))
    ax.set_xticklabels([f"{w:.3g}" for w in w_vals], rotation=45, ha="right")
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{a:.3g}" for a in a_vals])
    ax.set_xlabel("w_pyr_pyr_inter")
    ax.set_ylabel("Amplitude (× I_ext_pyr)")
    ax.set_title("end_val_mean (normalised bump at end of delay)")
    fig.colorbar(im2, ax=ax, label="end_val (0=decayed, 1=sustained)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    if not no_show:
        plt.show()
    return fig


def plot_threshold_curve(
    thresholds: dict[float, dict[float, float | None]],
    save_path: Path,
    no_show: bool,
) -> plt.Figure:
    """Line plot: minimum active amplitude vs w_inter, one line per w_pv."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colours = plt.cm.plasma(np.linspace(0.1, 0.8, len(thresholds)))
    for (wpv, wdict), colour in zip(sorted(thresholds.items()), colours):
        w_vals = sorted(wdict.keys())
        amp_vals = [wdict[w] for w in w_vals]
        # Replace None with NaN for plotting
        amp_plot = [a if a is not None else np.nan for a in amp_vals]
        ax.plot(w_vals, amp_plot, "o-", color=colour, label=f"w_pv = {wpv:.4g}")

    # Horizontal reference lines
    ax.axhline(7.0, color="red",    linestyle="--", linewidth=1, alpha=0.7,
               label="Old threshold (w_inter=0.002)")
    ax.axhline(1.5, color="green",  linestyle="--", linewidth=1, alpha=0.7,
               label="Physiological target (~1.5×)")

    ax.set_xscale("log")
    ax.set_xlabel("w_pyr_pyr_inter", fontsize=12)
    ax.set_ylabel("Min active amplitude (× I_ext_pyr)", fontsize=12)
    ax.set_title(
        "Low_fr: bifurcation threshold vs recurrent PYR coupling\n"
        "(lower is better — closer to biologically plausible range)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    if not no_show:
        plt.show()
    return fig


# ── JSON helpers ───────────────────────────────────────────────────────────

def _to_json_safe(obj):
    """Recursively convert numpy scalars and None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    print(f"\nLoading sweep data from: {sweep_dir}")
    data = load_sweep(sweep_dir)
    if not data:
        print("No data found — did the sweep run complete?")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────
    print("\nAggregating...")
    all_records: list[dict] = []
    for wpv, df in sorted(data.items()):
        recs = aggregate(df, wpv, args.active_thresh)
        all_records.extend(recs)
        n_active = sum(1 for r in recs if r["active"])
        print(f"  w_pv={wpv:.4g}: {len(recs)} combos, {n_active} active")

    # ── JSON summary ───────────────────────────────────────────────────────
    thresholds = extract_thresholds(all_records, condition="WT")

    summary = {
        "sweep_dir":            str(sweep_dir),
        "active_ref_amp_thresh": args.active_thresh,
        "n_records":            len(all_records),
        # Threshold table: w_pv → w_inter → min_active_amplitude
        "threshold_by_wpv_winter": {
            str(wpv): {str(w): v for w, v in wdict.items()}
            for wpv, wdict in thresholds.items()
        },
        "records": all_records,
    }

    json_path = sweep_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(_to_json_safe(summary), f, indent=2)
    print(f"\nSummary JSON → {json_path}")

    # Pretty-print threshold table to console
    print("\n── Threshold table (min amplitude for active bump, condition=WT) ──")
    all_winters = sorted({r["w_inter"] for r in all_records})
    all_wpvs    = sorted(thresholds.keys())
    header = "w_inter   " + "  ".join(f"w_pv={wpv:.4g}" for wpv in all_wpvs)
    print(header)
    print("-" * len(header))
    for w in all_winters:
        row = f"{w:<10.4g}"
        for wpv in all_wpvs:
            val = thresholds.get(wpv, {}).get(w)
            row += f"  {str(val) if val is not None else 'SILENT':>14}"
        print(row)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    conditions = sorted({r["condition"] for r in all_records})
    wpv_vals   = sorted({r["w_pv"]      for r in all_records})

    for cond in conditions:
        for wpv in wpv_vals:
            hm_path = sweep_dir / f"heatmap_{cond}_wpv_{wpv:.4g}.png"
            fig = plot_heatmap(all_records, cond, wpv, hm_path, args.no_show)
            if fig is not None:
                plt.close(fig)

    tc_path = sweep_dir / "threshold_curve.png"
    fig = plot_threshold_curve(thresholds, tc_path, args.no_show)
    plt.close(fig)

    print("\nDone.")
    print(f"  summary.json        → {json_path}")
    print(f"  threshold_curve.png → {tc_path}")
    print(f"  heatmap_*.png       → {sweep_dir}/heatmap_*.png")


if __name__ == "__main__":
    main()
