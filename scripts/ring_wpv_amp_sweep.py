#!/usr/bin/env python3
"""
Ring network sweep: w_pv_global × amplitude heatmap.

Runs ring-run for a 2-D grid:
  w_pv_global : 0.05, 0.06, ..., 0.10  (6 values)
  amplitude   : 0.40, 0.45, ..., 0.80  (9 values)
                                         total: 54 runs

--no_snapshot_mp4 is always set for speed.
Results are cached; re-running the script only executes missing grid points.

Produces a 4-panel heatmap:
  1. Pre-cue baseline PYR rate
  2. Peak PYR rate during cue  (200 Hz = saturation cap)
  3. Post-cue centre PYR rate  (proxy for bump sustenance)
  4. Bump centre std           (low = localised bump)

Usage:
    python scripts/ring_wpv_amp_sweep.py
    python scripts/ring_wpv_amp_sweep.py --n_workers 6 --no_show
    python scripts/ring_wpv_amp_sweep.py --sigma_pyr_deg 15 --n_workers 6 --no_show
    python scripts/ring_wpv_amp_sweep.py --no_run --no_show   # only plot cached data
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Static paths / grid constants
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
PARAMS_JSON = ROOT / "figs/optim/bistable_high_fr/best_params.json"

W_PYR     = 0.002
DELAY_MS  = 2000
CONDITION = "WT"
SEED      = 442

WPV_VALUES = [round(v, 4) for v in np.arange(0.05, 0.101, 0.01)]   # [0.05 … 0.10]
AMP_VALUES = [round(v, 3) for v in np.arange(0.40, 0.801, 0.05)]   # [0.40 … 0.80]

# Runtime context — populated by main() before any run_* call
_SIGMA_DEG: float = 30.0
_SWEEP_DIR: Path  = ROOT / "figs/ring/sweep/wpv_amp_sweep"


def _sweep_dir_for(sigma: float) -> Path:
    if sigma == 30.0:
        return ROOT / "figs/ring/sweep/wpv_amp_sweep"
    return ROOT / f"figs/ring/sweep/sigma{int(sigma)}/wpv_amp_sweep"


# ---------------------------------------------------------------------------
# Path helpers (replicates cli._fmt logic)
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    if abs(v) < 0.1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def run_dir(wpv: float, amp: float) -> Path:
    return _SWEEP_DIR / f"wpv{_fmt(wpv)}_amp{_fmt(amp)}"


# ---------------------------------------------------------------------------
# Single-point runner
# ---------------------------------------------------------------------------
def run_single(wpv: float, amp: float) -> dict | None:
    """Launch one ring-run subprocess and return its metrics dict."""
    out = run_dir(wpv, amp)
    metrics_path = out / "run_metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "circuit_model", "ring-run",
        "--params_json",     str(PARAMS_JSON),
        "--w_pyr_pyr_inter", str(W_PYR),
        "--w_pv_global",     str(wpv),
        "--amplitude",       str(amp),
        "--sigma_pyr_deg",   str(_SIGMA_DEG),
        "--delay_ms",        str(DELAY_MS),
        "--condition",       CONDITION,
        "--seed",            str(SEED),
        "--no_show",
        "--no_snapshot_mp4",
        "--output_dir",      str(out),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  [FAIL] wpv={wpv} amp={amp}\n{result.stderr[-500:]}")
        return None

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def _worker(args):
    wpv, amp = args
    try:
        return wpv, amp, run_single(wpv, amp), None
    except Exception as exc:
        return wpv, amp, None, str(exc)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
def run_grid(n_workers: int) -> dict:
    """Run all missing grid points; return full results dict."""
    grid = [(wpv, amp) for wpv in WPV_VALUES for amp in AMP_VALUES]
    missing = [(wpv, amp) for wpv, amp in grid
               if not (run_dir(wpv, amp) / "run_metrics.json").exists()]

    total = len(grid)
    cached_n = total - len(missing)
    print(f"Grid: {total} points — {cached_n} cached, {len(missing)} to run.")

    results: dict[tuple, dict] = {}

    # Load already-done results
    for wpv, amp in grid:
        p = run_dir(wpv, amp) / "run_metrics.json"
        if p.exists():
            with open(p) as f:
                results[(wpv, amp)] = json.load(f)

    if not missing:
        return results

    done = cached_n
    if n_workers == 1:
        for wpv, amp in missing:
            print(f"  Running wpv={wpv:.2f} amp={amp:.2f} …", flush=True)
            _, _, m, err = _worker((wpv, amp))
            done += 1
            if err:
                print(f"    ERROR: {err}")
            elif m:
                results[(wpv, amp)] = m
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, g): g for g in missing}
            for fut in as_completed(futures):
                wpv, amp, m, err = fut.result()
                done += 1
                tag = "ERROR" if err else ("ok" if m else "no metrics")
                print(f"  [{done}/{total}] wpv={wpv:.2f} amp={amp:.2f} → {tag}", flush=True)
                if m:
                    results[(wpv, amp)] = m

    return results


# ---------------------------------------------------------------------------
# Build metric arrays
# ---------------------------------------------------------------------------
def build_arrays(results: dict):
    """Return (baseline, peak_cue, delay_ctr, ctr_std) as (n_wpv × n_amp) arrays."""
    n_wpv, n_amp = len(WPV_VALUES), len(AMP_VALUES)
    baseline  = np.full((n_wpv, n_amp), np.nan)
    peak_cue  = np.full((n_wpv, n_amp), np.nan)
    delay_ctr = np.full((n_wpv, n_amp), np.nan)
    ctr_std   = np.full((n_wpv, n_amp), np.nan)

    for i, wpv in enumerate(WPV_VALUES):
        for j, amp in enumerate(AMP_VALUES):
            m = results.get((wpv, amp))
            if m is None:
                continue
            ss = m.get("steady_state", {})
            bm = m.get("bump_metrics", {})
            baseline[i, j]  = ss.get("baseline_pyr_hz",   np.nan)
            peak_cue[i, j]  = ss.get("peak_pyr_cue_hz",   np.nan)
            delay_ctr[i, j] = ss.get("delay_pyr_center_hz", np.nan)
            ctr_std[i, j]   = bm.get("center_std_deg",    np.nan)

    return baseline, peak_cue, delay_ctr, ctr_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
SAT_CAP = 200.0


def plot_heatmaps(baseline, peak_cue, delay_ctr, ctr_std, save_path: Path, no_show: bool):
    n_wpv, n_amp = len(WPV_VALUES), len(AMP_VALUES)
    amp_labels = [f"{v:.2f}" for v in AMP_VALUES]
    wpv_labels = [f"{v:.2f}" for v in WPV_VALUES]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Ring sweep  —  w_pv_global × amplitude\n"
        f"w_pyr_pyr_inter={W_PYR}  |  sigma={_SIGMA_DEG}°  |  delay={DELAY_MS} ms  |  {CONDITION}",
        fontsize=12, fontweight="bold",
    )

    def draw(ax, data, title, vmin, vmax, cmap, fmt=".1f", annot_thresh_dark=0.6):
        data_plot = np.where(np.isnan(data), np.nan, data)
        im = ax.imshow(data_plot, aspect="auto", vmin=vmin, vmax=vmax,
                       cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_xticks(range(n_amp))
        ax.set_xticklabels(amp_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_wpv))
        ax.set_yticklabels(wpv_labels, fontsize=9)
        ax.set_xlabel("Amplitude (× I_ext_pyr)", fontsize=9)
        ax.set_ylabel("w_pv_global", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(n_wpv):
            for j in range(n_amp):
                v = data[i, j]
                if np.isnan(v):
                    continue
                text_color = "white" if norm(v) > annot_thresh_dark else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    # --- Panel 1: pre-cue baseline ---
    draw(axes[0, 0], baseline,
         "Pre-cue baseline PYR (Hz)\n[target: ≈ 0–3 Hz]",
         vmin=0, vmax=max(10.0, float(np.nanmax(baseline)) if not np.all(np.isnan(baseline)) else 10),
         cmap="YlOrRd", fmt=".1f")

    # --- Panel 2: peak during cue ---
    peak_plot = np.clip(peak_cue, 0, SAT_CAP)
    draw(axes[0, 1], peak_plot,
         f"Peak PYR during cue (Hz)\n[{SAT_CAP:.0f} Hz = saturated at cap]",
         vmin=0, vmax=SAT_CAP, cmap="RdYlGn_r", fmt=".0f", annot_thresh_dark=0.55)
    sat_mask = (peak_cue >= SAT_CAP - 0.5).astype(float)
    if sat_mask.any() and not sat_mask.all():
        axes[0, 1].contour(sat_mask, levels=[0.5], colors="cyan",
                           linewidths=2, linestyles="--")
        axes[0, 1].text(
            0.02, 0.96, "— sat. boundary",
            transform=axes[0, 1].transAxes, fontsize=7, color="cyan", va="top"
        )

    # --- Panel 3: delay centre rate ---
    delay_max = max(15.0, float(np.nanmax(delay_ctr)) if not np.all(np.isnan(delay_ctr)) else 15)
    draw(axes[1, 0], delay_ctr,
         "Post-cue centre PYR (Hz)\n[higher = bump sustained]",
         vmin=0, vmax=delay_max, cmap="YlGn", fmt=".1f")

    # --- Panel 4: bump centre std ---
    draw(axes[1, 1], np.clip(ctr_std, 0, 180),
         "Bump centre std (deg)\n[< 30° = well-localised]",
         vmin=0, vmax=180, cmap="RdYlGn", fmt=".0f", annot_thresh_dark=0.4)
    axes[1, 1].contour((ctr_std < 45).astype(float), levels=[0.5],
                        colors="blue", linewidths=1.5, linestyles=":")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved: {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _SIGMA_DEG, _SWEEP_DIR

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sigma_pyr_deg", type=float, default=30.0,
                        help="Gaussian excitation kernel width in degrees (default: 30)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Parallel workers for ring-run (default: 4)")
    parser.add_argument("--no_run", action="store_true",
                        help="Skip running simulations; only plot cached data")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not display the figure interactively")
    args = parser.parse_args()

    _SIGMA_DEG = args.sigma_pyr_deg
    _SWEEP_DIR = _sweep_dir_for(_SIGMA_DEG)
    _SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_run:
        print("--no_run: loading cached results only.")
        results = {}
        for wpv in WPV_VALUES:
            for amp in AMP_VALUES:
                p = run_dir(wpv, amp) / "run_metrics.json"
                if p.exists():
                    with open(p) as f:
                        results[(wpv, amp)] = json.load(f)
        n_found = len(results)
        print(f"Found {n_found} / {len(WPV_VALUES)*len(AMP_VALUES)} cached results.")
    else:
        results = run_grid(n_workers=args.n_workers)

    if not results:
        print("No results to plot.")
        return

    baseline, peak_cue, delay_ctr, ctr_std = build_arrays(results)
    plot_heatmaps(
        baseline, peak_cue, delay_ctr, ctr_std,
        save_path=_SWEEP_DIR / "heatmap_wpv_amp.png",
        no_show=args.no_show,
    )

    # Print summary table
    print(f"\n{'wpv':>6}  {'amp':>6}  {'base':>6}  {'peak':>6}  {'delay_c':>8}  {'cstd':>7}")
    print("-" * 54)
    for wpv in WPV_VALUES:
        for amp in AMP_VALUES:
            m = results.get((wpv, amp))
            if m is None:
                print(f"{wpv:>6.2f}  {amp:>6.2f}  {'N/A':>6}")
                continue
            ss = m.get("steady_state", {})
            bm = m.get("bump_metrics", {})
            b   = ss.get("baseline_pyr_hz",    float("nan"))
            pk  = ss.get("peak_pyr_cue_hz",    float("nan"))
            dc  = ss.get("delay_pyr_center_hz", float("nan"))
            cs  = bm.get("center_std_deg",     float("nan"))
            sat = " SAT" if (not np.isnan(pk) and pk >= SAT_CAP - 0.5) else ""
            print(f"{wpv:>6.2f}  {amp:>6.2f}  {b:>6.2f}  {pk:>6.1f}{sat}  {dc:>8.3f}  {cs:>7.1f}")


if __name__ == "__main__":
    main()
