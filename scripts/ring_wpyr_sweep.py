#!/usr/bin/env python3
"""
Ring network sweep: w_pyr_pyr_inter (1-D).

Fixed: w_pv_global=0.05, amplitude=0.55  (best from Phase 4 wpv×amp sweep)

w_pyr_pyr_inter values: ~12 log-spaced points from 0.001 to 0.05

--no_snapshot_mp4 is always set for speed.
Results are cached; re-running the script only executes missing points.

Produces a 4-panel figure (x = w_pyr_pyr_inter, log scale):
  1. Pre-cue baseline PYR rate
  2. Peak PYR rate during cue  (200 Hz = saturation cap)
  3. Post-cue centre PYR rate  (proxy for bump sustenance)
  4. Bump centre std           (low = localised bump)

Usage:
    python scripts/ring_wpyr_sweep.py
    python scripts/ring_wpyr_sweep.py --n_workers 6 --no_show
    python scripts/ring_wpyr_sweep.py --w_pv_global 0.05 --amplitude 0.55 --n_workers 6 --no_show
    python scripts/ring_wpyr_sweep.py --no_run --no_show   # only plot cached data
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and grid
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
PARAMS_JSON = ROOT / "figs/optim/bistable_high_fr/best_params.json"

# Fixed from Phase 4 optimum
DEFAULT_WPV   = 0.05
DEFAULT_AMP   = 0.55
DEFAULT_SIGMA = 30.0
DELAY_MS      = 2000
CONDITION     = "WT"
SEED          = 442

# Log-spaced w_pyr_pyr_inter values: 0.001 → 0.05
WPYR_VALUES = [round(v, 6) for v in np.logspace(np.log10(0.001), np.log10(0.05), 12)]

# Runtime context — populated by main()
_SIGMA_DEG: float = DEFAULT_SIGMA
_SWEEP_DIR: Path  = ROOT / "figs/ring/sweep/wpyr_sweep"


def _sweep_dir_for(sigma: float) -> Path:
    if sigma == 30.0:
        return ROOT / "figs/ring/sweep/wpyr_sweep"
    return ROOT / f"figs/ring/sweep/sigma{int(sigma)}/wpyr_sweep"


# ---------------------------------------------------------------------------
# Path helpers (replicates cli._fmt logic)
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    if abs(v) < 0.1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def run_dir(wpyr: float, wpv: float, amp: float) -> Path:
    return _SWEEP_DIR / f"wpyr{_fmt(wpyr)}_wpv{_fmt(wpv)}_amp{_fmt(amp)}"


# ---------------------------------------------------------------------------
# Single-point runner
# ---------------------------------------------------------------------------
def run_single(wpyr: float, wpv: float, amp: float) -> dict | None:
    """Launch one ring-run subprocess and return its metrics dict."""
    out = run_dir(wpyr, wpv, amp)
    metrics_path = out / "run_metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "circuit_model", "ring-run",
        "--params_json",     str(PARAMS_JSON),
        "--w_pyr_pyr_inter", str(wpyr),
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
        print(f"\n  [FAIL] wpyr={wpyr} wpv={wpv} amp={amp}\n{result.stderr[-500:]}")
        return None

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def _worker(args):
    wpyr, wpv, amp = args
    try:
        return wpyr, run_single(wpyr, wpv, amp), None
    except Exception as exc:
        return wpyr, None, str(exc)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
def run_grid(wpv: float, amp: float, n_workers: int) -> dict:
    """Run all missing points; return full results dict keyed by wpyr."""
    missing = [wpyr for wpyr in WPYR_VALUES
               if not (run_dir(wpyr, wpv, amp) / "run_metrics.json").exists()]

    total    = len(WPYR_VALUES)
    cached_n = total - len(missing)
    print(f"Sweep: {total} points — {cached_n} cached, {len(missing)} to run.")

    results: dict[float, dict] = {}

    # Load already-done results
    for wpyr in WPYR_VALUES:
        p = run_dir(wpyr, wpv, amp) / "run_metrics.json"
        if p.exists():
            with open(p) as f:
                results[wpyr] = json.load(f)

    if not missing:
        return results

    done = cached_n
    args_list = [(wpyr, wpv, amp) for wpyr in missing]

    if n_workers == 1:
        for a in args_list:
            wpyr, m, err = _worker(a)
            done += 1
            if err:
                print(f"  [{done}/{total}] wpyr={wpyr:.5f} → ERROR: {err}")
            elif m:
                print(f"  [{done}/{total}] wpyr={wpyr:.5f} → ok")
                results[wpyr] = m
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, a): a for a in args_list}
            for fut in as_completed(futures):
                wpyr, m, err = fut.result()
                done += 1
                tag = "ERROR" if err else ("ok" if m else "no metrics")
                print(f"  [{done}/{total}] wpyr={wpyr:.5f} → {tag}", flush=True)
                if m:
                    results[wpyr] = m

    return results


# ---------------------------------------------------------------------------
# Build metric arrays
# ---------------------------------------------------------------------------
def build_arrays(results: dict):
    """Return (baseline, peak_cue, delay_ctr, ctr_std) as 1-D arrays."""
    n = len(WPYR_VALUES)
    baseline  = np.full(n, np.nan)
    peak_cue  = np.full(n, np.nan)
    delay_ctr = np.full(n, np.nan)
    ctr_std   = np.full(n, np.nan)

    for i, wpyr in enumerate(WPYR_VALUES):
        m = results.get(wpyr)
        if m is None:
            continue
        ss = m.get("steady_state", {})
        bm = m.get("bump_metrics", {})
        baseline[i]  = ss.get("baseline_pyr_hz",     np.nan)
        peak_cue[i]  = ss.get("peak_pyr_cue_hz",     np.nan)
        delay_ctr[i] = ss.get("delay_pyr_center_hz", np.nan)
        ctr_std[i]   = bm.get("center_std_deg",      np.nan)

    return baseline, peak_cue, delay_ctr, ctr_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
SAT_CAP = 200.0


def plot_sweep(baseline, peak_cue, delay_ctr, ctr_std,
               wpv: float, amp: float,
               save_path: Path, no_show: bool):
    x = np.array(WPYR_VALUES)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"Ring sweep — w_pyr_pyr_inter\n"
        f"w_pv_global={wpv}  |  amplitude={amp}  |  sigma={_SIGMA_DEG}°  |  delay={DELAY_MS} ms  |  {CONDITION}",
        fontsize=12, fontweight="bold",
    )

    def panel(ax, y, title, ylabel, color, hline=None, hline_label=None,
              ylim=None, fill_bad=None):
        ax.semilogx(x, y, "o-", color=color, lw=2, ms=5)
        if hline is not None:
            ax.axhline(hline, color="red", lw=1.2, ls="--",
                       label=hline_label or f"{hline}")
            ax.legend(fontsize=8)
        if fill_bad is not None:
            # shade region above (or below) a threshold
            threshold, direction = fill_bad
            bad = y >= threshold if direction == "above" else y <= threshold
            for i, (xi, bi) in enumerate(zip(x, bad)):
                if bi:
                    ax.axvline(xi, color="salmon", alpha=0.3, lw=6)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, which="both", alpha=0.3)
        # annotate points
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                ax.text(xi, yi, f" {yi:.1f}", fontsize=7, va="bottom", ha="left")

    panel(axes[0, 0], baseline,
          "Pre-cue baseline PYR (Hz)\n[target: ≈ 0–3 Hz]",
          "Rate (Hz)", "steelblue",
          hline=3.0, hline_label="3 Hz target")

    panel(axes[0, 1], np.clip(peak_cue, 0, SAT_CAP),
          f"Peak PYR during cue (Hz)\n[{SAT_CAP:.0f} Hz = saturated]",
          "Rate (Hz)", "darkorange",
          hline=SAT_CAP, hline_label="200 Hz cap",
          ylim=(0, SAT_CAP * 1.05))

    panel(axes[1, 0], delay_ctr,
          "Post-cue centre PYR (Hz)\n[higher = bump sustained]",
          "Rate (Hz)", "seagreen",
          hline=5.0, hline_label="5 Hz floor")

    panel(axes[1, 1], ctr_std,
          "Bump centre std (deg)\n[< 30° = well-localised]",
          "Std (deg)", "mediumpurple",
          hline=30.0, hline_label="30° threshold",
          ylim=(0, 185))

    for ax in axes[1]:
        ax.set_xlabel("w_pyr_pyr_inter", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--w_pv_global",  type=float, default=DEFAULT_WPV,
                        help=f"Fixed w_pv_global (default: {DEFAULT_WPV})")
    parser.add_argument("--amplitude",    type=float, default=DEFAULT_AMP,
                        help=f"Fixed amplitude (default: {DEFAULT_AMP})")
    parser.add_argument("--sigma_pyr_deg", type=float, default=DEFAULT_SIGMA,
                        help=f"Gaussian excitation kernel width in degrees (default: {DEFAULT_SIGMA})")
    parser.add_argument("--n_workers",    type=int,   default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--no_run",  action="store_true",
                        help="Skip simulations; only plot cached data")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not display the figure interactively")
    args = parser.parse_args()

    global _SIGMA_DEG, _SWEEP_DIR
    wpv = args.w_pv_global
    amp = args.amplitude
    _SIGMA_DEG = args.sigma_pyr_deg
    _SWEEP_DIR = _sweep_dir_for(_SIGMA_DEG)
    _SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_run:
        print("--no_run: loading cached results only.")
        results = {}
        for wpyr in WPYR_VALUES:
            p = run_dir(wpyr, wpv, amp) / "run_metrics.json"
            if p.exists():
                with open(p) as f:
                    results[wpyr] = json.load(f)
        print(f"Found {len(results)} / {len(WPYR_VALUES)} cached results.")
    else:
        results = run_grid(wpv=wpv, amp=amp, n_workers=args.n_workers)

    if not results:
        print("No results to plot.")
        return

    baseline, peak_cue, delay_ctr, ctr_std = build_arrays(results)

    plot_sweep(
        baseline, peak_cue, delay_ctr, ctr_std,
        wpv=wpv, amp=amp,
        save_path=_SWEEP_DIR / "sweep_wpyr.png",
        no_show=args.no_show,
    )

    # Summary table
    print(f"\n{'wpyr':>10}  {'base':>6}  {'peak':>7}  {'delay_c':>9}  {'cstd':>7}")
    print("-" * 50)
    for wpyr in WPYR_VALUES:
        m = results.get(wpyr)
        if m is None:
            print(f"{wpyr:>10.5f}  {'N/A':>6}")
            continue
        ss = m.get("steady_state", {})
        bm = m.get("bump_metrics", {})
        b  = ss.get("baseline_pyr_hz",     None) or float("nan")
        pk = ss.get("peak_pyr_cue_hz",     None) or float("nan")
        dc = ss.get("delay_pyr_center_hz", None) or float("nan")
        cs = bm.get("center_std_deg",      None) or float("nan")
        sat = " SAT" if (not np.isnan(pk) and pk >= SAT_CAP - 0.5) else ""
        print(f"{wpyr:>10.5f}  {b:>6.2f}  {pk:>6.1f}{sat}  {dc:>9.3f}  {cs:>7.1f}")


if __name__ == "__main__":
    main()
