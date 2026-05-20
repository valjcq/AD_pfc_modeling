#!/usr/bin/env python3
"""
Run w_pyr_pyr_inter × amplitude sweep for low_fr parameter set.

Goal
----
Previous finding: with w_pyr_pyr_inter = 0.002 (default), the low_fr network
only transitions from silent to bump at amplitude ≥ 7.0 × I_ext_pyr — an
unphysically strong stimulus.

Hypothesis: stronger recurrent PYR→PYR coupling (w_pyr_pyr_inter) shifts the
bifurcation threshold downward toward biologically plausible amplitudes (1–3×).

Sweep design
------------
  w_pyr_pyr_inter : 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.025, 0.040
                    (log-spaced, 9 values; covers sub- and supra-Turing regimes)
  amplitudes      : 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
                    (from well below old threshold to old threshold)
  w_pv_global     : 0.0001, 0.001  (primary + secondary; creates w_pv_X/ sub-dirs)

  Total: 2 × 9 × 8 × 10 = 1440 trials (≈ 15–25 min on 10 workers).

Outputs
-------
  figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep/
    w_pv_0.0001/
      bump_decay_trials.csv            ← per-trial raw data
      bump_decay_amp_sweep.png         ← line plot: amplitude vs mean end_val
      WT/bump_decay_heatmap.png        ← 2D heatmap: w_inter × amplitude
      experiment_config.txt
      amp{X}/w{Y}/
        bump_decay_timecourse.png
        bump_decay_boxplot.png
    w_pv_0.001/
      (same structure)

Post-analysis
-------------
  python3 scripts/analyze_low_fr_wInter_sweep.py \\
      --sweep_dir figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep \\
      --no_show

  Produces:
    summary.json       — aggregated metrics per (w_pv, w_inter, amplitude)
    threshold_curve.png — min amplitude vs w_inter per w_pv value
    heatmap_wpv_*.png  — 2D heatmap per w_pv value
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMS_JSON  = "figs/optim/bistable_low_fr/best_params.json"
OUTPUT_DIR   = "figs/ring/calibration/128_sigma_15_low_fr_wInter_amp_sweep"
SIGMA_DEG    = "15"
N_TRIALS     = "10"
N_WORKERS    = "10"

# w_pyr_pyr_inter: sub-Turing → near-Turing → supra-Turing (log-spaced)
W_INTER_VALUES = ["0.001", "0.002", "0.004", "0.006", "0.008",
                  "0.012", "0.016", "0.025", "0.040"]

# Amplitudes: from below known threshold down to physiological
AMPLITUDES = ["1.0", "1.5", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]

# w_pv sweep (secondary dimension)
W_PV_VALUES = ["0.0001", "0.001"]

# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "circuit_model", "ring-calibrate",
        "--params_json",    PARAMS_JSON,
        "--sigma_pyr_deg",  SIGMA_DEG,
        "--w_pv_values",    *W_PV_VALUES,
        "--w_inter_values", *W_INTER_VALUES,
        "--amplitudes",     *AMPLITUDES,
        "--n_trials",       N_TRIALS,
        "--n_workers",      N_WORKERS,
        "--output_dir",     OUTPUT_DIR,
        "--no_show",
    ]

    print("=" * 70)
    print("  Low-fr  w_pyr_pyr_inter × amplitude sweep")
    print("=" * 70)
    print(f"  w_inter  : {' '.join(W_INTER_VALUES)}")
    print(f"  amplitude: {' '.join(AMPLITUDES)}")
    print(f"  w_pv     : {' '.join(W_PV_VALUES)}")
    total = len(W_PV_VALUES) * len(W_INTER_VALUES) * len(AMPLITUDES) * int(N_TRIALS)
    print(f"  Total trials: {total}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)
    print()
    print("Command:")
    print("  " + " ".join(cmd))
    print()

    subprocess.run(cmd, check=True, cwd=str(root))

    print()
    print("=" * 70)
    print("  Sweep complete.  Now run the analysis:")
    print(f"    python3 scripts/analyze_low_fr_wInter_sweep.py \\")
    print(f"        --sweep_dir {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
