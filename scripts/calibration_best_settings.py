"""
calibration_best_settings.py
-----------------------------
For every network-parameter folder under figs/ring/calibration/, find the
(amplitude, w_inter) setting that satisfies:

    success_rate == 1.00
    17.5 Hz <= peak_pyr_rate <= 18.0 Hz

and among those, keep the one with the highest mean_A_hat (one per condition).
Write the result to  <network_label_folder>/best_settings.txt.

Usage:
    python scripts/calibration_best_settings.py
    python scripts/calibration_best_settings.py --root figs/ring/calibration
    python scripts/calibration_best_settings.py --min_rate 17.5 --max_rate 18.0
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

PEAK_MIN_DEFAULT = 17.5   # Hz
PEAK_MAX_DEFAULT = 18.0   # Hz
SUCCESS_RATE_TARGET = 1.0


def find_network_folders(root: str) -> list[str]:
    """
    Walk the root tree and return every folder that directly contains
    condition sub-folders with calibration_summary.csv files.

    Expected layout:
        root/
          <params_label>/
            <network_label>/         ← returned
              <condition>/
                calibration_summary.csv
    """
    folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        # A network folder is one whose direct children contain
        # calibration_summary.csv
        has_summary = any(
            os.path.isfile(os.path.join(dirpath, d, "calibration_summary.csv"))
            for d in dirnames
        )
        if has_summary:
            folders.append(dirpath)
    return sorted(folders)


def load_summary(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def best_row(rows: list[dict], peak_min: float, peak_max: float) -> dict | None:
    candidates = []
    for r in rows:
        try:
            sr = float(r["success_rate"])
            pr = float(r["peak_pyr_rate"])
            ah = float(r["mean_A_hat"])
        except (KeyError, ValueError):
            continue
        if abs(sr - SUCCESS_RATE_TARGET) < 1e-9 and peak_min <= pr <= peak_max:
            candidates.append((ah, r))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def write_txt(network_folder: str, results: dict, peak_min: float, peak_max: float) -> str:
    """
    results: {condition_key: row_dict | None}
    """
    out_path = os.path.join(network_folder, "best_settings.txt")
    lines = [
        f"Best calibration settings",
        f"Network: {os.path.basename(network_folder)}",
        f"Criteria: success_rate = 1.00, "
        f"peak_pyr_rate ∈ [{peak_min:.1f}, {peak_max:.1f}] Hz",
        f"Selection: highest mean_A_hat among candidates",
        "",
        f"{'Condition':<15} {'Amplitude':>10} {'w_inter':>9} "
        f"{'peak_pyr_rate':>14} {'mean_A_hat':>11} {'success_rate':>13}",
        "-" * 78,
    ]
    for cond in sorted(results):
        r = results[cond]
        if r is None:
            lines.append(f"{cond:<15}  {'— no match found':}")
        else:
            lines.append(
                f"{cond:<15} {float(r['amplitude']):>10.1f} "
                f"{float(r['w_inter']):>9.2f} "
                f"{float(r['peak_pyr_rate']):>14.3f} "
                f"{float(r['mean_A_hat']):>11.4f} "
                f"{float(r['success_rate']):>13.2f}"
            )
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root", default="figs/ring/calibration",
        help="Root calibration directory (default: figs/ring/calibration)",
    )
    parser.add_argument(
        "--min_rate", type=float, default=PEAK_MIN_DEFAULT,
        help=f"Minimum peak PYR firing rate in Hz (default: {PEAK_MIN_DEFAULT})",
    )
    parser.add_argument(
        "--max_rate", type=float, default=PEAK_MAX_DEFAULT,
        help=f"Maximum peak PYR firing rate in Hz (default: {PEAK_MAX_DEFAULT})",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    network_folders = find_network_folders(root)
    if not network_folders:
        print(f"No calibration folders found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(network_folders)} network folder(s).")

    for folder in network_folders:
        results = {}
        for entry in sorted(os.listdir(folder)):
            csv_path = os.path.join(folder, entry, "calibration_summary.csv")
            if not os.path.isfile(csv_path):
                continue
            rows = load_summary(csv_path)
            if not rows:
                continue
            # condition_key may differ from the folder name; read from CSV
            cond = rows[0].get("condition_key", entry)
            results[cond] = best_row(rows, args.min_rate, args.max_rate)

        if not results:
            print(f"  {folder}: no condition CSVs found, skipping.")
            continue

        out = write_txt(folder, results, args.min_rate, args.max_rate)
        print(f"  Written: {out}")
        for cond, r in sorted(results.items()):
            if r is None:
                print(f"    {cond}: no match")
            else:
                print(f"    {cond}: amp={r['amplitude']}, w_inter={r['w_inter']}, "
                      f"peak={float(r['peak_pyr_rate']):.3f} Hz, "
                      f"A_hat={float(r['mean_A_hat']):.4f}")


if __name__ == "__main__":
    main()
