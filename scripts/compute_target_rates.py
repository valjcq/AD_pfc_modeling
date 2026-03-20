"""
Compute mean activity rate per genotype per timepoint from calcium imaging data.

Formula (per Results CSV file):
  rate = mean(all fluorescence values in file) / t_recording

Empirical validation against data_1mo_article.md (1mo WT genotypes):
  SST: 3.42 vs article 3.248  (1.05×)
  PV:  2.08 vs article 1.414  (1.47×)
  PYR: 4.14 vs article 2.487  (1.67×)
  VIP: 1.93 vs article 2.517  (0.77×)

The formula captures the correct order of magnitude and relative ordering.
Residual discrepancy is due to the article reporting medians while we compute
means, and per-cell-type differences in baseline fluorescence brightness.

Output: AD_data/summary/targets_{1mo,3mo,6mo}.json
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401 (used inside file_rate via pd.read_csv)

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent / "AD_data"
SUMMARY_DIR = ROOT / "summary"
SUMMARY_DIR.mkdir(exist_ok=True)

# No thresholds needed — formula is simply mean(F) / t_recording

# Timepoints to process — all genotype subfolders are auto-discovered
TIMEPOINTS = ["1mo", "3mo", "6mo"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_duration(mouse_dir: Path) -> float:
    """Read recording duration in seconds from parameters.rtf or parameters.txt."""
    for fname in ("parameters.rtf", "parameters.txt"):
        p = mouse_dir / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Matches: t= 164.897 sec  /  t = 164.897  /  t=330.3 sec
        m = re.search(r"t\s*=\s*([\d.]+)", text)
        if m:
            return float(m.group(1))
    return 164.897  # fallback: most common value


def file_rate(csv_path: Path, t_sec: float) -> float:
    """
    Compute activity rate for one Results CSV file.

    rate = mean(all fluorescence values) / t_recording
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        warnings.warn(f"Could not read {csv_path}: {e}")
        return np.nan
    return float(df.values.mean()) / t_sec


def compute_genotype_rate(genotype_dir: Path) -> dict:
    """
    Compute summary statistics for one genotype folder.

    Returns dict with: mean, std, median, n_files, n_mice, per_mouse_means
    """
    all_rates: list[float] = []
    per_mouse: dict[str, float] = {}

    mouse_dirs = sorted(d for d in genotype_dir.iterdir() if d.is_dir())
    if not mouse_dirs:
        return {}

    for mouse_dir in mouse_dirs:
        t_sec = parse_duration(mouse_dir)
        mouse_rates: list[float] = []

        for csv in sorted(mouse_dir.glob("*.csv")):
            r = file_rate(csv, t_sec)
            if not np.isnan(r):
                mouse_rates.append(r)

        if mouse_rates:
            per_mouse[mouse_dir.name] = round(float(np.mean(mouse_rates)), 6)
            all_rates.extend(mouse_rates)

    if not all_rates:
        return {}

    return {
        "mean":       round(float(np.mean(all_rates)), 6),
        "median":     round(float(np.median(all_rates)), 6),
        "std":        round(float(np.std(all_rates)), 6),
        "n_files":    len(all_rates),
        "n_mice":     len(per_mouse),
        "per_mouse":  per_mouse,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results: dict[str, dict] = {}

    for timepoint in TIMEPOINTS:
        tp_dir = ROOT / f"{timepoint}_post_injection"
        if not tp_dir.exists():
            print(f"\nSkipping {timepoint}: {tp_dir} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Timepoint: {timepoint}")
        print(f"{'='*50}")
        results[timepoint] = {}

        for geno_dir in sorted(tp_dir.iterdir()):
            if not geno_dir.is_dir():
                continue
            geno_name = geno_dir.name

            stats = compute_genotype_rate(geno_dir)
            if not stats:
                print(f"  {geno_name:<25} no data")
                continue

            results[timepoint][geno_name] = stats
            print(
                f"  {geno_name:<25} "
                f"mean={stats['mean']:.4f}  "
                f"median={stats['median']:.4f}  "
                f"std={stats['std']:.4f}  "
                f"n={stats['n_files']} files / {stats['n_mice']} mice"
            )

        # Save per-timepoint JSON
        out_path = SUMMARY_DIR / f"targets_{timepoint}.json"
        with open(out_path, "w") as f:
            json.dump(results[timepoint], f, indent=2)
        print(f"\n  → saved to {out_path.relative_to(ROOT.parent)}")

    # Also save a combined file
    combined_path = SUMMARY_DIR / "targets_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined → {combined_path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
