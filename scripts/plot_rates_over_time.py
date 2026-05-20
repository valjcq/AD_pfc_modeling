"""
Plot mean activity rates per genotype over timepoints (1mo, 3mo, 6mo).
One subplot per genotype; each shows control (solid) vs APP (dashed).
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
T_DEFAULT = 164.897

# ── rate computation (fallback for genotypes missing from summary JSON) ────────

def read_t(mouse_dir: Path) -> float:
    for name in ("parameters.rtf", "parameters.txt"):
        p = mouse_dir / name
        if p.exists():
            import re
            m = re.search(r"t\s*=\s*([\d.]+)", p.read_text(errors="replace"))
            if m:
                return float(m.group(1))
    return T_DEFAULT

def folder_rate(folder: Path):
    """Return (mean, std, n_files) over all Results CSVs in a genotype folder."""
    rates = []
    for mouse_dir in sorted(folder.iterdir()):
        if not mouse_dir.is_dir():
            continue
        t = read_t(mouse_dir)
        for csv in sorted(mouse_dir.glob("Results*.csv")):
            try:
                df = pd.read_csv(csv, index_col=0)
                rates.append(float(df.values.mean()) / t)
            except Exception as e:
                warnings.warn(f"Skipping {csv}: {e}")
    if not rates:
        return None, None, 0
    return float(np.mean(rates)), float(np.std(rates)), len(rates)

# ── load all rates (summary JSON + on-the-fly for missing entries) ─────────────

summary = json.loads((ROOT / "AD_data/summary/targets_all.json").read_text())

_cache: dict = {}

def rate(tp: str, genotype: str):
    """Return (mean, std) or (None, None) if no data."""
    key = (tp, genotype)
    if key in _cache:
        return _cache[key]
    g = summary.get(tp, {}).get(genotype)
    if g is not None:
        result = (g["mean"], g["std"])
    else:
        folder = ROOT / "AD_data" / f"{tp}_post_injection" / genotype
        if folder.exists():
            m, s, n = folder_rate(folder)
            result = (m, s) if m is not None else (None, None)
        else:
            result = (None, None)
    _cache[key] = result
    return result

# ── subplot configuration ──────────────────────────────────────────────────────

# (subplot_title, ctrl_key, app_key)
PANELS = [
    ("WT (PYR)",  "WT",           "WT_APP"),
    ("a7KO (PYR)","a7KO_control", "a7KO_APP"),
    ("b2KO (PYR)","b2KO_control", "b2KO_APP"),
    ("a5KO (PYR)","a5KO_control", "a5KO_APP"),
    ("PV",        "PV_control",   "PV_APP"),
    ("SST",       "SST_control",  "SST_APP"),
    ("VIP",       "VIP_control",  "VIP_APP"),
]

TIMEPOINTS = ["1mo", "3mo", "6mo"]
TP_X       = [1, 3, 6]
TP_LABELS  = ["1 mo", "3 mo", "6 mo"]

# Within each subplot: one fixed color per condition (not per genotype)
COLOR_CTRL = "#2166ac"   # blue  — control
COLOR_APP  = "#d6604d"   # red   — APP

# ── figure ────────────────────────────────────────────────────────────────────

ncols = 4
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7), constrained_layout=True,
                         sharey=False)
fig.suptitle("Activity rate per genotype — control vs APP", fontsize=13)

axes_flat = axes.flatten()

for idx, (title, ctrl_key, app_key) in enumerate(PANELS):
    ax = axes_flat[idx]
    # --- control ---
    xs_c, ys_c, es_c = [], [], []
    for tp, x in zip(TIMEPOINTS, TP_X):
        m, s = rate(tp, ctrl_key)
        if m is not None:
            xs_c.append(x); ys_c.append(m); es_c.append(s)

    if xs_c:
        ax.errorbar(xs_c, ys_c, yerr=es_c, color=COLOR_CTRL,
                    marker="o", linewidth=2, markersize=7, capsize=4,
                    label="control", zorder=3)

    # --- APP ---
    xs_a, ys_a, es_a = [], [], []
    for tp, x in zip(TIMEPOINTS, TP_X):
        m, s = rate(tp, app_key)
        if m is not None:
            xs_a.append(x); ys_a.append(m); es_a.append(s)

    if xs_a:
        ax.errorbar(xs_a, ys_a, yerr=es_a, color=COLOR_APP,
                    marker="s", linewidth=2, markersize=7, capsize=4,
                    label="APP", zorder=3)

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(TP_X)
    ax.set_xticklabels(TP_LABELS, fontsize=8)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylabel("rate (a.u. / s)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

# hide the last (empty) subplot
axes_flat[-1].set_visible(False)

# ── save ──────────────────────────────────────────────────────────────────────
out = ROOT / "figs/data"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "rates_over_time.png", dpi=150, bbox_inches="tight")
print(f"Saved → {out / 'rates_over_time.png'}")
plt.show()
