#!/usr/bin/env python3
"""
Plot state-classification heatmap from bistable_transient_sweep.py output.

Reads run_metrics.json files from the directory structure
    <sweep_dir>/noise{sigma}/cue{amp}/run_metrics.json

and produces a 2-D heatmap (cue amplitude × noise sigma) where each cell
is coloured by the single-node bistable network state.

States
------
PRE-SAT   : Network already in high state *before* the transient (noise-driven).
            Detected via elevated pre_trans_pyr_hz.
SHIFTED   : Transient caused a permanent flip from low to high state.
            pre-transient rate low, post-transient rate high.
TRANSIENT : Network visited the high state during cue but returned to low.
            pre and post rates both low, but peak during transient was high.
SILENT    : Network barely responded — all rates stayed low throughout.

Usage
-----
    python scripts/bistable_sweep_heatmap.py --sweep_dir PATH [options]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# ── State definitions ──────────────────────────────────────────────────────────

STATES = ["PRE-SAT", "SHIFTED", "TRANSIENT", "SILENT", "MISSING"]

STATE_COLORS: dict[str, str] = {
    "PRE-SAT":   "#d62728",   # red     – saturated before cue
    "SHIFTED":   "#2ca02c",   # green   – successful flip to high state
    "TRANSIENT": "#ff7f0e",   # orange  – visited high state but returned
    "SILENT":    "#aec7e8",   # steel blue – no response
    "MISSING":   "#f0f0f0",   # near-white – no data
}

STATE_LABELS: dict[str, str] = {
    "PRE-SAT":   "Pre-saturated  (noise before cue)",
    "SHIFTED":   "Shifted        (cue triggered high state)",
    "TRANSIENT": "Transient only (returned to low state)",
    "SILENT":    "Silent         (no response)",
    "MISSING":   "No data",
}

STATE_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}


# ── Classification ─────────────────────────────────────────────────────────────

def classify_run(
    metrics: dict,
    high_thresh_hz: float,
    peak_thresh_hz: float,
) -> str:
    """
    Classify one run from its run_metrics.json content.

    Priority order
    --------------
    1. PRE-SAT   – pre_trans_pyr_hz > high_thresh_hz
    2. SHIFTED   – post_trans_pyr_hz > high_thresh_hz  (pre was low)
    3. TRANSIENT – trans_peak_pyr_hz > peak_thresh_hz  (post still low)
    4. SILENT    – all rates below thresholds
    """
    ss = metrics.get("steady_state", {})
    pre  = ss.get("pre_trans_pyr_hz",  0.0)
    peak = ss.get("trans_peak_pyr_hz", 0.0)
    post = ss.get("post_trans_pyr_hz", 0.0)

    if pre > high_thresh_hz:
        return "PRE-SAT"
    if post > high_thresh_hz:
        return "SHIFTED"
    if peak > peak_thresh_hz:
        return "TRANSIENT"
    return "SILENT"


# ── Directory parsing ──────────────────────────────────────────────────────────

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_float_suffix(prefix: str, dirname: str) -> float | None:
    if not dirname.startswith(prefix):
        return None
    m = _FLOAT_RE.search(dirname[len(prefix):])
    return float(m.group()) if m else None


def load_sweep(
    sweep_dir: Path,
    high_thresh_hz: float,
    peak_thresh_hz: float,
    noise_type: str | None = None,
) -> dict[tuple[float, float], str]:
    """
    Scan sweep_dir for run_metrics.json files and return
        { (noise_sigma, amplitude) : state_str }

    If noise_type is specified (e.g., "white" or "ou"), filter to only that subfolder.
    """
    result: dict[tuple[float, float], str] = {}

    for mf in sorted(sweep_dir.rglob("run_metrics.json")):
        parts = mf.relative_to(sweep_dir).parts[:-1]

        noise: float | None = None
        amp:   float | None = None
        found_noise_type: str | None = None

        for part in parts:
            v = _parse_float_suffix("noise", part)
            if v is not None:
                noise = v; continue
            v = _parse_float_suffix("cue", part)
            if v is not None:
                amp = v; continue
            # Detect noise type from path (white or ou folder)
            if part in ["white", "ou"]:
                found_noise_type = part

        if noise is None or amp is None:
            continue

        # Filter by requested noise_type if specified
        if noise_type is not None and found_noise_type != noise_type:
            continue

        try:
            metrics = json.loads(mf.read_text())
        except Exception:
            continue

        result[(noise, amp)] = classify_run(metrics, high_thresh_hz, peak_thresh_hz)

    return result


# ── Heatmap ────────────────────────────────────────────────────────────────────

def _nearest_idx(sorted_vals: list[float], v: float) -> int | None:
    if not sorted_vals:
        return None
    diffs = [abs(x - v) for x in sorted_vals]
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < 1e-9 else None


def build_array(
    cell_states: dict[tuple[float, float], str],
    noise_vals: list[float],
    amp_vals: list[float],
) -> np.ndarray:
    arr = np.full((len(amp_vals), len(noise_vals)), STATE_IDX["MISSING"], dtype=int)
    for (noise, amp), state in cell_states.items():
        ni = _nearest_idx(noise_vals, noise)
        ai = _nearest_idx(amp_vals, amp)
        if ni is not None and ai is not None:
            arr[ai, ni] = STATE_IDX[state]
    return arr


def plot_heatmap(
    data: dict[tuple[float, float], str],
    output_path: Path,
    title_extra: str = "",
    noise_type: str | None = None,
) -> None:
    noise_vals = sorted({n for (n, _) in data})
    amp_vals   = sorted({a for (_, a) in data})

    colors = [STATE_COLORS[s] for s in STATES]
    cmap   = ListedColormap(colors)
    bounds = np.arange(len(STATES) + 1) - 0.5
    norm   = BoundaryNorm(bounds, len(STATES))

    arr = build_array(data, noise_vals, amp_vals)

    fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * len(noise_vals) + 1.5), 5.0))

    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    ax.set_xticks(range(len(noise_vals)))
    ax.set_xticklabels([f"{v:.3g}" for v in noise_vals], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(amp_vals)))
    ax.set_yticklabels([f"{v:.3g}" for v in amp_vals], fontsize=8)

    # Cell borders via minor ticks
    ax.set_xticks(np.arange(-0.5, len(noise_vals)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(amp_vals)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    ax.set_xlabel("Noise σ", fontsize=9)
    ax.set_ylabel("Cue amplitude", fontsize=9)

    # Legend — only show states actually present (plus MISSING if any gap)
    present = set(data.values())
    has_missing = STATE_IDX["MISSING"] in arr
    if has_missing:
        present.add("MISSING")

    legend_handles = [
        mpatches.Patch(color=STATE_COLORS[s], label=STATE_LABELS[s])
        for s in STATES if s in present
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9,
    )

    title = "Heatmap of network state for varying noise and amplitude"
    if noise_type:
        title += f"  ({noise_type} noise)"
    fig.suptitle(title, fontsize=10, y=0.98)

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sweep_dir", required=True,
                   help="Root output directory of bistable_transient_sweep.py")
    p.add_argument("--output", default=None,
                   help="Output figure path (default: <sweep_dir>/state_heatmap.png)")
    p.add_argument("--high_thresh_hz", type=float, default=20.0,
                   help="PYR rate threshold for 'high state' (Hz). Used to detect "
                        "PRE-SAT (pre-transient) and SHIFTED (post-transient). Default: 20.")
    p.add_argument("--peak_thresh_hz", type=float, default=10.0,
                   help="Minimum peak PYR rate during transient to count as TRANSIENT "
                        "(vs SILENT). Default: 10 Hz.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()

    if not sweep_dir.is_dir():
        print(f"Error: sweep_dir not found: {sweep_dir}")
        return

    print(f"Sweep dir    : {sweep_dir}")
    print(f"Thresholds   : high={args.high_thresh_hz} Hz, peak={args.peak_thresh_hz} Hz")

    # Detect which noise_type folders exist
    noise_types = []
    for nt in ["white", "ou"]:
        if (sweep_dir / nt).exists():
            noise_types.append(nt)

    if not noise_types:
        print("No noise type folders (white/ or ou/) found — nothing to plot.")
        return

    title_extra = sweep_dir.name
    total_runs = 0

    for noise_type in noise_types:
        data = load_sweep(sweep_dir, args.high_thresh_hz, args.peak_thresh_hz, noise_type=noise_type)

        if not data:
            print(f"{noise_type}: No run_metrics.json found.")
            continue

        total_runs += len(data)
        counts = Counter(data.values())
        print(f"{noise_type}: {len(data)} runs — " + ", ".join(f"{s}={n}" for s, n in sorted(counts.items())))

        # Generate output path for this noise type
        if args.output:
            output_path = Path(args.output).resolve()
            # Insert noise_type before the extension
            output = output_path.parent / f"{output_path.stem}_{noise_type}{output_path.suffix}"
        else:
            output = sweep_dir / f"state_heatmap_{noise_type}.png"

        plot_heatmap(data, output, title_extra, noise_type=noise_type)

    if total_runs == 0:
        print("No run_metrics.json found across any noise type — nothing to plot.")


if __name__ == "__main__":
    main()
