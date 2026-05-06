#!/usr/bin/env python3
"""
Plot state-classification heatmaps from ring_transient_sweep.py output.

For each sigma_som value in the sweep directory, produces a 2-D heatmap
(cue amplitude × noise sigma) where each cell is coloured by the network
state inferred from delay-period metrics in run_metrics.json.

States
------
PRE-SAT  : Network in high state *before* cue (noise-driven saturation).
           Detected via elevated baseline PYR rate before stimulus onset.
GLOBAL   : Cue pushed the whole network to the high state.
           Baseline was low, but both cue-side and opposite-side nodes are
           in the high state during the delay → full network recruitment.
LOCAL    : Localised bump sustained near the cue angle (ideal WM).
           High activity at cue side, low activity on opposite side,
           and decoded bump centre close to the cue location.
DRIFTED  : Localised bump but far from cue location.
           Similar to LOCAL but the bump has drifted / diffused away.
SILENT   : No sustained PYR activity after cue (network back at silent FP).
MISSING  : No run_metrics.json found for this (noise, amplitude) combination.

Classification timing
---------------------
All metrics come from the delay period (100 ms after cue offset → end of
simulation), so the classification reflects the settled state well after
the cue transient, not the immediate response.

Usage
-----
    python scripts/ring_sweep_heatmap.py --sweep_dir PATH [options]

Examples
--------
    python scripts/ring_sweep_heatmap.py \\
        --sweep_dir params_bistable/capped0.05nosomadapt/ring_transient_sweep

    python scripts/ring_sweep_heatmap.py \\
        --sweep_dir params_bistable/capped0.05nosomadapt/ring_transient_sweep \\
        --response_factor 0.3 \\
        --output figs/state_heatmap_rfac03.png \\
        --high_thresh_hz 15 \\
        --local_error_deg 45
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# ── State definitions ──────────────────────────────────────────────────────────

# Ordered for display: most "interesting" / distinctive first
STATES = ["PRE-SAT", "GLOBAL", "LOCAL", "DRIFTED", "SILENT", "MISSING"]

STATE_COLORS: dict[str, str] = {
    "PRE-SAT": "#d62728",   # red      – saturated before cue
    "GLOBAL":  "#9467bd",   # purple   – cue drove full-network high state
    "LOCAL":   "#2ca02c",   # green    – good localised bump near cue
    "DRIFTED": "#ff7f0e",   # orange   – bump far from cue
    "SILENT":  "#aec7e8",   # steel blue – no sustained activity
    "MISSING": "#f0f0f0",   # near-white – no data
}

STATE_LABELS: dict[str, str] = {
    "PRE-SAT": "Pre-saturated  (noise before cue)",
    "GLOBAL":  "Global high    (cue-driven full shift)",
    "LOCAL":   "Local bump     (near cue — good WM)",
    "DRIFTED": "Drifted bump   (far from cue)",
    "SILENT":  "Silent         (no sustained activity)",
    "MISSING": "No data",
}

STATE_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}


# ── Classification ─────────────────────────────────────────────────────────────

def classify_run(
    metrics: dict,
    high_thresh_hz: float,
    bump_thresh_hz: float,
    local_error_deg: float,
) -> str:
    """
    Classify one run into a state from its run_metrics.json content.

    Priority order (first match wins):
        1. PRE-SAT  – baseline PYR rate > high_thresh_hz before cue
        2. SILENT   – max delay-period node rate < bump_thresh_hz
        3. GLOBAL   – opposite-side rate > high_thresh_hz (full shift)
        4. LOCAL    – error from cue <= local_error_deg
        5. DRIFTED  – (all other cases with a bump)
    """
    ss = metrics.get("steady_state", {})
    bm = metrics.get("bump_metrics", {})

    baseline    = ss.get("baseline_pyr_hz", 0.0)
    delay_ctr   = ss.get("delay_pyr_center_hz", 0.0)
    delay_opp   = ss.get("delay_pyr_opposite_hz", 0.0)
    error_deg   = bm.get("error_from_cue_deg", 180.0)

    # 1. Pre-cue saturation: noise already drove network to high state
    if baseline > high_thresh_hz:
        return "PRE-SAT"

    # 2. Silent: cue produced no lasting activity
    if delay_ctr < bump_thresh_hz:
        return "SILENT"

    # 3. Global high state: both sides active (not just a localised bump)
    if delay_opp > high_thresh_hz:
        return "GLOBAL"

    # 4 / 5. Localised bump — check angular error from cue
    return "LOCAL" if error_deg <= local_error_deg else "DRIFTED"


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
    bump_thresh_hz: float,
    local_error_deg: float,
    noise_type: str | None = None,
) -> dict[float | None, dict[tuple[float, float], str]]:
    """
    Recursively scan sweep_dir for run_metrics.json files and return

        { sigma_som_or_None : { (noise_sigma, amplitude) : state_str } }

    sigma_som_or_None is None when no sigma_som subdirectory is present.
    If noise_type is specified (e.g., "white" or "ou"), filter to only that subfolder.
    """
    result: dict[float | None, dict[tuple[float, float], str]] = defaultdict(dict)

    for mf in sorted(sweep_dir.rglob("run_metrics.json")):
        parts = mf.relative_to(sweep_dir).parts[:-1]   # drop filename

        sigma_som: float | None = None
        noise: float | None     = None
        amp: float | None       = None
        found_noise_type: str | None = None

        for part in parts:
            v = _parse_float_suffix("sigma_som", part)
            if v is not None:
                sigma_som = v; continue
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

        state = classify_run(metrics, high_thresh_hz, bump_thresh_hz, local_error_deg)
        result[sigma_som][(noise, amp)] = state

    return result


# ── Heatmap array builder ──────────────────────────────────────────────────────

def _nearest_idx(sorted_vals: list[float], v: float) -> int | None:
    """Return index of the nearest value in a sorted list, or None if empty."""
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
    """(n_amp, n_noise) integer array mapping each cell to a STATE_IDX."""
    arr = np.full(
        (len(amp_vals), len(noise_vals)),
        STATE_IDX["MISSING"],
        dtype=int,
    )
    for (noise, amp), state in cell_states.items():
        ni = _nearest_idx(noise_vals, noise)
        ai = _nearest_idx(amp_vals, amp)
        if ni is not None and ai is not None:
            arr[ai, ni] = STATE_IDX[state]
    return arr


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_heatmaps(
    data: dict[float | None, dict[tuple[float, float], str]],
    output_path: Path,
    title_extra: str = "",
    noise_type: str | None = None,
) -> None:
    # Gather global noise / amplitude axes from all sigma_som groups
    all_noise: list[float] = sorted(
        {noise for cells in data.values() for (noise, _) in cells}
    )
    all_amp: list[float] = sorted(
        {amp for cells in data.values() for (_, amp) in cells}
    )

    sigma_som_vals: list[float | None] = sorted(
        data.keys(), key=lambda x: (x is None, x or 0.0)
    )
    n_cols = max(len(sigma_som_vals), 1)

    # Colourmap
    colors = [STATE_COLORS[s] for s in STATES]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(STATES) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(STATES))

    fig_w = max(4.0, 3.5 * n_cols)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(fig_w, 5.2),
        sharey=True,
        squeeze=False,
    )
    ax_row = axes[0]

    for ax, sigma_som in zip(ax_row, sigma_som_vals):
        arr = build_array(data[sigma_som], all_noise, all_amp)

        ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
        )

        ax.set_xticks(range(len(all_noise)))
        ax.set_xticklabels(
            [f"{v:.3g}" for v in all_noise],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(all_amp)))
        ax.set_yticklabels([f"{v:.3g}" for v in all_amp], fontsize=8)

        # Cell borders via minor ticks
        ax.set_xticks(np.arange(-0.5, len(all_noise)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(all_amp)), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)

        ax.set_xlabel("Noise σ", fontsize=9)

        if sigma_som is None:
            ax.set_title("(no σ_som sweep)", fontsize=9)
        else:
            ax.set_title(f"σ_som = {sigma_som:.4g}°", fontsize=9)

    ax_row[0].set_ylabel("Cue amplitude", fontsize=9)

    # Detect which states are actually present to decide legend content
    all_states_present = {
        state
        for cells in data.values()
        for state in cells.values()
    }
    # Always include MISSING if any cell is missing in any panel
    has_missing = any(
        STATE_IDX["MISSING"] in build_array(data[ss], all_noise, all_amp)
        for ss in sigma_som_vals
    )
    if has_missing:
        all_states_present.add("MISSING")

    legend_handles = [
        mpatches.Patch(color=STATE_COLORS[s], label=STATE_LABELS[s])
        for s in STATES
        if s in all_states_present
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
    p.add_argument(
        "--sweep_dir", required=True,
        help="Root output directory of ring_transient_sweep.py",
    )
    p.add_argument(
        "--output", default=None,
        help="Output figure path (default: <sweep_dir>/state_heatmap.png)",
    )
    p.add_argument(
        "--high_thresh_hz", type=float, default=20.0,
        help="PYR firing-rate threshold distinguishing high-state from low-state (Hz). "
             "Used to detect PRE-SAT (baseline) and GLOBAL (opposite-side rate). "
             "Default: 20 Hz.",
    )
    p.add_argument(
        "--bump_thresh_hz", type=float, default=5.0,
        help="Minimum delay_pyr_center_hz for a bump to be considered present. "
             "Runs below this threshold are classified SILENT. Default: 5 Hz.",
    )
    p.add_argument(
        "--local_error_deg", type=float, default=60.0,
        help="Maximum angular error from cue to classify bump as LOCAL "
             "(rather than DRIFTED). Default: 60 degrees.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()

    if not sweep_dir.is_dir():
        print(f"Error: sweep_dir not found: {sweep_dir}")
        return

    print(f"Sweep dir       : {sweep_dir}")
    print(f"Thresholds      : high={args.high_thresh_hz} Hz, "
          f"bump={args.bump_thresh_hz} Hz, local_err={args.local_error_deg}°")

    # Detect which noise_type folders exist (at top level or within sigma_som dirs)
    noise_types = []
    for nt in ["white", "ou"]:
        if (sweep_dir / nt).exists():
            noise_types.append(nt)
        else:
            # Also check if noise_type exists within any sigma_som subdirectory
            for sigma_som_dir in sweep_dir.glob("sigma_som*"):
                if (sigma_som_dir / nt).exists():
                    noise_types.append(nt)
                    break

    noise_types = list(set(noise_types))  # Remove duplicates

    if not noise_types:
        print("No noise type folders (white/ or ou/) found — nothing to plot.")
        return

    total_runs = 0

    for noise_type in noise_types:
        data = load_sweep(
            sweep_dir,
            high_thresh_hz=args.high_thresh_hz,
            bump_thresh_hz=args.bump_thresh_hz,
            local_error_deg=args.local_error_deg,
            noise_type=noise_type,
        )

        if not data:
            print(f"{noise_type}: No run_metrics.json found.")
            continue

        total_runs += sum(len(v) for v in data.values())
        n_groups = len(data)
        print(f"{noise_type}: Loaded {sum(len(v) for v in data.values())} runs across {n_groups} sigma_som group(s)")

        for sigma_som in sorted(data.keys(), key=lambda x: (x is None, x or 0.0)):
            cells = data[sigma_som]
            counts = Counter(cells.values())
            label = f"σ_som={sigma_som}°" if sigma_som is not None else "(no sweep)"
            parts = ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
            print(f"    {label:18s}: {parts}")

        # Generate output path for this noise type
        if args.output:
            output_path = Path(args.output).resolve()
            # Insert noise_type before the extension
            output = output_path.parent / f"{output_path.stem}_{noise_type}{output_path.suffix}"
        else:
            output = sweep_dir / f"state_heatmap_{noise_type}.png"

        plot_heatmaps(data, output, "", noise_type=noise_type)

    if total_runs == 0:
        print("No run_metrics.json found across any noise type — nothing to plot.")


if __name__ == "__main__":
    main()
