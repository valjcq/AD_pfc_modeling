"""
Standalone script: plot asymmetry over time for the worst-case trial of each condition.

Reads seeds from the cached CSV, re-runs each worst trial at full temporal resolution,
and plots the L/R asymmetry index A(t) from pre-cue through the entire delay.

Usage:
    python scripts/asymmetry_timecourse.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CSV_PATH = os.path.join(
    ROOT,
    "figs/asymmetry/128/default/gauss_w7_s30-pv_unif_10/amp45_uncorrected",
    "asymmetry_trials.csv",
)
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "worst_case_timecourses.png")

# ---------------------------------------------------------------------------
# Simulation parameters  (must match the experiment that produced the CSV)
# ---------------------------------------------------------------------------
AMP_FACTOR = 45.0          # stimulus amplitude multiplier
DELAY_MS   = 5000.0        # delay period duration
SETTLING_MS = 6000.0       # noisy burn-in before cue onset  (ASYM_SETTLING_MS)
STIM_DURATION_MS = 250.0   # cue duration
STIM_SIGMA_DEG   = 18.0    # cue Gaussian width
RECORD_DT_MS = 1.0         # higher resolution than the cached 5 ms

# Ring network
N_NODES       = 128
W_PYR_INTER   = 7.0
SIGMA_PYR_DEG = 30.0
W_PV_GLOBAL   = 10.0

CONDITIONS = ["WT", "WT_APP", "a7_KO_APP"]

# ---------------------------------------------------------------------------
# 1. Load CSV — find worst-case (max |delay_asym|) trial per condition
# ---------------------------------------------------------------------------
worst: dict[str, dict] = {}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        cond = row["condition"]
        if cond not in CONDITIONS:
            continue
        val = abs(float(row["delay_asym"]))
        if cond not in worst or val > abs(float(worst[cond]["delay_asym"])):
            worst[cond] = row

for cond, row in worst.items():
    print(
        f"Worst {cond}: trial={row['trial_idx']}, seed={row['seed']}, "
        f"cue={float(row['cue_deg']):.2f}°, delay_asym={float(row['delay_asym']):+.4f}"
    )

# ---------------------------------------------------------------------------
# 2. Set up the circuit
# ---------------------------------------------------------------------------
from circuit_model.params import CircuitParams
from circuit_model.study import STUDY_CONDITIONS, apply_condition
from circuit_model.ring.params import RingParams
from circuit_model.ring.connectivity import RingConnectivity
from circuit_model.ring.stimulus import RingStimulus
from circuit_model.ring.simulation import simulate_ring
from circuit_model.ring.analysis import compute_bump_asymmetry, compute_asymmetry_temporal_metrics

base_params = CircuitParams()
ring_params = RingParams(
    n_nodes=N_NODES,
    w_pyr_pyr_inter=W_PYR_INTER,
    sigma_pyr_deg=SIGMA_PYR_DEG,
    w_pv_global=W_PV_GLOBAL,
)
connectivity = RingConnectivity.from_params(ring_params)

stim_onset_ms  = SETTLING_MS
stim_offset_ms = stim_onset_ms + STIM_DURATION_MS
T_ms           = stim_offset_ms + DELAY_MS

# ---------------------------------------------------------------------------
# 3. Re-run worst trial for each condition and collect A(t)
# ---------------------------------------------------------------------------
results_by_cond: dict[str, dict] = {}

for cond_key in CONDITIONS:
    row   = worst[cond_key]
    seed  = int(row["seed"])
    cue_deg = float(row["cue_deg"])

    print(f"\nRunning {cond_key} (seed={seed}) …", flush=True)

    local_params = apply_condition(base_params, STUDY_CONDITIONS[cond_key])
    actual_current = AMP_FACTOR * base_params.I_ext_pyr()

    stimuli = [RingStimulus(
        center_deg=cue_deg,
        amplitude=actual_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=stim_onset_ms,
        duration_ms=STIM_DURATION_MS,
    )]

    result = simulate_ring(
        local_params, ring_params,
        T_ms=T_ms,
        stimuli=stimuli,
        seed=seed,
        connectivity=connectivity,
        record_dt_ms=RECORD_DT_MS,
    )

    asym = compute_bump_asymmetry(result)   # shape (n_steps,)
    t_ms = result.t_ms                      # shape (n_steps,)

    # Time relative to cue onset (ms), so t=0 = cue on, t=250 = cue off
    t_rel = t_ms - stim_onset_ms

    # New temporal metrics over the delay window (after transient skip)
    TRANSIENT_SKIP_MS = 400.0
    delay_mask = (t_ms >= stim_onset_ms + STIM_DURATION_MS + TRANSIENT_SKIP_MS)
    temporal = compute_asymmetry_temporal_metrics(asym[delay_mask], t_ms[delay_mask])

    results_by_cond[cond_key] = {
        "t_rel":    t_rel,
        "asym":     asym,
        "label":    STUDY_CONDITIONS[cond_key].name,
        "delay_asym": float(row["delay_asym"]),
        **temporal,
    }

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
COLORS = {"WT": "#2196F3", "WT_APP": "#FF9800", "a7_KO_APP": "#F44336"}

fig, axes = plt.subplots(
    len(CONDITIONS), 1,
    figsize=(12, 3.5 * len(CONDITIONS)),
    sharex=True,
    constrained_layout=True,
)

for ax, cond_key in zip(axes, CONDITIONS):
    d = results_by_cond[cond_key]
    t = d["t_rel"]
    a = d["asym"]
    color = COLORS[cond_key]

    ax.plot(t, a, lw=1.2, color=color, label=d["label"])
    ax.axhline(0, color="k", lw=0.7, ls="--")

    # Shade cue window
    ax.axvspan(0, STIM_DURATION_MS, alpha=0.15, color="gold", label="Cue on")

    # Shade pre-cue window used for the metric
    ax.axvspan(-500, 0, alpha=0.10, color="gray", label="Pre-cue window")

    # Mark mean scalar metrics
    pre_mask  = (t >= -500) & (t < 0)
    delay_mask = (t >= STIM_DURATION_MS) & (t <= STIM_DURATION_MS + DELAY_MS)
    if pre_mask.any():
        ax.axhline(a[pre_mask].mean(), color="gray", lw=1.0, ls=":", alpha=0.8)
    if delay_mask.any():
        ax.axhline(a[delay_mask].mean(), color=color, lw=1.2, ls=":", alpha=0.8,
                   label=f"Mean delay asym = {d['delay_asym']:+.4f}")

    ax.set_ylabel("Asymmetry A(t)")
    metrics_str = (
        f"mean(A) = {d['delay_asym']:+.4f}  |  "
        f"mean|A| = {d['mean_abs_asym']:.4f}  |  "
        f"std(A) = {d['asym_std']:.4f}"
    )
    ax.set_title(
        f"{d['label']} — worst trial  (seed {worst[cond_key]['seed']})\n{metrics_str}",
        fontsize=9, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time relative to cue onset (ms)")

# Common x-axis limits: show last 500 ms of burn-in + cue + full delay
axes[0].set_xlim(-500, STIM_DURATION_MS + DELAY_MS)

fig.suptitle(
    "Asymmetry A(t) — worst-case trial per condition\n"
    f"(amp={AMP_FACTOR}×, uncorrected, n_nodes={N_NODES}, "
    f"w_inter={W_PYR_INTER}, σ={SIGMA_PYR_DEG}°, w_pv={W_PV_GLOBAL})",
    fontsize=11,
)

fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PATH}")
plt.show()
