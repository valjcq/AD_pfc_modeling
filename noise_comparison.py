"""
Noise level comparison for ring attractor bump position drift.

Runs 3 simulations with sigma_s = 0.5, 2.0, 5.0 and plots decoded bump
centre (deg) over the delay period for each, overlaid on a single figure.

Mirrors the ring-run command:
  --w_pyr_pyr_inter 5.5 --sigma_pyr_deg 30.0 --w_pv_global 5.0
  --amplitude 20 --delay_ms 10000 --n_nodes 512
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from circuit_model.params import CircuitParams
from circuit_model.ring.params import RingParams
from circuit_model.ring.simulation import simulate_ring, RingStimulus
from circuit_model.ring.analysis import decode_bump_center

# ── Constants matching ring-run CLI defaults ─────────────────────────────────
BURN_IN_MS       = 10_000.0
STIM_ONSET_MS    = BURN_IN_MS + 500.0   # 10 500 ms
STIM_DURATION_MS = 250.0
STIM_CENTER_DEG  = 180.0
STIM_SIGMA_DEG   = 20.0

DELAY_MS         = 10_000.0
AMP_FACTOR       = 20.0
N_NODES          = 128
SEED             = 42

STIM_OFFSET_MS   = STIM_ONSET_MS + STIM_DURATION_MS   # 10 750 ms
DELAY_END_MS     = STIM_OFFSET_MS + DELAY_MS           # 20 750 ms
T_MS             = DELAY_END_MS

RECORD_DT_MS     = 5.0   # record every 5 ms (enough for drift visualisation)

# ── Ring connectivity parameters ─────────────────────────────────────────────
ring_params = RingParams(
    n_nodes=N_NODES,
    w_pyr_pyr_inter=6,
    sigma_pyr_deg=10.0,
    w_pv_global=4.0,
    sigma_pv_deg=90.0,
    pv_global_type="gaussian",
)

# ── Noise levels to compare ──────────────────────────────────────────────────
NOISE_LEVELS = [0.5, 2.0, 5.0]
COLORS       = ["steelblue", "darkorange", "crimson"]

# ── Run simulations ──────────────────────────────────────────────────────────
base_params_default = CircuitParams()
actual_current = AMP_FACTOR * base_params_default.I_ext_pyr()
stimuli = [
    RingStimulus(
        center_deg=STIM_CENTER_DEG,
        amplitude=actual_current,
        sigma_deg=STIM_SIGMA_DEG,
        onset_ms=STIM_ONSET_MS,
        duration_ms=STIM_DURATION_MS,
    )
]

results = {}
for sigma_s in NOISE_LEVELS:
    print(f"\n=== sigma_s = {sigma_s} ===")
    local_params = replace(base_params_default, sigma_s=sigma_s)

    result = simulate_ring(
        local_params, ring_params,
        T_ms=T_MS,
        stimuli=stimuli,
        seed=SEED,
        noise_type="white",
        record_dt_ms=RECORD_DT_MS,
    )
    results[sigma_s] = result
    print(f"  Done. {result.r.shape[0]} recorded time points.")

# ── Decode bump centre and build comparison plot ──────────────────────────────
print("\nDecoding bump centres and plotting …")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, sigma_s, color in zip(axes, NOISE_LEVELS, COLORS):
    result = results[sigma_s]
    t_ms   = result.t_ms

    # Delay window: from stimulus offset to end of delay
    mask    = (t_ms >= STIM_OFFSET_MS) & (t_ms <= DELAY_END_MS)
    t_delay = (t_ms[mask] - STIM_OFFSET_MS) / 1000.0   # seconds from stim offset

    center_rad, amplitude = decode_bump_center(result, population=0)  # PYR
    center_deg_all = np.degrees(center_rad)   # [0, 360)

    c_delay = center_deg_all[mask]
    a_delay = amplitude[mask]

    # Compute signed angular deviation from cue, wrapped to [-180, +180)
    dev = c_delay - STIM_CENTER_DEG
    dev = (dev + 180.0) % 360.0 - 180.0   # wrap to [-180, 180)

    # Linear drift estimate (deg/s) from deviation
    if len(t_delay) > 1:
        drift_deg_s = np.polyfit(t_delay, dev, 1)[0]
    else:
        drift_deg_s = float("nan")

    ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=1, label="Cue (0° deviation)")
    ax.plot(t_delay, dev, color=color, lw=1.2, alpha=0.85, zorder=2)
    ax.set_ylabel("Deviation from cue (°)", fontsize=10)
    ax.set_ylim(-185, 185)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title(
        f"σ_s = {sigma_s}   |   linear drift = {drift_deg_s:+.2f} °/s   "
        f"|   mean decode amplitude = {a_delay.mean():.3f}",
        fontsize=10, color=color,
    )
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time from stimulus offset (s)", fontsize=12)

fig.suptitle(
    f"Bump position deviation from cue vs noise level (σ_s)\n"
    f"w_inter=5.5, σ_pyr=30°, w_pv=5.0, amp={AMP_FACTOR:.0f}×, "
    f"delay={int(DELAY_MS/1000)}s, N={N_NODES}",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()

os.makedirs("figs", exist_ok=True)
out_path = "figs/noise_comparison_bump_drift.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
