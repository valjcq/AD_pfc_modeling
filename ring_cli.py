import argparse
import os
from pathlib import Path

from circuit_model import CircuitParams, load_params_json
from ring_attractor import (
    RingParams, RingStimulus, simulate_ring,
    plot_ring_dashboard, plot_ring_connectome, plot_bump_metrics_over_time,
)

parser = argparse.ArgumentParser(description="Ring attractor simulation")
parser.add_argument("--params_json", type=str, default="",
                    help="Load local circuit parameters from JSON file")
args = parser.parse_args()

# === Timing ===
burn_in_ms = 10000.0
stim_onset_ms = burn_in_ms + 500.0
stim_duration_ms = 250.0
delay_ms = 3000.0
T_ms = stim_onset_ms + stim_duration_ms + delay_ms

# === Cutoff signal (brief global suppression after cue to test bump persistence) ===
cutoff_enabled = False
cutoff_amplitude = -50.0        # negative = suppressive
cutoff_delay_after_cue_ms = 100.0  # how long after cue offset
cutoff_duration_ms = 50.0

# Setup
if args.params_json:
    local_params = load_params_json(args.params_json)
    print(f"Loaded parameters from: {args.params_json}")
    out_dir = os.path.join("figs/ring", Path(args.params_json).stem)
else:
    local_params = CircuitParams()
    print("Using default parameters")
    out_dir = os.path.join("figs/ring", "default")
os.makedirs(out_dir, exist_ok=True)

ring_params = RingParams(n_nodes=128, w_pyr_pyr_inter=0.5, sigma_pyr_deg=10.0, w_pv_global=2)

# Build stimulus list
stimuli = [
    RingStimulus(center_deg=180.0, amplitude=10, onset_ms=stim_onset_ms, duration_ms=stim_duration_ms),
]
if cutoff_enabled:
    cutoff_onset = stim_onset_ms + stim_duration_ms + cutoff_delay_after_cue_ms
    stimuli.append(RingStimulus(
        center_deg=180.0, amplitude=cutoff_amplitude,
        sigma_deg=9999.0,  # global: covers all nodes
        onset_ms=cutoff_onset, duration_ms=cutoff_duration_ms,
    ))

# Simulate
result = simulate_ring(local_params, ring_params, T_ms=T_ms, stimuli=stimuli)

# Visualize (skip burn-in)
t_plot_start = burn_in_ms
plot_ring_dashboard(result, save_path=os.path.join(out_dir, "dashboard.png"), time_range=(t_plot_start, T_ms))
plot_bump_metrics_over_time(result, time_range=(t_plot_start, T_ms))
import matplotlib.pyplot as plt
plt.savefig(os.path.join(out_dir, "bump_metrics.png"), dpi=150, bbox_inches="tight")
plot_ring_connectome(ring_params, save_path=os.path.join(out_dir, "connectome.png"))

print(f"\nFigures saved to {out_dir}/")
