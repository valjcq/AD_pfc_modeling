from circuit_model import CircuitParams
from ring_attractor import RingParams, RingStimulus, simulate_ring, plot_ring_dashboard

# Setup
local_params = CircuitParams()
ring_params = RingParams(n_nodes=64, w_pyr_pyr_inter=0.5, sigma_pyr_deg=30.0)
stimulus = RingStimulus(center_deg=180.0, amplitude=10.0, onset_ms=500.0, duration_ms=250.0)

# Simulate
result = simulate_ring(local_params, ring_params, T_ms=4000.0, stimuli=[stimulus])

# Visualize
plot_ring_dashboard(result, save_path="dashboard.png")
