"""
Ring Attractor Network for Working Memory Simulations.

This subpackage implements a ring attractor network built on top of the
4-population PFC circuit model. Each node on the ring is a full local
circuit (PYR, PV, SOM, VIP) with inter-node connectivity enabling
persistent activity bumps for working memory.

Architecture:
- N nodes arranged in a circle (default: 64)
- Each node: 4-population circuit with existing dynamics
- Inter-node PYR->PYR: Local excitation with Gaussian profile
- Inter-node PYR->PV: Global excitation of PV (E->I->E inhibitory loop)
- SOM, VIP: Local only (no inter-node connections)

Usage:
    from circuit_model import CircuitParams
    from circuit_model.ring import (
        RingParams,
        RingStimulus,
        WorkingMemoryProtocol,
        simulate_ring,
        decode_bump_center,
        plot_ring_dashboard,
    )

    # Setup parameters
    local_params = CircuitParams()
    ring_params = RingParams(n_nodes=64, sigma_pyr_deg=30.0)

    # Define stimulus
    stimulus = RingStimulus(
        center_deg=180.0,
        amplitude=5.0,
        onset_ms=500.0,
        duration_ms=250.0,
    )

    # Run simulation
    result = simulate_ring(
        local_params,
        ring_params,
        T_ms=4000.0,
        stimuli=[stimulus],
    )

    # Analyze results
    center_deg, amplitude = decode_bump_center(result)

    # Visualize
    plot_ring_dashboard(result)
"""

# Ring parameters
from .params import RingParams

# Connectivity
from .connectivity import (
    angular_distance,
    gaussian_profile,
    build_pyr_pyr_weights,
    build_pv_pyr_weights,
    build_som_pyr_weights,
    RingConnectivity,
)

# Stimulus
from .stimulus import (
    RingStimulus,
    WorkingMemoryProtocol,
    compute_stimulus_current,
)

# Simulation
from .simulation import (
    RingSimulationResult,
    simulate_ring,
    simulate_ring_batch,
    mean_rates_ring,
)

# Analysis
from .analysis import (
    population_vector_decode,
    decode_bump_center,
    estimate_bump_width,
    angular_distance_deg,
    compute_bump_metrics,
    compute_bump_asymmetry,
    compute_metrics_at_delay_times,
    compute_working_memory_accuracy,
    aggregate_metrics_across_trials,
    aggregate_single_metrics,
)

# Plotting
from .plotting import (
    CONDITION_COLORS,
    plot_ring_activity_heatmap,
    plot_ring_snapshot,
    plot_bump_tracking,
    plot_node_activity,
    plot_bump_metrics_over_time,
    plot_ring_dashboard,
    plot_ring_connectome,
    extract_comparison_data,
    plot_bump_metrics_comparison,
    plot_metrics_vs_delay,
    plot_metrics_vs_amplitude,
    print_simulation_summary,
)

__all__ = [
    # Ring parameters
    "RingParams",
    # Connectivity
    "angular_distance",
    "gaussian_profile",
    "build_pyr_pyr_weights",
    "build_pv_pyr_weights",
    "build_som_pyr_weights",
    "RingConnectivity",
    # Stimulus
    "RingStimulus",
    "WorkingMemoryProtocol",
    "compute_stimulus_current",
    # Simulation
    "RingSimulationResult",
    "simulate_ring",
    "simulate_ring_batch",
    "mean_rates_ring",
    # Analysis
    "population_vector_decode",
    "decode_bump_center",
    "estimate_bump_width",
    "angular_distance_deg",
    "compute_bump_metrics",
    "compute_bump_asymmetry",
    "compute_metrics_at_delay_times",
    "compute_working_memory_accuracy",
    "aggregate_metrics_across_trials",
    "aggregate_single_metrics",
    # Plotting
    "CONDITION_COLORS",
    "plot_ring_activity_heatmap",
    "plot_ring_snapshot",
    "plot_bump_tracking",
    "plot_node_activity",
    "plot_bump_metrics_over_time",
    "plot_ring_dashboard",
    "plot_ring_connectome",
    "extract_comparison_data",
    "plot_bump_metrics_comparison",
    "plot_metrics_vs_delay",
    "plot_metrics_vs_amplitude",
    "print_simulation_summary",
]
