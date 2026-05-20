"""
PFC Circuit Model: 4-Population Rate Model with Parameter Optimization.

This package implements a computational model of the prefrontal cortex (PFC)
microcircuit with 4 neural populations (stored in arrays as [PYR, SOM, PV, VIP]):
- PYR: Pyramidal cells (excitatory)
- SOM: Somatostatin interneurons (inhibitory, dendritic targeting)
- PV: Parvalbumin interneurons (fast-spiking inhibitory)
- VIP: VIP interneurons (inhibitory, disinhibitory)

The model uses the Wong-Wang transfer function and supports:
- Nevergrad-based parameter optimization
- Nicotinic receptor knockout simulations (alpha7, alpha5, beta2)
- Multiple noise types (white, Ornstein-Uhlenbeck)
- Ring attractor network for working memory (see circuit_model.ring)

Usage:
    # Single-node circuit
    from circuit_model import CircuitParams, simulate_circuit, mean_rates

    params = CircuitParams()
    result = simulate_circuit(params, T_ms=1000)
    rates = mean_rates(result, burn_in_ms=500, window_ms=500)

    # Ring attractor network
    from circuit_model.ring import RingParams, simulate_ring, RingStimulus

    # From command line
    python -m circuit_model run                    # single-node simulation
    python -m circuit_model ring-run               # ring attractor simulation
    python -m circuit_model ring-optimize          # joint circuit + ring fit
"""

from .params import CircuitParams, ParamBound, default_bounds
from .transfer import phi_wong_wang
from .simulation import SimulationResult, simulate_circuit, mean_rates, NoiseType, validate_fast_loop
from .loss import TargetRates, FitConfig, loss_from_means, loss_from_ko_pyr, ach_ratio_penalty
from .optimization import (
    KOMeans,
    Candidate,
    run_trials,
    evaluate_params,
    build_nevergrad_parametrization,
    params_from_ng_dict,
    nevergrad_optimize,
)
from .io import load_params_json, save_params_json, format_params_as_code, log_best_result
from .plotting import (
    plot_firing_rates,
    plot_adaptation,
    plot_simulation_dashboard,
    plot_mean_rates_bar,
    print_simulation_summary,
    POPULATION_NAMES,
    POPULATION_COLORS,
)
from .study import (
    ExperimentalCondition,
    STUDY_CONDITIONS,
    CONDITION_ORDER,
    StudyConfig,
    StudyResults,
    apply_condition,
    run_single_simulation,
    run_condition_batch,
    run_study,
    plot_study_boxplots,
)
from .cli import main

__all__ = [
    # Parameters
    "CircuitParams",
    "ParamBound",
    "default_bounds",
    # Transfer function
    "phi_wong_wang",
    # Simulation
    "SimulationResult",
    "simulate_circuit",
    "mean_rates",
    "NoiseType",
    "validate_fast_loop",
    # Loss/targets
    "TargetRates",
    "FitConfig",
    "loss_from_means",
    "loss_from_ko_pyr",
    "ach_ratio_penalty",
    # Optimization
    "KOMeans",
    "Candidate",
    "run_trials",
    "evaluate_params",
    "build_nevergrad_parametrization",
    "params_from_ng_dict",
    "nevergrad_optimize",
    # I/O
    "load_params_json",
    "save_params_json",
    "format_params_as_code",
    "log_best_result",
    # Plotting
    "plot_firing_rates",
    "plot_adaptation",
    "plot_simulation_dashboard",
    "plot_mean_rates_bar",
    "print_simulation_summary",
    "POPULATION_NAMES",
    "POPULATION_COLORS",
    # Study
    "ExperimentalCondition",
    "STUDY_CONDITIONS",
    "CONDITION_ORDER",
    "StudyConfig",
    "StudyResults",
    "apply_condition",
    "run_single_simulation",
    "run_condition_batch",
    "run_study",
    "plot_study_boxplots",
    # CLI
    "main",
]
