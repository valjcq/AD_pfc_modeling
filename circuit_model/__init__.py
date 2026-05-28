"""
PFC Circuit Model: 5-Population Rate Model with Parameter Optimization.

This package implements a computational model of the prefrontal cortex (PFC)
microcircuit with 5 neural populations (stored in arrays as [PYR, SOM, PV, VIP, NDNF]):
- PYR:  Pyramidal cells (excitatory)
- SOM:  Somatostatin interneurons (subtractive dendritic inhibition)
- PV:   Parvalbumin interneurons (fast-spiking, divisive/shunting inhibition)
- VIP:  VIP interneurons (disinhibitory)
- NDNF: NDNF interneurons (subtractive dendritic inhibition; α7+β2 receptors)

The model uses the Wong-Wang transfer function and supports:
- Nevergrad-based parameter optimization with TwoPointsDE
- Nicotinic receptor knockout simulations (global α7, α5, β2)
- Multiple noise types (white, Ornstein-Uhlenbeck)

Usage:
    from circuit_model import CircuitParams, simulate_circuit, mean_rates

    params = CircuitParams()
    result = simulate_circuit(params, T_ms=1000)
    rates = mean_rates(result, burn_in_ms=500, window_ms=500)

    # From command line
    python -m circuit_model run
    python -m circuit_model optimize --target_pyr 4 --target_som 3 --target_pv 2 \\
        --target_vip 2 --target_ndnf 3 --optimizer twopointde
"""

from .params import CircuitParams, ParamBound, default_bounds
from .transfer import phi_wong_wang
from .simulation import SimulationResult, simulate_circuit, mean_rates, NoiseType, validate_fast_loop
from .loss import TargetRates, FitConfig, loss_from_means_normalized, loss_from_ko_normalized, DrugTarget
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
    "loss_from_means_normalized",
    "loss_from_ko_normalized",
    "DrugTarget",
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
