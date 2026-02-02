#!/usr/bin/env python3
"""
PFC Circuit Model: 4-Population Rate Model with Parameter Optimization.

This is a backward-compatible wrapper that re-exports all components from
the circuit_model package. For new code, prefer importing from the package
directly:

    from circuit_model import CircuitParams, simulate_circuit, mean_rates

This file maintains backward compatibility for scripts that import from
circuit_model.py directly.

For command-line usage:
    # Run simulation and plot
    python -m circuit_model run
    python -m circuit_model run --noise_type ou --T_ms 5000

    # Run optimization
    python -m circuit_model optimize --target_pyr 5 --target_som 10 --target_pv 15 --target_vip 8
"""

# Re-export everything from the package for backward compatibility
from circuit_model import (
    # Parameters
    CircuitParams,
    ParamBound,
    default_bounds,
    # Transfer function
    phi_wong_wang,
    # Simulation
    SimulationResult,
    simulate_circuit,
    mean_rates,
    NoiseType,
    # Loss/targets
    TargetRates,
    FitConfig,
    loss_from_means,
    loss_from_ko_pyr,
    # Optimization
    KOMeans,
    Candidate,
    run_trials,
    evaluate_params,
    build_nevergrad_parametrization,
    params_from_ng_dict,
    nevergrad_optimize,
    # I/O
    load_params_json,
    save_params_json,
    format_params_as_code,
    log_best_result,
    # Plotting
    plot_firing_rates,
    plot_adaptation,
    plot_simulation_dashboard,
    plot_mean_rates_bar,
    print_simulation_summary,
    POPULATION_NAMES,
    POPULATION_COLORS,
    # CLI
    main,
)

if __name__ == "__main__":
    main()
