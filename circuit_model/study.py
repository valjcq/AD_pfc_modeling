"""
Batch simulation study across experimental conditions.

This module provides functionality to run simulations across multiple
experimental conditions (WT, APP, KO variants) and generate box plots
comparing firing rate distributions.

Usage:
    python -m circuit_model study [options]
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from tqdm import tqdm

from .params import CircuitParams
from .simulation import simulate_circuit, mean_rates
from .plotting import POPULATION_NAMES, POPULATION_COLORS


# =============================================================================
# EXPERIMENTAL CONDITIONS
# =============================================================================

@dataclass(frozen=True)
class ActivationDistribution:
    """
    Distribution for receptor activation values.

    For APP conditions, activation values are sampled from distributions:
    - α7: ~90% inactivated → 5-15% remains (mean=0.10, std=0.03)
    - α5: ~40% inactivated → 50-70% remains (mean=0.60, std=0.05)
    - β2: ~0-25% inactivated → 75-100% remains (uniform)
    """
    mean: float
    std: float = 0.0  # If 0, use fixed value (no sampling)
    lo: float = 0.0   # Lower bound (clipped)
    hi: float = 1.0   # Upper bound (clipped)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from this distribution."""
        if self.std == 0:
            return self.mean
        value = rng.normal(self.mean, self.std)
        return float(np.clip(value, self.lo, self.hi))


# APP desensitization distributions
APP_ALPHA7 = ActivationDistribution(mean=0.10, std=0.03, lo=0.02, hi=0.20)  # 80-98% inactivated
APP_ALPHA5 = ActivationDistribution(mean=0.60, std=0.05, lo=0.45, hi=0.75)  # 25-55% inactivated
APP_BETA2 = ActivationDistribution(mean=0.875, std=0.06, lo=0.75, hi=1.0)   # 0-25% inactivated


@dataclass(frozen=True)
class ExperimentalCondition:
    """Definition of an experimental condition."""
    name: str           # Short name for plots (e.g., "WT", "WT APP")
    label: str          # Full descriptive label
    is_app: bool = False    # Uses app_params in dual-params mode
    ko_alpha7: bool = False # alpha7 receptor is knocked out
    ko_alpha5: bool = False # alpha5 receptor is knocked out
    ko_beta2: bool = False  # beta2 receptor is knocked out
    act_alpha7: ActivationDistribution = None  # Used in single-params mode only
    act_alpha5: ActivationDistribution = None
    act_beta2: ActivationDistribution = None
    g_alpha7: Optional[float] = None  # None means use default from base params

    def __post_init__(self):
        # Convert float values to fixed distributions for convenience
        object.__setattr__(self, 'act_alpha7',
            self.act_alpha7 if isinstance(self.act_alpha7, ActivationDistribution)
            else ActivationDistribution(mean=self.act_alpha7 if self.act_alpha7 is not None else 1.0))
        object.__setattr__(self, 'act_alpha5',
            self.act_alpha5 if isinstance(self.act_alpha5, ActivationDistribution)
            else ActivationDistribution(mean=self.act_alpha5 if self.act_alpha5 is not None else 1.0))
        object.__setattr__(self, 'act_beta2',
            self.act_beta2 if isinstance(self.act_beta2, ActivationDistribution)
            else ActivationDistribution(mean=self.act_beta2 if self.act_beta2 is not None else 1.0))


# The 8 conditions.
# In single-params mode: APP conditions sample activations from distributions.
# In dual-params mode:   APP conditions use app_params; KO flags set activations to 0.
STUDY_CONDITIONS: dict[str, ExperimentalCondition] = {
    "WT": ExperimentalCondition(
        name="WT",
        label="Wild Type",
        act_alpha7=1.0, act_alpha5=1.0, act_beta2=1.0
    ),
    "WT_APP": ExperimentalCondition(
        name="WT APP",
        label="Wild Type + APP",
        is_app=True,
        act_alpha7=APP_ALPHA7, act_alpha5=APP_ALPHA5, act_beta2=APP_BETA2
    ),
    "a7_KO": ExperimentalCondition(
        name="a7 KO",
        label="alpha7 Knockout",
        ko_alpha7=True,
        act_alpha7=0.0, g_alpha7=0.0
    ),
    "a7_KO_APP": ExperimentalCondition(
        name="a7 KO APP",
        label="alpha7 Knockout + APP",
        is_app=True, ko_alpha7=True,
        act_alpha7=0.0, g_alpha7=0.0, act_alpha5=APP_ALPHA5, act_beta2=APP_BETA2
    ),
    "b2_KO": ExperimentalCondition(
        name="b2 KO",
        label="beta2 Knockout",
        ko_beta2=True,
        act_beta2=0.0
    ),
    "b2_KO_APP": ExperimentalCondition(
        name="b2 KO APP",
        label="beta2 Knockout + APP",
        is_app=True, ko_beta2=True,
        act_beta2=0.0, act_alpha7=APP_ALPHA7, act_alpha5=APP_ALPHA5
    ),
    "a5_KO": ExperimentalCondition(
        name="a5 KO",
        label="alpha5 Knockout",
        ko_alpha5=True,
        act_alpha5=0.0
    ),
    "a5_KO_APP": ExperimentalCondition(
        name="a5 KO APP",
        label="alpha5 Knockout + APP",
        is_app=True, ko_alpha5=True,
        act_alpha5=0.0, act_alpha7=APP_ALPHA7, act_beta2=APP_BETA2
    ),
}

# Condition order for plotting
CONDITION_ORDER = ["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP"]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StudyConfig:
    """Configuration for batch study."""
    n_runs: int = 50
    T_ms: float = 2500.0
    dt_ms: float = 0.1
    burn_in_ms: float = 1800.0
    window_ms: float = 500.0
    noise_type: str = "white"  # Use white noise by default
    tau_noise_ms: float = 5.0
    n_workers: Optional[int] = None  # Auto-detect
    fixed_receptor_values: bool = False  # If True, use mean instead of sampling


@dataclass
class StudyResults:
    """Container for batch study results."""
    conditions: list[str]  # Condition keys in order
    population_names: list[str]  # ["PYR", "SOM", "PV", "VIP"]
    data: dict[str, np.ndarray]  # condition_key -> (n_runs, 4) array
    config: StudyConfig


# =============================================================================
# CONDITION APPLICATION
# =============================================================================

def apply_condition(
    base_params: CircuitParams,
    condition: ExperimentalCondition,
    rng: Optional[np.random.Generator] = None,
    app_params: Optional[CircuitParams] = None,
) -> CircuitParams:
    """
    Apply experimental condition parameters to base CircuitParams.

    Two modes:

    Dual-params mode (app_params is not None):
        APP conditions use app_params as base (APP physiology already baked into the
        fit). KO conditions set the relevant activation to 0 on whichever base is
        chosen. No activation sampling occurs.

    Single-params mode (app_params is None):
        APP conditions sample activations from distributions (APP_ALPHA7 etc.) on
        top of base_params to simulate biological variability.

    Parameters:
        base_params: WT CircuitParams
        condition: ExperimentalCondition
        rng: Random generator for sampling (single-params mode only)
        app_params: APP CircuitParams for dual-params mode (optional)

    Returns:
        Modified CircuitParams
    """
    if app_params is not None:
        # Dual-params mode: choose base, then apply KO modifications only
        chosen = app_params if condition.is_app else base_params
        kwargs: dict = {}
        if condition.ko_alpha7:
            kwargs['act_alpha7'] = 0.0
            if condition.g_alpha7 is not None:
                kwargs['g_alpha7'] = condition.g_alpha7
        if condition.ko_alpha5:
            kwargs['act_alpha5'] = 0.0
        if condition.ko_beta2:
            kwargs['act_beta2'] = 0.0
        return replace(chosen, **kwargs) if kwargs else chosen

    # Single-params mode: sample or use mean activation values
    if rng is None:
        act_a7 = condition.act_alpha7.mean
        act_a5 = condition.act_alpha5.mean
        act_b2 = condition.act_beta2.mean
    else:
        act_a7 = condition.act_alpha7.sample(rng)
        act_a5 = condition.act_alpha5.sample(rng)
        act_b2 = condition.act_beta2.sample(rng)

    kwargs = {
        'act_alpha7': act_a7,
        'act_alpha5': act_a5,
        'act_beta2': act_b2,
    }
    if condition.g_alpha7 is not None:
        kwargs['g_alpha7'] = condition.g_alpha7
    return replace(base_params, **kwargs)


# =============================================================================
# BATCH SIMULATION
# =============================================================================

# Module-level arguments for pickling with ProcessPoolExecutor
_sim_args: Optional[tuple] = None


def _init_worker(
    base_params: CircuitParams,
    condition: ExperimentalCondition,
    cfg: StudyConfig,
    app_params: Optional[CircuitParams],
) -> None:
    """Initialize worker process with shared parameters."""
    global _sim_args
    _sim_args = (base_params, condition, cfg, app_params)


def _run_single_sim(seed: int) -> np.ndarray:
    """Run a single simulation with the given seed. Used by ProcessPoolExecutor."""
    global _sim_args
    if _sim_args is None:
        raise RuntimeError("Worker not initialized")
    base_params, condition, cfg, app_params = _sim_args

    if app_params is not None:
        # Dual-params mode: no activation sampling
        params = apply_condition(base_params, condition, app_params=app_params)
    else:
        # Single-params mode: sample activation values unless fixed mode is enabled
        rng = None if cfg.fixed_receptor_values else np.random.default_rng(seed)
        params = apply_condition(base_params, condition, rng)

    result = simulate_circuit(
        params,
        T_ms=cfg.T_ms,
        dt_ms=cfg.dt_ms,
        seed=seed,
        noise_type=cfg.noise_type,
        tau_noise_ms=cfg.tau_noise_ms,
        use_transient=False,
    )
    return mean_rates(result, burn_in_ms=cfg.burn_in_ms, window_ms=cfg.window_ms)


def run_single_simulation(
    base_params: CircuitParams,
    condition: ExperimentalCondition,
    cfg: StudyConfig,
    seed: int,
    app_params: Optional[CircuitParams] = None,
) -> np.ndarray:
    """
    Run a single simulation and return mean rates.

    Parameters:
        base_params: WT CircuitParams
        condition: ExperimentalCondition
        cfg: StudyConfig with simulation settings
        seed: Random seed (used for both activation sampling and simulation noise)
        app_params: APP CircuitParams for dual-params mode (optional)

    Returns:
        Array of shape (4,) with mean rates [pyr, som, pv, vip]
    """
    if app_params is not None:
        params = apply_condition(base_params, condition, app_params=app_params)
    else:
        rng = None if cfg.fixed_receptor_values else np.random.default_rng(seed)
        params = apply_condition(base_params, condition, rng)

    result = simulate_circuit(
        params,
        T_ms=cfg.T_ms,
        dt_ms=cfg.dt_ms,
        seed=seed,
        noise_type=cfg.noise_type,
        tau_noise_ms=cfg.tau_noise_ms,
    )
    return mean_rates(result, burn_in_ms=cfg.burn_in_ms, window_ms=cfg.window_ms)


def run_condition_batch(
    base_params: CircuitParams,
    condition: ExperimentalCondition,
    cfg: StudyConfig,
    base_seed: int,
    app_params: Optional[CircuitParams] = None,
) -> np.ndarray:
    """
    Run N simulations for a condition and return all mean rates.

    In single-params mode, APP conditions sample activation values per run.
    In dual-params mode, APP conditions always use app_params as base.

    Parameters:
        base_params: WT CircuitParams
        condition: ExperimentalCondition to apply
        cfg: StudyConfig with simulation settings
        base_seed: Base random seed for generating per-run seeds
        app_params: APP CircuitParams for dual-params mode (optional)

    Returns:
        Array of shape (n_runs, 4) with mean rates for each run
    """
    # Generate seeds - each run gets unique seed for both activation sampling and noise
    rng = np.random.default_rng(base_seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(cfg.n_runs)]

    # Determine number of workers
    n_workers = cfg.n_workers
    if n_workers is None:
        n_workers = min(cfg.n_runs, os.cpu_count() or 4)

    # Run simulations
    if n_workers > 1 and cfg.n_runs > 1:
        # Parallel execution
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(base_params, condition, cfg, app_params)
        ) as executor:
            futures = [executor.submit(_run_single_sim, seed) for seed in seeds]
            results = []
            with tqdm(total=cfg.n_runs, desc="Simulations", leave=False) as pbar:
                for future in futures:
                    results.append(future.result())
                    pbar.update()
    else:
        # Sequential execution
        results = [
            run_single_simulation(base_params, condition, cfg, seed, app_params=app_params)
            for seed in seeds
        ]

    return np.array(results)  # Shape: (n_runs, 4)


def run_study(
    base_params: CircuitParams,
    cfg: StudyConfig,
    base_seed: int = 0,
    verbose: bool = True,
    app_params: Optional[CircuitParams] = None,
) -> StudyResults:
    """
    Run the full study across all conditions.

    Parameters:
        base_params: WT CircuitParams
        cfg: StudyConfig with simulation settings
        base_seed: Base random seed
        verbose: Whether to print progress
        app_params: APP CircuitParams for dual-params mode. When provided, APP
            conditions use app_params as their base (APP physiology baked into
            the fit) and KO conditions still set the relevant activation to 0.

    Returns:
        StudyResults containing all data
    """
    rng = np.random.default_rng(base_seed)
    data: dict[str, np.ndarray] = {}

    for cond_key in CONDITION_ORDER:
        cond = STUDY_CONDITIONS[cond_key]
        print(f"Running {cond.label}")

        seed = int(rng.integers(0, 2**31 - 1))
        data[cond_key] = run_condition_batch(base_params, cond, cfg, seed, app_params=app_params)

        if verbose:
            means = data[cond_key].mean(axis=0)
            print(f"  Mean rates: PYR={means[0]:.2f}, SOM={means[1]:.2f}, "
                  f"PV={means[2]:.2f}, VIP={means[3]:.2f}")

    return StudyResults(
        conditions=CONDITION_ORDER,
        population_names=list(POPULATION_NAMES),
        data=data,
        config=cfg,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_study_boxplots(
    results: StudyResults,
    title: str = "Firing Rate Distribution by Condition",
    figsize: tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    unit: str = "transients/min",
):
    """
    Create box plots showing firing rate distributions for the PYR population.

    Parameters:
        results: StudyResults from run_study
        title: Figure title
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        show: Whether to call plt.show()
        unit: Rate unit for Y-axis label

    Returns:
        The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from .plotting import _check_display_available

    n_conditions = len(results.conditions)

    fig, ax_pyr = plt.subplots(figsize=figsize, constrained_layout=True)

    pop_idx = 0  # PYR
    color = POPULATION_COLORS["PYR"]

    # Condition labels
    condition_labels = [STUDY_CONDITIONS[k].name for k in results.conditions]

    # Collect data for PYR across all conditions
    data = [results.data[cond_key][:, pop_idx] for cond_key in results.conditions]

    # Create box plot
    bp = ax_pyr.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
    )

    # Color all boxes with the population color
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    # X-axis
    ax_pyr.set_xticks(range(1, n_conditions + 1))
    ax_pyr.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=9)

    # Y-axis: zoom in on the data range with 20% padding
    all_vals = np.concatenate(data)
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = max((vmax - vmin) * 0.2, 0.05 * vmax)
    ax_pyr.set_ylabel(f"Rate ({unit})", fontsize=10)
    ax_pyr.set_ylim(max(0, vmin - margin), vmax + margin)

    # Title
    ax_pyr.set_title("PYR", fontsize=13, fontweight='bold', color=color)

    # Style
    ax_pyr.spines['top'].set_visible(False)
    ax_pyr.spines['right'].set_visible(False)
    ax_pyr.grid(axis='y', alpha=0.3)

    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        if _check_display_available():
            plt.show(block=True)
        else:
            noise_type = results.config.noise_type
            fallback_path = save_path or f"study_boxplots_{noise_type}.png"
            if not save_path:
                fig.savefig(fallback_path, dpi=150, bbox_inches='tight')
                print(f"No display available. Figure saved to: {fallback_path}")

    return fig
