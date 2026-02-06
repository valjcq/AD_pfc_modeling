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
    act_alpha7: ActivationDistribution = None
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


# The 8 conditions - APP conditions now use distributions
STUDY_CONDITIONS: dict[str, ExperimentalCondition] = {
    "WT": ExperimentalCondition(
        name="WT",
        label="Wild Type",
        act_alpha7=1.0, act_alpha5=1.0, act_beta2=1.0
    ),
    "WT_APP": ExperimentalCondition(
        name="WT APP",
        label="Wild Type + APP",
        act_alpha7=APP_ALPHA7, act_alpha5=APP_ALPHA5, act_beta2=APP_BETA2
    ),
    "a7_KO": ExperimentalCondition(
        name="a7 KO",
        label="alpha7 Knockout",
        act_alpha7=0.0, g_alpha7=0.0
    ),
    "a7_KO_APP": ExperimentalCondition(
        name="a7 KO APP",
        label="alpha7 Knockout + APP",
        act_alpha7=0.0, g_alpha7=0.0, act_alpha5=APP_ALPHA5, act_beta2=APP_BETA2
    ),
    "b2_KO": ExperimentalCondition(
        name="b2 KO",
        label="beta2 Knockout",
        act_beta2=0.0
    ),
    "b2_KO_APP": ExperimentalCondition(
        name="b2 KO APP",
        label="beta2 Knockout + APP",
        act_beta2=0.0, act_alpha7=APP_ALPHA7, act_alpha5=APP_ALPHA5
    ),
    "a5_KO": ExperimentalCondition(
        name="a5 KO",
        label="alpha5 Knockout",
        act_alpha5=0.0
    ),
    "a5_KO_APP": ExperimentalCondition(
        name="a5 KO APP",
        label="alpha5 Knockout + APP",
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
) -> CircuitParams:
    """
    Apply experimental condition parameters to base CircuitParams.

    For APP conditions, activation values are sampled from distributions
    to simulate biological variability in receptor desensitization.

    Parameters:
        base_params: Base CircuitParams to modify
        condition: ExperimentalCondition with activation distributions
        rng: Random generator for sampling (if None, uses mean values)

    Returns:
        Modified CircuitParams with sampled activation values
    """
    if rng is None:
        # Use mean values (no sampling)
        act_a7 = condition.act_alpha7.mean
        act_a5 = condition.act_alpha5.mean
        act_b2 = condition.act_beta2.mean
    else:
        # Sample from distributions
        act_a7 = condition.act_alpha7.sample(rng)
        act_a5 = condition.act_alpha5.sample(rng)
        act_b2 = condition.act_beta2.sample(rng)

    kwargs: dict = {
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


def _init_worker(base_params: CircuitParams, condition: ExperimentalCondition, cfg: StudyConfig) -> None:
    """Initialize worker process with shared parameters."""
    global _sim_args
    _sim_args = (base_params, condition, cfg)


def _run_single_sim(seed: int) -> np.ndarray:
    """Run a single simulation with the given seed. Used by ProcessPoolExecutor."""
    global _sim_args
    if _sim_args is None:
        raise RuntimeError("Worker not initialized")
    base_params, condition, cfg = _sim_args

    # Sample activation values unless fixed mode is enabled
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


def run_single_simulation(
    base_params: CircuitParams,
    condition: ExperimentalCondition,
    cfg: StudyConfig,
    seed: int,
) -> np.ndarray:
    """
    Run a single simulation and return mean rates.

    Parameters:
        base_params: Base CircuitParams
        condition: ExperimentalCondition (with distributions for APP)
        cfg: StudyConfig with simulation settings
        seed: Random seed (used for both activation sampling and simulation noise)

    Returns:
        Array of shape (4,) with mean rates [pyr, som, pv, vip]
    """
    # Sample activation values unless fixed mode is enabled
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
) -> np.ndarray:
    """
    Run N simulations for a condition and return all mean rates.

    For APP conditions, each run samples different activation values
    from the desensitization distributions.

    Parameters:
        base_params: Base CircuitParams
        condition: ExperimentalCondition to apply (may contain distributions)
        cfg: StudyConfig with simulation settings
        base_seed: Base random seed for generating per-run seeds

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
            initargs=(base_params, condition, cfg)
        ) as executor:
            futures = [executor.submit(_run_single_sim, seed) for seed in seeds]
            results = []
            with tqdm(total=cfg.n_runs, desc="Simulations", leave=False) as pbar:
                for future in futures:
                    results.append(future.result())
                    pbar.update()
    else:
        # Sequential execution
        results = [run_single_simulation(base_params, condition, cfg, seed) for seed in seeds]

    return np.array(results)  # Shape: (n_runs, 4)


def run_study(
    base_params: CircuitParams,
    cfg: StudyConfig,
    base_seed: int = 0,
    verbose: bool = True,
) -> StudyResults:
    """
    Run the full study across all conditions.

    Parameters:
        base_params: Base CircuitParams
        cfg: StudyConfig with simulation settings
        base_seed: Base random seed
        verbose: Whether to print progress

    Returns:
        StudyResults containing all data
    """
    rng = np.random.default_rng(base_seed)
    data: dict[str, np.ndarray] = {}

    for cond_key in CONDITION_ORDER:
        cond = STUDY_CONDITIONS[cond_key]
        print(f"Running {cond.label}")

        seed = int(rng.integers(0, 2**31 - 1))
        data[cond_key] = run_condition_batch(base_params, cond, cfg, seed)

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
    figsize: tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    unit: str = "transients/min",
):
    """
    Create box plots showing firing rate distributions, one per population.

    Layout: PYR (main population) on top (larger), then SOM, PV, VIP below.

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

    # Create figure with PYR larger on top, others below
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

    # PYR (main) - spans full width on top
    ax_pyr = fig.add_subplot(gs[0, :])

    # Other populations below
    ax_som = fig.add_subplot(gs[1, 0])
    ax_pv = fig.add_subplot(gs[1, 1])
    ax_vip = fig.add_subplot(gs[1, 2])

    axes = [ax_pyr, ax_som, ax_pv, ax_vip]
    pop_indices = {"PYR": 0, "SOM": 1, "PV": 2, "VIP": 3}

    # Condition labels
    condition_labels = [STUDY_CONDITIONS[k].name for k in results.conditions]

    for ax, pop_name in zip(axes, results.population_names):
        pop_idx = pop_indices[pop_name]
        color = POPULATION_COLORS[pop_name]

        # Collect data for this population across all conditions
        data = [results.data[cond_key][:, pop_idx] for cond_key in results.conditions]

        # Create box plot
        bp = ax.boxplot(
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
        ax.set_xticks(range(1, n_conditions + 1))
        ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=9 if pop_name == "PYR" else 8)

        # Y-axis
        ax.set_ylabel(f"Rate ({unit})", fontsize=10 if pop_name == "PYR" else 9)
        ax.set_ylim(bottom=0)

        # Title
        is_main = pop_name == "PYR"
        ax.set_title(
            pop_name,
            fontsize=13 if is_main else 11,
            fontweight='bold',
            color=color
        )

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

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
            fallback_path = save_path or "study_boxplots.png"
            if not save_path:
                fig.savefig(fallback_path, dpi=150, bbox_inches='tight')
                print(f"No display available. Figure saved to: {fallback_path}")

    return fig
