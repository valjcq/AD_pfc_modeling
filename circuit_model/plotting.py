"""
Visualization utilities for the circuit model.

This module provides functions to plot simulation results:
- Firing rates over time for all populations
- Adaptation currents over time
- Combined dashboard view

Note: Requires matplotlib. Install with: pip install matplotlib
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np


def _check_display_available() -> bool:
    """Check if a display is available for GUI plotting."""
    # Check common display environment variables
    if os.environ.get("DISPLAY"):
        return True
    # WSL with WSLg
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Check if running in a notebook
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

if TYPE_CHECKING:
    from .simulation import SimulationResult

# Population names and colors (colorblind-friendly palette)
POPULATION_NAMES = ["PYR", "SOM", "PV", "VIP"]
POPULATION_COLORS = {
    "PYR": "#E69F00",  # Orange - excitatory
    "SOM": "#56B4E9",  # Sky blue
    "PV": "#009E73",   # Bluish green
    "VIP": "#CC79A7",  # Reddish purple
}
ADAPTATION_COLORS = {
    "PYR": "#D55E00",  # Vermillion (darker orange)
    "SOM": "#0072B2",  # Blue (darker)
}
TRANSIENT_COLOR = "#888888"  # Gray for transient markers


def _add_transient_markers(
    ax,
    transient_window: tuple[float, float],
    time_range: Optional[tuple[float, float]] = None,
    add_legend: bool = False,
):
    """
    Add vertical lines and shading to indicate transient current window.

    Parameters:
        ax: Matplotlib axis
        transient_window: (start_ms, end_ms) of transient
        time_range: Optional (t_start, t_end) for clipping
        add_legend: Whether to add a legend entry
    """
    t_start, t_end = transient_window

    # Clip to visible range if specified
    if time_range is not None:
        vis_start, vis_end = time_range
        # Only draw if transient overlaps with visible range
        if t_end < vis_start or t_start > vis_end:
            return
        # Clip transient window to visible range
        t_start = max(t_start, vis_start)
        t_end = min(t_end, vis_end)

    # Shaded region
    ax.axvspan(t_start, t_end, alpha=0.15, color=TRANSIENT_COLOR,
               label="Transient ON" if add_legend else None)
    # Vertical lines at boundaries
    ax.axvline(transient_window[0], color=TRANSIENT_COLOR, linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.axvline(transient_window[1], color=TRANSIENT_COLOR, linestyle="--",
               linewidth=1.5, alpha=0.7)


def plot_firing_rates(
    result: "SimulationResult",
    ax=None,
    title: str = "Population Firing Rates",
    show_legend: bool = True,
    time_range: Optional[tuple[float, float]] = None,
    show_transient: bool = True,
    unit: str = "transients/min",
):
    """
    Plot firing rates over time for all 4 populations.

    Parameters:
        result: SimulationResult from simulate_circuit
        ax: Matplotlib axis (creates new figure if None)
        title: Plot title
        show_legend: Whether to show legend
        time_range: Optional (t_start, t_end) in ms to zoom in
        show_transient: Whether to show transient window markers (if present)
        unit: Rate unit for Y-axis label (default: "transients/min")

    Returns:
        The matplotlib axis object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    t = result.t_ms
    r = result.r

    # Apply time range filter if specified
    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t = t[mask]
        r = r[mask]

    # Draw transient markers first (so they're behind the data)
    if show_transient and result.transient_window is not None:
        _add_transient_markers(ax, result.transient_window, time_range, add_legend=show_legend)

    for i, name in enumerate(POPULATION_NAMES):
        ax.plot(t, r[:, i], label=name, color=POPULATION_COLORS[name], linewidth=1.5)

    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel(f"Firing Rate ({unit})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)

    return ax


def plot_adaptation(
    result: "SimulationResult",
    ax=None,
    title: str = "Adaptation Currents",
    show_legend: bool = True,
    time_range: Optional[tuple[float, float]] = None,
    show_transient: bool = True,
):
    """
    Plot adaptation currents (I_adapt) over time for PYR and SOM.

    Parameters:
        result: SimulationResult from simulate_circuit
        ax: Matplotlib axis (creates new figure if None)
        title: Plot title
        show_legend: Whether to show legend
        time_range: Optional (t_start, t_end) in ms to zoom in
        show_transient: Whether to show transient window markers (if present)

    Returns:
        The matplotlib axis object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    t = result.t_ms
    I_adapt = result.I_adapt

    # Apply time range filter if specified
    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t = t[mask]
        I_adapt = I_adapt[mask]

    # Draw transient markers first (so they're behind the data)
    if show_transient and result.transient_window is not None:
        _add_transient_markers(ax, result.transient_window, time_range, add_legend=False)

    ax.plot(t, I_adapt[:, 0], label="I_adapt (PYR)", color=ADAPTATION_COLORS["PYR"], linewidth=1.5)
    ax.plot(t, I_adapt[:, 1], label="I_adapt (SOM)", color=ADAPTATION_COLORS["SOM"], linewidth=1.5)

    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Adaptation Current", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)

    return ax


def plot_simulation_dashboard(
    result: "SimulationResult",
    title: str = "Circuit Model Simulation",
    time_range: Optional[tuple[float, float]] = None,
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    unit: str = "transients/min",
):
    """
    Create a comprehensive dashboard showing simulation results.

    Creates a figure with:
    - Top: Firing rates for all 4 populations
    - Middle: Individual population subplots
    - Bottom: Adaptation currents

    Parameters:
        result: SimulationResult from simulate_circuit
        title: Main figure title
        time_range: Optional (t_start, t_end) in ms to zoom in
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        show: Whether to call plt.show()
        unit: Rate unit for Y-axis labels (default: "transients/min")

    Returns:
        The matplotlib figure object
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Create grid: 3 rows
    # Row 0: Combined firing rates (spans full width)
    # Row 1: Individual populations (4 subplots)
    # Row 2: Adaptation currents (spans full width)
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1.5, 1])

    # Top plot: Combined firing rates
    ax_combined = fig.add_subplot(gs[0, :])
    plot_firing_rates(result, ax=ax_combined, title="All Populations", time_range=time_range, unit=unit)

    # Middle row: Individual populations
    t = result.t_ms
    r = result.r
    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[mask]
        r_plot = r[mask]
    else:
        t_plot = t
        r_plot = r

    for i, name in enumerate(POPULATION_NAMES):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t_plot, r_plot[:, i], color=POPULATION_COLORS[name], linewidth=1.2)
        ax.set_title(name, fontsize=11, fontweight="bold", color=POPULATION_COLORS[name])
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel(f"Rate ({unit})", fontsize=9)
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # Bottom plot: Adaptation currents
    ax_adapt = fig.add_subplot(gs[2, :])
    plot_adaptation(result, ax=ax_adapt, title="Adaptation Currents", time_range=time_range)

    # Main title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        if _check_display_available():
            plt.show(block=True)  # block=True ensures window stays open
        else:
            # No display available (e.g., WSL without X server), save to file
            fallback_path = save_path or "circuit_simulation.png"
            if not save_path:
                fig.savefig(fallback_path, dpi=150, bbox_inches="tight")
                print(f"No display available (WSL/headless). Figure saved to: {fallback_path}")
                print("Tip: Use --save_plot <filename> to specify output path")

    return fig


def plot_mean_rates_bar(
    means: np.ndarray,
    target: Optional[np.ndarray] = None,
    ax=None,
    title: str = "Mean Firing Rates",
    unit: str = "transients/min",
):
    """
    Plot mean firing rates as a bar chart, optionally with target comparison.

    Parameters:
        means: Array of shape (4,) with mean rates [pyr, som, pv, vip]
        target: Optional array of shape (4,) with target rates
        ax: Matplotlib axis (creates new figure if None)
        title: Plot title
        unit: Rate unit for Y-axis label (default: "transients/min")

    Returns:
        The matplotlib axis object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(POPULATION_NAMES))
    width = 0.35

    colors = [POPULATION_COLORS[name] for name in POPULATION_NAMES]

    if target is not None:
        # Side-by-side bars
        bars1 = ax.bar(x - width/2, means, width, label="Simulated", color=colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, target, width, label="Target", color=colors, alpha=0.4, edgecolor="black", linewidth=1.5)
        ax.legend()
    else:
        bars1 = ax.bar(x, means, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars1 if target is None else bars1, means):
        ax.annotate(f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(POPULATION_NAMES)
    ax.set_ylabel(f"Firing Rate ({unit})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def print_simulation_summary(result: "SimulationResult", burn_in_ms: float = 0.0) -> dict:
    """
    Print and return a summary of simulation results.

    Parameters:
        result: SimulationResult from simulate_circuit
        burn_in_ms: Time to skip for computing statistics

    Returns:
        Dictionary with summary statistics
    """
    from .simulation import mean_rates

    dt = float(result.t_ms[1] - result.t_ms[0])
    start_idx = int(np.floor(burn_in_ms / dt))

    r_after_burnin = result.r[start_idx:]
    t_after_burnin = result.t_ms[start_idx:]

    means = np.mean(r_after_burnin, axis=0)
    stds = np.std(r_after_burnin, axis=0)
    mins = np.min(r_after_burnin, axis=0)
    maxs = np.max(r_after_burnin, axis=0)

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Duration: {result.t_ms[-1]:.1f} ms | Burn-in: {burn_in_ms:.1f} ms | dt: {dt:.2f} ms")
    print("-" * 60)
    print(f"{'Population':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for i, name in enumerate(POPULATION_NAMES):
        print(f"{name:<10} {means[i]:>10.3f} {stds[i]:>10.3f} {mins[i]:>10.3f} {maxs[i]:>10.3f}")
    print("=" * 60 + "\n")

    return {
        "means": dict(zip(POPULATION_NAMES, means)),
        "stds": dict(zip(POPULATION_NAMES, stds)),
        "mins": dict(zip(POPULATION_NAMES, mins)),
        "maxs": dict(zip(POPULATION_NAMES, maxs)),
    }
