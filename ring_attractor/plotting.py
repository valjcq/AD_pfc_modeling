"""
Visualization utilities for the ring attractor network.

This module provides functions to plot simulation results:
- Activity heatmap (time x position)
- Polar snapshots of ring activity
- Bump tracking over time
- Combined dashboard view
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulation import RingSimulationResult


def _check_display_available() -> bool:
    """Check if a display is available for GUI plotting."""
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


# Population names and colors (colorblind-friendly palette)
POPULATION_NAMES = ["PYR", "SOM", "PV", "VIP"]
POPULATION_COLORS = {
    "PYR": "#E69F00",  # Orange - excitatory
    "SOM": "#56B4E9",  # Sky blue
    "PV": "#009E73",  # Bluish green
    "VIP": "#CC79A7",  # Reddish purple
}

# Condition colors for multi-condition comparison plots (Okabe-Ito palette)
CONDITION_COLORS: dict[str, str] = {
    "WT":         "#000000",  # Black
    "WT_APP":     "#E69F00",  # Orange
    "a7_KO":      "#56B4E9",  # Sky blue
    "a7_KO_APP":  "#009E73",  # Bluish green
    "b2_KO":      "#F0E442",  # Yellow
    "b2_KO_APP":  "#0072B2",  # Blue
    "a5_KO":      "#D55E00",  # Vermillion
    "a5_KO_APP":  "#CC79A7",  # Reddish purple
}


def plot_ring_activity_heatmap(
    result: "RingSimulationResult",
    population: int = 0,
    ax=None,
    title: Optional[str] = None,
    cmap: str = "hot",
    time_range: Optional[tuple[float, float]] = None,
    show_stimulus: bool = True,
    show_decoded: bool = True,
    t_offset: float = 0.0,
):
    """
    Plot activity as heatmap (time x position).

    X-axis: Angular position (degrees)
    Y-axis: Time (ms)
    Color: Firing rate

    Parameters:
        result: RingSimulationResult
        population: Which population (0=PYR, 1=SOM, 2=PV, 3=VIP)
        ax: Matplotlib axis (created if None)
        title: Plot title (default: population name)
        cmap: Colormap name
        time_range: Optional (start_ms, end_ms) to restrict time
        show_stimulus: Whether to mark stimulus window
        show_decoded: Whether to overlay decoded bump position

    Returns:
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    activity = result.r[:, :, population]
    t = result.t_ms
    angles = result.ring_params.node_angles_deg

    # Time range filtering
    if time_range:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        activity = activity[mask]
        t = t[mask]

    # Apply display offset
    t_display = t - t_offset

    # Create heatmap
    extent = [0, 360, t_display[-1], t_display[0]]
    im = ax.imshow(
        activity,
        aspect="auto",
        cmap=cmap,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Firing Rate (Hz)")

    # Mark stimulus
    if show_stimulus and result.stim_window[1] > result.stim_window[0]:
        ax.axhline(result.stim_window[0] - t_offset, color="white", linestyle="--", linewidth=1)
        ax.axhline(result.stim_window[1] - t_offset, color="white", linestyle="--", linewidth=1)
        ax.axvline(result.stim_angle_deg, color="white", linestyle=":", linewidth=1)

    # Overlay decoded position
    if show_decoded:
        from .analysis import decode_bump_center

        center_deg, amplitude = decode_bump_center(result, population)
        if time_range:
            mask_full = (result.t_ms >= time_range[0]) & (result.t_ms <= time_range[1])
            center_deg = center_deg[mask_full]
            t_plot = result.t_ms[mask_full] - t_offset
        else:
            t_plot = result.t_ms - t_offset

        # Only plot where amplitude is reasonable
        valid = amplitude > 0.2 if not time_range else amplitude[mask_full] > 0.2
        ax.scatter(
            center_deg[valid],
            t_plot[valid],
            c="cyan",
            s=1,
            alpha=0.5,
            label="Decoded",
        )

    pop_name = POPULATION_NAMES[population]
    ax.set_xlabel("Position (degrees)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title or f"{pop_name} Activity")
    ax.set_xlim(0, 360)

    return ax


def plot_ring_snapshot(
    result: "RingSimulationResult",
    t_ms: float,
    ax=None,
    polar: bool = True,
    show_all_populations: bool = False,
    t_offset: float = 0.0,
):
    """
    Plot activity pattern at a single time point.

    Parameters:
        result: RingSimulationResult
        t_ms: Time point to plot (ms)
        ax: Matplotlib axis (created if None)
        polar: Whether to use polar coordinates
        show_all_populations: Whether to show all 4 populations

    Returns:
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt

    # Find closest time index
    idx = np.argmin(np.abs(result.t_ms - t_ms))
    actual_t = result.t_ms[idx]

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar" if polar else None)

    if polar:
        angles = result.ring_params.node_angles_rad
        # Close the ring by appending first point
        angles_closed = np.append(angles, angles[0])
    else:
        angles = result.ring_params.node_angles_deg

    if show_all_populations:
        for i, (name, color) in enumerate(
            zip(POPULATION_NAMES, POPULATION_COLORS.values())
        ):
            r = result.r[idx, :, i]
            if polar:
                r_closed = np.append(r, r[0])
                ax.plot(angles_closed, r_closed, color=color, label=name, linewidth=2)
            else:
                ax.plot(angles, r, color=color, label=name, linewidth=2)
        ax.legend(loc="upper right")
    else:
        r_pyr = result.r[idx, :, 0]
        color = POPULATION_COLORS["PYR"]
        if polar:
            r_closed = np.append(r_pyr, r_pyr[0])
            ax.plot(angles_closed, r_closed, color=color, linewidth=2)
            ax.fill(angles_closed, r_closed, color=color, alpha=0.3)
        else:
            ax.plot(angles, r_pyr, color=color, linewidth=2)
            ax.fill_between(angles, 0, r_pyr, color=color, alpha=0.3)

    ax.set_title(f"t = {actual_t - t_offset:.1f} ms")

    if not polar:
        ax.set_xlabel("Position (degrees)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlim(0, 360)

    return ax


def plot_bump_tracking(
    result: "RingSimulationResult",
    population: int = 0,
    ax=None,
    show_cue: bool = True,
    t_offset: float = 0.0,
):
    """
    Plot decoded bump position over time.

    Parameters:
        result: RingSimulationResult
        population: Which population to decode (0=PYR)
        ax: Matplotlib axis (created if None)
        show_cue: Whether to mark stimulus location

    Returns:
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    from .analysis import decode_bump_center

    center_deg, amplitude = decode_bump_center(result, population)

    # Color by decoding confidence
    scatter = ax.scatter(
        result.t_ms - t_offset, center_deg, c=amplitude, cmap="viridis", s=1, alpha=0.5
    )
    plt.colorbar(scatter, ax=ax, label="Decoding Confidence")

    # Mark stimulus
    if show_cue and result.stim_window[1] > result.stim_window[0]:
        ax.axhline(
            result.stim_angle_deg,
            color="red",
            linestyle="--",
            label=f"Cue: {result.stim_angle_deg:.0f}°",
        )
        ax.axvspan(
            result.stim_window[0] - t_offset, result.stim_window[1] - t_offset, alpha=0.2, color="red"
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Decoded Position (degrees)")
    ax.set_ylim(0, 360)
    ax.set_title("Bump Position Over Time")

    return ax


def plot_node_activity(
    result: "RingSimulationResult",
    nodes: Optional[list[int]] = None,
    population: int = 0,
    ax=None,
    t_offset: float = 0.0,
):
    """
    Plot activity at specific nodes over time.

    Parameters:
        result: RingSimulationResult
        nodes: List of node indices to plot (default: stim node and opposite)
        population: Which population (0=PYR)
        ax: Matplotlib axis (created if None)

    Returns:
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if nodes is None:
        stim_node = result.stim_node
        opposite_node = (stim_node + result.n_nodes // 2) % result.n_nodes
        nodes = [stim_node, opposite_node]

    colors = plt.cm.tab10(np.linspace(0, 1, len(nodes)))

    for node, color in zip(nodes, colors):
        angle = result.ring_params.node_angles_deg[node]
        ax.plot(
            result.t_ms - t_offset,
            result.r[:, node, population],
            color=color,
            label=f"Node {node} ({angle:.0f}°)",
            linewidth=1,
        )

    # Mark stimulus window
    if result.stim_window[1] > result.stim_window[0]:
        ax.axvspan(
            result.stim_window[0] - t_offset, result.stim_window[1] - t_offset, alpha=0.2, color="gray"
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{POPULATION_NAMES[population]} Firing Rate (Hz)")
    ax.legend(loc="upper right")
    ax.set_title("Activity at Selected Nodes")

    return ax


def plot_bump_metrics_over_time(
    result: "RingSimulationResult",
    population: int = 0,
    ax=None,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
):
    """
    Plot decoded bump center, amplitude, and width over time.

    Parameters:
        result: RingSimulationResult
        population: Which population to decode (0=PYR)
        ax: Array of 3 axes (created if None)
        time_range: Optional (start_ms, end_ms) to restrict time

    Returns:
        axes: Array of 3 Matplotlib axes
    """
    import matplotlib.pyplot as plt
    from .analysis import decode_bump_center, estimate_bump_width

    center_deg, amplitude = decode_bump_center(result, population)
    t = result.t_ms

    # Time range filtering
    if time_range:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t = t[mask]
        center_deg = center_deg[mask]
        amplitude = amplitude[mask]
        activity = result.r[mask, :, population]
    else:
        mask = np.ones(len(t), dtype=bool)
        activity = result.r[:, :, population]

    # Compute width at sampled time points (expensive, so subsample)
    n_samples = min(200, len(t))
    sample_idx = np.linspace(0, len(t) - 1, n_samples, dtype=int)
    t_width = t[sample_idx]
    widths = np.array([
        estimate_bump_width(
            activity[i],
            result.ring_params.node_angles_rad,
            center_deg[sample_idx[j]] * np.pi / 180,
        )
        for j, i in enumerate(sample_idx)
    ])

    # Apply display offset
    t_display = t - t_offset
    t_width_display = t_width - t_offset

    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # --- Center position ---
    ax[0].scatter(t_display, center_deg, c=amplitude, cmap="viridis", s=1, alpha=0.5)
    if result.stim_angle_deg > 0:
        ax[0].axhline(result.stim_angle_deg, color="red", ls="--", lw=1,
                       label=f"Cue: {result.stim_angle_deg:.0f}°")
        ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_ylabel("Center (°)")
    ax[0].set_ylim(0, 360)
    ax[0].set_title("Bump Metrics Over Time")

    # --- Amplitude ---
    ax[1].plot(t_display, amplitude, color="#009E73", lw=1)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_ylim(0, max(1, amplitude.max() * 1.1))

    # --- Width ---
    ax[2].plot(t_width_display, widths, color="#CC79A7", lw=1)
    ax[2].set_ylabel("Width (°)")
    ax[2].set_xlabel("Time (ms)")

    # Mark stimulus window on all axes
    if result.stim_window[1] > result.stim_window[0]:
        for a in ax:
            a.axvspan(result.stim_window[0] - t_offset, result.stim_window[1] - t_offset,
                      alpha=0.15, color="red")

    return ax


def plot_ring_dashboard(
    result: "RingSimulationResult",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
):
    """
    Comprehensive visualization dashboard for ring attractor simulation.

    Parameters:
        result: RingSimulationResult
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        time_range: Optional (start_ms, end_ms) to restrict displayed time

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1])

    # Top row: Activity heatmap (spans 2 columns)
    ax_heat = fig.add_subplot(gs[0, :2])
    plot_ring_activity_heatmap(result, ax=ax_heat, time_range=time_range, t_offset=t_offset)

    # Top right: Snapshot at end of delay
    ax_snap = fig.add_subplot(gs[0, 2], projection="polar")
    t_snap = min(result.stim_window[1] + 500, result.t_ms[-1])
    plot_ring_snapshot(result, t_snap, ax=ax_snap, t_offset=t_offset)

    # Middle row: Bump tracking
    ax_track = fig.add_subplot(gs[1, :])
    plot_bump_tracking(result, ax=ax_track, t_offset=t_offset)
    if time_range:
        ax_track.set_xlim((time_range[0] - t_offset, time_range[1] - t_offset))

    # Bottom left: Activity at specific nodes
    ax_nodes = fig.add_subplot(gs[2, :2])
    plot_node_activity(result, ax=ax_nodes, t_offset=t_offset)
    if time_range:
        ax_nodes.set_xlim((time_range[0] - t_offset, time_range[1] - t_offset))

    # Bottom right: Metrics text
    ax_metrics = fig.add_subplot(gs[2, 2])
    ax_metrics.axis("off")

    from .analysis import compute_bump_metrics

    metrics = compute_bump_metrics(result)
    metrics_text = (
        f"Bump Metrics (delay period)\n"
        f"{'─' * 30}\n"
        f"Center: {metrics['center_mean_deg']:.1f}° ± {metrics['center_std_deg']:.1f}°\n"
        f"Width: {metrics['width_mean_deg']:.1f}°\n"
        f"Amplitude: {metrics['amplitude_mean']:.2f}\n"
        f"Drift: {metrics['drift_rate_deg_per_s']:.1f}°/s\n"
        f"Diffusion: {metrics['diffusion_deg2_per_s']:.1f}°²/s\n"
        f"Error from cue: {metrics['error_from_cue_deg']:.1f}°"
    )
    ax_metrics.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ring_connectome(
    ring_params,
    ax=None,
    n_highlight: int = 8,
    excit_color: str = "#D62728",
    inhib_color: str = "#1F77B4",
    weight_threshold: float = 0.05,
    save_path: Optional[str] = None,
):
    """
    Plot the ring network connectivity as a connectome diagram.

    Nodes are arranged in a circle. Excitatory (PYR→PYR) connections are
    drawn as solid lines and inhibitory (PV→PV) connections as dashed lines,
    with line width proportional to connection strength.

    Parameters:
        ring_params: RingParams configuration
        ax: Matplotlib axis (created if None)
        n_highlight: Number of evenly-spaced source nodes to show connections from
        excit_color: Color for excitatory connections
        inhib_color: Color for inhibitory connections
        weight_threshold: Fraction of peak weight below which connections are hidden
        save_path: If provided, save figure to this path

    Returns:
        ax: Matplotlib axis
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from .connectivity import build_pyr_pyr_weights, build_pv_pyr_weights

    n = ring_params.n_nodes
    angles = ring_params.node_angles_rad

    # Node positions on unit circle (0° at top, clockwise)
    x = np.sin(angles)
    y = np.cos(angles)

    # Build weight matrices
    W_exc = build_pyr_pyr_weights(ring_params)
    W_inh = build_pv_pyr_weights(ring_params)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Select source nodes evenly spaced around the ring
    sources = np.linspace(0, n, n_highlight, endpoint=False, dtype=int)

    max_exc = W_exc.max()
    max_inh = W_inh.max()
    lw_max = 3.0

    # --- Inhibitory connections (draw first, behind) ---
    # Show from 2 source nodes to a sparse subset of targets
    inh_sources = sources[::4]  # 2 source nodes
    tgt_step = max(1, n // 16)  # show ~16 target endpoints per source
    inh_segments = []
    inh_linewidths = []
    inh_alphas = []
    for src in inh_sources:
        for tgt in range(0, n, tgt_step):
            if tgt == src:
                continue
            w = W_inh[tgt, src]
            if w > 0:
                inh_segments.append([(x[src], y[src]), (x[tgt], y[tgt])])
                inh_linewidths.append(0.8)
                inh_alphas.append(0.3)

    if inh_segments:
        inh_lc = LineCollection(
            inh_segments,
            linewidths=inh_linewidths,
            colors=[(*plt.matplotlib.colors.to_rgb(inhib_color), a) for a in inh_alphas],
            linestyles="dashed",
            zorder=0,
        )
        ax.add_collection(inh_lc)

    # --- Excitatory connections (draw on top) ---
    exc_segments = []
    exc_linewidths = []
    exc_alphas = []
    for src in sources:
        for tgt in range(n):
            if tgt == src:
                continue
            w = W_exc[tgt, src]
            if w > weight_threshold * max_exc:
                exc_segments.append([(x[src], y[src]), (x[tgt], y[tgt])])
                exc_linewidths.append(lw_max * (w / max_exc))
                exc_alphas.append(0.25 + 0.55 * (w / max_exc))

    if exc_segments:
        exc_lc = LineCollection(
            exc_segments,
            linewidths=exc_linewidths,
            colors=[(*plt.matplotlib.colors.to_rgb(excit_color), a) for a in exc_alphas],
            zorder=1,
        )
        ax.add_collection(exc_lc)

    # --- Draw nodes ---
    ax.scatter(x, y, s=25, c="black", zorder=3)
    # Highlight source nodes
    ax.scatter(
        x[sources], y[sources],
        s=60, c=excit_color, edgecolors="black", linewidth=0.5, zorder=4,
    )

    # --- Degree labels around the ring ---
    label_nodes = np.linspace(0, n, 8, endpoint=False, dtype=int)
    for i in label_nodes:
        deg = ring_params.node_angles_deg[i]
        offset = 1.12
        ax.text(
            x[i] * offset, y[i] * offset,
            f"{deg:.0f}°",
            ha="center", va="center", fontsize=9, color="gray",
        )

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], color=excit_color, linewidth=2.5,
               label=f"PYR→PYR excitatory (σ={ring_params.sigma_pyr_deg:.0f}°)"),
        Line2D([0], [0], color=inhib_color, linewidth=1, linestyle="--",
               label=f"PV→PYR inhibitory ({ring_params.pv_global_type})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
              framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_title(f"Ring Connectome ({n} nodes)", fontsize=14, fontweight="bold")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_bump_metrics_comparison(
    results: dict[str, "RingSimulationResult"],
    population: int = 0,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Overlay bump metrics (center, amplitude, width) over time for multiple conditions.

    Creates a 3-panel figure with one line per condition.

    Parameters:
        results: dict mapping condition_key -> RingSimulationResult
        population: Which population to decode (0=PYR)
        time_range: (start_ms, end_ms) in absolute time for filtering
        t_offset: Subtracted from display time (e.g. burn_in_ms)
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from .analysis import decode_bump_center, estimate_bump_width

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for cond_key, result in results.items():
        color = condition_colors.get(cond_key, None)
        center_deg, amplitude = decode_bump_center(result, population)
        t = result.t_ms

        # Time range filtering
        if time_range:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            center_deg = center_deg[mask]
            amplitude = amplitude[mask]
            activity = result.r[mask, :, population]
        else:
            activity = result.r[:, :, population]

        t_display = t - t_offset

        # Subsample width
        n_samples = min(200, len(t))
        sample_idx = np.linspace(0, len(t) - 1, n_samples, dtype=int)
        t_w_display = t_display[sample_idx]
        widths = np.array([
            estimate_bump_width(
                activity[i],
                result.ring_params.node_angles_rad,
                center_deg[sample_idx[j]] * np.pi / 180,
            )
            for j, i in enumerate(sample_idx)
        ])

        # Use condition name from study if available
        from circuit_model.study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        axes[0].plot(t_display, center_deg, color=color, lw=1, alpha=0.7, label=label)
        axes[1].plot(t_display, amplitude, color=color, lw=1, alpha=0.7, label=label)
        axes[2].plot(t_w_display, widths, color=color, lw=1, alpha=0.7, label=label)

    # Mark stimulus window (from first result)
    first_result = next(iter(results.values()))
    if first_result.stim_window[1] > first_result.stim_window[0]:
        for ax in axes:
            ax.axvspan(
                first_result.stim_window[0] - t_offset,
                first_result.stim_window[1] - t_offset,
                alpha=0.15, color="red",
            )

    # Cue location
    if first_result.stim_angle_deg > 0:
        axes[0].axhline(first_result.stim_angle_deg, color="red", ls="--", lw=1)

    axes[0].set_ylabel("Center (°)")
    axes[0].set_ylim(0, 360)
    axes[0].set_title("Bump Metrics Comparison")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Amplitude")
    axes[2].set_ylabel("Width (°)")
    axes[2].set_xlabel("Time (ms)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# Mapping from metric dict keys to human-readable labels
_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "amplitude_mean": "Amplitude",
    "width_mean_deg": "Width (°)",
    "error_from_cue_deg": "Error from Cue (°)",
    "center_std_deg": "Center Std (°)",
    "diffusion_deg2_per_s": "Diffusion (°²/s)",
    "drift_rate_deg_per_s": "Drift (°/s)",
}


def plot_metrics_vs_delay(
    metrics_over_delay: dict[str, list[dict]],
    delay_labels: list[str],
    metrics_to_plot: tuple[str, ...] = ("amplitude_mean", "width_mean_deg", "error_from_cue_deg"),
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
):
    """
    Plot bump metrics at multiple delay timepoints, comparing conditions.

    Creates one subplot per metric with lines for each condition.

    Parameters:
        metrics_over_delay: dict mapping condition_key -> list of metric dicts
            (output of compute_metrics_at_delay_times)
        delay_labels: Human-readable labels for each timepoint (e.g. ["1s", "2s", "3s"])
        metrics_to_plot: Which metric keys to plot (one panel per metric)
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size
        save_path: If provided, save figure

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x_pos = np.arange(len(delay_labels))

    for cond_key, metric_list in metrics_over_delay.items():
        color = condition_colors.get(cond_key, None)

        from circuit_model.study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        for ax, metric_key in zip(axes, metrics_to_plot):
            values = [m[metric_key] for m in metric_list]
            ax.plot(x_pos, values, marker="o", color=color, label=label, lw=2, markersize=6)

    for ax, metric_key in zip(axes, metrics_to_plot):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(delay_labels)
        ax.set_xlabel("Delay time")
        ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bump Metrics During Delay Period", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def print_simulation_summary(result: "RingSimulationResult") -> None:
    """Print a text summary of the simulation results."""
    from .analysis import compute_bump_metrics, compute_working_memory_accuracy

    print("=" * 50)
    print("Ring Attractor Simulation Summary")
    print("=" * 50)
    print(f"Network: {result.n_nodes} nodes")
    print(f"Duration: {result.t_ms[-1]:.0f} ms")
    print(f"Stimulus: {result.stim_angle_deg:.0f}° ({result.stim_window[0]:.0f}-{result.stim_window[1]:.0f} ms)")
    print()

    # Bump metrics during delay
    metrics = compute_bump_metrics(result)
    print("Bump Metrics (delay period):")
    print(f"  Center: {metrics['center_mean_deg']:.1f}° ± {metrics['center_std_deg']:.1f}°")
    print(f"  Width: {metrics['width_mean_deg']:.1f}°")
    print(f"  Decoding amplitude: {metrics['amplitude_mean']:.2f}")
    print(f"  Drift rate: {metrics['drift_rate_deg_per_s']:.1f}°/s")
    print(f"  Diffusion: {metrics['diffusion_deg2_per_s']:.1f}°²/s")
    print()

    # Working memory accuracy
    accuracy = compute_working_memory_accuracy(result)
    print("Working Memory Performance:")
    print(f"  Cue position: {accuracy['cue_position_deg']:.0f}°")
    print(f"  Final position: {accuracy['final_position_deg']:.1f}°")
    print(f"  Error: {accuracy['error_deg']:.1f}°")
    print(f"  Bump maintained: {'Yes' if accuracy['maintained'] else 'No'}")
    print("=" * 50)
