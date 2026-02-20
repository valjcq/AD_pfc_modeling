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

from ..plotting import POPULATION_NAMES, POPULATION_COLORS

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


def _mark_transient(ax, result: "RingSimulationResult", t_offset: float = 0.0,
                    orientation: str = "vertical"):
    """Draw markers for the response transient window if enabled.

    Parameters:
        ax: Matplotlib axis (or list of axes)
        result: RingSimulationResult (uses result.local_params transient fields)
        t_offset: Display offset subtracted from absolute time
        orientation: 'vertical' for time-on-x-axis, 'horizontal' for heatmap (time-on-y)
    """
    p = result.local_params
    if not p.trans_enabled:
        return
    t_start = p.trans_start_ms - t_offset
    t_end = (p.trans_start_ms + p.trans_duration_ms) - t_offset

    axes = ax if hasattr(ax, '__len__') else [ax]
    for a in axes:
        if orientation == "vertical":
            a.axvspan(t_start, t_end, alpha=0.15, color="blue", zorder=0)
            a.axvline(t_start, color="blue", ls="--", lw=0.8, alpha=0.6)
            a.axvline(t_end, color="blue", ls="--", lw=0.8, alpha=0.6)
        else:  # horizontal (heatmap with time on y-axis)
            a.axhline(t_start, color="cyan", ls="--", lw=1, alpha=0.7)
            a.axhline(t_end, color="cyan", ls="--", lw=1, alpha=0.7)


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

    # Mark response transient
    _mark_transient(ax, result, t_offset=t_offset, orientation="horizontal")

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
            label=f"Cue: {result.stim_angle_deg:.0f}",
        )
        ax.axvspan(
            result.stim_window[0] - t_offset, result.stim_window[1] - t_offset, alpha=0.2, color="red"
        )
        ax.legend(loc="upper right")

    # Mark response transient
    _mark_transient(ax, result, t_offset=t_offset)

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
            label=f"Node {node} ({angle:.0f})",
            linewidth=1,
        )

    # Mark stimulus window
    if result.stim_window[1] > result.stim_window[0]:
        ax.axvspan(
            result.stim_window[0] - t_offset, result.stim_window[1] - t_offset, alpha=0.2, color="gray"
        )

    # Mark response transient
    _mark_transient(ax, result, t_offset=t_offset)

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
                       label=f"Cue: {result.stim_angle_deg:.0f}")
        ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_ylabel("Center (deg)")
    ax[0].set_ylim(0, 360)
    ax[0].set_title("Bump Metrics Over Time")

    # --- Amplitude ---
    ax[1].plot(t_display, amplitude, color="#009E73", lw=1)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_ylim(0, max(1, amplitude.max() * 1.1))

    # --- Width ---
    ax[2].plot(t_width_display, widths, color="#CC79A7", lw=1)
    ax[2].set_ylabel("Width (deg)")
    ax[2].set_xlabel("Time (ms)")

    # Mark stimulus window on all axes
    if result.stim_window[1] > result.stim_window[0]:
        for a in ax:
            a.axvspan(result.stim_window[0] - t_offset, result.stim_window[1] - t_offset,
                      alpha=0.15, color="red")

    # Mark response transient on all axes
    _mark_transient(ax, result, t_offset=t_offset)

    return ax


def plot_ring_dashboard(
    result: "RingSimulationResult",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
    suptitle: Optional[str] = None,
):
    """
    Comprehensive visualization dashboard for ring attractor simulation.

    Parameters:
        result: RingSimulationResult
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        time_range: Optional (start_ms, end_ms) to restrict displayed time
        suptitle: Optional figure super-title (e.g. stimulus amplitude info)

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1])

    # Top row: Activity heatmap (spans 2 columns)
    ax_heat = fig.add_subplot(gs[0, :2])
    plot_ring_activity_heatmap(result, ax=ax_heat, time_range=time_range, t_offset=t_offset, show_decoded=False)

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
        f"Center: {metrics['center_mean_deg']:.1f} +/- {metrics['center_std_deg']:.1f} deg\n"
        f"Width: {metrics['width_mean_deg']:.1f} deg\n"
        f"Amplitude: {metrics['amplitude_mean']:.2f}\n"
        f"Drift: {metrics['drift_rate_deg_per_s']:.1f} deg/s\n"
        f"Diffusion: {metrics['diffusion_deg2_per_s']:.1f} deg^2/s\n"
        f"Error from cue: {metrics['error_from_cue_deg']:.1f} deg"
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

    Nodes are arranged in a circle. Excitatory (PYR->PYR) connections are
    drawn as solid lines and inhibitory (PV->PV) connections as dashed lines,
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

    # Node positions on unit circle (0 at top, clockwise)
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
            f"{deg:.0f} deg",
            ha="center", va="center", fontsize=9, color="gray",
        )

    # --- Legend ---
    if ring_params.pyr_profile_type == "compte":
        exc_label = (f"PYR->PYR Compte (J+={ring_params.J_plus:.2f}, "
                     f"sigma={ring_params.sigma_pyr_deg:.0f} deg)")
    else:
        exc_label = f"PYR->PYR excitatory (sigma={ring_params.sigma_pyr_deg:.0f} deg)"
    legend_elements = [
        Line2D([0], [0], color=excit_color, linewidth=2.5, label=exc_label),
        Line2D([0], [0], color=inhib_color, linewidth=1, linestyle="--",
               label=f"PV->PYR inhibitory ({ring_params.pv_global_type})"),
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


def extract_comparison_data(
    result: "RingSimulationResult",
    population: int = 0,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
) -> dict:
    """Extract lightweight comparison data from a full simulation result.

    This avoids keeping the large r array (~500 MB) in memory.
    Call this before deleting the result.

    Parameters:
        result: Full RingSimulationResult
        population: Which population to decode (0=PYR)
        time_range: (start_ms, end_ms) in absolute time for filtering
        t_offset: Subtracted from display time

    Returns:
        dict with keys: t_display, center_deg, amplitude, t_w_display, widths,
                        stim_window, stim_angle_deg, local_params
    """
    from .analysis import decode_bump_center, estimate_bump_width

    center_deg, amplitude = decode_bump_center(result, population)
    t = result.t_ms

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

    return {
        "t_display": t_display,
        "center_deg": center_deg,
        "amplitude": amplitude,
        "t_w_display": t_w_display,
        "widths": widths,
        "stim_window": result.stim_window,
        "stim_angle_deg": result.stim_angle_deg,
        "local_params": result.local_params,
        "t_offset": t_offset,
    }


def plot_bump_metrics_comparison(
    comparison_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """
    Overlay bump metrics (center, amplitude, width) over time for multiple conditions.

    Creates a 3-panel figure with one line per condition.

    Parameters:
        comparison_data: dict mapping condition_key -> lightweight dict from
            extract_comparison_data()
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size
        save_path: If provided, save figure
        suptitle: Optional super-title

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for cond_key, data in comparison_data.items():
        color = condition_colors.get(cond_key, None)

        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        axes[0].plot(data["t_display"], data["center_deg"], color=color, lw=1, alpha=0.7, label=label)
        axes[1].plot(data["t_display"], data["amplitude"], color=color, lw=1, alpha=0.7, label=label)
        axes[2].plot(data["t_w_display"], data["widths"], color=color, lw=1, alpha=0.7, label=label)

    # Mark stimulus window (from first entry)
    first = next(iter(comparison_data.values()))
    stim_w = first["stim_window"]

    t_offset = first.get("t_offset", 0.0)
    if stim_w[1] > stim_w[0]:
        for ax in axes:
            ax.axvspan(stim_w[0] - t_offset, stim_w[1] - t_offset,
                       alpha=0.15, color="red")

    if first["stim_angle_deg"] > 0:
        axes[0].axhline(first["stim_angle_deg"], color="red", ls="--", lw=1)

    # Mark transient window if enabled
    p = first["local_params"]
    if p.trans_enabled:
        t_start = p.trans_start_ms - t_offset
        t_end = (p.trans_start_ms + p.trans_duration_ms) - t_offset
        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.15, color="blue", zorder=0)
            ax.axvline(t_start, color="blue", ls="--", lw=0.8, alpha=0.6)
            ax.axvline(t_end, color="blue", ls="--", lw=0.8, alpha=0.6)

    axes[0].set_ylabel("Center (deg)")
    axes[0].set_ylim(0, 360)
    axes[0].set_title("Bump Metrics Comparison")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Amplitude")
    axes[2].set_ylabel("Width (deg)")
    axes[2].set_xlabel("Time (ms)")

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# Mapping from metric dict keys to human-readable labels
_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "amplitude_mean": "Amplitude",
    "width_mean_deg": "Width (deg)",
    "error_from_cue_deg": "Error from Cue (deg)",
    "center_std_deg": "Center Std (deg)",
    "diffusion_deg2_per_s": "Diffusion (deg^2/s)",
    "drift_rate_deg_per_s": "Drift (deg/s)",
}


def plot_metrics_vs_delay(
    metrics_over_delay: dict[str, list[dict]],
    delay_labels: list[str],
    metrics_to_plot: tuple[str, ...] = ("amplitude_mean", "width_mean_deg", "error_from_cue_deg"),
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    error_band: str = "sem",
):
    """
    Plot bump metrics at multiple delay timepoints, comparing conditions.

    Parameters:
        metrics_over_delay: dict mapping condition_key -> list of metric dicts
        delay_labels: Human-readable labels for each timepoint
        error_band: ``"sem"`` (default) or ``"sd"`` — controls the shaded band.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    band_suffix = "_sem" if error_band == "sem" else "_sd"

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Parse delay labels to numeric seconds for the x-axis
    x_seconds = []
    for lbl in delay_labels:
        try:
            x_seconds.append(float(lbl.rstrip("s")))
        except ValueError:
            x_seconds.append(float("nan"))
    x_seconds = np.array(x_seconds)

    for cond_key, metric_list in metrics_over_delay.items():
        color = condition_colors.get(cond_key, None)

        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        n_pts = min(len(x_seconds), len(metric_list))
        x = x_seconds[:n_pts]

        for ax, metric_key in zip(axes, metrics_to_plot):
            # Detect aggregated format (mean/sd/sem from multi-trial)
            if f"{metric_key}_mean" in metric_list[0]:
                means = np.array([m[f"{metric_key}_mean"] for m in metric_list[:n_pts]])
                errs = np.array([m.get(f"{metric_key}{band_suffix}",
                                       m.get(f"{metric_key}_sd", 0.0))
                                 for m in metric_list[:n_pts]])
                ax.plot(x, means, marker="o", color=color, label=label, lw=2, markersize=4)
                if np.any(errs > 0):
                    ax.fill_between(x, means - errs, means + errs,
                                    color=color, alpha=0.2)
            else:
                values = [m[metric_key] for m in metric_list[:n_pts]]
                ax.plot(x, values, marker="o", color=color, label=label, lw=2, markersize=4)

    for ax, metric_key in zip(axes, metrics_to_plot):
        ax.set_xlabel("Delay time (s)")
        ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Use sensible tick spacing: ~5-8 ticks max
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))

    plt.suptitle(suptitle or "Bump Metrics During Delay Period", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_vs_amplitude(
    all_delay_metrics: dict[float, dict[str, dict]],
    amplitude_values: list[float],
    metrics_to_plot: tuple[str, ...] = ("amplitude_mean", "width_mean_deg", "error_from_cue_deg"),
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    error_band: str = "sem",
):
    """
    Plot bump metrics as a function of stimulus amplitude, comparing conditions.

    Creates one subplot per metric with lines for each condition.

    Parameters:
        all_delay_metrics: dict mapping amplitude -> {condition_key -> metric_dict}.
            Each metric_dict is the output of compute_bump_metrics at delay end.
        amplitude_values: Stimulus amplitude values (x-axis), in order.
        metrics_to_plot: Which metric keys to plot (one panel per metric).
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    band_suffix = "_sem" if error_band == "sem" else "_sd"

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Collect all condition keys across amplitudes
    cond_keys = []
    for amp in amplitude_values:
        for k in all_delay_metrics.get(amp, {}):
            if k not in cond_keys:
                cond_keys.append(k)

    for cond_key in cond_keys:
        color = condition_colors.get(cond_key, None)

        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        for ax, metric_key in zip(axes, metrics_to_plot):
            means = []
            errs = []
            for amp in amplitude_values:
                m = all_delay_metrics.get(amp, {}).get(cond_key, {})
                # Detect aggregated format
                if f"{metric_key}_mean" in m:
                    means.append(m.get(f"{metric_key}_mean", float("nan")))
                    errs.append(m.get(f"{metric_key}{band_suffix}",
                                      m.get(f"{metric_key}_sd", 0.0)))
                else:
                    means.append(m.get(metric_key, float("nan")))
                    errs.append(0.0)
            means_arr = np.array(means)
            errs_arr = np.array(errs)
            ax.plot(amplitude_values, means_arr, marker="o", color=color,
                    label=label, lw=2, markersize=6)
            if np.any(errs_arr > 0):
                ax.fill_between(amplitude_values,
                                means_arr - errs_arr, means_arr + errs_arr,
                                color=color, alpha=0.2)

    for ax, metric_key in zip(axes, metrics_to_plot):
        ax.set_xlabel("Stimulus Amplitude (× I_ext_pyr)")
        ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Bump Metrics vs Stimulus Amplitude",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_msd_curves(
    msd_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    error_band: str = "sem",
):
    """Plot MSD vs time and B_hat bar chart across conditions.

    Parameters:
        msd_data: Dict mapping condition_key -> dict with keys:
            'lag_times' (s), 'msd_mean' (deg²), 'msd_sem' (deg²),
            'msd_sd' (deg²), 'fit_line' (deg²), 'B_hat' (deg²/s),
            'r_squared'.
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.
        suptitle: Optional super-title.
        error_band: ``"sem"`` (default) or ``"sd"`` — controls the shaded band.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    has_amp = any('amp_t_s' in data for data in msd_data.values())
    if has_amp:
        fig, (ax_msd, ax_bar, ax_amp) = plt.subplots(
            1, 3,
            figsize=(figsize[0] * 1.5, figsize[1]),
            gridspec_kw={"width_ratios": [2, 1, 2]},
        )
    else:
        fig, (ax_msd, ax_bar) = plt.subplots(1, 2, figsize=figsize,
                                              gridspec_kw={"width_ratios": [2, 1]})
        ax_amp = None

    # Left panel: MSD vs time
    for cond_key, data in msd_data.items():
        color = condition_colors.get(cond_key, None)
        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        lag = data["lag_times"]
        msd = data["msd_mean"]
        band_key = "msd_sd" if error_band == "sd" else "msd_sem"
        err = data.get(band_key, data["msd_sem"])

        ax_msd.plot(lag, msd, color=color, lw=2, label=label)
        if np.any(err > 0):
            ax_msd.fill_between(lag, msd - err, msd + err, color=color, alpha=0.2)
        # Overlay linear fit (dashed)
        fit = data["fit_line"]
        valid = ~np.isnan(fit)
        ax_msd.plot(lag[valid], fit[valid], color=color, ls="--", lw=1, alpha=0.7)

    ax_msd.set_xlabel("Lag (s)")
    ax_msd.set_ylabel("MSD (deg²)")
    ax_msd.set_title("Mean Squared Displacement")
    ax_msd.legend(fontsize=8)
    ax_msd.grid(True, alpha=0.3)

    # Right panel: B_hat bar chart
    cond_keys = list(msd_data.keys())
    B_values = [msd_data[k]["B_hat"] for k in cond_keys]
    colors = [condition_colors.get(k, "#666666") for k in cond_keys]
    labels = []
    for k in cond_keys:
        from ..study import STUDY_CONDITIONS
        labels.append(STUDY_CONDITIONS[k].name if k in STUDY_CONDITIONS else k)

    x = np.arange(len(cond_keys))
    bar_vals = [v if not np.isnan(v) else 0.0 for v in B_values]
    ax_bar.bar(x, bar_vals, color=colors, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, B_values):
        if np.isnan(v):
            ax_bar.text(xi, 0, "N/A\n(all melted)", ha="center", va="bottom",
                        fontsize=7, color="gray")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("$\\hat{B}$ (deg²/s)")
    ax_bar.set_title("Diffusion Strength")
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Right panel: amplitude evolution (only when amplitude data is available)
    if ax_amp is not None:
        noise_threshold_drawn = False
        for cond_key, data in msd_data.items():
            if 'amp_t_s' not in data:
                continue
            color = condition_colors.get(cond_key, None)
            from ..study import STUDY_CONDITIONS
            label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

            t = data['amp_t_s']
            amp = data['amp_mean']
            amp_err = data.get('amp_sem', np.zeros_like(amp))

            ax_amp.plot(t, amp, color=color, lw=2, label=label)
            if np.any(amp_err > 0):
                ax_amp.fill_between(t, amp - amp_err, amp + amp_err,
                                    color=color, alpha=0.2)

            nt = data.get('noise_threshold')
            if nt is not None and not noise_threshold_drawn:
                ax_amp.axhline(nt, color='black', ls='--', lw=1.2,
                               label='Noise floor')
                noise_threshold_drawn = True
            elif nt is not None:
                ax_amp.axhline(nt, color='black', ls='--', lw=1.2)

        ax_amp.set_xlabel("Time in delay (s)")
        ax_amp.set_ylabel("Bump amplitude (pop. vector length)")
        ax_amp.set_title("Bump Amplitude Over Time")
        ax_amp.set_ylim(bottom=0)
        ax_amp.legend(fontsize=8)
        ax_amp.grid(True, alpha=0.3)

    # Shade the oscillation-dominated region on the MSD panel
    max_osc_period = 0.0
    for data in msd_data.values():
        osc = data.get('osc_spectrum', {})
        p = osc.get('dominant_period_s')
        if p is not None and p > max_osc_period:
            max_osc_period = p
    if max_osc_period > 0:
        ax_msd.axvspan(0, max_osc_period, color='gray', alpha=0.12,
                       label=f'Osc. regime (<{max_osc_period * 1000:.0f} ms)')
        ax_msd.legend(fontsize=8)

    plt.suptitle(suptitle or "Diffusion Analysis (MSD)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_oscillation_spectrum(
    osc_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Plot bump amplitude oscillation power spectrum across conditions.

    Two-panel figure:
    - Left: PSD (power spectral density) vs frequency (Hz), per condition.
      A vertical dashed line marks the detected dominant frequency for each
      condition.
    - Right: Bar chart of detected oscillation period (ms) per condition.
      Conditions with no detected oscillation show an empty bar.

    Parameters:
        osc_data: Dict mapping condition_key -> result of
            ``compute_oscillation_spectrum``.
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, (ax_psd, ax_bar) = plt.subplots(1, 2, figsize=figsize,
                                          gridspec_kw={"width_ratios": [2, 1]})

    periods_ms: list[float] = []
    labels: list[str] = []
    colors: list[str] = []

    # Compute a global power floor = 1% of the peak power across all conditions,
    # so that near-zero FFT bins (e.g. 10^-27) don't compress the log-y scale.
    all_peaks = []
    for data in osc_data.values():
        p = data['power_mean']
        mask0 = data['freqs'] <= 30.0
        if np.any(mask0) and np.any(p[mask0] > 0):
            all_peaks.append(float(np.nanmax(p[mask0])))
    global_floor = (min(all_peaks) * 1e-2) if all_peaks else 1e-10

    for cond_key, data in osc_data.items():
        color = condition_colors.get(cond_key, "#666666")
        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        freqs = data['freqs']
        power = data['power_mean']
        power_err = data.get('power_sem', np.zeros_like(power))
        dominant_freq = data.get('dominant_freq_hz')

        # Only plot up to 30 Hz (physiologically relevant range); clip to floor
        mask = freqs <= 30.0
        p_clipped = np.clip(power[mask], global_floor, None)
        ax_psd.semilogy(freqs[mask], p_clipped, color=color, lw=1.5, label=label)
        if np.any(power_err[mask] > 0):
            lo = np.clip(power[mask] - power_err[mask], global_floor, None)
            hi = np.maximum(power[mask] + power_err[mask], global_floor)
            ax_psd.fill_between(freqs[mask], lo, hi, color=color, alpha=0.15)

        if dominant_freq is not None:
            ax_psd.axvline(dominant_freq, color=color, ls='--', lw=1.2, alpha=0.8)
            periods_ms.append(1000.0 / dominant_freq)
        else:
            periods_ms.append(0.0)

        labels.append(label)
        colors.append(color)

    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (a.u., log scale)")
    ax_psd.set_title("Amplitude Power Spectrum")
    ax_psd.legend(fontsize=8)
    ax_psd.grid(True, alpha=0.3, which='both')
    ax_psd.set_xlim(0, 30)

    # Bar chart: dominant period in ms
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, periods_ms, color=colors, edgecolor="black", linewidth=0.5)
    for xi, p in zip(x, periods_ms):
        if p == 0.0:
            ax_bar.text(xi, 0.5, "n.d.", ha="center", va="bottom",
                        fontsize=8, color="gray")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Dominant period (ms)")
    ax_bar.set_title("Oscillation Period")
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.suptitle(suptitle or "Bump Amplitude Oscillation Spectrum",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_displacement_distribution(
    disp_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "plt.Figure":
    """Plot distribution of final bump displacement from cue position.

    Two-panel figure:
    - Left: Violin plot (or box+strip if few trials) of per-trial displacements
      per condition.  Shows both the spread and the individual trials.
    - Right: Bar chart of mean |displacement| per condition with ±1 SD error bar.

    Parameters
    ----------
    disp_data : dict
        Mapping condition_key → dict with keys ``displacements_deg``,
        ``mean_deg``, ``std_deg``, ``abs_mean_deg``, ``n_valid``, ``n_total``.
    condition_colors : optional color map
    figsize : figure size
    save_path : if given, save figure here
    suptitle : figure super-title

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, ax_viol = plt.subplots(1, 1, figsize=figsize)

    cond_keys = list(disp_data.keys())
    from ..study import STUDY_CONDITIONS
    labels = [STUDY_CONDITIONS[ck].name if ck in STUDY_CONDITIONS else ck
              for ck in cond_keys]
    colors = [condition_colors.get(ck, "#666666") for ck in cond_keys]
    x = np.arange(len(cond_keys))

    # --- Violin / strip plot ---
    for xi, (ck, color) in enumerate(zip(cond_keys, colors)):
        disps = disp_data[ck].get('displacements_deg', np.array([]))
        n_valid = int(disp_data[ck].get('n_valid', len(disps)))
        n_total = int(disp_data[ck].get('n_total', len(disps)))
        if len(disps) == 0:
            continue
        # Violin
        vp = ax_viol.violinplot([disps], positions=[xi], widths=0.6,
                                showmeans=False, showmedians=True,
                                showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        vp['cmedians'].set_color(color)
        vp['cmedians'].set_linewidth(2)
        # Individual points (jitter)
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(disps))
        ax_viol.scatter(xi + jitter, disps, color=color, s=6, alpha=0.35,
                        zorder=3)
        # Mean marker
        mean_val = float(np.mean(disps))
        ax_viol.scatter([xi], [mean_val], color=color, s=60,
                        marker='D', zorder=5, edgecolors='black', linewidths=0.5)
        # n annotation
        ax_viol.text(xi, ax_viol.get_ylim()[0] if xi == 0 else 0,
                     f"n={n_valid}/{n_total}", ha='center', va='top',
                     fontsize=7, color='gray')

    ax_viol.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax_viol.set_xticks(x)
    ax_viol.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_viol.set_ylabel("Minimum displacement from cue (°)")
    ax_viol.set_title("Displacement distribution (◆ = mean)")
    ax_viol.grid(True, alpha=0.25, axis='y')

    plt.suptitle(suptitle or "Final Bump Displacement from Cue",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_diffusion_ring_snapshot(
    disp_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "plt.Figure":
    """Activity heatmap of the most extreme drift trial per condition.

    For each condition with a full ``extreme_result`` (RingSimulationResult),
    two stacked panels are shown:

    * **Top (large)** — activity heatmap: angle on x-axis, time on y-axis,
      PYR firing rate as colour, decoded bump centre overlaid as cyan dots.
      Matches the dashboard top-left panel from ``ring-run``.
    * **Bottom (small)** — decoded amplitude over the delay period, with the
      noise threshold as a horizontal dashed line when available.

    Falls back to a "no data" placeholder when ``extreme_result`` is absent
    (e.g. when loading results from cache).

    Parameters
    ----------
    disp_data : dict
        Mapping condition_key → dict produced by ``cmd_diffusion``.
        Expected keys: ``extreme_result`` (RingSimulationResult or None),
        ``extreme_displacement_deg`` (float), ``delay_start_ms``,
        ``delay_end_ms``, ``noise_threshold`` (float or None).
    condition_colors : optional color map
    figsize : figure size; auto-computed if None
    save_path : if given, save figure here
    suptitle : figure super-title

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from .analysis import decode_bump_center
    from ..study import STUDY_CONDITIONS

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    valid_conds = [
        ck for ck, d in disp_data.items()
        if d.get('extreme_result') is not None
    ]
    n_cond = len(valid_conds)

    if n_cond == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5,
                "No heatmap data available\n(run without --load_cache to regenerate)",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    if figsize is None:
        figsize = (7 * n_cond, 9)

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(1, n_cond, figure=fig, wspace=0.35)

    for col, ck in enumerate(valid_conds):
        d = disp_data[ck]
        result = d['extreme_result']
        delay_start = float(d.get('delay_start_ms', result.t_ms[0]))
        delay_end = float(d.get('delay_end_ms', result.t_ms[-1]))
        disp_deg = float(d.get('extreme_displacement_deg', 0.0))
        noise_thr = d.get('noise_threshold', None)
        label = STUDY_CONDITIONS[ck].name if ck in STUDY_CONDITIONS else ck

        # Two rows: heatmap (75%) + amplitude (25%)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1], hspace=0.08,
        )
        ax_heat = fig.add_subplot(inner[0])
        # Amplitude panel has its own independent x-axis (time, not angle)
        ax_amp = fig.add_subplot(inner[1])

        # --- Heatmap panel ---
        plot_ring_activity_heatmap(
            result,
            population=0,
            ax=ax_heat,
            title=f"{label} — most extreme trial ({disp_deg:+.1f}°)",
            cmap="hot",
            time_range=(delay_start, delay_end),
            show_stimulus=False,
            show_decoded=True,
            t_offset=delay_start,
        )
        ax_heat.set_ylabel("Time from delay onset (ms)")
        # Mark cue position with a vertical line
        ax_heat.axvline(result.stim_angle_deg, color="white", ls="--", lw=1.2,
                        alpha=0.8, label=f"Cue ({result.stim_angle_deg:.0f}°)")
        ax_heat.legend(fontsize=7, loc="upper right", framealpha=0.4)

        # --- Amplitude panel ---
        _, amplitude = decode_bump_center(result, population=0)
        t_ms = result.t_ms
        mask = (t_ms >= delay_start) & (t_ms <= delay_end)
        t_plot = t_ms[mask] - delay_start
        amp_delay = amplitude[mask]

        ax_amp.plot(t_plot, amp_delay, color=condition_colors.get(ck, "#444444"), lw=1.5)
        if noise_thr is not None:
            ax_amp.axhline(noise_thr, color='red', ls='--', lw=1.0,
                           label=f'Noise thr. ({noise_thr:.3f})')
            ax_amp.legend(fontsize=7, loc='upper right')
        ax_amp.set_xlabel("Time from delay onset (ms)")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.set_xlim(0, delay_end - delay_start)
        ax_amp.grid(True, alpha=0.2)

    plt.suptitle(
        suptitle or "Most Extreme Drift Trial — Activity Heatmap",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_extreme_drift_trials(
    extreme_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "plt.Figure":
    """Plot the most prominent drift trial per condition as a sanity check.

    Each subplot shows the bump center position (in degrees, relative to start)
    versus time for the trial that ended farthest from its starting position.
    The raw (pre-low-pass-filter) trajectory is shown so oscillatory motion is
    visible alongside any genuine drift.

    Parameters
    ----------
    extreme_data : dict
        Mapping condition_key → dict with keys ``t_s``, ``center_deg``,
        ``displacement_deg``.  Built by ``cmd_diffusion``.
    condition_colors : optional color map
    figsize : figure size; auto-computed from n_conditions if not provided
    save_path : if given, save figure here
    suptitle : figure super-title

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    n_cond = len(extreme_data)
    if figsize is None:
        figsize = (max(5 * n_cond, 8), 4)

    fig, axes = plt.subplots(1, n_cond, figsize=figsize, sharey=False)
    if n_cond == 1:
        axes = [axes]

    for ax, (cond_key, data) in zip(axes, extreme_data.items()):
        color = condition_colors.get(cond_key, "#666666")
        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        t_s = data.get('t_s')
        center_deg = data.get('center_deg')
        disp = data.get('displacement_deg', 0.0)

        if t_s is not None and center_deg is not None and len(center_deg) >= 2:
            t_ms = np.asarray(t_s) * 1000.0
            shifted = np.asarray(center_deg) - center_deg[0]
            ax.plot(t_ms, shifted, color=color, lw=1.2)
            ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4,
                       label='Start position')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Position shift (°)")
            ax.set_title(f"{label}\nMax final drift: {disp:.1f}°")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(label)

    plt.suptitle(suptitle or "Most Prominent Drift Trial per Condition",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_field(
    drift_data: dict[str, dict],
    condition_colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    error_band: str = "sem",
):
    """Plot distractor-induced drift field A_hat(Δφ) across conditions.

    Parameters:
        drift_data: Dict mapping condition_key -> dict with keys:
            'offsets_deg', 'A_hat' (rad/s), 'A_hat_sem' (rad/s),
            'A_hat_sd' (rad/s).
        condition_colors: Optional color mapping. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.
        suptitle: Optional super-title.
        error_band: ``"sem"`` (default) or ``"sd"`` — controls the shaded band.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    for cond_key, data in drift_data.items():
        color = condition_colors.get(cond_key, None)
        from ..study import STUDY_CONDITIONS
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        offsets = data["offsets_deg"]
        A = data["A_hat"]
        band_key = "A_hat_sd" if error_band == "sd" else "A_hat_sem"
        err = data.get(band_key, data["A_hat_sem"])

        ax.plot(offsets, A, color=color, lw=2, marker="o", markersize=4, label=label)
        if np.any(err > 0):
            ax.fill_between(offsets, A - err, A + err, color=color, alpha=0.2)

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Distractor Offset Δφ (deg)")
    ax.set_ylabel("$\\hat{A}(\\Delta\\varphi)$ (rad/s)")
    ax.set_title("Distractor Drift Field")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Drift Field Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_noise_floor_histogram(
    baseline_data: dict[float, np.ndarray],
    thresholds: dict[float, float],
    figsize: tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Plot histogram of Â_hat from no-stimulus baseline trials.

    Parameters:
        baseline_data: Dict mapping w_inter -> array of Â_hat values.
        thresholds: Dict mapping w_inter -> noise floor threshold.
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    w_values = sorted(baseline_data.keys())
    n = len(w_values)
    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, figsize[0]), figsize[1]),
                              squeeze=False)
    axes = axes[0]

    for ax, w in zip(axes, w_values):
        vals = baseline_data[w]
        thresh = thresholds[w]
        ax.hist(vals.ravel(), bins=40, color="#56B4E9", edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(thresh, color="red", ls="--", lw=2,
                   label=f"threshold = {thresh:.3f}")
        ax.set_xlabel("$\\hat{A}$ (pop. vector amplitude)")
        ax.set_ylabel("Count")
        ax.set_title(f"w_inter = {w:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Noise Floor: Â_hat Distribution (No Stimulus)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration_heatmap(
    grid_data: dict,
    metric: str,
    amplitude_values: list[float],
    w_inter_values: list[float],
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot a 2D heatmap of a calibration metric.

    Parameters:
        grid_data: Dict mapping (amplitude, w_inter) -> dict of aggregated metrics.
        metric: Key to extract from each grid point dict (e.g. 'success_rate').
        amplitude_values: List of amplitude factors (x-axis).
        w_inter_values: List of w_pyr_pyr_inter values (y-axis).
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_amp = len(amplitude_values)
    n_w = len(w_inter_values)
    mat = np.full((n_w, n_amp), np.nan)

    for i, w in enumerate(w_inter_values):
        for j, amp in enumerate(amplitude_values):
            d = grid_data.get((amp, w), {})
            mat[i, j] = d.get(metric, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    # Labels
    ax.set_xticks(range(n_amp))
    ax.set_xticklabels([f"{a:.0f}" for a in amplitude_values])
    ax.set_yticks(range(n_w))
    ax.set_yticklabels([f"{w:.2f}" for w in w_inter_values])
    ax.set_xlabel("Stimulus Amplitude (× I_ext_pyr)")
    ax.set_ylabel("w_pyr_pyr_inter")

    # Annotate cells
    for i in range(n_w):
        for j in range(n_amp):
            val = mat[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}" if val < 100 else f"{val:.0f}"
                text_color = "white" if val < (np.nanmax(mat) + np.nanmin(mat)) / 2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=text_color)

    metric_labels = {
        "success_rate": "Success Rate",
        "mean_A_hat": "Mean $\\hat{A}$",
        "peak_pyr_rate": "Peak PYR Rate (Hz)",
    }
    ax.set_title(metric_labels.get(metric, metric))

    plt.suptitle(suptitle or "Parameter Calibration", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration_timecourses(
    timecourse_data: dict[tuple[float, float], dict],
    eval_times_s: np.ndarray,
    figsize: tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    error_band: str = "sem",
):
    """Plot Â_hat time courses for selected grid points.

    Parameters:
        timecourse_data: Dict mapping (amplitude, w_inter) -> dict with keys
            'A_hat_mean' (array), 'A_hat_sem' (array), 'A_hat_sd' (array).
        eval_times_s: Time points in seconds.
        save_path: If provided, save figure.
        suptitle: Optional super-title.
        error_band: 'sem' or 'sd'.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis
    keys = sorted(timecourse_data.keys())
    n = len(keys)

    for idx, (amp, w) in enumerate(keys):
        d = timecourse_data[(amp, w)]
        color = cmap(idx / max(n - 1, 1))
        label = f"amp={amp:.0f}, w={w:.2f}"
        means = d["A_hat_mean"]
        err_key = "A_hat_sd" if error_band == "sd" else "A_hat_sem"
        errs = d.get(err_key, d.get("A_hat_sem", np.zeros_like(means)))

        ax.plot(eval_times_s, means, color=color, lw=2, marker="o",
                markersize=3, label=label)
        if np.any(errs > 0):
            ax.fill_between(eval_times_s, means - errs, means + errs,
                            color=color, alpha=0.15)

    ax.set_xlabel("Delay Time (s)")
    ax.set_ylabel("$\\hat{A}$ (pop. vector amplitude)")
    ax.set_title("Bump Amplitude During Delay")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.suptitle(suptitle or "Â_hat Time Courses", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration_scatter(
    grid_data: dict,
    figsize: tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Scatter plot: mean Â_hat vs success rate, colored by peak PYR rate.

    Parameters:
        grid_data: Dict mapping (amplitude, w_inter) -> dict of aggregated metrics.
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    x_vals, y_vals, c_vals, labels = [], [], [], []
    for (amp, w), d in sorted(grid_data.items()):
        x_vals.append(d.get("mean_A_hat", 0.0))
        y_vals.append(d.get("success_rate", 0.0))
        c_vals.append(d.get("peak_pyr_rate", 0.0))
        labels.append(f"({amp:.0f}, {w:.1f})")

    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="RdYlGn_r", s=80,
                    edgecolors="black", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Peak PYR Rate (Hz)")

    # Annotate points
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x_vals[i], y_vals[i]), fontsize=6,
                    textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Mean $\\hat{A}$ (pop. vector amplitude)")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Calibration Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================
# DISTRACTOR SWEEP FIGURES
# ============================================================================

def plot_distractor_sweep_heatmaps(
    grid_summary: dict,
    offsets_deg: list[float],
    amp_factors: list[float],
    collapse_threshold: float = 0.3,
    figsize: tuple[float, float] = (7, 5),
    save_dir: Optional[str] = None,
    suptitle_prefix: str = "",
):
    """Plot drift and collapse-probability heatmaps for the 2-D distractor sweep.

    Parameters:
        grid_summary: Dict mapping ``(offset_deg, amp_factor)`` to a dict with
            keys ``'drift_mean_deg'``, ``'drift_sem_deg'``, ``'collapse_prob'``.
        offsets_deg: Sorted list of distractor angular offsets (degrees).
        amp_factors: Sorted list of distractor amplitude factors (relative to cue).
        collapse_threshold: Â threshold used to declare bump collapse.
        figsize: Per-figure size.
        save_dir: If given, save both figures there.
        suptitle_prefix: Prepended to each figure title.

    Returns:
        (fig_drift, fig_collapse): Two Matplotlib Figure objects.
    """
    import matplotlib.pyplot as plt

    n_off = len(offsets_deg)
    n_amp = len(amp_factors)

    drift_mat = np.full((n_amp, n_off), np.nan)
    collapse_mat = np.full((n_amp, n_off), np.nan)

    for i_amp, amp in enumerate(amp_factors):
        for j_off, off in enumerate(offsets_deg):
            d = grid_summary.get((off, amp), {})
            drift_mat[i_amp, j_off] = d.get('drift_mean_deg', np.nan)
            collapse_mat[i_amp, j_off] = d.get('collapse_prob', np.nan)

    x_labels = [f"{off:.0f}°" for off in offsets_deg]
    y_labels = [f"{amp:.2g}×" for amp in amp_factors]

    def _annotate(ax, mat):
        for i in range(n_amp):
            for j in range(n_off):
                val = mat[i, j]
                if not np.isnan(val):
                    txt = f"{val:.1f}"
                    mid = (np.nanmax(mat) + np.nanmin(mat)) / 2
                    color = "white" if val < mid else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, color=color)

    # --- Figure 1: Drift heatmap ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    vmax = np.nanmax(np.abs(drift_mat))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    im1 = ax1.imshow(drift_mat, aspect="auto", cmap="RdBu_r", origin="lower",
                     vmin=-vmax, vmax=vmax)
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label("Mean bump shift (deg)")
    ax1.set_xticks(range(n_off))
    ax1.set_xticklabels(x_labels)
    ax1.set_yticks(range(n_amp))
    ax1.set_yticklabels(y_labels)
    ax1.set_xlabel("Distractor angular offset Δφ")
    ax1.set_ylabel("Distractor amplitude (relative to cue)")
    ax1.set_title("Bump Drift")
    _annotate(ax1, drift_mat)
    plt.suptitle(f"{suptitle_prefix}Distractor Sweep — Drift Field",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, "drift.png"),
                     dpi=150, bbox_inches="tight")

    # --- Figure 2: Collapse heatmap ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    im2 = ax2.imshow(collapse_mat, aspect="auto", cmap="YlOrRd", origin="lower",
                     vmin=0.0, vmax=1.0)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label(f"Collapse probability (Â < {collapse_threshold:.2g})")
    ax2.set_xticks(range(n_off))
    ax2.set_xticklabels(x_labels)
    ax2.set_yticks(range(n_amp))
    ax2.set_yticklabels(y_labels)
    ax2.set_xlabel("Distractor angular offset Δφ")
    ax2.set_ylabel("Distractor amplitude (relative to cue)")
    ax2.set_title("Bump Collapse Probability")
    # Annotate with percentage
    for i in range(n_amp):
        for j in range(n_off):
            val = collapse_mat[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.0%}", ha="center", va="center",
                         fontsize=8,
                         color="black" if val < 0.5 else "white")
    plt.suptitle(f"{suptitle_prefix}Distractor Sweep — Collapse Probability",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig2.savefig(os.path.join(save_dir, "collapse.png"),
                     dpi=150, bbox_inches="tight")

    return fig1, fig2


def plot_distractor_sweep_activity_grid(
    tc_data: list[dict],
    cue_onset_ms: float,
    cue_offset_ms: float,
    dist_onset_ms: float,
    dist_offset_ms: float,
    burn_in_ms: float = 10000.0,
    figsize_per_panel: tuple[float, float] = (4.0, 4.5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Grid of PYR activity heatmaps (time × position) for representative distractor-sweep cells.

    Parameters:
        tc_data: List of dicts, each with keys:
            ``'offset_deg'``, ``'amp_factor'``, ``'full_result'`` (RingSimulationResult).
        cue_onset_ms, cue_offset_ms: Cue window (absolute ms).
        dist_onset_ms, dist_offset_ms: Distractor window (absolute ms).
        burn_in_ms: Burn-in duration; subtracted to align displayed time to experiment start.
        figsize_per_panel: Width × height per subplot.
        save_path: If given, save figure there.
        suptitle: Optional figure super-title.

    Returns:
        fig: Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from .analysis import decode_bump_center

    n = len(tc_data)
    # Single row for ≤4 panels; wrap at 3 columns for larger grids
    ncols = n if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(pw * ncols, ph * nrows),
        squeeze=False,
        layout="constrained",
    )

    # Times relative to experiment start (0 = burn-in end)
    cue_on_rel = cue_onset_ms - burn_in_ms
    cue_off_rel = cue_offset_ms - burn_in_ms
    dist_on_rel = dist_onset_ms - burn_in_ms
    dist_off_rel = dist_offset_ms - burn_in_ms

    # Global vmax across all panels for a shared colour scale
    vmax = max(
        np.nanmax(entry['full_result'].r[:, :, 0])
        for entry in tc_data
    )

    ims = []
    for idx, entry in enumerate(tc_data):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        result = entry['full_result']
        off = entry['offset_deg']
        amp = entry['amp_factor']

        # Crop to post-burn-in window
        t_abs = result.t_ms  # already shifted by BURN_IN_MS in cmd_distractor_sweep
        t_rel = t_abs - burn_in_ms
        mask = t_rel >= 0
        activity = result.r[mask, :, 0]  # PYR only
        t_plot = t_rel[mask]

        # Angles
        angles_deg = result.ring_params.node_angles_deg  # shape (n_nodes,)

        # Heatmap: time on y-axis (top=early, bottom=late), position on x-axis
        extent = [angles_deg[0], angles_deg[-1], t_plot[-1], t_plot[0]]
        im = ax.imshow(
            activity,
            aspect="auto",
            cmap="hot",
            origin="upper",
            extent=extent,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)

        # Cue window — white dashed lines
        ax.axhline(cue_on_rel, color="white", ls="--", lw=1.2, alpha=0.9)
        ax.axhline(cue_off_rel, color="white", ls="--", lw=1.2, alpha=0.9)
        # Cue position — white dotted vertical line
        ax.axvline(180.0, color="white", ls=":", lw=1.0, alpha=0.7)

        # Distractor window — orange lines
        ax.axhline(dist_on_rel, color="#E69F00", ls="--", lw=1.2, alpha=0.9)
        ax.axhline(dist_off_rel, color="#E69F00", ls="--", lw=1.2, alpha=0.9)
        # Distractor angular position
        dist_angle = (180.0 + off) % 360.0
        ax.axvline(dist_angle, color="#E69F00", ls=":", lw=1.0, alpha=0.7)

        # Decoded bump trajectory (cyan dots)
        center_deg, amplitude = decode_bump_center(result, population=0)
        t_full_rel = result.t_ms - burn_in_ms
        valid = (t_full_rel >= 0) & (amplitude > 0.2)
        ax.scatter(
            center_deg[valid], t_full_rel[valid],
            c="cyan", s=1, alpha=0.5,
        )

        ax.set_title(f"Δφ={off:.0f}°,  {amp:.2g}× cue", fontsize=9)
        ax.set_xlabel("Position (deg)", fontsize=8)
        ax.set_ylabel("Time (ms)", fontsize=8)
        ax.set_xlim(angles_deg[0], angles_deg[-1])
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colorbar on the right
    cbar = fig.colorbar(ims[-1], ax=axes, label="PYR firing rate (Hz)",
                         fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(suptitle or "Distractor Sweep — PYR Activity",
                 fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distractor_sweep_timecourses(
    tc_data: list[dict],
    cue_onset_ms: float,
    cue_offset_ms: float,
    dist_onset_ms: float,
    dist_offset_ms: float,
    figsize_per_panel: tuple[float, float] = (4.5, 3.0),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Plot bump position θ(t) for representative distractor-sweep conditions.

    Parameters:
        tc_data: List of dicts, each with keys:
            ``'offset_deg'``, ``'amp_factor'``, ``'t_ms'`` (array),
            ``'center_deg'`` (array), ``'amplitude'`` (array).
        cue_onset_ms, cue_offset_ms: Cue window (absolute ms).
        dist_onset_ms, dist_offset_ms: Distractor window (absolute ms).
        figsize_per_panel: Width × height for each subplot panel.
        save_path: If given, save figure there.
        suptitle: Optional figure super-title.

    Returns:
        fig: Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = len(tc_data)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows),
                              squeeze=False)

    cue_location_deg = 180.0  # canonical cue position

    for idx, entry in enumerate(tc_data):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        t_s = entry['t_ms'] / 1000.0
        center = entry['center_deg']

        # Shaded regions
        ax.axvspan(cue_onset_ms / 1000, cue_offset_ms / 1000,
                   color="#b0b0b0", alpha=0.35, label="Cue")
        ax.axvspan(dist_onset_ms / 1000, dist_offset_ms / 1000,
                   color="#E69F00", alpha=0.40, label="Distractor")

        ax.plot(t_s, center, color="#333333", lw=1.2)
        ax.axhline(cue_location_deg, color="#56B4E9", ls="--", lw=1.0,
                   label=f"Cue pos ({cue_location_deg:.0f}°)")

        off = entry['offset_deg']
        amp = entry['amp_factor']
        ax.set_title(f"Δφ={off:.0f}°,  {amp:.2g}× cue", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Bump position (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.suptitle(suptitle or "Distractor Sweep — Bump Trajectories",
                 fontsize=12, fontweight="bold")
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
    print(f"Stimulus: {result.stim_angle_deg:.0f} deg ({result.stim_window[0]:.0f}-{result.stim_window[1]:.0f} ms)")
    print()

    # Bump metrics during delay
    metrics = compute_bump_metrics(result)
    print("Bump Metrics (delay period):")
    print(f"  Center: {metrics['center_mean_deg']:.1f} deg +/- {metrics['center_std_deg']:.1f} deg")
    print(f"  Width: {metrics['width_mean_deg']:.1f} deg")
    print(f"  Decoding amplitude: {metrics['amplitude_mean']:.2f}")
    print(f"  Drift rate: {metrics['drift_rate_deg_per_s']:.1f} deg/s")
    print(f"  Diffusion: {metrics['diffusion_deg2_per_s']:.1f} deg^2/s")
    print()

    # Working memory accuracy
    accuracy = compute_working_memory_accuracy(result)
    print("Working Memory Performance:")
    print(f"  Cue position: {accuracy['cue_position_deg']:.0f} deg")
    print(f"  Final position: {accuracy['final_position_deg']:.1f} deg")
    print(f"  Error: {accuracy['error_deg']:.1f} deg")
    print(f"  Bump maintained: {'Yes' if accuracy['maintained'] else 'No'}")
    print("=" * 50)
