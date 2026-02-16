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
    import matplotlib.ticker as mticker

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

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
            # Detect aggregated format (mean/sd from multi-trial)
            if f"{metric_key}_mean" in metric_list[0]:
                means = np.array([m[f"{metric_key}_mean"] for m in metric_list[:n_pts]])
                sds = np.array([m[f"{metric_key}_sd"] for m in metric_list[:n_pts]])
                ax.plot(x, means, marker="o", color=color, label=label, lw=2, markersize=4)
                if np.any(sds > 0):
                    ax.fill_between(x, means - sds, means + sds,
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
            sds = []
            for amp in amplitude_values:
                m = all_delay_metrics.get(amp, {}).get(cond_key, {})
                # Detect aggregated format
                if f"{metric_key}_mean" in m:
                    means.append(m.get(f"{metric_key}_mean", float("nan")))
                    sds.append(m.get(f"{metric_key}_sd", 0.0))
                else:
                    means.append(m.get(metric_key, float("nan")))
                    sds.append(0.0)
            means_arr = np.array(means)
            sds_arr = np.array(sds)
            ax.plot(amplitude_values, means_arr, marker="o", color=color,
                    label=label, lw=2, markersize=6)
            if np.any(sds_arr > 0):
                ax.fill_between(amplitude_values,
                                means_arr - sds_arr, means_arr + sds_arr,
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
