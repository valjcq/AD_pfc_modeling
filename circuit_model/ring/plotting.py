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

from ..plotting import POPULATION_NAMES, POPULATION_COLORS, ADAPTATION_COLORS
from .constants import TRANSIENT_SKIP_TIME_MS

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
            ax.legend(["PYR"], loc="upper right", fontsize=7)
        else:
            ax.plot(angles, r_pyr, color=color, linewidth=2, label="PYR")
            ax.fill_between(angles, 0, r_pyr, color=color, alpha=0.3)
            ax.legend(["PYR"], loc="upper right", fontsize=7)

    ax.set_title(f"t = {actual_t - t_offset:.1f} ms")

    if not polar:
        ax.set_xlabel("Position (degrees)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlim(0, 360)

    return ax


def animate_ring_snapshot_evolution(
    result: "RingSimulationResult",
    save_path: str,
    population: int = 0,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
    frame_step_ms: float = 2.0,
    fps: int = 30,
    figsize: tuple[float, float] = (8.0, 9.0),
    suptitle: Optional[str] = None,
    cue_window: Optional[tuple[float, float]] = None,
    cue_angle_deg: Optional[float] = None,
    distractor_window: Optional[tuple[float, float]] = None,
    distractor_angle_deg: Optional[float] = None,
    show_asymmetry: bool = False,
    n_workers: int = 4,
    dpi: int = 100,
    av1_crf: int = 35,
    av1_preset: int = 8,
):
    """Animate ring snapshot evolution over time and save to MP4.

    Parameters:
        result: RingSimulationResult
        save_path: Output path. Extension must be .mp4
        population: Population index (0=PYR)
        time_range: Optional absolute time bounds (start_ms, end_ms)
        t_offset: Display offset subtracted from shown time labels
        frame_step_ms: Temporal step between frames in ms
        fps: Animation frame rate
        figsize: Figure size
        suptitle: Optional figure title
        cue_window: Optional (onset_ms, offset_ms) for cue shading
        cue_angle_deg: Optional cue angle marker override
        distractor_window: Optional (onset_ms, offset_ms) for distractor shading
        distractor_angle_deg: Optional distractor angle marker on snapshot/profile
        av1_crf: AV1 constant-rate-factor (lower = better quality, slower/larger)
        av1_preset: AV1 speed/quality preset (lower = better quality, slower)

    Returns:
        fig: Matplotlib figure used for animation
        ani: Matplotlib FuncAnimation object
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    t = result.t_ms
    if len(t) < 2:
        raise ValueError("Need at least 2 recorded time points to build animation.")

    if time_range is None:
        mask = np.ones_like(t, dtype=bool)
    else:
        mask = (t >= time_range[0]) & (t <= time_range[1])

    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("No time points found in requested time_range.")

    dt_ms = float(np.median(np.diff(t)))
    stride = max(1, int(round(frame_step_ms / max(dt_ms, 1e-9))))
    frame_idx = idx[::stride]
    if frame_idx[-1] != idx[-1]:
        frame_idx = np.append(frame_idx, idx[-1])

    angles = result.ring_params.node_angles_rad
    angles_closed = np.append(angles, angles[0])

    pop_name = POPULATION_NAMES[population]
    color = list(POPULATION_COLORS.values())[population]

    cue_angle = float(result.stim_angle_deg) if cue_angle_deg is None else float(cue_angle_deg)
    cue_time_window = result.stim_window if cue_window is None else cue_window

    distractor_angle = None if distractor_angle_deg is None else float(distractor_angle_deg)
    has_distractor = distractor_angle is not None

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax_asym_anim = None
    if has_distractor:
        if show_asymmetry:
            gs = fig.add_gridspec(5, 1, height_ratios=[2.0, 1.2, 1.0, 1.0, 0.8], hspace=0.5)
            ax_asym_anim = fig.add_subplot(gs[4])
        else:
            gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.2, 1.0, 1.0], hspace=0.5)
        ax = fig.add_subplot(gs[0], projection="polar")
        ax_profile = fig.add_subplot(gs[1])
        ax_nodes = fig.add_subplot(gs[2])
        ax_diff = fig.add_subplot(gs[3])
    else:
        if show_asymmetry:
            gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.3, 1.0, 0.8], hspace=0.45)
            ax_asym_anim = fig.add_subplot(gs[3])
        else:
            gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.3, 1.0], hspace=0.45)
        ax = fig.add_subplot(gs[0], projection="polar")
        ax_profile = fig.add_subplot(gs[1])
        ax_nodes = fig.add_subplot(gs[2])
        ax_diff = None
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    first = result.r[frame_idx[0], :, population]
    first_closed = np.append(first, first[0])

    line, = ax.plot(angles_closed, first_closed, color=color, linewidth=2)
    fill = ax.fill(angles_closed, first_closed, color=color, alpha=0.3)[0]
    title_text = ax.set_title(f"{pop_name} snapshot — t = {t[frame_idx[0]] - t_offset:.1f} ms")

    # Cue marker on ring
    cue_rad = np.deg2rad(cue_angle)
    ax.plot([cue_rad, cue_rad], [0, 1], color="red", ls="--", lw=1.2, alpha=0.9)
    if has_distractor:
        dist_rad = np.deg2rad(distractor_angle)
        ax.plot([dist_rad, dist_rad], [0, 1], color="#E69F00", ls="--", lw=1.2, alpha=0.9)

    # Instantaneous angular activity profile (same frame as snapshot)
    angles_deg = result.ring_params.node_angles_deg
    line_profile, = ax_profile.plot(angles_deg, first, color=color, linewidth=2)
    _fp_x = np.concatenate([angles_deg, angles_deg[::-1]])
    _fp_y = np.concatenate([first, np.zeros(len(first))])
    fill_profile = ax_profile.fill(_fp_x, _fp_y, color=color, alpha=0.25)[0]
    ax_profile.axvline(cue_angle, color="red", ls="--", lw=1, alpha=0.8, label="Cue")
    if has_distractor:
        ax_profile.axvline(
            distractor_angle,
            color="#E69F00",
            ls="--",
            lw=1,
            alpha=0.85,
            label="Distractor",
        )
    ax_profile.set_xlim(0, 360)
    ax_profile.set_ylabel("Rate (Hz)")
    ax_profile.set_xlabel("Position (deg)")
    ax_profile.set_title("Ring activity profile at current frame")

    if has_distractor:
        ax_profile.legend(loc="upper right", fontsize=8)

    # Time traces at cue/distractor nodes + difference, with moving cursors
    t_display = t - t_offset
    angle_diffs_cue = np.abs(((angles_deg - cue_angle + 180.0) % 360.0) - 180.0)
    cue_node = int(np.argmin(angle_diffs_cue))
    cue_node_angle = float(angles_deg[cue_node])
    cue_trace = result.r[:, cue_node, population]
    diff_cursor = None
    dist_trace = None
    if has_distractor:
        if ax_diff is None:
            raise RuntimeError("Internal plotting error: distractor axis is missing.")
        angle_diffs_dist = np.abs(((angles_deg - distractor_angle + 180.0) % 360.0) - 180.0)
        dist_node = int(np.argmin(angle_diffs_dist))
        dist_node_angle = float(angles_deg[dist_node])
        dist_trace = result.r[:, dist_node, population]
        ax_diff_local = ax_diff
        ax_nodes.plot(t_display, cue_trace, color="red", lw=1.6, label=f"Cue node ({cue_node_angle:.1f}°)")
        ax_nodes.plot(t_display, dist_trace, color="#E69F00", lw=1.6,
                      label=f"Distractor node ({dist_node_angle:.1f}°)")
        if cue_time_window[1] > cue_time_window[0]:
            ax_nodes.axvspan(cue_time_window[0] - t_offset, cue_time_window[1] - t_offset,
                             color="red", alpha=0.12)
        if distractor_window is not None and distractor_window[1] > distractor_window[0]:
            ax_nodes.axvspan(distractor_window[0] - t_offset, distractor_window[1] - t_offset,
                             color="#E69F00", alpha=0.14)
        _mark_transient(ax_nodes, result, t_offset=t_offset, orientation="vertical")
        cue_cursor = ax_nodes.axvline(float(t_display[frame_idx[0]]), color="black", lw=1.2, alpha=0.9)
        ax_nodes.set_xlim(float(t_display[idx[0]]), float(t_display[idx[-1]]))
        ax_nodes.set_ylabel("Rate (Hz)")
        ax_nodes.set_title("Node firing rates at cue and distractor locations")
        ax_nodes.legend(loc="upper right", fontsize=8)

        diff_trace = cue_trace - dist_trace
        ax_diff_local.plot(t_display, diff_trace, color="#0072B2", lw=1.6)
        ax_diff_local.axhline(0.0, color="black", lw=0.9, alpha=0.6)
        if cue_time_window[1] > cue_time_window[0]:
            ax_diff_local.axvspan(cue_time_window[0] - t_offset, cue_time_window[1] - t_offset,
                                  color="red", alpha=0.12)
        if distractor_window is not None and distractor_window[1] > distractor_window[0]:
            ax_diff_local.axvspan(distractor_window[0] - t_offset, distractor_window[1] - t_offset,
                                  color="#E69F00", alpha=0.14)
        _mark_transient(ax_diff_local, result, t_offset=t_offset, orientation="vertical")
        diff_cursor = ax_diff_local.axvline(float(t_display[frame_idx[0]]), color="black", lw=1.2, alpha=0.9)
        ax_diff_local.set_xlim(float(t_display[idx[0]]), float(t_display[idx[-1]]))
        ax_diff_local.set_ylabel("Cue - Distractor (Hz)")
        if not show_asymmetry:
            ax_diff_local.set_xlabel("Time (ms)")
        ax_diff_local.set_title("Difference between cue and distractor node rates")
    else:
        ax_nodes.plot(t_display, cue_trace, color="red", lw=1.6, label=f"Cue node ({cue_node_angle:.1f}°)")
        if cue_time_window[1] > cue_time_window[0]:
            ax_nodes.axvspan(cue_time_window[0] - t_offset, cue_time_window[1] - t_offset,
                             color="red", alpha=0.14)
        _mark_transient(ax_nodes, result, t_offset=t_offset, orientation="vertical")
        cue_cursor = ax_nodes.axvline(float(t_display[frame_idx[0]]), color="black", lw=1.2, alpha=0.9)
        ax_nodes.set_xlim(float(t_display[idx[0]]), float(t_display[idx[-1]]))
        ax_nodes.set_ylabel("Rate (Hz)")
        if not show_asymmetry:
            ax_nodes.set_xlabel("Time (ms)")
        ax_nodes.set_title(f"Cue node firing rate ({cue_node_angle:.1f}°)")

    r_max = float(np.max(result.r[:, :, population]))
    ax.set_ylim(0, max(1.0, r_max * 1.05))
    ax_profile.set_ylim(0, max(1.0, r_max * 1.05))
    ax_nodes.set_ylim(0, max(1.0, r_max * 1.05))
    if ax_diff is not None and has_distractor and dist_trace is not None:
        diff_abs = float(np.max(np.abs(cue_trace - dist_trace)))
        ax_diff.set_ylim(-max(1.0, diff_abs * 1.1), max(1.0, diff_abs * 1.1))

    # --- Asymmetry panel (non-distractor only) ---
    asym_cursor = None
    if show_asymmetry and ax_asym_anim is not None:
        from .analysis import compute_bump_asymmetry
        cmap_asym_anim, norm_asym_anim = _asym_cmap_norm()
        asym_full = compute_bump_asymmetry(result, population)
        asym_masked = asym_full[mask]
        t_asym_display = t_display[mask]
        asym_ylim = max(float(np.max(np.abs(asym_masked))), 0.05) * 1.25
        ax_asym_anim.scatter(
            t_asym_display, asym_masked,
            c=asym_masked, cmap=cmap_asym_anim, norm=norm_asym_anim,
            s=4, alpha=0.7, linewidths=0, zorder=3,
        )
        ax_asym_anim.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.7)
        if cue_time_window[1] > cue_time_window[0]:
            ax_asym_anim.axvspan(
                cue_time_window[0] - t_offset, cue_time_window[1] - t_offset,
                color="red", alpha=0.12,
            )
        _mark_transient(ax_asym_anim, result, t_offset=t_offset, orientation="vertical")
        ax_asym_anim.set_xlim(float(t_display[idx[0]]), float(t_display[idx[-1]]))
        ax_asym_anim.set_ylim(-asym_ylim, asym_ylim)
        ax_asym_anim.set_ylabel("Asymmetry\n(R−L)", fontsize=9)
        ax_asym_anim.set_xlabel("Time (ms)")
        asym_cursor = ax_asym_anim.axvline(
            float(t_display[frame_idx[0]]), color="black", lw=1.2, alpha=0.9,
        )

    def _update(k: int):
        ti = frame_idx[k]
        values = result.r[ti, :, population]
        values_closed = np.append(values, values[0])
        line.set_data(angles_closed, values_closed)
        fill.set_xy(np.column_stack([angles_closed, values_closed]))
        line_profile.set_data(angles_deg, values)
        fill_profile.set_xy(np.column_stack([
            np.concatenate([angles_deg, angles_deg[::-1]]),
            np.concatenate([values, np.zeros(len(values))]),
        ]))
        cue_cursor.set_xdata([t_display[ti], t_display[ti]])
        title_text.set_text(f"{pop_name} snapshot — t = {t[ti] - t_offset:.1f} ms")
        artists = [line, fill, line_profile, fill_profile, cue_cursor, title_text]
        if diff_cursor is not None:
            diff_cursor.set_xdata([t_display[ti], t_display[ti]])
            artists.append(diff_cursor)
        if asym_cursor is not None:
            asym_cursor.set_xdata([t_display[ti], t_display[ti]])
            artists.append(asym_cursor)
        return artists

    import subprocess as _sp
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA

    if os.path.splitext(save_path)[1].lower() != ".mp4":
        raise ValueError("save_path must use .mp4 extension for video output.")

    # Switch to Agg (no X server connection) — required for fork safety and
    # for saving to file with a non-interactive backend.
    if not isinstance(fig.canvas, _FCA):
        fig.set_canvas(_FCA(fig))

    # Initial draw to lock in frame dimensions
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    n_total = len(frame_idx)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgba", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libsvtav1", "-preset", str(av1_preset), "-crf", str(av1_crf),
        "-pix_fmt", "yuv420p", "-svtav1-params", f"lp={n_workers}",
        save_path,
    ]

    if n_workers > 1:
        import multiprocessing as _mp
        import tempfile as _tf

        frame_size = W * H * 4  # RGBA bytes
        with _tf.NamedTemporaryFile(delete=False, suffix=".raw") as _f:
            tmp_path = _f.name
        raw_buf = np.memmap(tmp_path, dtype=np.uint8, mode="w+",
                            shape=(n_total * frame_size,))

        def _render_chunk(chunk_ks: list) -> None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
            fig.set_canvas(_FCA(fig))  # detach inherited X connection in child
            for k in chunk_ks:
                _update(k)
                fig.canvas.draw()
                raw_buf[k * frame_size:(k + 1) * frame_size] = np.frombuffer(
                    fig.canvas.buffer_rgba(), dtype=np.uint8
                )

        chunk_size = max(1, (n_total + n_workers - 1) // n_workers)
        chunks = [
            list(range(i * chunk_size, min((i + 1) * chunk_size, n_total)))
            for i in range(n_workers) if i * chunk_size < n_total
        ]
        ctx = _mp.get_context("fork")
        procs = [ctx.Process(target=_render_chunk, args=(c,)) for c in chunks]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        raw_buf.flush()

        proc = _sp.Popen(ffmpeg_cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        for k in range(n_total):
            proc.stdin.write(raw_buf[k * frame_size:(k + 1) * frame_size].tobytes())
        proc.stdin.close()
        proc.wait()
        del raw_buf
        os.unlink(tmp_path)
    else:
        proc = _sp.Popen(ffmpeg_cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        for k in range(n_total):
            _update(k)
            fig.canvas.draw()
            proc.stdin.write(bytes(fig.canvas.buffer_rgba()))
        proc.stdin.close()
        proc.wait()

    return fig, None


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


def plot_population_activity(
    result: "RingSimulationResult",
    pre_cue_ms: float = 200.0,
    save_path: Optional[str] = None,
    t_offset: float = 0.0,
):
    """
    Plot firing rate time courses at the cue location and opposite side for all populations,
    plus adaptation currents when available.

    Layout (all subplots share the x-axis):
      - PYR: cue + opposite on the same subplot (solid / dashed)
      - SOM: cue + opposite on the same subplot
      - PV:  cue + opposite on the same subplot
      - VIP @ cue location   (separate subplot)
      - VIP @ opposite location (separate subplot)
      - PYR adaptation current: cue + opposite  (only when I_adapt_stored is available)
      - SOM adaptation current: cue + opposite  (only when I_adapt_stored is available)

    The time window starts pre_cue_ms before cue onset and covers the full delay period.

    Parameters:
        result: RingSimulationResult
        pre_cue_ms: How many ms before cue onset to include
        save_path: If provided, save figure to this path
        t_offset: Subtracted from absolute time for display (typically BURN_IN_MS)

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stim_node = result.stim_node
    n_nodes = result.n_nodes
    opp_node = (stim_node + n_nodes // 2) % n_nodes
    stim_angle = result.ring_params.node_angles_deg[stim_node]
    opp_angle = result.ring_params.node_angles_deg[opp_node]

    t_start_abs = result.stim_window[0] - pre_cue_ms
    mask = (result.t_ms >= t_start_abs) & (result.t_ms <= result.t_ms[-1])
    t_display = result.t_ms[mask] - t_offset

    cue_on_disp = result.stim_window[0] - t_offset
    cue_off_disp = result.stim_window[1] - t_offset

    has_adapt = result.I_adapt_stored is not None

    # Build row list: (label, y-label, data_fn)
    # data_fn(node) -> 1-D array for that node
    rows = []
    for i, name in enumerate(POPULATION_NAMES):  # PYR, SOM, PV, VIP
        if name == "VIP":
            # Two separate rows for VIP
            rows.append(("VIP  —  cue", "VIP\n(Hz)", name,
                         lambda m=mask, nd=stim_node, pi=i: result.r[m, nd, pi],
                         None))
            rows.append(("VIP  —  opposite", "VIP\n(Hz)", name,
                         lambda m=mask, nd=opp_node, pi=i: result.r[m, nd, pi],
                         None))
        else:
            rows.append((name, f"{name}\n(Hz)", name,
                         lambda m=mask, nd=stim_node, pi=i: result.r[m, nd, pi],
                         lambda m=mask, nd=opp_node, pi=i: result.r[m, nd, pi]))

    if has_adapt:
        for adapt_idx, name in enumerate(["PYR", "SOM"]):
            rows.append(
                (f"{name} adapt.", f"I_adapt\n(a.u.)", name,
                 lambda m=mask, nd=stim_node, ai=adapt_idx:
                     result.I_adapt_stored[m, nd, ai],
                 lambda m=mask, nd=opp_node, ai=adapt_idx:
                     result.I_adapt_stored[m, nd, ai])
            )

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.5 * n_rows + 1), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (title, ylabel, pop_name, fn_cue, fn_opp) in zip(axes, rows):
        # Choose color: population color for firing rates, adaptation color for adapt rows
        if "adapt" in title:
            color = ADAPTATION_COLORS[pop_name]
        else:
            color = POPULATION_COLORS[pop_name]

        data_cue = fn_cue()
        ax.plot(t_display, data_cue, color=color, lw=1.5, ls="-")

        if fn_opp is not None:
            data_opp = fn_opp()
            ax.plot(t_display, data_opp, color=color, lw=1.5, ls="--")

        if cue_off_disp > cue_on_disp:
            ax.axvspan(cue_on_disp, cue_off_disp, alpha=0.2, color="gray")

        _mark_transient(ax, result, t_offset=t_offset)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=8, loc="right", pad=2)
        ax.tick_params(labelsize=8)

    # Shared legend in the first subplot
    legend_handles = [
        Line2D([0], [0], color="k", lw=1.5, ls="-",
               label=f"Cue location ({stim_angle:.0f}°)"),
        Line2D([0], [0], color="k", lw=1.5, ls="--",
               label=f"Opposite ({opp_angle:.0f}°)"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Population Activity: Cue vs. Opposite Location",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_bump_metrics_over_time(
    result: "RingSimulationResult",
    population: int = 0,
    ax=None,
    time_range: Optional[tuple[float, float]] = None,
    t_offset: float = 0.0,
):
    """
    Plot decoded bump center, amplitude, width, and L/R asymmetry over time.

    Parameters:
        result: RingSimulationResult
        population: Which population to decode (0=PYR)
        ax: Array of 4 axes (created if None)
        time_range: Optional (start_ms, end_ms) to restrict time

    Returns:
        axes: Array of 4 Matplotlib axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from .analysis import decode_bump_center, estimate_bump_width, compute_bump_asymmetry

    center_deg, amplitude = decode_bump_center(result, population)
    asymmetry = compute_bump_asymmetry(result, population)
    t = result.t_ms

    # Time range filtering
    if time_range:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t = t[mask]
        center_deg = center_deg[mask]
        amplitude = amplitude[mask]
        asymmetry = asymmetry[mask]
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

    # Diverging colormap: blue (left) → black (symmetric) → yellow (right)
    cmap_asym = mcolors.LinearSegmentedColormap.from_list(
        "asymmetry", ["#0072B2", "#000000", "#F0E442"]
    )
    norm_asym = mcolors.Normalize(vmin=-1, vmax=1)

    # Apply display offset
    t_display = t - t_offset
    t_width_display = t_width - t_offset

    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

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

    # --- Asymmetry ---
    sc = ax[3].scatter(
        t_display, asymmetry,
        c=asymmetry, cmap=cmap_asym, norm=norm_asym,
        s=2, alpha=0.7,
    )
    ax[3].axhline(0, color="gray", ls="--", lw=0.8)
    ax[3].set_ylim(-1, 1)
    ax[3].set_ylabel("Asymmetry\n(right − left)")
    ax[3].set_xlabel("Time (ms)")
    plt.colorbar(sc, ax=ax[3], label="← left    right →", orientation="vertical",
                 fraction=0.03, pad=0.01)

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
    t_snap = min(result.stim_window[1] + TRANSIENT_SKIP_TIME_MS, result.t_ms[-1])
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
    exc_label = f"PYR->PYR excitatory (sigma={ring_params.sigma_pyr_deg:.0f} deg)"
    legend_elements = [
        Line2D([0], [0], color=excit_color, linewidth=2.5, label=exc_label),
        Line2D([0], [0], color=inhib_color, linewidth=1, linestyle="--",
               label="PV->PYR inhibitory (uniform)"),
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
    separate_app: bool = False,
):
    """
    Plot bump metrics at multiple delay timepoints, comparing conditions.

    Parameters:
        metrics_over_delay: dict mapping condition_key -> list of metric dicts
        delay_labels: Human-readable labels for each timepoint
        error_band: ``"sem"`` (default) or ``"sd"`` — controls the shaded band.
        separate_app: If True, split into two rows: Non-APP (top) and APP (bottom).

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    band_suffix = "_sem" if error_band == "sem" else "_sd"
    n_metrics = len(metrics_to_plot)

    from ..study import STUDY_CONDITIONS

    # Parse delay labels to numeric seconds for the x-axis
    x_seconds = []
    for lbl in delay_labels:
        try:
            x_seconds.append(float(lbl.rstrip("s")))
        except ValueError:
            x_seconds.append(float("nan"))
    x_seconds = np.array(x_seconds)

    def _plot_conditions_delay(keys, row_axes, x_seconds, band_suffix):
        for cond_key in keys:
            if cond_key not in metrics_over_delay:
                continue
            metric_list = metrics_over_delay[cond_key]
            color = condition_colors.get(cond_key, None)
            label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key
            n_pts = min(len(x_seconds), len(metric_list))
            x = x_seconds[:n_pts]
            for ax, metric_key in zip(row_axes, metrics_to_plot):
                if metric_list and f"{metric_key}_mean" in metric_list[0]:
                    means = np.array([m[f"{metric_key}_mean"] for m in metric_list[:n_pts]])
                    errs  = np.array([m.get(f"{metric_key}{band_suffix}",
                                            m.get(f"{metric_key}_sd", 0.0))
                                      for m in metric_list[:n_pts]])
                    ax.plot(x, means, marker="o", color=color, label=label, lw=2, markersize=4)
                    if np.any(errs > 0):
                        ax.fill_between(x, means - errs, means + errs, color=color, alpha=0.2)
                else:
                    values = [m[metric_key] for m in metric_list[:n_pts]]
                    ax.plot(x, values, marker="o", color=color, label=label, lw=2, markersize=4)

    if separate_app:
        non_app_keys = [k for k in metrics_over_delay if not k.endswith("_APP")]
        app_keys     = [k for k in metrics_over_delay if k.endswith("_APP")]

        fig, axes_2d = plt.subplots(
            2, n_metrics,
            figsize=(figsize[0], figsize[1] * 2),
            sharey="col",
            squeeze=False,
        )

        groups = [
            (non_app_keys, axes_2d[0], "Non-APP"),
            (app_keys,     axes_2d[1], "APP"),
        ]

        for row_idx, (keys, row_axes, row_label) in enumerate(groups):
            _plot_conditions_delay(keys, row_axes, x_seconds, band_suffix)
            for ax, metric_key in zip(row_axes, metrics_to_plot):
                ax.set_xlabel("Delay time (s)")
                ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
                ax.legend(fontsize=8, title=row_label, title_fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))

        # Metric names as column titles on top row only
        for ax, metric_key in zip(axes_2d[0], metrics_to_plot):
            ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))

        plt.suptitle(suptitle or "Bump Metrics During Delay Period", fontsize=13, fontweight="bold")
        

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # --- single-row (default) path ---
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for cond_key, metric_list in metrics_over_delay.items():
        color = condition_colors.get(cond_key, None)
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
    separate_app: bool = False,
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
        separate_app: If True, split into two rows: Non-APP (top) and APP (bottom).

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    band_suffix = "_sem" if error_band == "sem" else "_sd"
    n_metrics = len(metrics_to_plot)

    from ..study import STUDY_CONDITIONS

    # Collect all condition keys across amplitudes (preserving order)
    cond_keys = []
    for amp in amplitude_values:
        for k in all_delay_metrics.get(amp, {}):
            if k not in cond_keys:
                cond_keys.append(k)

    def _plot_conditions_amplitude(keys, row_axes):
        for cond_key in keys:
            color = condition_colors.get(cond_key, None)
            label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key
            for ax, metric_key in zip(row_axes, metrics_to_plot):
                means = []
                errs = []
                for amp in amplitude_values:
                    m = all_delay_metrics.get(amp, {}).get(cond_key, {})
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

    if separate_app:
        non_app_keys = [k for k in cond_keys if not k.endswith("_APP")]
        app_keys     = [k for k in cond_keys if k.endswith("_APP")]

        fig, axes_2d = plt.subplots(
            2, n_metrics,
            figsize=(figsize[0], figsize[1] * 2),
            sharey="col",
            squeeze=False,
        )

        groups = [
            (non_app_keys, axes_2d[0], "Non-APP"),
            (app_keys,     axes_2d[1], "APP"),
        ]

        for row_idx, (keys, row_axes, row_label) in enumerate(groups):
            _plot_conditions_amplitude(keys, row_axes)
            for ax, metric_key in zip(row_axes, metrics_to_plot):
                ax.set_xlabel("Stimulus Amplitude (× I_ext_pyr)")
                ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
                ax.legend(fontsize=8, title=row_label, title_fontsize=9)
                ax.grid(True, alpha=0.3)

        # Metric names as column titles on top row only
        for ax, metric_key in zip(axes_2d[0], metrics_to_plot):
            ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))

        plt.suptitle(suptitle or "Bump Metrics vs Stimulus Amplitude",
                     fontsize=13, fontweight="bold")
        

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # --- single-row (default) path ---
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    _plot_conditions_amplitude(cond_keys, axes)

    for ax, metric_key in zip(axes, metrics_to_plot):
        ax.set_xlabel("Stimulus Amplitude (× I_ext_pyr)")
        ax.set_ylabel(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.set_title(_METRIC_DISPLAY_NAMES.get(metric_key, metric_key))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Bump Metrics vs Stimulus Amplitude",
                 fontsize=13, fontweight="bold")
    

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

    # Robust y-limits to avoid a single extreme outlier ruining readability
    all_disps = []
    for ck in cond_keys:
        dvals = np.asarray(disp_data[ck].get('displacements_deg', np.array([])), dtype=float)
        if dvals.size > 0:
            all_disps.append(dvals)
    y_clip_low = None
    y_clip_high = None
    if all_disps:
        stacked = np.concatenate(all_disps)
        if stacked.size >= 20:
            p_low, p_high = np.percentile(stacked, [0.5, 99.5])
            span = max(1.0, float(p_high - p_low))
            margin = 0.08 * span
            y_clip_low = float(p_low - margin)
            y_clip_high = float(p_high + margin)

    # --- Violin / strip plot ---
    n_low_clip = 0
    n_high_clip = 0
    n_total_pts = 0
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
        disps_plot = np.asarray(disps, dtype=float)
        if y_clip_low is not None and y_clip_high is not None:
            n_low_clip += int(np.sum(disps_plot < y_clip_low))
            n_high_clip += int(np.sum(disps_plot > y_clip_high))
            n_total_pts += int(disps_plot.size)
            disps_plot = np.clip(disps_plot, y_clip_low, y_clip_high)
        ax_viol.scatter(xi + jitter, disps_plot, color=color, s=6, alpha=0.35,
                        zorder=3)
        # Mean marker
        mean_val = float(np.mean(disps))
        ax_viol.scatter([xi], [mean_val], color=color, s=60,
                        marker='D', zorder=5, edgecolors='black', linewidths=0.5)
        # n annotation
        ax_viol.text(xi, 0, f"n={n_valid}/{n_total}", ha='center', va='top',
                     fontsize=7, color='gray')

    if y_clip_low is not None and y_clip_high is not None:
        ax_viol.set_ylim(y_clip_low, y_clip_high)
        # Reposition n labels at the bottom edge after limits are fixed
        for txt in ax_viol.texts:
            if txt.get_text().startswith("n="):
                txt.set_y(y_clip_low)

        if (n_low_clip + n_high_clip) > 0 and n_total_pts > 0:
            ax_viol.text(
                0.99, 0.99,
                f"robust scale (0.5–99.5%): clipped {n_low_clip + n_high_clip}/{n_total_pts} pts",
                transform=ax_viol.transAxes,
                ha='right', va='top', fontsize=7, color='gray',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    ax_viol.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax_viol.set_xticks(x)
    ax_viol.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_viol.set_ylabel("Minimum displacement from cue (°)")
    ax_viol.set_title("Displacement distribution (◆ = mean)")
    ax_viol.grid(True, alpha=0.25, axis='y')

    plt.suptitle(suptitle or "Final Bump Displacement from Cue",
                 fontsize=12, fontweight="bold")
    

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
    """Ring activity during delay across conditions (one random sample each).

    For each condition with a full ``sample_result`` (RingSimulationResult),
    two stacked panels are shown:

        * **Top (large)** — activity heatmap: angle on x-axis, time on y-axis,
            PYR firing rate as colour, decoded bump centre overlaid as cyan dots.
        * **Bottom (small)** — decoded amplitude over cue+delay period, with the
      noise threshold as a horizontal dashed line when available.

    Falls back to a "no data" placeholder when ``sample_result`` is absent
    (e.g. when loading results from cache).

    Parameters
    ----------
    disp_data : dict
        Mapping condition_key → dict produced by ``cmd_diffusion``.
        Expected keys: ``sample_result`` (RingSimulationResult or None),
        ``sample_displacement_deg`` (float),
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
        if d.get('sample_result') is not None or d.get('extreme_result') is not None
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

    ncols = min(4, n_cond)
    nrows = int(np.ceil(n_cond / ncols))

    if figsize is None:
        figsize = (7 * ncols, 9 * nrows)

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.35, hspace=0.35)
    pre_cue_ms = 100.0

    for idx, ck in enumerate(valid_conds):
        row, col = divmod(idx, ncols)
        d = disp_data[ck]
        result = d.get('sample_result')
        if result is None:
            result = d.get('extreme_result')
        if result is None:
            continue
        delay_start = float(d.get('delay_start_ms', result.t_ms[0]))
        cue_duration = max(0.0, float(result.stim_window[1] - result.stim_window[0]))
        cue_end = delay_start - TRANSIENT_SKIP_TIME_MS
        cue_start = cue_end - cue_duration
        window_start = max(float(result.t_ms[0]), cue_start - pre_cue_ms)
        delay_end = float(d.get('delay_end_ms', result.t_ms[-1]))
        disp_deg = float(d.get('sample_displacement_deg', d.get('extreme_displacement_deg', 0.0)))
        noise_thr = d.get('noise_threshold', None)
        amp_factor = d.get('amplitude_factor', None)
        if amp_factor is None:
            stim_current = d.get('stim_current', None)
            if stim_current is not None:
                try:
                    amp_factor = float(stim_current) / float(result.local_params.I_ext_pyr())
                except Exception:
                    amp_factor = None
        label = STUDY_CONDITIONS[ck].name if ck in STUDY_CONDITIONS else ck

        # Two rows: heatmap (75%) + amplitude (25%)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[3, 1], hspace=0.08,
        )
        ax_heat = fig.add_subplot(inner[0])
        # Amplitude panel has its own independent x-axis (time, not angle)
        ax_amp = fig.add_subplot(inner[1])

        title = f"{label} — random sample ({disp_deg:+.1f}°)"
        if amp_factor is not None:
            title += f" | {float(amp_factor):.0f}× baseline"
        # --- Heatmap panel (angle on x, time on y — original orientation) ---
        plot_ring_activity_heatmap(
            result,
            population=0,
            ax=ax_heat,
            title=title,
            cmap="hot",
            time_range=(window_start, delay_end),
            show_stimulus=False,
            show_decoded=True,
            t_offset=cue_start,
        )
        ax_heat.set_ylabel("Time from cue onset (ms)")
        ax_heat.axhline(0.0, color="cyan", ls="-", lw=1.0, alpha=0.9,
                        label="Cue ON")
        if cue_duration > 0:
            ax_heat.axhline(cue_duration, color="cyan", ls="--", lw=1.0, alpha=0.9,
                            label="Cue OFF")
        ax_heat.axvline(result.stim_angle_deg, color="white", ls="--", lw=1.2,
                        alpha=0.8, label=f"Cue ({result.stim_angle_deg:.0f}°)")
        ax_heat.legend(fontsize=7, loc="upper right", framealpha=0.4)

        # --- Amplitude panel ---
        _, amplitude = decode_bump_center(result, population=0)
        t_ms = result.t_ms
        mask = (t_ms >= window_start) & (t_ms <= delay_end)
        t_plot = t_ms[mask] - cue_start
        amp_delay = amplitude[mask]

        ax_amp.plot(t_plot, amp_delay, color=condition_colors.get(ck, "#444444"), lw=1.5)
        ax_amp.axvline(0.0, color='cyan', ls='-', lw=1.0, alpha=0.9)
        if cue_duration > 0:
            ax_amp.axvline(cue_duration, color='cyan', ls='--', lw=1.0, alpha=0.9)
        if noise_thr is not None:
            ax_amp.axhline(noise_thr, color='red', ls='--', lw=1.0,
                           label=f'Noise thr. ({noise_thr:.3f})')
            ax_amp.legend(fontsize=7, loc='upper right')
        ax_amp.set_xlabel("Time from cue onset (ms)")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.set_xlim(window_start - cue_start, delay_end - cue_start)
        ax_amp.grid(True, alpha=0.2)

    plt.suptitle(
        suptitle or "Ring Activity from Cue to End of Delay Across Conditions",
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
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = max(4 * ncols, figsize[0])
    fig_h = max(3.5 * nrows, figsize[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    for ax, w in zip(axes_flat, w_values):
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

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle(suptitle or "Noise Floor: Â_hat Distribution (No Stimulus)",
                 fontsize=13, fontweight="bold")
    

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

    keys = sorted(timecourse_data.keys())
    n = len(keys)

    # Widen figure when there are many curves so the legend fits
    if n > 8:
        figsize = (max(figsize[0], 10 + (n - 8) * 0.3), figsize[1])

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis

    for idx, (amp, w) in enumerate(keys):
        d = timecourse_data[(amp, w)]
        color = cmap(idx / max(n - 1, 1))
        sr = d.get('success_rate')
        sr_str = f", sr={sr:.0%}" if sr is not None else ""
        label = f"amp={amp:.0f}, w={w:.2f}{sr_str}"
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
    ncol = max(1, n // 10 + 1)
    ax.legend(fontsize=7, loc="best", ncol=ncol)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.suptitle(suptitle or "Â_hat Time Courses", fontsize=13, fontweight="bold")
    

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration_scatter(
    all_grid_data: dict,
    condition_colors: Optional[dict] = None,
    figsize: tuple[float, float] = (9, 6),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Scatter plot: mean Â_hat vs success rate, one color per condition.

    Parameters:
        all_grid_data: Dict mapping condition_key -> {(amplitude, w_inter): metrics_dict}.
            A single-condition dict can be passed as ``{"WT": grid_data}``.
        condition_colors: Color per condition key. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from ..study import STUDY_CONDITIONS

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    # Collect all amplitude values to map to marker sizes
    all_amps = sorted({amp for gd in all_grid_data.values() for (amp, _w) in gd})
    base_size = 60
    amp_sizes = {a: base_size + base_size * i for i, a in enumerate(all_amps)}

    fig, ax = plt.subplots(figsize=figsize)

    for cond_key, grid_data in all_grid_data.items():
        color = condition_colors.get(cond_key, "#666666")
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key

        x_vals, y_vals, s_vals = [], [], []
        for (amp, _w), d in sorted(grid_data.items()):
            x_vals.append(d.get("mean_A_hat", 0.0))
            y_vals.append(d.get("success_rate", 0.0))
            s_vals.append(amp_sizes.get(amp, base_size))

        ax.scatter(x_vals, y_vals, c=color, s=s_vals, label=label,
                   edgecolors="white", linewidth=0.5, alpha=0.85)

    # Condition legend
    cond_legend = ax.legend(title="Condition", fontsize=9, title_fontsize=10,
                            loc="upper left")

    # Amplitude size legend (only if multiple amplitude values)
    if len(all_amps) > 1:
        size_handles = [
            plt.scatter([], [], s=amp_sizes[a], c="gray",
                        edgecolors="white", linewidth=0.5, label=f"{a:.0f}")
            for a in all_amps
        ]
        ax.legend(handles=size_handles, title="Amplitude", fontsize=8,
                  title_fontsize=9, loc="lower right")
        ax.add_artist(cond_legend)

    ax.set_xlabel("Mean $\\hat{A}$ (pop. vector amplitude)")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Calibration Summary", fontsize=13, fontweight="bold")
    

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_noise_summary(
    noise_data: dict,
    condition_colors: Optional[dict] = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Line plot of noise threshold vs w_pyr_pyr_inter, one line per condition.

    Parameters:
        noise_data: Dict mapping condition_key -> {w_inter: threshold}.
        condition_colors: Color per condition key. Defaults to CONDITION_COLORS.
        figsize: Figure size.
        save_path: If provided, save figure.
        suptitle: Optional super-title.

    Returns:
        fig: Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from ..study import STUDY_CONDITIONS

    if condition_colors is None:
        condition_colors = CONDITION_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    for cond_key, thresholds in noise_data.items():
        color = condition_colors.get(cond_key, "#666666")
        label = STUDY_CONDITIONS[cond_key].name if cond_key in STUDY_CONDITIONS else cond_key
        ws = sorted(thresholds.keys())
        ys = [thresholds[w] for w in ws]
        ax.plot(ws, ys, color=color, marker="o", markersize=5, lw=2, label=label)

    ax.set_xlabel("$w_{pyr-pyr}^{inter}$")
    ax.set_ylabel("Noise threshold ($\\hat{A}$)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Condition", fontsize=9, title_fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle or "Noise Floor Summary", fontsize=13, fontweight="bold")
    

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
    """Grid of PYR activity heatmaps (cell index × time) for representative distractor-sweep cells.

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

        # Angles and node indices
        angles_deg = result.ring_params.node_angles_deg  # shape (n_nodes,)
        n_nodes = len(angles_deg)

        # Heatmap: time on x-axis (left=early, right=late), cell index on y-axis
        extent = [t_plot[0], t_plot[-1], 0, n_nodes - 1]
        im = ax.imshow(
            activity.T,
            aspect="auto",
            cmap="hot",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)

        # Cue/distractor windows as vertical lines in time
        ax.axvline(cue_on_rel, color="white", ls="--", lw=1.2, alpha=0.9)
        ax.axvline(cue_off_rel, color="white", ls="--", lw=1.2, alpha=0.9)
        ax.axvline(dist_on_rel, color="#E69F00", ls="--", lw=1.2, alpha=0.9)
        ax.axvline(dist_off_rel, color="#E69F00", ls="--", lw=1.2, alpha=0.9)

        # Cue/distractor positions as horizontal lines in cell index
        cue_idx = int(np.argmin(np.abs(((angles_deg - 180.0 + 180.0) % 360.0) - 180.0)))
        ax.axhline(cue_idx, color="white", ls=":", lw=1.0, alpha=0.7)
        dist_angle = (180.0 + off) % 360.0
        dist_idx = int(np.argmin(np.abs(((angles_deg - dist_angle + 180.0) % 360.0) - 180.0)))
        ax.axhline(dist_idx, color="#E69F00", ls=":", lw=1.0, alpha=0.7)

        # Decoded bump trajectory (cyan dots)
        center_deg, amplitude = decode_bump_center(result, population=0)
        t_full_rel = result.t_ms - burn_in_ms
        valid = (t_full_rel >= 0) & (amplitude > 0.2)
        center_node = (np.mod(center_deg, 360.0) / 360.0) * n_nodes
        ax.scatter(
            t_full_rel[valid], center_node[valid],
            c="cyan", s=1, alpha=0.5,
        )

        ax.set_title(f"Δφ={off:.0f}°,  {amp:.2g}× cue", fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Cell index", fontsize=8)
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_ylim(0, n_nodes - 1)
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
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distractor_sweep_node_differences(
    tc_data: list[dict],
    cue_onset_ms: float,
    cue_offset_ms: float,
    dist_onset_ms: float,
    dist_offset_ms: float,
    burn_in_ms: float = 10000.0,
    cue_center_deg: float = 180.0,
    figsize_per_panel: tuple[float, float] = (4.5, 3.0),
    save_paths: Optional[tuple] = None,
    suptitle_prefix: str = "",
):
    """Three figures showing pairwise firing-rate differences between key nodes.

    Produces one figure per difference type:
      1. Cue node − Distractor node
      2. End node − Distractor node
      3. Cue node − End node

    Each figure has one panel per offset condition. The legend in each panel
    shows the actual node angles involved in the subtraction.

    Parameters:
        tc_data: List of dicts with keys ``'offset_deg'``, ``'amp_factor'``,
            ``'full_result'``, ``'center_deg'``, ``'t_ms'``.
        cue_onset_ms, cue_offset_ms: Cue window (absolute ms).
        dist_onset_ms, dist_offset_ms: Distractor window (absolute ms).
        burn_in_ms: Burn-in duration; subtracted to align displayed time.
        cue_center_deg: Angular position of the cue (default 180°).
        figsize_per_panel: Width × height per subplot panel.
        save_paths: Optional tuple of 3 file paths
            (cue−dist, end−dist, cue−end).
        suptitle_prefix: String prepended to each figure title.

    Returns:
        (fig_cue_dist, fig_end_dist, fig_cue_end): three Matplotlib Figures.
    """
    import matplotlib.pyplot as plt

    n = len(tc_data)
    ncols = n if n <= 4 else min(5, n)
    nrows = int(np.ceil(n / ncols))
    pw, ph = figsize_per_panel

    cue_on_rel = (cue_onset_ms - burn_in_ms) / 1000.0
    cue_off_rel = (cue_offset_ms - burn_in_ms) / 1000.0
    dist_on_rel = (dist_onset_ms - burn_in_ms) / 1000.0
    dist_off_rel = (dist_offset_ms - burn_in_ms) / 1000.0

    # Pre-compute node indices and difference timecourses for all entries
    panels = []
    for entry in tc_data:
        result = entry['full_result']
        t_rel = result.t_ms - burn_in_ms
        mask = t_rel >= 0
        t_plot = t_rel[mask] / 1000.0
        activity = result.r[mask, :, 0]  # PYR only

        angles_deg = result.ring_params.node_angles_deg
        cue_idx = int(np.argmin(np.abs(((angles_deg - cue_center_deg + 180.0) % 360.0) - 180.0)))
        dist_deg = (cue_center_deg + entry['offset_deg']) % 360.0
        dist_idx = int(np.argmin(np.abs(((angles_deg - dist_deg + 180.0) % 360.0) - 180.0)))
        post_mask = entry['t_ms'] >= burn_in_ms
        final_center = float(entry['center_deg'][post_mask][-1])
        final_idx = int(np.argmin(np.abs(((angles_deg - final_center + 180.0) % 360.0) - 180.0)))

        panels.append({
            'off': entry['offset_deg'],
            'amp': entry['amp_factor'],
            't_plot': t_plot,
            'diff_cue_dist': activity[:, cue_idx]  - activity[:, dist_idx],
            'diff_dist_end': activity[:, dist_idx]  - activity[:, final_idx],
            'diff_cue_end':  activity[:, cue_idx]  - activity[:, final_idx],
            'cue_deg':   float(angles_deg[cue_idx]),
            'dist_deg':  float(angles_deg[dist_idx]),
            'final_deg': final_center,
        })

    configs = [
        ('diff_cue_dist', '#56B4E9', 'Cue − Distractor node firing rate'),
        ('diff_dist_end', '#009E73', 'Distractor − End node firing rate'),
        ('diff_cue_end',  '#CC79A7', 'Cue − End node firing rate'),
    ]

    figs = []
    for i, (diff_key, color, title) in enumerate(configs):
        max_abs = max(
            np.nanmax(np.abs(p[diff_key]))
            for p in panels
        )
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(pw * ncols, ph * nrows),
            squeeze=False,
            layout="constrained",
        )

        for idx, p in enumerate(panels):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            ax.axvspan(cue_on_rel, cue_off_rel, color="#b0b0b0", alpha=0.35)
            ax.axvspan(dist_on_rel, dist_off_rel, color="#E69F00", alpha=0.40)
            ax.axhline(0, color="k", lw=0.8, ls=":")

            if diff_key == 'diff_cue_dist':
                label = f"{p['cue_deg']:.0f}° − {p['dist_deg']:.0f}°"
            elif diff_key == 'diff_dist_end':
                label = f"{p['dist_deg']:.0f}° − {p['final_deg']:.0f}°"
            else:
                label = f"{p['cue_deg']:.0f}° − {p['final_deg']:.0f}°"

            ax.plot(p['t_plot'], p[diff_key], color=color, lw=1.5, label=label)
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title(f"Δφ={p['off']:.0f}°, {p['amp']:.2g}×", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("ΔFiring rate (Hz)", fontsize=8)
            ax.set_ylim(-max_abs, max_abs)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"{suptitle_prefix}{title}", fontsize=12, fontweight="bold")
        if save_paths and i < len(save_paths) and save_paths[i]:
            fig.savefig(save_paths[i], dpi=150, bbox_inches="tight")

        figs.append(fig)

    return tuple(figs)


def plot_distractor_sweep_node_timecourses(
    tc_data: list[dict],
    cue_onset_ms: float,
    cue_offset_ms: float,
    dist_onset_ms: float,
    dist_offset_ms: float,
    burn_in_ms: float = 10000.0,
    cue_center_deg: float = 180.0,
    figsize_per_panel: tuple[float, float] = (4.5, 3.0),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """Grid of firing-rate timecourses for the cue, distractor, and end nodes.

    Each panel shows three lines (cue node, distractor node, end node) for
    one (offset, amplitude) condition.
    """
    import matplotlib.pyplot as plt

    n = len(tc_data)
    ncols = n if n <= 4 else min(5, n)
    nrows = int(np.ceil(n / ncols))
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(pw * ncols, ph * nrows),
        squeeze=False,
        layout="constrained",
    )

    cue_on_rel = (cue_onset_ms - burn_in_ms) / 1000.0
    cue_off_rel = (cue_offset_ms - burn_in_ms) / 1000.0
    dist_on_rel = (dist_onset_ms - burn_in_ms) / 1000.0
    dist_off_rel = (dist_offset_ms - burn_in_ms) / 1000.0

    for idx, entry in enumerate(tc_data):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        result = entry['full_result']
        off = entry['offset_deg']
        amp = entry['amp_factor']

        t_rel = result.t_ms - burn_in_ms
        mask = t_rel >= 0
        t_plot = t_rel[mask] / 1000.0
        activity = result.r[mask, :, 0]

        angles_deg = result.ring_params.node_angles_deg
        cue_idx = int(np.argmin(np.abs(((angles_deg - cue_center_deg + 180.0) % 360.0) - 180.0)))
        dist_deg = (cue_center_deg + off) % 360.0
        dist_idx = int(np.argmin(np.abs(((angles_deg - dist_deg + 180.0) % 360.0) - 180.0)))
        post_mask = entry['t_ms'] >= burn_in_ms
        final_center = float(entry['center_deg'][post_mask][-1])
        final_idx = int(np.argmin(np.abs(((angles_deg - final_center + 180.0) % 360.0) - 180.0)))

        ax.axvspan(cue_on_rel, cue_off_rel, color="#b0b0b0", alpha=0.35)
        ax.axvspan(dist_on_rel, dist_off_rel, color="#E69F00", alpha=0.40)
        ax.plot(t_plot, activity[:, cue_idx],   color="#56B4E9", lw=1.5,
                label=f"Cue ({angles_deg[cue_idx]:.0f}°)")
        ax.plot(t_plot, activity[:, dist_idx],  color="#E69F00", lw=1.5,
                label=f"Dist ({angles_deg[dist_idx]:.0f}°)")
        ax.plot(t_plot, activity[:, final_idx], color="#009E73", lw=1.5, ls='--',
                label=f"End ({final_center:.0f}°)")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"Δφ={off:.0f}°, {amp:.2g}×", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Firing rate (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle or "Distractor Sweep — Node Activity",
                 fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# New plot functions for the 4-experiment battery
# ---------------------------------------------------------------------------

def plot_lesion_study(
    lesion_data: dict,
    noise_floor: float,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "Figure":
    """4×2 panel grid showing the effect of knockdown on formation and survival.

    Parameters:
        lesion_data: Nested dict
            {population_name: {knockdown_pct: dict from compute_lesion_metrics}}
        noise_floor: Used to annotate plots.
        save_path: Optional path to save the figure.
        suptitle: Optional super-title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    populations = list(lesion_data.keys())
    n_pops = len(populations)
    fig, axes = plt.subplots(n_pops, 2, figsize=(10, 3 * n_pops), squeeze=False)

    pop_colors = {
        'PYR_recurrence': '#D62728',
        'PV': '#1F77B4',
        'SOM': '#2CA02C',
        'VIP': '#FF7F0E',
    }

    for row, pop_name in enumerate(populations):
        kd_data = lesion_data[pop_name]
        kd_levels = sorted(kd_data.keys())
        formation_rates = [kd_data[k]['formation_rate'] * 100 for k in kd_levels]
        survival_means = [kd_data[k]['survival_time_mean_ms'] for k in kd_levels]
        survival_sems = [kd_data[k]['survival_time_sem_ms'] for k in kd_levels]

        color = pop_colors.get(pop_name, '#333333')

        # Left panel: formation success rate
        ax_form = axes[row, 0]
        ax_form.plot(kd_levels, formation_rates, 'o-', color=color, lw=2, markersize=6)
        ax_form.set_ylim(0, 105)
        ax_form.set_ylabel('Formation rate (%)', fontsize=9)
        ax_form.set_title(f'{pop_name} — formation', fontsize=10)
        ax_form.axhline(50, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax_form.set_xlabel('Knockdown (%)', fontsize=8)
        ax_form.tick_params(labelsize=8)
        ax_form.grid(True, alpha=0.25)
        ax_form.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

        # Right panel: bump survival time
        ax_surv = axes[row, 1]
        s_means = np.array(survival_means, dtype=float)
        s_sems = np.array(survival_sems, dtype=float)
        valid = ~np.isnan(s_means)
        kd_arr = np.array(kd_levels)
        ax_surv.plot(kd_arr[valid], s_means[valid], 'o-', color=color, lw=2, markersize=6)
        ax_surv.fill_between(
            kd_arr[valid],
            s_means[valid] - s_sems[valid],
            s_means[valid] + s_sems[valid],
            color=color, alpha=0.25,
        )
        ax_surv.set_ylabel('Survival time (ms)', fontsize=9)
        ax_surv.set_title(f'{pop_name} — survival', fontsize=10)
        ax_surv.set_xlabel('Knockdown (%)', fontsize=8)
        ax_surv.tick_params(labelsize=8)
        ax_surv.grid(True, alpha=0.25)
        ax_surv.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

    plt.suptitle(
        suptitle or 'Population Lesion Study: Formation & Survival',
        fontsize=13, fontweight='bold',
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_tau_adapt_sweep(
    sweep_metrics: dict,
    tau_values_ms: list,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "Figure":
    """3-panel figure with shared log x-axis (τ_adapt_pyr in ms).

    Parameters:
        sweep_metrics: {tau_ms: dict from extract_tau_sweep_metrics}
        tau_values_ms: List of τ values tested (for x-axis ordering).
        save_path: Optional path to save the figure.
        suptitle: Optional super-title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    tau_arr = np.array(sorted(sweep_metrics.keys()), dtype=float)

    survival_mean = np.array([sweep_metrics[t]['survival_time_mean_ms'] for t in tau_arr])
    survival_sem  = np.array([sweep_metrics[t]['survival_time_sem_ms'] for t in tau_arr])
    diffusion     = np.array([sweep_metrics[t]['diffusion_deg2_per_s'] for t in tau_arr])
    osc_freq      = np.array([
        sweep_metrics[t]['osc_freq_hz'] if sweep_metrics[t]['osc_freq_hz'] is not None else np.nan
        for t in tau_arr
    ])

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    # Panel 0: survival time
    ax = axes[0]
    valid = ~np.isnan(survival_mean)
    ax.errorbar(tau_arr[valid], survival_mean[valid], yerr=survival_sem[valid],
                fmt='o-', color='#1F77B4', lw=2, markersize=6, capsize=4)
    ax.set_ylabel('Survival time (ms)', fontsize=10)
    ax.set_title('Bump survival during delay', fontsize=10)
    ax.grid(True, alpha=0.25)

    # Panel 1: diffusion coefficient
    ax = axes[1]
    valid = ~np.isnan(diffusion)
    ax.plot(tau_arr[valid], diffusion[valid], 's-', color='#D62728', lw=2, markersize=6)
    ax.set_ylabel('Diffusion D (deg²/s)', fontsize=10)
    ax.set_title('Bump diffusion coefficient', fontsize=10)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.25)

    # Panel 2: oscillation frequency
    ax = axes[2]
    valid = ~np.isnan(osc_freq)
    ax.scatter(tau_arr[valid], osc_freq[valid], color='#2CA02C', s=60, zorder=3)
    ax.plot(tau_arr[valid], osc_freq[valid], '-', color='#2CA02C', lw=1.5, alpha=0.7)
    ax.set_ylabel('Oscillation freq (Hz)', fontsize=10)
    ax.set_title('Bump amplitude oscillation', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('τ_adapt_pyr (ms)', fontsize=10)

    for ax in axes:
        ax.set_xscale('log')
        xticks = [t for t in [50, 100, 200, 400, 600, 1000, 2000] if t >= tau_arr.min() - 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.tick_params(labelsize=8)

    plt.suptitle(
        suptitle or 'τ_adapt Sweep: Survival, Diffusion & Oscillation',
        fontsize=12, fontweight='bold',
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_phase_plane(
    phase_data: dict,
    conditions: list,
    operating_points: Optional[dict] = None,
    population_names: Optional[list] = None,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "Figure":
    """4-condition × 4-population grid of bifurcation S-curves.

    Parameters:
        phase_data: {condition_key: dict from run_phase_plane_sweep}
        conditions: List of condition keys (rows of the grid).
        operating_points: Dict with keys 'spontaneous' and/or 'cue',
            each mapping to a delta_I value (float). These are drawn as
            vertical dashed lines on all panels.
        population_names: List of 4 population names (default: PYR,SOM,PV,VIP).
        save_path: Optional path to save the figure.
        suptitle: Optional super-title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if population_names is None:
        population_names = ['PYR', 'SOM', 'PV', 'VIP']

    n_conds = len(conditions)
    n_pops = 4
    fig, axes = plt.subplots(
        n_conds, n_pops,
        figsize=(4 * n_pops, 3 * n_conds),
        squeeze=False,
    )

    op_styles = {
        'spontaneous': dict(color='gray',   ls='--', lw=1.2, label='Spontaneous/delay'),
        'cue':         dict(color='crimson', ls=':',  lw=1.5, label='Cue'),
    }

    for row, cond in enumerate(conditions):
        if cond not in phase_data:
            for col in range(n_pops):
                axes[row, col].set_visible(False)
            continue
        data = phase_data[cond]
        delta_I = data['delta_I']
        up_rates   = data['up_rates']    # (n_steps, 4)
        down_rates = data['down_rates']  # (n_steps, 4)
        bistable   = data['bistable_mask']  # bool (n_steps,)

        for col, pop_name in enumerate(population_names):
            ax = axes[row, col]

            # Grey shading for bistable region
            if np.any(bistable):
                for k in range(len(delta_I) - 1):
                    if bistable[k]:
                        ax.axvspan(delta_I[k], delta_I[k + 1],
                                   color='lightgray', alpha=0.5, zorder=0)

            ax.plot(delta_I, up_rates[:, col], color='#1F77B4', lw=2, label='UP')
            ax.plot(delta_I, down_rates[:, col], color='#D62728', lw=1.5,
                    ls='--', label='DOWN')

            # Operating point vertical lines
            if operating_points:
                for op_name, dI_val in operating_points.items():
                    style = op_styles.get(op_name, dict(color='black', ls=':', lw=1))
                    ax.axvline(dI_val, **{k: v for k, v in style.items() if k != 'label'},
                               zorder=5)

            ax.set_xlabel('ΔI_PYR (pA)', fontsize=8)
            ax.set_ylabel('Rate (Hz)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(pop_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{cond}\nRate (Hz)', fontsize=8)

            # Legend only on first panel
            if row == 0 and col == 0:
                handles = [
                    mpatches.Patch(color='#1F77B4', label='UP sweep'),
                    mpatches.Patch(color='#D62728', label='DOWN sweep'),
                    mpatches.Patch(color='lightgray', label='Bistable region'),
                ]
                if operating_points:
                    for op_name, dI_val in operating_points.items():
                        style = op_styles.get(op_name, {})
                        handles.append(plt.Line2D(
                            [], [], color=style.get('color', 'k'),
                            ls=style.get('ls', '--'), lw=1.5,
                            label=style.get('label', op_name),
                        ))
                ax.legend(handles=handles, fontsize=7, loc='upper left')

    plt.suptitle(
        suptitle or 'Phase Plane Analysis: Bifurcation Fingerprinting',
        fontsize=12, fontweight='bold',
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_temporal_dissection(
    result: "RingSimulationResult",
    t_offset: float = 0.0,
    time_range: Optional[tuple] = None,
    noise_floor: Optional[float] = None,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> "Figure":
    """3-location × (4 populations + adaptation current) time-course grid.

    Layout: 5 rows × 3 columns = 15 subplots.
    Rows:    0=PYR, 1=SOM, 2=PV, 3=VIP (firing rates, Hz)
             4=PYR adaptation current (requires result.I_adapt_stored)
    Columns: center node (at stimulus), +90°, antipodal (+180°)

    Vertical markers:
        - Cue onset: red dashed
        - Cue offset: red solid
        - Collapse time (auto-detected, blue dotted) — optional

    Parameters:
        result: RingSimulationResult (I_adapt_stored required for row 4).
        t_offset: Absolute time offset to subtract for display (ms).
        time_range: Optional (t_start_ms, t_end_ms) for zooming.
        noise_floor: Threshold for auto-detecting collapse time.
        save_path: Optional path to save the figure.
        suptitle: Optional super-title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from .analysis import compute_bump_survival_time

    n_nodes = result.n_nodes
    stim_node = result.stim_node
    node_90  = (stim_node + n_nodes // 4) % n_nodes
    node_180 = (stim_node + n_nodes // 2) % n_nodes
    nodes = [stim_node, node_90, node_180]
    node_labels = ['Center (0°)', '+90°', 'Antipodal (+180°)']

    pop_names = ['PYR', 'SOM', 'PV', 'VIP']
    pop_colors = ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E']

    t = result.t_ms - t_offset

    # Time mask — time_range is in absolute ms (matches result.t_ms)
    if time_range is not None:
        tmask = (result.t_ms >= time_range[0]) & (result.t_ms <= time_range[1])
    else:
        tmask = np.ones(len(result.t_ms), dtype=bool)

    t_plot = t[tmask]

    has_adapt = result.I_adapt_stored is not None
    n_rows = 5 if has_adapt else 4
    row_labels = pop_names + (['I_adapt PYR'] if has_adapt else [])
    row_colors = pop_colors + (['#9467BD'] if has_adapt else [])

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 2.5 * n_rows),
        squeeze=False,
        sharex=True,
    )

    # Vertical marker times (relative to t_offset)
    cue_onset  = result.stim_window[0] - t_offset
    cue_offset = result.stim_window[1] - t_offset

    # Auto-detect collapse time
    collapse_time_rel = None
    if noise_floor is not None:
        st = compute_bump_survival_time(result, noise_floor)
        if st is not None:
            collapse_time_rel = cue_offset + st  # relative time

    for row in range(n_rows):
        for col, (node, node_label) in enumerate(zip(nodes, node_labels)):
            ax = axes[row, col]

            if row < 4:
                # Firing rate
                trace = result.r[tmask, node, row]
                color = row_colors[row]
                ax.plot(t_plot, trace, color=color, lw=1.2)
                ax.set_ylabel(f'{row_labels[row]} (Hz)', fontsize=8)
            else:
                # Adaptation current
                if has_adapt:
                    trace = result.I_adapt_stored[tmask, node, 0]
                    ax.plot(t_plot, trace, color=row_colors[row], lw=1.2)
                    ax.set_ylabel('I_adapt (AU)', fontsize=8)

            # Vertical markers
            ax.axvline(cue_onset,  color='red', ls='--', lw=1.2, alpha=0.8)
            ax.axvline(cue_offset, color='red', ls='-',  lw=1.5, alpha=0.8)
            if collapse_time_rel is not None:
                ax.axvline(collapse_time_rel, color='blue', ls=':', lw=1.5, alpha=0.9)

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(node_label, fontsize=10, fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel('Time (ms)', fontsize=8)

    plt.suptitle(
        suptitle or 'Temporal Dissection: Population Time Courses',
        fontsize=12, fontweight='bold',
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
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


# ============================================================================
# ASYMMETRY EXPERIMENT PLOTS
# ============================================================================

def _asym_cmap_norm():
    """Return (cmap, norm) for the blue→black→yellow asymmetry colormap."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "asymmetry", ["#0072B2", "#000000", "#F0E442"]
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    return cmap, norm


def plot_asymmetry_distribution(
    data_by_condition: dict,
    condition_order: list,
    save_path: Optional[str] = None,
    title_suffix: str = "",
    stats_by_condition: Optional[dict] = None,
):
    """Plot distribution of pre-cue and delay asymmetry per condition.

    Violin + jittered strip plots, one panel per condition. Points are
    colored blue→black→yellow by asymmetry value.

    Parameters:
        data_by_condition: {cond_key: {'pre_cue': np.ndarray, 'delay': np.ndarray}}
        condition_order: Condition keys to display (left→right)
        save_path: If provided, save figure here

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from ..study import STUDY_CONDITIONS

    conds = [k for k in condition_order if k in data_by_condition]
    n_conds = len(conds)
    if n_conds == 0:
        return None

    cmap, norm = _asym_cmap_norm()
    rng = np.random.default_rng(0)

    # Symmetric y-scale around 0 so distributions stay visually centered.
    # Keep a small headroom so the most extreme points nearly touch bounds.
    all_vals = np.concatenate([
        np.concatenate([
            np.asarray(data_by_condition[k]['pre_cue'], dtype=float),
            np.asarray(data_by_condition[k]['delay'], dtype=float),
        ])
        for k in conds
    ])
    y_lim = float(np.max(np.abs(all_vals))) * 1.05
    y_lim = max(y_lim, 0.1)

    fig, axes = plt.subplots(1, n_conds, figsize=(4.5 * n_conds, 5.5), sharey=True)
    if n_conds == 1:
        axes = [axes]

    groups = [('pre_cue', 'Pre-cue'), ('delay', 'Delay')]

    for ax, cond_key in zip(axes, conds):
        d = data_by_condition[cond_key]
        cname = STUDY_CONDITIONS[cond_key].name
        ccolor = CONDITION_COLORS.get(cond_key, '#888888')

        for xi, (key, label) in enumerate(groups):
            vals = np.asarray(d[key], dtype=float)

            # Violin
            vp = ax.violinplot(vals, positions=[xi], showmedians=True,
                               widths=0.55, showextrema=False)
            for body in vp['bodies']:
                body.set_facecolor(ccolor)
                body.set_alpha(0.35)
                body.set_edgecolor('none')
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(2.0)

            # Jittered strip, colored by asymmetry value
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                xi + jitter, vals,
                c=vals, cmap=cmap, norm=norm,
                s=12, alpha=0.65, linewidths=0, zorder=3,
            )

            # Mean and variance annotation below each group
            mean_val = float(np.mean(vals))
            var_val = float(np.var(vals, ddof=1))
            annot = f"μ={mean_val:+.3f}\nσ²={var_val:.4f}"
            if stats_by_condition and cond_key in stats_by_condition:
                cond_s = stats_by_condition[cond_key]
                # support both nested {period: {...}} and legacy flat structure
                s = cond_s.get(key) if isinstance(cond_s.get(key), dict) else (
                    cond_s if key == 'delay' else None)
                if s is not None:
                    p_use = s.get('p_w') if s.get('p_w') is not None else s.get('p_t')

                    def _stars(p):
                        if p is None: return ''
                        if p < 0.001: return '***'
                        if p < 0.01:  return '**'
                        if p < 0.05:  return '*'
                        return 'n.s.'

                    annot += f"\np={s['p_t']:.3f} {_stars(p_use)}"
            # Put stats text below the axis in axes coordinates so it does not
            # alter y-limits or push data upward.
            ax.text(
                xi, -0.22,
                annot,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7.5,
                color='#333333', style='italic',
                clip_on=False,
            )

        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g[1] for g in groups], fontsize=10)
        ax.set_title(cname, fontsize=11, fontweight='bold', color=ccolor)
        ax.set_ylim(-y_lim, y_lim)
        if ax is axes[0]:
            ax.set_ylabel("Asymmetry index (right − left)", fontsize=10)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], fraction=0.04, pad=0.02,
                 label='← left        right →')

    fig.suptitle(f"L/R Asymmetry Distribution{title_suffix}", fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_asymmetry_correlation(
    data_by_condition: dict,
    condition_order: list,
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    """Scatter plot of pre-cue vs delay asymmetry per condition.

    One panel per condition. Points colored by delay asymmetry value
    (blue→black→yellow). Pearson r is annotated.

    Parameters:
        data_by_condition: {cond_key: {'pre_cue': np.ndarray, 'delay': np.ndarray}}
        condition_order: Condition keys to display
        save_path: If provided, save figure here

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from ..study import STUDY_CONDITIONS

    conds = [k for k in condition_order if k in data_by_condition]
    n_conds = len(conds)
    if n_conds == 0:
        return None

    cmap, norm = _asym_cmap_norm()

    # Two rows: top = mean pre-cue, bottom = last time-step before cue
    has_last = all('last_pre_cue' in data_by_condition[k] and
                   not np.all(np.isnan(data_by_condition[k]['last_pre_cue']))
                   for k in conds)
    n_rows = 2 if has_last else 1
    row_keys   = ['pre_cue', 'last_pre_cue'] if has_last else ['pre_cue']
    row_labels = ['Mean pre-cue A (500 ms window)', 'Last A before cue (instantaneous)'] \
                 if has_last else ['Mean pre-cue A (500 ms window)']

    fig, axes_2d = plt.subplots(n_rows, n_conds,
                                figsize=(4.5 * n_conds, 4.5 * n_rows),
                                squeeze=False)

    def _scatter_panel(ax, pre, delay, cname, xlabel, cond_key, first_col):
        valid = ~(np.isnan(pre) | np.isnan(delay))
        pre_v, delay_v = pre[valid], delay[valid]
        if len(pre_v) == 0:
            ax.set_visible(False)
            return
        lim = float(np.max(np.abs(np.concatenate([pre_v, delay_v])))) * 1.05
        lim = max(lim, 0.01)
        ax.scatter(pre_v, delay_v, c=delay_v, cmap=cmap, norm=norm,
                   s=25, alpha=0.7, linewidths=0)
        if len(pre_v) > 2:
            try:
                from scipy.stats import pearsonr
                r, p = pearsonr(pre_v, delay_v)
                star = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "ns")
                ax.text(0.05, 0.95, f"r = {r:.2f} {star}",
                        transform=ax.transAxes, fontsize=9, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            except ImportError:
                pass
        ax.plot([-lim, lim], [-lim, lim], color='gray', ls='--', lw=0.8, alpha=0.5, zorder=0)
        ax.axhline(0, color='gray', ls=':', lw=0.6, alpha=0.5)
        ax.axvline(0, color='gray', ls=':', lw=0.6, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlabel, fontsize=9)
        if first_col:
            ax.set_ylabel("Delay asymmetry", fontsize=10)
        ax.set_title(cname, fontsize=11, fontweight='bold',
                     color=CONDITION_COLORS.get(cond_key, 'black'))
        ax.set_aspect('equal')

    for row_i, (rkey, rlabel) in enumerate(zip(row_keys, row_labels)):
        for col_i, cond_key in enumerate(conds):
            d = data_by_condition[cond_key]
            _scatter_panel(
                ax=axes_2d[row_i, col_i],
                pre=np.asarray(d[rkey], dtype=float),
                delay=np.asarray(d['delay'], dtype=float),
                cname=STUDY_CONDITIONS[cond_key].name,
                xlabel=rlabel,
                cond_key=cond_key,
                first_col=(col_i == 0),
            )

    # Shared colorbar on the last column of last row
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_2d[-1, -1], fraction=0.04, pad=0.02,
                 label='← left        right →')

    fig.suptitle(f"Pre-cue vs Delay Asymmetry{title_suffix}", fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _pairwise_bracket_stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n.s.'
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def _draw_pairwise_brackets(ax, conds, pairwise_stats, period_key, base_ylim):
    """Draw all pairwise significance brackets on a bar-chart axes.

    Parameters
    ----------
    ax : Axes
    conds : list[str]   condition keys in x-axis order
    pairwise_stats : list[dict]  each dict has 'period', 'cond_a', 'cond_b', 'p_u'
    period_key : str  value of 'period' to filter on
    base_ylim : float  top of the bar region (used to anchor brackets above bars)
    """
    if not pairwise_stats:
        return
    pairs = [
        (pw['cond_a'], pw['cond_b'], pw['p_u'])
        for pw in pairwise_stats
        if pw.get('period') == period_key
        and pw['cond_a'] in conds and pw['cond_b'] in conds
    ]
    pairs.sort(key=lambda t: abs(conds.index(t[1]) - conds.index(t[0])))
    bracket_unit = base_ylim * 0.10
    occupied: dict = {}
    max_level = 0
    for ca, cb, p_val in pairs:
        xi_a = conds.index(ca)
        xi_b = conds.index(cb)
        lo, hi = min(xi_a, xi_b), max(xi_a, xi_b)
        level = max((occupied.get(xi, 0) for xi in range(lo, hi + 1)), default=0) + 1
        for xi in range(lo, hi + 1):
            occupied[xi] = max(occupied.get(xi, 0), level)
        y_bot = base_ylim * 1.02 + (level - 1) * bracket_unit * 1.5
        color = 'black' if p_val < 0.05 else '#999999'
        label = _pairwise_bracket_stars(p_val)
        ax.plot([lo, lo, hi, hi], [y_bot, y_bot + bracket_unit * 0.6,
                                    y_bot + bracket_unit * 0.6, y_bot],
                lw=0.9, c=color, clip_on=False)
        ax.text((lo + hi) / 2, y_bot + bracket_unit * 0.6, label,
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color=color, clip_on=False)
        max_level = max(max_level, level)
    if max_level > 0:
        new_top = base_ylim * 1.02 + max_level * bracket_unit * 1.5 + bracket_unit
        ax.set_ylim(0, max(ax.get_ylim()[1], new_top))


def _new_metric_bar(ax, conds, data_by_condition, key, title, ylabel,
                    pairwise_stats=None):
    """Bar chart ± SEM for a per-trial scalar metric stored in data_by_condition.

    If *pairwise_stats* is provided and contains entries whose 'period' matches
    *key*, significance brackets are drawn above the bars.
    """
    from ..study import STUDY_CONDITIONS
    x = np.arange(len(conds))
    labels = [STUDY_CONDITIONS[k].name for k in conds]
    vals = np.array([
        np.nanmean(data_by_condition[k].get(key, np.array([np.nan]))) for k in conds
    ])
    sems = np.array([
        np.nanstd(data_by_condition[k].get(key, np.array([np.nan])), ddof=1)
        / np.sqrt(np.sum(~np.isnan(data_by_condition[k].get(key, np.array([np.nan])))))
        if np.sum(~np.isnan(data_by_condition[k].get(key, np.array([np.nan])))) > 1 else 0.0
        for k in conds
    ])
    colors = [CONDITION_COLORS.get(k, '#888888') for k in conds]
    ax.bar(x, vals, yerr=sems, capsize=5, color=colors,
           edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    base = max(float(np.nanmax(vals + sems)) * 1.6, 1e-6)
    ax.set_ylim(0, base)
    if pairwise_stats:
        _draw_pairwise_brackets(ax, conds, pairwise_stats, key, base)
    legend_pw = "pairwise (MWU):\n* p<0.05\n** p<0.01\n*** p<0.001\nn.s. = not significant"
    ax.text(0.98, 0.98, legend_pw, transform=ax.transAxes,
            ha='right', va='top', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))


def plot_asymmetry_summary(
    data_by_condition: dict,
    condition_order: list,
    save_path: Optional[str] = None,
    title_suffix: str = "",
    stats_by_condition: Optional[dict] = None,
    pairwise_stats: Optional[list] = None,
):
    """Summary bar chart of asymmetry statistics across conditions.

    Three panels:
    1. Mean asymmetry ± SEM for both pre-cue and delay, annotated with vs-0 stars
    2. Mean |asymmetry| ± SEM — Delay period, with all pairwise brackets
    3. Mean |asymmetry| ± SEM — Pre-cue period, with all pairwise brackets

    Parameters:
        data_by_condition: {cond_key: {'pre_cue': np.ndarray, 'delay': np.ndarray}}
        condition_order: Condition keys to display
        save_path: If provided, save figure here
        stats_by_condition: {cond_key: {'pre_cue': {...}, 'delay': {...}}} one-sample stats
        pairwise_stats: List of {period, cond_a, cond_b, p_u, ...} dicts

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ..study import STUDY_CONDITIONS

    conds = [k for k in condition_order if k in data_by_condition]
    n_conds = len(conds)
    if n_conds == 0:
        return None

    cmap, norm = _asym_cmap_norm()
    x = np.arange(n_conds)
    labels = [STUDY_CONDITIONS[k].name for k in conds]

    periods = [('pre_cue', 'Pre-cue'), ('delay', 'Delay')]
    bar_w = 0.35
    offsets = np.array([-bar_w / 2, bar_w / 2])
    period_alphas = [0.55, 1.0]
    period_hatches = ['///', None]

    def _stars(p):
        return _pairwise_bracket_stars(p)

    def _add_all_pairwise_brackets(ax, period_key, base_ylim):
        _draw_pairwise_brackets(ax, conds, pairwise_stats, period_key, base_ylim)

    fig, axes_2d = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes_2d[0]           # top row: existing three panels
    axes_new = axes_2d[1]       # bottom row: two new temporal panels + one empty

    # --- Panel 1: mean asymmetry ± SEM (both periods, both annotated vs 0) ---
    ax = axes[0]
    means_by_period: dict = {}
    sems_by_period: dict = {}
    all_peak_tops = []
    for pi, (pkey, _) in enumerate(periods):
        means = np.array([np.mean(data_by_condition[k][pkey]) for k in conds])
        sems = np.array([
            np.std(data_by_condition[k][pkey], ddof=1)
            / np.sqrt(len(data_by_condition[k][pkey]))
            for k in conds
        ])
        means_by_period[pkey] = means
        sems_by_period[pkey] = sems
        all_peak_tops.append((np.abs(means) + sems).max())
        ax.bar(x + offsets[pi], means, width=bar_w, yerr=sems, capsize=4,
               color=[cmap(norm(m)) for m in means],
               edgecolor='black', linewidth=0.8,
               alpha=period_alphas[pi], hatch=period_hatches[pi])

    ax.axhline(0, color='black', ls='-', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Mean asymmetry ± SEM")
    ax.set_title("Average asymmetry\n(0 = perfectly balanced)")
    ylim1 = max(max(all_peak_tops) * 2.5, 0.005)
    ax.set_ylim(-ylim1, ylim1)

    # Stars vs 0 for both periods
    if stats_by_condition:
        for pi, (pkey, _) in enumerate(periods):
            m = means_by_period[pkey]
            s = sems_by_period[pkey]
            for xi, ck in enumerate(conds):
                cond_s = stats_by_condition.get(ck, {})
                # support nested {period: {...}} structure
                ps = cond_s.get(pkey) if isinstance(cond_s.get(pkey), dict) else (
                    cond_s if pkey == 'delay' else None)
                if ps is None:
                    continue
                p_use = ps.get('p_w') if ps.get('p_w') is not None else ps.get('p_t')
                bar_top = abs(m[xi]) + s[xi] + ylim1 * 0.04
                y_star = np.sign(m[xi] + 1e-12) * bar_top
                ax.text(xi + offsets[pi], y_star, _stars(p_use),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    legend_txt = "vs. 0:\n*** p<0.001\n ** p<0.01\n  * p<0.05\nn.s. p≥0.05"
    ax.text(0.98, 0.98, legend_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    # Pre-cue / Delay legend
    legend_patches = [
        Patch(fc='gray', alpha=0.55, hatch='///', ec='black', label='Pre-cue'),
        Patch(fc='gray', alpha=1.0, ec='black', label='Delay'),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc='upper left', frameon=True)

    # --- Panel 2: |asymmetry| Delay with all pairwise brackets ---
    ax2 = axes[1]
    abs_d = np.array([np.mean(np.abs(data_by_condition[k]['delay'])) for k in conds])
    sem_d = np.array([
        np.std(np.abs(data_by_condition[k]['delay']), ddof=1)
        / np.sqrt(len(data_by_condition[k]['delay']))
        for k in conds
    ])
    ax2.bar(x, abs_d, yerr=sem_d, capsize=5,
            color=[CONDITION_COLORS.get(k, '#888888') for k in conds],
            alpha=1.0, edgecolor='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel("Mean |asymmetry| ± SEM")
    ax2.set_title("Magnitude — Delay")
    base_d = max((abs_d + sem_d).max() * 1.6, 0.005)
    ax2.set_ylim(0, base_d)
    _add_all_pairwise_brackets(ax2, 'delay', base_d)

    legend_pw = "pairwise (MWU):\n* p<0.05\n** p<0.01\n*** p<0.001\nn.s. = not significant"
    ax2.text(0.98, 0.98, legend_pw, transform=ax2.transAxes,
             ha='right', va='top', fontsize=7, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    # --- Panel 3: |asymmetry| Pre-cue with all pairwise brackets ---
    ax3 = axes[2]
    abs_p = np.array([np.mean(np.abs(data_by_condition[k]['pre_cue'])) for k in conds])
    sem_p = np.array([
        np.std(np.abs(data_by_condition[k]['pre_cue']), ddof=1)
        / np.sqrt(len(data_by_condition[k]['pre_cue']))
        for k in conds
    ])
    ax3.bar(x, abs_p, yerr=sem_p, capsize=5,
            color=[CONDITION_COLORS.get(k, '#888888') for k in conds],
            alpha=0.55, edgecolor='black', linewidth=0.8, hatch='///')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax3.set_ylabel("Mean |asymmetry| ± SEM")
    ax3.set_title("Magnitude — Pre-cue")
    base_p = max((abs_p + sem_p).max() * 1.6, 0.005)
    ax3.set_ylim(0, base_p)
    _add_all_pairwise_brackets(ax3, 'pre_cue', base_p)
    ax3.text(0.98, 0.98, legend_pw, transform=ax3.transAxes,
             ha='right', va='top', fontsize=7, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    # --- Panel 4: Mean |A(t)| per trial (temporal, does not cancel) ---
    ax4 = axes_new[0]
    _new_metric_bar(ax4, conds, data_by_condition, 'mean_abs_asym',
                    "Mean |A(t)| — Delay\n(magnitude, does not cancel)",
                    "Mean |A(t)| ± SEM",
                    pairwise_stats=pairwise_stats)

    # --- Panel 5: Std of A(t) (amplitude + side variability) ---
    ax5 = axes_new[1]
    _new_metric_bar(ax5, conds, data_by_condition, 'asym_std',
                    "Std(A(t)) — Delay\n(amplitude + side variability)",
                    "Std(A(t)) ± SEM",
                    pairwise_stats=pairwise_stats)

    axes_new[2].set_visible(False)

    fig.suptitle(f"L/R Asymmetry Summary{title_suffix}", fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
