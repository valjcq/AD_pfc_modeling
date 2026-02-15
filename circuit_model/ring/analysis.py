"""
Analysis functions for the ring attractor network.

This module contains:
- Population vector decoding for bump center estimation
- Bump width estimation
- Drift and diffusion analysis
- Comprehensive bump metrics computation
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .simulation import RingSimulationResult
from .connectivity import angular_distance


def population_vector_decode(
    activity: np.ndarray,
    node_angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode bump center using population vector (center of mass) method.

    For each time step, computes the circular mean of activity weighted
    by node positions.

    Parameters:
        activity: Firing rates, shape (n_steps, n_nodes) or (n_nodes,)
        node_angles_rad: Node positions in radians, shape (n_nodes,)

    Returns:
        center_rad: Decoded bump center (radians), shape (n_steps,) or scalar
        amplitude: Vector length (confidence measure), shape (n_steps,) or scalar
    """
    # Handle single time point
    squeeze_output = False
    if activity.ndim == 1:
        activity = activity[np.newaxis, :]
        squeeze_output = True

    # Convert to complex representation on unit circle
    z = np.exp(1j * node_angles_rad)  # Shape: (n_nodes,)

    # Weighted sum of unit vectors
    weighted_z = activity @ z  # Shape: (n_steps,)

    # Normalize by total activity
    total_activity = np.sum(activity, axis=1)  # Shape: (n_steps,)
    total_activity = np.maximum(total_activity, 1e-10)  # Avoid division by zero

    normalized_z = weighted_z / total_activity

    # Extract angle and amplitude
    center_rad = np.angle(normalized_z)  # Range: [-pi, pi]
    center_rad = np.mod(center_rad, 2 * np.pi)  # Convert to [0, 2pi)

    amplitude = np.abs(normalized_z)  # Range: [0, 1], 1 = perfect bump

    if squeeze_output:
        return center_rad[0], amplitude[0]

    return center_rad, amplitude


def decode_bump_center(
    result: RingSimulationResult,
    population: int = 0,  # Default: PYR
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode bump center from simulation result.

    Parameters:
        result: RingSimulationResult
        population: Which population to decode (0=PYR, 1=SOM, 2=PV, 3=VIP)

    Returns:
        center_deg: Decoded bump center (degrees), shape (n_steps,)
        amplitude: Decoding confidence, shape (n_steps,)
    """
    activity = result.r[:, :, population]
    node_angles = result.ring_params.node_angles_rad

    center_rad, amplitude = population_vector_decode(activity, node_angles)
    center_deg = center_rad * 180 / np.pi

    return center_deg, amplitude


def estimate_bump_width(
    activity: np.ndarray,
    node_angles_rad: np.ndarray,
    center_rad: Optional[float] = None,
) -> float:
    """
    Estimate bump width using circular standard deviation.

    Parameters:
        activity: Firing rates at a single time, shape (n_nodes,)
        node_angles_rad: Node positions in radians
        center_rad: Bump center (radians), computed if None

    Returns:
        width_deg: Bump width in degrees (circular std dev)
    """
    if center_rad is None:
        center_rad, _ = population_vector_decode(activity, node_angles_rad)

    # Weighted circular variance
    total = np.sum(activity)
    if total < 1e-10:
        return 0.0

    # Angular deviation from center
    dev = angular_distance(node_angles_rad, center_rad)

    # Circular variance using second moment
    # R = resultant length = sqrt(mean(cos)^2 + mean(sin)^2)
    cos_dev = np.cos(dev)
    sin_dev = np.sin(dev)
    mean_cos = np.sum(activity * cos_dev) / total
    mean_sin = np.sum(activity * sin_dev) / total
    R = np.sqrt(mean_cos**2 + mean_sin**2)

    # Convert to standard deviation
    # R = 1 means perfectly peaked, R = 0 means uniform
    # Circular variance = 1 - R
    # Circular std = sqrt(-2 * log(R)) for von Mises approximation
    if R > 0.99:
        R = 0.99  # Cap to avoid log(0)
    circular_variance = 1 - R
    if circular_variance >= 1:
        return 180.0  # Uniform distribution

    width_rad = np.sqrt(-2 * np.log(1 - circular_variance))
    width_deg = width_rad * 180 / np.pi

    return min(width_deg, 180.0)  # Cap at 180 degrees


def angular_distance_deg(angle1: float, angle2: float) -> float:
    """Angular distance in degrees, handling wraparound."""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def compute_bump_metrics(
    result: RingSimulationResult,
    time_window: Optional[tuple[float, float]] = None,
    population: int = 0,
) -> dict:
    """
    Compute comprehensive bump metrics from simulation.

    Parameters:
        result: RingSimulationResult
        time_window: (start_ms, end_ms) for analysis (default: delay period)
        population: Which population to analyze (0=PYR)

    Returns:
        Dictionary with metrics:
        - 'center_mean_deg': Mean bump center (degrees)
        - 'center_std_deg': Std of bump center (circular)
        - 'amplitude_mean': Mean decoding confidence
        - 'width_mean_deg': Mean bump width (degrees)
        - 'drift_rate_deg_per_s': Systematic drift (degrees/second)
        - 'diffusion_deg2_per_s': Diffusion coefficient (degrees^2/second)
        - 'error_from_cue_deg': Angular error from stimulus location
    """
    center_deg, amplitude = decode_bump_center(result, population)

    # Select time window
    if time_window is None:
        # Default: delay period (after stimulus offset)
        t_start = result.stim_window[1] + 100  # 100ms after stim
        t_end = result.t_ms[-1]
    else:
        t_start, t_end = time_window

    mask = (result.t_ms >= t_start) & (result.t_ms <= t_end)
    if not np.any(mask):
        return {
            "center_mean_deg": np.nan,
            "center_std_deg": np.nan,
            "amplitude_mean": np.nan,
            "width_mean_deg": np.nan,
            "drift_rate_deg_per_s": np.nan,
            "diffusion_deg2_per_s": np.nan,
            "error_from_cue_deg": np.nan,
        }

    t_window = result.t_ms[mask]
    center_window = center_deg[mask]
    amp_window = amplitude[mask]

    # Handle wraparound for statistics using complex representation
    z = np.exp(1j * center_window * np.pi / 180)
    mean_z = np.mean(z)
    center_mean = np.angle(mean_z) * 180 / np.pi
    center_mean = np.mod(center_mean, 360)

    # Circular standard deviation
    R = np.abs(mean_z)
    if R > 0.01:
        center_std = np.sqrt(-2 * np.log(R)) * 180 / np.pi
    else:
        center_std = 180.0  # Essentially uniform

    # Drift rate (linear fit on unwrapped center)
    center_unwrapped = np.unwrap(center_window * np.pi / 180) * 180 / np.pi
    dt_s = (t_window[-1] - t_window[0]) / 1000  # seconds
    if dt_s > 0 and len(center_unwrapped) > 1:
        drift_rate = (center_unwrapped[-1] - center_unwrapped[0]) / dt_s
    else:
        drift_rate = 0.0

    # Diffusion coefficient (MSD analysis)
    # D = <(x(t+tau) - x(t))^2> / (2*tau)
    dt_ms = result.t_ms[1] - result.t_ms[0]
    lag_steps = max(1, int(100 / dt_ms))  # 100ms lag
    if len(center_unwrapped) > lag_steps:
        displacements = center_unwrapped[lag_steps:] - center_unwrapped[:-lag_steps]
        msd = np.mean(displacements**2)
        tau_s = lag_steps * dt_ms / 1000
        diffusion = msd / (2 * tau_s)
    else:
        diffusion = 0.0

    # Bump width (average over window, sample 20 points)
    activity = result.r[mask, :, population]
    widths = []
    n_samples = min(20, len(activity))
    indices = np.linspace(0, len(activity) - 1, n_samples, dtype=int)
    for i in indices:
        w = estimate_bump_width(
            activity[i],
            result.ring_params.node_angles_rad,
            center_window[i] * np.pi / 180,
        )
        widths.append(w)
    width_mean_deg = np.nanmean(widths)

    # Error from cue
    error_from_cue = angular_distance_deg(center_mean, result.stim_angle_deg)

    return {
        "center_mean_deg": center_mean,
        "center_std_deg": center_std,
        "amplitude_mean": np.mean(amp_window),
        "width_mean_deg": width_mean_deg,
        "drift_rate_deg_per_s": drift_rate,
        "diffusion_deg2_per_s": diffusion,
        "error_from_cue_deg": error_from_cue,
    }


def compute_metrics_at_delay_times(
    result: RingSimulationResult,
    delay_times_ms: list[float],
    window_ms: float = 200.0,
    population: int = 0,
) -> list[dict]:
    """
    Compute bump metrics at multiple timepoints during the delay.

    Parameters:
        result: RingSimulationResult
        delay_times_ms: Absolute time points (ms) at which to evaluate metrics.
            Each defines the center of a window of width window_ms.
        window_ms: Averaging window around each timepoint (ms)
        population: Which population to analyze (0=PYR)

    Returns:
        List of metric dicts (same format as compute_bump_metrics), one per
        timepoint.  Each dict also includes 'eval_time_ms'.
    """
    metrics_list = []
    half_w = window_ms / 2
    for t in delay_times_ms:
        t_start = max(t - half_w, result.t_ms[0])
        t_end = min(t + half_w, result.t_ms[-1])
        m = compute_bump_metrics(result, time_window=(t_start, t_end), population=population)
        m["eval_time_ms"] = t
        metrics_list.append(m)
    return metrics_list


def compute_working_memory_accuracy(
    result: RingSimulationResult,
    delay_end_ms: Optional[float] = None,
    population: int = 0,
) -> dict:
    """
    Compute working memory task accuracy metrics.

    Parameters:
        result: RingSimulationResult
        delay_end_ms: End of delay period (default: simulation end)
        population: Which population to decode (0=PYR)

    Returns:
        Dictionary with:
        - 'final_position_deg': Decoded position at end of delay
        - 'cue_position_deg': Original stimulus location
        - 'error_deg': Absolute angular error
        - 'maintained': Whether bump was maintained (amplitude > 0.3)
    """
    if delay_end_ms is None:
        delay_end_ms = result.t_ms[-1]

    # Find time index closest to delay end
    idx = np.argmin(np.abs(result.t_ms - delay_end_ms))

    # Decode position at that time
    activity = result.r[idx, :, population]
    center_rad, amplitude = population_vector_decode(
        activity, result.ring_params.node_angles_rad
    )
    center_deg = center_rad * 180 / np.pi

    error = angular_distance_deg(center_deg, result.stim_angle_deg)
    maintained = amplitude > 0.3

    return {
        "final_position_deg": center_deg,
        "cue_position_deg": result.stim_angle_deg,
        "error_deg": error,
        "maintained": maintained,
        "amplitude": amplitude,
    }


def aggregate_metrics_across_trials(
    all_trial_metrics: list[list[dict]],
) -> list[dict]:
    """Aggregate per-timepoint metrics from multiple trials into mean +/- SEM.

    Parameters:
        all_trial_metrics: list of length n_trials, each element is a list
            of metric dicts (one per delay timepoint), as returned by
            compute_metrics_at_delay_times().

    Returns:
        List of dicts (one per timepoint), each containing eval_time_ms
        plus {key}_mean and {key}_sem for each metric key.
    """
    n_timepoints = len(all_trial_metrics[0])
    metric_keys = [k for k in all_trial_metrics[0][0] if k != "eval_time_ms"]

    aggregated = []
    for tp_idx in range(n_timepoints):
        entry = {"eval_time_ms": all_trial_metrics[0][tp_idx]["eval_time_ms"]}
        for key in metric_keys:
            values = np.array(
                [trial[tp_idx][key] for trial in all_trial_metrics], dtype=float
            )
            valid = values[~np.isnan(values)]
            entry[f"{key}_mean"] = float(np.mean(valid)) if len(valid) > 0 else np.nan
            entry[f"{key}_sem"] = (
                float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
                if len(valid) > 1
                else 0.0
            )
        aggregated.append(entry)
    return aggregated


def aggregate_single_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate single-timepoint metric dicts (one per trial) into mean +/- SEM.

    Parameters:
        all_metrics: list of metric dicts (e.g. from compute_bump_metrics),
            one per trial.

    Returns:
        dict with {key}_mean and {key}_sem for each numeric metric key.
    """
    metric_keys = list(all_metrics[0].keys())
    result = {}
    for key in metric_keys:
        values = np.array([m[key] for m in all_metrics], dtype=float)
        valid = values[~np.isnan(values)]
        result[f"{key}_mean"] = float(np.mean(valid)) if len(valid) > 0 else np.nan
        result[f"{key}_sem"] = (
            float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
            if len(valid) > 1
            else 0.0
        )
    return result
