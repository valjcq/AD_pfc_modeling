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
from .constants import TRANSIENT_SKIP_TIME_MS


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


def compute_bump_asymmetry(
    result: RingSimulationResult,
    population: int = 0,
) -> np.ndarray:
    """Compute a left/right asymmetry index of activity around the cue location.

    For each time step, compares total activity on the left vs right side of the
    cue presentation angle, returning a normalized index in [-1, 1]:

        -1  → all activity concentrated on the left side of the cue
         0  → perfectly symmetric
        +1  → all activity concentrated on the right side of the cue

    "Left" means nodes with a negative signed angular offset from the cue
    (counter-clockwise direction); "right" means a positive signed offset
    (clockwise direction).

    Parameters:
        result: RingSimulationResult
        population: Which population to analyze (0 = PYR)

    Returns:
        asymmetry: Normalized asymmetry index, shape (n_steps,).
    """
    activity = result.r[:, :, population]           # (n_steps, n_nodes)
    node_angles = result.ring_params.node_angles_deg  # (n_nodes,)
    cue_deg = result.stim_angle_deg

    # Signed angular offset from cue: in (-180, 180]
    offsets = ((node_angles - cue_deg + 180.0) % 360.0) - 180.0

    left_mask = offsets < 0
    right_mask = offsets > 0

    left_activity = activity[:, left_mask].sum(axis=1)   # (n_steps,)
    right_activity = activity[:, right_mask].sum(axis=1)  # (n_steps,)

    total = left_activity + right_activity
    asymmetry = np.where(total > 1e-10, (right_activity - left_activity) / total, 0.0)

    return asymmetry


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
    """Aggregate per-timepoint metrics from multiple trials into mean ± SD/SEM.

    Parameters:
        all_trial_metrics: list of length n_trials, each element is a list
            of metric dicts (one per delay timepoint), as returned by
            compute_metrics_at_delay_times().

    Returns:
        List of dicts (one per timepoint), each containing eval_time_ms
        plus {key}_mean, {key}_sd, and {key}_sem for each metric key.
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
            n = len(valid)
            entry[f"{key}_mean"] = float(np.mean(valid)) if n > 0 else np.nan
            sd = float(np.std(valid, ddof=1)) if n > 1 else 0.0
            entry[f"{key}_sd"] = sd
            entry[f"{key}_sem"] = sd / np.sqrt(n) if n > 1 else 0.0
        aggregated.append(entry)
    return aggregated


def compute_msd_curve(
    centers_rad_trials: list[np.ndarray],
    t_s: np.ndarray,
    max_lag_frac: float = 0.75,
    n_lags: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean squared displacement (MSD) of bump center over time.

    For each lag τ, computes ⟨[φ(t+τ) - φ(0)]²⟩ averaged across trials.
    Trajectories should be *unwrapped* (no 2π jumps) and referenced to
    their initial position (i.e. displacement from t=0).

    Parameters:
        centers_rad_trials: List of per-trial unwrapped bump center
            trajectories in radians, each shape (n_steps,).  Each
            trajectory is shifted so that φ(0) = 0 internally.
        t_s: Time vector in seconds, shape (n_steps,).
        max_lag_frac: Maximum lag as fraction of total duration.
        n_lags: Number of lag points to evaluate.

    Returns:
        lag_times: Lag values in seconds, shape (n_lags,)
        msd_mean: Mean MSD in rad², shape (n_lags,)
        msd_sem: SEM of MSD across trials, shape (n_lags,)
        msd_sd: SD of MSD across trials, shape (n_lags,)
    """
    T = t_s[-1] - t_s[0]
    max_lag = max_lag_frac * T
    lag_times = np.linspace(0, max_lag, n_lags)
    dt = t_s[1] - t_s[0]

    n_trials = len(centers_rad_trials)
    msd_per_trial = np.zeros((n_trials, n_lags))

    for trial_idx, traj in enumerate(centers_rad_trials):
        # Reference to initial position
        disp = traj - traj[0]
        for i, lag in enumerate(lag_times):
            lag_steps = int(round(lag / dt))
            if lag_steps == 0:
                msd_per_trial[trial_idx, i] = 0.0
            elif lag_steps < len(disp):
                # MSD at this lag: average over all starting points
                squared_displacements = (disp[lag_steps:] - disp[:-lag_steps]) ** 2
                msd_per_trial[trial_idx, i] = np.mean(squared_displacements)
            else:
                msd_per_trial[trial_idx, i] = np.nan

    msd_mean = np.nanmean(msd_per_trial, axis=0)
    msd_sd = np.nanstd(msd_per_trial, axis=0, ddof=1) if n_trials > 1 else np.zeros(n_lags)
    msd_sem = msd_sd / np.sqrt(n_trials) if n_trials > 1 else np.zeros(n_lags)

    return lag_times, msd_mean, msd_sem, msd_sd


def fit_diffusion_coefficient(
    lag_times: np.ndarray,
    msd: np.ndarray,
    fit_range: tuple[float, float] = (0.1, 1.0),
) -> tuple[float, np.ndarray, float]:
    """Fit a line to the linear regime of the MSD to extract B_hat.

    Parameters:
        lag_times: Lag values in seconds.
        msd: MSD values in rad².
        fit_range: (t_min, t_max) in seconds for the linear fit region.

    Returns:
        B_hat: Diffusion strength (slope of MSD vs t, in rad²/s).
        fit_line: Fitted MSD values at all lag_times, for plotting.
        r_squared: Coefficient of determination of the fit.
    """
    mask = (lag_times >= fit_range[0]) & (lag_times <= fit_range[1])
    mask &= ~np.isnan(msd)
    if np.sum(mask) < 2:
        return 0.0, np.full_like(lag_times, np.nan), 0.0

    t_fit = lag_times[mask]
    msd_fit = msd[mask]

    # Linear fit: MSD = B * t + intercept
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    B_hat = coeffs[0]  # slope = diffusion strength
    intercept = coeffs[1]

    fit_line = B_hat * lag_times + intercept

    # R²
    ss_res = np.sum((msd_fit - (B_hat * t_fit + intercept)) ** 2)
    ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return B_hat, fit_line, r_squared


def compute_oscillation_band_timecourse(
    amplitude: np.ndarray,
    t_s: np.ndarray,
    min_freq_hz: float = 2.0,
    max_freq_hz: float = 12.0,
    window_s: float = 1.0,
    overlap_frac: float = 0.8,
) -> dict:
    """Compute a time-frequency map and dominant-band trajectory for one trial.

    Uses a Hann-window STFT on the bump-amplitude trajectory. In each time
    bin, the dominant frequency is selected as the frequency with maximum power
    within [min_freq_hz, max_freq_hz].

    Parameters:
        amplitude: 1D bump-amplitude trajectory, shape (n_steps,).
        t_s: Matching time vector in seconds, shape (n_steps,).
        min_freq_hz: Low edge of frequency band to analyze.
        max_freq_hz: High edge of frequency band to analyze.
        window_s: STFT window length in seconds.
        overlap_frac: Fractional overlap between consecutive windows [0, 0.99].

    Returns:
        dict with keys:
            ``freqs_hz``            – frequency axis (band-limited), shape (n_freqs,)
            ``times_s``             – STFT time bins (absolute seconds), shape (n_times,)
            ``power``               – power map, shape (n_freqs, n_times)
            ``dominant_freq_hz``    – dominant frequency per time bin, shape (n_times,)
            ``dominant_power``      – dominant power per time bin, shape (n_times,)
    """
    from scipy.signal import spectrogram

    amp = np.asarray(amplitude, dtype=float).ravel()
    tt = np.asarray(t_s, dtype=float).ravel()
    if len(amp) != len(tt):
        raise ValueError("amplitude and t_s must have same length")
    if len(amp) < 8:
        raise ValueError("Need at least 8 samples for oscillation timecourse")

    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1e-3
    fs = 1.0 / max(dt, 1e-9)

    nperseg = int(max(8, round(window_s * fs)))
    nperseg = min(nperseg, len(amp))
    overlap_frac = float(np.clip(overlap_frac, 0.0, 0.99))
    noverlap = int(round(overlap_frac * nperseg))
    noverlap = min(noverlap, max(0, nperseg - 1))
    # Aggressive zero-padding: very dense frequency grid for thin, smooth bins.
    # Cap nfft to avoid pathological runtimes on very long windows.
    nfft_base = max(64, nperseg)
    nfft_pow2 = 1 << int(np.ceil(np.log2(nfft_base)))
    nfft = int(max(nfft_pow2, 16 * nperseg))
    nfft = min(nfft, 32768)

    # Remove slow trend and DC offset so low-frequency drift does not dominate.
    trend = np.polyval(np.polyfit(tt, amp, 1), tt)
    detrended = amp - trend
    detrended = detrended - np.mean(detrended)

    freqs, times_rel, power = spectrogram(
        detrended,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling='density',
        mode='psd',
    )

    band = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    if not np.any(band):
        return {
            'freqs_hz': np.array([], dtype=float),
            'times_s': np.array([], dtype=float),
            'power': np.zeros((0, 0), dtype=float),
            'dominant_freq_hz': np.array([], dtype=float),
            'dominant_power': np.array([], dtype=float),
        }

    freqs_band = freqs[band]
    power_band = power[band, :]
    if power_band.shape[1] == 0:
        times_abs = np.array([], dtype=float)
        dom_freq = np.array([], dtype=float)
        dom_power = np.array([], dtype=float)
    else:
        peak_idx = np.argmax(power_band, axis=0)
        # Refine peak location with quadratic interpolation in log-power around
        # the winning bin to avoid frequency values being locked to FFT bin centers.
        dom_freq = freqs_band[peak_idx].astype(float)
        if len(freqs_band) >= 3:
            for ti, k in enumerate(peak_idx):
                if k <= 0 or k >= len(freqs_band) - 1:
                    continue
                y1 = float(np.log(max(power_band[k - 1, ti], 1e-30)))
                y2 = float(np.log(max(power_band[k, ti], 1e-30)))
                y3 = float(np.log(max(power_band[k + 1, ti], 1e-30)))
                denom = (y1 - 2.0 * y2 + y3)
                if abs(denom) < 1e-12:
                    continue
                delta = 0.5 * (y1 - y3) / denom
                # Keep interpolation local to neighboring bins.
                delta = float(np.clip(delta, -1.0, 1.0))
                f_lo = float(freqs_band[k - 1])
                f_mid = float(freqs_band[k])
                f_hi = float(freqs_band[k + 1])
                if f_hi <= f_lo:
                    continue
                dom_freq[ti] = f_mid + delta * (f_hi - f_lo) * 0.5
        dom_power = power_band[peak_idx, np.arange(power_band.shape[1])].astype(float)
        dom_freq = dom_freq.astype(float)
        # SNR threshold: suppress timepoints where the dominant bin does not
        # stand above the local noise floor (median across all band frequencies).
        noise_floor_t = np.median(power_band, axis=0)
        below_snr = dom_power <= 2.0 * noise_floor_t
        dom_freq[below_snr] = np.nan
        dom_power[below_snr] = np.nan
        times_abs = float(tt[0]) + times_rel

    return {
        'freqs_hz': freqs_band,
        'times_s': times_abs,
        'power': power_band,
        'dominant_freq_hz': dom_freq,
        'dominant_power': dom_power,
    }


def summarize_oscillation_timecourse(
    dominant_freq_hz: np.ndarray,
    dominant_power: np.ndarray,
    times_s: np.ndarray,
    sample_time_s: Optional[float] = None,
) -> dict:
    """Summarize dominant oscillation trajectories into trial-level metrics.

    Returns both delay-averaged metrics and one-timepoint metrics (nearest STFT
    bin to sample_time_s, or center time if sample_time_s is None).
    """
    f = np.asarray(dominant_freq_hz, dtype=float).ravel()
    p = np.asarray(dominant_power, dtype=float).ravel()
    t = np.asarray(times_s, dtype=float).ravel()

    _nan_result = {
        'freq_median_hz': np.nan,
        'power_median': np.nan,
        'freq_sample_hz': np.nan,
        'power_sample': np.nan,
        'sample_time_s': np.nan,
    }

    if len(f) == 0 or len(p) == 0 or len(t) == 0:
        return _nan_result

    # t must be non-NaN to find sample index; f and p may have NaNs from SNR mask
    valid_t = ~np.isnan(t)
    if not np.any(valid_t):
        return _nan_result

    t = t[valid_t]
    f = f[valid_t]
    p = p[valid_t]

    if sample_time_s is None:
        sample_time_s = float(t[len(t) // 2])
    else:
        sample_time_s = float(sample_time_s)

    idx = int(np.argmin(np.abs(t - sample_time_s)))
    return {
        'freq_median_hz': float(np.nanmedian(f)),
        'power_median': float(np.nanmedian(p)),
        'freq_sample_hz': float(f[idx]),
        'power_sample': float(p[idx]),
        'sample_time_s': float(t[idx]),
    }


def lowpass_filter_trajectory(
    trajectory: np.ndarray,
    t_s: np.ndarray,
    cutoff_hz: float,
) -> np.ndarray:
    """Apply a zero-phase 4th-order Butterworth low-pass filter to a trajectory.

    Uses ``scipy.signal.filtfilt`` so there is no phase shift.  If the cutoff
    is at or above the Nyquist frequency the raw trajectory is returned
    unchanged.

    Parameters:
        trajectory: 1-D array of bump center positions (radians, unwrapped).
        t_s: Time vector in seconds (used to compute sampling frequency).
        cutoff_hz: Low-pass cutoff frequency in Hz.

    Returns:
        Filtered trajectory, same shape as input.
    """
    from scipy.signal import butter, filtfilt

    dt = float(t_s[1] - t_s[0]) if len(t_s) > 1 else 1e-3
    fs = 1.0 / dt
    nyq = fs / 2.0
    normalized_cutoff = cutoff_hz / nyq
    if normalized_cutoff >= 1.0:
        return trajectory  # cutoff above Nyquist — nothing to filter
    b, a = butter(4, normalized_cutoff, btype='low')
    return filtfilt(b, a, trajectory)


def fit_oscillation_corrected_diffusion(
    lag_times: np.ndarray,
    msd: np.ndarray,
    fit_range: tuple[float, float] = (0.1, 1.0),
    osc_period_s: Optional[float] = None,
) -> tuple[float, np.ndarray, float]:
    """Fit the MSD accounting for a known oscillation period.

    When ``osc_period_s`` is provided the model

        MSD(τ) = B·τ + C·(1 − cos(2π·τ / T)) + offset

    is fitted, separating the genuine diffusion coefficient *B* from the
    oscillatory contribution *C*.  The oscillation period *T* is treated as
    fixed (determined externally by FFT).

    When ``osc_period_s`` is None the function falls back to the standard
    linear fit (same as ``fit_diffusion_coefficient``).

    Parameters:
        lag_times: Lag values in seconds.
        msd: MSD values in rad².
        fit_range: (t_min, t_max) in seconds for the fit region.
        osc_period_s: Known oscillation period in seconds (or None).

    Returns:
        B_hat: Diffusion strength in rad²/s.
        fit_line: Model-evaluated MSD at all lag_times, for plotting.
        r_squared: Coefficient of determination of the fit.
    """
    if osc_period_s is None:
        return fit_diffusion_coefficient(lag_times, msd, fit_range)

    from scipy.optimize import curve_fit

    mask = (lag_times >= fit_range[0]) & (lag_times <= fit_range[1]) & ~np.isnan(msd)
    if np.sum(mask) < 4:
        return fit_diffusion_coefficient(lag_times, msd, fit_range)

    omega = 2.0 * np.pi / osc_period_s
    t_fit = lag_times[mask]
    msd_fit = msd[mask]

    def model(t: np.ndarray, B: float, C: float, offset: float) -> np.ndarray:
        return B * t + C * (1.0 - np.cos(omega * t)) + offset

    # Initial guess: slope from endpoints, zero oscillation component
    slope0 = float(np.mean(np.diff(msd_fit) / np.diff(t_fit))) if len(t_fit) > 1 else 0.0
    p0 = [max(slope0, 0.0), 0.01, float(msd_fit[0])]

    try:
        popt, _ = curve_fit(model, t_fit, msd_fit, p0=p0, maxfev=10_000)
        B_hat = float(popt[0])
        fit_line = model(lag_times, *popt)
        ss_res = float(np.sum((msd_fit - model(t_fit, *popt)) ** 2))
        ss_tot = float(np.sum((msd_fit - np.mean(msd_fit)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return B_hat, fit_line, r_squared
    except Exception:
        return fit_diffusion_coefficient(lag_times, msd, fit_range)


def compute_asymmetry_temporal_metrics(
    asym: np.ndarray,
    t_ms: np.ndarray,
) -> dict:
    """Compute temporal metrics of an asymmetry timecourse.

    These metrics avoid the cancellation problem of the simple time-average:
    oscillations that are symmetric around zero (same signed area on each side)
    give a mean close to zero even though the bump is severely displaced most
    of the time.

    Parameters
    ----------
    asym : np.ndarray, shape (n_steps,)
        Instantaneous asymmetry A(t) for the window of interest (e.g. the delay
        period after the transient skip, already masked).
    t_ms : np.ndarray, shape (n_steps,)
        Corresponding time vector in milliseconds.

    Returns
    -------
    dict with:
        ``mean_abs_asym``
            Mean of |A(t)|.  Unlike mean(A(t)) this never cancels: a bump that
            oscillates ±0.3 gives mean_abs ≈ 0.3 even though mean(A) ≈ 0.
        ``asym_std``
            Standard deviation of A(t).  Captures both amplitude of variation
            and side-switching together; independent of the DC offset.
    """
    if len(asym) == 0:
        return {'mean_abs_asym': np.nan, 'asym_std': np.nan}

    mean_abs_asym = float(np.mean(np.abs(asym)))
    asym_std = float(np.std(asym, ddof=1)) if len(asym) > 1 else 0.0

    return {'mean_abs_asym': mean_abs_asym, 'asym_std': asym_std}


def compute_noise_floor(A_hat_values: np.ndarray, percentile: float = 95.0) -> float:
    """Compute noise floor threshold from no-stimulus Â_hat values.

    Parameters:
        A_hat_values: Array of population-vector amplitudes from baseline
            (no-stimulus) trials, any shape.
        percentile: Percentile to use as threshold (default: 95th).

    Returns:
        Noise floor threshold (scalar float).
    """
    valid = A_hat_values[~np.isnan(A_hat_values.ravel())]
    if len(valid) == 0:
        return 0.0
    return float(np.percentile(valid, percentile))


# A_hat values below this indicate all nodes hit the firing-rate ceiling.
SATURATION_A_HAT_THRESHOLD: float = 1e-6


def aggregate_single_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate single-timepoint metric dicts (one per trial) into mean ± SD/SEM.

    Parameters:
        all_metrics: list of metric dicts (e.g. from compute_bump_metrics),
            one per trial.

    Returns:
        dict with {key}_mean, {key}_sd, and {key}_sem for each numeric key.
    """
    metric_keys = list(all_metrics[0].keys())
    result = {}
    for key in metric_keys:
        values = np.array([m[key] for m in all_metrics], dtype=float)
        valid = values[~np.isnan(values)]
        n = len(valid)
        result[f"{key}_mean"] = float(np.mean(valid)) if n > 0 else np.nan
        sd = float(np.std(valid, ddof=1)) if n > 1 else 0.0
        result[f"{key}_sd"] = sd
        result[f"{key}_sem"] = sd / np.sqrt(n) if n > 1 else 0.0
    return result


# ---------------------------------------------------------------------------
# New analysis functions for the 4-experiment battery
# ---------------------------------------------------------------------------


def compute_plv_timecourse(
    signal1: np.ndarray,
    signal2: np.ndarray,
    t_s: np.ndarray,
    min_freq_hz: float = 2.0,
    max_freq_hz: float = 12.0,
    filter_order: int = 4,
    window_s: float = 1.0,
    overlap_frac: float = 0.8,
) -> dict:
    """Compute a Phase Locking Value (PLV) timecourse between two signals.

    Measures phase synchrony in the oscillation band by:
    1. Bandpass-filtering both signals (zero-phase Butterworth)
    2. Extracting instantaneous phases via Hilbert transform
    3. Computing PLV in sliding windows matching the STFT bin grid

    The window/step formula is identical to ``compute_oscillation_band_timecourse``
    so that PLV and STFT timecourses share the same time axis.

    Parameters
    ----------
    signal1, signal2 : np.ndarray
        1-D raw signals (e.g. firing rate at cue node and distractor node),
        shape ``(n_steps,)``.
    t_s : np.ndarray
        Matching time vector in seconds, shape ``(n_steps,)``.
    min_freq_hz, max_freq_hz : float
        Bandpass edges for the Butterworth filter.
    filter_order : int
        Butterworth filter order (applied forward+backward by ``filtfilt``).
    window_s : float
        Sliding-window length in seconds.
    overlap_frac : float
        Fractional overlap between consecutive windows [0, 0.99].

    Returns
    -------
    dict with keys:
        ``times_s`` – window-center times (absolute seconds), shape ``(n_windows,)``
        ``plv``     – PLV values in [0, 1], shape ``(n_windows,)``
    """
    from scipy.signal import butter, filtfilt, hilbert

    _empty = {'times_s': np.array([], dtype=float), 'plv': np.array([], dtype=float)}

    s1 = np.asarray(signal1, dtype=float).ravel()
    s2 = np.asarray(signal2, dtype=float).ravel()
    tt = np.asarray(t_s, dtype=float).ravel()

    if len(s1) != len(tt) or len(s2) != len(tt):
        raise ValueError("signal1, signal2, and t_s must have the same length")
    if len(s1) < 8:
        return _empty

    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1e-3
    fs = 1.0 / max(dt, 1e-9)
    nyq = fs / 2.0

    lo = float(np.clip(min_freq_hz, 0.0, nyq - 1e-3))
    hi = float(np.clip(max_freq_hz, lo + 1e-3, nyq - 1e-3))
    if lo >= nyq or hi >= nyq or lo >= hi:
        return _empty

    b, a = butter(filter_order, [lo / nyq, hi / nyq], btype='bandpass')
    filt1 = filtfilt(b, a, s1)
    filt2 = filtfilt(b, a, s2)

    phase1 = np.angle(hilbert(filt1))
    phase2 = np.angle(hilbert(filt2))
    dphase = phase1 - phase2

    # Sliding window matching compute_oscillation_band_timecourse bin grid
    nperseg = int(max(8, round(window_s * fs)))
    nperseg = min(nperseg, len(s1))
    overlap_frac = float(np.clip(overlap_frac, 0.0, 0.99))
    noverlap = int(round(overlap_frac * nperseg))
    noverlap = min(noverlap, max(0, nperseg - 1))
    step = max(1, nperseg - noverlap)

    centers: list[float] = []
    plv_vals: list[float] = []
    start = 0
    while start + nperseg <= len(s1):
        end = start + nperseg
        window_dphase = dphase[start:end]
        plv_val = float(np.abs(np.mean(np.exp(1j * window_dphase))))
        t_center = float(tt[start + nperseg // 2])
        centers.append(t_center)
        plv_vals.append(plv_val)
        start += step

    return {
        'times_s': np.array(centers, dtype=float),
        'plv': np.array(plv_vals, dtype=float),
    }

