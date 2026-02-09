"""
External Current Sensitivity Analysis for the 4-Population PFC Circuit Model.

Analyzes how external currents (I_ext_pyr, I_ext_pv, I_ext_som, I_ext_vip)
affect network behavior across 7 metric categories:
  1. Firing Rate Statistics
  2. Gain Modulation
  3. E/I Balance
  4. Response to Perturbations
  5. Sensory Responsiveness
  6. Disinhibition Efficacy
  7. Adaptation Strength

Usage:
    python analysis_external_currents.py                  # run all metrics
    python analysis_external_currents.py --fast           # skip noisy CV calculations
    python analysis_external_currents.py --metrics 1 3 6  # run only selected metrics
    python analysis_external_currents.py --no_show        # save figures without display
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from circuit_model import (
    CircuitParams,
    simulate_circuit,
    mean_rates,
    load_params_json,
    POPULATION_NAMES,
    POPULATION_COLORS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POP_INDICES = {"PYR": 0, "SOM": 1, "PV": 2, "VIP": 3}
I0_FIELDS = {"PYR": "I0_pyr", "PV": "I0_pv", "SOM": "I0_som", "VIP": "I0_vip"}

# Sweep ranges for I0 (receptor components stay fixed)
SWEEP_RANGES = {
    "PYR": (0.0, 15.0),
    "PV":  (0.0, 15.0),
    "SOM": (0.0, 20.0),
    "VIP": (0.0, 15.0),
}
N_SWEEP = 25

# Simulation defaults
T_SWEEP_MS = 3000.0
T_ADAPT_MS = 5000.0
DT_MS = 0.1
BURN_IN_MS = 1500.0
N_NOISE_SEEDS = 5

OUT_DIR = "figs/analysis"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def _save_fig(fig, filename, show=True, out_dir=None):
    """Save figure and optionally display."""
    path = os.path.join(out_dir or OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _get_total_I_ext(params: CircuitParams, pop: str) -> float:
    """Return the total static I_ext for a population."""
    return {
        "PYR": params.I_ext_pyr(),
        "PV":  params.I_ext_pv(),
        "SOM": params.I_ext_som(),
        "VIP": params.I_ext_vip(),
    }[pop]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_sweep(
    params: CircuitParams,
    pop: str,
    I0_values: np.ndarray,
    T_ms: float = T_SWEEP_MS,
    noise_type: str = "none",
    seed: int = 42,
) -> np.ndarray:
    """Sweep one population's I0 and return steady-state rates.

    Returns array of shape (len(I0_values), 4).
    """
    field = I0_FIELDS[pop]
    rates = np.zeros((len(I0_values), 4))
    for i, val in enumerate(I0_values):
        p = replace(params, **{field: val})
        result = simulate_circuit(p, T_ms=T_ms, dt_ms=DT_MS, seed=seed,
                                  noise_type=noise_type)
        rates[i] = mean_rates(result, burn_in_ms=BURN_IN_MS, window_ms=0)
    return rates


def run_sweep_stats(
    params: CircuitParams,
    pop: str,
    I0_values: np.ndarray,
    n_seeds: int = N_NOISE_SEEDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep with noise across seeds. Returns (mean_rates, std_rates), each (n_pts, 4)."""
    field = I0_FIELDS[pop]
    all_rates = np.zeros((len(I0_values), n_seeds, 4))
    for i, val in enumerate(I0_values):
        p = replace(params, **{field: val})
        for s in range(n_seeds):
            result = simulate_circuit(p, T_ms=T_SWEEP_MS, dt_ms=DT_MS,
                                      seed=42 + s, noise_type="white")
            all_rates[i, s] = mean_rates(result, burn_in_ms=BURN_IN_MS, window_ms=0)
    return np.mean(all_rates, axis=1), np.std(all_rates, axis=1)


def get_steady_state(
    params: CircuitParams,
    T_ms: float = 8000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a long simulation to reach true steady state.

    Returns (r_ss, I_adapt_ss) — final firing rates (4,) and adaptation (2,).
    Uses T_ms=8000 by default (~3.5x SOM tau_adapt) to ensure convergence.
    """
    result = simulate_circuit(params, T_ms=T_ms, dt_ms=DT_MS, noise_type="none")
    return result.r[-1].copy(), result.I_adapt[-1].copy()


def run_perturbation(
    params: CircuitParams,
    pop: str,
    delta: float,
    r0: Optional[np.ndarray] = None,
    I_adapt0: Optional[np.ndarray] = None,
    baseline_ms: float = 500.0,
    pulse_ms: float = 500.0,
    recovery_ms: float = 1500.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """3-phase simulation: baseline -> pulse ON -> recovery.

    If r0/I_adapt0 are provided, starts from that state (should be the true
    steady state from get_steady_state). Otherwise runs from default init.

    Returns (t_ms, r) concatenated across all phases. Time starts at 0.
    """
    field = I0_FIELDS[pop]
    I0_base = getattr(params, field)

    # Phase 1: short baseline from steady state (to show flat pre-pulse trace)
    res1 = simulate_circuit(params, T_ms=baseline_ms, dt_ms=DT_MS, seed=seed,
                            noise_type="none", r0=r0, I_adapt0=I_adapt0)
    r_ss = res1.r[-1]
    adapt_ss = res1.I_adapt[-1]

    # Phase 2: pulse ON
    p_on = replace(params, **{field: I0_base + delta})
    res2 = simulate_circuit(p_on, T_ms=pulse_ms, dt_ms=DT_MS,
                            r0=r_ss, I_adapt0=adapt_ss,
                            seed=seed, noise_type="none")

    # Phase 3: recovery
    res3 = simulate_circuit(params, T_ms=recovery_ms, dt_ms=DT_MS,
                            r0=res2.r[-1], I_adapt0=res2.I_adapt[-1],
                            seed=seed, noise_type="none")

    # Concatenate (skip first sample of phases 2,3 to avoid duplication)
    t = np.concatenate([
        res1.t_ms,
        res1.t_ms[-1] + res2.t_ms[1:],
        res1.t_ms[-1] + res2.t_ms[-1] + res3.t_ms[1:],
    ])
    r = np.concatenate([res1.r, res2.r[1:], res3.r[1:]], axis=0)
    return t, r


def reconstruct_pyr_currents(
    result, params: CircuitParams
) -> dict[str, np.ndarray]:
    """Reconstruct PYR input current components post-hoc."""
    r = result.r
    I_adapt = result.I_adapt
    ggaba = params.g_gaba()

    r_pyr, r_som, r_pv = r[:, 0], r[:, 1], r[:, 2]
    Iap = I_adapt[:, 0]

    denom = 1.0 + ggaba * params.w_pe * r_pv
    I_exc = (params.w_ee * r_pyr) / denom
    I_inh_som = ggaba * params.w_se * r_som
    I_ext = params.I_ext_pyr()

    E = I_exc + I_ext
    I = I_inh_som + Iap

    return {
        "I_exc": I_exc,
        "I_inh_som": I_inh_som,
        "I_adapt": Iap,
        "I_ext": np.full_like(r_pyr, I_ext),
        "shunting_denom": denom,
        "E": E,
        "I": I,
        "EI_ratio": E / np.maximum(I, 1e-6),
    }


def fit_exponential_decay(t, y):
    """Fit y = A*exp(-t/tau) + C. Returns (A, tau, C) or None on failure."""
    try:
        t_shifted = t - t[0]
        y0, yf = y[0], y[-1]
        A0 = y0 - yf
        tau0 = (t_shifted[-1] - t_shifted[0]) / 4.0

        def exp_model(x, A, tau, C):
            return A * np.exp(-x / tau) + C

        bounds = ([-np.inf, 1.0, -np.inf], [np.inf, 1e5, np.inf])
        popt, _ = curve_fit(exp_model, t_shifted, y, p0=[A0, tau0, yf],
                            bounds=bounds, maxfev=5000)
        return popt  # (A, tau, C)
    except (RuntimeError, ValueError):
        return None


# =========================================================================
# METRIC 1: Firing Rate Statistics
# =========================================================================
def compute_metric1(params: CircuitParams, fast: bool = False):
    """Sweep each population's I0, compute mean/std/CV of all rates."""
    data = {}
    n_steps = len(POPULATION_NAMES) * (1 if fast else 2)
    pbar = tqdm(total=n_steps, desc="M1: Firing Rate Stats", unit="sweep")
    for pop in POPULATION_NAMES:
        lo, hi = SWEEP_RANGES[pop]
        I0_vals = np.linspace(lo, hi, N_SWEEP)

        # Deterministic sweep for mean rates
        rates_det = run_sweep(params, pop, I0_vals)
        pbar.update(1)

        # Noisy sweep for variability (CV)
        if fast:
            rates_mean = rates_det
            rates_std = np.zeros_like(rates_det)
        else:
            rates_mean, rates_std = run_sweep_stats(params, pop, I0_vals)
            pbar.update(1)

        cv = rates_std / np.maximum(rates_mean, 1e-6)

        # Compute total I_ext at each sweep point (for x-axis)
        field = I0_FIELDS[pop]
        baseline_receptor = _get_total_I_ext(params, pop) - getattr(params, field)
        I_ext_vals = I0_vals + baseline_receptor

        data[pop] = {
            "I0_values": I0_vals,
            "I_ext_values": I_ext_vals,
            "rates_det": rates_det,
            "rates_mean": rates_mean,
            "rates_std": rates_std,
            "cv": cv,
        }
    pbar.close()
    return data


def plot_metric1(data, show=True, out_dir=None):
    """Figure 1: Firing rate statistics — mean±std and CV."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    for j, swept_pop in enumerate(POPULATION_NAMES):
        d = data[swept_pop]
        x = d["I_ext_values"]

        # Top row: mean ± std
        ax = axes[0, j]
        for i, pop in enumerate(POPULATION_NAMES):
            color = POPULATION_COLORS[pop]
            ax.plot(x, d["rates_mean"][:, i], color=color, label=pop, linewidth=1.5)
            if np.any(d["rates_std"] > 0):
                ax.fill_between(x,
                                d["rates_mean"][:, i] - d["rates_std"][:, i],
                                d["rates_mean"][:, i] + d["rates_std"][:, i],
                                color=color, alpha=0.2)
        # Mark baseline
        baseline_I = _get_total_I_ext(CircuitParams(), swept_pop)
        ax.axvline(baseline_I, color="gray", ls="--", lw=1, alpha=0.6)
        _style_ax(ax, xlabel=f"I_ext_{swept_pop.lower()}", ylabel="Firing rate",
                  title=f"Sweep I_ext_{swept_pop}")
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.legend(fontsize=8, loc="best")

        # Bottom row: CV
        ax = axes[1, j]
        for i, pop in enumerate(POPULATION_NAMES):
            color = POPULATION_COLORS[pop]
            ax.plot(x, d["cv"][:, i], color=color, label=pop, linewidth=1.5)
        ax.axvline(baseline_I, color="gray", ls="--", lw=1, alpha=0.6)
        _style_ax(ax, xlabel=f"I_ext_{swept_pop.lower()}", ylabel="CV",
                  title=f"CV vs I_ext_{swept_pop}")
        ax.set_ylim(bottom=0)

    fig.suptitle("Metric 1: Firing Rate Statistics", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric1_firing_rate_sweeps.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 2: Gain Modulation
# =========================================================================
def compute_metric2(params: CircuitParams):
    """Compute gain matrix (Jacobian) and gain curves."""
    delta = 0.5
    pbar = tqdm(total=8, desc="M2: Gain Modulation", unit="sweep")

    # --- Gain matrix at baseline (4x4) ---
    gain_matrix = np.zeros((4, 4))  # [target_pop, swept_pop]
    for j, pop in enumerate(POPULATION_NAMES):
        field = I0_FIELDS[pop]
        I0_base = getattr(params, field)

        p_plus = replace(params, **{field: I0_base + delta})
        p_minus = replace(params, **{field: I0_base - delta})

        r_plus = mean_rates(
            simulate_circuit(p_plus, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none"),
            burn_in_ms=BURN_IN_MS, window_ms=0,
        )
        r_minus = mean_rates(
            simulate_circuit(p_minus, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none"),
            burn_in_ms=BURN_IN_MS, window_ms=0,
        )
        gain_matrix[:, j] = (r_plus - r_minus) / (2 * delta)
        pbar.update(1)

    # --- Gain curves along each sweep ---
    gain_curves = {}
    for pop in POPULATION_NAMES:
        lo, hi = SWEEP_RANGES[pop]
        I0_vals = np.linspace(lo, hi, N_SWEEP)
        rates = run_sweep(params, pop, I0_vals)

        # Numerical gradient along sweep
        field = I0_FIELDS[pop]
        baseline_receptor = _get_total_I_ext(params, pop) - getattr(params, field)
        I_ext_vals = I0_vals + baseline_receptor
        gains = np.gradient(rates, I0_vals, axis=0)  # d(rate)/d(I0)

        gain_curves[pop] = {
            "I_ext_values": I_ext_vals,
            "gains": gains,  # (N_SWEEP, 4)
        }
        pbar.update(1)
    pbar.close()

    return {"gain_matrix": gain_matrix, "gain_curves": gain_curves}


def plot_metric2(data, show=True, out_dir=None):
    """Figure 2: Gain modulation — heatmap + gain curves."""
    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 1, 1, 1, 1])

    # Panel A: gain matrix heatmap
    ax_heat = fig.add_subplot(gs[0, 0])
    gm = data["gain_matrix"]
    im = ax_heat.imshow(gm, cmap="RdBu_r", aspect="auto",
                        vmin=-np.max(np.abs(gm)), vmax=np.max(np.abs(gm)))
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels(POPULATION_NAMES, fontsize=10)
    ax_heat.set_yticks(range(4))
    ax_heat.set_yticklabels(POPULATION_NAMES, fontsize=10)
    ax_heat.set_xlabel("Swept population (I_ext)", fontsize=11)
    ax_heat.set_ylabel("Measured population", fontsize=11)
    ax_heat.set_title("Gain matrix at baseline", fontsize=12, fontweight="bold")
    # Annotate cells
    for ii in range(4):
        for jj in range(4):
            ax_heat.text(jj, ii, f"{gm[ii, jj]:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if abs(gm[ii, jj]) > 0.5 * np.max(np.abs(gm)) else "black")
    fig.colorbar(im, ax=ax_heat, shrink=0.7, label="dF/dI")

    # Panels B-E: gain curves
    for j, swept_pop in enumerate(POPULATION_NAMES):
        ax = fig.add_subplot(gs[0, j + 1])
        gc = data["gain_curves"][swept_pop]
        x = gc["I_ext_values"]
        for i, pop in enumerate(POPULATION_NAMES):
            ax.plot(x, gc["gains"][:, i], color=POPULATION_COLORS[pop],
                    label=pop, linewidth=1.5)
        baseline_I = _get_total_I_ext(CircuitParams(), swept_pop)
        ax.axvline(baseline_I, color="gray", ls="--", lw=1, alpha=0.6)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        _style_ax(ax, xlabel=f"I_ext_{swept_pop.lower()}", ylabel="Gain (dF/dI)",
                  title=f"Gain vs I_ext_{swept_pop}")
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("Metric 2: Gain Modulation", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric2_gain_modulation.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 3: E/I Balance
# =========================================================================
def compute_metric3(params: CircuitParams):
    """E/I decomposition for PYR across I_ext sweeps."""
    pbar = tqdm(total=N_SWEEP * 2 + 1, desc="M3: E/I Balance", unit="sim")

    # --- Sweep I_ext_pyr ---
    I0_pyr_vals = np.linspace(*SWEEP_RANGES["PYR"], N_SWEEP)
    ei_pyr_sweep = {k: [] for k in ["I_exc", "I_inh_som", "I_adapt", "I_ext",
                                      "shunting_denom", "E", "I", "EI_ratio"]}
    for val in I0_pyr_vals:
        p = replace(params, I0_pyr=val)
        result = simulate_circuit(p, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none")
        c = reconstruct_pyr_currents(result, p)
        for k in ei_pyr_sweep:
            ei_pyr_sweep[k].append(np.mean(c[k][-1000:]))  # last 100ms
        pbar.update(1)

    for k in ei_pyr_sweep:
        ei_pyr_sweep[k] = np.array(ei_pyr_sweep[k])

    # --- Sweep I_ext_pv ---
    I0_pv_vals = np.linspace(*SWEEP_RANGES["PV"], N_SWEEP)
    baseline_receptor_pv = _get_total_I_ext(params, "PV") - params.I0_pv
    I_ext_pv_vals = I0_pv_vals + baseline_receptor_pv
    shunting_vs_pv = []
    ei_vs_pv = []
    for val in I0_pv_vals:
        p = replace(params, I0_pv=val)
        result = simulate_circuit(p, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none")
        c = reconstruct_pyr_currents(result, p)
        shunting_vs_pv.append(np.mean(c["shunting_denom"][-1000:]))
        ei_vs_pv.append(np.mean(c["EI_ratio"][-1000:]))
        pbar.update(1)
    shunting_vs_pv = np.array(shunting_vs_pv)
    ei_vs_pv = np.array(ei_vs_pv)

    # --- Time series at baseline ---
    result_bl = simulate_circuit(params, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="white",
                                 seed=42)
    currents_bl = reconstruct_pyr_currents(result_bl, params)
    pbar.update(1)
    pbar.close()

    return {
        "I0_pyr_vals": I0_pyr_vals,
        "I_ext_pyr_vals": I0_pyr_vals,  # PYR has no receptor component
        "ei_pyr_sweep": ei_pyr_sweep,
        "I_ext_pv_vals": I_ext_pv_vals,
        "shunting_vs_pv": shunting_vs_pv,
        "ei_vs_pv": ei_vs_pv,
        "t_baseline": result_bl.t_ms,
        "currents_baseline": currents_bl,
    }


def plot_metric3(data, show=True, out_dir=None):
    """Figure 3: E/I balance for PYR population."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    baseline_pyr = CircuitParams().I_ext_pyr()
    baseline_pv = CircuitParams().I_ext_pv()

    # Panel A: E/I ratio vs I_ext_pyr
    ax = axes[0, 0]
    x = data["I_ext_pyr_vals"]
    ax.plot(x, data["ei_pyr_sweep"]["EI_ratio"], color="#333333", linewidth=2)
    ax.axvline(baseline_pyr, color="gray", ls="--", lw=1, alpha=0.6, label="Baseline")
    ax.axhline(1.0, color="red", ls=":", lw=1, alpha=0.5, label="E/I = 1")
    _style_ax(ax, xlabel="I_ext_pyr", ylabel="E/I ratio", title="E/I ratio vs I_ext_PYR")
    ax.legend(fontsize=9)

    # Panel B: Shunting factor vs I_ext_pv
    ax = axes[0, 1]
    ax.plot(data["I_ext_pv_vals"], data["shunting_vs_pv"], color=POPULATION_COLORS["PV"],
            linewidth=2)
    ax.axvline(baseline_pv, color="gray", ls="--", lw=1, alpha=0.6)
    _style_ax(ax, xlabel="I_ext_pv", ylabel="Shunting factor (1 + g*w*r_PV)",
              title="PV shunting vs I_ext_PV")

    # Panel C: Time series of E and I at baseline
    ax = axes[1, 0]
    t = data["t_baseline"]
    mask = t >= BURN_IN_MS
    t_plot = t[mask]
    c = data["currents_baseline"]
    ax.plot(t_plot, c["E"][mask], color=POPULATION_COLORS["PYR"], label="Excitation (E)",
            linewidth=1, alpha=0.8)
    ax.plot(t_plot, c["I"][mask], color=POPULATION_COLORS["PV"], label="Inhibition (I)",
            linewidth=1, alpha=0.8)
    _style_ax(ax, xlabel="Time (ms)", ylabel="Current", title="E and I currents to PYR (baseline)")
    ax.legend(fontsize=9)

    # Panel D: Stacked decomposition vs I_ext_pyr
    ax = axes[1, 1]
    x = data["I_ext_pyr_vals"]
    sw = data["ei_pyr_sweep"]
    ax.plot(x, sw["I_exc"], color=POPULATION_COLORS["PYR"], label="Recurrent exc (shunted)",
            linewidth=1.5)
    ax.plot(x, sw["I_ext"], color="#888888", label="I_ext_pyr", linewidth=1.5, ls="--")
    ax.plot(x, -sw["I_inh_som"], color=POPULATION_COLORS["SOM"],
            label="-SOM inh", linewidth=1.5)
    ax.plot(x, -sw["I_adapt"], color="#D55E00", label="-I_adapt", linewidth=1.5)
    ax.axvline(baseline_pyr, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    _style_ax(ax, xlabel="I_ext_pyr", ylabel="Current component",
              title="PYR input decomposition vs I_ext_PYR")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Metric 3: E/I Balance", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric3_ei_balance.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 4: Response to Perturbations
# =========================================================================
def compute_metric4(params: CircuitParams):
    """Per-population perturbation: 500ms pulse, measure response."""
    results = {}

    # Pre-compute true steady state (long warmup, ~3.5x SOM tau_adapt)
    print("  Computing steady state (warmup)...")
    r_ss, adapt_ss = get_steady_state(params)

    baseline_ms = 500.0
    pulse_ms = 500.0
    recovery_ms = 1500.0
    pbar = tqdm(POPULATION_NAMES, desc="M4: Perturbations", unit="pop")

    for pop in pbar:
        field = I0_FIELDS[pop]
        I0_base = getattr(params, field)
        delta = 0.2 * I0_base  # +20% of baseline I0

        t, r = run_perturbation(params, pop, delta,
                                r0=r_ss, I_adapt0=adapt_ss,
                                baseline_ms=baseline_ms, pulse_ms=pulse_ms,
                                recovery_ms=recovery_ms)

        # Steady-state rate (end of baseline phase, should be flat)
        idx_ss = int((baseline_ms * 0.5) / DT_MS)

        # Pulse region
        pulse_start_idx = int(baseline_ms / DT_MS)
        pulse_end_idx = pulse_start_idx + int(pulse_ms / DT_MS)
        recovery_start_idx = pulse_end_idx

        # Peak deviation (all pops)
        peak_dev = np.zeros(4)
        time_to_peak = np.zeros(4)
        for i in range(4):
            r_ss_i = r[idx_ss, i]
            deviation = r[pulse_start_idx:, i] - r_ss_i
            abs_dev = np.abs(deviation)
            peak_idx = np.argmax(abs_dev)
            peak_dev[i] = deviation[peak_idx]
            time_to_peak[i] = peak_idx * DT_MS

        # Decay tau: fit exponential to PYR recovery phase
        t_recovery = t[recovery_start_idx:]
        r_pyr_recovery = r[recovery_start_idx:, 0]
        fit_result = fit_exponential_decay(t_recovery, r_pyr_recovery)
        decay_tau = fit_result[1] if fit_result is not None else np.nan

        results[pop] = {
            "t": t,
            "r": r,
            "peak_deviation": peak_dev,
            "time_to_peak": time_to_peak,
            "decay_tau": decay_tau,
            "delta": delta,
            "pulse_start_ms": baseline_ms,
            "pulse_end_ms": baseline_ms + pulse_ms,
        }

    pbar.close()
    return results


def plot_metric4(data, show=True, out_dir=None):
    """Figure 4: Perturbation responses."""
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1])

    # Top row: time traces (one panel per perturbed pop)
    for j, perturbed_pop in enumerate(POPULATION_NAMES):
        ax = fig.add_subplot(gs[0, j])
        d = data[perturbed_pop]
        t = d["t"]
        r = d["r"]
        pulse_start = d["pulse_start_ms"]
        pulse_end = d["pulse_end_ms"]

        for i, pop in enumerate(POPULATION_NAMES):
            ax.plot(t, r[:, i], color=POPULATION_COLORS[pop],
                    label=pop, linewidth=1.2)

        # Shade pulse window
        ax.axvspan(pulse_start, pulse_end, alpha=0.15, color="#888888")
        _style_ax(ax, xlabel="Time (ms)", ylabel="Firing rate",
                  title=f"Perturb {perturbed_pop}\n(+{d['delta']:.2f})")
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    # Middle row: peak deviation bar plots
    pops_list = POPULATION_NAMES
    for j, perturbed_pop in enumerate(POPULATION_NAMES):
        ax = fig.add_subplot(gs[1, j])
        d = data[perturbed_pop]
        colors = [POPULATION_COLORS[p] for p in pops_list]
        bars = ax.bar(range(4), d["peak_deviation"], color=colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(pops_list, fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        _style_ax(ax, ylabel="Peak deviation",
                  title=f"Peak response\n(perturb {perturbed_pop})")

    # Bottom row: summary heatmap of peak deviations + decay tau bar
    ax_heat = fig.add_subplot(gs[2, :2])
    peak_matrix = np.zeros((4, 4))
    for j, perturbed_pop in enumerate(POPULATION_NAMES):
        peak_matrix[:, j] = data[perturbed_pop]["peak_deviation"]
    im = ax_heat.imshow(peak_matrix, cmap="RdBu_r", aspect="auto",
                        vmin=-np.max(np.abs(peak_matrix)),
                        vmax=np.max(np.abs(peak_matrix)))
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels(pops_list, fontsize=10)
    ax_heat.set_yticks(range(4))
    ax_heat.set_yticklabels(pops_list, fontsize=10)
    ax_heat.set_xlabel("Perturbed population")
    ax_heat.set_ylabel("Measured population")
    ax_heat.set_title("Peak deviation matrix", fontweight="bold")
    for ii in range(4):
        for jj in range(4):
            ax_heat.text(jj, ii, f"{peak_matrix[ii, jj]:.2f}", ha="center",
                         va="center", fontsize=8,
                         color="white" if abs(peak_matrix[ii, jj]) > 0.5 * np.max(np.abs(peak_matrix)) else "black")
    fig.colorbar(im, ax=ax_heat, shrink=0.7)

    ax_tau = fig.add_subplot(gs[2, 2:])
    taus = [data[p]["decay_tau"] for p in POPULATION_NAMES]
    colors = [POPULATION_COLORS[p] for p in POPULATION_NAMES]
    bars = ax_tau.bar(range(4), taus, color=colors, alpha=0.8)
    ax_tau.set_xticks(range(4))
    ax_tau.set_xticklabels([f"Perturb\n{p}" for p in POPULATION_NAMES], fontsize=9)
    _style_ax(ax_tau, ylabel="Decay tau (ms)", title="PYR recovery time constant")
    for bar, val in zip(bars, taus):
        if not np.isnan(val):
            ax_tau.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Metric 4: Response to Perturbations", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric4_perturbation_responses.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 5: Sensory Responsiveness
# =========================================================================
def compute_metric5(params: CircuitParams):
    """PYR transient response at different baseline I_ext levels."""
    I0_levels = np.linspace(2.0, 12.0, 8)
    pulse_amplitude = 2.0
    baseline_ms = 500.0
    pulse_ms = 500.0
    recovery_ms = 1000.0

    results = {
        "I0_levels": I0_levels,
        "traces": [],       # list of (t, r) per level
        "baseline_rate": [],
        "peak_rate": [],
        "sensitivity": [],
    }

    for I0_val in tqdm(I0_levels, desc="M5: Sensory Response", unit="level"):
        base_params = replace(params, I0_pyr=I0_val)

        # Each baseline level has its own steady state
        r_ss, adapt_ss = get_steady_state(base_params)

        t, r = run_perturbation(base_params, "PYR", pulse_amplitude,
                                r0=r_ss, I_adapt0=adapt_ss,
                                baseline_ms=baseline_ms, pulse_ms=pulse_ms,
                                recovery_ms=recovery_ms)

        # Baseline PYR rate (mid-baseline, should be flat)
        idx_ss = int((baseline_ms * 0.5) / DT_MS)
        pyr_baseline = r[idx_ss, 0]

        # Peak during pulse
        pulse_start = int(baseline_ms / DT_MS)
        pulse_end = pulse_start + int(pulse_ms / DT_MS)
        pyr_peak = np.max(r[pulse_start:pulse_end, 0])

        delta_rate = pyr_peak - pyr_baseline
        sensitivity = delta_rate / pulse_amplitude

        results["traces"].append((t, r))
        results["baseline_rate"].append(pyr_baseline)
        results["peak_rate"].append(pyr_peak)
        results["sensitivity"].append(sensitivity)

    for k in ["baseline_rate", "peak_rate", "sensitivity"]:
        results[k] = np.array(results[k])

    return results


def plot_metric5(data, show=True, out_dir=None):
    """Figure 5: Sensory responsiveness."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cmap = plt.cm.viridis
    n_levels = len(data["I0_levels"])

    # Panel A: overlaid PYR traces
    ax = axes[0]
    for idx, (t, r) in enumerate(data["traces"]):
        color = cmap(idx / max(n_levels - 1, 1))
        label = f"I0={data['I0_levels'][idx]:.1f}"
        ax.plot(t, r[:, 0], color=color, linewidth=1.2, label=label)
    # Shade pulse window (baseline_ms to baseline_ms + pulse_ms)
    ax.axvspan(500.0, 1000.0, alpha=0.1, color="#888888")
    _style_ax(ax, xlabel="Time (ms)", ylabel="PYR firing rate",
              title="PYR response at different baselines")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)

    # Panel B: sensitivity vs baseline
    ax = axes[1]
    ax.plot(data["I0_levels"], data["sensitivity"], "o-", color=POPULATION_COLORS["PYR"],
            linewidth=2, markersize=6)
    _style_ax(ax, xlabel="Baseline I0_pyr", ylabel="Sensitivity (dRate / dI)",
              title="Transient sensitivity")

    # Panel C: peak rate vs baseline
    ax = axes[2]
    ax.plot(data["I0_levels"], data["baseline_rate"], "s--", color="#888888",
            label="Baseline rate", linewidth=1.5, markersize=5)
    ax.plot(data["I0_levels"], data["peak_rate"], "o-", color=POPULATION_COLORS["PYR"],
            label="Peak rate", linewidth=2, markersize=6)
    _style_ax(ax, xlabel="Baseline I0_pyr", ylabel="PYR firing rate",
              title="Baseline vs peak rate")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.suptitle("Metric 5: Sensory Responsiveness", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric5_sensory_responsiveness.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 6: Disinhibition Efficacy
# =========================================================================
def compute_metric6(params: CircuitParams):
    """Sweep VIP current, track VIP->SOM->PYR cascade."""
    I0_vip_vals = np.linspace(*SWEEP_RANGES["VIP"], N_SWEEP)
    baseline_receptor_vip = _get_total_I_ext(params, "VIP") - params.I0_vip
    I_ext_vip_vals = I0_vip_vals + baseline_receptor_vip

    pbar = tqdm(total=N_SWEEP + 1, desc="M6: Disinhibition", unit="sim")

    rates = run_sweep(params, "VIP", I0_vip_vals)
    pbar.update(1)

    # Reconstruct SOM inhibitory current onto PYR at each sweep point
    som_inh_on_pyr = []
    ei_ratio = []
    for val in I0_vip_vals:
        p = replace(params, I0_vip=val)
        result = simulate_circuit(p, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none")
        c = reconstruct_pyr_currents(result, p)
        som_inh_on_pyr.append(np.mean(c["I_inh_som"][-1000:]))
        ei_ratio.append(np.mean(c["EI_ratio"][-1000:]))
        pbar.update(1)
    pbar.close()
    som_inh_on_pyr = np.array(som_inh_on_pyr)
    ei_ratio = np.array(ei_ratio)

    # Disinhibition gain: d(r_PYR)/d(I_ext_VIP)
    disinhibition_gain = np.gradient(rates[:, 0], I0_vip_vals)

    return {
        "I0_vip_vals": I0_vip_vals,
        "I_ext_vip_vals": I_ext_vip_vals,
        "rates": rates,
        "som_inh_on_pyr": som_inh_on_pyr,
        "ei_ratio": ei_ratio,
        "disinhibition_gain": disinhibition_gain,
    }


def plot_metric6(data, show=True, out_dir=None):
    """Figure 6: Disinhibition efficacy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    x = data["I_ext_vip_vals"]
    baseline_vip = CircuitParams().I_ext_vip()

    # Panel A: VIP, SOM, PYR rates vs I_ext_vip
    ax = axes[0]
    for i, pop in enumerate(["VIP", "SOM", "PYR"]):
        idx = POP_INDICES[pop]
        ax.plot(x, data["rates"][:, idx], color=POPULATION_COLORS[pop],
                label=pop, linewidth=2)
    ax.axvline(baseline_vip, color="gray", ls="--", lw=1, alpha=0.6)
    _style_ax(ax, xlabel="I_ext_vip", ylabel="Firing rate",
              title="VIP → SOM → PYR cascade")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    # Panel B: SOM inhibitory current on PYR
    ax = axes[1]
    ax.plot(x, data["som_inh_on_pyr"], color=POPULATION_COLORS["SOM"], linewidth=2)
    ax.axvline(baseline_vip, color="gray", ls="--", lw=1, alpha=0.6)
    _style_ax(ax, xlabel="I_ext_vip", ylabel="SOM → PYR inhibitory current",
              title="SOM inhibition on PYR")

    # Panel C: Disinhibition gain
    ax = axes[2]
    ax.plot(x, data["disinhibition_gain"], color=POPULATION_COLORS["PYR"], linewidth=2)
    ax.axvline(baseline_vip, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    _style_ax(ax, xlabel="I_ext_vip", ylabel="d(r_PYR) / d(I_ext_VIP)",
              title="Disinhibition gain")

    fig.suptitle("Metric 6: Disinhibition Efficacy", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric6_disinhibition_efficacy.png", show=show, out_dir=out_dir)


# =========================================================================
# METRIC 7: Adaptation Strength
# =========================================================================
def compute_metric7(params: CircuitParams):
    """Adaptation buildup at different I0_pyr levels."""
    I0_levels = np.linspace(2.0, 12.0, 6)

    results = {
        "I0_levels": I0_levels,
        "traces_adapt_pyr": [],
        "traces_adapt_som": [],
        "traces_r_pyr": [],
        "t_ms": None,
        "adapt_ss_pyr": [],
        "adapt_ss_som": [],
        "rate_peak_pyr": [],
        "rate_ss_pyr": [],
        "rate_suppression": [],
    }

    for I0_val in tqdm(I0_levels, desc="M7: Adaptation", unit="level"):
        p = replace(params, I0_pyr=I0_val)
        result = simulate_circuit(p, T_ms=T_ADAPT_MS, dt_ms=DT_MS, noise_type="none")

        if results["t_ms"] is None:
            results["t_ms"] = result.t_ms

        results["traces_adapt_pyr"].append(result.I_adapt[:, 0])
        results["traces_adapt_som"].append(result.I_adapt[:, 1])
        results["traces_r_pyr"].append(result.r[:, 0])

        # Steady-state adaptation (last 500ms)
        adapt_ss_pyr = np.mean(result.I_adapt[-5000:, 0])
        adapt_ss_som = np.mean(result.I_adapt[-5000:, 1])
        results["adapt_ss_pyr"].append(adapt_ss_pyr)
        results["adapt_ss_som"].append(adapt_ss_som)

        # Peak rate: global max of PYR trace (transient overshoot before
        # adaptation settles). Clamp suppression to >= 0 (no overshoot = 0%).
        rate_peak = np.max(result.r[:, 0])
        rate_ss = np.mean(result.r[-5000:, 0])
        results["rate_peak_pyr"].append(rate_peak)
        results["rate_ss_pyr"].append(rate_ss)
        suppression = (rate_peak - rate_ss) / max(rate_peak, 1e-6)
        results["rate_suppression"].append(max(suppression, 0.0))

    for k in ["adapt_ss_pyr", "adapt_ss_som", "rate_peak_pyr", "rate_ss_pyr",
              "rate_suppression"]:
        results[k] = np.array(results[k])

    return results


def plot_metric7(data, show=True, out_dir=None):
    """Figure 7: Adaptation strength."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    t = data["t_ms"]
    cmap = plt.cm.copper
    n_levels = len(data["I0_levels"])

    # Panel A: I_adapt_pyr time courses
    ax = axes[0, 0]
    for idx in range(n_levels):
        color = cmap(idx / max(n_levels - 1, 1))
        label = f"I0={data['I0_levels'][idx]:.1f}"
        ax.plot(t, data["traces_adapt_pyr"][idx], color=color, linewidth=1.2,
                label=label)
    _style_ax(ax, xlabel="Time (ms)", ylabel="I_adapt_pyr",
              title="PYR adaptation current buildup")
    ax.legend(fontsize=7, loc="best")

    # Panel B: I_adapt_som time courses
    ax = axes[0, 1]
    for idx in range(n_levels):
        color = cmap(idx / max(n_levels - 1, 1))
        label = f"I0={data['I0_levels'][idx]:.1f}"
        ax.plot(t, data["traces_adapt_som"][idx], color=color, linewidth=1.2,
                label=label)
    _style_ax(ax, xlabel="Time (ms)", ylabel="I_adapt_som",
              title="SOM adaptation current buildup")
    ax.legend(fontsize=7, loc="best")

    # Panel C: Steady-state adaptation vs I0
    ax = axes[1, 0]
    ax.plot(data["I0_levels"], data["adapt_ss_pyr"], "o-",
            color="#D55E00", linewidth=2, markersize=6, label="PYR adapt")
    ax.plot(data["I0_levels"], data["adapt_ss_som"], "s-",
            color="#0072B2", linewidth=2, markersize=6, label="SOM adapt")
    _style_ax(ax, xlabel="I0_pyr", ylabel="Steady-state I_adapt",
              title="Steady-state adaptation vs input")
    ax.legend(fontsize=9)

    # Panel D: Rate suppression vs I0
    ax = axes[1, 1]
    ax.plot(data["I0_levels"], data["rate_suppression"] * 100, "o-",
            color=POPULATION_COLORS["PYR"], linewidth=2, markersize=6)
    _style_ax(ax, xlabel="I0_pyr",
              ylabel="Rate suppression (%)",
              title="Adaptation-induced rate suppression")
    ax.set_ylim(bottom=0)

    fig.suptitle("Metric 7: Adaptation Strength", fontsize=14, fontweight="bold")
    _save_fig(fig, "metric7_adaptation_strength.png", show=show, out_dir=out_dir)


# =========================================================================
# Summary metrics serialization
# =========================================================================
def _to_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


def _count_elements(obj) -> int:
    """Count total elements in a nested structure."""
    if isinstance(obj, np.ndarray):
        return obj.size
    if isinstance(obj, (list, tuple)):
        return sum(_count_elements(v) for v in obj)
    return 1


def _prune_large(obj, max_elements=500):
    """Recursively prune numpy arrays / lists / tuples that exceed max_elements."""
    if isinstance(obj, np.ndarray):
        if obj.size > max_elements:
            return None
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pruned = _prune_large(v, max_elements)
            if pruned is not None:
                out[k] = pruned
        return out
    if isinstance(obj, (list, tuple)):
        if _count_elements(obj) > max_elements:
            return None
        cleaned = [_prune_large(v, max_elements) for v in obj]
        cleaned = [v for v in cleaned if v is not None]
        return cleaned
    return obj


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON, pruning large trace arrays."""
    pruned = _prune_large(metrics)
    serializable = _to_serializable(pruned)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics saved to: {path}")


# =========================================================================
# Main
# =========================================================================
def _print_metric_summary(m: int, data: dict):
    """Print a concise summary for each metric."""
    if m == 1:
        for pop in POPULATION_NAMES:
            d = data[pop]
            rates = d["rates_det"]
            print(f"  {pop} sweep: rate range = [{rates[:, POP_INDICES[pop]].min():.2f}, "
                  f"{rates[:, POP_INDICES[pop]].max():.2f}]")
    elif m == 2:
        gm = data["gain_matrix"]
        print("  Gain matrix (dF/dI) at baseline:")
        header = "         " + "  ".join(f"{p:>6}" for p in POPULATION_NAMES)
        print(header)
        for i, pop in enumerate(POPULATION_NAMES):
            row = "  ".join(f"{gm[i, j]:+6.2f}" for j in range(4))
            print(f"  {pop:>6}  {row}")
    elif m == 3:
        ei = data["ei_pyr_sweep"]["EI_ratio"]
        print(f"  E/I ratio range (PYR sweep): [{ei.min():.2f}, {ei.max():.2f}]")
    elif m == 4:
        for pop in POPULATION_NAMES:
            d = data[pop]
            tau = d["decay_tau"]
            peak = d["peak_deviation"][0]
            tau_str = f"{tau:.1f} ms" if not np.isnan(tau) else "N/A"
            print(f"  Perturb {pop}: PYR peak dev = {peak:+.3f}, recovery tau = {tau_str}")
    elif m == 5:
        sens = data["sensitivity"]
        print(f"  Sensitivity range: [{sens.min():.3f}, {sens.max():.3f}]")
    elif m == 6:
        gain = data["disinhibition_gain"]
        print(f"  Disinhibition gain range: [{gain.min():.3f}, {gain.max():.3f}]")
    elif m == 7:
        supp = data["rate_suppression"] * 100
        print(f"  Rate suppression range: [{supp.min():.1f}%, {supp.max():.1f}%]")
        adapt = data["adapt_ss_pyr"]
        print(f"  SS adaptation (PYR) range: [{adapt.min():.3f}, {adapt.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description="External current sensitivity analysis for 4-population PFC circuit model."
    )
    parser.add_argument("--fast", action="store_true",
                        help="Skip noisy CV calculations (faster)")
    parser.add_argument("--metrics", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                        help="Which metrics to compute (1-7)")
    parser.add_argument("--params_json", type=str, default="",
                        help="Load parameters from JSON file")
    parser.add_argument("--no_show", action="store_true",
                        help="Save figures without displaying")
    args = parser.parse_args()

    # Detect display availability
    show = not args.no_show
    if show:
        display_ok = (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not display_ok:
            try:
                from IPython import get_ipython
                if get_ipython() is None:
                    show = False
            except ImportError:
                show = False
        if not show:
            print("No display detected, saving figures only.")
            matplotlib.use("Agg")

    # Load parameters
    if args.params_json:
        params = load_params_json(args.params_json)
        print(f"Loaded parameters from: {args.params_json}")
    else:
        params = CircuitParams()
        print("Using default parameters.")

    # Derive output directory from params file
    if args.params_json:
        from pathlib import Path
        out_dir = os.path.join(OUT_DIR, Path(args.params_json).stem)
    else:
        out_dir = os.path.join(OUT_DIR, "default")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline characterization
    print("\nBaseline external currents:")
    for pop in POPULATION_NAMES:
        print(f"  {pop}: I_ext = {_get_total_I_ext(params, pop):.3f}")

    result_bl = simulate_circuit(params, T_ms=T_SWEEP_MS, dt_ms=DT_MS, noise_type="none")
    baseline_rates = mean_rates(result_bl, burn_in_ms=BURN_IN_MS, window_ms=0)
    print("\nBaseline firing rates:")
    for i, pop in enumerate(POPULATION_NAMES):
        print(f"  {pop}: {baseline_rates[i]:.3f}")

    all_metrics: dict = {
        "baseline_rates": dict(zip(POPULATION_NAMES, baseline_rates.tolist())),
        "baseline_I_ext": {pop: _get_total_I_ext(params, pop) for pop in POPULATION_NAMES},
    }

    # Run selected metrics
    metric_funcs = {
        1: ("Firing Rate Statistics", compute_metric1, plot_metric1,
            {"params": params, "fast": args.fast}),
        2: ("Gain Modulation", compute_metric2, plot_metric2,
            {"params": params}),
        3: ("E/I Balance", compute_metric3, plot_metric3,
            {"params": params}),
        4: ("Response to Perturbations", compute_metric4, plot_metric4,
            {"params": params}),
        5: ("Sensory Responsiveness", compute_metric5, plot_metric5,
            {"params": params}),
        6: ("Disinhibition Efficacy", compute_metric6, plot_metric6,
            {"params": params}),
        7: ("Adaptation Strength", compute_metric7, plot_metric7,
            {"params": params}),
    }

    selected = [m for m in sorted(args.metrics) if m in metric_funcs]
    unknown = [m for m in args.metrics if m not in metric_funcs]
    for m in unknown:
        print(f"Unknown metric: {m}, skipping.")

    for m in selected:
        name, compute_fn, plot_fn, kwargs = metric_funcs[m]
        data = compute_fn(**kwargs)
        all_metrics[f"metric{m}"] = data
        plot_fn(data, show=show, out_dir=out_dir)

    # Print results summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline firing rates:")
    for pop in POPULATION_NAMES:
        print(f"  {pop}: {all_metrics['baseline_rates'][pop]:.3f}")

    for m in selected:
        name = metric_funcs[m][0]
        data = all_metrics[f"metric{m}"]
        print(f"\nMetric {m}: {name}")
        _print_metric_summary(m, data)

    # Save all metrics
    save_metrics(all_metrics, os.path.join(out_dir, "metrics_summary.json"))

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
