"""
Diagnostic plotting for circuit parameters: Turing gain product and transfer functions.

This module provides analytical (no-simulation) diagnostic plots:
  1. Turing gain product vs PYR firing rate (with marked operating points)
  2. Transfer functions for all 4 populations with operating point markers
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .params import CircuitParams
from .transfer import phi_wong_wang
from .loss import transfer_function_slope
from .jacobian import _phi_derivative, _total_inputs


def _invert_transfer_function(
    params: CircuitParams,
    target_rate: float,
    population: str = "PYR",
    I_bounds: tuple[float, float] = (0.0, 2.0),
) -> float:
    """
    Invert the transfer function to find I such that Phi(I) = target_rate.

    Uses Brent's method (robust root-finding).

    Parameters
    ----------
    params : CircuitParams
    target_rate : Desired firing rate (Hz)
    population : "PYR", "SOM", "PV", or "VIP"
    I_bounds : (I_min, I_max) search range (nA)

    Returns
    -------
    I_star : Input current such that Phi(I_star) = target_rate (nA)
    """
    if population == "PYR":
        A, theta, c, g = params.A_pyr, params.Theta_pyr, params.alpha_pyr, params.g_exc
    elif population == "SOM":
        A, theta, c, g = params.A_som, params.Theta_som, params.alpha_som, params.g_inh
    elif population == "PV":
        A, theta, c, g = params.A_pv, params.Theta_pv, params.alpha_pv, params.g_inh
    elif population == "VIP":
        A, theta, c, g = params.A_vip, params.Theta_vip, params.alpha_vip, params.g_inh
    else:
        raise ValueError(f"Unknown population: {population}")

    def objective(I):
        rate = phi_wong_wang(I, theta=theta, c=c, g=g, A=A)
        return float(rate) - target_rate

    # Check bounds
    I_min, I_max = I_bounds
    f_min = objective(I_min)
    f_max = objective(I_max)

    if f_min * f_max > 0:
        # No sign change: clamp to boundary
        if abs(f_min) < abs(f_max):
            return I_min
        else:
            return I_max

    return brentq(objective, I_min, I_max)


def _solve_full_fixed_point(params: CircuitParams) -> Optional[float]:
    """Find the true 4-population fixed-point for a given set of circuit params.

    Solves r_pyr such that Phi_pyr(I_pyr(r_pyr, r_som(r_pyr), r_pv(r_pyr), r_vip(r_pyr))) = r_pyr,
    where r_som, r_pv, r_vip are themselves solved consistently via
    _solve_constrained_steady_state.

    Returns
    -------
    r_pyr : equilibrium PYR rate (Hz), or None if no root found.
    """
    def residual(r_pyr: float) -> float:
        r_pyr_c = max(r_pyr, 0.0)
        try:
            r_som, r_pv, r_vip = _solve_constrained_steady_state(params, r_pyr_c)
        except ValueError:
            return 1.0
        r_ss = np.array([r_pyr_c, r_som, r_pv, r_vip])
        I_pyr, _, _, _ = _total_inputs(params, r_ss)
        r_pred = float(phi_wong_wang(
            I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr,
            g=params.g_exc, A=params.A_pyr,
        ))
        return r_pred - r_pyr

    try:
        f_lo = residual(0.01)
        f_hi = residual(200.0)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return None
        if f_lo * f_hi <= 0:
            return float(brentq(residual, 0.01, 200.0))
        # No sign change between 0–200: scan finely for a crossing
        r_scan = np.linspace(0.01, 200.0, 200)
        f_prev = f_lo
        for k in range(1, len(r_scan)):
            f_cur = residual(r_scan[k])
            if np.isfinite(f_prev) and np.isfinite(f_cur) and f_prev * f_cur <= 0:
                return float(brentq(residual, r_scan[k - 1], r_scan[k]))
            f_prev = f_cur
        return None
    except ValueError:
        return None


def _find_I0_scale_for_rate(
    params: CircuitParams,
    target_hz: float,
    n_scan: int = 60,
) -> Optional[float]:
    """Find the I0_pyr scale such that the true equilibrium r_pyr ≈ target_hz.

    Scans scales in [0.5, 2.5] to find a sign-crossing bracket, then refines
    with brentq.

    Parameters
    ----------
    params : base CircuitParams
    target_hz : desired equilibrium PYR firing rate (Hz)
    n_scan : number of scan points (default 60)

    Returns
    -------
    scale : I0_pyr multiplier, or None if no solution found in the scan range.
    """
    from dataclasses import replace as _replace

    base_I0 = params.I0_pyr
    scales = np.linspace(0.5, 2.5, n_scan)

    def r_eq_at_scale(scale: float) -> Optional[float]:
        p = _replace(params, I0_pyr=base_I0 * scale)
        return _solve_full_fixed_point(p)

    # Scan to find a bracket [s_lo, s_hi] where r_eq crosses target_hz
    r_prev = r_eq_at_scale(scales[0])
    for k in range(1, len(scales)):
        r_cur = r_eq_at_scale(scales[k])
        if r_prev is not None and r_cur is not None:
            if (r_prev - target_hz) * (r_cur - target_hz) <= 0:
                # Sign change found — refine with brentq
                def residual(scale: float) -> float:
                    r = r_eq_at_scale(scale)
                    return (r - target_hz) if r is not None else -target_hz
                try:
                    return float(brentq(residual, scales[k - 1], scales[k], xtol=1e-4))
                except ValueError:
                    pass
        r_prev = r_cur

    return None


def _solve_constrained_steady_state(
    params: CircuitParams,
    r_pyr: float,
) -> tuple[float, float, float]:
    """
    Given a fixed r_pyr, solve for the consistent steady-state rates of SOM, PV, VIP.

    The circuit structure allows sequential resolution:
      - VIP depends only on r_pyr  → solved directly.
      - SOM depends on r_pyr and r_vip → solved directly after VIP.
      - PV has self-inhibition (w_pp) → solved with Brent's method.

    Parameters
    ----------
    params : CircuitParams
    r_pyr : PYR firing rate held fixed (Hz)

    Returns
    -------
    (r_som, r_pv, r_vip)
    """
    # VIP: I_vip = w_ev * r_pyr + I0_vip  (no feedback from other pops)
    I_vip = params.w_ev * r_pyr + params.I_ext_vip()
    r_vip = float(phi_wong_wang(I_vip, theta=params.Theta_vip,
                                 c=params.alpha_vip, g=params.g_inh, A=params.A_vip))

    # SOM: I_som = w_es * r_pyr - w_vs * r_vip + I0_som
    I_som = params.w_es * r_pyr - params.w_vs * r_vip + params.I_ext_som()
    r_som = float(phi_wong_wang(I_som, theta=params.Theta_som,
                                 c=params.alpha_som, g=params.g_inh, A=params.A_som))

    # PV: implicit via self-inhibition
    # r_pv = Phi_pv(w_ep*r_pyr - g*w_pp*r_pv - g*w_sp*r_som - w_vp*r_vip + I0_pv)
    ggaba = params.g_gaba()

    def pv_residual(r_pv: float) -> float:
        I_pv = (params.w_ep * r_pyr
                - ggaba * params.w_pp * r_pv
                - ggaba * params.w_sp * r_som
                - params.w_vp * r_vip
                + params.I_ext_pv())
        return float(phi_wong_wang(I_pv, theta=params.Theta_pv,
                                    c=params.alpha_pv, g=params.g_inh, A=params.A_pv)) - r_pv

    # The residual is guaranteed to cross zero: at r_pv=0 it is positive (drive > 0),
    # and at large r_pv it is negative (self-inhibition dominates).
    r_pv = brentq(pv_residual, 0.0, 500.0)

    return r_som, r_pv, r_vip


def _turing_gain_at_ss(
    params: CircuitParams,
    r_ss: np.ndarray,
    w_pyr_inter: float,
) -> float:
    """Compute Turing gain product at a given steady-state rate vector."""
    I_pyr, _, I_pv, _ = _total_inputs(params, r_ss)
    ggaba = params.g_gaba()
    phi_prime_pyr = params.A_pyr * _phi_derivative(
        I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc,
    )
    phi_prime_pv = params.A_pv * _phi_derivative(
        I_pv, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_inh,
    )
    denom = 1.0 + ggaba * params.w_pe * phi_prime_pv * params.w_ep * phi_prime_pyr
    if denom == 0.0:
        return 0.0
    return (phi_prime_pyr / denom) * w_pyr_inter


def compute_turing_gain_product(
    circuit_params: CircuitParams,
    ring_params: "RingParams",  # type: ignore
    r_pyr_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute Turing gain product = G_eff * w_pyr_inter along the true equilibrium manifold.

    Sweeps I0_pyr from 0.4× to 8× the base value, finds the true 4-population
    fixed point at each value (PYR self-consistent via _solve_full_fixed_point),
    computes the gain at each equilibrium, then interpolates onto r_pyr_grid.

    Parameters
    ----------
    circuit_params : CircuitParams
    ring_params : RingParams (must have w_pyr_pyr_inter)
    r_pyr_grid : array of PYR firing rates (Hz) for interpolation output

    Returns
    -------
    gain_product : array of Turing gain products (dimensionless), on r_pyr_grid
    """
    from dataclasses import replace as _replace

    base_I0 = circuit_params.I0_pyr
    scales = np.linspace(0.4, 8.0, 150)

    eq_r_pyrs: list[float] = []
    eq_gains: list[float] = []

    for scale in scales:
        p = _replace(circuit_params, I0_pyr=base_I0 * scale)
        r_eq = _solve_full_fixed_point(p)
        if r_eq is None:
            continue
        try:
            r_som, r_pv, r_vip = _solve_constrained_steady_state(p, r_eq)
        except ValueError:
            continue
        r_ss = np.array([r_eq, r_som, r_pv, r_vip])
        gain = _turing_gain_at_ss(p, r_ss, ring_params.w_pyr_pyr_inter)
        eq_r_pyrs.append(r_eq)
        eq_gains.append(gain)

    if len(eq_r_pyrs) < 2:
        return np.zeros_like(r_pyr_grid, dtype=float)

    eq_r_arr = np.array(eq_r_pyrs)
    eq_g_arr = np.array(eq_gains)
    sort_idx = np.argsort(eq_r_arr)
    eq_r_arr = eq_r_arr[sort_idx]
    eq_g_arr = eq_g_arr[sort_idx]

    return np.interp(r_pyr_grid, eq_r_arr, eq_g_arr,
                     left=eq_g_arr[0], right=eq_g_arr[-1])


def plot_turing_gain_product(
    circuit_params: CircuitParams,
    ring_params: "RingParams",  # type: ignore
    target_pyr: float = 8.0,
    turing_bump_hz: float = 40.0,
    turing_cue_hz: float = 60.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot Turing gain product vs PYR firing rate with marked operating points.

    Parameters
    ----------
    circuit_params : CircuitParams
    ring_params : RingParams
    target_pyr : Rest PYR firing rate (Hz), used to mark rest operating point
    turing_bump_hz : Target PYR rate (Hz) for the bump operating point marker
    turing_cue_hz : Target PYR rate (Hz) for the cue operating point marker
    save_path : Path to save figure (PNG)
    show : Whether to display the figure
    """
    import matplotlib.pyplot as plt

    # Create fine grid for PYR firing rate (0 to 80 Hz)
    r_pyr_grid = np.linspace(0.1, 80.0, 500)
    gain_product = compute_turing_gain_product(circuit_params, ring_params, r_pyr_grid)

    # Operating points are specified directly in Hz — clamp to grid range
    r_pyr_bump: Optional[float] = min(turing_bump_hz, 80.0) if turing_bump_hz > 0 else None
    r_pyr_cue:  Optional[float] = min(turing_cue_hz,  80.0) if turing_cue_hz  > 0 else None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot gain product
    ax.plot(r_pyr_grid, gain_product, linewidth=2.5, color="#1f77b4", label="G_eff · w_pyr_inter")

    # Horizontal dashed line at gain = 1
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Turing threshold (gain = 1)")

    # Shade regions
    ax.fill_between(r_pyr_grid, 1.0, gain_product, where=(gain_product >= 1.0), alpha=0.2, color="green", label="Gain > 1 (Turing unstable)")
    ax.fill_between(r_pyr_grid, gain_product, 1.0, where=(gain_product < 1.0), alpha=0.2, color="red", label="Gain < 1 (Turing stable)")

    # Mark rest operating point (blue)
    if target_pyr is not None and target_pyr < 80:
        gain_rest = np.interp(target_pyr, r_pyr_grid, gain_product)
        ax.axvline(target_pyr, color="blue", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot(target_pyr, gain_rest, "o", markersize=10, color="blue", zorder=5)
        ax.text(target_pyr, gain_rest + 0.1, f"Rest\n{target_pyr:.1f} Hz\nG={gain_rest:.3f}",
                ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    # Mark bump operating point (green)
    if r_pyr_bump is not None and 0 < r_pyr_bump < 80:
        gain_bump = np.interp(r_pyr_bump, r_pyr_grid, gain_product)
        ax.axvline(r_pyr_bump, color="green", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot(r_pyr_bump, gain_bump, "s", markersize=10, color="green", zorder=5)
        ax.text(r_pyr_bump, gain_bump + 0.1, f"Bump\n{r_pyr_bump:.1f} Hz\nG={gain_bump:.3f}",
                ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    # Mark cue operating point (orange)
    if r_pyr_cue is not None and 0 < r_pyr_cue < 80:
        gain_cue = np.interp(r_pyr_cue, r_pyr_grid, gain_product)
        ax.axvline(r_pyr_cue, color="orange", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot(r_pyr_cue, gain_cue, "^", markersize=10, color="orange", zorder=5)
        ax.text(r_pyr_cue, gain_cue - 0.15, f"Cue\n{r_pyr_cue:.1f} Hz\nG={gain_cue:.3f}",
                ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    # Formatting
    ax.set_xlabel("PYR Firing Rate (Hz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Turing Gain Product (G_eff · w_pyr_inter)", fontsize=12, fontweight="bold")
    ax.set_title("Turing Gain Product vs PYR Firing Rate", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(0, 80)

    # Print operating point values to console
    print("\n[TURING GAIN PRODUCT ANALYSIS]")
    print("=" * 65)
    print(f"Rest operating point:  r_pyr = {target_pyr:6.2f} Hz  →  G = {np.interp(target_pyr, r_pyr_grid, gain_product):6.4f}")
    if r_pyr_bump is not None:
        print(f"Bump operating point:  r_pyr = {r_pyr_bump:6.2f} Hz  →  G = {np.interp(r_pyr_bump, r_pyr_grid, gain_product):6.4f}")
    if r_pyr_cue is not None:
        print(f"Cue operating point:   r_pyr = {r_pyr_cue:6.2f} Hz  →  G = {np.interp(r_pyr_cue, r_pyr_grid, gain_product):6.4f}")
    print("=" * 65 + "\n")

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_transfer_functions_diagnostic(
    circuit_params: CircuitParams,
    target_pyr: float = 8.0,
    turing_bump_hz: float = 40.0,
    turing_cue_hz: float = 60.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot transfer functions for all 4 populations with operating point markers.

    Four subplots (2x2), one per population. Each shows:
      - Transfer function curve Phi(I)
      - Rest operating point (vertical dashed line, blue marker)
      - Bump operating point (green marker)
      - Cue operating point (orange marker)

    Operating points are expressed directly in Hz. For each target rate,
    the code finds the I0_pyr scale that produces that equilibrium, then solves
    all four populations to get the true steady-state currents.

    Parameters
    ----------
    circuit_params : CircuitParams
    target_pyr : Rest PYR firing rate (Hz)
    turing_bump_hz : Bump operating point PYR rate (Hz), default 40.0
    turing_cue_hz : Cue operating point PYR rate (Hz), default 60.0
    save_path : Path to save figure (PNG)
    show : Whether to display the figure
    """
    import matplotlib.pyplot as plt
    from dataclasses import replace as _replace

    from matplotlib.lines import Line2D

    populations = ["PYR", "SOM", "PV", "VIP"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # --- Solve TRUE fixed points for each operating point ---
    # REST: solve all 4 pops at base I0_pyr (with target_pyr as given)
    r_rest_pyr = _solve_full_fixed_point(circuit_params) or target_pyr
    r_som_rest, r_pv_rest, r_vip_rest = _solve_constrained_steady_state(circuit_params, r_rest_pyr)
    r_rest_all = np.array([r_rest_pyr, r_som_rest, r_pv_rest, r_vip_rest])

    # BUMP: find I0_pyr scale for turing_bump_hz, then solve all pops
    bump_scale = _find_I0_scale_for_rate(circuit_params, turing_bump_hz)
    if bump_scale is not None:
        p_bump = _replace(circuit_params, I0_pyr=bump_scale * circuit_params.I0_pyr)
        r_bump_pyr = _solve_full_fixed_point(p_bump) or turing_bump_hz
        r_som_bump, r_pv_bump, r_vip_bump = _solve_constrained_steady_state(p_bump, r_bump_pyr)
        r_bump_all = np.array([r_bump_pyr, r_som_bump, r_pv_bump, r_vip_bump])
    else:
        r_bump_all = r_rest_all * (turing_bump_hz / max(r_rest_pyr, 1.0))

    # CUE: find I0_pyr scale for turing_cue_hz, then solve all pops
    cue_scale = _find_I0_scale_for_rate(circuit_params, turing_cue_hz)
    if cue_scale is not None:
        p_cue = _replace(circuit_params, I0_pyr=cue_scale * circuit_params.I0_pyr)
        r_cue_pyr = _solve_full_fixed_point(p_cue) or turing_cue_hz
        r_som_cue, r_pv_cue, r_vip_cue = _solve_constrained_steady_state(p_cue, r_cue_pyr)
        r_cue_all = np.array([r_cue_pyr, r_som_cue, r_pv_cue, r_vip_cue])
    else:
        r_cue_all = np.minimum(r_rest_all * (turing_cue_hz / max(r_rest_pyr, 1.0)), 80.0)

    # PYR rates for the shared legend
    r_pyr_rest = float(r_rest_all[0])
    r_pyr_bump = float(r_bump_all[0])
    r_pyr_cue  = float(r_cue_all[0])

    # Operating point styles: (rate_array_index, color, marker, label_key)
    OP_STYLES = [
        ("rest", "blue",   "o"),
        ("bump", "green",  "s"),
        ("cue",  "orange", "^"),
    ]

    for idx, pop in enumerate(populations):
        ax = axes[idx]

        # Population-specific TF parameters
        if pop == "PYR":
            A, theta, c, g = circuit_params.A_pyr, circuit_params.Theta_pyr, circuit_params.alpha_pyr, circuit_params.g_exc
        elif pop == "SOM":
            A, theta, c, g = circuit_params.A_som, circuit_params.Theta_som, circuit_params.alpha_som, circuit_params.g_inh
        elif pop == "PV":
            A, theta, c, g = circuit_params.A_pv, circuit_params.Theta_pv, circuit_params.alpha_pv, circuit_params.g_inh
        else:  # VIP
            A, theta, c, g = circuit_params.A_vip, circuit_params.Theta_vip, circuit_params.alpha_vip, circuit_params.g_inh

        # Resolve I_star for each operating point
        rates = {"rest": r_rest_all[idx], "bump": r_bump_all[idx], "cue": r_cue_all[idx]}
        I_stars: dict[str, float] = {}
        for key, r_pop in rates.items():
            try:
                I_stars[key] = _invert_transfer_function(circuit_params, r_pop, population=pop, I_bounds=(0.0, 2.0))
            except ValueError:
                pass

        # Adaptive x range: span all operating points with padding; also always
        # include a bit below threshold so the knee is visible.
        all_I = list(I_stars.values())
        if all_I:
            I_span_lo = min(all_I)
            I_span_hi = max(all_I)
            span = max(I_span_hi - I_span_lo, 0.02)
            pad = max(span * 0.5, 0.05)
            # Also show ~0.05 nA below the threshold so the zero-rate region is visible
            x_lo = max(0.0, min(I_span_lo - pad, theta - 0.05))
            x_hi = min(2.0, I_span_hi + pad)
        else:
            x_lo, x_hi = 0.0, 1.5

        # Plot transfer function over the adaptive range
        I_grid = np.linspace(x_lo, x_hi, 500)
        phi = phi_wong_wang(I_grid, theta=theta, c=c, g=g, A=A)
        ax.plot(I_grid, phi, linewidth=2.5, color="#1f77b4")

        # Adaptive y range: 0 to max of (curve in range, highest operating point) + 20%
        r_max_in_range = max(float(phi.max()), max(rates.values()))
        ax.set_ylim(0, r_max_in_range * 1.20)

        # Plot operating points with vertical dashed lines
        for key, color, marker in OP_STYLES:
            if key not in I_stars:
                continue
            I_star = I_stars[key]
            r_op   = rates[key]
            ax.axvline(I_star, color=color, linestyle="--", linewidth=1.5, alpha=0.55)
            ax.plot(I_star, r_op, marker, markersize=9, color=color, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
            # Minimal annotation: just the nA value, offset so it doesn't overlap the marker
            ax.annotate(f"{I_star:.3f} nA", xy=(I_star, r_op),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_xlabel("Input Current (nA)", fontsize=11)
        ax.set_ylabel("Firing Rate (Hz)", fontsize=11)
        ax.set_title(f"{pop}  (A={A:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlim(x_lo, x_hi)
        ax.grid(True, alpha=0.3)

    # Shared figure legend: one entry per operating point, stating the PYR rate
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",   markersize=9,
               markeredgecolor="white", label=f"Rest  — PYR = {r_pyr_rest:.1f} Hz"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green",  markersize=9,
               markeredgecolor="white", label=f"Bump  — PYR = {r_pyr_bump:.1f} Hz"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="orange", markersize=9,
               markeredgecolor="white", label=f"Cue   — PYR = {r_pyr_cue:.1f} Hz"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11,
               title="Operating points (PYR firing rate)", title_fontsize=11,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.suptitle("Transfer Functions with Operating Points", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Turing gain timecourse from ring simulation
# ---------------------------------------------------------------------------

def compute_turing_gain_timecourse(
    result: "RingSimulationResult",  # type: ignore
    circuit_params: CircuitParams,
    ring_params: "RingParams",  # type: ignore
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Turing gain product at every recorded timestep of a ring simulation.

    At each timestep two operating points are evaluated:
      - mean : average rates across all nodes (background / homogeneous state)
      - peak : rates at the node with maximum PYR firing rate (bump node)

    Parameters
    ----------
    result : RingSimulationResult with r of shape (n_recorded, n_nodes, 4)
    circuit_params : CircuitParams
    ring_params : RingParams

    Returns
    -------
    (t_ms, gain_mean, gain_peak) — each array of length n_recorded
    """
    r = result.r          # (n_recorded, n_nodes, 4): [pyr, som, pv, vip]
    t_ms = result.t_ms    # (n_recorded,)
    n_t = r.shape[0]

    gain_mean = np.zeros(n_t, dtype=float)
    gain_peak = np.zeros(n_t, dtype=float)
    w_inter = ring_params.w_pyr_pyr_inter

    for t in range(n_t):
        rates_t = r[t]  # (n_nodes, 4)
        # Background: mean rates
        r_mean = rates_t.mean(axis=0)
        gain_mean[t] = _turing_gain_at_ss(circuit_params, r_mean, w_inter)
        # Bump node: max PYR rate
        peak_node = int(np.argmax(rates_t[:, 0]))
        gain_peak[t] = _turing_gain_at_ss(circuit_params, rates_t[peak_node], w_inter)

    return t_ms, gain_mean, gain_peak


def plot_turing_gain_timecourse(
    result: "RingSimulationResult",  # type: ignore
    circuit_params: CircuitParams,
    ring_params: "RingParams",  # type: ignore
    t_offset: float = 0.0,
    time_range: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> "plt.Figure":  # type: ignore
    """
    Plot Turing gain product over time from a ring simulation.

    Two curves:
      - blue  solid : gain at mean rates across nodes (background state)
      - red   dashed: gain at peak PYR-rate node (bump node)

    A horizontal dashed line marks the Turing threshold at gain = 1.
    The cue / stimulus window is shaded if available in `result`.

    Parameters
    ----------
    result : RingSimulationResult
    circuit_params : CircuitParams
    ring_params : RingParams
    t_offset : Subtract from timestamps for display (e.g. burn-in duration in ms)
    time_range : Optional (t_start_ms, t_end_ms) in original (un-offset) time
    save_path : Path to save figure (PNG)
    show : Whether to display the figure

    Returns
    -------
    fig : Matplotlib figure
    """
    import matplotlib.pyplot as plt

    t_ms, gain_mean, gain_peak = compute_turing_gain_timecourse(
        result, circuit_params, ring_params
    )
    t_disp = t_ms - t_offset

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t_disp, gain_mean, linewidth=1.8, color="#1f77b4",
            label="Background (mean across nodes)")
    ax.plot(t_disp, gain_peak, linewidth=1.5, color="#d62728",
            linestyle="--", label="Bump node (peak PYR)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Turing threshold (gain = 1)")
    ax.fill_between(t_disp, 1.0, gain_mean,
                    where=(gain_mean >= 1.0), alpha=0.15, color="green")
    ax.fill_between(t_disp, gain_mean, 1.0,
                    where=(gain_mean < 1.0), alpha=0.15, color="red")

    # Shade cue/stimulus window if available
    if hasattr(result, "stim_window") and result.stim_window is not None:
        t_on  = result.stim_window[0] - t_offset
        t_off = result.stim_window[1] - t_offset
        ax.axvspan(t_on, t_off, color="orange", alpha=0.12, label="Cue period")

    if time_range is not None:
        ax.set_xlim(time_range[0] - t_offset, time_range[1] - t_offset)

    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Turing Gain Product  (G_eff · w_pyr_inter)", fontsize=11)
    ax.set_title("Turing Gain Product Over Time", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
