"""
Bistability loss function for finding parameters with bistable PYR nullcline.

This module implements the bistable_loss function which evaluates circuit parameters
based on whether they produce a bistable nullcline (two stable fixed points).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

from .params import CircuitParams
from .transfer import phi_wong_wang
from .jacobian import compute_jacobian
from .constants import R_MAX_PHYS, R_HIGH_MAX


@dataclass
class BistableConfig:
    """Configuration for bistability loss function."""
    # Target fixed point rates (Hz)
    r_low_target: float = 8.0       # Resting PYR rate
    r_high_target: float = 30.0     # Bump-active PYR rate
    r_mid_probe: float = 15.0       # Probe point, must be in unstable branch

    # Interneuron targets at rest (low fixed point)
    r_pv_target: float = 3.0        # PV rate
    r_som_target: float = 5.0       # SOM rate
    r_vip_target: float = 2.0       # VIP rate

    # Margin: minimum gap between low and high fixed points
    delta_r_min: float = 15.0       # Hz

    # Physiological ceiling: prevents optimizer from pushing high FP into clamp region
    r_high_max: float = R_HIGH_MAX  # Hz (default from constants, ~80)

    # Loss weights
    w_bistab: float = 1.0
    w_rate: float = 1.0
    w_margin: float = 0.5
    w_jacobian: float = 0.1         # Regularizer on max |J| entry

    # Condition (for informational purposes)
    condition: str = "WT"


def _phi_pyr(I: float, params: CircuitParams) -> float:
    """Transfer function for PYR."""
    return float(phi_wong_wang(I, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc))


def _phi_pv(I: float, params: CircuitParams) -> float:
    """Transfer function for PV."""
    return float(phi_wong_wang(I, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_inh))


def _phi_som(I: float, params: CircuitParams) -> float:
    """Transfer function for SOM."""
    return float(phi_wong_wang(I, theta=params.Theta_som, c=params.alpha_som, g=params.g_inh))


def _phi_vip(I: float, params: CircuitParams) -> float:
    """Transfer function for VIP."""
    return float(phi_wong_wang(I, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_inh))


def _solve_interneurons(r_pyr: float, params: CircuitParams) -> tuple[float, float, float]:
    """
    Given r_PYR, solve for (r_SOM, r_PV, r_VIP) at steady state.

    VIP is direct (no recurrence). SOM and PV are solved jointly via fsolve.
    Tries two initial guesses and returns solution with lower residual.

    Returns:
        (r_som, r_pv, r_vip) at steady state
    """
    ggaba = params.g_gaba()

    # VIP is direct (no recurrence)
    I_vip = params.w_ev * r_pyr + params.I_ext_vip()
    r_vip = _phi_vip(I_vip, params)

    # Solve jointly for (r_SOM, r_PV)
    def residuals(x: np.ndarray) -> list[float]:
        r_som, r_pv = x[0], x[1]
        # SOM: includes adaptation term -J_adapt_som*r_som
        I_som = params.w_es * r_pyr - params.w_vs * r_vip - params.J_adapt_som * r_som + params.I_ext_som()
        # PV: divisive PV inhibition, adaptation omitted (not in params)
        I_pv = (
            params.w_ep * r_pyr
            - ggaba * params.w_pp * r_pv
            - ggaba * params.w_sp * r_som
            - params.w_vp * r_vip
            + params.I_ext_pv()
        )
        return [_phi_som(I_som, params) - r_som, _phi_pv(I_pv, params) - r_pv]

    # Try from two initial guesses, take solution with lower residual
    results: list[tuple[float, np.ndarray]] = []
    for x0 in [(0.0, 0.0), (30.0, 30.0)]:
        sol = fsolve(residuals, x0, full_output=True)
        x, info, ier, msg = sol
        residual = float(np.sum(np.abs(info["fvec"])))
        results.append((residual, x))

    _, (r_som, r_pv) = min(results, key=lambda t: t[0])
    # Ensure non-negative
    r_som = max(0.0, float(r_som))
    r_pv = max(0.0, float(r_pv))

    return r_som, r_pv, r_vip


def _compute_F_sweep(r_sweep: np.ndarray, params: CircuitParams) -> np.ndarray:
    """
    Compute F(r_PYR) = Phi_PYR(I_net) - r_PYR for a sweep of r_PYR values.

    This is the PYR nullcline: intersection with identity line gives fixed points.

    Args:
        r_sweep: Array of r_PYR values (Hz)
        params: Circuit parameters

    Returns:
        F values at each point in r_sweep
    """
    ggaba = params.g_gaba()
    F = np.zeros_like(r_sweep, dtype=float)

    for i, r_pyr in enumerate(r_sweep):
        # Solve for interneuron rates at this r_PYR
        r_som, r_pv, r_vip = _solve_interneurons(float(r_pyr), params)

        # PYR adaptation at steady state
        I_adapt_pyr = params.J_adapt_pyr * r_pyr

        # PYR net input with divisive PV inhibition
        denom = 1.0 + ggaba * params.w_pe * r_pv
        I_net = (
            (params.w_ee * r_pyr) / denom
            - ggaba * params.w_se * r_som
            - I_adapt_pyr
            + params.I_ext_pyr()
        )

        F[i] = _phi_pyr(I_net, params) - r_pyr

    return F


def bistable_loss(
    params: CircuitParams,
    cfg: BistableConfig,
    *,
    return_components: bool = False,
) -> Union[float, tuple[float, dict]]:
    """
    Compute the bistability loss for given circuit parameters.

    The loss has four components:
    - L_bistab: Sign pattern enforcement (F must be positive at low/high targets, negative at mid)
    - L_rate: Rate matching at the low fixed point
    - L_margin: Separation between fixed points
    - L_jac: Jacobian regularizer to avoid degenerate solutions

    Args:
        params: Circuit parameters to evaluate
        cfg: BistableConfig with targets and weights
        return_components: If True, return (loss, components_dict)

    Returns:
        Loss as float, or (loss, components) dict if return_components=True
    """
    # Compute nullcline sweep (capped at R_MAX_PHYS to exclude clamp-induced artifacts)
    n_sweep = 500
    r_sweep = np.linspace(0.0, min(R_MAX_PHYS, 80.0), n_sweep)
    F = _compute_F_sweep(r_sweep, params)

    # ========================================================================
    # A. Bistability component L_bistab
    # ========================================================================
    # Interpolate F at specific points
    F_low = float(np.interp(cfg.r_low_target, r_sweep, F))
    F_mid = float(np.interp(cfg.r_mid_probe, r_sweep, F))
    F_high = float(np.interp(cfg.r_high_target, r_sweep, F))

    def relu(x: float) -> float:
        return max(0.0, x)

    # Point-wise penalties
    L_3pt = relu(-F_low) + relu(F_mid) + relu(-F_high)

    # Zone penalties for robustness to probe point placement
    mask1 = r_sweep <= cfg.r_low_target
    mask2 = (r_sweep > cfg.r_low_target) & (r_sweep <= cfg.r_high_target)
    mask3 = r_sweep > cfg.r_high_target

    # Zone 1: F should be >= 0 (nullcline above identity)
    L_zone1 = float(np.mean(np.maximum(-F[mask1], 0.0))) if np.any(mask1) else 0.0
    # Zone 2: F must go negative somewhere (existence of unstable branch)
    L_zone2 = relu(float(np.max(F[mask2]))) if np.any(mask2) else 0.0
    # Zone 3: F should be >= 0 again (nullcline above identity)
    L_zone3 = float(np.mean(np.maximum(-F[mask3], 0.0))) if np.any(mask3) else 0.0

    L_bistab = L_3pt + L_zone1 + L_zone2 + L_zone3

    # ========================================================================
    # B. Fixed point classification with stability analysis
    # ========================================================================
    # Compute gradient for stability classification (dF/dr)
    dF_dr_sweep = np.gradient(F, r_sweep)

    # Find zero-crossings of F (potential fixed points)
    sign_changes = np.where(np.diff(np.sign(F)))[0]

    # Classify each crossing as stable (dF/dr < 0) or unstable (dF/dr > 0)
    stable_fps = []  # (r_fp, dF_dr)
    unstable_fps = []  # (r_fp, dF_dr)
    spurious_fps = []  # Crossings above R_MAX_PHYS

    for idx in sign_changes:
        # Refine crossing location via linear interpolation
        r_cross = float(
            np.interp(0.0, [F[idx], F[idx + 1]], [r_sweep[idx], r_sweep[idx + 1]])
        )

        # Filter out crossings above R_MAX_PHYS (clamp artifacts)
        if r_cross >= R_MAX_PHYS:
            spurious_fps.append(r_cross)
            continue

        # Classify by stability: dF/dr at the crossing index
        dF_dr_at_cross = float(dF_dr_sweep[idx])
        if dF_dr_at_cross < 0:
            stable_fps.append((r_cross, dF_dr_at_cross))
        else:
            unstable_fps.append((r_cross, dF_dr_at_cross))

    # ========================================================================
    # C. Rate matching component L_rate
    # ========================================================================
    # Use the LOWEST stable fixed point
    r_low_fp = None
    if stable_fps:
        # Sort by rate and take the minimum
        stable_fps_sorted = sorted(stable_fps, key=lambda x: x[0])
        r_low_fp = stable_fps_sorted[0][0]
    else:
        # Fallback: use argmin |F| in [0, 15] Hz if no stable crossing found
        mask_low = r_sweep <= 15.0
        if np.any(mask_low):
            idx_min = np.argmin(np.abs(F[mask_low]))
            r_low_fp = float(r_sweep[mask_low][idx_min])
        else:
            r_low_fp = cfg.r_low_target

    # Compute interneuron rates at the low fixed point
    r_som_fp, r_pv_fp, r_vip_fp = _solve_interneurons(r_low_fp, params)

    # Relative squared errors
    L_rate = (
        ((r_low_fp - cfg.r_low_target) / cfg.r_low_target) ** 2
        + ((r_pv_fp - cfg.r_pv_target) / cfg.r_pv_target) ** 2
        + ((r_som_fp - cfg.r_som_target) / cfg.r_som_target) ** 2
        + ((r_vip_fp - cfg.r_vip_target) / cfg.r_vip_target) ** 2
    )

    # ========================================================================
    # D. Separation margin L_margin
    # ========================================================================
    # Bistability margin computed ONLY from stable fixed points
    n_stable = len(stable_fps)

    if n_stable < 2:
        # Monostable: large penalty
        L_margin = cfg.delta_r_min * 2.0
        r_high_fp_candidate = None
    else:
        # Compute margin between lowest and highest stable FPs
        stable_rates = [r for r, _ in stable_fps]
        r_low_stable = min(stable_rates)
        r_high_stable = max(stable_rates)
        L_margin = relu(cfg.delta_r_min - (r_high_stable - r_low_stable))
        r_high_fp_candidate = r_high_stable

    # ========================================================================
    # E. Ceiling loss L_ceiling
    # ========================================================================
    # Penalize if high stable fixed point exceeds physiological ceiling
    L_ceiling = 0.0
    if r_high_fp_candidate is not None and r_high_fp_candidate > cfg.r_high_max:
        L_ceiling = relu(r_high_fp_candidate - cfg.r_high_max) ** 2

    # ========================================================================
    # F. Jacobian regularizer L_jac
    # ========================================================================
    r_ss = np.array([r_low_fp, r_som_fp, r_pv_fp, r_vip_fp], dtype=float)
    J = compute_jacobian(params, r_ss)
    max_abs_J = float(np.max(np.abs(J)))
    L_jac = relu(max_abs_J - 5.0) ** 2

    # ========================================================================
    # Total loss
    # ========================================================================
    L_total = (
        cfg.w_bistab * (L_bistab + L_ceiling)
        + cfg.w_rate * L_rate
        + cfg.w_margin * L_margin
        + cfg.w_jacobian * L_jac
    )

    if return_components:
        # Use the high FP candidate found earlier (if bistable with 2+ stable FPs)
        r_high_fp = r_high_fp_candidate

        components = {
            "L_bistab": float(L_bistab),
            "L_ceiling": float(L_ceiling),
            "L_rate": float(L_rate),
            "L_margin": float(L_margin),
            "L_jac": float(L_jac),
            "L_total": float(L_total),
            "r_low_fp": float(r_low_fp),
            "r_high_fp": float(r_high_fp) if r_high_fp is not None else None,
            "n_stable": int(n_stable),
            "n_unstable": int(len(unstable_fps)),
            "n_spurious": int(len(spurious_fps)),
            "r_pv_fp": float(r_pv_fp),
            "r_som_fp": float(r_som_fp),
            "r_vip_fp": float(r_vip_fp),
        }
        return L_total, components

    return L_total


def save_bistable_summary(
    output_dir: str,
    params: CircuitParams,
    components: dict,
    cfg: BistableConfig,
) -> None:
    """
    Write a human-readable summary of bistable optimization results.

    Args:
        output_dir: Directory to save the summary file
        params: Optimized circuit parameters
        components: Component breakdown from bistable_loss(..., return_components=True)
        cfg: BistableConfig used
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "bistable_summary.txt"

    # Determine bistability classification based on STABLE fixed points only
    n_stable = components.get("n_stable", 0)
    n_unstable = components.get("n_unstable", 0)
    n_spurious = components.get("n_spurious", 0)
    is_bistable = n_stable >= 2
    status = "BISTABLE" if is_bistable else "MONOSTABLE"

    lines = []
    lines.append("=" * 70)
    lines.append("  BISTABLE OPTIMIZATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Loss components
    lines.append("  LOSS COMPONENTS:")
    lines.append(f"    L_bistab   (sign pattern):     {components.get('L_bistab', 0.0):10.4g}")
    lines.append(f"    L_ceiling  (high FP cap):      {components.get('L_ceiling', 0.0):10.4g}")
    lines.append(f"    L_rate     (rate matching):    {components.get('L_rate', 0.0):10.4g}")
    lines.append(f"    L_margin   (FP separation):    {components.get('L_margin', 0.0):10.4g}")
    lines.append(f"    L_jac      (Jacobian reg.):    {components.get('L_jac', 0.0):10.4g}")
    lines.append(f"    ─" * 35)
    lines.append(f"    L_total                        {components.get('L_total', 0.0):10.4g}")
    lines.append("")

    # Fixed points
    lines.append("  FIXED POINTS FOUND:")
    total_crossings = n_stable + n_unstable + n_spurious
    lines.append(f"    Total crossings:               {total_crossings}")
    lines.append(f"      Stable:                      {n_stable}")
    lines.append(f"      Unstable:                    {n_unstable}")
    if n_spurious > 0:
        lines.append(f"      Above R_MAX_PHYS (artifact):   {n_spurious}")
    lines.append(f"    Regime:                        {status}")
    if is_bistable:
        lines.append(f"    Low FP:  r_PYR = {components.get('r_low_fp', 0.0):6.2f} Hz")
        lines.append(f"    High FP: r_PYR = {components.get('r_high_fp', 0.0):6.2f} Hz")
    else:
        lines.append(f"    (Single or no stable FP found)")
    lines.append("")

    # Interneuron rates at low fixed point
    lines.append("  INTERNEURON RATES AT LOW FIXED POINT:")
    pops = ["PYR", "SOM", "PV", "VIP"]
    targets = [cfg.r_low_target, cfg.r_som_target, cfg.r_pv_target, cfg.r_vip_target]
    actuals = [
        components.get("r_low_fp", 0.0),
        components.get("r_som_fp", 0.0),
        components.get("r_pv_fp", 0.0),
        components.get("r_vip_fp", 0.0),
    ]
    lines.append(f"    {'Pop':<6}  {'Actual':>8}  {'Target':>8}  {'Error %':>8}")
    lines.append("    " + "-" * 35)
    for pop, actual, target in zip(pops, actuals, targets):
        if target > 0.01:
            err = 100.0 * (actual - target) / target
        else:
            err = 0.0
        lines.append(f"    {pop:<6}  {actual:8.2f}  {target:8.2f}  {err:+8.1f}")
    lines.append("")

    # Config summary
    lines.append("  CONFIGURATION:")
    lines.append(f"    r_low_target:     {cfg.r_low_target} Hz")
    lines.append(f"    r_high_target:    {cfg.r_high_target} Hz")
    lines.append(f"    r_mid_probe:      {cfg.r_mid_probe} Hz")
    lines.append(f"    delta_r_min:      {cfg.delta_r_min} Hz")
    lines.append(f"    condition:        {cfg.condition}")
    lines.append("")
    lines.append("=" * 70)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
