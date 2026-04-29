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
from scipy.optimize import fsolve, brentq

from .params import CircuitParams
from .transfer import phi_wong_wang, phi_capped
from .jacobian import compute_jacobian
from .constants import R_MAX_PHYS, R_HIGH_MAX, GAMMA_NMDA, TAU_NMDA_MS, R_MAX_PV, R_MAX_SOM, R_MAX_VIP


@dataclass
class BistableConfig:
    """Configuration for bistability loss function."""
    # Rate target for the low fixed point (used in L_rate only)
    r_low_target: float = 8.0

    # Interneuron targets at the LOW fixed point (Hz)
    r_pv_target: float = 3.0
    r_som_target: float = 5.0
    r_vip_target: float = 2.0

    # Rate targets at the HIGH fixed point — Rooy 2021 active-state values (Hz)
    r_pyr_high_target: float = 60.2
    r_som_high_target: float = 35.2
    r_pv_high_target: float = 35.3
    r_vip_high_target: float = 68.8

    # Margin: minimum gap between low and high fixed points
    delta_r_min: float = 15.0       # Hz

    # Minimum F amplitude required in the high basin.
    # Prevents the optimizer from satisfying L_high_basin by making F barely
    # negative (→ 0⁻) rather than actually positive. Must be > 0.
    f_high_margin: float = 1.0      # Hz

    # Window around r_pyr_high_target where F must be positive (high basin check).
    # Prevents the optimizer from satisfying the high-basin condition with a
    # spurious low-rate bump (e.g., at 17 Hz instead of 60 Hz).
    r_high_basin_lo_frac: float = 0.7   # lower bound = r_pyr_high_target × this
    r_high_basin_hi_frac: float = 1.2   # upper bound = r_pyr_high_target × this

    # Nullcline peak constraint: penalises max(Φ) above this value
    nullcline_peak_max: float = 200.0   # Hz — default 200 = effectively off
    w_peak: float = 0.0                 # default off (backward compatible)

    # Loss weights — bistability is the priority; rate matching is secondary
    w_bistab: float = 5.0           # adaptive sign-pattern check; must dominate when monostable
    w_rate: float = 1.0             # Low FP rate matching
    w_rate_high: float = 1.5        # High FP rate matching
    w_margin: float = 2.0           # Separation penalty; also large when monostable (×delta_r_min×2)
    w_jacobian: float = 0.1         # Regularizer on max |J| entry

    # Condition (for informational purposes)
    condition: str = "WT"


def _phi_pyr(I: float, params: CircuitParams) -> float:
    """Transfer function for PYR."""
    return float(phi_wong_wang(I, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc))


def _phi_pv(I: float, params: CircuitParams) -> float:
    """Transfer function for PV — with hyperbolic soft ceiling."""
    return float(phi_capped(I, R_MAX_PV, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_inh))


def _phi_som(I: float, params: CircuitParams) -> float:
    """Transfer function for SOM — with hyperbolic soft ceiling."""
    return float(phi_capped(I, R_MAX_SOM, theta=params.Theta_som, c=params.alpha_som, g=params.g_inh))


def _phi_vip(I: float, params: CircuitParams) -> float:
    """Transfer function for VIP — with hyperbolic soft ceiling."""
    return float(phi_capped(I, R_MAX_VIP, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_inh))


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

        # PYR net input with divisive PV inhibition and NMDA gating
        denom = 1.0 + ggaba * params.w_pe * r_pv
        # NMDA gating: use steady-state formula S* for fixed-point analysis
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
        I_net = (
            (params.J_NMDA * S_star) / denom
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

    The loss has six components:
    - L_bistab: Sign pattern enforcement — simplified 2-condition check
    - L_rate: Rate matching at the low fixed point
    - L_rate_high: Rate matching at the high fixed point (Rooy 2021 active-state targets)
    - L_margin: Separation between fixed points
    - L_ceiling: High FP ceiling (decoupled from w_bistab)
    - L_jac: Jacobian regularizer to avoid degenerate solutions

    Fixed points are refined using Brentq root-finding for numerical accuracy.

    Args:
        params: Circuit parameters to evaluate
        cfg: BistableConfig with targets and weights
        return_components: If True, return (loss, components_dict)

    Returns:
        Loss as float, or (loss, components) dict if return_components=True
    """
    # Compute nullcline sweep (capped at R_MAX_PHYS to exclude clamp-induced artifacts)
    # Use higher resolution (1000 points) for better FP detection
    n_sweep = 1000
    r_sweep = np.linspace(0.0, min(R_MAX_PHYS, 80.0), n_sweep)
    F = _compute_F_sweep(r_sweep, params)

    # ========================================================================
    # A. Bistability component L_bistab
    # ========================================================================
    # Two-part check that requires no fixed probe locations.
    #
    # Part 1 — Adaptive low basin:
    #   Detect the actual low FP (first downward crossing of F, i.e. F goes
    #   from + to -). Penalise if F is not positive throughout [0, r_low_actual].
    #   Penalty is scaled by the normalised displacement of the actual FP from
    #   its target: if the low FP drifts to 70 Hz instead of 1.75 Hz, the whole
    #   [0, 70] region must be positive AND the penalty is ~40× larger, creating
    #   a strong coupled gradient toward the target position.
    #
    # Part 2 — Full sign pattern (+, -, +, -):
    #   After the low FP, three conditions enforce the bistable shape:
    #     a) F < 0 somewhere  → valley exists  (creates the unstable FP)
    #     b) F > 0 somewhere  → high basin exists  (creates the high stable FP)
    #     c) F < 0 at the far end of the sweep  → high FP is stable, not a bump
    #   The unstable FP position is not constrained; only the pattern matters.

    def relu(x: float) -> float:
        return max(0.0, x)

    # Detect the first downward zero-crossing (F goes from + to -)
    down_cross_idx = np.where(np.diff(np.sign(F)) < 0)[0]
    r_low_actual = float(r_sweep[down_cross_idx[0]]) if len(down_cross_idx) > 0 else cfg.r_low_target

    # Part 1: F > 0 in [0, r_low_actual], scaled by FP displacement from target
    mask_low_basin = r_sweep <= r_low_actual
    F_max_low_basin = float(np.max(F[mask_low_basin])) if mask_low_basin.any() else -1.0
    fp_scale = 1.0 + abs(r_low_actual - cfg.r_low_target) / max(cfg.r_low_target, 1.0)
    L_low_basin = relu(-F_max_low_basin) * fp_scale

    # Part 2: sign pattern after the low FP
    mask_after_low = r_sweep > r_low_actual
    if mask_after_low.any():
        F_after = F[mask_after_low]
        L_valley     = relu(float(np.min(F_after)))   # 2a: F must go negative (valley)
        # 2b: F must be positive within the target high-state window.
        #     Using a windowed max (not global) prevents satisfying this with
        #     a spurious low-rate crossing (e.g., nullcline bump at 17 Hz).
        r_hb_lo = cfg.r_pyr_high_target * cfg.r_high_basin_lo_frac
        r_hb_hi = cfg.r_pyr_high_target * cfg.r_high_basin_hi_frac
        mask_hb = (r_sweep > r_hb_lo) & (r_sweep <= r_hb_hi)
        F_max_hb = float(np.max(F[mask_hb])) if mask_hb.any() else -np.inf
        L_high_basin = relu(cfg.f_high_margin - F_max_hb)
        tail_mask    = r_sweep >= r_sweep[-1] * 0.85
        L_tail       = relu(float(np.max(F[tail_mask])))  # 2c: F < 0 at far end (stable high FP)
    else:
        L_valley = L_high_basin = L_tail = 1.0

    L_bistab = L_low_basin + L_valley + L_high_basin + L_tail

    # ========================================================================
    # B. Fixed point classification with stability analysis
    # ========================================================================
    # Compute gradient for stability classification (dF/dr)
    dF_dr_sweep = np.gradient(F, r_sweep)

    # Find zero-crossings of F (potential fixed points) with Brentq refinement
    sign_changes = np.where(np.diff(np.sign(F)))[0]

    # Classify each crossing as stable (dF/dr < 0) or unstable (dF/dr > 0)
    stable_fps = []  # (r_fp, dF_dr)
    unstable_fps = []  # (r_fp, dF_dr)
    spurious_fps = []  # Crossings above R_MAX_PHYS

    for idx in sign_changes:
        r_min = r_sweep[idx]
        r_max = r_sweep[idx + 1]

        # Refine crossing via Brentq for better numerical accuracy
        try:
            def f_to_root(r_pyr: float) -> float:
                r_som, r_pv, r_vip = _solve_interneurons(r_pyr, params)
                ggaba = params.g_gaba()
                I_adapt_pyr = params.J_adapt_pyr * r_pyr
                denom = 1.0 + ggaba * params.w_pe * r_pv
                S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
                I_net = (
                    (params.J_NMDA * S_star) / denom
                    - ggaba * params.w_se * r_som
                    - I_adapt_pyr
                    + params.I_ext_pyr()
                )
                return _phi_pyr(I_net, params) - r_pyr

            r_cross = float(brentq(f_to_root, r_min, r_max))
        except ValueError:
            # Brentq failed, fall back to interpolation
            r_cross = float(np.interp(0.0, [F[idx], F[idx + 1]], [r_min, r_max]))

        # Filter out crossings above R_MAX_PHYS (clamp artifacts)
        if r_cross >= R_MAX_PHYS:
            spurious_fps.append(r_cross)
            continue

        # Classify by stability: dF/dr at the refined crossing point
        # Use interpolation for stability, not just the index
        dF_dr_at_cross = float(np.interp(r_cross, r_sweep, dF_dr_sweep))
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
    # E. Jacobian regularizer L_jac
    # ========================================================================
    r_ss = np.array([r_low_fp, r_som_fp, r_pv_fp, r_vip_fp], dtype=float)
    J = compute_jacobian(params, r_ss)
    max_abs_J = float(np.max(np.abs(J)))
    L_jac = relu(max_abs_J - 5.0) ** 2

    # ========================================================================
    # F. Nullcline peak penalty L_peak
    # ========================================================================
    # Φ(r) = F(r) + r  →  peak of the nullcline above the identity line
    phi_sweep = F + r_sweep
    nullcline_peak = float(np.max(phi_sweep))
    L_peak = relu(nullcline_peak - cfg.nullcline_peak_max) ** 2

    # ========================================================================
    # G. High fixed-point rate matching L_rate_high
    # ========================================================================
    # Only active when the network is bistable (second stable FP exists).
    # Symmetric MSPE between actual high-FP rates and Rooy 2021 targets.
    # When monostable, L_rate_high = 0 — the bistability term already applies a
    # strong gradient; adding a phantom high-FP penalty would be misleading.
    L_rate_high = 0.0
    r_som_high_fp: Optional[float] = None
    r_pv_high_fp: Optional[float] = None
    r_vip_high_fp: Optional[float] = None
    if r_high_fp_candidate is not None:
        r_som_high_fp, r_pv_high_fp, r_vip_high_fp = _solve_interneurons(r_high_fp_candidate, params)
        L_rate_high = (
            ((r_high_fp_candidate - cfg.r_pyr_high_target) / cfg.r_pyr_high_target) ** 2
            + ((r_som_high_fp - cfg.r_som_high_target) / cfg.r_som_high_target) ** 2
            + ((r_pv_high_fp - cfg.r_pv_high_target) / cfg.r_pv_high_target) ** 2
            + ((r_vip_high_fp - cfg.r_vip_high_target) / cfg.r_vip_high_target) ** 2
        )

    # ========================================================================
    # Total loss
    # ========================================================================
    L_total = (
        cfg.w_bistab * L_bistab
        + cfg.w_rate * L_rate
        + cfg.w_rate_high * L_rate_high
        + cfg.w_margin * L_margin
        + cfg.w_jacobian * L_jac
        + cfg.w_peak * L_peak
    )

    if return_components:
        r_high_fp = r_high_fp_candidate
        components = {
            "L_bistab": float(L_bistab),
            "L_rate": float(L_rate),
            "L_rate_high": float(L_rate_high),
            "L_margin": float(L_margin),
            "L_jac": float(L_jac),
            "L_peak": float(L_peak),
            "nullcline_peak_hz": float(nullcline_peak),
            "L_total": float(L_total),
            "r_low_fp": float(r_low_fp),
            "r_high_fp": float(r_high_fp) if r_high_fp is not None else None,
            "n_stable": int(n_stable),
            "n_unstable": int(len(unstable_fps)),
            "n_spurious": int(len(spurious_fps)),
            # Low FP interneuron rates
            "r_pv_fp": float(r_pv_fp),
            "r_som_fp": float(r_som_fp),
            "r_vip_fp": float(r_vip_fp),
            # High FP interneuron rates (None if monostable)
            "r_pv_high_fp": float(r_pv_high_fp) if r_pv_high_fp is not None else None,
            "r_som_high_fp": float(r_som_high_fp) if r_som_high_fp is not None else None,
            "r_vip_high_fp": float(r_vip_high_fp) if r_vip_high_fp is not None else None,
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
    lines.append(f"    L_rate     (low FP rates):     {components.get('L_rate', 0.0):10.4g}")
    lines.append(f"    L_rate_high(high FP rates):    {components.get('L_rate_high', 0.0):10.4g}")
    lines.append(f"    L_margin   (FP separation):    {components.get('L_margin', 0.0):10.4g}")
    lines.append(f"    L_jac      (Jacobian reg.):    {components.get('L_jac', 0.0):10.4g}")
    lines.append(f"    L_peak     (nullcline peak):   {components.get('L_peak', 0.0):10.4g}  [peak={components.get('nullcline_peak_hz', float('nan')):.1f} Hz]")
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
    low_pops = ["PYR", "SOM", "PV", "VIP"]
    low_targets = [cfg.r_low_target, cfg.r_som_target, cfg.r_pv_target, cfg.r_vip_target]
    low_actuals = [
        components.get("r_low_fp", 0.0),
        components.get("r_som_fp", 0.0),
        components.get("r_pv_fp", 0.0),
        components.get("r_vip_fp", 0.0),
    ]
    lines.append(f"    {'Pop':<6}  {'Actual':>8}  {'Target':>8}  {'Error %':>8}")
    lines.append("    " + "-" * 35)
    for pop, actual, target in zip(low_pops, low_actuals, low_targets):
        err = 100.0 * (actual - target) / target if target > 0.01 else 0.0
        lines.append(f"    {pop:<6}  {actual:8.2f}  {target:8.2f}  {err:+8.1f}")
    lines.append("")

    # Interneuron rates at high fixed point
    lines.append("  INTERNEURON RATES AT HIGH FIXED POINT:")
    if is_bistable:
        high_pops = ["PYR", "SOM", "PV", "VIP"]
        high_targets = [cfg.r_pyr_high_target, cfg.r_som_high_target, cfg.r_pv_high_target, cfg.r_vip_high_target]
        high_actuals = [
            components.get("r_high_fp", 0.0),
            components.get("r_som_high_fp") or 0.0,
            components.get("r_pv_high_fp") or 0.0,
            components.get("r_vip_high_fp") or 0.0,
        ]
        lines.append(f"    {'Pop':<6}  {'Actual':>8}  {'Target':>8}  {'Error %':>8}")
        lines.append("    " + "-" * 35)
        for pop, actual, target in zip(high_pops, high_actuals, high_targets):
            err = 100.0 * (actual - target) / target if target > 0.01 else 0.0
            lines.append(f"    {pop:<6}  {actual:8.2f}  {target:8.2f}  {err:+8.1f}")
    else:
        lines.append("    N/A — network is MONOSTABLE (no high fixed point found)")
        lines.append(f"    Targets were: PYR={cfg.r_pyr_high_target} SOM={cfg.r_som_high_target} "
                     f"PV={cfg.r_pv_high_target} VIP={cfg.r_vip_high_target} Hz")
    lines.append("")

    # Config summary
    lines.append("  CONFIGURATION:")
    lines.append(f"    r_low_target:     {cfg.r_low_target} Hz")
    lines.append(f"    r_pyr_high_target:{cfg.r_pyr_high_target} Hz")
    lines.append(f"    delta_r_min:      {cfg.delta_r_min} Hz")
    lines.append(f"    condition:        {cfg.condition}")
    lines.append("")
    lines.append("=" * 70)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
