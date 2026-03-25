"""
Jacobian and effective-connectivity analysis for the 4-population circuit.

The raw synaptic weights are NOT the same as functional connection strengths.
Because each population has its own transfer function (threshold, gain, curvature),
the same weight w_XY can have very different effects depending on where population X
is operating relative to its threshold.

The Jacobian element J[i, j] = ∂r_i/∂r_j evaluated at a steady-state operating
point gives the *effective gain*: "if population j fires 1 Hz more, how much does
population i change?"  This is a model-independent, scale-independent measure of
functional connection strength.

Population order throughout: [PYR=0, SOM=1, PV=2, VIP=3]

Usage
-----
    from circuit_model.jacobian import compute_jacobian, print_sanity_check

    r_ss = np.array([4.14, 3.42, 2.08, 1.93])   # fitted steady-state rates
    J = compute_jacobian(params, r_ss)
    print_sanity_check(params, r_ss)
"""

from __future__ import annotations

import numpy as np

from .params import CircuitParams


# ---------------------------------------------------------------------------
# Transfer function derivative
# ---------------------------------------------------------------------------

def _phi_derivative(I: float, *, theta: float, c: float, g: float) -> float:
    """Derivative dΦ/dI of the Wong-Wang transfer function at input I.

    Φ(I) = max(0,  u / (1 − exp(−g·u))  )   where  u = c·(I − θ)

    dΦ/dI = c · (1 − e·(1 + g·u)) / (1 − e)²   where  e = exp(−g·u)
    Near u ≈ 0 uses the Taylor limit  dΦ/dI ≈ c/2.
    Returns 0 when Φ would be ≤ 0 (below threshold, rectified).
    """
    u = c * (I - theta)
    # If the unrectified output would be ≤ 0, derivative is 0
    # (practically never happens at a real operating point, but just in case)
    if u < -700.0 / max(g, 1e-9):
        return 0.0

    z = g * u
    eps = 1e-8
    if abs(z) < eps:
        return c / 2.0

    e = np.exp(-z)
    denom = 1.0 - e
    return float(c * (1.0 - e * (1.0 + z)) / (denom ** 2))


# ---------------------------------------------------------------------------
# Total inputs at a given rate vector
# ---------------------------------------------------------------------------

def _total_inputs(
    params: CircuitParams,
    r: np.ndarray,
    i_adapt: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Compute total synaptic input currents at steady-state rates r.

    Parameters
    ----------
    r : [r_pyr, r_som, r_pv, r_vip]
    i_adapt : [I_adapt_pyr, I_adapt_som] at steady state.
        If None, uses the fixed-point values I_adapt_X = J_adapt_X * r_X.

    Returns
    -------
    (I_pyr, I_som, I_pv, I_vip)
    """
    r_pyr, r_som, r_pv, r_vip = r
    ggaba = params.g_gaba()

    if i_adapt is None:
        I_ap = params.J_adapt_pyr * r_pyr
    else:
        I_ap = i_adapt[0]

    denom = 1.0 + ggaba * params.w_pe * r_pv
    I_pyr = (params.w_ee * r_pyr) / denom - ggaba * params.w_se * r_som - I_ap + params.I_ext_pyr()
    I_som = params.w_es * r_pyr - params.w_vs * r_vip + params.I_ext_som()
    I_pv  = (params.w_ep * r_pyr
             - ggaba * params.w_pp * r_pv
             - ggaba * params.w_sp * r_som
             - params.w_vp * r_vip
             + params.I_ext_pv())
    I_vip = params.w_ev * r_pyr + params.I_ext_vip()

    return I_pyr, I_som, I_pv, I_vip


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def compute_jacobian(
    params: CircuitParams,
    r_ss: np.ndarray,
) -> np.ndarray:
    """Compute the 4×4 effective-gain Jacobian at steady-state rates r_ss.

    J[i, j] = ∂r_i/∂r_j   (treating adaptation as fixed at its s.s. value)

    Row/column order: [PYR, SOM, PV, VIP]

    Positive J[i,j] means j excites i; negative means j inhibits i.

    Parameters
    ----------
    params : CircuitParams
        Fitted circuit parameters.
    r_ss : array of shape (4,)
        Steady-state firing rates [r_pyr, r_som, r_pv, r_vip] (Hz)
        units cancel in the ratio, so the Jacobian is dimensionless).

    Returns
    -------
    J : (4, 4) float array
    """
    r_pyr, r_som, r_pv, r_vip = r_ss
    ggaba = params.g_gaba()

    # Transfer function derivatives at the operating point
    I_pyr, I_som, I_pv, I_vip = _total_inputs(params, r_ss)
    dphi_pyr = _phi_derivative(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g)
    dphi_som = _phi_derivative(I_som, theta=params.Theta_som, c=params.alpha_som, g=params.g)
    dphi_pv  = _phi_derivative(I_pv,  theta=params.Theta_pv,  c=params.alpha_pv,  g=params.g)
    dphi_vip = _phi_derivative(I_vip, theta=params.Theta_vip, c=params.alpha_vip, g=params.g)

    # Partial derivatives ∂I_i/∂r_j  (explicit, adaptation treated as fixed)
    denom = 1.0 + ggaba * params.w_pe * r_pv

    # ∂I_pyr / ∂r_j
    dIpyr_drpyr = params.w_ee / denom
    dIpyr_drsom = -ggaba * params.w_se
    dIpyr_drpv  = -params.w_ee * r_pyr * ggaba * params.w_pe / (denom ** 2)
    dIpyr_drvip = 0.0

    # ∂I_som / ∂r_j
    dIsom_drpyr = params.w_es
    dIsom_drsom = 0.0
    dIsom_drpv  = 0.0
    dIsom_drvip = -params.w_vs

    # ∂I_pv / ∂r_j
    dIpv_drpyr = params.w_ep
    dIpv_drsom = -ggaba * params.w_sp
    dIpv_drpv  = -ggaba * params.w_pp
    dIpv_drvip = -params.w_vp

    # ∂I_vip / ∂r_j
    dIvip_drpyr = params.w_ev
    dIvip_drsom = 0.0
    dIvip_drpv  = 0.0
    dIvip_drvip = 0.0

    # J[i, j] = dphi_i * dI_i/dr_j
    J = np.array([
        [dphi_pyr * dIpyr_drpyr, dphi_pyr * dIpyr_drsom, dphi_pyr * dIpyr_drpv,  dphi_pyr * dIpyr_drvip],
        [dphi_som * dIsom_drpyr, dphi_som * dIsom_drsom, dphi_som * dIsom_drpv,  dphi_som * dIsom_drvip],
        [dphi_pv  * dIpv_drpyr,  dphi_pv  * dIpv_drsom,  dphi_pv  * dIpv_drpv,   dphi_pv  * dIpv_drvip],
        [dphi_vip * dIvip_drpyr, dphi_vip * dIvip_drsom, dphi_vip * dIvip_drpv,  dphi_vip * dIvip_drvip],
    ])
    return J


# ---------------------------------------------------------------------------
# Sanity check report
# ---------------------------------------------------------------------------

# Expected-present connections and their biological role.
# Each entry: (row_pop, col_pop, weight_attr, description, expected_sign)
_CONNECTIONS = [
    (0, 0, "w_ee", "PYR → PYR  (recurrent excitation)",    "+"),
    (1, 0, "w_es", "PYR → SOM  (recruits dendritic inh.)", "+"),
    (2, 0, "w_ep", "PYR → PV   (fast feedback inh.)",      "+"),
    (3, 0, "w_ev", "PYR → VIP  (recruits disinhibition)",  "+"),
    (0, 1, "w_se", "SOM → PYR  (dendritic inhibition)",    "-"),
    (0, 2, "w_pe", "PV  → PYR  (perisomatic inhibition)",  "-"),
    (2, 2, "w_pp", "PV  → PV   (self-inhibition)",         "-"),
    (2, 1, "w_sp", "SOM → PV   (cross-inhibition)",        "-"),
    (1, 3, "w_vs", "VIP → SOM  (disinhibition pathway)",   "-"),
    (2, 3, "w_vp", "VIP → PV   (weak disinhibition)",      "-"),
]

_POPS = ["PYR", "SOM", "PV ", "VIP"]


def print_sanity_check(
    params: CircuitParams,
    r_ss: np.ndarray,
    *,
    negligible_threshold: float = 0.005,
) -> None:
    """Print a human-readable sanity check of the circuit at the operating point.

    Shows:
    - The 4×4 Jacobian (effective gains)
    - Per-connection flagging: STRONG / MODERATE / WEAK / NEGLIGIBLE
    - Warning for any connection that is negligible (|J| ≤ negligible_threshold)

    Parameters
    ----------
    params : CircuitParams
    r_ss : [r_pyr, r_som, r_pv, r_vip]
    negligible_threshold : flag connection as negligible if |J[i,j]| ≤ this value.
    """
    J = compute_jacobian(params, r_ss)

    print("\n" + "=" * 62)
    print("  CIRCUIT JACOBIAN — effective gains at operating point")
    print(f"  r = [PYR={r_ss[0]:.2f}, SOM={r_ss[1]:.2f}, PV={r_ss[2]:.2f}, VIP={r_ss[3]:.2f}]")
    print("=" * 62)

    # Print full matrix
    header = "         " + "".join(f"  {p:>6}" for p in _POPS)
    print(header)
    print("         " + "-" * (len(header) - 9))
    for i, row_name in enumerate(_POPS):
        row = "  ".join(f"{J[i, j]:+.4f}" for j in range(4))
        print(f"  {row_name}  |  {row}")

    print()
    print("  Connection details  (raw weight → effective gain):")
    print("  " + "-" * 58)

    warnings = []
    for (i, j, attr, desc, sign) in _CONNECTIONS:
        w_raw = getattr(params, attr)
        gain = J[i, j]
        abs_gain = abs(gain)

        if abs_gain > 0.1:
            label = "STRONG  "
        elif abs_gain > 0.01:
            label = "moderate"
        elif abs_gain > negligible_threshold:
            label = "weak    "
        else:
            label = "NEGLIGIBLE ⚠"
            warnings.append(desc)

        # Direction check
        direction_ok = (gain > 0 and sign == "+") or (gain < 0 and sign == "-")
        dir_flag = "" if direction_ok else "  [WRONG SIGN ⚠]"

        print(f"  {desc:<42}  w={w_raw:8.3f}  J={gain:+.4f}  [{label}]{dir_flag}")

    print("  " + "-" * 58)
    if warnings:
        print(f"\n  ⚠  {len(warnings)} negligible connection(s) — circuit may be degenerate:")
        for w in warnings:
            print(f"      • {w}")
    else:
        print("\n  ✓  All expected connections have non-negligible effective gain.")
    print("=" * 62 + "\n")
