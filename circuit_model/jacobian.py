"""
Jacobian and effective-connectivity analysis for the 5-population circuit.

Population order: [PYR=0, SOM=1, PV=2, VIP=3, NDNF=4]
"""

from __future__ import annotations

import numpy as np

from .params import CircuitParams
from .constants import GAMMA_NMDA, TAU_NMDA_MS


def _phi_derivative(I: float, *, theta: float, c: float, g: float) -> float:
    """dΦ/dI of the Wong-Wang transfer function at input I."""
    u = c * (I - theta)
    if u < -700.0 / max(g, 1e-9):
        return 0.0
    z = g * u
    eps = 1e-8
    if abs(z) < eps:
        return c / 2.0
    e = np.exp(-z)
    denom = 1.0 - e
    return float(c * (1.0 - e * (1.0 + z)) / (denom ** 2))


def _total_inputs(
    params: CircuitParams,
    r: np.ndarray,
    i_adapt: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute total synaptic input currents at steady-state rates r.

    Parameters
    ----------
    r : [r_pyr, r_som, r_pv, r_vip, r_ndnf]
    i_adapt : [I_adapt_pyr, I_adapt_som] at steady state; if None, uses fixed-point.

    Returns
    -------
    (I_pyr, I_som, I_pv, I_vip, I_ndnf)
    """
    r_pyr, r_som, r_pv, r_vip, r_ndnf = r
    ggaba = params.g_gaba()

    if i_adapt is None:
        I_ap = params.J_adapt_pyr * r_pyr
    else:
        I_ap = i_adapt[0]

    denom = 1.0 + ggaba * params.w_pe * r_pv
    S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
    I_pyr = ((params.J_NMDA * S_star) / denom
             - ggaba * params.w_se * r_som
             - ggaba * params.w_ne * r_ndnf
             - I_ap + params.I_ext_pyr())
    I_som = (params.w_es * r_pyr - params.w_vs * r_vip + params.I_ext_som())
    I_pv  = (params.w_ep * r_pyr
             - ggaba * params.w_pp * r_pv
             - ggaba * params.w_sp * r_som
             - params.w_vp * r_vip
             - ggaba * params.w_np * r_ndnf
             + params.I_ext_pv())
    I_vip = (params.w_ev * r_pyr
             - ggaba * params.w_nv * r_ndnf
             + params.I_ext_vip())
    I_ndnf = (- ggaba * params.w_sn * r_som
              + params.I_ext_ndnf())

    return I_pyr, I_som, I_pv, I_vip, I_ndnf


def compute_jacobian(
    params: CircuitParams,
    r_ss: np.ndarray,
) -> np.ndarray:
    """5×5 effective-gain Jacobian at steady-state rates.

    Row/column order: [PYR, SOM, PV, VIP, NDNF].
    """
    r_pyr, r_som, r_pv, r_vip, r_ndnf = r_ss
    ggaba = params.g_gaba()

    I_pyr, I_som, I_pv, I_vip, I_ndnf = _total_inputs(params, r_ss)
    dphi_pyr  = _phi_derivative(I_pyr,  theta=params.Theta_pyr,  c=params.alpha_pyr,  g=params.g_exc)
    dphi_som  = _phi_derivative(I_som,  theta=params.Theta_som,  c=params.alpha_som,  g=params.g_inh)
    dphi_pv   = _phi_derivative(I_pv,   theta=params.Theta_pv,   c=params.alpha_pv,   g=params.g_inh)
    dphi_vip  = _phi_derivative(I_vip,  theta=params.Theta_vip,  c=params.alpha_vip,  g=params.g_inh)
    dphi_ndnf = _phi_derivative(I_ndnf, theta=params.Theta_ndnf, c=params.alpha_ndnf, g=params.g_inh)

    denom = 1.0 + ggaba * params.w_pe * r_pv
    S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
    dSdr = GAMMA_NMDA * TAU_NMDA_MS / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS) ** 2

    # ∂I_pyr / ∂r_j  (j in [pyr, som, pv, vip, ndnf])
    dIpyr = (
        params.J_NMDA * dSdr / denom,
        -ggaba * params.w_se,
        -params.J_NMDA * S_star * ggaba * params.w_pe / (denom ** 2),
        0.0,
        -ggaba * params.w_ne,
    )
    # ∂I_som / ∂r_j
    dIsom = (params.w_es, 0.0, 0.0, -params.w_vs, 0.0)
    # ∂I_pv / ∂r_j
    dIpv  = (params.w_ep, -ggaba * params.w_sp, -ggaba * params.w_pp,
             -params.w_vp, -ggaba * params.w_np)
    # ∂I_vip / ∂r_j
    dIvip = (params.w_ev, 0.0, 0.0, 0.0, -ggaba * params.w_nv)
    # ∂I_ndnf / ∂r_j  (no PYR -> NDNF; only SOM -> NDNF)
    dIndnf = (0.0, -ggaba * params.w_sn, 0.0, 0.0, 0.0)

    J = np.array([
        [dphi_pyr  * x for x in dIpyr],
        [dphi_som  * x for x in dIsom],
        [dphi_pv   * x for x in dIpv],
        [dphi_vip  * x for x in dIvip],
        [dphi_ndnf * x for x in dIndnf],
    ])
    return J


# (row, col, attr, description, expected_sign)
# Row = target (output channel), Col = source. Names follow the
# `w_XY = X (source) -> Y (target)` convention used throughout the codebase.
_CONNECTIONS = [
    (0, 0, "J_NMDA", "PYR  → PYR  (NMDA recurrent excitation)", "+"),
    (1, 0, "w_es",   "PYR  → SOM  (recruits dendritic inh.)",   "+"),
    (2, 0, "w_ep",   "PYR  → PV   (fast feedback inh.)",         "+"),
    (3, 0, "w_ev",   "PYR  → VIP  (recruits disinhibition)",     "+"),
    (0, 1, "w_se",   "SOM  → PYR  (dendritic inhibition)",       "-"),
    (0, 2, "w_pe",   "PV   → PYR  (perisomatic inhibition)",     "-"),
    (2, 2, "w_pp",   "PV   → PV   (self-inhibition)",            "-"),
    (2, 1, "w_sp",   "SOM  → PV   (cross-inhibition)",           "-"),
    (1, 3, "w_vs",   "VIP  → SOM  (disinhibition pathway)",      "-"),
    (2, 3, "w_vp",   "VIP  → PV   (weak disinhibition)",         "-"),
    (4, 1, "w_sn",   "SOM  → NDNF (subtractive inhibition)",     "-"),
    (0, 4, "w_ne",   "NDNF → PYR  (dendritic inhibition)",       "-"),
    (2, 4, "w_np",   "NDNF → PV   (subtractive inhibition)",     "-"),
    (3, 4, "w_nv",   "NDNF → VIP  (subtractive inhibition)",     "-"),
]

_POPS = ["PYR ", "SOM ", "PV  ", "VIP ", "NDNF"]


def print_sanity_check(
    params: CircuitParams,
    r_ss: np.ndarray,
    *,
    negligible_threshold: float = 0.005,
) -> None:
    """Human-readable sanity check of the 5-population circuit Jacobian."""
    J = compute_jacobian(params, r_ss)

    print("\n" + "=" * 70)
    print("  CIRCUIT JACOBIAN — effective gains at operating point")
    print(f"  r = [PYR={r_ss[0]:.2f}, SOM={r_ss[1]:.2f}, PV={r_ss[2]:.2f}, "
          f"VIP={r_ss[3]:.2f}, NDNF={r_ss[4]:.2f}]")
    print("=" * 70)

    header = "         " + "".join(f"  {p:>6}" for p in _POPS)
    print(header)
    print("         " + "-" * (len(header) - 9))
    for i, row_name in enumerate(_POPS):
        row = "  ".join(f"{J[i, j]:+.4f}" for j in range(5))
        print(f"  {row_name}  |  {row}")

    print()
    print("  Connection details  (raw weight → effective gain):")
    print("  " + "-" * 66)

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

        direction_ok = (gain > 0 and sign == "+") or (gain < 0 and sign == "-")
        dir_flag = "" if direction_ok else "  [WRONG SIGN ⚠]"

        print(f"  {desc:<44}  w={w_raw:8.3f}  J={gain:+.4f}  [{label}]{dir_flag}")

    print("  " + "-" * 66)
    if warnings:
        print(f"\n  ⚠  {len(warnings)} negligible connection(s) — circuit may be degenerate:")
        for w in warnings:
            print(f"      • {w}")
    else:
        print("\n  ✓  All expected connections have non-negligible effective gain.")
    print("=" * 70 + "\n")
