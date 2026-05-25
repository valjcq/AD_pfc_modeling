"""Numba-compiled Euler integration for the 5-population circuit model.

Population order: [PYR, SOM, PV, VIP, NDNF]  (indices 0..4)

NDNF (added 2026-05): subtractive dendritic inhibitor like SOM.
  Receives PYR (w_ne) and SOM (w_ns).
  Projects to PYR (w_en), PV (w_pn), VIP (w_vn) — all subtractive ×g_gaba.
"""

from __future__ import annotations

import math

import numpy as np

GAMMA_NMDA = 0.641
TAU_NMDA_MS = 100.0

try:
    from numba import njit as _njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def _njit(fn=None, **kwargs):  # type: ignore[misc]
        if fn is not None:
            return fn
        return lambda fn: fn


@_njit(cache=True)
def _phi_scalar(I: float, theta: float, c: float, g: float) -> float:
    u = c * (I - theta)
    z = g * u
    if abs(z) < 1e-8:
        return max(0.0, 1.0 / g + u * 0.5)
    denom = -math.expm1(min(-z, 700.0))
    return max(0.0, u / denom)


@_njit(cache=True)
def _phi_capped_scalar(I: float, r_max: float, theta: float, c: float, g: float) -> float:
    phi = _phi_scalar(I, theta, c, g)
    return r_max * phi / (r_max + phi)


@_njit(cache=True)
def _euler_loop(
    r_out: np.ndarray,       # (n_steps, 5) — [PYR, SOM, PV, VIP, NDNF]
    I_adapt_out: np.ndarray, # (n_steps, 2)
    noise_arr: np.ndarray,
    n_steps: int,
    dt_ms: float,
    noise_scale_pyr: float,
    noise_scale_som: float,
    noise_scale_pv: float,
    noise_scale_vip: float,
    noise_scale_ndnf: float,
    tau_s: float,
    ggaba: float,
    # PYR-side / NMDA
    J_NMDA: float, S_pyr_init: float,
    w_pe: float, w_se: float, w_en: float,
    # SOM input
    w_es: float, w_vs: float,
    # PV input
    w_ep: float, w_pp: float, w_sp: float,
    w_vp: float, w_pn: float,
    # VIP input
    w_ev: float, w_vn: float,
    # NDNF input
    w_ne: float, w_ns: float,
    # Adaptation
    J_adapt_pyr: float, tau_adapt_pyr: float,
    J_adapt_som: float, tau_adapt_som: float,
    # External currents
    I_ext_pyr: float, I_ext_som: float, I_ext_pv: float, I_ext_vip: float,
    I_ext_ndnf: float,
    # Transfer function params
    Theta_pyr: float, alpha_pyr: float, g_exc: float,
    g_inh: float,
    Theta_som: float,  alpha_som: float,
    Theta_pv: float,   alpha_pv: float,
    Theta_vip: float,  alpha_vip: float,
    Theta_ndnf: float, alpha_ndnf: float,
    # Soft ceilings
    r_max_pv: float, r_max_som: float, r_max_vip: float, r_max_ndnf: float,
) -> None:
    S_pyr = S_pyr_init

    for k in range(n_steps - 1):
        r_pyr  = r_out[k, 0]
        r_som  = r_out[k, 1]
        r_pv   = r_out[k, 2]
        r_vip  = r_out[k, 3]
        r_ndnf = r_out[k, 4]
        Iap = I_adapt_out[k, 0]
        Ias = I_adapt_out[k, 1]

        # NMDA gating
        dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
        S_pyr = max(0.0, min(1.0, S_pyr + dS))

        denom = 1.0 + ggaba * w_pe * r_pv

        xi = noise_arr[k]
        I_pyr = (J_NMDA * S_pyr) / denom \
                - ggaba * w_se * r_som \
                - ggaba * w_en * r_ndnf \
                - Iap \
                + I_ext_pyr \
                + noise_scale_pyr * xi
        I_som = w_es * r_pyr \
                - w_vs * r_vip \
                - J_adapt_som * r_som \
                + I_ext_som \
                + noise_scale_som * xi
        I_pv  = w_ep * r_pyr \
                - ggaba * w_pp * r_pv \
                - ggaba * w_sp * r_som \
                - w_vp * r_vip \
                - ggaba * w_pn * r_ndnf \
                + I_ext_pv \
                + noise_scale_pv * xi
        I_vip = w_ev * r_pyr \
                - ggaba * w_vn * r_ndnf \
                + I_ext_vip \
                + noise_scale_vip * xi
        I_ndnf = w_ne * r_pyr \
                 - ggaba * w_ns * r_som \
                 + I_ext_ndnf \
                 + noise_scale_ndnf * xi

        phi_pyr  = _phi_scalar(I_pyr, Theta_pyr, alpha_pyr, g_exc)
        phi_som  = _phi_capped_scalar(I_som,  r_max_som,  Theta_som,  alpha_som,  g_inh)
        phi_pv   = _phi_capped_scalar(I_pv,   r_max_pv,   Theta_pv,   alpha_pv,   g_inh)
        phi_vip  = _phi_capped_scalar(I_vip,  r_max_vip,  Theta_vip,  alpha_vip,  g_inh)
        phi_ndnf = _phi_capped_scalar(I_ndnf, r_max_ndnf, Theta_ndnf, alpha_ndnf, g_inh)

        dr_pyr  = (-r_pyr  + phi_pyr)  / tau_s
        dr_som  = (-r_som  + phi_som)  / tau_s
        dr_pv   = (-r_pv   + phi_pv)   / tau_s
        dr_vip  = (-r_vip  + phi_vip)  / tau_s
        dr_ndnf = (-r_ndnf + phi_ndnf) / tau_s
        r_out[k + 1, 0] = max(0.0, r_pyr  + dt_ms * dr_pyr)
        r_out[k + 1, 1] = max(0.0, r_som  + dt_ms * dr_som)
        r_out[k + 1, 2] = max(0.0, r_pv   + dt_ms * dr_pv)
        r_out[k + 1, 3] = max(0.0, r_vip  + dt_ms * dr_vip)
        r_out[k + 1, 4] = max(0.0, r_ndnf + dt_ms * dr_ndnf)

        I_adapt_out[k + 1, 0] = Iap + dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
        I_adapt_out[k + 1, 1] = Ias + dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som
