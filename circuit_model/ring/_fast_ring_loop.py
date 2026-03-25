"""Numba-compiled Euler integration for the ring attractor network."""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit as _njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def _njit(fn=None, **kwargs):  # type: ignore[misc]
        """No-op decorator used when numba is not installed."""
        if fn is not None:
            return fn
        return lambda fn: fn


@_njit(cache=True)
def _phi_scalar(I: float, theta: float, c: float, g: float) -> float:
    """Wong-Wang transfer function on a scalar value."""
    u = c * (I - theta)
    z = g * u
    if abs(z) < 1e-8:
        return max(0.0, 1.0 / g + u * 0.5)
    denom = -math.expm1(min(-z, 700.0))
    return max(0.0, u / denom)


@_njit(cache=True)
def _ring_euler_loop(
    r_out: np.ndarray,
    I_adapt_out: np.ndarray,
    noise_arr: np.ndarray,
    I_stim_arr: np.ndarray,
    I_ext_pyr_arr: np.ndarray,
    I_ext_som_arr: np.ndarray,
    I_ext_pv_arr: np.ndarray,
    I_ext_vip_arr: np.ndarray,
    W_pyr_pyr: np.ndarray,
    W_pv_pyr: np.ndarray,
    n_steps: int,
    n_nodes: int,
    dt_ms: float,
    sigma_s: float,
    tau_s: float,
    ggaba: float,
    w_ee: float,
    w_pe: float,
    w_se: float,
    w_es: float,
    w_vs: float,
    w_ep: float,
    w_pp: float,
    w_sp: float,
    w_vp: float,
    w_ev: float,
    J_adapt_pyr: float,
    tau_adapt_pyr: float,
    J_adapt_som: float,
    tau_adapt_som: float,
    Theta_pyr: float,
    alpha_pyr: float,
    g: float,
    Theta_som: float,
    alpha_som: float,
    Theta_pv: float,
    alpha_pv: float,
    Theta_vip: float,
    alpha_vip: float,
) -> None:
    """Core Euler integration loop for ring simulations."""
    I_pyr_inter = np.zeros(n_nodes)
    I_pv_inter = np.zeros(n_nodes)
    r_pyr_k = np.zeros(n_nodes)
    r_pv_k = np.zeros(n_nodes)

    for k in range(n_steps - 1):
        # Build contiguous vectors for fast BLAS-backed dot products.
        for j in range(n_nodes):
            r_pyr_k[j] = r_out[k, j, 0]
            r_pv_k[j] = r_out[k, j, 2]

        # BLAS-backed matrix-vector products are significantly faster than
        # explicit nested loops for n_nodes=O(100) and long trajectories.
        I_pyr_inter[:] = np.dot(W_pyr_pyr, r_pyr_k)
        I_pv_inter[:] = np.dot(W_pv_pyr, r_pv_k)

        for j in range(n_nodes):
            r_pyr = r_out[k, j, 0]
            r_som = r_out[k, j, 1]
            r_pv = r_out[k, j, 2]
            r_vip = r_out[k, j, 3]
            Iap = I_adapt_out[k, j, 0]
            Ias = I_adapt_out[k, j, 1]

            denom = 1.0 + ggaba * w_pe * r_pv

            I_pyr_j = (
                (w_ee * r_pyr) / denom
                + I_pyr_inter[j]
                - ggaba * I_pv_inter[j]
                - ggaba * w_se * r_som
                - Iap
                + I_ext_pyr_arr[k]
                + I_stim_arr[k, j]
            )
            I_som_j = w_es * r_pyr - w_vs * r_vip - Ias + I_ext_som_arr[k]
            I_pv_j = (
                w_ep * r_pyr
                - ggaba * w_pp * r_pv
                - ggaba * w_sp * r_som
                - w_vp * r_vip
                + I_ext_pv_arr[k]
            )
            I_vip_j = w_ev * r_pyr + I_ext_vip_arr[k]

            phi_pyr = _phi_scalar(I_pyr_j, Theta_pyr, alpha_pyr, g)
            phi_som = _phi_scalar(I_som_j, Theta_som, alpha_som, g)
            phi_pv = _phi_scalar(I_pv_j, Theta_pv, alpha_pv, g)
            phi_vip = _phi_scalar(I_vip_j, Theta_vip, alpha_vip, g)

            dr_pyr = (-r_pyr + phi_pyr + sigma_s * noise_arr[k, j, 0]) / tau_s
            dr_som = (-r_som + phi_som + sigma_s * noise_arr[k, j, 1]) / tau_s
            dr_pv = (-r_pv + phi_pv + sigma_s * noise_arr[k, j, 2]) / tau_s
            dr_vip = (-r_vip + phi_vip + sigma_s * noise_arr[k, j, 3]) / tau_s

            r_out[k + 1, j, 0] = min(200.0, max(0.0, r_pyr + dt_ms * dr_pyr))
            r_out[k + 1, j, 1] = min(200.0, max(0.0, r_som + dt_ms * dr_som))
            r_out[k + 1, j, 2] = min(200.0, max(0.0, r_pv + dt_ms * dr_pv))
            r_out[k + 1, j, 3] = min(200.0, max(0.0, r_vip + dt_ms * dr_vip))

            I_adapt_out[k + 1, j, 0] = Iap + dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
            I_adapt_out[k + 1, j, 1] = Ias + dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som
