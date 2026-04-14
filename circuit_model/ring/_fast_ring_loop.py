"""Numba-compiled Euler integration for the ring attractor network."""

from __future__ import annotations

import math

import numpy as np

# NMDA gating constants (fixed physics, not fitted)
GAMMA_NMDA = 0.641
TAU_NMDA_MS = 100.0

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
    r_stored: np.ndarray,          # (n_recorded, n_nodes, 4) — r_stored[0]=initial state
    i_adapt_stored: np.ndarray,    # (n_recorded, n_nodes, 2) — first slot=initial adaptation
    r_final: np.ndarray,           # (n_nodes, 4) — OUTPUT: final state after loop
    i_adapt_final: np.ndarray,     # (n_nodes, 2) — OUTPUT: final adaptation after loop
    noise_arr: np.ndarray,         # (n_steps-1, n_nodes) — shared noise samples, or zeros
    I_stim_arr: np.ndarray,
    I_ext_pyr_arr: np.ndarray,
    I_ext_som_arr: np.ndarray,
    I_ext_pv_arr: np.ndarray,
    I_ext_vip_arr: np.ndarray,
    W_pyr_pyr: np.ndarray,
    W_pv_pyr: np.ndarray,
    n_steps: int,
    n_nodes: int,
    record_step: int,
    dt_ms: float,
    noise_scale_pyr: float,        # = sigma_noise * I_ext_pyr (nA)
    noise_scale_som: float,        # = sigma_noise * I_ext_som (nA)
    noise_scale_pv: float,         # = sigma_noise * I_ext_pv (nA)
    noise_scale_vip: float,        # = sigma_noise * I_ext_vip (nA)
    tau_s: float,
    ggaba: float,
    J_NMDA: float,
    S_pyr_init: np.ndarray,    # (n_nodes,) — initial NMDA gating per node
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
    g_exc: float,
    g_inh: float,
    Theta_som: float,
    alpha_som: float,
    Theta_pv: float,
    alpha_pv: float,
    Theta_vip: float,
    alpha_vip: float,
) -> None:
    """Core Euler integration loop for ring simulations.

    Uses O(n_nodes) working memory instead of O(n_steps * n_nodes): the working
    state is kept in small arrays (r_curr, Iap_curr, Ias_curr) that fit in L1 cache,
    and subsampled recordings are written directly into r_stored at every record_step
    steps.  The full trajectory is never materialised, reducing both peak memory and
    memory-bandwidth pressure.

    Parameters
    ----------
    r_stored : (n_recorded, n_nodes, 4) — written in-place; slot 0 must contain the
        initial firing-rate state on entry.
    i_adapt_stored : (n_recorded, n_nodes, 2) — written in-place; slot 0 must contain
        the initial adaptation currents on entry.
    r_final : (n_nodes, 4) — always overwritten with the state after the last step.
    i_adapt_final : (n_nodes, 2) — always overwritten with the adaptation after the
        last step.
    noise_arr : (n_steps-1, n_nodes) — pre-drawn N(0,1) samples. Injected as an
        additive current perturbation into each population: I_X += noise_scale_X * noise_arr[k, j].
        The same xi sample is shared across populations at each node; only the scale differs.
        Pass an all-zeros array to disable noise.
    noise_scale_pyr/som/pv/vip : scalar (nA) = sigma_noise * I_ext_X. Each population's
        noise amplitude is proportional to its own baseline external drive.
    record_step : write a recording every record_step integration steps (>= 1).
    n_steps : total number of time points including t=0 (loop runs n_steps-1 steps).
    """
    # Initialise working state from the first (pre-filled) slot
    r_curr = np.empty((n_nodes, 4))
    Iap_curr = np.empty(n_nodes)
    Ias_curr = np.empty(n_nodes)
    for j in range(n_nodes):
        r_curr[j, 0] = r_stored[0, j, 0]
        r_curr[j, 1] = r_stored[0, j, 1]
        r_curr[j, 2] = r_stored[0, j, 2]
        r_curr[j, 3] = r_stored[0, j, 3]
        Iap_curr[j] = i_adapt_stored[0, j, 0]
        Ias_curr[j] = i_adapt_stored[0, j, 1]

    I_pyr_inter = np.zeros(n_nodes)
    I_pv_inter = np.zeros(n_nodes)
    r_pyr_k = np.zeros(n_nodes)
    r_pv_k = np.zeros(n_nodes)
    S_pyr_curr = np.zeros(n_nodes)
    for j in range(n_nodes):
        S_pyr_curr[j] = S_pyr_init[j]

    rec_i = 1  # next recording slot index

    for k in range(n_steps - 1):
        # Build contiguous vectors for BLAS-backed dot products.
        for j in range(n_nodes):
            r_pyr_k[j] = r_curr[j, 0]
            r_pv_k[j] = r_curr[j, 2]

        I_pyr_inter[:] = np.dot(W_pyr_pyr, r_pyr_k)
        I_pv_inter[:] = np.dot(W_pv_pyr, r_pv_k)

        for j in range(n_nodes):
            r_pyr = r_curr[j, 0]
            r_som = r_curr[j, 1]
            r_pv  = r_curr[j, 2]
            r_vip = r_curr[j, 3]
            Iap   = Iap_curr[j]
            Ias   = Ias_curr[j]

            # NMDA gating update (before computing I_pyr_j)
            S_pyr_j = S_pyr_curr[j]
            dS = (-S_pyr_j + (1.0 - S_pyr_j) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
            S_pyr_j = max(0.0, min(1.0, S_pyr_j + dS))
            S_pyr_curr[j] = S_pyr_j

            denom = 1.0 + ggaba * w_pe * r_pv

            xi_j = noise_arr[k, j]  # shared noise sample at this node/step
            I_pyr_j = (
                (J_NMDA * S_pyr_j) / denom
                + I_pyr_inter[j]
                - ggaba * I_pv_inter[j]
                - ggaba * w_se * r_som
                - Iap
                + I_ext_pyr_arr[k]
                + I_stim_arr[k, j]
                + noise_scale_pyr * xi_j
            )
            I_som_j = w_es * r_pyr - w_vs * r_vip - Ias + I_ext_som_arr[k] + noise_scale_som * xi_j
            I_pv_j = (
                w_ep * r_pyr
                - ggaba * w_pp * r_pv
                - ggaba * w_sp * r_som
                - w_vp * r_vip
                + I_ext_pv_arr[k]
                + noise_scale_pv * xi_j
            )
            I_vip_j = w_ev * r_pyr + I_ext_vip_arr[k] + noise_scale_vip * xi_j

            phi_pyr = _phi_scalar(I_pyr_j, Theta_pyr, alpha_pyr, g_exc)
            phi_som = _phi_scalar(I_som_j, Theta_som, alpha_som, g_inh)
            phi_pv  = _phi_scalar(I_pv_j,  Theta_pv,  alpha_pv,  g_inh)
            phi_vip = _phi_scalar(I_vip_j, Theta_vip, alpha_vip, g_inh)

            dr_pyr = (-r_pyr + phi_pyr) / tau_s
            dr_som = (-r_som + phi_som) / tau_s
            dr_pv  = (-r_pv  + phi_pv)  / tau_s
            dr_vip = (-r_vip + phi_vip) / tau_s

            r_curr[j, 0] = min(200.0, max(0.0, r_pyr + dt_ms * dr_pyr))
            r_curr[j, 1] = min(200.0, max(0.0, r_som + dt_ms * dr_som))
            r_curr[j, 2] = min(200.0, max(0.0, r_pv  + dt_ms * dr_pv))
            r_curr[j, 3] = min(200.0, max(0.0, r_vip + dt_ms * dr_vip))

            Iap_curr[j] = Iap + dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
            Ias_curr[j] = Ias + dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som

        # Subsample directly into the output recording array
        if (k + 1) % record_step == 0:
            for j in range(n_nodes):
                r_stored[rec_i, j, 0] = r_curr[j, 0]
                r_stored[rec_i, j, 1] = r_curr[j, 1]
                r_stored[rec_i, j, 2] = r_curr[j, 2]
                r_stored[rec_i, j, 3] = r_curr[j, 3]
                i_adapt_stored[rec_i, j, 0] = Iap_curr[j]
                i_adapt_stored[rec_i, j, 1] = Ias_curr[j]
            rec_i += 1

    # Always expose the final state so the caller can handle need_extra_final
    for j in range(n_nodes):
        r_final[j, 0] = r_curr[j, 0]
        r_final[j, 1] = r_curr[j, 1]
        r_final[j, 2] = r_curr[j, 2]
        r_final[j, 3] = r_curr[j, 3]
        i_adapt_final[j, 0] = Iap_curr[j]
        i_adapt_final[j, 1] = Ias_curr[j]
