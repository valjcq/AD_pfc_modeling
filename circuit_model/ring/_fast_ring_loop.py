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
def _phi_capped_scalar(I: float, r_max: float, theta: float, c: float, g: float) -> float:
    """Hyperbolic soft ceiling applied to the Wong-Wang transfer function.

    Used for interneurons: Phi_capped = r_max * Phi / (r_max + Phi).
    """
    phi = _phi_scalar(I, theta, c, g)
    return r_max * phi / (r_max + phi)


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
    W_pyr_pyr: np.ndarray,         # (n_nodes, n_nodes) — unified PYR→PYR; used as W @ S_pyr
    W_pv_pyr: np.ndarray,          # (n_nodes, n_nodes) — uniform PV→PYR; W @ r_pv → denom
    W_som_pyr: np.ndarray,         # (n_nodes, n_nodes) — lateral SOM→PYR; W @ r_som → subtr.
    n_steps: int,
    n_nodes: int,
    record_step: int,
    dt_ms: float,
    noise_scale_pyr: float,
    noise_scale_som: float,
    noise_scale_pv: float,
    noise_scale_vip: float,
    tau_s: float,
    ggaba: float,
    S_pyr_init: np.ndarray,    # (n_nodes,) — initial NMDA gating per node
    # Local scalar weights (PV/SOM/VIP population inputs — unchanged from single node)
    w_es: float,   # PYR→SOM
    w_vs: float,   # VIP→SOM
    w_ep: float,   # PYR→PV
    w_pp: float,   # PV→PV self
    w_sp: float,   # SOM→PV
    w_vp: float,   # VIP→PV
    w_ev: float,   # PYR→VIP
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
    # Interneuron soft ceilings (Hz)
    r_max_pv: float,
    r_max_som: float,
    r_max_vip: float,
) -> None:
    """Core Euler integration loop for ring simulations.

    Architecture (matches thesis §2.4):
    - PYR NMDA drive   : (W_pyr_pyr @ S_pyr)[k] / denom  — unified kernel, NMDA-gated
    - PV divisive denom: 1 + ggaba * (W_pv_pyr @ r_pv)[k] — all PV nodes in denominator
    - SOM lateral inhib: ggaba * (W_som_pyr @ r_som)[k]  — purely lateral, subtractive

    Loop order within each time step:
    1. Extract rate vectors for the previous step.
    2. Update ALL S_pyr values (using previous-step rates) — ensures the matrix
       product uses a consistent S snapshot (correct Euler ordering).
    3. Compute the three matrix products before the per-node inner loop.
    4. Per-node inner loop: compute inputs, transfer function, Euler update.
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

    # Working arrays for rate vectors and pre-computed inter-node currents
    r_pyr_k  = np.zeros(n_nodes)
    r_pv_k   = np.zeros(n_nodes)
    r_som_k  = np.zeros(n_nodes)
    I_pyr_nmda = np.zeros(n_nodes)  # W_pyr_pyr @ S_pyr — unified NMDA numerator
    I_pv_denom = np.zeros(n_nodes)  # W_pv_pyr  @ r_pv  — divisive PV denominator
    I_som_lat  = np.zeros(n_nodes)  # W_som_pyr @ r_som — lateral SOM inhibition
    S_pyr_curr = np.zeros(n_nodes)
    for j in range(n_nodes):
        S_pyr_curr[j] = S_pyr_init[j]

    rec_i = 1  # next recording slot index

    for k in range(n_steps - 1):
        # === 1. Extract rate vectors (previous step) ===
        for j in range(n_nodes):
            r_pyr_k[j] = r_curr[j, 0]
            r_pv_k[j]  = r_curr[j, 2]
            r_som_k[j] = r_curr[j, 1]

        # === 2. Update ALL S_pyr values using previous-step PYR rates ===
        for j in range(n_nodes):
            S_j = S_pyr_curr[j]
            dS  = (-S_j + (1.0 - S_j) * GAMMA_NMDA * r_pyr_k[j]) * (dt_ms / TAU_NMDA_MS)
            S_pyr_curr[j] = max(0.0, min(1.0, S_j + dS))

        # === 3. Matrix products (all use previous-step quantities) ===
        I_pyr_nmda[:] = np.dot(W_pyr_pyr, S_pyr_curr)   # unified NMDA numerator
        I_pv_denom[:] = np.dot(W_pv_pyr,  r_pv_k)        # PV for divisive denominator
        I_som_lat[:]  = np.dot(W_som_pyr, r_som_k)        # lateral SOM inhibition

        # === 4. Per-node Euler update ===
        for j in range(n_nodes):
            r_pyr = r_curr[j, 0]
            r_som = r_curr[j, 1]
            r_pv  = r_curr[j, 2]
            r_vip = r_curr[j, 3]
            Iap   = Iap_curr[j]
            Ias   = Ias_curr[j]

            xi_j = noise_arr[k, j]

            # Fully divisive PV denominator (local + inter-node all in denom)
            denom = 1.0 + ggaba * I_pv_denom[j]

            I_pyr_j = (
                I_pyr_nmda[j] / denom           # unified NMDA drive (Gaussian incl. self)
                - ggaba * I_som_lat[j]           # lateral SOM inhibition (subtractive)
                - Iap
                + I_ext_pyr_arr[k]
                + I_stim_arr[k, j]
                + noise_scale_pyr * xi_j
            )
            I_som_j = (
                w_es * r_pyr
                - w_vs * r_vip
                - Ias
                + I_ext_som_arr[k]
                + noise_scale_som * xi_j
            )
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
            phi_som = _phi_capped_scalar(I_som_j, r_max_som, Theta_som, alpha_som, g_inh)
            phi_pv  = _phi_capped_scalar(I_pv_j,  r_max_pv,  Theta_pv,  alpha_pv,  g_inh)
            phi_vip = _phi_capped_scalar(I_vip_j, r_max_vip, Theta_vip, alpha_vip, g_inh)

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
