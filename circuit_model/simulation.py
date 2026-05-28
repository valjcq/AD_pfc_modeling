"""
Circuit simulation functions for the 5-population PFC model.

Population order in r: [PYR, SOM, PV, VIP, NDNF]  (indices 0..4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .params import CircuitParams
from .transfer import phi_wong_wang, phi_capped
from ._fast_loop import _euler_loop, NUMBA_AVAILABLE
from .constants import (
    GAMMA_NMDA, TAU_NMDA_MS,
    R_MAX_PV, R_MAX_SOM, R_MAX_VIP, R_MAX_NDNF,
)


NoiseType = Literal["none", "white", "ou"]


@dataclass
class SimulationResult:
    """Container for simulation output."""
    t_ms: np.ndarray      # (n_steps,)
    r: np.ndarray         # (n_steps, 5) — [PYR, SOM, PV, VIP, NDNF]
    I_adapt: np.ndarray   # (n_steps, 2) — [PYR, SOM]
    transient_window: Optional[tuple[float, float]] = None
    transient_window2: Optional[tuple[float, float]] = None


N_POPS = 5


def simulate_circuit(
    params: CircuitParams,
    T_ms: float,
    dt_ms: float = 0.1,
    r0: Optional[np.ndarray] = None,
    I_adapt0: Optional[np.ndarray] = None,
    *,
    seed: Optional[int] = None,
    noise_type: NoiseType = "none",
    tau_noise_ms: float = 5.0,
    use_transient: bool = False,
) -> SimulationResult:
    """Simulate the 5-population circuit using Euler integration."""
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    t = np.linspace(0.0, dt_ms * (n_steps - 1), n_steps, dtype=float)

    r = np.zeros((n_steps, N_POPS), dtype=float)
    I_adapt = np.zeros((n_steps, 2), dtype=float)

    if r0 is None:
        r[0] = np.full(N_POPS, 0.1, dtype=float)
    else:
        r0 = np.asarray(r0, dtype=float)
        if r0.shape != (N_POPS,):
            raise ValueError(f"r0 must have shape ({N_POPS},)")
        r[0] = r0

    if I_adapt0 is None:
        I_adapt[0] = np.array([0.0, 0.0], dtype=float)
    else:
        I_adapt0 = np.asarray(I_adapt0, dtype=float)
        if I_adapt0.shape != (2,):
            raise ValueError("I_adapt0 must have shape (2,)")
        I_adapt[0] = I_adapt0

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    ggaba = params.g_gaba()

    r_pyr_init = float(r[0, 0])
    S_pyr_init = (GAMMA_NMDA * r_pyr_init * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr_init * TAU_NMDA_MS)

    _can_use_fast = not use_transient and noise_type in ("none", "white")

    if _can_use_fast:
        noise_scale_pyr  = params.sigma_noise * params.I_ext_pyr()
        noise_scale_som  = params.sigma_noise * params.I_ext_som()
        noise_scale_pv   = params.sigma_noise * params.I_ext_pv()
        noise_scale_vip  = params.sigma_noise * params.I_ext_vip()
        noise_scale_ndnf = params.sigma_noise * params.I_ext_ndnf()
        any_noise = any(s != 0.0 for s in (
            noise_scale_pyr, noise_scale_som, noise_scale_pv, noise_scale_vip, noise_scale_ndnf,
        ))
        if not any_noise or noise_type == "none":
            noise_arr = np.zeros(n_steps - 1, dtype=np.float64)
        else:
            noise_arr = rng.standard_normal(n_steps - 1)

        _euler_loop(
            r, I_adapt, noise_arr,
            n_steps, dt_ms,
            float(noise_scale_pyr), float(noise_scale_som),
            float(noise_scale_pv),  float(noise_scale_vip),
            float(noise_scale_ndnf),
            float(params.tau_s),
            float(ggaba),
            # PYR-side / NMDA
            float(params.J_NMDA), float(S_pyr_init),
            float(params.w_pe), float(params.w_se), float(params.w_ne),
            # SOM input
            float(params.w_es), float(params.w_vs),
            # PV input
            float(params.w_ep), float(params.w_pp), float(params.w_sp),
            float(params.w_vp), float(params.w_np),
            # VIP input
            float(params.w_ev), float(params.w_nv),
            # NDNF input (PYR -> NDNF removed; NDNF receives SOM only)
            float(params.w_sn),
            # Adaptation
            float(params.J_adapt_pyr), float(params.tau_adapt_pyr),
            float(params.J_adapt_som), float(params.tau_adapt_som),
            # External currents
            float(params.I_ext_pyr()), float(params.I_ext_som()),
            float(params.I_ext_pv()),  float(params.I_ext_vip()),
            float(params.I_ext_ndnf()),
            # Transfer function params
            float(params.Theta_pyr), float(params.alpha_pyr), float(params.g_exc),
            float(params.g_inh),
            float(params.Theta_som),  float(params.alpha_som),
            float(params.Theta_pv),   float(params.alpha_pv),
            float(params.Theta_vip),  float(params.alpha_vip),
            float(params.Theta_ndnf), float(params.alpha_ndnf),
            # Soft ceilings
            float(R_MAX_PV), float(R_MAX_SOM), float(R_MAX_VIP), float(R_MAX_NDNF),
        )

    else:
        # =====================================================================
        # REFERENCE PATH — OU noise or transient
        # =====================================================================
        noise_scale_pyr  = params.sigma_noise * params.I_ext_pyr()
        noise_scale_som  = params.sigma_noise * params.I_ext_som()
        noise_scale_pv   = params.sigma_noise * params.I_ext_pv()
        noise_scale_vip  = params.sigma_noise * params.I_ext_vip()
        noise_scale_ndnf = params.sigma_noise * params.I_ext_ndnf()
        xi_state = 0.0
        S_pyr = S_pyr_init

        for k in range(n_steps - 1):
            r_pyr, r_som, r_pv, r_vip, r_ndnf = r[k]
            Iap = I_adapt[k, 0]
            Ias = I_adapt[k, 1]

            if noise_type == "none":
                xi = 0.0
            elif noise_type == "white":
                xi = float(rng.standard_normal())
            elif noise_type == "ou":
                if tau_noise_ms <= 0:
                    raise ValueError("tau_noise_ms must be > 0 for OU noise")
                xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                    2.0 * dt_ms / tau_noise_ms
                ) * float(rng.standard_normal())
                xi = xi_state
            else:
                raise ValueError(f"Unknown noise_type: {noise_type!r}")

            if use_transient:
                I_ext_pyr_val  = params.I_ext_pyr_at_time(t[k])
                I_ext_som_val  = params.I_ext_som_at_time(t[k])
                I_ext_pv_val   = params.I_ext_pv_at_time(t[k])
                I_ext_vip_val  = params.I_ext_vip_at_time(t[k])
                I_ext_ndnf_val = params.I_ext_ndnf_at_time(t[k])
            else:
                I_ext_pyr_val  = params.I_ext_pyr()
                I_ext_som_val  = params.I_ext_som()
                I_ext_pv_val   = params.I_ext_pv()
                I_ext_vip_val  = params.I_ext_vip()
                I_ext_ndnf_val = params.I_ext_ndnf()

            dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
            S_pyr = float(np.clip(S_pyr + dS, 0.0, 1.0))

            denom = 1.0 + ggaba * params.w_pe * r_pv
            I_pyr = (
                (params.J_NMDA * S_pyr) / denom
                - ggaba * params.w_se * r_som
                - ggaba * params.w_ne * r_ndnf
                - Iap
                + I_ext_pyr_val
                + noise_scale_pyr * xi
            )
            I_som = (
                params.w_es * r_pyr
                - params.w_vs * r_vip
                - params.J_adapt_som * r_som
                + I_ext_som_val
                + noise_scale_som * xi
            )
            I_pv = (
                params.w_ep * r_pyr
                - ggaba * params.w_pp * r_pv
                - ggaba * params.w_sp * r_som
                - params.w_vp * r_vip
                - ggaba * params.w_np * r_ndnf
                + I_ext_pv_val
                + noise_scale_pv * xi
            )
            I_vip = (
                params.w_ev * r_pyr
                - ggaba * params.w_nv * r_ndnf
                + I_ext_vip_val
                + noise_scale_vip * xi
            )
            I_ndnf = (
                - ggaba * params.w_sn * r_som
                + I_ext_ndnf_val
                + noise_scale_ndnf * xi
            )

            Phi = np.array(
                [
                    phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc).item(),
                    phi_capped(I_som,  R_MAX_SOM,  theta=params.Theta_som,  c=params.alpha_som,  g=params.g_inh).item(),
                    phi_capped(I_pv,   R_MAX_PV,   theta=params.Theta_pv,   c=params.alpha_pv,   g=params.g_inh).item(),
                    phi_capped(I_vip,  R_MAX_VIP,  theta=params.Theta_vip,  c=params.alpha_vip,  g=params.g_inh).item(),
                    phi_capped(I_ndnf, R_MAX_NDNF, theta=params.Theta_ndnf, c=params.alpha_ndnf, g=params.g_inh).item(),
                ],
                dtype=float,
            )

            dr = (-r[k] + Phi) / params.tau_s
            r[k + 1] = np.maximum(r[k] + dt_ms * dr, 0.0)

            dIap = (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
            I_adapt[k + 1, 0] = Iap + dt_ms * dIap
            dIas = (-Ias + params.J_adapt_som * r_som) / params.tau_adapt_som
            I_adapt[k + 1, 1] = Ias + dt_ms * dIas

    transient_window = None
    if use_transient and params.trans_enabled:
        transient_window = (params.trans_start_ms, params.trans_start_ms + params.trans_duration_ms)
    transient_window2 = None
    if use_transient and params.trans2_enabled:
        transient_window2 = (params.trans2_start_ms, params.trans2_start_ms + params.trans2_duration_ms)

    return SimulationResult(t_ms=t, r=r, I_adapt=I_adapt,
                            transient_window=transient_window, transient_window2=transient_window2)


def validate_fast_loop(
    params: Optional[CircuitParams] = None,
    T_ms: float = 500.0,
    dt_ms: float = 0.1,
    seed: int = 42,
) -> None:
    """Verify the fast path is bit-identical to the reference NumPy path."""
    if params is None:
        params = CircuitParams()

    r0 = np.array([2.0, 5.0, 15.0, 3.0, 4.0], dtype=float)
    I_adapt0 = np.array([0.1, 0.5], dtype=float)

    # Reference (slow) — re-run by calling simulate_circuit with conditions
    # that force the slow path. The simplest trick: use OU noise with seed,
    # which avoids the fast path. But we need bit-identical noise sequences.
    # Easier: directly replicate the slow loop here.
    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    r_ref = np.zeros((n_steps, N_POPS), dtype=float)
    I_adapt_ref = np.zeros((n_steps, 2), dtype=float)
    r_ref[0] = r0
    I_adapt_ref[0] = I_adapt0

    ns_pyr  = params.sigma_noise * params.I_ext_pyr()
    ns_som  = params.sigma_noise * params.I_ext_som()
    ns_pv   = params.sigma_noise * params.I_ext_pv()
    ns_vip  = params.sigma_noise * params.I_ext_vip()
    ns_ndnf = params.sigma_noise * params.I_ext_ndnf()

    rng_ref = np.random.default_rng(seed)
    noise_ref = rng_ref.standard_normal(n_steps - 1)

    r_pyr_init = float(r0[0])
    S_pyr_init = (GAMMA_NMDA * r_pyr_init * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr_init * TAU_NMDA_MS)
    S_pyr = S_pyr_init
    ggaba = params.g_gaba()

    for k in range(n_steps - 1):
        r_pyr, r_som, r_pv, r_vip, r_ndnf = r_ref[k]
        Iap = I_adapt_ref[k, 0]
        Ias = I_adapt_ref[k, 1]
        dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
        S_pyr = float(np.clip(S_pyr + dS, 0.0, 1.0))
        denom = 1.0 + ggaba * params.w_pe * r_pv
        xi = noise_ref[k]
        I_pyr = ((params.J_NMDA * S_pyr) / denom
                 - ggaba * params.w_se * r_som
                 - ggaba * params.w_ne * r_ndnf
                 - Iap + params.I_ext_pyr() + ns_pyr * xi)
        I_som = (params.w_es * r_pyr - params.w_vs * r_vip
                 - params.J_adapt_som * r_som + params.I_ext_som() + ns_som * xi)
        I_pv  = (params.w_ep * r_pyr - ggaba * params.w_pp * r_pv
                 - ggaba * params.w_sp * r_som - params.w_vp * r_vip
                 - ggaba * params.w_np * r_ndnf
                 + params.I_ext_pv() + ns_pv * xi)
        I_vip = (params.w_ev * r_pyr - ggaba * params.w_nv * r_ndnf
                 + params.I_ext_vip() + ns_vip * xi)
        I_ndnf = (- ggaba * params.w_sn * r_som
                  + params.I_ext_ndnf() + ns_ndnf * xi)
        Phi = np.array([
            phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc).item(),
            phi_capped(I_som,  R_MAX_SOM,  theta=params.Theta_som,  c=params.alpha_som,  g=params.g_inh).item(),
            phi_capped(I_pv,   R_MAX_PV,   theta=params.Theta_pv,   c=params.alpha_pv,   g=params.g_inh).item(),
            phi_capped(I_vip,  R_MAX_VIP,  theta=params.Theta_vip,  c=params.alpha_vip,  g=params.g_inh).item(),
            phi_capped(I_ndnf, R_MAX_NDNF, theta=params.Theta_ndnf, c=params.alpha_ndnf, g=params.g_inh).item(),
        ])
        dr = (-r_ref[k] + Phi) / params.tau_s
        r_ref[k + 1] = np.maximum(r_ref[k] + dt_ms * dr, 0.0)
        I_adapt_ref[k + 1, 0] = Iap + dt_ms * (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
        I_adapt_ref[k + 1, 1] = Ias + dt_ms * (-Ias + params.J_adapt_som * r_som) / params.tau_adapt_som

    # Fast path
    r_fast = np.zeros((n_steps, N_POPS), dtype=float)
    I_adapt_fast = np.zeros((n_steps, 2), dtype=float)
    r_fast[0] = r0
    I_adapt_fast[0] = I_adapt0

    rng_fast = np.random.default_rng(seed)
    noise_fast = rng_fast.standard_normal(n_steps - 1)

    _euler_loop(
        r_fast, I_adapt_fast, noise_fast,
        n_steps, dt_ms,
        float(ns_pyr), float(ns_som), float(ns_pv), float(ns_vip), float(ns_ndnf),
        float(params.tau_s),
        float(ggaba),
        float(params.J_NMDA), float(S_pyr_init),
        float(params.w_pe), float(params.w_se), float(params.w_ne),
        float(params.w_es), float(params.w_vs),
        float(params.w_ep), float(params.w_pp), float(params.w_sp),
        float(params.w_vp), float(params.w_np),
        float(params.w_ev), float(params.w_nv),
        float(params.w_sn),
        float(params.J_adapt_pyr), float(params.tau_adapt_pyr),
        float(params.J_adapt_som), float(params.tau_adapt_som),
        float(params.I_ext_pyr()), float(params.I_ext_som()),
        float(params.I_ext_pv()),  float(params.I_ext_vip()),
        float(params.I_ext_ndnf()),
        float(params.Theta_pyr), float(params.alpha_pyr), float(params.g_exc),
        float(params.g_inh),
        float(params.Theta_som),  float(params.alpha_som),
        float(params.Theta_pv),   float(params.alpha_pv),
        float(params.Theta_vip),  float(params.alpha_vip),
        float(params.Theta_ndnf), float(params.alpha_ndnf),
        float(R_MAX_PV), float(R_MAX_SOM), float(R_MAX_VIP), float(R_MAX_NDNF),
    )

    if not np.array_equal(r_fast, r_ref):
        max_err = np.max(np.abs(r_fast - r_ref) / (np.abs(r_ref) + 1e-30))
        raise AssertionError(
            f"Fast loop r is NOT bit-identical: max relative error = {max_err:.3e}"
        )
    if not np.array_equal(I_adapt_fast, I_adapt_ref):
        max_err = np.max(np.abs(I_adapt_fast - I_adapt_ref) / (np.abs(I_adapt_ref) + 1e-30))
        raise AssertionError(
            f"Fast loop I_adapt is NOT bit-identical: max relative error = {max_err:.3e}"
        )

    numba_status = "Numba JIT" if NUMBA_AVAILABLE else "plain-scalar fallback"
    print(f"validate_fast_loop OK — bit-identical ({numba_status}, T={T_ms}ms, {n_steps} steps)")


def mean_rates(result: SimulationResult, burn_in_ms: float, window_ms: float) -> np.ndarray:
    """Compute mean firing rates after burn-in. Returns shape (5,)."""
    dt = float(result.t_ms[1] - result.t_ms[0])
    start = int(np.floor(burn_in_ms / dt))

    if window_ms <= 0:
        rr = result.r[start:]
    else:
        end = result.r.shape[0]
        window_steps = int(np.floor(window_ms / dt))
        rr = result.r[max(start, end - window_steps) : end]

    return np.mean(rr, axis=0)
