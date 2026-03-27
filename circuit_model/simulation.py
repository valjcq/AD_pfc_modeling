"""
Circuit simulation functions.

This module contains:
- SimulationResult: Data class for simulation output
- simulate_circuit: Main simulation function using Euler integration
- mean_rates: Compute mean firing rates after burn-in
- validate_fast_loop: Check that the Numba fast path matches the NumPy reference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .params import CircuitParams
from .transfer import phi_wong_wang
from ._fast_loop import _euler_loop, NUMBA_AVAILABLE


NoiseType = Literal["none", "white", "ou"]


@dataclass
class SimulationResult:
    """Container for simulation output."""
    t_ms: np.ndarray      # Shape: (n_steps,) - Time points in ms
    r: np.ndarray         # Shape: (n_steps, 4) - Firing rates [pyr, som, pv, vip]
    I_adapt: np.ndarray   # Shape: (n_steps, 2) - Adaptation currents [pyr, som]
    # Optional transient window info for plotting
    transient_window: Optional[tuple[float, float]] = None  # (start_ms, end_ms)


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
    """
    Simulate the 4-population circuit using Euler integration.

    Implements the rate equation:
        tau_s * dr/dt = -r + Phi(I_det) + sigma_s * xi(t)

    where each population (PYR, SOM, PV, VIP) has its own input current
    computed from synaptic connectivity and external inputs.

    Parameters:
        params: CircuitParams containing all model parameters
        T_ms: Total simulation time in milliseconds
        dt_ms: Integration time step (default 0.1ms for stability)
        r0: Initial firing rates [pyr, som, pv, vip] (default: 0.1 for all)
        I_adapt0: Initial adaptation currents [pyr, som] (default: 0)
        seed: Random seed for reproducibility
        noise_type: "none", "white" (Gaussian), or "ou" (Ornstein-Uhlenbeck)
        tau_noise_ms: Time constant for OU noise (if used)
        use_transient: If True and params.trans_enabled=True, apply time-dependent
                       transient current to ALL populations (trans_factor * I0_pop
                       is added during the transient window)

    Returns:
        SimulationResult with time points, firing rates, and adaptation currents
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    t = np.linspace(0.0, dt_ms * (n_steps - 1), n_steps, dtype=float)

    r = np.zeros((n_steps, 4), dtype=float)
    I_adapt = np.zeros((n_steps, 2), dtype=float)

    if r0 is None:
        r[0] = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
    else:
        r0 = np.asarray(r0, dtype=float)
        if r0.shape != (4,):
            raise ValueError("r0 must have shape (4,)")
        r[0] = r0

    if I_adapt0 is None:
        I_adapt[0] = 0.0
    else:
        I_adapt0 = np.asarray(I_adapt0, dtype=float)
        if I_adapt0.shape != (2,):
            raise ValueError("I_adapt0 must have shape (2,)")
        I_adapt[0] = I_adapt0

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    ggaba = params.g_gaba()

    # =========================================================================
    # FAST PATH — Numba JIT or pure-scalar fallback (both avoid NumPy overhead)
    #   Conditions: no time-varying transient, no OU noise.
    #   Speedup: ~50-100x with Numba, ~3-5x without (vs. current NumPy loop).
    # =========================================================================
    _can_use_fast = not use_transient and noise_type in ("none", "white")

    if _can_use_fast:
        # Pre-generate noise array — one vectorized call, no per-step overhead
        noise_scale = params.sigma_noise * params.I_ext_pyr()
        if noise_scale == 0.0 or noise_type == "none":
            noise_arr = np.zeros(n_steps - 1, dtype=np.float64)
        else:
            noise_arr = rng.standard_normal(n_steps - 1)

        _euler_loop(
            r, I_adapt, noise_arr,
            n_steps, dt_ms, float(noise_scale), float(params.tau_s),
            float(ggaba),
            float(params.w_ee), float(params.w_pe), float(params.w_se),
            float(params.w_es), float(params.w_vs),
            float(params.w_ep), float(params.w_pp), float(params.w_sp),
            float(params.w_vp), float(params.w_ev),
            float(params.J_adapt_pyr),
            float(params.tau_adapt_pyr),
            float(params.I_ext_pyr()), float(params.I_ext_som()),
            float(params.I_ext_pv()),  float(params.I_ext_vip()),
            float(params.Theta_pyr), float(params.alpha_pyr), float(params.g_exc),
            float(params.g_inh),
            float(params.Theta_som), float(params.alpha_som),
            float(params.Theta_pv),  float(params.alpha_pv),
            float(params.Theta_vip), float(params.alpha_vip),
            float(params.A_pyr), float(params.A_pv),
            float(params.A_som), float(params.A_vip),
        )

    else:
        # =====================================================================
        # REFERENCE PATH — original NumPy loop (OU noise or transient cases)
        # =====================================================================
        noise_scale = params.sigma_noise * params.I_ext_pyr()
        xi_state = 0.0  # scalar OU state for PYR

        for k in range(n_steps - 1):
            r_pyr, r_som, r_pv, r_vip = r[k]
            Iap = I_adapt[k, 0]  # Adaptation current for PYR only

            # NOISE GENERATION (PYR only, current-space)
            if noise_scale == 0.0 or noise_type == "none":
                xi_pyr = 0.0
            elif noise_type == "white":
                xi_pyr = float(rng.standard_normal())
            elif noise_type == "ou":
                if tau_noise_ms <= 0:
                    raise ValueError("tau_noise_ms must be > 0 for OU noise")
                xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                    2.0 * dt_ms / tau_noise_ms
                ) * float(rng.standard_normal())
                xi_pyr = xi_state
            else:
                raise ValueError(f"Unknown noise_type: {noise_type!r}")

            # EXTERNAL CURRENTS (time-dependent if transient enabled)
            if use_transient:
                I_ext_pyr_val = params.I_ext_pyr_at_time(t[k])
                I_ext_som_val = params.I_ext_som_at_time(t[k])
                I_ext_pv_val = params.I_ext_pv_at_time(t[k])
                I_ext_vip_val = params.I_ext_vip_at_time(t[k])
            else:
                I_ext_pyr_val = params.I_ext_pyr()
                I_ext_som_val = params.I_ext_som()
                I_ext_pv_val = params.I_ext_pv()
                I_ext_vip_val = params.I_ext_vip()

            # INPUT CURRENTS
            # PV provides DIVISIVE (shunting) inhibition: models perisomatic GABA
            denom = 1.0 + ggaba * params.w_pe * r_pv
            I_pyr = (
                (params.w_ee * r_pyr) / denom
                - ggaba * params.w_se * r_som
                - Iap
                + I_ext_pyr_val
                + noise_scale * xi_pyr  # current-space noise proportional to baseline PYR drive
            )
            I_som = (
                params.w_es * r_pyr
                - params.w_vs * r_vip
                + I_ext_som_val
            )
            I_pv = (
                params.w_ep * r_pyr
                - ggaba * params.w_pp * r_pv
                - ggaba * params.w_sp * r_som
                - params.w_vp * r_vip
                + I_ext_pv_val
            )
            I_vip = params.w_ev * r_pyr + I_ext_vip_val

            # TRANSFER FUNCTION
            Phi = np.array(
                [
                    phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc, A=params.A_pyr).item(),
                    phi_wong_wang(I_som, theta=params.Theta_som, c=params.alpha_som, g=params.g_inh, A=params.A_som).item(),
                    phi_wong_wang(I_pv, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_inh, A=params.A_pv).item(),
                    phi_wong_wang(I_vip, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_inh, A=params.A_vip).item(),
                ],
                dtype=float,
            )

            # EULER UPDATE: FIRING RATES
            dr = (-r[k] + Phi) / params.tau_s  # noise already in I_pyr above
            r[k + 1] = np.maximum(r[k] + dt_ms * dr, 0.0)

            # EULER UPDATE: ADAPTATION CURRENTS
            dIap = (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
            I_adapt[k + 1, 0] = Iap + dt_ms * dIap
            I_adapt[k + 1, 1] = 0.0

    # Compute transient window for plotting if enabled
    transient_window = None
    if use_transient and params.trans_enabled:
        trans_end = params.trans_start_ms + params.trans_duration_ms
        transient_window = (params.trans_start_ms, trans_end)

    return SimulationResult(t_ms=t, r=r, I_adapt=I_adapt, transient_window=transient_window)


def validate_fast_loop(
    params: Optional[CircuitParams] = None,
    T_ms: float = 500.0,
    dt_ms: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Verify that the fast (Numba/scalar) path is bit-identical to the reference.

    Runs both paths on the same inputs and asserts np.array_equal (exact match,
    not just close). Any divergence raises AssertionError.

    Usage:
        from circuit_model.simulation import validate_fast_loop
        validate_fast_loop()  # prints "OK — bit-identical" on success
    """
    if params is None:
        params = CircuitParams()

    r0 = np.array([2.0, 5.0, 15.0, 3.0], dtype=float)
    I_adapt0 = np.array([0.1, 0.5], dtype=float)

    # ---- Reference: original NumPy loop (force slow path) ----
    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    r_ref = np.zeros((n_steps, 4), dtype=float)
    I_adapt_ref = np.zeros((n_steps, 2), dtype=float)
    r_ref[0] = r0
    I_adapt_ref[0] = I_adapt0

    rng_ref = np.random.default_rng(seed)
    noise_ref = rng_ref.standard_normal(n_steps - 1)
    noise_scale = params.sigma_noise * params.I_ext_pyr()

    ggaba = params.g_gaba()
    for k in range(n_steps - 1):
        r_pyr, r_som, r_pv, r_vip = r_ref[k]
        Iap = I_adapt_ref[k, 0]
        xi_pyr = noise_ref[k]
        denom = 1.0 + ggaba * params.w_pe * r_pv
        I_pyr = (params.w_ee * r_pyr) / denom - ggaba * params.w_se * r_som - Iap + params.I_ext_pyr() + noise_scale * xi_pyr
        I_som = params.w_es * r_pyr - params.w_vs * r_vip + params.I_ext_som()
        I_pv  = params.w_ep * r_pyr - ggaba * params.w_pp * r_pv - ggaba * params.w_sp * r_som - params.w_vp * r_vip + params.I_ext_pv()
        I_vip = params.w_ev * r_pyr + params.I_ext_vip()
        Phi = np.array([
            phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_exc, A=params.A_pyr).item(),
            phi_wong_wang(I_som, theta=params.Theta_som, c=params.alpha_som, g=params.g_inh, A=params.A_som).item(),
            phi_wong_wang(I_pv,  theta=params.Theta_pv,  c=params.alpha_pv,  g=params.g_inh, A=params.A_pv).item(),
            phi_wong_wang(I_vip, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_inh, A=params.A_vip).item(),
        ])
        dr = (-r_ref[k] + Phi) / params.tau_s  # noise already in I_pyr
        r_ref[k + 1] = np.maximum(r_ref[k] + dt_ms * dr, 0.0)
        I_adapt_ref[k + 1, 0] = Iap + dt_ms * (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
        I_adapt_ref[k + 1, 1] = 0.0

    # ---- Fast path ----
    r_fast = np.zeros((n_steps, 4), dtype=float)
    I_adapt_fast = np.zeros((n_steps, 2), dtype=float)
    r_fast[0] = r0
    I_adapt_fast[0] = I_adapt0

    # Re-generate noise with same seed so the sequences are identical
    rng_fast = np.random.default_rng(seed)
    noise_fast = rng_fast.standard_normal(n_steps - 1)

    _euler_loop(
        r_fast, I_adapt_fast, noise_fast,
        n_steps, dt_ms, float(noise_scale), float(params.tau_s),
        float(ggaba),
        float(params.w_ee), float(params.w_pe), float(params.w_se),
        float(params.w_es), float(params.w_vs),
        float(params.w_ep), float(params.w_pp), float(params.w_sp),
        float(params.w_vp), float(params.w_ev),
        float(params.J_adapt_pyr),
        float(params.tau_adapt_pyr),
        float(params.I_ext_pyr()), float(params.I_ext_som()),
        float(params.I_ext_pv()),  float(params.I_ext_vip()),
        float(params.Theta_pyr), float(params.alpha_pyr), float(params.g_exc),
        float(params.g_inh),
        float(params.Theta_som), float(params.alpha_som),
        float(params.Theta_pv),  float(params.alpha_pv),
        float(params.Theta_vip), float(params.alpha_vip),
        float(params.A_pyr), float(params.A_pv),
        float(params.A_som), float(params.A_vip),
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
    """
    Compute mean firing rates after burn-in period.

    Parameters:
        result: SimulationResult from simulate_circuit
        burn_in_ms: Time to skip at start (for transients to settle)
        window_ms: Averaging window at end (0 = use all after burn-in)

    Returns:
        Array of shape (4,) with mean rates [pyr, som, pv, vip]
    """
    dt = float(result.t_ms[1] - result.t_ms[0])
    start = int(np.floor(burn_in_ms / dt))

    if window_ms <= 0:
        rr = result.r[start:]
    else:
        end = result.r.shape[0]
        window_steps = int(np.floor(window_ms / dt))
        rr = result.r[max(start, end - window_steps) : end]

    return np.mean(rr, axis=0)
