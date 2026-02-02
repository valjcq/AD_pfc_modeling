"""
Circuit simulation functions.

This module contains:
- SimulationResult: Data class for simulation output
- simulate_circuit: Main simulation function using Euler integration
- mean_rates: Compute mean firing rates after burn-in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .params import CircuitParams
from .transfer import phi_wong_wang


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
                       transient current to PYR (only during transient window)

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

    rng = np.random.default_rng(seed)
    xi_state = np.zeros(4, dtype=float)

    ggaba = params.g_gaba()

    for k in range(n_steps - 1):
        r_pyr, r_som, r_pv, r_vip = r[k]
        Iap, Ias = I_adapt[k]  # Adaptation currents for PYR and SOM

        # =====================================================================
        # NOISE GENERATION
        # =====================================================================
        # Noise represents stochastic synaptic input from unmodeled populations
        if params.sigma_s == 0.0 or noise_type == "none":
            xi = np.zeros(4, dtype=float)
        elif noise_type == "white":
            # White noise: independent Gaussian at each time step
            xi = rng.standard_normal(4)
        elif noise_type == "ou":
            # Ornstein-Uhlenbeck: temporally correlated noise (more realistic)
            # dxi = -xi/tau dt + sqrt(2/tau) dW
            if tau_noise_ms <= 0:
                raise ValueError("tau_noise_ms must be > 0 for OU noise")
            xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                2.0 * dt_ms / tau_noise_ms
            ) * rng.standard_normal(4)
            xi = xi_state
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # =====================================================================
        # COMPUTE INPUT CURRENTS FOR EACH POPULATION
        # =====================================================================

        # --- PYR INPUT ---
        # PV provides DIVISIVE (shunting) inhibition: models perisomatic GABA
        # synapses that reduce input resistance, effectively dividing excitation.
        # This is biologically accurate: PV targets soma/proximal dendrites.
        denom = 1.0 + ggaba * params.w_pe * r_pv  # Shunting denominator
        # Use time-dependent current if transient mode is enabled
        if use_transient:
            I_ext_pyr_val = params.I_ext_pyr_at_time(t[k])
        else:
            I_ext_pyr_val = params.I_ext_pyr()
        I_pyr = (
            (params.w_ee * r_pyr) / denom  # Recurrent excitation (divided by PV)
            - ggaba * params.w_se * r_som  # SOM dendritic inhibition (subtractive)
            - Iap                          # Spike-frequency adaptation
            + I_ext_pyr_val                # External input (baseline + transient)
        )

        # --- SOM INPUT ---
        # SOM receives excitation from PYR and inhibition from PV and VIP.
        # VIP->SOM is the core "disinhibition" pathway.
        I_som = (
            params.w_es * r_pyr            # Excitation from PYR
            - ggaba * params.w_ps * r_pv   # Inhibition from PV
            - params.w_vs * r_vip          # Inhibition from VIP (disinhibition pathway)
            - Ias                          # Spike-frequency adaptation
            + params.I_ext_som()           # External (baseline + alpha7 + beta2 currents)
        )

        # --- PV INPUT ---
        # PV receives strong excitation from PYR and inhibits itself.
        I_pv = (
            params.w_ep * r_pyr            # Strong excitation from PYR
            - ggaba * params.w_pp * r_pv   # Self-inhibition (limits PV rate)
            - ggaba * params.w_sp * r_som  # Weak inhibition from SOM
            - params.w_vp * r_vip          # Weak inhibition from VIP
            + params.I_ext_pv()            # External (baseline + alpha7 current)
        )

        # --- VIP INPUT ---
        # VIP receives weak input from PYR and self-inhibits.
        # VIP is largely driven by top-down or neuromodulatory inputs.
        I_vip = (
            params.w_ev * r_pyr   # Very weak excitation from PYR
            - params.w_vv * r_vip  # Self-inhibition
            + params.I_ext_vip()   # External (baseline + alpha5 current)
        )

        # =====================================================================
        # APPLY TRANSFER FUNCTION
        # =====================================================================
        # Convert input currents to firing rates via Wong-Wang function
        Phi = np.array(
            [
                phi_wong_wang(I_pyr, theta=params.Theta_pyr, c=params.alpha_pyr, g=params.g_e).item(),
                phi_wong_wang(I_som, theta=params.Theta_som, c=params.alpha_som, g=params.g_i).item(),
                phi_wong_wang(I_pv, theta=params.Theta_pv, c=params.alpha_pv, g=params.g_i).item(),
                phi_wong_wang(I_vip, theta=params.Theta_vip, c=params.alpha_vip, g=params.g_i).item(),
            ],
            dtype=float,
        )

        # =====================================================================
        # EULER UPDATE: FIRING RATES
        # =====================================================================
        # tau_s * dr/dt = -r + Phi(I) + sigma*xi
        dr = (-r[k] + Phi + params.sigma_s * xi) / params.tau_s
        r[k + 1] = np.maximum(r[k] + dt_ms * dr, 0.0)  # Enforce non-negative rates

        # =====================================================================
        # EULER UPDATE: ADAPTATION CURRENTS
        # =====================================================================
        # tau_adapt * dI_adapt/dt = -I_adapt + J_adapt * r
        # Adaptation builds up with firing and decays when silent
        dIap = (-Iap + params.J_adapt_pyr * r_pyr) / params.tau_adapt_pyr
        dIas = (-Ias + params.J_adapt_som * r_som) / params.tau_adapt_som
        I_adapt[k + 1, 0] = Iap + dt_ms * dIap
        I_adapt[k + 1, 1] = Ias + dt_ms * dIas

    # Compute transient window for plotting if enabled
    transient_window = None
    if use_transient and params.trans_enabled:
        trans_end = params.trans_start_ms + params.trans_duration_ms
        transient_window = (params.trans_start_ms, trans_end)

    return SimulationResult(t_ms=t, r=r, I_adapt=I_adapt, transient_window=transient_window)


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
