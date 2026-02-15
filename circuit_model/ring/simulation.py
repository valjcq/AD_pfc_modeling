"""
Ring attractor simulation functions.

This module contains:
- RingSimulationResult: Data class for simulation output
- simulate_ring: Main simulation function using Euler integration
- mean_rates_ring: Compute mean firing rates after burn-in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..params import CircuitParams
from ..transfer import phi_wong_wang
from ..simulation import NoiseType

from .params import RingParams
from .connectivity import RingConnectivity
from .stimulus import RingStimulus, compute_stimulus_current


@dataclass
class RingSimulationResult:
    """Container for ring attractor simulation output."""

    # Time information (recorded time points only)
    t_ms: np.ndarray  # Shape: (n_recorded,)

    # Firing rates: shape (n_recorded, n_nodes, 4) where 4 = [pyr, som, pv, vip]
    r: np.ndarray

    # Final adaptation currents: shape (n_nodes, 2) where 2 = [pyr, som]
    I_adapt_final: np.ndarray

    # Stimulus information
    stim_angle_deg: float  # Stimulus location in degrees (0 if no stimulus)
    stim_window: tuple[float, float]  # (onset_ms, offset_ms)

    # Parameters (for reference)
    ring_params: RingParams
    local_params: CircuitParams

    # Convenience properties
    @property
    def n_nodes(self) -> int:
        return self.r.shape[1]

    @property
    def n_steps(self) -> int:
        return self.r.shape[0]

    @property
    def stim_node(self) -> int:
        """Node index closest to stimulus location."""
        return self.ring_params.angle_to_node(self.stim_angle_deg)

    def get_pyr_activity(self) -> np.ndarray:
        """Return PYR firing rates: shape (n_recorded, n_nodes)."""
        return self.r[:, :, 0]

    def get_population(self, pop: int) -> np.ndarray:
        """Return activity for a specific population (0=PYR, 1=SOM, 2=PV, 3=VIP)."""
        return self.r[:, :, pop]


def simulate_ring(
    local_params: CircuitParams,
    ring_params: RingParams,
    T_ms: float,
    dt_ms: float = 0.1,
    *,
    stimuli: Optional[list[RingStimulus]] = None,
    r0: Optional[np.ndarray] = None,
    I_adapt0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    noise_type: NoiseType = "white",
    tau_noise_ms: float = 5.0,
    connectivity: Optional[RingConnectivity] = None,
    record_dt_ms: float = 1.0,
) -> RingSimulationResult:
    """
    Simulate the ring attractor network using Euler integration.

    The integration runs at *dt_ms* resolution for numerical accuracy, but
    only records the state every *record_dt_ms* (default 1 ms) to save memory.

    Parameters:
        local_params: CircuitParams for local 4-population dynamics
        ring_params: RingParams for network structure
        T_ms: Total simulation time (ms)
        dt_ms: Integration time step (ms), default 0.1
        stimuli: List of RingStimulus objects (optional)
        r0: Initial firing rates, shape (n_nodes, 4) or None
        I_adapt0: Initial adaptation currents, shape (n_nodes, 2) or None
        seed: Random seed for reproducibility
        noise_type: "none", "white", or "ou"
        tau_noise_ms: OU noise time constant
        connectivity: Pre-computed connectivity (computed if None)
        record_dt_ms: Recording time step (ms). Only every record_dt_ms
            the state is stored in the output arrays. Default 1.0.

    Returns:
        RingSimulationResult with recorded simulation output
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_nodes = ring_params.n_nodes
    n_steps = int(np.floor(T_ms / dt_ms)) + 1

    # Recording interval
    record_step = max(1, round(record_dt_ms / dt_ms))
    # Recorded indices: 0, record_step, 2*record_step, ..., and always the last step
    n_recorded = (n_steps - 1) // record_step + 1
    # Check if we need an extra slot for the final step
    last_recorded_k = (n_recorded - 1) * record_step
    need_extra_final = last_recorded_k < (n_steps - 1)
    if need_extra_final:
        n_recorded += 1

    # Pre-compute connectivity if not provided
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params)

    # Allocate recorded arrays
    r_stored = np.zeros((n_recorded, n_nodes, 4), dtype=float)
    t_stored = np.zeros(n_recorded, dtype=float)

    # Working state variables (small, not stored per step)
    r_curr = np.zeros((n_nodes, 4), dtype=float)
    Iap_curr = np.zeros(n_nodes, dtype=float)  # PYR adaptation
    Ias_curr = np.zeros(n_nodes, dtype=float)  # SOM adaptation

    # Set initial conditions
    if r0 is None:
        r_curr[:] = 0.1
    else:
        r0 = np.asarray(r0, dtype=float)
        if r0.shape != (n_nodes, 4):
            raise ValueError(f"r0 must have shape ({n_nodes}, 4)")
        r_curr[:] = r0

    if I_adapt0 is not None:
        I_adapt0 = np.asarray(I_adapt0, dtype=float)
        if I_adapt0.shape != (n_nodes, 2):
            raise ValueError(f"I_adapt0 must have shape ({n_nodes}, 2)")
        Iap_curr[:] = I_adapt0[:, 0]
        Ias_curr[:] = I_adapt0[:, 1]

    # Record initial state
    r_stored[0] = r_curr
    t_stored[0] = 0.0
    rec_idx = 0

    # Setup noise
    rng = np.random.default_rng(seed)
    xi_state = np.zeros((n_nodes, 4), dtype=float)

    # Cache parameters
    ggaba = local_params.g_gaba()
    p = local_params  # Shorthand
    node_angles = ring_params.node_angles_rad

    # External currents (base values, always computed)
    I_ext_pyr_base = p.I_ext_pyr()
    I_ext_som_base = p.I_ext_som()
    I_ext_pv_base = p.I_ext_pv()
    I_ext_vip_base = p.I_ext_vip()

    # Pre-compute transient additions (nonspecific current to all populations)
    use_transient = p.trans_enabled
    if use_transient:
        trans_k0 = int(p.trans_start_ms / dt_ms)
        trans_k1 = int((p.trans_start_ms + p.trans_duration_ms) / dt_ms)
        dI_pyr = p.trans_factor * p.I0_pyr
        dI_som = p.trans_factor * p.I0_som
        dI_pv = p.trans_factor * p.I0_pv
        dI_vip = p.trans_factor * p.I0_vip
    else:
        trans_k0 = n_steps + 1  # never reached
        trans_k1 = n_steps + 1

    # Main simulation loop
    for k in range(n_steps - 1):
        t_ms_k = k * dt_ms

        # External currents (with transient if in window)
        if trans_k0 <= k < trans_k1:
            I_ext_pyr_val = I_ext_pyr_base + dI_pyr
            I_ext_som_val = I_ext_som_base + dI_som
            I_ext_pv_val = I_ext_pv_base + dI_pv
            I_ext_vip_val = I_ext_vip_base + dI_vip
        else:
            I_ext_pyr_val = I_ext_pyr_base
            I_ext_som_val = I_ext_som_base
            I_ext_pv_val = I_ext_pv_base
            I_ext_vip_val = I_ext_vip_base

        # Current state
        r_pyr = r_curr[:, 0]
        r_som = r_curr[:, 1]
        r_pv = r_curr[:, 2]
        r_vip = r_curr[:, 3]

        # === INTER-NODE CURRENTS ===
        I_pyr_inter, I_pv_pyr_inter = connectivity.compute_inter_node_inputs(r_pyr, r_pv)

        # === STIMULUS CURRENT ===
        I_stim = np.zeros(n_nodes)
        if stimuli:
            for stim in stimuli:
                I_stim += compute_stimulus_current(stim, node_angles, t_ms_k)

        # === NOISE ===
        if p.sigma_s == 0.0 or noise_type == "none":
            xi = np.zeros((n_nodes, 4))
        elif noise_type == "white":
            xi = rng.standard_normal((n_nodes, 4))
        elif noise_type == "ou":
            if tau_noise_ms <= 0:
                raise ValueError("tau_noise_ms must be > 0 for OU noise")
            xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                2.0 * dt_ms / tau_noise_ms
            ) * rng.standard_normal((n_nodes, 4))
            xi = xi_state
        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # === COMPUTE INPUT CURRENTS (vectorized over nodes) ===

        # PYR: local + inter-node excitation + stimulus
        # PV provides DIVISIVE (shunting) inhibition
        denom = 1.0 + ggaba * p.w_pe * r_pv
        I_pyr = (
            (p.w_ee * r_pyr) / denom  # Local recurrent excitation (divided by PV)
            + I_pyr_inter  # Inter-node PYR excitation (from neighbors)
            - ggaba * I_pv_pyr_inter  # Global PV→PYR inhibition (from all nodes)
            - ggaba * p.w_se * r_som  # SOM dendritic inhibition (subtractive)
            - Iap_curr  # Spike-frequency adaptation
            + I_ext_pyr_val  # External input
            + I_stim  # Stimulus current
        )

        # SOM: local only (no inter-node connections)
        I_som = (
            p.w_es * r_pyr  # Excitation from PYR
            - ggaba * p.w_ps * r_pv  # Inhibition from PV
            - p.w_vs * r_vip  # Inhibition from VIP (disinhibition pathway)
            - Ias_curr  # Spike-frequency adaptation
            + I_ext_som_val  # External input
        )

        # PV: local only (inter-node PV effect is on PYR, not PV)
        I_pv_curr = (
            p.w_ep * r_pyr  # Strong excitation from local PYR
            - ggaba * p.w_pp * r_pv  # Self-inhibition
            - ggaba * p.w_sp * r_som  # Weak inhibition from SOM
            - p.w_vp * r_vip  # Weak inhibition from VIP
            + I_ext_pv_val  # External input
        )

        # VIP: local only (no inter-node connections)
        I_vip = p.w_ev * r_pyr - p.w_vv * r_vip + I_ext_vip_val

        # === TRANSFER FUNCTION (vectorized) ===
        Phi_pyr = phi_wong_wang(I_pyr, theta=p.Theta_pyr, c=p.alpha_pyr, g=p.g_e)
        Phi_som = phi_wong_wang(I_som, theta=p.Theta_som, c=p.alpha_som, g=p.g_i)
        Phi_pv = phi_wong_wang(I_pv_curr, theta=p.Theta_pv, c=p.alpha_pv, g=p.g_i)
        Phi_vip = phi_wong_wang(I_vip, theta=p.Theta_vip, c=p.alpha_vip, g=p.g_i)

        Phi = np.stack([Phi_pyr, Phi_som, Phi_pv, Phi_vip], axis=1)

        # === EULER UPDATE: FIRING RATES ===
        # tau_s * dr/dt = -r + Phi(I) + sigma*xi
        dr = (-r_curr + Phi + p.sigma_s * xi) / p.tau_s
        r_curr = np.clip(r_curr + dt_ms * dr, 0.0, 200.0)

        # === EULER UPDATE: ADAPTATION ===
        # tau_adapt * dI_adapt/dt = -I_adapt + J_adapt * r
        dIap = (-Iap_curr + p.J_adapt_pyr * r_pyr) / p.tau_adapt_pyr
        dIas = (-Ias_curr + p.J_adapt_som * r_som) / p.tau_adapt_som
        Iap_curr = Iap_curr + dt_ms * dIap
        Ias_curr = Ias_curr + dt_ms * dIas

        # === RECORD ===
        next_k = k + 1
        if next_k % record_step == 0:
            rec_idx += 1
            r_stored[rec_idx] = r_curr
            t_stored[rec_idx] = next_k * dt_ms

    # Always record the final step if not already recorded
    if need_extra_final:
        rec_idx += 1
        r_stored[rec_idx] = r_curr
        t_stored[rec_idx] = (n_steps - 1) * dt_ms

    # Final adaptation state (for burn-in cache)
    I_adapt_final = np.stack([Iap_curr, Ias_curr], axis=1)  # (n_nodes, 2)

    # Build result
    stim_info = stimuli[0] if stimuli else None
    return RingSimulationResult(
        t_ms=t_stored,
        r=r_stored,
        I_adapt_final=I_adapt_final,
        stim_angle_deg=stim_info.center_deg if stim_info else 0.0,
        stim_window=(stim_info.onset_ms, stim_info.offset_ms) if stim_info else (0, 0),
        ring_params=ring_params,
        local_params=local_params,
    )


def mean_rates_ring(
    result: RingSimulationResult, burn_in_ms: float, window_ms: float
) -> np.ndarray:
    """
    Compute mean firing rates after burn-in period.

    Parameters:
        result: RingSimulationResult from simulate_ring
        burn_in_ms: Time to skip at start (for transients to settle)
        window_ms: Averaging window at end (0 = use all after burn-in)

    Returns:
        Array of shape (n_nodes, 4) with mean rates [pyr, som, pv, vip] per node
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
