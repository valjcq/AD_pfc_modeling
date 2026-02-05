"""
Ring attractor simulation functions.

This module contains:
- RingSimulationResult: Data class for simulation output
- simulate_ring: Main simulation function using Euler integration
- mean_rates_ring: Compute mean firing rates after burn-in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

# Import from circuit_model package
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from circuit_model import CircuitParams, phi_wong_wang

from .ring_params import RingParams
from .connectivity import RingConnectivity
from .stimulus import RingStimulus, compute_stimulus_current


NoiseType = Literal["none", "white", "ou"]


@dataclass
class RingSimulationResult:
    """Container for ring attractor simulation output."""

    # Time information
    t_ms: np.ndarray  # Shape: (n_steps,)

    # Firing rates: shape (n_steps, n_nodes, 4) where 4 = [pyr, som, pv, vip]
    r: np.ndarray

    # Adaptation currents: shape (n_steps, n_nodes, 2) where 2 = [pyr, som]
    I_adapt: np.ndarray

    # Inter-node currents (for debugging/analysis)
    I_inter_pyr: np.ndarray  # Shape: (n_steps, n_nodes) - inter-node input to PYR
    I_inter_pv: np.ndarray  # Shape: (n_steps, n_nodes) - inter-node input to PV

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
        """Return PYR firing rates: shape (n_steps, n_nodes)."""
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
    noise_type: NoiseType = "none",
    tau_noise_ms: float = 5.0,
    connectivity: Optional[RingConnectivity] = None,
) -> RingSimulationResult:
    """
    Simulate the ring attractor network using Euler integration.

    Each node follows the same local dynamics as the original 4-population
    circuit, but with additional inter-node currents:
    - PYR receives: local recurrent + inter-node PYR excitation + stimulus
    - PV receives: local input + global PV inhibition from other nodes
    - SOM: local only
    - VIP: local only

    The rate equation for each population at each node:
        tau_s * dr/dt = -r + Phi(I_total) + sigma_s * xi(t)

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

    Returns:
        RingSimulationResult with full simulation output
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_nodes = ring_params.n_nodes
    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    t = np.linspace(0.0, dt_ms * (n_steps - 1), n_steps)

    # Pre-compute connectivity if not provided
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params)

    # Initialize state arrays
    r = np.zeros((n_steps, n_nodes, 4), dtype=float)
    I_adapt = np.zeros((n_steps, n_nodes, 2), dtype=float)
    I_inter_pyr = np.zeros((n_steps, n_nodes), dtype=float)
    I_inter_pv = np.zeros((n_steps, n_nodes), dtype=float)

    # Set initial conditions
    if r0 is None:
        r[0] = 0.1 * np.ones((n_nodes, 4))
    else:
        r0 = np.asarray(r0, dtype=float)
        if r0.shape != (n_nodes, 4):
            raise ValueError(f"r0 must have shape ({n_nodes}, 4)")
        r[0] = r0

    if I_adapt0 is not None:
        I_adapt0 = np.asarray(I_adapt0, dtype=float)
        if I_adapt0.shape != (n_nodes, 2):
            raise ValueError(f"I_adapt0 must have shape ({n_nodes}, 2)")
        I_adapt[0] = I_adapt0

    # Setup noise
    rng = np.random.default_rng(seed)
    xi_state = np.zeros((n_nodes, 4), dtype=float)

    # Cache parameters
    ggaba = local_params.g_gaba()
    p = local_params  # Shorthand
    node_angles = ring_params.node_angles_rad

    # External currents (static, no time-dependent transient for ring)
    I_ext_pyr_val = p.I_ext_pyr()
    I_ext_som_val = p.I_ext_som()
    I_ext_pv_val = p.I_ext_pv()
    I_ext_vip_val = p.I_ext_vip()

    # Main simulation loop
    for k in range(n_steps - 1):
        t_ms = t[k]

        # Current state
        r_k = r[k]  # Shape: (n_nodes, 4)
        r_pyr = r_k[:, 0]
        r_som = r_k[:, 1]
        r_pv = r_k[:, 2]
        r_vip = r_k[:, 3]
        Iap = I_adapt[k, :, 0]  # PYR adaptation
        Ias = I_adapt[k, :, 1]  # SOM adaptation

        # === INTER-NODE CURRENTS ===
        I_pyr_inter, I_pv_inter = connectivity.compute_inter_node_inputs(r_pyr, r_pv)
        I_inter_pyr[k] = I_pyr_inter
        I_inter_pv[k] = I_pv_inter

        # === STIMULUS CURRENT ===
        I_stim = np.zeros(n_nodes)
        if stimuli:
            for stim in stimuli:
                I_stim += compute_stimulus_current(stim, node_angles, t_ms)

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
            - ggaba * p.w_se * r_som  # SOM dendritic inhibition (subtractive)
            - Iap  # Spike-frequency adaptation
            + I_ext_pyr_val  # External input
            + I_stim  # Stimulus current
        )

        # SOM: local only (no inter-node connections)
        I_som = (
            p.w_es * r_pyr  # Excitation from PYR
            - ggaba * p.w_ps * r_pv  # Inhibition from PV
            - p.w_vs * r_vip  # Inhibition from VIP (disinhibition pathway)
            - Ias  # Spike-frequency adaptation
            + I_ext_som_val  # External input
        )

        # PV: local + global PV inhibition from other nodes
        I_pv = (
            p.w_ep * r_pyr  # Strong excitation from PYR
            - ggaba * p.w_pp * r_pv  # Self-inhibition
            - ggaba * p.w_sp * r_som  # Weak inhibition from SOM
            - p.w_vp * r_vip  # Weak inhibition from VIP
            + I_ext_pv_val  # External input
            - ggaba * I_pv_inter  # Global PV inhibition (inhibitory, hence minus)
        )

        # VIP: local only (no inter-node connections)
        I_vip = p.w_ev * r_pyr - p.w_vv * r_vip + I_ext_vip_val

        # === TRANSFER FUNCTION (vectorized) ===
        Phi_pyr = phi_wong_wang(I_pyr, theta=p.Theta_pyr, c=p.alpha_pyr, g=p.g_e)
        Phi_som = phi_wong_wang(I_som, theta=p.Theta_som, c=p.alpha_som, g=p.g_i)
        Phi_pv = phi_wong_wang(I_pv, theta=p.Theta_pv, c=p.alpha_pv, g=p.g_i)
        Phi_vip = phi_wong_wang(I_vip, theta=p.Theta_vip, c=p.alpha_vip, g=p.g_i)

        Phi = np.stack([Phi_pyr, Phi_som, Phi_pv, Phi_vip], axis=1)

        # === EULER UPDATE: FIRING RATES ===
        # tau_s * dr/dt = -r + Phi(I) + sigma*xi
        dr = (-r_k + Phi + p.sigma_s * xi) / p.tau_s
        r[k + 1] = np.maximum(r_k + dt_ms * dr, 0.0)

        # === EULER UPDATE: ADAPTATION ===
        # tau_adapt * dI_adapt/dt = -I_adapt + J_adapt * r
        dIap = (-Iap + p.J_adapt_pyr * r_pyr) / p.tau_adapt_pyr
        dIas = (-Ias + p.J_adapt_som * r_som) / p.tau_adapt_som
        I_adapt[k + 1, :, 0] = Iap + dt_ms * dIap
        I_adapt[k + 1, :, 1] = Ias + dt_ms * dIas

    # Fill last step of inter-node currents
    r_pyr_last = r[-1, :, 0]
    r_pv_last = r[-1, :, 2]
    I_inter_pyr[-1], I_inter_pv[-1] = connectivity.compute_inter_node_inputs(
        r_pyr_last, r_pv_last
    )

    # Build result
    stim_info = stimuli[0] if stimuli else None
    return RingSimulationResult(
        t_ms=t,
        r=r,
        I_adapt=I_adapt,
        I_inter_pyr=I_inter_pyr,
        I_inter_pv=I_inter_pv,
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
