"""
Ring attractor simulation functions — CPU/numpy implementation.

This module contains:
- RingSimulationResult: Data class for simulation output
- simulate_ring: Single simulation using numpy Euler integration
- simulate_ring_batch: Batch simulation using numpy vectorization (n_batch trials in parallel)
- mean_rates_ring: Compute mean firing rates after burn-in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..params import CircuitParams
from ..transfer import phi_wong_wang, phi_capped
from ..simulation import NoiseType
from ..constants import GAMMA_NMDA, TAU_NMDA_MS, R_MAX_PV, R_MAX_SOM, R_MAX_VIP
from ._fast_ring_loop import (
    _ring_euler_loop,
    NUMBA_AVAILABLE as RING_NUMBA_AVAILABLE,
)

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

    # Optional: adaptation current time courses (only when record_adaptation=True)
    # Shape: (n_recorded, n_nodes, 2) where 2 = [pyr_adapt, som_adapt]
    I_adapt_stored: Optional[np.ndarray] = None

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phi_numpy(I, theta, c, g):
    """
    Vectorized Wong-Wang transfer function. Works on arrays of any shape.
    Handles the z→0 limit analytically to avoid division by zero.
    """
    u = c * (I - theta)
    z = g * u
    denom = -np.expm1(np.minimum(-z, 700.0))
    out = np.where(np.abs(z) < 1e-8, 1.0 / g + u / 2.0, u / denom)
    return np.maximum(out, 0.0)


def _phi_capped_numpy(I, r_max, theta, c, g):
    """Vectorized hyperbolic soft ceiling applied to the Wong-Wang transfer function."""
    phi = _phi_numpy(I, theta, c, g)
    return r_max * phi / (r_max + phi)


def _precompute_stimulus(stimuli, node_angles_rad, dt_ms, n_steps):
    """
    Pre-compute stimulus current for all n_steps timesteps.

    Returns a numpy array of shape (n_steps, n_nodes).
    Runs once before the Euler loop.
    """
    n_nodes = len(node_angles_rad)
    if not stimuli:
        return np.zeros((n_steps, n_nodes))
    I_stim = np.zeros((n_steps, n_nodes))
    for k in range(n_steps):
        t = k * dt_ms
        for stim in stimuli:
            I_stim[k] += compute_stimulus_current(stim, node_angles_rad, t)
    return I_stim


def _precompute_ext_currents(p: CircuitParams, n_steps: int, dt_ms: float):
    """
    Pre-compute external currents for all timesteps (handles transient).

    Returns four numpy arrays of shape (n_steps,): pyr, som, pv, vip.
    """
    I_pyr = np.full(n_steps, p.I_ext_pyr())
    I_som = np.full(n_steps, p.I_ext_som())
    I_pv = np.full(n_steps, p.I_ext_pv())
    I_vip = np.full(n_steps, p.I_ext_vip())
    if p.trans_enabled:
        k0 = min(int(p.trans_start_ms / dt_ms), n_steps)
        k1 = min(int((p.trans_start_ms + p.trans_duration_ms) / dt_ms), n_steps)
        I_pyr[k0:k1] += p.trans_factor * p.I0_pyr
    return I_pyr, I_som, I_pv, I_vip


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

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
    record_adaptation: bool = False,
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
        record_adaptation: If True, also record adaptation currents at every
            recording step. Result will have I_adapt_stored of shape
            (n_recorded, n_nodes, 2). Default False.

    Returns:
        RingSimulationResult with recorded simulation output
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be > 0")

    n_nodes = ring_params.n_nodes
    n_steps = int(np.floor(T_ms / dt_ms)) + 1

    # Recording interval
    record_step = max(1, round(record_dt_ms / dt_ms))
    n_recorded = (n_steps - 1) // record_step + 1
    last_recorded_k = (n_recorded - 1) * record_step
    need_extra_final = last_recorded_k < (n_steps - 1)
    if need_extra_final:
        n_recorded += 1

    # Pre-compute connectivity if not provided
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params, local_params)

    # Allocate recorded arrays
    r_stored = np.zeros((n_recorded, n_nodes, 4), dtype=float)
    t_stored = np.zeros(n_recorded, dtype=float)
    if record_adaptation:
        I_adapt_stored = np.zeros((n_recorded, n_nodes, 2), dtype=float)
    else:
        I_adapt_stored = None

    # Working state variables
    r_curr = np.zeros((n_nodes, 4), dtype=float)
    Iap_curr = np.zeros(n_nodes, dtype=float)  # PYR adaptation
    Ias_curr = np.zeros(n_nodes, dtype=float)  # SOM adaptation
    S_pyr = np.zeros(n_nodes, dtype=float)     # NMDA gating variable

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

    # Initialize NMDA gating from initial PYR rates
    S_pyr = (GAMMA_NMDA * r_curr[:, 0] * TAU_NMDA_MS) / \
            (1.0 + GAMMA_NMDA * r_curr[:, 0] * TAU_NMDA_MS)

    # Record initial state
    r_stored[0] = r_curr
    t_stored[0] = 0.0
    if record_adaptation:
        I_adapt_stored[0, :, 0] = Iap_curr
        I_adapt_stored[0, :, 1] = Ias_curr
    rec_idx = 0

    # Setup noise
    rng = np.random.default_rng(seed)
    xi_state = np.zeros(n_nodes, dtype=float)  # OU state

    # Cache parameters
    ggaba = local_params.g_gaba()
    p = local_params
    node_angles = ring_params.node_angles_rad

    noise_scale_pyr = p.sigma_noise * p.I_ext_pyr()
    noise_scale_som = p.sigma_noise * p.I_ext_som()
    noise_scale_pv  = p.sigma_noise * p.I_ext_pv()
    noise_scale_vip = p.sigma_noise * p.I_ext_vip()
    any_noise = p.sigma_noise != 0.0

    if any_noise and noise_type == "white":
        noise_arr = rng.standard_normal((n_steps - 1, n_nodes))
        wiener_arr = None
    elif any_noise and noise_type == "ou":
        noise_arr = None
        wiener_arr = rng.standard_normal((n_steps - 1, n_nodes))
    else:
        noise_arr = None
        wiener_arr = None

    I_stim_arr = _precompute_stimulus(stimuli, node_angles, dt_ms, n_steps - 1)
    I_ext_pyr_arr, I_ext_som_arr, I_ext_pv_arr, I_ext_vip_arr = _precompute_ext_currents(
        p, n_steps - 1, dt_ms,
    )

    # Fast path: use Numba when available for white/no-noise integration.
    if RING_NUMBA_AVAILABLE and noise_type in ("white", "none"):
        noise_nb = (
            noise_arr
            if (noise_arr is not None and any_noise)
            else np.zeros((n_steps - 1, n_nodes), dtype=float)
        )

        i_adapt_nb = np.zeros((n_recorded, n_nodes, 2), dtype=float)
        i_adapt_nb[0, :, 0] = Iap_curr
        i_adapt_nb[0, :, 1] = Ias_curr

        r_final_nb = np.empty((n_nodes, 4), dtype=float)
        i_adapt_final_nb = np.empty((n_nodes, 2), dtype=float)

        _ring_euler_loop(
            r_stored,
            i_adapt_nb,
            r_final_nb,
            i_adapt_final_nb,
            noise_nb,
            I_stim_arr,
            I_ext_pyr_arr,
            I_ext_som_arr,
            I_ext_pv_arr,
            I_ext_vip_arr,
            connectivity.W_pyr_pyr,
            connectivity.W_pv_pyr,
            connectivity.W_som_pyr,
            n_steps,
            n_nodes,
            record_step,
            dt_ms,
            float(noise_scale_pyr),
            float(noise_scale_som),
            float(noise_scale_pv),
            float(noise_scale_vip),
            float(p.tau_s),
            float(ggaba),
            S_pyr,
            float(p.w_es),
            float(p.w_vs),
            float(p.w_ep),
            float(p.w_pp),
            float(p.w_sp),
            float(p.w_vp),
            float(p.w_ev),
            float(p.J_adapt_pyr),
            float(p.tau_adapt_pyr),
            float(p.J_adapt_som),
            float(p.tau_adapt_som),
            float(p.Theta_pyr),
            float(p.alpha_pyr),
            float(p.g_exc),
            float(p.g_inh),
            float(p.Theta_som),
            float(p.alpha_som),
            float(p.Theta_pv),
            float(p.alpha_pv),
            float(p.Theta_vip),
            float(p.alpha_vip),
            float(R_MAX_PV),
            float(R_MAX_SOM),
            float(R_MAX_VIP),
        )

        # Fill time vector analytically
        n_base_rec = (n_steps - 1) // record_step
        t_stored[1:n_base_rec + 1] = (
            np.arange(1, n_base_rec + 1, dtype=float) * record_step * dt_ms
        )

        if need_extra_final:
            r_stored[-1] = r_final_nb
            t_stored[-1] = (n_steps - 1) * dt_ms
            i_adapt_nb[-1] = i_adapt_final_nb

        if record_adaptation:
            I_adapt_stored = i_adapt_nb

        r_curr = r_final_nb
        Iap_curr = i_adapt_final_nb[:, 0]
        Ias_curr = i_adapt_final_nb[:, 1]
    else:
        # Python fallback (OU noise or Numba unavailable)
        for k in range(n_steps - 1):
            r_pyr = r_curr[:, 0]
            r_som = r_curr[:, 1]
            r_pv  = r_curr[:, 2]
            r_vip = r_curr[:, 3]

            # === NOISE ===
            if not any_noise or noise_type == "none":
                xi = np.zeros(n_nodes)
            elif noise_type == "white":
                xi = noise_arr[k] if noise_arr is not None else np.zeros(n_nodes)
            elif noise_type == "ou":
                if tau_noise_ms <= 0:
                    raise ValueError("tau_noise_ms must be > 0 for OU noise")
                if wiener_arr is None:
                    w_step = rng.standard_normal(n_nodes)
                else:
                    w_step = wiener_arr[k]
                xi_state += (-xi_state / tau_noise_ms) * dt_ms + np.sqrt(
                    2.0 * dt_ms / tau_noise_ms,
                ) * w_step
                xi = xi_state
            else:
                raise ValueError(f"Unknown noise_type: {noise_type!r}")

            # === UPDATE ALL S_PYR (previous-step rates) ===
            dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
            S_pyr = np.clip(S_pyr + dS, 0.0, 1.0)

            # === MATRIX PRODUCTS ===
            I_pyr_nmda = connectivity.W_pyr_pyr @ S_pyr   # unified NMDA numerator
            I_pv_denom = connectivity.W_pv_pyr  @ r_pv    # for divisive denominator
            I_som_lat  = connectivity.W_som_pyr @ r_som   # lateral SOM inhibition

            I_stim = I_stim_arr[k]

            # === PER-NODE INPUT CURRENTS (vectorized) ===
            denom = 1.0 + ggaba * I_pv_denom            # fully divisive PV
            I_pyr = (
                I_pyr_nmda / denom                       # unified NMDA (incl. self)
                - ggaba * I_som_lat                      # lateral SOM (subtractive)
                - Iap_curr
                + I_ext_pyr_arr[k]
                + I_stim
                + noise_scale_pyr * xi
            )
            I_som = (
                p.w_es * r_pyr
                - p.w_vs * r_vip
                - Ias_curr
                + I_ext_som_arr[k]
                + noise_scale_som * xi
            )
            I_pv_curr = (
                p.w_ep * r_pyr
                - ggaba * p.w_pp * r_pv
                - ggaba * p.w_sp * r_som
                - p.w_vp * r_vip
                + I_ext_pv_arr[k]
                + noise_scale_pv * xi
            )
            I_vip = p.w_ev * r_pyr + I_ext_vip_arr[k] + noise_scale_vip * xi

            # === TRANSFER FUNCTION ===
            Phi_pyr = phi_wong_wang(I_pyr, theta=p.Theta_pyr, c=p.alpha_pyr, g=p.g_exc)
            Phi_som = _phi_capped_numpy(I_som, R_MAX_SOM, p.Theta_som, p.alpha_som, p.g_inh)
            Phi_pv  = _phi_capped_numpy(I_pv_curr, R_MAX_PV, p.Theta_pv, p.alpha_pv, p.g_inh)
            Phi_vip = _phi_capped_numpy(I_vip, R_MAX_VIP, p.Theta_vip, p.alpha_vip, p.g_inh)

            Phi = np.stack([Phi_pyr, Phi_som, Phi_pv, Phi_vip], axis=1)

            # === EULER UPDATE: FIRING RATES ===
            dr = (-r_curr + Phi) / p.tau_s
            r_curr = np.clip(r_curr + dt_ms * dr, 0.0, 200.0)

            # === EULER UPDATE: ADAPTATION ===
            dIap = (-Iap_curr + p.J_adapt_pyr * r_pyr) / p.tau_adapt_pyr
            Iap_curr = Iap_curr + dt_ms * dIap
            dIas = (-Ias_curr + p.J_adapt_som * r_som) / p.tau_adapt_som
            Ias_curr = Ias_curr + dt_ms * dIas

            # === RECORD ===
            next_k = k + 1
            if next_k % record_step == 0:
                rec_idx += 1
                r_stored[rec_idx] = r_curr
                t_stored[rec_idx] = next_k * dt_ms
                if record_adaptation:
                    I_adapt_stored[rec_idx, :, 0] = Iap_curr
                    I_adapt_stored[rec_idx, :, 1] = Ias_curr

    # Always record the final step if not already recorded
    if need_extra_final and noise_type not in ("white", "none"):
        rec_idx += 1
        r_stored[rec_idx] = r_curr
        t_stored[rec_idx] = (n_steps - 1) * dt_ms
        if record_adaptation:
            I_adapt_stored[rec_idx, :, 0] = Iap_curr
            I_adapt_stored[rec_idx, :, 1] = Ias_curr

    # Final adaptation state
    I_adapt_final = np.stack([Iap_curr, Ias_curr], axis=1)  # (n_nodes, 2)

    stim_info = stimuli[0] if stimuli else None
    return RingSimulationResult(
        t_ms=t_stored,
        r=r_stored,
        I_adapt_final=I_adapt_final,
        stim_angle_deg=stim_info.center_deg if stim_info else 0.0,
        stim_window=(stim_info.onset_ms, stim_info.offset_ms) if stim_info else (0, 0),
        ring_params=ring_params,
        local_params=local_params,
        I_adapt_stored=I_adapt_stored,
    )


# ---------------------------------------------------------------------------
# Batch simulation — numpy-vectorized over n_batch trials
# ---------------------------------------------------------------------------

def simulate_ring_batch(
    local_params_list: list[CircuitParams],
    ring_params: RingParams,
    T_ms: float,
    seeds: Optional[list[int]] = None,
    *,
    stimuli: Optional[list[RingStimulus]] = None,
    noise_type: NoiseType = "white",
    dt_ms: float = 0.1,
    record_dt_ms: float = 1.0,
    connectivity: Optional[RingConnectivity] = None,
    r0: Optional[np.ndarray] = None,
    I_adapt0: Optional[np.ndarray] = None,
) -> list[RingSimulationResult]:
    """
    Run multiple simulations in parallel using numpy batch vectorization.

    All simulations share the same ring_params, stimuli, and initial state.
    Each simulation can have different CircuitParams. Only ``seeds[0]`` is
    used to seed a single shared noise stream — trials within the batch
    share the same Wiener increments. To get independent noise per trial,
    call ``simulate_ring`` individually with distinct seeds.
    """
    if noise_type == "ou":
        raise ValueError("OU noise is not supported in batch mode. Use 'white' or 'none'.")

    n_batch = len(local_params_list)
    if seeds is None:
        seeds = list(range(n_batch))

    n_nodes = ring_params.n_nodes
    n_steps = int(np.floor(T_ms / dt_ms)) + 1
    record_step = max(1, round(record_dt_ms / dt_ms))
    n_recorded = (n_steps - 1) // record_step + 1
    n_scan_steps = n_recorded - 1
    n_total_used = n_scan_steps * record_step

    # Shared connectivity — use first params as representative for row-sums
    if connectivity is None:
        connectivity = RingConnectivity.from_params(ring_params, local_params_list[0])
    W_pyr_pyr = connectivity.W_pyr_pyr  # (n_nodes, n_nodes)
    W_pv_pyr  = connectivity.W_pv_pyr
    W_som_pyr = connectivity.W_som_pyr

    # Shared stimulus: (n_total_used, n_nodes)
    node_angles = ring_params.node_angles_rad
    I_stim_all = _precompute_stimulus(stimuli, node_angles, dt_ms, n_total_used)

    ext = [_precompute_ext_currents(p, n_total_used, dt_ms) for p in local_params_list]
    I_ext_pyr = np.stack([e[0] for e in ext], axis=1)  # (n_total_used, n_batch)
    I_ext_som = np.stack([e[1] for e in ext], axis=1)
    I_ext_pv  = np.stack([e[2] for e in ext], axis=1)
    I_ext_vip = np.stack([e[3] for e in ext], axis=1)

    def _arr(attr_fn):
        return np.array([float(attr_fn(p)) for p in local_params_list])[:, None]

    ggaba     = _arr(lambda p: p.g_gaba())
    w_es      = _arr(lambda p: p.w_es);   w_vs = _arr(lambda p: p.w_vs)
    w_ep      = _arr(lambda p: p.w_ep);   w_pp = _arr(lambda p: p.w_pp)
    w_sp      = _arr(lambda p: p.w_sp);   w_vp = _arr(lambda p: p.w_vp)
    w_ev      = _arr(lambda p: p.w_ev)
    tau_adapt_pyr = _arr(lambda p: p.tau_adapt_pyr)
    J_adapt_pyr   = _arr(lambda p: p.J_adapt_pyr)
    tau_adapt_som = _arr(lambda p: p.tau_adapt_som)
    J_adapt_som   = _arr(lambda p: p.J_adapt_som)
    Theta_pyr = _arr(lambda p: p.Theta_pyr);  alpha_pyr = _arr(lambda p: p.alpha_pyr)
    Theta_som = _arr(lambda p: p.Theta_som);  alpha_som = _arr(lambda p: p.alpha_som)
    Theta_pv  = _arr(lambda p: p.Theta_pv);   alpha_pv  = _arr(lambda p: p.alpha_pv)
    Theta_vip = _arr(lambda p: p.Theta_vip);  alpha_vip = _arr(lambda p: p.alpha_vip)
    g_exc = _arr(lambda p: p.g_exc)
    g_inh = _arr(lambda p: p.g_inh)
    noise_scale_pyr_batch = np.array([float(p.sigma_noise * p.I_ext_pyr()) for p in local_params_list])[:, None]
    noise_scale_som_batch = np.array([float(p.sigma_noise * p.I_ext_som()) for p in local_params_list])[:, None]
    noise_scale_pv_batch  = np.array([float(p.sigma_noise * p.I_ext_pv())  for p in local_params_list])[:, None]
    noise_scale_vip_batch = np.array([float(p.sigma_noise * p.I_ext_vip()) for p in local_params_list])[:, None]
    tau_s = np.array([float(p.tau_s) for p in local_params_list])[:, None, None]

    # Initial state: (n_batch, n_nodes, 4)
    if r0 is None:
        r = np.full((n_batch, n_nodes, 4), 0.1)
    else:
        r = np.tile(np.asarray(r0, dtype=float), (n_batch, 1, 1))

    # Adaptation: (n_batch, n_nodes)
    Iap = np.zeros((n_batch, n_nodes))
    Ias = np.zeros((n_batch, n_nodes))
    if I_adapt0 is not None:
        I_adapt0_np = np.asarray(I_adapt0, dtype=float)
        Iap = np.tile(I_adapt0_np[:, 0], (n_batch, 1))
        Ias = np.tile(I_adapt0_np[:, 1], (n_batch, 1))

    # NMDA gating: (n_batch, n_nodes)
    S_pyr = (GAMMA_NMDA * r[:, :, 0] * TAU_NMDA_MS) / \
            (1.0 + GAMMA_NMDA * r[:, :, 0] * TAU_NMDA_MS)

    use_noise = noise_type == "white" and any(p.sigma_noise != 0.0 for p in local_params_list)
    rng = np.random.default_rng(seeds[0] if seeds else 0) if use_noise else None

    r_all = np.empty((n_batch, n_recorded, n_nodes, 4))
    r_all[:, 0] = r
    rec_idx = 1

    for k in range(n_total_used):
        r_pyr = r[:, :, 0]  # (n_batch, n_nodes)
        r_som = r[:, :, 1]
        r_pv  = r[:, :, 2]
        r_vip = r[:, :, 3]

        # Update ALL S_pyr (previous-step rates)
        dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
        S_pyr = np.clip(S_pyr + dS, 0.0, 1.0)

        # Matrix products: (n_batch, n_nodes) = (n_batch, n_nodes) @ (n_nodes, n_nodes).T
        I_pyr_nmda = S_pyr @ W_pyr_pyr.T   # unified NMDA numerator
        I_pv_denom = r_pv  @ W_pv_pyr.T    # PV for divisive denominator
        I_som_lat  = r_som @ W_som_pyr.T   # lateral SOM inhibition

        I_ext_pyr_k = I_ext_pyr[k, :, None]  # (n_batch, 1)
        I_ext_som_k = I_ext_som[k, :, None]
        I_ext_pv_k  = I_ext_pv[k, :, None]
        I_ext_vip_k = I_ext_vip[k, :, None]

        denom  = 1.0 + ggaba * I_pv_denom          # (n_batch, n_nodes)
        I_pyr  = (I_pyr_nmda / denom
                  - ggaba * I_som_lat
                  - Iap + I_ext_pyr_k + I_stim_all[k])
        I_som  = w_es * r_pyr - w_vs * r_vip - Ias + I_ext_som_k
        I_pv_c = w_ep * r_pyr - ggaba * w_pp * r_pv - ggaba * w_sp * r_som \
                 - w_vp * r_vip + I_ext_pv_k
        I_vip  = w_ev * r_pyr + I_ext_vip_k

        if use_noise:
            xi = rng.standard_normal((n_batch, n_nodes))
            I_pyr  = I_pyr  + noise_scale_pyr_batch * xi
            I_som  = I_som  + noise_scale_som_batch * xi
            I_pv_c = I_pv_c + noise_scale_pv_batch  * xi
            I_vip  = I_vip  + noise_scale_vip_batch * xi

        Phi = np.stack([
            _phi_numpy(I_pyr,  Theta_pyr, alpha_pyr, g_exc),
            _phi_capped_numpy(I_som,  R_MAX_SOM, Theta_som, alpha_som, g_inh),
            _phi_capped_numpy(I_pv_c, R_MAX_PV,  Theta_pv,  alpha_pv,  g_inh),
            _phi_capped_numpy(I_vip,  R_MAX_VIP, Theta_vip, alpha_vip, g_inh),
        ], axis=-1)  # (n_batch, n_nodes, 4)

        dr = (-r + Phi) / tau_s
        r = np.clip(r + dt_ms * dr, 0.0, 200.0)

        Iap += dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
        Ias += dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som

        if (k + 1) % record_step == 0:
            r_all[:, rec_idx] = r
            rec_idx += 1

    t_np = np.arange(n_recorded, dtype=float) * record_step * dt_ms

    stim_info = stimuli[0] if stimuli else None
    results = []
    for i, lp in enumerate(local_params_list):
        I_adapt_final = np.stack([Iap[i], Ias[i]], axis=1)  # (n_nodes, 2)
        results.append(RingSimulationResult(
            t_ms=t_np.copy(),
            r=r_all[i],
            I_adapt_final=I_adapt_final,
            stim_angle_deg=stim_info.center_deg if stim_info else 0.0,
            stim_window=(stim_info.onset_ms, stim_info.offset_ms) if stim_info else (0, 0),
            ring_params=ring_params,
            local_params=lp,
        ))
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

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
        rr = result.r[max(start, end - window_steps):end]

    return np.mean(rr, axis=0)
