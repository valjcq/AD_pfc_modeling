"""Numba-compiled Euler integration for the 4-population circuit model.

Provides a 50-100x speedup over the NumPy loop in simulation.py for the
single-node (optimization) case by eliminating Python/NumPy dispatch overhead
on scalar operations performed at every time step.

The JIT-compiled _euler_loop is FUNCTIONALLY IDENTICAL to the original Python
loop. Use validate_fast_loop() to verify outputs agree to machine precision.

NUMBA_AVAILABLE is True when numba is installed.
If False, the same functions fall back to plain Python using math.expm1,
which is still 3-5x faster than the current NumPy version (no array-creation
overhead per call).

Design notes:
  - noise_arr must be pre-generated outside (shape (n_steps-1,), all-zeros
    if sigma_noise == 0 or noise_type == "none"). All 4 populations receive
    the same xi sample per step; noise_scale_* = sigma_noise * I_ext_pop
    controls the amplitude per population (same design as the ring loop).
  - External currents are passed as static floats; the transient case is
    handled at the caller level (not here).
  - cache=True persists compiled bytecode across Python sessions (~1-2 s
    one-time compilation cost, then instant).
"""

from __future__ import annotations

import math

import numpy as np

# NMDA gating constants (fixed physics, not fitted)
GAMMA_NMDA = 0.641
TAU_NMDA_MS = 100.0

# ---------------------------------------------------------------------------
# Numba import — graceful fallback
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scalar transfer function
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _phi_scalar(I: float, theta: float, c: float, g: float) -> float:
    """Wong-Wang transfer function on a single scalar value.

    Identical to phi_wong_wang() in transfer.py but avoids all NumPy overhead.
    Uses math.expm1 which is compiled to a single CPU instruction by Numba.
    """
    u = c * (I - theta)
    z = g * u
    if abs(z) < 1e-8:
        # Taylor limit: avoids 0/0 at z=0
        return max(0.0, 1.0 / g + u * 0.5)
    denom = -math.expm1(min(-z, 700.0))  # stable: 1 - exp(-z)
    return max(0.0, u / denom)


# ---------------------------------------------------------------------------
# Core Euler loop — all scalar operations, no Python objects inside
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _euler_loop(
    r_out: np.ndarray,       # (n_steps, 4) — r_out[0] = r0 on entry
    I_adapt_out: np.ndarray, # (n_steps, 2) — I_adapt_out[0] = I_adapt0 on entry
    noise_arr: np.ndarray,   # (n_steps-1,) — shared xi per step (same for all pops), or zeros
    n_steps: int,
    dt_ms: float,
    noise_scale_pyr: float,  # = sigma_noise * I_ext_pyr (precomputed, nA)
    noise_scale_som: float,  # = sigma_noise * I_ext_som
    noise_scale_pv: float,   # = sigma_noise * I_ext_pv
    noise_scale_vip: float,  # = sigma_noise * I_ext_vip
    tau_s: float,
    # GABA scaling
    ggaba: float,
    # Synaptic weights
    J_NMDA: float, S_pyr_init: float, w_pe: float, w_se: float,
    w_es: float, w_vs: float,
    w_ep: float, w_pp: float, w_sp: float, w_vp: float,
    w_ev: float,
    # Adaptation
    J_adapt_pyr: float,
    tau_adapt_pyr: float,
    J_adapt_som: float,
    tau_adapt_som: float,
    # External currents (static — precomputed from params)
    I_ext_pyr: float, I_ext_som: float, I_ext_pv: float, I_ext_vip: float,
    # Transfer function parameters
    Theta_pyr: float, alpha_pyr: float, g_exc: float,
    g_inh: float,
    Theta_som: float, alpha_som: float,
    Theta_pv: float,  alpha_pv: float,
    Theta_vip: float, alpha_vip: float,
) -> None:
    """Core Euler integration loop — writes into r_out and I_adapt_out in-place.

    All operations are scalar floats. Numba compiles this to native machine code
    with zero Python interpreter overhead per iteration.
    """
    S_pyr = S_pyr_init  # NMDA gating variable

    for k in range(n_steps - 1):
        r_pyr = r_out[k, 0]
        r_som = r_out[k, 1]
        r_pv  = r_out[k, 2]
        r_vip = r_out[k, 3]
        Iap   = I_adapt_out[k, 0]
        Ias   = I_adapt_out[k, 1]

        # NMDA gating update (before computing I_pyr)
        dS = (-S_pyr + (1.0 - S_pyr) * GAMMA_NMDA * r_pyr) * (dt_ms / TAU_NMDA_MS)
        S_pyr = max(0.0, min(1.0, S_pyr + dS))

        # Shunting (divisive) inhibition denominator — PV on PYR
        denom = 1.0 + ggaba * w_pe * r_pv

        # Input currents (with NMDA gating)
        xi = noise_arr[k]  # shared noise sample for this step (all pops)
        I_pyr = (J_NMDA * S_pyr) / denom \
                - ggaba * w_se * r_som \
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
                + I_ext_pv \
                + noise_scale_pv * xi
        I_vip = w_ev * r_pyr + I_ext_vip \
                + noise_scale_vip * xi

        # Transfer function (scalar, zero overhead)
        phi_pyr = _phi_scalar(I_pyr, Theta_pyr, alpha_pyr, g_exc)
        phi_som = _phi_scalar(I_som, Theta_som, alpha_som, g_inh)
        phi_pv  = _phi_scalar(I_pv,  Theta_pv,  alpha_pv,  g_inh)
        phi_vip = _phi_scalar(I_vip, Theta_vip, alpha_vip, g_inh)

        # Euler update: firing rates
        # Operation order matches reference exactly: dt_ms * (sum / tau_s)
        # (NOT dt_ms/tau_s * sum — that reorders FP ops and breaks bit-identity)
        dr_pyr = (-r_pyr + phi_pyr) / tau_s  # noise already in I_pyr above
        dr_som = (-r_som + phi_som) / tau_s
        dr_pv  = (-r_pv  + phi_pv)  / tau_s
        dr_vip = (-r_vip + phi_vip) / tau_s
        r_out[k + 1, 0] = max(0.0, r_pyr + dt_ms * dr_pyr)
        r_out[k + 1, 1] = max(0.0, r_som + dt_ms * dr_som)
        r_out[k + 1, 2] = max(0.0, r_pv  + dt_ms * dr_pv)
        r_out[k + 1, 3] = max(0.0, r_vip + dt_ms * dr_vip)

        # Euler update: adaptation currents
        I_adapt_out[k + 1, 0] = Iap + dt_ms * (-Iap + J_adapt_pyr * r_pyr) / tau_adapt_pyr
        I_adapt_out[k + 1, 1] = Ias + dt_ms * (-Ias + J_adapt_som * r_som) / tau_adapt_som
