"""
Transfer function for the circuit model.

This module contains the Wong-Wang transfer function that converts
synaptic input current to firing rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def phi_wong_wang(I: Any, *, theta: float, c: float, g: float) -> np.ndarray:
    """
    Wong-Wang transfer function: converts synaptic input current to firing rate.

    This function originates from mean-field reduction of spiking neural networks
    (Wong & Wang, 2006, J. Neurosci.). It maps total synaptic input I to an
    output firing rate with biologically realistic saturation properties.

    Mathematical form:
        Phi(I) = u / (1 - exp(-g*u))  where u = c*(I - theta)

    Parameters:
        I: Input current (can be array)
        theta: Threshold current - input below this produces near-zero output
        c: Gain parameter - controls slope/sensitivity of the response
        g: Curvature parameter - controls saturation behavior

    Properties:
        - Monotonically increasing
        - Approximately linear near threshold (like f-I curve of neurons)
        - Saturates at high inputs (metabolic/biophysical limits)
        - Bounded below at 0 (firing rates cannot be negative)

    The function reduces to ReLU-like behavior as g -> infinity and to linear as g -> 0.
    """
    if g <= 0:
        raise ValueError("g must be > 0")
    if c < 0:
        raise ValueError("c must be >= 0")

    I = np.asarray(I, dtype=float)
    u = c * (I - theta)  # Shifted and scaled input
    z = g * u

    # Numerical stability: use expm1 for accurate computation of 1-exp(-z)
    # Cap z to prevent overflow in exp() for very large negative z
    denom = -np.expm1(np.minimum(-z, 700.0))

    # Near z=0, use Taylor expansion: u/(1-exp(-gu)) approx 1/g + u/2
    eps = 1e-8
    out = np.where(np.abs(z) < eps, 1.0 / g + u / 2.0, u / denom)
    return np.maximum(out, 0.0)  # Firing rates must be non-negative
