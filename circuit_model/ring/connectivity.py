"""
Connectivity matrices for the ring attractor network.

All weight matrices are row-sum normalised to the corresponding single-node
fitted scalar from CircuitParams, implementing the row-sum principle:

  PYR→PYR : Gaussian kernel WITH diagonal; row-sum = J_NMDA
  PV→PYR  : Uniform all-to-all WITH diagonal; row-sum = w_pe
  SOM→PYR : Gaussian kernel with ZERO diagonal (purely lateral); row-sum = w_se

At the homogeneous fixed point (all nodes firing identically), each kernel
produces the same total drive as the single-node model, preserving all
resting-state firing rates and bistable fixed points exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .params import RingParams

if TYPE_CHECKING:
    from ..params import CircuitParams


def angular_distance(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """
    Compute angular distance on a circle, handling wraparound.

    Parameters:
        theta1, theta2: Angles in radians (can be arrays)

    Returns:
        Angular distance in radians, range [0, pi]
    """
    diff = np.abs(theta1 - theta2)
    return np.minimum(diff, 2 * np.pi - diff)


def gaussian_profile(distance: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian connectivity profile: exp(-d^2 / (2*sigma^2))

    Parameters:
        distance: Angular distance in radians
        sigma: Width parameter in radians

    Returns:
        Weight values in [0, 1]
    """
    return np.exp(-distance**2 / (2 * sigma**2))


def build_pyr_pyr_weights(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build PYR→PYR weight matrix: Gaussian kernel WITH non-zero diagonal.

    The diagonal (self-weight) is the Gaussian evaluated at distance zero and
    replaces the former separate local J_NMDA term. The matrix-vector product
    W_pyr @ S_pyr gives the full NMDA-gated excitatory numerator at each node.

    Row-sum = local_params.J_NMDA (single-node fitted value).

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    angles = ring_params.node_angles_rad
    dist = angular_distance(angles[:, None], angles[None, :])
    W = gaussian_profile(dist, ring_params.sigma_pyr_rad)
    # Diagonal is Gaussian at distance 0 = 1.0 (kept, not zeroed)
    row_sum = W.sum(axis=1, keepdims=True)
    W = local_params.J_NMDA * W / np.maximum(row_sum, 1e-12)
    return W


def build_pv_pyr_weights(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build PV→PYR weight matrix: uniform all-to-all INCLUDING diagonal.

    All entries = w_pe / N, so row-sum = w_pe (single-node fitted value).
    PV inhibition enters the DIVISIVE denominator at each node; the uniform
    kernel means the denominator depends on the mean PV rate across all nodes.

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    n = ring_params.n_nodes
    W = np.full((n, n), local_params.w_pe / n)
    return W


def build_som_pyr_weights(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build SOM→PYR weight matrix: Gaussian kernel with ZERO diagonal (purely lateral).

    SOM interneurons receive input from local PYR only but project their
    inhibitory output to neighbouring columns (PYR_k → SOM_k → PYR_{l≠k}).
    The active node is released from SOM self-inhibition at the bump centre.

    Row-sum (excluding diagonal) = local_params.w_se (single-node fitted value).

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    angles = ring_params.node_angles_rad
    dist = angular_distance(angles[:, None], angles[None, :])
    W = gaussian_profile(dist, ring_params.sigma_som_rad)
    np.fill_diagonal(W, 0.0)  # zero diagonal — no self-inhibition via SOM
    row_sum = W.sum(axis=1, keepdims=True)
    W = local_params.w_se * W / np.maximum(row_sum, 1e-12)
    return W


@dataclass
class RingConnectivity:
    """Pre-computed connectivity matrices for efficient ring simulation."""

    W_pyr_pyr: np.ndarray  # PYR→PYR, shape (n_nodes, n_nodes); used as W @ S_pyr
    W_pv_pyr: np.ndarray   # PV→PYR,  shape (n_nodes, n_nodes); W @ r_pv goes into denom
    W_som_pyr: np.ndarray  # SOM→PYR, shape (n_nodes, n_nodes); W @ r_som is lateral inhib

    @classmethod
    def from_params(
        cls, ring_params: RingParams, local_params: "CircuitParams"
    ) -> "RingConnectivity":
        """Build all three connectivity matrices from ring and local parameters."""
        return cls(
            W_pyr_pyr=build_pyr_pyr_weights(ring_params, local_params),
            W_pv_pyr=build_pv_pyr_weights(ring_params, local_params),
            W_som_pyr=build_som_pyr_weights(ring_params, local_params),
        )
