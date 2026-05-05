"""
Connectivity matrices for the ring attractor network.

All weight matrices are row-sum normalised to the corresponding single-node
fitted scalar from CircuitParams, implementing the row-sum principle:

  PYR→PYR : Gaussian kernel WITH diagonal; row-sum = J_NMDA
  PV→PYR  : Uniform all-to-all WITH diagonal; row-sum = w_pe
  SOM→PYR : Gaussian (annular), uniform, or local-only kernel; row-sum = w_se

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


def annular_gaussian_profile(distance: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Offset (annular) Gaussian peaked at distance mu from the source:
        exp(-(d - mu)^2 / (2*sigma^2))

    Used for SOM→PYR: inhibition is strongest at the edge of the PYR
    excitation zone (d ≈ mu) and falls off on both sides.

    Parameters:
        distance: Angular distance in radians
        mu:    Peak distance in radians (derived from sigma_pyr, not hardcoded)
        sigma: Width of the annular ring in radians

    Returns:
        Weight values in [0, 1]
    """
    return np.exp(-(distance - mu) ** 2 / (2 * sigma**2))


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


def build_som_pyr_weights_gaussian(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build SOM→PYR weight matrix: annular (offset) Gaussian that peaks at
    the edge of the PYR excitation zone and falls off on both sides.

    The peak distance mu is set to 2*sigma_pyr so the SOM surround starts
    where PYR excitation stops — derived automatically, never hardcoded.
    sigma_som controls the width of the inhibitory ring.

    Row-sum = local_params.w_se (single-node fitted value).

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    angles = ring_params.node_angles_rad
    dist = angular_distance(angles[:, None], angles[None, :])
    mu = 3.0 * ring_params.sigma_pyr_rad  # peak at 3-sigma of PYR Gaussian (edge of excitation zone)
    W = annular_gaussian_profile(dist, mu, ring_params.sigma_som_rad)
    np.fill_diagonal(W, 0.0)  # zero self-connections
    row_sum = W.sum(axis=1, keepdims=True)
    W = local_params.w_se * W / np.maximum(row_sum, 1e-12)
    return W


def build_som_pyr_weights_uniform(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build SOM→PYR weight matrix: uniform outside the PYR excitation zone.

    Profile: 1 - gaussian_profile(d, 2*sigma_som). The factor of 2 ensures
    neighboring nodes receive negligible inhibition (~1.7% of far-field at
    node+1 vs 6.8% without it). sigma_som_deg controls the hole size.

    Diagonal is zeroed (no self-inhibition). Row-sum = w_se.

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    angles = ring_params.node_angles_rad
    dist = angular_distance(angles[:, None], angles[None, :])
    W = 1.0 - gaussian_profile(dist, 2.0 * ring_params.sigma_som_rad)
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    W = local_params.w_se * W / np.maximum(row_sum, 1e-12)
    return W


def build_som_pyr_weights_none(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """
    Build SOM→PYR weight matrix: local only (no inter-node connections).

    Each node inhibits only itself with weight w_se, replicating the
    single-node fitted behaviour exactly. The matrix is diagonal with
    w_se on every entry.

    Returns:
        W: Shape (n_nodes, n_nodes)
    """
    return np.diag(np.full(ring_params.n_nodes, local_params.w_se))


def build_som_pyr_weights(ring_params: RingParams, local_params: "CircuitParams") -> np.ndarray:
    """Dispatch SOM→PYR weight builder based on ring_params.som_pattern."""
    if ring_params.som_pattern == "uniform":
        return build_som_pyr_weights_uniform(ring_params, local_params)
    if ring_params.som_pattern == "gaussian":
        return build_som_pyr_weights_gaussian(ring_params, local_params)
    if ring_params.som_pattern == "none":
        return build_som_pyr_weights_none(ring_params, local_params)
    raise ValueError(f"Unknown som_pattern: {ring_params.som_pattern!r}. Choose 'gaussian', 'uniform', or 'none'.")


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
