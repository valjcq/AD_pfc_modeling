"""
Connectivity matrices for the ring attractor network.

This module contains functions to build pre-computed weight matrices
for inter-node connections in the ring attractor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ring_params import RingParams


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


def build_pyr_pyr_weights(ring_params: RingParams) -> np.ndarray:
    """
    Build inter-node PYR→PYR weight matrix with Gaussian profile.

    The weight from node j to node i depends on their angular distance.
    Self-connections (diagonal) are set to 0 (handled by local w_ee).

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from j to i
    """
    n = ring_params.n_nodes
    angles = ring_params.node_angles_rad  # Shape: (n,)

    # Compute pairwise angular distances
    # angles[:, None] has shape (n, 1), angles[None, :] has shape (1, n)
    dist = angular_distance(angles[:, None], angles[None, :])  # Shape: (n, n)

    # Apply Gaussian profile
    W = ring_params.w_pyr_pyr_inter * gaussian_profile(dist, ring_params.sigma_pyr_rad)

    # Zero out diagonal (self-connections handled by local w_ee)
    np.fill_diagonal(W, 0.0)

    return W


def build_pv_global_weights(ring_params: RingParams) -> np.ndarray:
    """
    Build inter-node PV global inhibition weight matrix.

    Can be uniform (all-to-all equal) or Gaussian (distance-dependent).

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from j to i
    """
    n = ring_params.n_nodes

    if ring_params.pv_global_type == "uniform":
        # All-to-all with equal weights (excluding self)
        W = np.ones((n, n)) * ring_params.w_pv_global / (n - 1)
        np.fill_diagonal(W, 0.0)

    elif ring_params.pv_global_type == "gaussian":
        angles = ring_params.node_angles_rad
        dist = angular_distance(angles[:, None], angles[None, :])
        W = ring_params.w_pv_global * gaussian_profile(dist, ring_params.sigma_pv_rad)
        np.fill_diagonal(W, 0.0)

    else:
        raise ValueError(f"Unknown pv_global_type: {ring_params.pv_global_type}")

    return W


@dataclass
class RingConnectivity:
    """Pre-computed connectivity matrices for efficient simulation."""

    W_pyr_pyr: np.ndarray  # Inter-node PYR→PYR, shape (n_nodes, n_nodes)
    W_pv_global: np.ndarray  # Inter-node PV inhibition, shape (n_nodes, n_nodes)

    @classmethod
    def from_params(cls, ring_params: RingParams) -> "RingConnectivity":
        """Build connectivity from parameters."""
        return cls(
            W_pyr_pyr=build_pyr_pyr_weights(ring_params),
            W_pv_global=build_pv_global_weights(ring_params),
        )

    def compute_inter_node_inputs(
        self, r_pyr: np.ndarray, r_pv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute inter-node currents for all nodes.

        Parameters:
            r_pyr: PYR firing rates, shape (n_nodes,)
            r_pv: PV firing rates, shape (n_nodes,)

        Returns:
            I_pyr_inter: Inter-node excitatory input to PYR, shape (n_nodes,)
            I_pv_inter: Inter-node inhibitory input (from global PV), shape (n_nodes,)
        """
        # PYR receives excitation from neighboring PYR
        I_pyr_inter = self.W_pyr_pyr @ r_pyr  # Matrix-vector multiply

        # All nodes receive global PV inhibition
        I_pv_inter = self.W_pv_global @ r_pv

        return I_pyr_inter, I_pv_inter
