"""
Connectivity matrices for the ring attractor network.

This module contains functions to build pre-computed weight matrices
for inter-node connections in the ring attractor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import RingParams


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
    Build inter-node PYR→PYR weight matrix with row-sum normalized Gaussian profile.

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from j to i
    """
    angles = ring_params.node_angles_rad

    dist = angular_distance(angles[:, None], angles[None, :])
    W = gaussian_profile(dist, ring_params.sigma_pyr_rad)
    np.fill_diagonal(W, 0.0)

    row_sum = W.sum(axis=1, keepdims=True)
    W = ring_params.w_pyr_pyr_inter * W / np.maximum(row_sum, 1e-12)

    return W


def build_pv_pyr_weights(ring_params: RingParams) -> np.ndarray:
    """
    Build inter-node PV→PYR global inhibition weight matrix (uniform all-to-all).

    PV from all nodes inhibits PYR at each node. Combined with local
    PYR→PV excitation (w_ep), this creates the E→I→E loop:
    local PYR excites local PV, then PV globally inhibits PYR.

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from PV at j to PYR at i
    """
    n = ring_params.n_nodes

    W = np.ones((n, n)) * ring_params.w_pv_global / (n - 1)
    np.fill_diagonal(W, 0.0)

    return W


@dataclass
class RingConnectivity:
    """Pre-computed connectivity matrices for efficient simulation."""

    W_pyr_pyr: np.ndarray  # Inter-node PYR→PYR, shape (n_nodes, n_nodes)
    W_pv_pyr: np.ndarray  # Inter-node PV→PYR, shape (n_nodes, n_nodes)

    @classmethod
    def from_params(cls, ring_params: RingParams) -> "RingConnectivity":
        """Build connectivity from parameters."""
        return cls(
            W_pyr_pyr=build_pyr_pyr_weights(ring_params),
            W_pv_pyr=build_pv_pyr_weights(ring_params),
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
            I_pyr_inter: Inter-node excitatory input to PYR (from PYR→PYR), shape (n_nodes,)
            I_pv_pyr_inter: Inter-node inhibitory input to PYR (from global PV), shape (n_nodes,)
        """
        # PYR receives local excitation from neighboring PYR
        I_pyr_inter = self.W_pyr_pyr @ r_pyr

        # PYR receives global inhibition from PV at all nodes
        I_pv_pyr_inter = self.W_pv_pyr @ r_pv

        return I_pyr_inter, I_pv_pyr_inter
