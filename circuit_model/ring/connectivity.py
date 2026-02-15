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
    Build inter-node PYR→PYR weight matrix.

    Dispatches based on ring_params.pyr_profile_type:
    - "gaussian": Row-sum normalized Gaussian (original)
    - "compte": Compte et al. (2000) with surround inhibition

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from j to i
    """
    if ring_params.pyr_profile_type == "gaussian":
        return _build_pyr_pyr_weights_gaussian(ring_params)
    elif ring_params.pyr_profile_type == "compte":
        return build_pyr_pyr_weights_compte(ring_params)
    else:
        raise ValueError(
            f"Unknown pyr_profile_type: {ring_params.pyr_profile_type!r}. "
            f"Valid: 'gaussian', 'compte'"
        )


def _build_pyr_pyr_weights_gaussian(ring_params: RingParams) -> np.ndarray:
    """Build PYR→PYR weights with row-sum normalized Gaussian profile."""
    n = ring_params.n_nodes
    angles = ring_params.node_angles_rad

    dist = angular_distance(angles[:, None], angles[None, :])
    W = gaussian_profile(dist, ring_params.sigma_pyr_rad)
    np.fill_diagonal(W, 0.0)

    row_sum = W.sum(axis=1, keepdims=True)
    W = ring_params.w_pyr_pyr_inter * W / np.maximum(row_sum, 1e-12)

    return W


def build_pyr_pyr_weights_compte(ring_params: RingParams) -> np.ndarray:
    """
    Build inter-node PYR→PYR weight matrix using Compte et al. (2000) profile.

    W_ij = J_- + (J_+ - J_-) * exp(-d(theta_i, theta_j)^2 / (2*sigma^2))

    J_- is determined by the normalization constraint sum_{j!=i} W_ij = 1:
        J_- = (1 - J_+ * S) / (N - 1 - S)
    where S = sum_{j!=i} exp(-d_ij^2 / (2*sigma^2)).

    When J_- < 0, distant weights become negative (surround inhibition).
    Final matrix scaled by 1/N to preserve total input across network sizes.

    Reference:
        Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X.-J. (2000).
        Synaptic mechanisms and network dynamics underlying spatial working
        memory in a cortical network model. Cerebral Cortex, 10(9), 910-923.

    Parameters:
        ring_params: RingParams configuration (uses J_plus, sigma_pyr_rad)

    Returns:
        W: Shape (n_nodes, n_nodes), rows sum to 1/N (off-diagonal)
    """
    n = ring_params.n_nodes
    J_plus = ring_params.J_plus
    sigma = ring_params.sigma_pyr_rad
    angles = ring_params.node_angles_rad

    dist = angular_distance(angles[:, None], angles[None, :])
    G = np.exp(-dist**2 / (2 * sigma**2))
    np.fill_diagonal(G, 0.0)

    # S = off-diagonal Gaussian row sum (same for all rows by symmetry)
    S = G[0].sum()

    denom = n - 1 - S
    if abs(denom) < 1e-12:
        raise ValueError(
            f"Compte profile degenerate: N-1-S = {denom:.2e}. "
            f"sigma is too large relative to N."
        )
    J_minus = (1.0 - J_plus * S) / denom

    # W_ij = J_- + (J_+ - J_-) * G_ij
    W = J_minus + (J_plus - J_minus) * G
    np.fill_diagonal(W, 0.0)

    # Scale by 1/N to preserve total synaptic input across network sizes
    W /= n

    return W


def build_pv_pyr_weights(ring_params: RingParams) -> np.ndarray:
    """
    Build inter-node PV→PYR global inhibition weight matrix.

    PV from all nodes inhibits PYR at each node. Combined with local
    PYR→PV excitation (w_ep), this creates the E→I→E loop:
    local PYR excites local PV, then PV globally inhibits PYR.

    Can be uniform (all-to-all equal) or Gaussian (distance-dependent).

    Parameters:
        ring_params: RingParams configuration

    Returns:
        W: Shape (n_nodes, n_nodes), W[i,j] = weight from PV at j to PYR at i
    """
    n = ring_params.n_nodes

    if ring_params.pv_global_type == "uniform":
        # All-to-all with equal weights (excluding self)
        W = np.ones((n, n)) * ring_params.w_pv_global / (n - 1)
        np.fill_diagonal(W, 0.0)

    elif ring_params.pv_global_type == "gaussian":
        angles = ring_params.node_angles_rad
        dist = angular_distance(angles[:, None], angles[None, :])
        W = gaussian_profile(dist, ring_params.sigma_pv_rad)
        np.fill_diagonal(W, 0.0)
        # Normalize rows (consistent with uniform case where row sum = w_pv_global)
        row_sum = W.sum(axis=1, keepdims=True)
        W = ring_params.w_pv_global * W / np.maximum(row_sum, 1e-12)

    else:
        raise ValueError(f"Unknown pv_global_type: {ring_params.pv_global_type}")

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
