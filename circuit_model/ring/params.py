"""
Ring attractor network parameters.

This module contains the RingParams dataclass that defines the network
geometry and inter-node connectivity for the ring attractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RingParams:
    """
    Parameters for the ring attractor network.

    The ring consists of N nodes arranged in a circle. Each node is a full
    4-population local circuit (PYR, PV, SOM, VIP) with dynamics defined
    by CircuitParams (from circuit_model).

    Inter-node connectivity:
    - PYR→PYR: Local excitation with Gaussian profile (angular distance)
    - PV→PYR: Global inhibition of PYR (PV driven by local PYR, then inhibits PYR globally)
    - SOM, VIP: Local only (no inter-node connections)

    Attributes:
        n_nodes: Number of nodes on the ring (default: 64)
        w_pyr_pyr_inter: Total inter-node PYR→PYR coupling strength
            (row-sum normalized, independent of n_nodes). Used with Gaussian profile.
        sigma_pyr_deg: Width of Gaussian connectivity profile (degrees)
        pyr_profile_type: PYR→PYR connectivity profile ("gaussian" or "compte")
        J_plus: Local excitation strength for Compte et al. (2000) profile.
            Must be > 1.0 for bump formation. Only used when pyr_profile_type="compte".
        w_pv_global: Total inter-node PV→PYR global inhibition strength
            (row-sum normalized, independent of n_nodes)
        pv_global_type: Type of PV→PYR connectivity ("uniform" or "gaussian")
        sigma_pv_deg: Width of PV→PYR profile if gaussian (degrees)
    """

    # === Network geometry ===
    n_nodes: int = 64  # Number of nodes on ring (power of 2 recommended)

    # === Inter-node PYR→PYR excitation ===
    w_pyr_pyr_inter: float = 18.55  # Total coupling strength (Gaussian profile)
    sigma_pyr_deg: float = 30.0  # Width of Gaussian (degrees), ~30-60 typical
    pyr_profile_type: Literal["gaussian", "compte"] = "gaussian"
    J_plus: float = 1.6  # Compte profile local excitation strength (J+)

    # === Inter-node PV→PYR global inhibition ===
    w_pv_global: float = 0.3  # Total global PV→PYR inhibition strength
    pv_global_type: Literal["uniform", "gaussian"] = "uniform"
    sigma_pv_deg: float = 180.0  # If gaussian, width (180 = almost uniform)

    # === Derived properties ===
    @property
    def angular_spacing_deg(self) -> float:
        """Angular spacing between adjacent nodes (degrees)."""
        return 360.0 / self.n_nodes

    @property
    def angular_spacing_rad(self) -> float:
        """Angular spacing between adjacent nodes (radians)."""
        return 2 * np.pi / self.n_nodes

    @property
    def sigma_pyr_rad(self) -> float:
        """PYR connectivity width in radians."""
        return self.sigma_pyr_deg * np.pi / 180.0

    @property
    def sigma_pv_rad(self) -> float:
        """PV connectivity width in radians."""
        return self.sigma_pv_deg * np.pi / 180.0

    @property
    def node_angles_rad(self) -> np.ndarray:
        """Angular positions of all nodes in radians [0, 2pi)."""
        return np.linspace(0, 2 * np.pi, self.n_nodes, endpoint=False)

    @property
    def node_angles_deg(self) -> np.ndarray:
        """Angular positions of all nodes in degrees [0, 360)."""
        return np.linspace(0, 360, self.n_nodes, endpoint=False)

    def angle_to_node(self, angle_deg: float) -> int:
        """Convert an angle (degrees) to the nearest node index."""
        angle_normalized = angle_deg % 360.0
        return int(round(angle_normalized / self.angular_spacing_deg)) % self.n_nodes

    def node_to_angle_deg(self, node: int) -> float:
        """Convert a node index to its angular position (degrees)."""
        return (node % self.n_nodes) * self.angular_spacing_deg

    def node_to_angle_rad(self, node: int) -> float:
        """Convert a node index to its angular position (radians)."""
        return (node % self.n_nodes) * self.angular_spacing_rad
