"""
Ring attractor network parameters.

This module contains the RingParams dataclass that defines the network
geometry and inter-node connectivity for the ring attractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..params import ParamBound


@dataclass(frozen=True)
class RingParams:
    """
    Parameters for the ring attractor network.

    The ring consists of N nodes arranged in a circle. Each node is a full
    4-population local circuit (PYR, PV, SOM, VIP) with dynamics defined
    by CircuitParams (from circuit_model).

    Inter-node connectivity:
    - PYR→PYR: Local excitation with Gaussian profile (angular distance)
    - PV→PYR: Global uniform inhibition of PYR (PV driven by local PYR, then inhibits PYR globally)
    - SOM, VIP: Local only (no inter-node connections)

    Attributes:
        n_nodes: Number of nodes on the ring (default: 64)
        w_pyr_pyr_inter: Total inter-node PYR→PYR coupling strength
            (row-sum normalized, independent of n_nodes)
        sigma_pyr_deg: Width of Gaussian connectivity profile (degrees)
        w_pv_global: Total inter-node PV→PYR global inhibition strength
            (uniform all-to-all, independent of n_nodes)
    """

    # === Required inter-node connectivity (no defaults) ===
    w_pyr_pyr_inter: float  # Total coupling strength (Gaussian profile)
    w_pv_global: float  # Total global PV→PYR inhibition strength (uniform)

    # === Network geometry ===
    n_nodes: int = 64  # Number of nodes on ring (power of 2 recommended)

    # === Inter-node PYR→PYR excitation ===
    sigma_pyr_deg: float = 30.0  # Width of Gaussian (degrees), ~30-60 typical

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


def default_ring_bounds() -> "dict[str, ParamBound]":
    """
    Default search bounds for ring network parameters.

    Only covers the three optimizable scalar fields of RingParams.
    n_nodes is fixed by the user and not optimized.

    Units follow the W&W-grounded physical convention:
    - w_pyr_pyr_inter: nA/Hz  (inter-node PYR→PYR Gaussian row-sum)
    - w_pv_global:     nA/Hz  (uniform PV→PYR global inhibition)
    - sigma_pyr_deg:   degrees (spatial width, unit-independent)

    Working initialization (see params/init/network_ring_init.json):
    - w_pyr_pyr_inter = 4e-3 nA/Hz
    - w_pv_global     = 8e-3 nA/Hz
    - sigma_pyr_deg   = 15°

    These values are based on the W&W transfer function with A_x=1 (fixed),
    I0_pyr=0.44 nA.

    Turing window (analytical, 10× cue): [4.2e-3, 6.1e-3] nA/Hz for w_inter.
    Global PV effectively raises the practical upper bound for w_inter, so
    w_inter can safely reach ~1e-2 with w_pv balanced accordingly.

    w_pv_global upper bound (3e-2) corresponds to near-silent regime where
    inhibition suppresses all activity.
    """
    # Working WT solution: w_pyr_pyr_inter=0.0051, w_pv_global=0.027, sigma_pyr_deg=15.
    # Tightened from [5e-4, 5] → [5e-4, 0.1] / [5e-4, 0.2]; sigma from [5, 60] → [5, 40].
    from ..params import ParamBound
    return {
        "w_pyr_pyr_inter": ParamBound(lo=5e-4,  hi=0.1,  mode="log"),
        "w_pv_global":     ParamBound(lo=5e-4,  hi=0.2,  mode="log"),
        "sigma_pyr_deg":   ParamBound(lo=5.0,   hi=40.0, mode="lin"),
    }
