"""
Ring attractor network parameters.

This module contains the RingParams dataclass that defines the network
geometry for the ring attractor. Inter-node connectivity row-sums are
derived directly from the single-node fitted CircuitParams (J_NMDA, w_pe,
w_se) via the row-sum normalisation principle — no additional ring-level
free parameters are introduced for the connection strengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RingParams:
    """
    Parameters for the ring attractor network.

    The ring consists of N nodes arranged on a circle. Each node is a full
    4-population local circuit (PYR, SOM, PV, VIP) with dynamics defined
    by CircuitParams.

    Inter-node connectivity (all row-sum normalised to fitted single-node scalars):
    - PYR→PYR: Gaussian kernel INCLUDING diagonal; row-sum = J_NMDA (from local params)
    - PV→PYR: Uniform all-to-all INCLUDING diagonal; row-sum = w_pe (from local params)
    - SOM→PYR: Gaussian/uniform kernel (lateral) or local-only diagonal; row-sum = w_se (from local params)

    The Gaussian widths are the only structural free parameters at the ring level.

    Attributes:
        n_nodes: Number of nodes on the ring (default: 64)
        sigma_pyr_deg: Width of Gaussian PYR→PYR kernel (degrees, default 15)
        sigma_som_deg: Width of Gaussian SOM→PYR lateral kernel (degrees, default 15)
        som_pattern: SOM→PYR connectivity pattern: "gaussian" (annular surround, default),
                     "uniform" (flat all-to-all with zero diagonal, same row-sum w_se), or
                     "none" (local only — diagonal matrix, no inter-node connections)
    """

    # === Network geometry ===
    n_nodes: int = 64  # Number of nodes on ring (power of 2 recommended)

    # === Gaussian kernel widths (degrees) — only free structural parameters ===
    sigma_pyr_deg: float = 15.0  # PYR→PYR Gaussian width
    sigma_som_deg: float = 15.0  # SOM→PYR annular ring half-width (peak at 3*sigma_pyr)

    # === SOM connectivity pattern ===
    som_pattern: str = "gaussian"  # "gaussian" (annular surround), "uniform" (all-to-all, zero diagonal), or "none" (local only)

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
        """PYR→PYR connectivity width in radians."""
        return self.sigma_pyr_deg * np.pi / 180.0

    @property
    def sigma_som_rad(self) -> float:
        """SOM→PYR lateral connectivity width in radians."""
        return self.sigma_som_deg * np.pi / 180.0

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


def default_ring_bounds() -> "dict[str, object]":
    """
    Default search bounds for ring network structural parameters.

    Only the Gaussian widths are optimisable at the ring level.
    Connection strengths are derived from the single-node fitted CircuitParams.
    """
    from ..params import ParamBound
    return {
        "sigma_pyr_deg": ParamBound(lo=5.0, hi=40.0, mode="lin"),
        "sigma_som_deg": ParamBound(lo=5.0, hi=60.0, mode="lin"),
    }
