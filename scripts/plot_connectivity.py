"""
Generate ring attractor connectivity matrix + weight profile plot.

Usage:
    python plot_connectivity.py [--save PATH]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from circuit_model.ring.params import RingParams
from circuit_model.ring.connectivity import build_pyr_pyr_weights, build_pv_pyr_weights


def plot_connectivity_matrices(
    ring_params: RingParams,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot PYR→PYR and PV→PYR weight matrices + row-0 weight profile.

    Returns the matplotlib Figure.
    """
    W_exc = build_pyr_pyr_weights(ring_params)
    W_inh = build_pv_pyr_weights(ring_params)

    n = ring_params.n_nodes
    angles_deg = ring_params.node_angles_deg  # shape (n,)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Title ---
    pyr_label = f"w_pyr_inter={ring_params.w_pyr_pyr_inter}, σ_pyr={ring_params.sigma_pyr_deg:.1f}°"
    fig.suptitle(
        f"Ring connectivity  |  N={n},  {pyr_label},  w_pv_global={ring_params.w_pv_global}",
        fontsize=11,
    )

    # --- Panel 1: PYR→PYR matrix ---
    ax = axes[0]
    im = ax.imshow(W_exc, aspect="auto", origin="lower", cmap="Reds")
    plt.colorbar(im, ax=ax, label="Weight $W_{ij}$")
    ax.set_title("PYR → PYR (excitatory inter-node)")
    ax.set_xlabel("Source node j")
    ax.set_ylabel("Target node i")

    # --- Panel 2: PV→PYR matrix ---
    ax = axes[1]
    im = ax.imshow(W_inh, aspect="auto", origin="lower", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Weight $W_{ij}$")
    ax.set_title("PV → PYR (inhibitory inter-node)")
    ax.set_xlabel("Source node j")
    ax.set_ylabel("Target node i")

    # --- Panel 3: Weight profile for row 0 ---
    ax = axes[2]
    ax.plot(angles_deg, W_exc[0], color="#D62728", label="PYR→PYR (exc)")
    ax.plot(angles_deg, -W_inh[0], color="#1F77B4", label="-PV→PYR (inh)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Weight profile (row 0)")
    ax.set_xlabel("Source node angle (°)")
    ax.set_ylabel("Weight from node 0")
    ax.legend()
    ax.set_xlim(0, 360)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot ring connectivity matrices")
    parser.add_argument("--n_nodes", type=int, default=128)
    parser.add_argument("--w_pyr_pyr_inter", type=float, default=4.0)
    parser.add_argument("--sigma_pyr_deg", type=float, default=30.0)
    parser.add_argument("--w_pv_global", type=float, default=4.0)
    parser.add_argument("--save", type=str, default=None, help="Save path for the figure")
    args = parser.parse_args()

    ring_params = RingParams(
        n_nodes=args.n_nodes,
        w_pyr_pyr_inter=args.w_pyr_pyr_inter,
        sigma_pyr_deg=args.sigma_pyr_deg,
        w_pv_global=args.w_pv_global,
    )

    fig = plot_connectivity_matrices(ring_params, save_path=args.save)
    plt.show()


if __name__ == "__main__":
    main()
