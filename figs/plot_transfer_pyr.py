"""
PYR transfer function: raw Wong-Wang without any soft ceiling.
Single panel to accompany the interneuron transfer_ceiling_comparison figure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from circuit_model.transfer import phi_wong_wang
from circuit_model.params import CircuitParams

p = CircuitParams()

I = np.linspace(-0.1, 1.5, 2000)
phi_pyr = phi_wong_wang(I, theta=p.Theta_pyr, c=p.alpha_pyr, g=p.g_exc)

fig, ax = plt.subplots(figsize=(4.5, 4))
fig.suptitle("PYR transfer function", fontsize=12)

color = "#c44e52"
ax.plot(I, phi_pyr, color=color, lw=2.5, label="$\\Phi_{\\rm PYR}(I)$", zorder=3)

ax.axvline(p.Theta_pyr, color="gray", lw=1, ls=":", alpha=0.7,
           label=f"Threshold $\\Theta_e$ = {p.Theta_pyr:.3f} nA")

ax.set_title("PYR", fontsize=13, fontweight="bold")
ax.set_xlabel("Input current $I$ (nA)", fontsize=10)
ax.set_ylabel("Firing rate (Hz)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.set_xlim(I[0], I[-1])
ax.set_ylim(0, 300)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.25)
ax.grid(True, which="minor", alpha=0.1)

plt.tight_layout()
out = "figs/transfer_pyr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
