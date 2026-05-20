"""
Compare interneuron transfer functions with and without hyperbolic soft ceiling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from circuit_model.transfer import phi_wong_wang, phi_capped
from circuit_model.params import CircuitParams
from circuit_model.constants import R_MAX_PV, R_MAX_SOM, R_MAX_VIP

p = CircuitParams()

I = np.linspace(-0.1, 1.5, 2000)

interneurons = [
    ("SOM", p.Theta_som, p.alpha_som, p.g_inh, R_MAX_SOM, "#e07b39"),
    ("PV",  p.Theta_pv,  p.alpha_pv,  p.g_inh, R_MAX_PV,  "#4c72b0"),
    ("VIP", p.Theta_vip, p.alpha_vip, p.g_inh, R_MAX_VIP, "#55a868"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
fig.suptitle("Interneuron transfer functions: original vs. soft-ceiling", fontsize=12)

for ax, (name, theta, alpha, g, r_max, color) in zip(axes, interneurons):
    phi_orig   = phi_wong_wang(I, theta=theta, c=alpha, g=g)
    phi_capped_ = phi_capped(I, r_max, theta=theta, c=alpha, g=g)

    ax.plot(I, phi_orig,    color="gray",  lw=2,   ls="--", label="Original $\\Phi(I)$", zorder=2)
    ax.plot(I, phi_capped_, color=color,   lw=2.5,          label="Capped $\\Phi_{\\rm cap}(I)$", zorder=3)
    ax.axhline(r_max, color=color, lw=1, ls=":", alpha=0.7, label=f"$r_{{\\rm max}}$ = {r_max:.0f} Hz")

    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Input current $I$ (nA)", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("Firing rate (Hz)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(I[0], I[-1])
    ax.set_ylim(0, max(r_max * 2.5, 130))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.1)

plt.tight_layout()
out = "figs/transfer_ceiling_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
