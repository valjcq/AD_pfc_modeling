"""
Generate all figures for docs/bistable_loss_guide.md.
Outputs PNGs to figs/docs/.

Usage (from project root):
    python scripts/generate_loss_guide_figs.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import fsolve, brentq

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from circuit_model.params import CircuitParams
from circuit_model.bistable_loss import (
    _compute_F_sweep, _solve_interneurons, bistable_loss, BistableConfig,
)
from circuit_model.jacobian import compute_jacobian
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "lines.linewidth": 2.2,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLORS = {
    "pyr": "#E05C5C",
    "som": "#5CA0E0",
    "pv":  "#5CB85C",
    "vip": "#E0A040",
    "zero": "#888888",
    "stable": "#1F77B4",
    "unstable": "#D62728",
    "zone1": "#AED6F1",
    "zone2": "#FAD7A0",
    "zone3": "#A9DFBF",
    "target": "#2C3E50",
    "actual": "#E74C3C",
    "proposed": "#27AE60",
}

OUT = ROOT / "figs" / "docs"
OUT.mkdir(parents=True, exist_ok=True)

# ── load params ───────────────────────────────────────────────────────────────
PARAMS_PATH = ROOT / "figs" / "optim" / "bistable_high_fr" / "best_params.json"
LOW_PEAK_PATH = ROOT / "figs" / "optim" / "bistable_high_fr_low_peak" / "best_params.json"

with open(PARAMS_PATH) as f:
    raw = json.load(f)

params = CircuitParams(**{k: v for k, v in raw.items() if k in CircuitParams.__dataclass_fields__})

# ── helpers ───────────────────────────────────────────────────────────────────
def compute_nullcline(p, r_max=80.0, n=1000):
    r_sweep = np.linspace(0.0, r_max, n)
    F = _compute_F_sweep(r_sweep, p)
    phi = F + r_sweep
    return r_sweep, F, phi


def find_fixed_points(r_sweep, F):
    """Return (stable_fps, unstable_fps) as lists of r values."""
    dF = np.gradient(F, r_sweep)
    sign_changes = np.where(np.diff(np.sign(F)))[0]
    stable, unstable = [], []
    for idx in sign_changes:
        r0, r1 = r_sweep[idx], r_sweep[idx + 1]
        try:
            # make a local closure capturing idx explicitly
            def _f(r, _r=r_sweep, _F=F, _idx=idx):
                return float(np.interp(r, _r, _F))
            rc = brentq(_f, r0, r1)
        except Exception:
            rc = 0.5 * (r0 + r1)
        if rc >= 98.0:
            continue
        dF_rc = float(np.interp(rc, r_sweep, dF))
        (stable if dF_rc < 0 else unstable).append(rc)
    return stable, unstable


def get_interneuron_sweep(p, r_sweep):
    rs = np.zeros_like(r_sweep)
    rp = np.zeros_like(r_sweep)
    rv = np.zeros_like(r_sweep)
    for i, r in enumerate(r_sweep):
        rs[i], rp[i], rv[i] = _solve_interneurons(float(r), p)
    return rs, rp, rv


# Pre-compute for main params
r_sweep, F, phi = compute_nullcline(params)
stable_fps, unstable_fps = find_fixed_points(r_sweep, F)
r_som_sw, r_pv_sw, r_vip_sw = get_interneuron_sweep(params, r_sweep)

# Fixed points
r_low  = stable_fps[0] if stable_fps else 0.0
r_high = stable_fps[-1] if len(stable_fps) > 1 else 78.0
r_uns  = unstable_fps[0] if unstable_fps else 35.0

r_som_low, r_pv_low, r_vip_low = _solve_interneurons(r_low, params)
r_som_high, r_pv_high, r_vip_high = _solve_interneurons(r_high, params)

# Rooy 2021 targets
LOW_TARGETS  = {"PYR": 1.75, "SOM": 1.12, "PV": 1.04, "VIP": 1.33}
HIGH_TARGETS = {"PYR": 60.2, "SOM": 35.2, "PV": 35.3, "VIP": 68.8}

print(f"Low FP:  PYR={r_low:.2f}, SOM={r_som_low:.2f}, PV={r_pv_low:.2f}, VIP={r_vip_low:.2f}")
print(f"High FP: PYR={r_high:.2f}, SOM={r_som_high:.2f}, PV={r_pv_high:.2f}, VIP={r_vip_high:.2f}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 1  —  Monostable vs Bistable nullcline concept
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig1_concept_nullcline.png …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left panel: monostable (synthetic, just a smooth decreasing F) ─────────
ax = axes[0]
r = np.linspace(0, 80, 500)
F_mono = 5 * np.exp(-r / 12) - 0.08 * r + 1.5   # always crosses zero once
F_mono -= F_mono[-1] + 0.5                         # shift so it's negative at the end
ax.plot(r, F_mono, color="#555555", lw=2.5, label=r"$F(r) = \Phi(I_\mathrm{net}(r)) - r$")
ax.axhline(0, color=COLORS["zero"], lw=1.2, ls="--", alpha=0.7)
fp_mono = float(np.interp(0.0, F_mono[::-1], r[::-1]))
ax.plot(fp_mono, 0, "o", color=COLORS["stable"], ms=12, zorder=5, label=f"Stable FP ({fp_mono:.0f} Hz)")
ax.fill_between(r, F_mono, 0, where=(F_mono > 0), alpha=0.15, color=COLORS["stable"])
ax.fill_between(r, F_mono, 0, where=(F_mono < 0), alpha=0.15, color=COLORS["unstable"])

ax.annotate(r"$F>0$: $\dot r>0$, rate rises", xy=(3, 2.5), fontsize=9,
            color=COLORS["stable"], fontstyle="italic")
ax.annotate(r"$F<0$: $\dot r<0$, rate falls", xy=(22, -3.5), fontsize=9,
            color=COLORS["unstable"], fontstyle="italic")

ax.set_xlabel("PYR firing rate $r$ (Hz)")
ax.set_ylabel(r"$F(r) = \Phi(I_\mathrm{net}(r)) - r$  (Hz)")
ax.set_title("Monostable — one fixed point")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 80)

# ── Right panel: real bistable from best_params ───────────────────────────────
ax = axes[1]
ax.plot(r_sweep, F, color="#333333", lw=2.5, label=r"$F(r)$  (bistable_high_fr)")
ax.axhline(0, color=COLORS["zero"], lw=1.2, ls="--", alpha=0.7)

for rfp in stable_fps:
    ax.plot(rfp, 0, "o", color=COLORS["stable"], ms=12, zorder=5)
for rfp in unstable_fps:
    ax.plot(rfp, 0, "s", color=COLORS["unstable"], ms=10, zorder=5,
            markerfacecolor="white", markeredgewidth=2)

ax.annotate("Low stable FP\n(resting state)", xy=(r_low, 0), xytext=(r_low + 6, 8),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9, ha="left")
ax.annotate("Unstable FP\n(threshold)", xy=(r_uns, 0), xytext=(r_uns + 5, -12),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9, ha="left")
ax.annotate("High stable FP\n(active state)", xy=(r_high, 0), xytext=(r_high - 20, 10),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9, ha="left")

legend_handles = [
    mpatches.Patch(color=COLORS["stable"], label="Stable FP (●)"),
    mpatches.Patch(facecolor="white", edgecolor=COLORS["unstable"], linewidth=2, label="Unstable FP (□)"),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_xlabel("PYR firing rate $r$ (Hz)")
ax.set_title("Bistable — two stable fixed points")
ax.set_xlim(0, 80)

for ax in axes:
    ax.set_ylim(-25, 30)

fig.suptitle("PYR Nullcline Shape: $F(r) = \\Phi(I_\\mathrm{net}(r)) - r$\n"
             "Fixed points where $F = 0$;  stable when slope $F'<0$", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig1_concept_nullcline.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 2  —  L_bistab: zone penalties visualized
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig2_L_bistab.png …")

r_low_probe  = 8.0    # BistableConfig default
r_mid_probe  = 15.0
r_high_probe = 30.0

fig, ax = plt.subplots(figsize=(10, 5.5))

# Shaded zones
ax.axvspan(0,             r_low_probe,  alpha=0.18, color=COLORS["zone1"], label="Zone 1: F ≥ 0  (mean penalty)")
ax.axvspan(r_low_probe,  r_high_probe, alpha=0.18, color=COLORS["zone2"], label="Zone 2: must dip F < 0  (max penalty)")
ax.axvspan(r_high_probe, 80,           alpha=0.18, color=COLORS["zone3"], label="Zone 3: F ≥ 0  (mean penalty)")

ax.plot(r_sweep, F, color="#333333", lw=2.5, zorder=3)
ax.axhline(0, color=COLORS["zero"], lw=1.2, ls="--", alpha=0.8)

# Probe points
for r_probe, label, sign_ok in [
    (r_low_probe,  f"r_low = {r_low_probe} Hz\nF > 0 required", True),
    (r_mid_probe,  f"r_mid = {r_mid_probe} Hz\nF < 0 required", False),
    (r_high_probe, f"r_high = {r_high_probe} Hz\nF > 0 required", True),
]:
    F_at = float(np.interp(r_probe, r_sweep, F))
    color = "#27AE60" if (sign_ok and F_at > 0) or (not sign_ok and F_at < 0) else "#E74C3C"
    ax.plot(r_probe, F_at, "D", color=color, ms=10, zorder=6)
    ax.annotate(label, xy=(r_probe, F_at), xytext=(r_probe + 2, F_at + (8 if F_at > 0 else -11)),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2))

# Actual fixed points
for rfp in stable_fps:
    ax.plot(rfp, 0, "o", color=COLORS["stable"], ms=11, zorder=5)
for rfp in unstable_fps:
    ax.plot(rfp, 0, "s", color=COLORS["unstable"], ms=9, zorder=5,
            markerfacecolor="white", markeredgewidth=2)

ax.set_xlabel("PYR firing rate $r$ (Hz)")
ax.set_ylabel(r"$F(r)$  (Hz)")
ax.set_title(r"$L_\mathrm{bistab}$: Sign-Pattern Enforcement"
             "\nThree probe-point checks + mean zone penalties", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 80)
ax.set_ylim(-22, 25)

# Add formula annotation
formula = ("3-point check:\n"
           r"  relu($-F_{low}$) + relu($F_{mid}$) + relu($-F_{high}$)" + "\n"
           "Zone penalties:\n"
           r"  mean(relu($-F$)) per zone")
ax.text(42, 18, formula, fontsize=9.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

plt.tight_layout()
plt.savefig(OUT / "fig2_L_bistab.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 3  —  L_rate: low-FP rate matching
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig3_L_rate_low.png …")

pops = ["PYR", "SOM", "PV", "VIP"]
actual_low = [r_low, r_som_low, r_pv_low, r_vip_low]
target_low = [LOW_TARGETS[p] for p in pops]
pop_colors = [COLORS["pyr"], COLORS["som"], COLORS["pv"], COLORS["vip"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Left: absolute rates
ax = axes[0]
x = np.arange(len(pops))
w = 0.35
bars_t = ax.bar(x - w/2, target_low, w, label="Target (Rooy 2021)", color=[c + "99" for c in pop_colors],
                edgecolor=pop_colors, linewidth=2)
bars_a = ax.bar(x + w/2, actual_low, w, label="Actual (low FP)", color=pop_colors, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(pops, fontsize=12)
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("Low FP — Absolute Rates")
ax.legend(fontsize=9)

for bar, val in zip(bars_a, actual_low):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
for bar, val in zip(bars_t, target_low):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color="#555")

# Right: MSPE contributions
ax = axes[1]
mspe_terms = [((a - t) / t) ** 2 for a, t in zip(actual_low, target_low)]
bars = ax.bar(x, mspe_terms, color=pop_colors, alpha=0.85, edgecolor="white", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(pops, fontsize=12)
ax.set_ylabel(r"$\left(\frac{r_\mathrm{actual} - r_\mathrm{target}}{r_\mathrm{target}}\right)^2$")
ax.set_title(r"$L_\mathrm{rate}$ Contributions per Population"
             f"\nTotal $L_{{\\mathrm{{rate}}}}$ = {sum(mspe_terms):.4f}")

pct_errors = [100 * (a - t) / t for a, t in zip(actual_low, target_low)]
for bar, val, pct in zip(bars, mspe_terms, pct_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{pct:+.0f}%", ha="center", va="bottom", fontsize=9, color="#333")

fig.suptitle(r"$L_\mathrm{rate}$ — Low Fixed-Point Rate Matching (MSPE)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig3_L_rate_low.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 4  —  L_margin: separation between fixed points
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig4_L_margin.png …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

delta_r_min = 15.0
separation = r_high - r_low

for ax, scenario, r_lo, r_hi, r_un, title in [
    (axes[0], "bad",  r_low,  r_low + 6,  r_low + 3,
     f"Barely bistable: Δr={6:.0f} Hz < {delta_r_min:.0f} Hz\n"
     f"→ $L_{{\\mathrm{{margin}}}}$ = relu(15 − 6) = 9  ← penalised"),
    (axes[1], "good", r_low, r_high, r_uns,
     f"Well separated: Δr={separation:.0f} Hz > {delta_r_min:.0f} Hz\n"
     f"→ $L_{{\\mathrm{{margin}}}}$ = relu(15 − {separation:.0f}) = 0  ← OK"),
]:
    if scenario == "bad":
        r_arr = np.linspace(0, 20, 500)
        # synthetic barely-bistable nullcline
        F_s = 0.5 * np.sin(np.pi * r_arr / 9) * np.exp(-r_arr / 15) * 12
        F_s = F_s - F_s[0] * 0.2
    else:
        r_arr = r_sweep
        F_s = F

    ax.plot(r_arr, F_s, color="#333333", lw=2.5)
    ax.axhline(0, color=COLORS["zero"], lw=1.2, ls="--", alpha=0.7)

    if scenario == "bad":
        s_fps, _ = find_fixed_points(r_arr, F_s)
        if len(s_fps) >= 2:
            lo, hi = s_fps[0], s_fps[-1]
        else:
            lo, hi = r_lo, r_hi
        ax.plot(lo, 0, "o", color=COLORS["stable"], ms=12, zorder=5)
        ax.plot(hi, 0, "o", color=COLORS["stable"], ms=12, zorder=5)
        ax.annotate("", xy=(hi, -6), xytext=(lo, -6),
                    arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=2))
        ax.text((lo + hi) / 2, -8.5, f"Δr = {hi-lo:.0f} Hz", ha="center", fontsize=10, color="#E74C3C")
        ax.set_xlim(-1, 22)
        ax.set_ylim(-15, 15)
    else:
        ax.plot(r_low, 0, "o", color=COLORS["stable"], ms=12, zorder=5)
        ax.plot(r_high, 0, "o", color=COLORS["stable"], ms=12, zorder=5)
        ax.annotate("", xy=(r_high, -8), xytext=(r_low, -8),
                    arrowprops=dict(arrowstyle="<->", color="#27AE60", lw=2))
        ax.text((r_low + r_high) / 2, -11, f"Δr = {separation:.0f} Hz", ha="center",
                fontsize=10, color="#27AE60")
        ax.axvline(r_low + delta_r_min, color="#27AE60", ls=":", lw=1.5, alpha=0.5)
        ax.set_xlim(-2, 82)
        ax.set_ylim(-18, 20)

    ax.set_xlabel("PYR firing rate $r$ (Hz)")
    ax.set_ylabel(r"$F(r)$  (Hz)")
    ax.set_title(title, fontsize=10)

fig.suptitle(r"$L_\mathrm{margin} = \mathrm{relu}(\Delta r_\mathrm{min} - (r_\mathrm{high} - r_\mathrm{low}))$"
             "\nMinimum separation enforced: 15 Hz", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig4_L_margin.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 5  —  L_physiol: interneuron sweep ceilings
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig5_L_physiol.png …")

som_ceil = 50.0
pv_ceil  = 100.0
vip_ceil = 80.0

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, r_inh, ceil, name, color, label in [
    (axes[0], r_som_sw, som_ceil, "SOM", COLORS["som"], f"SOM ceiling = {som_ceil:.0f} Hz"),
    (axes[1], r_pv_sw,  pv_ceil,  "PV",  COLORS["pv"],  f"PV ceiling  = {pv_ceil:.0f} Hz"),
    (axes[2], r_vip_sw, vip_ceil, "VIP", COLORS["vip"], f"VIP ceiling = {vip_ceil:.0f} Hz"),
]:
    above = r_inh > ceil
    ax.plot(r_sweep, r_inh, color=color, lw=2.2)
    ax.axhline(ceil, color="#E74C3C", lw=2, ls="--", alpha=0.85, label=label)
    ax.fill_between(r_sweep, r_inh, ceil, where=above, color="#E74C3C", alpha=0.25, label="Penalised region")
    ax.set_xlabel("PYR rate $r$ (Hz)")
    ax.set_ylabel(f"{name} rate (Hz)")
    ax.set_title(f"{name} interneuron across sweep")
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 80)

    # Mark low and high FPs
    ax.axvline(r_low, color="#999", ls=":", lw=1.5, alpha=0.7)
    ax.axvline(r_high, color="#999", ls=":", lw=1.5, alpha=0.7, label="FPs")
    ax.text(r_low + 0.5, ax.get_ylim()[1] * 0.95, "low\nFP", fontsize=7.5, color="#666", va="top")
    ax.text(r_high + 0.5, ax.get_ylim()[1] * 0.95, "high\nFP", fontsize=7.5, color="#666", va="top")

fig.suptitle(r"$L_\mathrm{physiol}$: Interneuron Ceiling Across Full Sweep"
             "\nPenalises supraphysiological rates at any $r_\\mathrm{PYR}$, not just at fixed points",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig5_L_physiol.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 6  —  L_jac: Jacobian heatmap
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig6_L_jac.png …")

r_ss = np.array([r_low, r_som_low, r_pv_low, r_vip_low])
J = compute_jacobian(params, r_ss)
pop_labels = ["PYR", "SOM", "PV", "VIP"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Heatmap
ax = axes[0]
abs_J = np.abs(J)
vmax = max(5.0, abs_J.max())
im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([f"from {p}" for p in pop_labels], fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels([f"to {p}" for p in pop_labels], fontsize=9)
ax.set_title(r"Jacobian $J_{ij}$ at low fixed point")
plt.colorbar(im, ax=ax, label=r"$J_{ij}$ (Hz/Hz)")

for i in range(4):
    for j in range(4):
        c = "white" if abs(J[i, j]) > 0.4 * vmax else "black"
        ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center", color=c, fontsize=10, fontweight="bold")

# Bar: max |J| by row (target population)
ax = axes[1]
max_abs_per_row = [np.max(np.abs(J[i, :])) for i in range(4)]
colors_bar = [COLORS["pyr"], COLORS["som"], COLORS["pv"], COLORS["vip"]]
bars = ax.bar(pop_labels, max_abs_per_row, color=colors_bar, alpha=0.85, edgecolor="white", lw=1.5)
ax.axhline(5.0, color="#E74C3C", lw=2, ls="--", label="Threshold = 5.0")

for bar, val in zip(bars, max_abs_per_row):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)

max_J = float(np.max(abs_J))
L_jac = max(0, max_J - 5.0) ** 2
ax.set_ylabel(r"$\max_j |J_{ij}|$  per target population")
ax.set_title(f"Max |J| per row  [global max = {max_J:.2f}]\n"
             r"$L_\mathrm{jac} = \mathrm{relu}(\max|J|-5)^2$" + f" = {L_jac:.3f}")
ax.legend(fontsize=9)

fig.suptitle(r"$L_\mathrm{jac}$: Jacobian Regularizer"
             "\nPrevents any connection from dominating overwhelmingly", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig6_L_jac.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 7  —  L_peak: nullcline overshoot
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig7_L_peak.png …")

# Load low-peak params too if available
try:
    with open(LOW_PEAK_PATH) as f:
        raw_lp = json.load(f)
    params_lp = CircuitParams(**{k: v for k, v in raw_lp.items() if k in CircuitParams.__dataclass_fields__})
    r_sw_lp, F_lp, phi_lp = compute_nullcline(params_lp)
    stable_lp, _ = find_fixed_points(r_sw_lp, F_lp)
    r_high_lp = stable_lp[-1] if len(stable_lp) > 1 else 66.0
    has_lp = True
except Exception as e:
    has_lp = False
    print(f"  (low-peak params not loaded: {e})")

fig, ax = plt.subplots(figsize=(10, 5.5))

peak_idx = np.argmax(phi)
peak_r   = r_sweep[peak_idx]
peak_val = phi[peak_idx]

ax.plot(r_sweep, phi, color="#333333", lw=2.5, label="bistable_high_fr  (no L_peak)")
ax.plot(r_sweep, r_sweep, color="#aaaaaa", lw=1.5, ls="--", label="Identity line $\\Phi = r$")

if has_lp:
    peak_lp   = np.max(phi_lp)
    ax.plot(r_sw_lp, phi_lp, color="#5588CC", lw=2.2, ls="-", alpha=0.8,
            label=f"bistable_high_fr_low_peak  (L_peak active, peak={peak_lp:.0f} Hz)")
    ax.annotate("", xy=(r_sw_lp[np.argmax(phi_lp)], peak_lp),
                xytext=(r_sw_lp[np.argmax(phi_lp)] + 3, peak_lp + 10),
                arrowprops=dict(arrowstyle="->", color="#5588CC", lw=2))

# Annotate peak on main curve
ax.annotate(f"Nullcline peak\n{peak_val:.0f} Hz",
            xy=(peak_r, peak_val), xytext=(peak_r - 18, peak_val + 12),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
            fontsize=10, color="#E74C3C")

# Annotate high FP
ax.plot(r_high, r_high, "o", color=COLORS["stable"], ms=12, zorder=5)
ax.annotate(f"High FP\n{r_high:.0f} Hz",
            xy=(r_high, r_high), xytext=(r_high - 22, r_high - 20),
            arrowprops=dict(arrowstyle="->", color=COLORS["stable"], lw=2),
            fontsize=10, color=COLORS["stable"])

# Overshoot arrow
ax.annotate("", xy=(peak_r, peak_val), xytext=(peak_r, r_high),
            arrowprops=dict(arrowstyle="<->", color="#888", lw=2))
ax.text(peak_r + 2, (peak_val + r_high) / 2,
        f"Overshoot\n{peak_val-r_high:.0f} Hz", fontsize=9.5, color="#555", va="center")

ax.set_xlabel("PYR firing rate $r$ (Hz)")
ax.set_ylabel(r"Transfer output $\Phi(I_\mathrm{net}(r))$  (Hz)")
ax.set_title(r"$L_\mathrm{peak}$: Nullcline Peak Penalty"
             "\nLarge gap between peak and high FP → overshoot → saturation at 200 Hz during transitions",
             fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 80)
ax.set_ylim(0, 160)

plt.tight_layout()
plt.savefig(OUT / "fig7_L_peak.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 8  —  Missing high-state: current vs targets
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig8_missing_high_state.png …")

pops = ["PYR", "SOM", "PV", "VIP"]
actual_high = [r_high, r_som_high, r_pv_high, r_vip_high]
target_high = [HIGH_TARGETS[p] for p in pops]
pop_colors  = [COLORS["pyr"], COLORS["som"], COLORS["pv"], COLORS["vip"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: absolute comparison
ax = axes[0]
x = np.arange(len(pops))
w = 0.35
bars_t = ax.bar(x - w/2, target_high, w, label="Target (Rooy 2021 H-state)",
                color=[c + "88" for c in pop_colors], edgecolor=pop_colors, linewidth=2)
bars_a = ax.bar(x + w/2, actual_high, w, label="Actual (high FP, bistable_high_fr)",
                color=pop_colors, alpha=0.9)

for bar, val in zip(bars_t, target_high):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#555")
for bar, val in zip(bars_a, actual_high):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# Annotate the big misses
for i, (a, t, p) in enumerate(zip(actual_high, target_high, pops)):
    if abs(a - t) > 5:
        ax.annotate(f"[!] {100*(a-t)/t:+.0f}%",
                    xy=(x[i] + w/2, max(a, t) + 1),
                    xytext=(x[i] + w/2, max(a, t) + 12),
                    ha="center", fontsize=9, color="#E74C3C",
                    arrowprops=dict(arrowstyle="-", color="#E74C3C", lw=1.2))

ax.set_xticks(x); ax.set_xticklabels(pops, fontsize=12)
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("High FP — Current vs Target")
ax.legend(fontsize=9)
ax.set_ylim(0, 100)

# Right: "what's missing" — proposed L_rate_high MSPE terms
ax = axes[1]
mspe_high = [((a - t) / t) ** 2 for a, t in zip(actual_high, target_high)]
colors_miss = ["#27AE60" if v < 0.1 else "#E74C3C" for v in mspe_high]
bars2 = ax.bar(x, mspe_high, color=colors_miss, alpha=0.85, edgecolor="white", lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(pops, fontsize=12)
ax.set_ylabel(r"$\left(\frac{r_\mathrm{actual}-r_\mathrm{target}}{r_\mathrm{target}}\right)^2$")
ax.set_title(r"$L_\mathrm{rate,high}$ if we added it now"
             f"\n(what the optimizer is ignoring)",
             fontsize=10)

for bar, val, a, t in zip(bars2, mspe_high, actual_high, target_high):
    pct = 100 * (a - t) / t
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{pct:+.0f}%", ha="center", va="bottom", fontsize=9)

fig.suptitle("The Missing Piece: No Loss Constrains the High Fixed-Point Rates\n"
             "SOM and PV are completely wrong in the active state (target: ~35 Hz, actual: ~0 and ~4 Hz)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig8_missing_high_state.png")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# FIG 9  —  Proposed loss overview: before vs after
# ════════════════════════════════════════════════════════════════════════════════
print("Generating fig9_proposed_loss_overview.png …")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Color scheme
c_keep    = "#27AE60"
c_remove  = "#E74C3C"
c_add     = "#2980B9"
c_simplify = "#E67E22"

rows = [
    # (term,                 weight_old,  weight_new,  action,     color,       note)
    ("L_bistab (7 sub-terms)",  "w=1.0",  "w=2.0 (2 sub-terms)",  "Simplify",   c_simplify, "3-pt check + zones → 2-pt check"),
    ("L_ceiling",               "w=1.0*", "w=0.5",                "Keep",       c_keep,     "Bundled with L_bistab → decoupled"),
    ("L_rate_low",              "w=1.0",  "w=1.0",                "Keep",       c_keep,     "No change"),
    ("L_rate_high",             "—",      "w=1.5",                "NEW",        c_add,      "Add high-FP targets (Rooy 2021)"),
    ("L_margin",                "w=0.5",  "w=0.5",                "Keep",       c_keep,     "No change"),
    ("L_physiol",               "w=1.0",  "—",                    "Remove",     c_remove,   "Replaced by L_rate_high"),
    ("L_jac",                   "w=0.1",  "w=0.1",                "Keep",       c_keep,     "No change"),
    ("L_peak",                  "off",    "off (optional)",        "Keep",       c_keep,     "Enable with --w_peak 1.0"),
]

headers = ["Loss term", "Current weight", "Proposed weight", "Action", "Note"]
col_x = [0.1, 2.4, 4.1, 5.75, 6.75]
row_h = 0.85
y0 = 9.3

# Header
for hdr, cx in zip(headers, col_x):
    ax.text(cx, y0, hdr, fontsize=9.5, fontweight="bold", color="#222", va="top")

ax.axhline(y0 - 0.3, xmin=0.01, xmax=0.99, color="#333", lw=1.5)

for i, (term, w_old, w_new, action, color, note) in enumerate(rows):
    y = y0 - 0.45 - (i + 1) * row_h

    # Background highlight
    bg = "#f9f9f9" if i % 2 == 0 else "white"
    ax.barh(y + row_h/2 - 0.1, 9.8, height=row_h * 0.85, left=0.1, color=bg, zorder=0)

    ax.text(col_x[0], y + 0.25, term, fontsize=9, va="center", color="#222")
    ax.text(col_x[1], y + 0.25, w_old, fontsize=9, va="center", color="#666")
    ax.text(col_x[2], y + 0.25, w_new, fontsize=9, va="center", color=color, fontweight="bold")
    # Action badge
    badge_bg = {"Keep": "#EAF5EA", "Remove": "#FDECEA", "NEW": "#EBF5FB", "Simplify": "#FEF9E7"}
    ax.text(col_x[3], y + 0.25, action, fontsize=9, va="center", ha="left",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_bg.get(action, "white"),
                      edgecolor=color, linewidth=1))
    ax.text(col_x[4], y + 0.25, note, fontsize=8.5, va="center", color="#555",
            style="italic")

ax.set_title("Proposed Loss Redesign: Current vs After", fontsize=13, y=1.0, pad=8)

# Legend
leg_items = [
    (c_keep,    "Keep"),
    (c_simplify, "Simplify"),
    (c_add,     "New"),
    (c_remove,  "Remove"),
]
patches = [mpatches.Patch(color=c, label=l) for c, l in leg_items]
ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT / "fig9_proposed_loss_overview.png")
plt.close()


print(f"\n✓  All figures saved to {OUT}/")
print("  fig1_concept_nullcline.png")
print("  fig2_L_bistab.png")
print("  fig3_L_rate_low.png")
print("  fig4_L_margin.png")
print("  fig5_L_physiol.png")
print("  fig6_L_jac.png")
print("  fig7_L_peak.png")
print("  fig8_missing_high_state.png")
print("  fig9_proposed_loss_overview.png")
