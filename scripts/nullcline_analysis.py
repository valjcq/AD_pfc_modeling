#!/usr/bin/env python3
"""
Single-node nullcline analysis for CircuitParams.

Loads a fitted CircuitParams JSON and produces a nullcline plot for PYR,
determining whether the network is monostable or bistable.

Usage:
    python nullcline_analysis.py --params_json WT_1mo_article_ko.json
    python nullcline_analysis.py --params_json WT_1mo_article_ko.json --condition alpha7KO
    python nullcline_analysis.py --params_json WT_1mo_article_ko.json --all_conditions
"""

import argparse
import json
import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.constants import R_MAX_PHYS


class SingleNodeCircuit:
    """Single-node circuit with nullcline analysis."""

    def __init__(self, params_dict):
        """Initialize from a CircuitParams dictionary."""
        # Synaptic weights
        # J_NMDA (NMDA recurrent, replaces w_ee); fallback to w_ee for old JSON compatibility
        self.J_NMDA = params_dict.get('J_NMDA', params_dict.get('w_ee', 0.0))
        self.w_ep = params_dict.get('w_ep', 0.0)
        self.w_pe = params_dict.get('w_pe', 0.0)
        self.w_pp = params_dict.get('w_pp', 0.0)
        self.w_es = params_dict.get('w_es', 0.0)
        self.w_se = params_dict.get('w_se', 0.0)
        self.w_vs = params_dict.get('w_vs', 0.0)
        self.w_sp = params_dict.get('w_sp', 0.0)
        self.w_vp = params_dict.get('w_vp', 0.0)
        self.w_ev = params_dict.get('w_ev', 0.0)

        # External inputs
        self.I0_pyr = params_dict.get('I0_pyr', 0.0)
        self.I0_pv = params_dict.get('I0_pv', 0.0)
        self.I0_som = params_dict.get('I0_som', 0.0)
        self.I0_vip = params_dict.get('I0_vip', 0.0)

        # Transfer function parameters
        self.Theta_pyr = params_dict.get('Theta_pyr', 0.0)
        self.Theta_pv = params_dict.get('Theta_pv', 0.0)
        self.Theta_som = params_dict.get('Theta_som', 0.0)
        self.Theta_vip = params_dict.get('Theta_vip', 0.0)

        self.alpha_pyr = params_dict.get('alpha_pyr', 0.0)
        self.alpha_pv = params_dict.get('alpha_pv', 0.0)
        self.alpha_som = params_dict.get('alpha_som', 0.0)
        self.alpha_vip = params_dict.get('alpha_vip', 0.0)

        # Transfer function curvature (try both naming conventions)
        self.g_e = params_dict.get('g_e', params_dict.get('g_exc', 0.16))
        self.g_i = params_dict.get('g_i', params_dict.get('g_inh', 0.087))

        # GABA modulation (try both naming conventions)
        self.g_GABA_base = params_dict.get('g_GABA_base', params_dict.get('g_gaba_base', 0.0))
        self.g_alpha7 = params_dict.get('g_alpha7', 0.0)

        # Adaptation
        self.J_adapt_pyr = params_dict.get('J_adapt_pyr', 0.0)
        self.J_adapt_som = params_dict.get('J_adapt_som', 0.0)

        # Condition flags (can be overridden by set_condition)
        self.act_alpha7 = params_dict.get('act_alpha7', 1.0)
        self.act_beta2 = params_dict.get('act_beta2', 1.0)
        self.act_alpha5 = params_dict.get('act_alpha5', 1.0)

        # nAChR currents (optional, zero if missing)
        self.I_alpha7_pv = params_dict.get('I_alpha7_pv', 0.0)
        self.I_alpha7_som = params_dict.get('I_alpha7_som', 0.0)
        self.I_beta2_som = params_dict.get('I_beta2_som', 0.0)
        self.I_alpha5_vip = params_dict.get('I_alpha5_vip', 0.0)

    def set_condition(self, condition):
        """Set act_* flags based on condition (WT, alpha7KO, beta2KO, alpha5KO)."""
        self.act_alpha7 = 1.0
        self.act_beta2 = 1.0
        self.act_alpha5 = 1.0

        if condition == 'alpha7KO':
            self.act_alpha7 = 0.0
        elif condition == 'beta2KO':
            self.act_beta2 = 0.0
        elif condition == 'alpha5KO':
            self.act_alpha5 = 0.0

    def transfer_function(self, I, Theta, alpha, g):
        """Wong-Wang transfer function with numerically stable overflow guards."""
        u = alpha * (I - Theta)

        if np.isscalar(u):
            gu = g * u
            if gu > 500:
                # Linear regime: exp(-gu) ≈ 0, denominator ≈ 1
                phi = u
            elif np.abs(gu) < 1e-6:
                # Taylor expansion for small gu
                phi = 1.0 / g + u / 2.0
            elif gu < -500:
                # Saturation regime: exp(-gu) ≈ ∞, denominator ≈ -∞, phi → 0
                phi = 0.0
            else:
                # Safe middle ground (no overflow risk)
                phi = u / (1.0 - np.exp(-gu))
        else:
            phi = np.zeros_like(u)
            gu = g * u

            # Linear regime (large positive gu)
            large_pos = gu > 500
            phi[large_pos] = u[large_pos]

            # Saturation regime (large negative gu)
            large_neg = gu < -500
            phi[large_neg] = 0.0

            # Small gu (use Taylor expansion)
            small_gu = (np.abs(gu) < 1e-6) & ~large_pos & ~large_neg
            phi[small_gu] = 1.0 / g + u[small_gu] / 2.0

            # Regular regime (safe to compute)
            regular = ~large_pos & ~large_neg & ~small_gu
            phi[regular] = u[regular] / (1.0 - np.exp(-gu[regular]))

        # Clamp to [0, 200] Hz
        phi = np.clip(phi, 0.0, 200.0)
        return phi

    def get_g_GABA(self):
        """Compute effective g_GABA based on condition."""
        return self.g_GABA_base + self.act_alpha7 * self.g_alpha7

    def interneuron_steady_state(self, r_pyr):
        """
        Solve for steady-state interneuron rates given r_PYR.
        Returns (r_som, r_pv, r_vip).
        """
        # VIP input with nAChR modulation
        I_vip_nach = self.act_alpha5 * self.I_alpha5_vip
        I_vip = self.w_ev * r_pyr + self.I0_vip + I_vip_nach
        r_vip = self.transfer_function(I_vip, self.Theta_vip, self.alpha_vip, self.g_e)

        # Define the 2D system for SOM and PV
        def equations(x):
            r_som, r_pv = x

            # SOM input with nAChR modulation
            I_som_nach = self.act_alpha7 * self.I_alpha7_som + self.act_beta2 * self.I_beta2_som
            I_som = (self.w_es * r_pyr - self.w_vs * r_vip
                     - self.J_adapt_som * r_som + self.I0_som + I_som_nach)
            f_som = self.transfer_function(I_som, self.Theta_som, self.alpha_som, self.g_i) - r_som

            # PV input with nAChR modulation
            I_pv_nach = self.act_alpha7 * self.I_alpha7_pv
            g_GABA = self.get_g_GABA()
            I_pv = (self.w_ep * r_pyr - g_GABA * self.w_pp * r_pv
                    - g_GABA * self.w_sp * r_som - self.w_vp * r_vip + self.I0_pv + I_pv_nach)
            f_pv = self.transfer_function(I_pv, self.Theta_pv, self.alpha_pv, self.g_i) - r_pv

            return [f_som, f_pv]

        # Solve with initial guess [0, 0] (suppress convergence warning at high rates)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*not making good progress.*')
            sol = fsolve(equations, [0.0, 0.0], full_output=False)
        r_som, r_pv = sol

        return r_som, r_pv, r_vip

    def pyr_input_current(self, r_pyr, r_som, r_pv):
        """Compute net PYR input current (DIVISIVE inhibition from PV) with NMDA gating."""
        g_GABA = self.get_g_GABA()
        # NMDA gating: use steady-state formula S* for fixed-point analysis
        TAU_NMDA_MS, GAMMA_NMDA = 100.0, 0.641
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
        I_pyr = (self.J_NMDA * S_star / (1.0 + g_GABA * self.w_pe * r_pv)
                 - g_GABA * self.w_se * r_som
                 - self.J_adapt_pyr * r_pyr
                 + self.I0_pyr)
        return I_pyr

    def compute_nullcline(self, r_pyr_sweep):
        """
        Compute PYR nullcline and interneuron responses.

        Returns:
            r_pyr_sweep: input sweep (Hz)
            phi_pyr: Phi_PYR(I_net(r_PYR)) for each r_pyr
            r_som_sweep: steady-state r_SOM for each r_pyr
            r_pv_sweep: steady-state r_PV for each r_pyr
            r_vip_sweep: steady-state r_VIP for each r_pyr
            F: nullcline function F(r_pyr) = Phi_PYR(I_net) - r_pyr
        """
        phi_pyr = np.zeros_like(r_pyr_sweep)
        r_som_sweep = np.zeros_like(r_pyr_sweep)
        r_pv_sweep = np.zeros_like(r_pyr_sweep)
        r_vip_sweep = np.zeros_like(r_pyr_sweep)

        for i, r_pyr in enumerate(r_pyr_sweep):
            r_som, r_pv, r_vip = self.interneuron_steady_state(r_pyr)
            r_som_sweep[i] = r_som
            r_pv_sweep[i] = r_pv
            r_vip_sweep[i] = r_vip

            I_pyr = self.pyr_input_current(r_pyr, r_som, r_pv)
            phi_pyr[i] = self.transfer_function(I_pyr, self.Theta_pyr, self.alpha_pyr, self.g_e)

        F = phi_pyr - r_pyr_sweep

        return r_pyr_sweep, phi_pyr, r_som_sweep, r_pv_sweep, r_vip_sweep, F

    def find_fixed_points(self, r_pyr_sweep, F):
        """
        Find fixed points by detecting sign changes in F.
        Classify stability using np.gradient(F, r_sweep) at crossing points.
        Filter out any crossings above R_MAX_PHYS (clamp artifacts).

        Returns:
            fixed_points: list of (r_pyr, stability) tuples below R_MAX_PHYS
            spurious_count: number of crossings above R_MAX_PHYS that were discarded
        """
        fixed_points = []
        spurious_count = 0

        # Compute gradient of F across the sweep (smoother than finite differences)
        dF_dr_sweep = np.gradient(F, r_pyr_sweep)

        # Detect sign changes
        sign_changes = np.where(np.diff(np.sign(F)))[0]

        for idx in sign_changes:
            # Refine crossing location with brentq
            r_pyr_min = r_pyr_sweep[idx]
            r_pyr_max = r_pyr_sweep[idx + 1]

            try:
                def f_to_root(r):
                    r_som, r_pv, r_vip = self.interneuron_steady_state(r)
                    I_pyr = self.pyr_input_current(r, r_som, r_pv)
                    phi = self.transfer_function(I_pyr, self.Theta_pyr, self.alpha_pyr, self.g_e)
                    return phi - r

                r_fp = brentq(f_to_root, r_pyr_min, r_pyr_max)

                # Filter out clamp artifacts (crossings above R_MAX_PHYS)
                if r_fp >= R_MAX_PHYS:
                    spurious_count += 1
                    continue

                # Use gradient at the crossing index for stability (more robust than finite diff)
                dF_dr_at_crossing = dF_dr_sweep[idx]

                stability = 'stable' if dF_dr_at_crossing < 0 else 'unstable'
                fixed_points.append((r_fp, stability))
            except ValueError:
                # brentq failed (e.g., function doesn't change sign in interval)
                pass

        return fixed_points, spurious_count

    def compute_loop_gain(self, r_pyr_sweep, r_som_sweep, r_pv_sweep):
        """
        Compute loop gain decomposition.

        Returns:
            dPhi_dr: slope of transfer function at PYR operating point
            effective_gain_ee: J_NMDA * dS*/dr / (1 + g_GABA * w_pe * r_pv) [NMDA saturation]
            adaptation_reduction: J_adapt_pyr
            loop_gain: effective gain * dPhi/dr
        """
        TAU_NMDA_MS, GAMMA_NMDA = 100.0, 0.641

        dPhi_dr = np.zeros_like(r_pyr_sweep)
        effective_gain_ee = np.zeros_like(r_pyr_sweep)
        adaptation_reduction = self.J_adapt_pyr * np.ones_like(r_pyr_sweep)

        dr = 1e-4

        for i, r_pyr in enumerate(r_pyr_sweep):
            r_som = r_som_sweep[i]
            r_pv = r_pv_sweep[i]

            # Transfer function slope at operating point
            I_pyr_here = self.pyr_input_current(r_pyr, r_som, r_pv)
            I_pyr_plus = self.pyr_input_current(r_pyr + dr, r_som, r_pv)

            phi_here = self.transfer_function(I_pyr_here, self.Theta_pyr, self.alpha_pyr, self.g_e)
            phi_plus = self.transfer_function(I_pyr_plus, self.Theta_pyr, self.alpha_pyr, self.g_e)

            dPhi_dI = (phi_plus - phi_here) / dr

            # dI/dr_pyr at this operating point (accounts for NMDA saturation and divisive inhibition)
            g_GABA = self.get_g_GABA()
            # NMDA saturation derivative: dS*/dr
            dSdr = GAMMA_NMDA * TAU_NMDA_MS / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS) ** 2
            # Total effective gain with NMDA gating
            dI_dr = self.J_NMDA * dSdr / (1.0 + g_GABA * self.w_pe * r_pv)

            dPhi_dr[i] = dPhi_dI * dI_dr
            effective_gain_ee[i] = dI_dr

        loop_gain = dPhi_dr

        return dPhi_dr, effective_gain_ee, adaptation_reduction, loop_gain


def print_param_summary(params_dict):
    """Print a summary of key loaded parameters."""
    print("\nKey parameters:")
    # Use J_NMDA if present, else fall back to w_ee for old JSON compatibility
    J_NMDA_val = params_dict.get('J_NMDA', params_dict.get('w_ee', 0))
    print(f"  Recurrent excitation: J_NMDA={J_NMDA_val:.4f} nA (NMDA gating, tau=100ms, gamma=0.641)")
    print(f"  Feedback inhibition: w_pe={params_dict.get('w_pe', 0):.4f}, w_ep={params_dict.get('w_ep', 0):.4f}")
    print(f"  Inhibitory recurrence: w_pp={params_dict.get('w_pp', 0):.4f}, w_sp={params_dict.get('w_sp', 0):.4f}")
    print(f"  Disinhibition: w_vs={params_dict.get('w_vs', 0):.4f}, w_vp={params_dict.get('w_vp', 0):.4f}")
    print(f"  Adaptation: J_adapt_pyr={params_dict.get('J_adapt_pyr', 0):.4f}, "
          f"J_adapt_som={params_dict.get('J_adapt_som', 0):.4f}")
    # GABA modulation: try uppercase first, then lowercase (g_GABA_base vs g_gaba_base)
    g_gaba_base_val = params_dict.get('g_GABA_base', params_dict.get('g_gaba_base', 0.0))
    g_alpha7_val = params_dict.get('g_alpha7', 0.0)
    print(f"  GABA modulation: g_gaba_base={g_gaba_base_val:.4f}, "
          f"g_alpha7={g_alpha7_val:.4f}")
    print(f"  External inputs: I0_pyr={params_dict.get('I0_pyr', 0):.4f}, "
          f"I0_pv={params_dict.get('I0_pv', 0):.4f}, "
          f"I0_som={params_dict.get('I0_som', 0):.4f}, "
          f"I0_vip={params_dict.get('I0_vip', 0):.4f}")


def save_nullcline_json(circuit, r_pyr_sweep, phi_pyr, r_som, r_pv, r_vip, F,
                        fixed_points, condition, params_stem, params_dir=Path(".")):
    """Save nullcline evolution and fixed points to JSON."""
    # Classify regime
    stable_fps = [(r, s) for r, s in fixed_points if s == 'stable']
    unstable_fps = [(r, s) for r, s in fixed_points if s == 'unstable']
    n_stable = len(stable_fps)

    if n_stable >= 2:
        regime = 'BISTABLE'
    elif n_stable == 1:
        regime = 'MONOSTABLE'
    else:
        regime = 'NO STABLE FIXED POINTS'

    # Build fixed points list with interneuron rates at each FP
    fps_list = []
    for r_fp, stability in sorted(fixed_points):
        r_som_fp, r_pv_fp, r_vip_fp = circuit.interneuron_steady_state(r_fp)
        fps_list.append({
            "rate_hz": round(float(r_fp), 4),
            "stability": stability,
            "r_pv_hz": round(float(r_pv_fp), 4),
            "r_som_hz": round(float(r_som_fp), 4),
            "r_vip_hz": round(float(r_vip_fp), 4)
        })

    data = {
        "condition": condition,
        "regime": regime,
        "params_file": params_stem,
        "fixed_points": fps_list,
        "nullcline_sweep": {
            "r_pyr_hz": [round(float(x), 4) for x in r_pyr_sweep],
            "phi_pyr_hz": [round(float(x), 4) for x in phi_pyr],
            "r_pv_hz": [round(float(x), 4) for x in r_pv],
            "r_som_hz": [round(float(x), 4) for x in r_som],
            "r_vip_hz": [round(float(x), 4) for x in r_vip]
        }
    }

    out = params_dir / f'nullcline_{condition}_{params_stem}.json'
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {out}")


def plot_single_condition(circuit, args, params_stem, condition, params_dir=Path("."), no_show=False):
    """Plot 2-panel nullcline analysis for a single condition."""
    # Compute nullcline with extended sweep to [0, 250] Hz to capture all stable FPs
    r_pyr_sweep = np.linspace(0, 250, 1500)
    r_pyr_sweep, phi_pyr, r_som, r_pv, r_vip, F = circuit.compute_nullcline(r_pyr_sweep)

    # Find fixed points (excluding clamp artifacts above R_MAX_PHYS)
    fixed_points, spurious_count = circuit.find_fixed_points(r_pyr_sweep, F)

    # Classify regime: count stable fixed points only
    stable_fps = [(r, s) for r, s in fixed_points if s == 'stable']
    unstable_fps = [(r, s) for r, s in fixed_points if s == 'unstable']
    n_stable = len(stable_fps)

    if n_stable >= 2:
        regime = 'BISTABLE'
    elif n_stable == 1:
        regime = 'MONOSTABLE'
    else:
        regime = 'NO STABLE FIXED POINTS'

    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"Regime: {regime}")
    print(f"Fixed points found: {len(fixed_points)} total ({len(stable_fps)} stable, {len(unstable_fps)} unstable)")
    print(f"Crossings (sorted by rate, below R_MAX_PHYS={R_MAX_PHYS} Hz):")
    for r_fp, stab in sorted(fixed_points):
        print(f"  {r_fp:.2f} Hz — {stab}")
    if spurious_count > 0:
        print(f"Note: {spurious_count} crossing(s) above R_MAX_PHYS Hz ignored (clamp artifact)")
    print(f"{'='*60}\n")

    # Save JSON with nullcline evolution and fixed points
    save_nullcline_json(circuit, r_pyr_sweep, phi_pyr, r_som, r_pv, r_vip, F,
                        fixed_points, condition, params_stem, params_dir)

    # Single-panel figure: PYR nullcline only
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(r_pyr_sweep, phi_pyr, 'b-', linewidth=2.5, label='Phi_PYR(I_net)')
    ax.plot(r_pyr_sweep, r_pyr_sweep, 'k--', linewidth=1.5, alpha=0.5, label='Identity line')

    # Shade region above R_MAX_PHYS (clamp artifacts)
    ax.axvspan(R_MAX_PHYS, r_pyr_sweep.max(), alpha=0.15, color='gray', label='Clamp artifact region')
    ax.axvline(R_MAX_PHYS, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(R_MAX_PHYS + 5, r_pyr_sweep.max() * 0.92, f'R_MAX_PHYS\n({R_MAX_PHYS:.0f} Hz)',
            fontsize=9, color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

    # Mark stable fixed points (filled green)
    for r_fp, _ in stable_fps:
        r_som_fp, r_pv_fp, r_vip_fp = circuit.interneuron_steady_state(r_fp)
        I_pyr_fp = circuit.pyr_input_current(r_fp, r_som_fp, r_pv_fp)
        phi_fp = circuit.transfer_function(I_pyr_fp, circuit.Theta_pyr, circuit.alpha_pyr, circuit.g_e)
        ax.plot(r_fp, phi_fp, 'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
               label='Stable fixed point' if r_fp == stable_fps[0][0] else '')

    # Mark unstable fixed points (open red)
    for r_fp, _ in unstable_fps:
        r_som_fp, r_pv_fp, r_vip_fp = circuit.interneuron_steady_state(r_fp)
        I_pyr_fp = circuit.pyr_input_current(r_fp, r_som_fp, r_pv_fp)
        phi_fp = circuit.transfer_function(I_pyr_fp, circuit.Theta_pyr, circuit.alpha_pyr, circuit.g_e)
        ax.plot(r_fp, phi_fp, 'o', color='red', markersize=10, fillstyle='none', markeredgewidth=2.5,
               label='Unstable fixed point' if r_fp == unstable_fps[0][0] else '')

    # Set x-axis limit to extend slightly beyond the rightmost fixed point, at least 80 Hz
    if fixed_points:
        max_fp = max([r for r, _ in fixed_points])
        x_limit = max(max_fp * 1.1, 80)
    else:
        x_limit = 80
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, x_limit)
    ax.set_xlabel('r_PYR (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Phi_PYR(I_net) (Hz)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    ax.text(0.98, 0.97, regime, transform=ax.transAxes, fontsize=13,
           fontweight='bold', verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.5))

    fig.suptitle(f'PYR Nullcline — {condition} — {params_stem}', fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.90, left=0.12, right=0.97, bottom=0.12)
    out = params_dir / f'nullcline_{condition}_{params_stem}.png'
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_all_conditions(params_dict, params_stem, params_dir=Path("."), no_show=False):
    """Plot PYR nullclines for all 4 conditions on one figure and save JSON for each."""
    conditions = ['WT', 'alpha7KO', 'beta2KO', 'alpha5KO']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for cond_idx, cond in enumerate(conditions):
        circuit = SingleNodeCircuit(params_dict)
        circuit.set_condition(cond)

        # Compute nullcline with extended sweep to [0, 250] Hz
        r_pyr_sweep = np.linspace(0, 250, 1500)
        r_pyr_sweep, phi_pyr, r_som, r_pv, r_vip, F = circuit.compute_nullcline(r_pyr_sweep)

        # Find fixed points (excluding clamp artifacts above R_MAX_PHYS)
        fixed_points, spurious_count = circuit.find_fixed_points(r_pyr_sweep, F)

        # Classify regime: count stable fixed points only
        stable_fps = [(r, s) for r, s in fixed_points if s == 'stable']
        unstable_fps = [(r, s) for r, s in fixed_points if s == 'unstable']
        n_stable = len(stable_fps)

        if n_stable >= 2:
            regime = 'BISTABLE'
        elif n_stable == 1:
            regime = 'MONOSTABLE'
        else:
            regime = 'NO STABLE FIXED POINTS'

        # Save JSON for this condition
        save_nullcline_json(circuit, r_pyr_sweep, phi_pyr, r_som, r_pv, r_vip, F,
                           fixed_points, cond, params_stem, params_dir)

        # Plot on subplot
        ax = axes[cond_idx]
        ax.plot(r_pyr_sweep, phi_pyr, 'b-', linewidth=2.5, label='Phi_PYR(I_net)')
        ax.plot(r_pyr_sweep, r_pyr_sweep, 'k--', linewidth=1.5, alpha=0.5, label='Identity')

        # Shade region above R_MAX_PHYS (clamp artifacts)
        ax.axvspan(R_MAX_PHYS, r_pyr_sweep.max(), alpha=0.15, color='gray')
        ax.axvline(R_MAX_PHYS, color='red', linestyle='--', linewidth=1, alpha=0.6)

        # Mark stable fixed points
        for r_fp, _ in stable_fps:
            r_som_fp, r_pv_fp, r_vip_fp = circuit.interneuron_steady_state(r_fp)
            I_pyr_fp = circuit.pyr_input_current(r_fp, r_som_fp, r_pv_fp)
            phi_fp = circuit.transfer_function(I_pyr_fp, circuit.Theta_pyr, circuit.alpha_pyr, circuit.g_e)
            ax.plot(r_fp, phi_fp, 'go', markersize=11, markeredgecolor='darkgreen', markeredgewidth=2)

        # Mark unstable fixed points
        for r_fp, _ in unstable_fps:
            r_som_fp, r_pv_fp, r_vip_fp = circuit.interneuron_steady_state(r_fp)
            I_pyr_fp = circuit.pyr_input_current(r_fp, r_som_fp, r_pv_fp)
            phi_fp = circuit.transfer_function(I_pyr_fp, circuit.Theta_pyr, circuit.alpha_pyr, circuit.g_e)
            ax.plot(r_fp, phi_fp, 'o', color='red', markersize=9, fillstyle='none', markeredgewidth=2)

        # Set x-axis limit to extend slightly beyond the rightmost fixed point, at least 80 Hz
        if fixed_points:
            max_fp = max([r for r, _ in fixed_points])
            x_limit = max(max_fp * 1.1, 80)
        else:
            x_limit = 80
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, x_limit)
        ax.set_xlabel('r_PYR (Hz)', fontsize=10)
        ax.set_ylabel('Phi_PYR(I_net) (Hz)', fontsize=10)
        ax.set_title(f'{cond} — {regime}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.text(0.98, 0.97, regime, transform=ax.transAxes, fontsize=11,
               fontweight='bold', verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.5))

    fig.suptitle(f'PYR Nullcline — All Conditions — {params_stem}',
                fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.94, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    out = params_dir / f'nullcline_all_conditions_{params_stem}.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    if not no_show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Single-node nullcline analysis for CircuitParams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nullcline_analysis.py --params_json WT_1mo_article_ko.json
  python nullcline_analysis.py --params_json WT_1mo_article_ko.json --condition alpha7KO
  python nullcline_analysis.py --params_json WT_1mo_article_ko.json --all_conditions
        """
    )
    parser.add_argument('--params_json', required=True, help='Path to CircuitParams JSON file')
    parser.add_argument('--condition', default='WT',
                        choices=['WT', 'alpha7KO', 'beta2KO', 'alpha5KO'],
                        help='Condition to analyze (default: WT)')
    parser.add_argument('--all_conditions', action='store_true',
                        help='Plot all 4 conditions on one figure')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not call plt.show() (for batch runs)')
    args = parser.parse_args()

    # Load JSON
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"Error: File not found: {args.params_json}")
        return

    with open(params_path, 'r') as f:
        params_dict = json.load(f)

    params_stem = params_path.stem
    params_dir = params_path.parent

    print(f"{'='*60}")
    print(f"Loaded: {args.params_json}")
    print_param_summary(params_dict)
    print(f"{'='*60}")

    if args.all_conditions:
        plot_all_conditions(params_dict, params_stem, params_dir, no_show=args.no_show)
    else:
        circuit = SingleNodeCircuit(params_dict)
        circuit.set_condition(args.condition)
        plot_single_condition(circuit, args, params_stem, args.condition, params_dir, no_show=args.no_show)


if __name__ == '__main__':
    main()
