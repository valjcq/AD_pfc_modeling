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
        self.w_ee = params_dict.get('w_ee', 0.0)
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
        """Compute net PYR input current (DIVISIVE inhibition from PV)."""
        g_GABA = self.get_g_GABA()
        I_pyr = (self.w_ee * r_pyr / (1.0 + g_GABA * self.w_pe * r_pv)
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
            effective_gain_ee: w_ee / (1 + g_GABA * w_pe * r_pv)
            adaptation_reduction: J_adapt_pyr
            loop_gain: effective gain * dPhi/dr
        """
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

            # dI/dr_pyr at this operating point (accounts for divisive inhibition)
            g_GABA = self.get_g_GABA()
            dI_dr = self.w_ee / (1.0 + g_GABA * self.w_pe * r_pv)

            dPhi_dr[i] = dPhi_dI * dI_dr
            effective_gain_ee[i] = dI_dr

        loop_gain = dPhi_dr

        return dPhi_dr, effective_gain_ee, adaptation_reduction, loop_gain


def print_param_summary(params_dict):
    """Print a summary of key loaded parameters."""
    print("\nKey parameters:")
    print(f"  Recurrent excitation: w_ee={params_dict.get('w_ee', 0):.4f}")
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


def plot_single_condition(circuit, args, params_stem, condition):
    """Plot 3-panel nullcline analysis for a single condition."""
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

    # Compute loop gain
    dPhi_dr, eff_gain, adapt_red, loop_gain = circuit.compute_loop_gain(r_pyr_sweep, r_som, r_pv)

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

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: PYR nullcline
    ax = axes[0]
    ax.plot(r_pyr_sweep, phi_pyr, 'b-', linewidth=2.5, label='Phi_PYR(I_net)')
    ax.plot(r_pyr_sweep, r_pyr_sweep, 'k--', linewidth=1.5, alpha=0.5, label='Identity line')

    # Shade region above R_MAX_PHYS (clamp artifacts)
    ax.axvspan(R_MAX_PHYS, r_pyr_sweep.max(), alpha=0.15, color='gray', label='Clamp artifact region')
    ax.axvline(R_MAX_PHYS, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(R_MAX_PHYS + 5, 250, f'R_MAX_PHYS\n({R_MAX_PHYS:.0f} Hz)', fontsize=9, color='red',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

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

    ax.set_xlim(0, 260)
    ax.set_ylim(0, 260)
    ax.set_xlabel('r_PYR (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Phi_PYR(I_net) (Hz)', fontsize=11, fontweight='bold')
    ax.set_title(f'PYR Nullcline — {condition}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    ax.text(0.98, 0.97, regime, transform=ax.transAxes, fontsize=13,
           fontweight='bold', verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.5))

    # Subplot 2: Interneuron steady states
    ax = axes[1]
    ax.plot(r_pyr_sweep, r_pv, 'r-', linewidth=2.5, label='r_PV')
    ax.plot(r_pyr_sweep, r_som, 'orange', linewidth=2.5, label='r_SOM')
    ax.plot(r_pyr_sweep, r_vip, 'purple', linewidth=2.5, label='r_VIP')

    # Mark fixed point locations
    for r_fp, _ in fixed_points:
        ax.axvline(r_fp, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_xlim(0, 120)
    ax.set_xlabel('r_PYR (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Interneuron rate (Hz)', fontsize=11, fontweight='bold')
    ax.set_title('Interneuron Steady States', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Subplot 3: Loop gain decomposition
    ax = axes[2]
    ax.plot(r_pyr_sweep, loop_gain, 'b-', linewidth=2.5, label='Loop gain (dPhi/dr)')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Bistability threshold')
    ax.fill_between(r_pyr_sweep, 0, 1, alpha=0.1, color='green', label='Monostable region')
    ax.fill_between(r_pyr_sweep, 1, loop_gain.max() * 1.1, alpha=0.1, color='orange', label='Bistable region')

    # Mark fixed point locations
    for r_fp, _ in fixed_points:
        ax.axvline(r_fp, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_xlim(0, 120)
    ax.set_xlabel('r_PYR (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loop gain', fontsize=11, fontweight='bold')
    ax.set_title('Effective Gain (>1 → bistability)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f'Single-Node Nullcline Analysis — {params_stem}',
                fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(f'nullcline_{condition}_{params_stem}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: nullcline_{condition}_{params_stem}.png")
    plt.show()


def plot_all_conditions(params_dict, params_stem):
    """Plot PYR nullclines for all 4 conditions on one figure."""
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

        ax.set_xlim(0, 120)
        ax.set_ylim(0, 120)
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
    fig.tight_layout()
    fig.savefig(f'nullcline_all_conditions_{params_stem}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: nullcline_all_conditions_{params_stem}.png")
    plt.show()


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
    args = parser.parse_args()

    # Load JSON
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"Error: File not found: {args.params_json}")
        return

    with open(params_path, 'r') as f:
        params_dict = json.load(f)

    params_stem = params_path.stem

    print(f"{'='*60}")
    print(f"Loaded: {args.params_json}")
    print_param_summary(params_dict)
    print(f"{'='*60}")

    if args.all_conditions:
        plot_all_conditions(params_dict, params_stem)
    else:
        circuit = SingleNodeCircuit(params_dict)
        circuit.set_condition(args.condition)
        plot_single_condition(circuit, args, params_stem, args.condition)


if __name__ == '__main__':
    main()
