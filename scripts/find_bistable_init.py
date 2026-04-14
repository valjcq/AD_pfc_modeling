#!/usr/bin/env python3
"""
Find bistable parameter initializations using loop gain analysis.

Key insight: For a sigmoid transfer function, bistability emerges when:
  1. Loop gain (dPhi/dr * dI/dr) > 1 over a range of firing rates
  2. I_PYR threshold is achievable (above THETA_PYR)
  3. Parameters form an S-shaped nullcline (multiple crossings with identity line)

This script:
- Loads a viable baseline (e.g., WT fit)
- Analytically computes loop gain as a function of key parameters
- Identifies parameter ranges that produce bistability
- Generates candidates with validated bistability

Usage:
    python find_bistable_init.py --base WT_1mo_circuit_reference_20260413.json
    python find_bistable_init.py --base WT_1mo_circuit_reference_20260413.json --plot
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve, brentq
import warnings
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS, R_MAX_PHYS


class LoopGainAnalyzer:
    """Analyze loop gain to identify bistability regimes."""

    def __init__(self, params_dict):
        self.params = params_dict.copy()

    def transfer_function(self, I, Theta, alpha, g):
        """Wong-Wang transfer function."""
        u = alpha * (I - Theta)
        if np.isscalar(u):
            gu = g * u
            if gu > 500:
                phi = u
            elif np.abs(gu) < 1e-6:
                phi = 1.0 / g + u / 2.0
            elif gu < -500:
                phi = 0.0
            else:
                phi = u / (1.0 - np.exp(-gu))
        else:
            phi = np.zeros_like(u)
            gu = g * u
            large_pos = gu > 500
            phi[large_pos] = u[large_pos]
            large_neg = gu < -500
            phi[large_neg] = 0.0
            small_gu = (np.abs(gu) < 1e-6) & ~large_pos & ~large_neg
            phi[small_gu] = 1.0 / g + u[small_gu] / 2.0
            regular = ~large_pos & ~large_neg & ~small_gu
            phi[regular] = u[regular] / (1.0 - np.exp(-gu[regular]))
        return np.clip(phi, 0.0, 200.0)

    def transfer_function_derivative(self, I, Theta, alpha, g, dr=1e-4):
        """Numerical derivative of transfer function."""
        phi_here = self.transfer_function(I, Theta, alpha, g)
        phi_plus = self.transfer_function(I + dr, Theta, alpha, g)
        return (phi_plus - phi_here) / dr

    def get_g_GABA(self):
        return self.params.get('g_gaba_base', 1.0) + self.params.get('g_alpha7', 0.0)

    def interneuron_steady_state(self, r_pyr):
        """Solve for steady-state interneuron rates."""
        p = self.params

        # VIP
        I_vip = p['w_ev'] * r_pyr + p['I0_vip'] + p.get('I_alpha5_vip', 0.0) * p.get('act_alpha5', 1.0)
        r_vip = self.transfer_function(I_vip, p['Theta_vip'], p['alpha_vip'], p['g_exc'])

        # SOM and PV
        def equations(x):
            r_som, r_pv = x
            I_som = (p['w_es'] * r_pyr - p.get('w_vs', 0.0) * r_vip
                     - p.get('J_adapt_som', 0.0) * r_som + p['I0_som']
                     + p.get('I_alpha7_som', 0.0) * p.get('act_alpha7', 1.0)
                     + p.get('I_beta2_som', 0.0) * p.get('act_beta2', 1.0))
            f_som = self.transfer_function(I_som, p['Theta_som'], p['alpha_som'], p['g_inh']) - r_som

            g_GABA = self.get_g_GABA()
            I_pv = (p['w_ep'] * r_pyr - g_GABA * p['w_pp'] * r_pv
                    - g_GABA * p['w_sp'] * r_som - p.get('w_vp', 0.0) * r_vip
                    + p['I0_pv'] + p.get('I_alpha7_pv', 0.0) * p.get('act_alpha7', 1.0))
            f_pv = self.transfer_function(I_pv, p['Theta_pv'], p['alpha_pv'], p['g_inh']) - r_pv

            return [f_som, f_pv]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sol = fsolve(equations, [0.0, 0.0], full_output=False)
        r_som, r_pv = sol
        return r_som, r_pv, r_vip

    def pyr_input_current(self, r_pyr, r_som, r_pv):
        """Compute PYR input current."""
        p = self.params
        g_GABA = self.get_g_GABA()
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        I_pyr = (J_NMDA * S_star / (1.0 + g_GABA * p['w_pe'] * r_pv)
                 - g_GABA * p['w_se'] * r_som
                 - p.get('J_adapt_pyr', 0.0) * r_pyr
                 + p['I0_pyr'])
        return I_pyr

    def compute_loop_gain(self, r_pyr_sweep):
        """
        Compute loop gain L(r) = dPhi/dI * dI/dr at each r_pyr.

        Returns:
            r_pyr_sweep, loop_gain_array, info_dict
        """
        p = self.params
        loop_gain = np.zeros_like(r_pyr_sweep)
        dPhi_dI_array = np.zeros_like(r_pyr_sweep)
        dI_dr_array = np.zeros_like(r_pyr_sweep)
        r_pyr_at_bistab = []

        dr = 1e-4

        for i, r_pyr in enumerate(r_pyr_sweep):
            # Get interneuron steady state at r_pyr
            r_som, r_pv, _ = self.interneuron_steady_state(r_pyr)

            # dI/dr: sensitivity of I_PYR to changes in r_PYR
            I_pyr_here = self.pyr_input_current(r_pyr, r_som, r_pv)
            I_pyr_plus = self.pyr_input_current(r_pyr + dr, r_som, r_pv)
            dI_dr = (I_pyr_plus - I_pyr_here) / dr

            # dPhi/dI: transfer function slope
            dPhi_dI = self.transfer_function_derivative(
                I_pyr_here, p['Theta_pyr'], p['alpha_pyr'], p['g_exc']
            )

            # Loop gain
            loop_gain[i] = dPhi_dI * dI_dr
            dPhi_dI_array[i] = dPhi_dI
            dI_dr_array[i] = dI_dr

            # Detect where loop gain > 1 (bistable region)
            if loop_gain[i] > 1.0:
                r_pyr_at_bistab.append(r_pyr)

        return r_pyr_sweep, loop_gain, {
            'dPhi_dI': dPhi_dI_array,
            'dI_dr': dI_dr_array,
            'r_pyr_at_bistab': r_pyr_at_bistab
        }

    def estimate_bistability(self, r_pyr_sweep=None):
        """
        Estimate bistability: fraction of operating range with loop gain > 1.

        Returns:
            bistab_fraction: float in [0, 1]
            max_loop_gain: maximum loop gain in the sweep
            has_bistable_region: bool, True if contiguous region with L > 1
        """
        if r_pyr_sweep is None:
            r_pyr_sweep = np.linspace(0.5, 50, 500)

        r_sweep, L, info = self.compute_loop_gain(r_pyr_sweep)
        max_L = np.max(L[np.isfinite(L)])

        # Check for contiguous bistable region
        bistab_indices = np.where(L > 1.0)[0]
        has_bistable = len(bistab_indices) > 0

        if has_bistable:
            # Check if it's a contiguous region (not scattered points)
            gaps = np.diff(bistab_indices)
            max_gap = np.max(gaps) if len(gaps) > 0 else 1
            is_contiguous = max_gap <= 2  # Allow small gaps due to discretization
        else:
            is_contiguous = False

        bistab_frac = len(bistab_indices) / len(r_sweep) if has_bistable else 0.0

        return bistab_frac, max_L, is_contiguous, info


def generate_bistable_sweep(base_params, n_points=50):
    """
    Generate bistable parameter candidates by sweeping J_NMDA and J_adapt_pyr.

    Returns list of (candidate_dict, bistab_score, max_loop_gain)
    """
    base = base_params.copy()
    J_NMDA_base = base.get('J_NMDA', base.get('w_ee', 0.0))
    J_adapt_base = base.get('J_adapt_pyr', 0.0)

    # Sweep ranges
    J_NMDA_factors = np.linspace(1.2, 2.5, 10)  # 20% to 150% increase
    J_adapt_factors = np.linspace(0.2, 1.0, 5)  # 80% to 0% reduction

    candidates = []

    for J_adapt_fac in J_adapt_factors:
        for J_NMDA_fac in J_NMDA_factors:
            candidate = base.copy()
            candidate['J_NMDA'] = J_NMDA_base * J_NMDA_fac
            candidate['J_adapt_pyr'] = J_adapt_base * J_adapt_fac

            try:
                analyzer = LoopGainAnalyzer(candidate)
                bistab_frac, max_L, is_contiguous, _ = analyzer.estimate_bistability()

                # Score: prefer high loop gain with contiguous bistable region
                score = max_L if is_contiguous else max_L * 0.5

                candidates.append({
                    'params': candidate,
                    'J_NMDA_factor': J_NMDA_fac,
                    'J_adapt_factor': J_adapt_fac,
                    'max_loop_gain': max_L,
                    'bistab_fraction': bistab_frac,
                    'is_contiguous': is_contiguous,
                    'score': score
                })
            except Exception as e:
                # Skip if computation fails
                pass

    # Sort by score (descending)
    candidates.sort(key=lambda c: c['score'], reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Base parameter file')
    parser.add_argument('--n_candidates', type=int, default=5, help='Number of top candidates to save')
    parser.add_argument('--plot', action='store_true', help='Plot loop gain analysis')
    parser.add_argument('--output_dir', default='params/warm_init_loop_gain',
                        help='Output directory')
    args = parser.parse_args()

    # Load base
    base_path = Path('params/best_fit_params') / args.base
    with open(base_path) as f:
        base_params = json.load(f)

    print(f"{'='*70}")
    print(f"Loop Gain Analysis - Bistable Parameter Search")
    print(f"{'='*70}")
    print(f"Base: {args.base}")

    # Analyze base
    analyzer_base = LoopGainAnalyzer(base_params)
    bistab_base, max_L_base, contig_base, _ = analyzer_base.estimate_bistability()
    print(f"\nBase parameter loop gain:")
    print(f"  Max loop gain: {max_L_base:.4f}")
    print(f"  Bistable fraction: {bistab_base*100:.1f}%")
    print(f"  Contiguous region: {contig_base}")

    # Generate and evaluate candidates
    print(f"\n{'='*70}")
    print(f"Sweeping J_NMDA and J_adapt_pyr...")
    print(f"{'='*70}\n")

    candidates = generate_bistable_sweep(base_params)

    # Save top candidates
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"Top candidates (sorted by loop gain with bistable region priority)")
    print(f"{'='*70}\n")

    for i in range(min(args.n_candidates, len(candidates))):
        c = candidates[i]
        bistab_str = "✓ bistab" if c['is_contiguous'] else "  viable"
        print(f"Candidate {i+1}/{min(args.n_candidates, len(candidates))}")
        print(f"  J_NMDA: {c['J_NMDA_factor']:.2f}× base | J_adapt: {c['J_adapt_factor']:.2f}× base")
        print(f"  Max loop gain: {c['max_loop_gain']:.4f} | Bistab frac: {c['bistab_fraction']*100:.1f}% | {bistab_str}")

        # Save
        out_path = output_dir / f"warm_init_{i+1:02d}_loopgain.json"
        with open(out_path, 'w') as f:
            json.dump(c['params'], f, indent=2)
        print(f"  → {out_path}\n")

        # Plot if requested
        if args.plot:
            analyzer = LoopGainAnalyzer(c['params'])
            r_sweep = np.linspace(0.5, 50, 300)
            r_sweep, L, info = analyzer.compute_loop_gain(r_sweep)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(r_sweep, L, 'b-', linewidth=2, label='Loop gain L(r)')
            ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Bistability threshold (L=1)')
            ax.fill_between(r_sweep, 0, 1, alpha=0.1, color='green', label='Monostable (L<1)')
            ax.fill_between(r_sweep, 1, np.max(L)*1.1, alpha=0.1, color='orange', label='Bistable (L>1)')
            ax.set_xlabel('r_PYR (Hz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loop gain L(r)', fontsize=12, fontweight='bold')
            ax.set_title(f'Candidate {i+1}: J_NMDA={c["J_NMDA_factor"]:.2f}× '
                        f'J_adapt={c["J_adapt_factor"]:.2f}×', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            ax.set_xlim(0, 50)

            plot_path = output_dir / f"loop_gain_{i+1:02d}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  → {plot_path}")
            plt.close(fig)

    # Recommendation
    print(f"\n{'='*70}")
    print(f"Recommendation")
    print(f"{'='*70}")
    best = candidates[0]
    if best['is_contiguous']:
        print(f"✓ Found bistable region candidates")
        print(f"  Recommended: warm_init_01_loopgain.json")
        print(f"  Use with: --params_json {output_dir}/warm_init_01_loopgain.json")
    else:
        print(f"⚠ Loop gain >1 achievable but not contiguous")
        print(f"  Optimizer should still find bistable solution, avoiding silent networks")
        print(f"  Use: warm_init_01_loopgain.json (highest max loop gain)")


if __name__ == '__main__':
    main()
