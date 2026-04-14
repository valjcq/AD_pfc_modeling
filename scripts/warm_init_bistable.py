#!/usr/bin/env python3
"""
Warm initialization for bistable mode optimization.

Generates parameter candidates that:
1. Promote bistable nullcline (two stable fixed points)
2. Avoid silent networks (ensure firing above I_PYR threshold)
3. Stay within viable parameter bounds

Key bistability mechanisms:
- Recurrent excitation (J_NMDA): increases loop gain
- Adaptation (J_adapt_pyr): reduces adaptation to allow bifurcations
- PV feedback (w_pe): divisive inhibition stabilizes low-firing state
- External input (I0_pyr): ensures baseline firing

Usage:
    python warm_init_bistable.py --base WT_1mo_circuit_reference_20260413.json
    python warm_init_bistable.py --base WT_1mo_circuit_reference_20260413.json --n_candidates 5
    python warm_init_bistable.py --base WT_1mo_circuit_reference_20260413.json --verbose
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve, brentq
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS, R_MAX_PHYS


class BistabilityAnalyzer:
    """Analyze and generate bistable parameter configurations."""

    def __init__(self, params_dict):
        """Load parameters from dict (CircuitParams JSON format)."""
        self.params = params_dict.copy()
        # Key bistability parameters (target for modification)
        self.bistability_keys = [
            'J_NMDA', 'J_adapt_pyr', 'w_pe', 'w_se', 'I0_pyr', 'g_gaba_base'
        ]

    def transfer_function(self, I, Theta, alpha, g):
        """Wong-Wang transfer function with overflow guards."""
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

    def get_g_GABA(self):
        """Effective GABA gain."""
        return self.params.get('g_gaba_base', 1.0) + self.params.get('g_alpha7', 0.0)

    def interneuron_steady_state(self, r_pyr):
        """Solve for steady-state interneuron rates given r_PYR."""
        p = self.params

        # VIP
        I_vip = p['w_ev'] * r_pyr + p['I0_vip'] + p.get('I_alpha5_vip', 0.0) * p.get('act_alpha5', 1.0)
        r_vip = self.transfer_function(I_vip, p['Theta_vip'], p['alpha_vip'], p['g_exc'])

        # SOM and PV (coupled system)
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
        """Compute PYR input with NMDA gating."""
        p = self.params
        g_GABA = self.get_g_GABA()

        # NMDA gating at steady state
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)

        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        I_pyr = (J_NMDA * S_star / (1.0 + g_GABA * p['w_pe'] * r_pv)
                 - g_GABA * p['w_se'] * r_som
                 - p.get('J_adapt_pyr', 0.0) * r_pyr
                 + p['I0_pyr'])
        return I_pyr

    def nullcline_stability(self, r_pyr_sweep):
        """Compute nullcline and detect stability regime."""
        p = self.params
        F = np.zeros_like(r_pyr_sweep)
        r_som_sweep = np.zeros_like(r_pyr_sweep)
        r_pv_sweep = np.zeros_like(r_pyr_sweep)

        for i, r_pyr in enumerate(r_pyr_sweep):
            r_som, r_pv, _ = self.interneuron_steady_state(r_pyr)
            r_som_sweep[i] = r_som
            r_pv_sweep[i] = r_pv

            I_pyr = self.pyr_input_current(r_pyr, r_som, r_pv)
            phi_pyr = self.transfer_function(I_pyr, p['Theta_pyr'], p['alpha_pyr'], p['g_exc'])
            F[i] = phi_pyr - r_pyr

        return F, r_som_sweep, r_pv_sweep

    def count_stable_fixed_points(self, r_pyr_sweep=None, F=None):
        """Count number of stable fixed points."""
        if r_pyr_sweep is None:
            r_pyr_sweep = np.linspace(0, 200, 1000)
        if F is None:
            F, _, _ = self.nullcline_stability(r_pyr_sweep)

        dF_dr = np.gradient(F, r_pyr_sweep)
        sign_changes = np.where(np.diff(np.sign(F)))[0]

        stable_count = 0
        for idx in sign_changes:
            try:
                r_pyr_min = r_pyr_sweep[idx]
                r_pyr_max = r_pyr_sweep[idx + 1]

                def f_to_root(r):
                    r_som, r_pv, _ = self.interneuron_steady_state(r)
                    I_pyr = self.pyr_input_current(r, r_som, r_pv)
                    phi = self.transfer_function(I_pyr, self.params['Theta_pyr'],
                                                self.params['alpha_pyr'], self.params['g_exc'])
                    return phi - r

                r_fp = brentq(f_to_root, r_pyr_min, r_pyr_max)
                if r_fp < R_MAX_PHYS and dF_dr[idx] < 0:  # stable
                    stable_count += 1
            except (ValueError, RuntimeError):
                pass

        return stable_count

    def is_viable(self):
        """Check if parameters avoid silent network (I_PYR > threshold at rest)."""
        p = self.params
        # At r_pyr = 0, check interneuron response
        r_som, r_pv, _ = self.interneuron_steady_state(0.0)
        I_pyr = self.pyr_input_current(0.0, r_som, r_pv)
        THETA_PYR = p['Theta_pyr']
        return I_pyr > THETA_PYR

    def estimate_min_firing_rate(self):
        """Estimate minimum achievable firing rate."""
        try:
            # Solve for zero-firing equilibrium
            def f_at_zero(eps):
                """Small perturbation analysis."""
                r_som, r_pv, _ = self.interneuron_steady_state(eps)
                I_pyr = self.pyr_input_current(eps, r_som, r_pv)
                return I_pyr - self.params['Theta_pyr']

            # Try to find small r_pyr where I_pyr crosses threshold
            for r_test in np.linspace(0.1, 10, 50):
                r_som, r_pv, _ = self.interneuron_steady_state(r_test)
                I_pyr = self.pyr_input_current(r_test, r_som, r_pv)
                if I_pyr > self.params['Theta_pyr']:
                    return r_test
            return None
        except:
            return None


def generate_bistable_candidate(base_params, strategy='moderate'):
    """
    Generate a bistable candidate by adjusting key parameters.

    Strategies:
    - 'moderate': conservative increases in recurrence, modest reduction in adaptation
    - 'strong': strong increase in recurrence, stronger reduction in adaptation
    - 'conservative': minimal changes, focus on viability
    """
    p = base_params.copy()

    if strategy == 'moderate':
        # Increase recurrent excitation (NMDA)
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        p['J_NMDA'] = J_NMDA * 1.5  # 50% increase

        # Reduce adaptation to allow bistable bifurcation
        p['J_adapt_pyr'] = p.get('J_adapt_pyr', 0.0) * 0.6  # 40% reduction

        # Slightly increase external input to ensure baseline firing
        p['I0_pyr'] = p['I0_pyr'] * 1.1  # 10% increase

    elif strategy == 'strong':
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        p['J_NMDA'] = J_NMDA * 2.0  # 100% increase
        p['J_adapt_pyr'] = p.get('J_adapt_pyr', 0.0) * 0.4  # 60% reduction
        p['I0_pyr'] = p['I0_pyr'] * 1.2  # 20% increase

    elif strategy == 'conservative':
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        p['J_NMDA'] = J_NMDA * 1.2  # 20% increase
        p['J_adapt_pyr'] = p.get('J_adapt_pyr', 0.0) * 0.8  # 20% reduction
        p['I0_pyr'] = p['I0_pyr'] * 1.05  # 5% increase

    return p


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--base', required=True, help='Base parameter file (JSON)')
    parser.add_argument('--n_candidates', type=int, default=3,
                        help='Number of candidates to generate (default: 3)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output_dir', default='params/warm_init',
                        help='Output directory for candidates (default: params/warm_init)')
    args = parser.parse_args()

    # Load base parameters
    base_path = Path('params/best_fit_params') / args.base
    if not base_path.exists():
        print(f"Error: base params not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    with open(base_path) as f:
        base_params = json.load(f)

    print(f"{'='*70}")
    print(f"Bistable Warm Initialization Generator")
    print(f"{'='*70}")
    print(f"Base: {args.base}")

    # Analyze base
    analyzer_base = BistabilityAnalyzer(base_params)
    is_viable = analyzer_base.is_viable()
    stable_fps = analyzer_base.count_stable_fixed_points()
    min_rate = analyzer_base.estimate_min_firing_rate()

    print(f"\nBase parameter analysis:")
    print(f"  Viable (avoids silent): {is_viable}")
    print(f"  Stable fixed points: {stable_fps}")
    print(f"  Min firing rate: {min_rate:.3f} Hz if viable" if min_rate else "  Min firing rate: unable to compute")

    # Generate candidates
    strategies = ['conservative', 'moderate', 'strong']
    candidates = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating {args.n_candidates} warm-init candidates...")
    print(f"{'='*70}\n")

    for i in range(args.n_candidates):
        strategy = strategies[i % len(strategies)]
        candidate = generate_bistable_candidate(base_params, strategy=strategy)

        analyzer = BistabilityAnalyzer(candidate)
        is_viable_cand = analyzer.is_viable()
        stable_fps_cand = analyzer.count_stable_fixed_points()
        min_rate_cand = analyzer.estimate_min_firing_rate()

        # Evaluate quality
        bistable = stable_fps_cand >= 2
        viable = is_viable_cand
        quality = "✓ good" if (viable and bistable) else ("⚠ viable" if viable else "✗ silent")

        print(f"Candidate {i+1}/{args.n_candidates} [{strategy:12s}]")
        print(f"  Viable: {viable:5} | Bistable: {bistable:5} ({stable_fps_cand} FPs) | Quality: {quality}")
        if min_rate_cand:
            print(f"  Min rate: {min_rate_cand:6.3f} Hz")

        # Save candidate
        out_path = output_dir / f"warm_init_{i+1:02d}_{strategy}.json"
        with open(out_path, 'w') as f:
            json.dump(candidate, f, indent=2)
        print(f"  → {out_path}")
        candidates.append((i+1, strategy, candidate, viable, stable_fps_cand))
        print()

    # Summary and recommendation
    print(f"{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    good_candidates = [c for c in candidates if c[3] and c[4] >= 2]
    if good_candidates:
        best_idx, best_strat, _, _, _ = good_candidates[0]
        print(f"✓ Found {len(good_candidates)} viable bistable candidate(s)")
        print(f"→ Recommended: Candidate {best_idx} ({best_strat})")
        print(f"  Use: --params_json params/warm_init/warm_init_{best_idx:02d}_{best_strat}.json")
    else:
        viable_count = sum(1 for c in candidates if c[3])
        if viable_count > 0:
            print(f"⚠ Found {viable_count} viable (but non-bistable) candidate(s)")
            print(f"  These should avoid silent networks but may not support bistability")
            print(f"  Recommendation: try the 'strong' strategy candidate first")
        else:
            print(f"✗ No viable candidates found with current strategies")
            print(f"  Consider adjusting adjustment factors in the script")


if __name__ == '__main__':
    main()
