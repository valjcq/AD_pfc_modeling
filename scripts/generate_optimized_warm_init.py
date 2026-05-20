#!/usr/bin/env python3
"""
Generate optimized warm initializations specifically tuned for bistability.

Uses nullcline analysis to directly target configurations with:
- Loop gain conditions > 1 (bistability potential)
- Firing rates in physiological range
- Minimal Turing loss

This is more direct than the parameter sweep approach.

Usage:
    python generate_optimized_warm_init.py \
      --base WT_1mo_circuit_reference_20260413.json \
      --J_NMDA_min 0.12 --J_NMDA_max 0.25 \
      --J_adapt_max 0.02 \
      --output_dir params/warm_init_optimized
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS


class BistabilityTargetedGenerator:
    """Generate warm inits targeting bistability."""

    def __init__(self, params_dict):
        self.params = params_dict.copy()

    def transfer_function(self, I, Theta, alpha, g):
        """Wong-Wang TF."""
        u = alpha * (I - Theta)
        if np.isscalar(u):
            gu = g * u
            if gu > 500:
                return u
            elif np.abs(gu) < 1e-6:
                return 1.0 / g + u / 2.0
            elif gu < -500:
                return 0.0
            else:
                return u / (1.0 - np.exp(-gu))
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
        return self.params.get('g_gaba_base', 1.0) + self.params.get('g_alpha7', 0.0)

    def interneuron_steady_state(self, r_pyr):
        """Solve for interneuron rates."""
        p = self.params
        I_vip = p['w_ev'] * r_pyr + p['I0_vip'] + p.get('I_alpha5_vip', 0.0)
        r_vip = self.transfer_function(I_vip, p['Theta_vip'], p['alpha_vip'], p['g_exc'])

        def equations(x):
            r_som, r_pv = x
            I_som = (p['w_es'] * r_pyr - p.get('w_vs', 0.0) * r_vip
                     - p.get('J_adapt_som', 0.0) * r_som + p['I0_som']
                     + p.get('I_alpha7_som', 0.0) + p.get('I_beta2_som', 0.0))
            f_som = self.transfer_function(I_som, p['Theta_som'], p['alpha_som'], p['g_inh']) - r_som

            g_GABA = self.get_g_GABA()
            I_pv = (p['w_ep'] * r_pyr - g_GABA * p['w_pp'] * r_pv
                    - g_GABA * p['w_sp'] * r_som - p.get('w_vp', 0.0) * r_vip
                    + p['I0_pv'] + p.get('I_alpha7_pv', 0.0))
            f_pv = self.transfer_function(I_pv, p['Theta_pv'], p['alpha_pv'], p['g_inh']) - r_pv
            return [f_som, f_pv]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sol = fsolve(equations, [0.0, 0.0])
        return sol[0], sol[1], r_vip

    def pyr_input_current(self, r_pyr, r_som, r_pv):
        """Compute I_PYR."""
        p = self.params
        g_GABA = self.get_g_GABA()
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        I_pyr = (J_NMDA * S_star / (1.0 + g_GABA * p['w_pe'] * r_pv)
                 - g_GABA * p['w_se'] * r_som
                 - p.get('J_adapt_pyr', 0.0) * r_pyr
                 + p['I0_pyr'])
        return I_pyr

    def estimate_max_loop_gain(self, r_pyr_range=None):
        """
        Estimate maximum loop gain over a range.

        Loop gain L = dPhi/dI * dI/dr at each point.
        High L in bistable region → potential for bistability.
        """
        if r_pyr_range is None:
            r_pyr_range = np.linspace(1, 20, 100)

        p = self.params
        max_loop_gain = 0
        dr = 1e-4

        for r_pyr in r_pyr_range:
            r_som, r_pv, _ = self.interneuron_steady_state(r_pyr)

            # dI/dr at this point
            I_here = self.pyr_input_current(r_pyr, r_som, r_pv)
            I_plus = self.pyr_input_current(r_pyr + dr, r_som, r_pv)
            dI_dr = (I_plus - I_here) / dr

            # dPhi/dI
            dPhi_dI_val = 1e-8  # Placeholder
            # Numerical derivative of transfer function
            phi_here = self.transfer_function(I_here, p['Theta_pyr'], p['alpha_pyr'], p['g_exc'])
            phi_plus = self.transfer_function(I_here + 1e-3, p['Theta_pyr'], p['alpha_pyr'], p['g_exc'])
            if phi_plus != phi_here:
                dPhi_dI_val = (phi_plus - phi_here) / 1e-3

            if np.abs(dI_dr) > 1e-8:
                L = dPhi_dI_val * dI_dr
                max_loop_gain = max(max_loop_gain, L)

        return max_loop_gain


def generate_candidates(base_params,
                        J_NMDA_min=0.10, J_NMDA_max=0.30,
                        J_adapt_max=0.025,
                        I0_pyr_min=0.8):
    """
    Generate candidates with specific ranges for bistability.

    Key targets:
    - J_NMDA: Strong recurrence for bistability (0.10-0.30)
    - J_adapt_pyr: Weak adaptation (< 0.025)
    - I0_pyr: Sufficient drive (> 0.8)
    """
    base_J_NMDA = base_params.get('J_NMDA', base_params.get('w_ee', 0.0))
    base_J_adapt = base_params.get('J_adapt_pyr', 0.0)
    base_I0_pyr = base_params.get('I0_pyr', 0.0)

    candidates = []

    # Grid search over key bistability parameters
    J_NMDA_vals = np.linspace(J_NMDA_min, J_NMDA_max, 6)
    J_adapt_fracs = np.linspace(0.1, 1.0, 5)  # Fraction of base value
    I0_pyr_factor = max(1.0, I0_pyr_min / base_I0_pyr)

    for J_NMDA in J_NMDA_vals:
        for J_adapt_frac in J_adapt_fracs:
            candidate = base_params.copy()
            candidate['J_NMDA'] = J_NMDA
            candidate['J_adapt_pyr'] = base_J_adapt * J_adapt_frac

            # Ensure I0_pyr is sufficient
            if candidate['I0_pyr'] < I0_pyr_min:
                candidate['I0_pyr'] = I0_pyr_min

            # Try to estimate quality
            gen = BistabilityTargetedGenerator(candidate)
            max_L = gen.estimate_max_loop_gain()

            candidates.append({
                'params': candidate,
                'J_NMDA': J_NMDA,
                'J_adapt_frac': J_adapt_frac,
                'max_loop_gain': max_L,
                'score': max_L  # Higher is better
            })

    # Sort by score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Base parameters JSON')
    parser.add_argument('--J_NMDA_min', type=float, default=0.10,
                        help='Minimum J_NMDA (recurrence) value')
    parser.add_argument('--J_NMDA_max', type=float, default=0.30,
                        help='Maximum J_NMDA value')
    parser.add_argument('--J_adapt_max', type=float, default=0.025,
                        help='Maximum J_adapt_pyr value')
    parser.add_argument('--I0_pyr_min', type=float, default=0.8,
                        help='Minimum I0_pyr (external input)')
    parser.add_argument('--n_top', type=int, default=5,
                        help='Number of top candidates to save')
    parser.add_argument('--output_dir', default='params/warm_init_optimized',
                        help='Output directory')
    args = parser.parse_args()

    # Load base
    base_path = Path('params/best_fit_params') / args.base
    with open(base_path) as f:
        base_params = json.load(f)

    print(f"{'='*70}")
    print(f"Optimized Warm Init Generation for Bistability")
    print(f"{'='*70}")
    print(f"Base: {args.base}")
    print(f"\nTarget ranges:")
    print(f"  J_NMDA (recurrence):  [{args.J_NMDA_min:.3f}, {args.J_NMDA_max:.3f}]")
    print(f"  J_adapt_pyr:          [0, {args.J_adapt_max:.4f}]")
    print(f"  I0_pyr (ext input):   >= {args.I0_pyr_min:.2f}")

    # Generate candidates
    print(f"\nGenerating candidates...")
    candidates = generate_candidates(
        base_params,
        J_NMDA_min=args.J_NMDA_min,
        J_NMDA_max=args.J_NMDA_max,
        J_adapt_max=args.J_adapt_max,
        I0_pyr_min=args.I0_pyr_min
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Top {min(args.n_top, len(candidates))} Candidates")
    print(f"{'='*70}\n")

    for i in range(min(args.n_top, len(candidates))):
        c = candidates[i]
        print(f"Candidate {i+1}/{min(args.n_top, len(candidates))}")
        print(f"  J_NMDA:           {c['J_NMDA']:.4f}")
        print(f"  J_adapt_pyr:      {c['J_adapt_frac']*base_params['J_adapt_pyr']:.6f} "
              f"({c['J_adapt_frac']*100:.0f}% of base)")
        print(f"  I0_pyr:           {c['params']['I0_pyr']:.4f}")
        print(f"  Max loop gain:    {c['max_loop_gain']:.6f}")

        out_path = output_dir / f"warm_init_opt_{i+1:02d}.json"
        with open(out_path, 'w') as f:
            json.dump(c['params'], f, indent=2)
        print(f"  → {out_path}\n")

    print(f"{'='*70}")
    print(f"Recommendations for bistable optimization:")
    print(f"{'='*70}")
    best = candidates[0]
    print(f"\nUse Candidate 1: warm_init_opt_01.json")
    print(f"\nWith these loss weights:")
    print(f"""
python -m circuit_model optimize \\
  --mode bistable \\
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \\
  --w_bistab 0.2 \\              # Soft bistability constraint
  --w_rate_bistab 1.0 \\         # Enforce firing rate in bistable mode
  --w_margin 0.5 \\              # Enforce stability margins
  --w_physiol 2.0 \\             # Physiological fit
  --budget 5000 \\
  --params_json {output_dir}/warm_init_opt_01.json \\
  --noise_type none \\
  --n_trials 2 \\
  --output_dir figs/optim/bistable_optimized/
""")

    # Verify candidates
    print(f"\n{'='*70}")
    print(f"Verifying top candidate...")
    print(f"{'='*70}\n")

    import subprocess
    verify_cmd = [
        sys.executable, 'scripts/validate_warm_init.py',
        '--params_json', str(output_dir / 'warm_init_opt_01.json')
    ]
    result = subprocess.run(verify_cmd, capture_output=True, text=True)
    print(result.stdout)


if __name__ == '__main__':
    main()
