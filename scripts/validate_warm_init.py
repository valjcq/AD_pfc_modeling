#!/usr/bin/env python3
"""
Validate warm initialization candidates.

Checks if parameter sets:
1. Avoid silent networks (I_PYR > threshold at rest)
2. Produce reasonable firing rates in simulation
3. Show bistable nullcline structure

Usage:
    python validate_warm_init.py --params_json params/warm_init/warm_init_03_strong.json
    python validate_warm_init.py --params_json params/warm_init/warm_init_03_strong.json --simulate
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve, brentq
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from circuit_model.constants import GAMMA_NMDA, TAU_NMDA_MS, R_MAX_PHYS


def load_params(params_json):
    """Load parameters from JSON."""
    with open(params_json) as f:
        return json.load(f)


class QuickValidator:
    """Lightweight validation without full simulation."""

    def __init__(self, params_dict):
        self.p = params_dict

    def get_g_GABA(self):
        return self.p.get('g_gaba_base', 1.0) + self.p.get('g_alpha7', 0.0)

    def transfer_function(self, I, Theta, alpha, g):
        """Wong-Wang transfer function."""
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
        return 0.0

    def interneuron_steady_state(self, r_pyr):
        """Solve for interneuron rates at given r_pyr."""
        p = self.p

        # VIP
        I_vip = p['w_ev'] * r_pyr + p['I0_vip'] + p.get('I_alpha5_vip', 0.0)
        r_vip = self.transfer_function(I_vip, p['Theta_vip'], p['alpha_vip'], p['g_exc'])

        # SOM and PV
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
        """Compute I_PYR with NMDA gating."""
        p = self.p
        g_GABA = self.get_g_GABA()
        S_star = (GAMMA_NMDA * r_pyr * TAU_NMDA_MS) / (1.0 + GAMMA_NMDA * r_pyr * TAU_NMDA_MS)
        J_NMDA = p.get('J_NMDA', p.get('w_ee', 0.0))
        I_pyr = (J_NMDA * S_star / (1.0 + g_GABA * p['w_pe'] * r_pv)
                 - g_GABA * p['w_se'] * r_som
                 - p.get('J_adapt_pyr', 0.0) * r_pyr
                 + p['I0_pyr'])
        return I_pyr

    def check_silent_network(self):
        """Check if network can produce non-zero firing."""
        r_som, r_pv, r_vip = self.interneuron_steady_state(0.0)
        I_pyr = self.pyr_input_current(0.0, r_som, r_pv)
        THETA = self.p['Theta_pyr']
        is_viable = I_pyr > THETA
        return is_viable, I_pyr, THETA

    def estimate_steady_state(self):
        """Estimate steady-state firing rates."""
        try:
            # Solve for fixed point of the full system
            def fixed_point_eqs(r):
                r_som, r_pv, r_vip = self.interneuron_steady_state(r)
                I_pyr = self.pyr_input_current(r, r_som, r_pv)
                phi_pyr = self.transfer_function(I_pyr, self.p['Theta_pyr'],
                                                 self.p['alpha_pyr'], self.p['g_exc'])
                return phi_pyr - r

            # Try to find a fixed point
            for r_init in [0.1, 1.0, 5.0, 10.0, 15.0]:
                try:
                    from scipy.optimize import fsolve
                    r_fp = fsolve(fixed_point_eqs, r_init)[0]
                    if r_fp > 0 and r_fp < 100:
                        r_som, r_pv, r_vip = self.interneuron_steady_state(r_fp)
                        return r_fp, r_som, r_pv, r_vip
                except:
                    pass
            return None
        except Exception as e:
            return None

    def count_bistable_fps(self):
        """Count stable fixed points."""
        r_pyr_sweep = np.linspace(0, 150, 500)
        F = np.zeros_like(r_pyr_sweep)

        for i, r_pyr in enumerate(r_pyr_sweep):
            r_som, r_pv, _ = self.interneuron_steady_state(r_pyr)
            I_pyr = self.pyr_input_current(r_pyr, r_som, r_pv)
            phi = self.transfer_function(I_pyr, self.p['Theta_pyr'],
                                        self.p['alpha_pyr'], self.p['g_exc'])
            F[i] = phi - r_pyr

        # Count sign changes
        sign_changes = np.where(np.diff(np.sign(F)))[0]
        stable_count = 0

        dF_dr = np.gradient(F, r_pyr_sweep)
        for idx in sign_changes:
            if dF_dr[idx] < 0:
                stable_count += 1

        return stable_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params_json', required=True, help='Parameters JSON file')
    parser.add_argument('--simulate', action='store_true', help='Run full simulation (slower)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"Error: {args.params_json} not found", file=sys.stderr)
        sys.exit(1)

    params = load_params(params_path)
    validator = QuickValidator(params)

    print(f"{'='*70}")
    print(f"Warm Init Validation: {params_path.name}")
    print(f"{'='*70}\n")

    # Check 1: Silent network
    print("1. Silent Network Check")
    print("-" * 70)
    is_viable, I_pyr_at_0, THETA = validator.check_silent_network()
    status = "✓ VIABLE" if is_viable else "✗ SILENT"
    print(f"   Status: {status}")
    print(f"   I_PYR(r=0): {I_pyr_at_0:.4f} nA")
    print(f"   Threshold:  {THETA:.4f} nA")
    print(f"   Margin:     {I_pyr_at_0 - THETA:.4f} nA {'(good)' if is_viable else '(BAD - network won\'t fire!)'}")

    # Check 2: Steady-state rates
    print("\n2. Steady-State Firing Rate Estimate")
    print("-" * 70)
    ss = validator.estimate_steady_state()
    if ss:
        r_pyr, r_som, r_pv, r_vip = ss
        print(f"   Estimated rates:")
        print(f"     PYR: {r_pyr:6.2f} Hz")
        print(f"     SOM: {r_som:6.2f} Hz")
        print(f"     PV:  {r_pv:6.2f} Hz")
        print(f"     VIP: {r_vip:6.2f} Hz")
    else:
        print(f"   ✗ Could not find stable fixed point (likely silent)")

    # Check 3: Bistability
    print("\n3. Bistable Nullcline Structure")
    print("-" * 70)
    n_stable = validator.count_bistable_fps()
    regime = "BISTABLE" if n_stable >= 2 else ("MONOSTABLE" if n_stable == 1 else "SILENT")
    print(f"   Regime: {regime}")
    print(f"   Stable fixed points: {n_stable}")
    print(f"   {'✓ Bistable - good warm init!' if n_stable >= 2 else ('◐ Monostable - can still work' if n_stable == 1 else '✗ Silent - will not work')}")

    # Check 4: Key parameters
    print("\n4. Key Parameters for Bistability")
    print("-" * 70)
    J_NMDA = params.get('J_NMDA', params.get('w_ee', 0.0))
    J_adapt = params.get('J_adapt_pyr', 0.0)
    I0_pyr = params.get('I0_pyr', 0.0)
    g_gaba = params.get('g_gaba_base', 0.0) + params.get('g_alpha7', 0.0)
    w_pe = params.get('w_pe', 0.0)

    print(f"   J_NMDA (recurrence):   {J_NMDA:.4f}  {'✓' if J_NMDA > 0.1 else '(low)'}")
    print(f"   J_adapt_pyr:           {J_adapt:.4f}  {'✓' if J_adapt < 0.05 else '(high)'}")
    print(f"   I0_pyr (ext input):    {I0_pyr:.4f}  {'✓' if I0_pyr > 0.5 else '✗ (too low!)'}")
    print(f"   g_GABA:                {g_gaba:.4f}")
    print(f"   w_pe (PV feedback):    {w_pe:.4f}")

    # Summary and recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if is_viable and n_stable >= 1:
        if n_stable >= 2:
            print("✓ EXCELLENT: Use this as warm init - it's already bistable!")
            print("  Command: --params_json", args.params_json)
        else:
            print("◐ ACCEPTABLE: Use this as warm init - network fires but not yet bistable")
            print("  The optimizer should be able to find bistability from here")
            print("  Command: --params_json", args.params_json)
            print("  Tip: Use lower bistability weight (w_bistab=0.3) for smoother search")
    elif is_viable:
        print("⚠ MARGINAL: Network barely avoids silence")
        print("  Try increasing I0_pyr manually before using as warm init")
    else:
        print("✗ DO NOT USE: This warm init is SILENT")
        print("  The optimizer will get stuck at zero firing rates")
        print("  Try a different warm init candidate")

    if args.simulate:
        print(f"\n{'='*70}")
        print("Running full simulation (slower)...")
        print(f"{'='*70}")
        try:
            from circuit_model.simulation import NetworkSimulator
            sim = NetworkSimulator(params, dt_ms=0.1, duration_ms=2000)
            rates_over_time = sim.integrate()
            mean_rates = np.mean(rates_over_time[-100:], axis=0)
            print(f"\nFull simulation mean firing rates (last 100ms):")
            print(f"  PYR: {mean_rates[0]:6.2f} Hz")
            print(f"  SOM: {mean_rates[1]:6.2f} Hz")
            print(f"  PV:  {mean_rates[2]:6.2f} Hz")
            print(f"  VIP: {mean_rates[3]:6.2f} Hz")
        except Exception as e:
            print(f"  Simulation failed: {e}")


if __name__ == '__main__':
    main()
