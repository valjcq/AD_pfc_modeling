#!/usr/bin/env python3
"""
Run batch study with varying tau_noise_ms values.

Usage:
    python run_tau_study.py                          # defaults: 1-10ms, 10 steps, 200 runs
    python run_tau_study.py --tau_min 1 --tau_max 20 --n_tau 5
    python run_tau_study.py --n_runs 100 --tau_min 0.5 --tau_max 15 --n_tau 8
"""

import argparse
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run batch study with varying tau_noise_ms values"
    )
    parser.add_argument("--tau_min", type=float, default=1.0,
                        help="Minimum tau_noise_ms value (default: 1.0)")
    parser.add_argument("--tau_max", type=float, default=10.0,
                        help="Maximum tau_noise_ms value (default: 10.0)")
    parser.add_argument("--n_tau", type=int, default=10,
                        help="Number of tau values to test (default: 10)")
    parser.add_argument("--n_runs", type=int, default=200,
                        help="Number of simulation runs per condition (default: 200)")
    parser.add_argument("--noise_type", type=str, default="white",
                        choices=["white", "ou", "none"],
                        help="Noise type (default: white)")
    args = parser.parse_args()

    # Create linear space of tau values
    tau_values = np.linspace(args.tau_min, args.tau_max, args.n_tau)

    print(f"\nTau study configuration:")
    print(f"  tau_noise_ms: {args.tau_min} to {args.tau_max} ms ({args.n_tau} values)")
    print(f"  Values: {[f'{t:.2f}' for t in tau_values]}")
    print(f"  Runs per condition: {args.n_runs}")
    print(f"  Noise type: {args.noise_type}")

    generated_files = []

    for tau in tau_values:
        output_file = f"study_tau{tau:.1f}ms.png"
        generated_files.append(output_file)

        print(f"\n{'='*60}")
        print(f"Running study with tau_noise_ms = {tau:.2f} ms")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "-m", "circuit_model", "study",
            "--n_runs", str(args.n_runs),
            "--noise_type", args.noise_type,
            "--tau_noise_ms", str(tau),
            "--save_plot", output_file,
            "--no_show",
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Error running study with tau={tau:.2f}ms")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All studies completed!")
    print(f"Generated {len(generated_files)} files:")
    for f in generated_files:
        print(f"  - {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
