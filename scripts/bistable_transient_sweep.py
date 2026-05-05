#!/usr/bin/env python3
"""
Sweep transient amplitude × noise sigma for a single-node bistable run.

sigma=0 runs with no noise; sigma>0 uses OU noise (same as ring-run).
Each combination saves one PNG using the naming convention:
    cue{amp}_sigma{sigma}.png

Usage:
    python scripts/bistable_transient_sweep.py --input_params PATH [options]

Examples:
    python scripts/bistable_transient_sweep.py \
        --input_params params_bistable/vold/bistable_params.json

    python scripts/bistable_transient_sweep.py \
        --input_params params_bistable/vnew_no_somadapt/best_params.json \
        --amplitudes 0.2 0.5 1.0 2.0 \
        --sigmas 0.0 0.1 0.3 \
        --T_ms 6000 \
        --output_dir /tmp/sweep_out
"""

import argparse
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_params", required=True,
                   help="Path to the params JSON file")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: <params_dir>/transient_sweep)")
    p.add_argument("--amplitudes", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                   help="Transient amplitudes as fraction of PYR I0 (default: 0.1 0.2 0.3 0.5 0.7 1.0)")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3],
                   help="Noise sigma values (0 = no noise, >0 uses OU) (default: 0.0 0.1 0.3)")
    p.add_argument("--trans_start_ms", type=float, default=1000.0,
                   help="Transient start time in ms (default: 1000)")
    p.add_argument("--trans_duration_ms", type=float, default=500.0,
                   help="Transient duration in ms (default: 500)")
    p.add_argument("--T_ms", type=float, default=4000.0,
                   help="Total simulation duration in ms (default: 4000)")
    p.add_argument("--seed", default=None,
                   help="Random seed for reproducibility (default: none)")
    p.add_argument("--condition", default=None,
                   choices=["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP"],
                   help="Experimental condition preset (default: none)")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                   help="Parallel worker processes (default: all CPUs)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def build_jobs(args):
    params = Path(args.input_params).resolve()
    outdir = Path(args.output_dir).resolve() if args.output_dir else params.parent / "transient_sweep"

    jobs = []
    for amp in args.amplitudes:
        for sigma in args.sigmas:
            name = f"cue{amp}_sigma{sigma}.png"
            noise_type = "none" if sigma == 0.0 else "ou"

            cmd = [
                sys.executable, "-m", "circuit_model", "run",
                "--params_json", str(params),
                "--enable_transient",
                "--trans_start_ms", str(args.trans_start_ms),
                "--trans_duration_ms", str(args.trans_duration_ms),
                "--trans_factor", str(amp),
                "--noise_type", noise_type,
                "--T_ms", str(args.T_ms),
                "--no_show",
                "--save_plot", str(outdir / name),
            ]

            if sigma > 0.0:
                cmd += ["--sigma_noise", str(sigma)]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.condition is not None:
                cmd += ["--condition", args.condition]

            jobs.append((name, cmd))

    return jobs, outdir


def run_job(name, cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    return name, result.returncode, tail


def main():
    args = parse_args()
    jobs, outdir = build_jobs(args)
    total = len(jobs)

    print(f"Params     : {args.input_params}")
    print(f"Output     : {outdir}")
    print(f"Amplitudes : {args.amplitudes}")
    print(f"Sigmas     : {args.sigmas}")
    print(f"Total runs : {total}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for name, cmd in jobs:
            print(f"  {name}")
            print("    " + " ".join(cmd))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Workers    : {args.workers}")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, n, c): n for n, c in jobs}
        with tqdm(total=total, unit="run") as bar:
            for future in as_completed(futures):
                name, rc, tail = future.result()
                if rc != 0:
                    failed.append(name)
                    tqdm.write(f"FAIL {name}\n  {tail}")
                bar.set_postfix_str(name[:55])
                bar.update(1)

    if failed:
        print(f"\n{len(failed)} failure(s):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nDone. {total} plots saved to {outdir}")


if __name__ == "__main__":
    main()
