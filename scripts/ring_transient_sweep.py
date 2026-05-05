#!/usr/bin/env python3
"""
Sweep cue amplitude × noise sigma × response transient factor for ring-run.

Each combination produces a subdirectory inside --output_dir named:
    cue{amp}_noise{sigma}_response{factor}/

When --sigma_som_values is provided, the sweep is repeated for each sigma_som
value, each in its own subdirectory: sigma_som{value}/cue{amp}_...

Usage:
    python scripts/ring_transient_sweep.py --input_params PATH [options]

Examples:
    python scripts/ring_transient_sweep.py \
        --input_params params_bistable/vold/bistable_params.json

    python scripts/ring_transient_sweep.py \
        --input_params params_bistable/vold/bistable_params.json \
        --amplitudes 0.3 0.5 0.7 \
        --sigmas 0.0 0.1 0.3 \
        --response_factors 0.0 0.3 0.5 \
        --som_pattern uniform \
        --sigma_som_values 5 10 15 20 30 \
        --delay_ms 3000 \
        --workers 8
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
                   help="Path to the circuit params JSON file")
    p.add_argument("--ring_params_json", default=None,
                   help="Path to ring params JSON (sigma_pyr_deg, n_nodes, etc.)")
    p.add_argument("--n_nodes", type=int, default=None,
                   help="Number of ring nodes (default: from ring params JSON or 64)")
    p.add_argument("--output_dir", default=None,
                   help="Root output directory (default: <params_dir>/ring_transient_sweep[_som_<pattern>])")

    # Sweep axes
    p.add_argument("--amplitudes", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 1.0],
                   help="Cue amplitude values (default: 0.3 0.5 0.7 1.0)")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3],
                   help="Noise sigma values (0 = no noise) (default: 0.0 0.1 0.3)")
    p.add_argument("--response_factors", type=float, nargs="+",
                   default=[0.0, 0.3, 0.5],
                   help="Response transient factors (0 = no transient) (default: 0.0 0.3 0.5)")
    p.add_argument("--sigma_som_values", type=float, nargs="+", default=None,
                   help="Sweep over these sigma_som_deg values; each gets its own subdir (default: use ring params JSON value)")

    # Ring-run fixed options
    p.add_argument("--delay_ms", type=float, default=5000.0,
                   help="Delay duration in ms (default: 5000)")
    p.add_argument("--response_duration_ms", type=float, default=500.0,
                   help="Response transient duration in ms (default: 500)")
    p.add_argument("--post_response_ms", type=float, default=3000.0,
                   help="Recording duration after response transient in ms (default: 3000)")
    p.add_argument("--seed", default="rdm",
                   help="Random seed or 'rdm' (default: rdm)")
    p.add_argument("--condition", default="WT",
                   help="Experimental condition (default: WT)")
    p.add_argument("--som_pattern", default=None, choices=["gaussian", "uniform", "none"],
                   help="SOM→PYR connectivity pattern (default: gaussian)")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                   help="Parallel worker processes (default: all CPUs)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def _fmt_sigma(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


def build_jobs(args):
    params = Path(args.input_params).resolve()
    sweep_dir = (
        "ring_transient_sweep"
        if not args.som_pattern or args.som_pattern == "gaussian"
        else f"ring_transient_sweep_som_{args.som_pattern}"
    )
    outdir = Path(args.output_dir).resolve() if args.output_dir else params.parent / sweep_dir

    sigma_som_list = args.sigma_som_values if args.sigma_som_values else [None]

    jobs = []
    for sigma_som in sigma_som_list:
        run_root = outdir / f"sigma_som{_fmt_sigma(sigma_som)}" if sigma_som is not None else outdir

        for sigma in args.sigmas:
            for amp in args.amplitudes:
                for rfactor in args.response_factors:
                    run_outdir = run_root / f"noise{sigma}" / f"cue{amp}" / f"response{rfactor}"

                    cmd = [
                        sys.executable, "-m", "circuit_model", "ring-run",
                        "--params_json", str(params),
                        "--amplitude", str(amp),
                        "--sigma_noise", str(sigma),
                        "--delay_ms", str(args.delay_ms),
                        "--seed", str(args.seed),
                        "--condition", args.condition,
                        "--no_show",
                        "--no_snapshot_mp4",
                        "--output_dir", str(run_outdir),
                    ]

                    if args.ring_params_json:
                        cmd += ["--ring_params_json", str(Path(args.ring_params_json).resolve())]
                    if args.n_nodes is not None:
                        cmd += ["--n_nodes", str(args.n_nodes)]
                    if args.som_pattern is not None:
                        cmd += ["--som_pattern", args.som_pattern]
                    if sigma_som is not None:
                        cmd += ["--sigma_som_deg", str(sigma_som)]

                    if rfactor > 0.0:
                        cmd += [
                            "--response_onset_ms", "0",
                            "--response_duration_ms", str(args.response_duration_ms),
                            "--response_factor", str(rfactor),
                            "--post_response_ms", str(args.post_response_ms),
                        ]

                    label_base = f"noise{sigma}/cue{amp}/response{rfactor}"
                    label = f"sigma_som{_fmt_sigma(sigma_som)}/{label_base}" if sigma_som is not None else label_base
                    jobs.append((label, cmd))

    return jobs, outdir


def run_job(name, cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    return name, result.returncode, tail


def main():
    args = parse_args()
    jobs, outdir = build_jobs(args)
    total = len(jobs)

    print(f"Params          : {args.input_params}")
    print(f"Output          : {outdir}")
    print(f"Amplitudes      : {args.amplitudes}")
    print(f"Noise sigmas    : {args.sigmas}")
    print(f"Response factors: {args.response_factors}")
    if args.sigma_som_values:
        print(f"sigma_som sweep : {args.sigma_som_values}")
    print(f"Total runs      : {total}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for name, cmd in jobs:
            print(f"  {name}")
            print("    " + " ".join(cmd))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Workers         : {args.workers}")

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
        print(f"\nDone. {total} runs saved to {outdir}")


if __name__ == "__main__":
    main()
