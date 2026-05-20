#!/usr/bin/env python3
"""
Sweep transient amplitude × noise sigma for a single-node bistable run.

Output layout (mirrors ring_transient_sweep.py)
-----------------------------------------------
    <output_dir>/
        {noise_type}/
            noise{sigma}/
                cue{amp}/
                    plot.png
                    run_metrics.json

After every noise-sigma column completes, the state heatmap is updated.

Usage:
    python scripts/bistable_transient_sweep.py --input_params PATH --noise_type {white,ou} [options]

Examples:
    # White noise sweep (sigma 0.0-1.0)
    python scripts/bistable_transient_sweep.py \
        --input_params params_bistable/last_opti/bistable_params.json \
        --noise_type white \
        --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --sigmas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

    # OU noise sweep (sigma 0.0-0.4)
    python scripts/bistable_transient_sweep.py \
        --input_params params_bistable/last_opti/bistable_params.json \
        --noise_type ou \
        --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
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
                   choices=["WT", "WT_APP", "a7_KO", "a7_KO_APP", "b2_KO", "b2_KO_APP", "a5_KO", "a5_KO_APP", "APP_sim"],
                   help="Experimental condition preset (default: none)")
    p.add_argument("--noise_type", required=True, choices=["white", "ou"],
                   help="Noise type: 'white' for white noise, 'ou' for Ornstein-Uhlenbeck noise")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                   help="Parallel worker processes (default: all CPUs)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def build_jobs(args):
    params = Path(args.input_params).resolve()
    outdir = Path(args.output_dir).resolve() if args.output_dir else params.parent / "transient_sweep"

    # Group jobs by noise sigma (so we can plot heatmap after each column)
    sigma_groups: dict[float, list] = {}
    for sigma in args.sigmas:
        group = []
        for amp in args.amplitudes:
            run_dir = outdir / args.noise_type / f"noise{sigma}" / f"cue{amp}"

            cmd = [
                sys.executable, "-m", "circuit_model", "run",
                "--params_json", str(params),
                "--enable_transient",
                "--trans_start_ms", str(args.trans_start_ms),
                "--trans_duration_ms", str(args.trans_duration_ms),
                "--trans_factor", str(amp),
                "--noise_type", args.noise_type,
                "--T_ms", str(args.T_ms),
                "--no_show",
                "--save_plot", str(run_dir / "plot.png"),
                "--save_metrics", str(run_dir / "run_metrics.json"),
            ]

            if sigma > 0.0:
                cmd += ["--sigma_noise", str(sigma)]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.condition is not None:
                cmd += ["--condition", args.condition]

            label = f"{args.noise_type}/noise{sigma}/cue{amp}"
            group.append((label, cmd))
        sigma_groups[sigma] = group

    return sigma_groups, outdir


def run_job(name, cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    return name, result.returncode, tail


def _plot_heatmap(outdir: Path) -> None:
    """Regenerate the state heatmap from all runs completed so far."""
    heatmap_script = Path(__file__).parent / "bistable_sweep_heatmap.py"
    if not heatmap_script.exists():
        tqdm.write("  (bistable_sweep_heatmap.py not found — skipping heatmap)")
        return
    output_path = outdir / "state_heatmap.png"
    result = subprocess.run(
        [sys.executable, str(heatmap_script),
         "--sweep_dir", str(outdir),
         "--output", str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        tqdm.write(f"  → Heatmap updated: {output_path}")
    else:
        tqdm.write(f"  → Heatmap failed:\n{result.stderr.strip()[:400]}")


def main():
    args = parse_args()
    sigma_groups, outdir = build_jobs(args)
    total = sum(len(g) for g in sigma_groups.values())

    print(f"Params     : {args.input_params}")
    print(f"Output     : {outdir}")
    print(f"Amplitudes : {args.amplitudes}")
    print(f"Noise type : {args.noise_type}")
    print(f"Sigmas     : {args.sigmas}")
    print(f"Total runs : {total}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for sigma, group in sigma_groups.items():
            for name, cmd in group:
                print(f"  {name}")
                print("    " + " ".join(cmd))
        return

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Workers    : {args.workers}")

    failed = []
    with tqdm(total=total, unit="run") as bar:
        for sigma in sorted(sigma_groups.keys()):
            group = sigma_groups[sigma]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_job, n, c): n for n, c in group}
                for future in as_completed(futures):
                    name, rc, tail = future.result()
                    if rc != 0:
                        failed.append(name)
                        tqdm.write(f"FAIL {name}\n  {tail}")
                    bar.set_postfix_str(name[:55])
                    bar.update(1)

            tqdm.write(f"\nCompleted noise={sigma} — plotting heatmap...")
            _plot_heatmap(outdir)

    if failed:
        print(f"\n{len(failed)} failure(s):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nDone. {total} runs saved to {outdir}")


if __name__ == "__main__":
    main()
