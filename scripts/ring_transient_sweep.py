#!/usr/bin/env python3
"""
Sweep cue amplitude × noise sigma for ring-run.

Output structure:
    <output_dir>/
        {noise_type}/noise{sigma}/cue{amp}/

When --sigma_som_values is provided, each sigma_som gets its own subtree:
    <output_dir>/
        sigma_som{value}/
            {noise_type}/noise{sigma}/cue{amp}/

Usage:
    python scripts/ring_transient_sweep.py --input_params PATH --noise_type {white,ou} [options]

Examples:
    # White noise sweep (sigma 0.0-1.0)
    python scripts/ring_transient_sweep.py \
        --input_params params_bistable/last_opti/bistable_params.json \
        --noise_type white \
        --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --sigmas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

    # OU noise sweep (sigma 0.0-0.4)
    python scripts/ring_transient_sweep.py \
        --input_params params_bistable/last_opti/bistable_params.json \
        --noise_type ou \
        --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
"""

import argparse
import multiprocessing
import re
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
    p.add_argument("--sigma_som_values", type=float, nargs="+", default=None,
                   help="Sweep over these sigma_som_deg values; each gets its own subdir (default: use ring params JSON value)")

    # Ring-run fixed options
    p.add_argument("--delay_ms", type=float, default=5000.0,
                   help="Delay duration in ms (default: 5000)")
    p.add_argument("--seed", default="rdm",
                   help="Random seed or 'rdm' (default: rdm)")
    p.add_argument("--condition", default="WT",
                   help="Experimental condition (default: WT)")
    p.add_argument("--som_pattern", default=None, choices=["gaussian", "uniform", "none"],
                   help="SOM→PYR connectivity pattern (default: gaussian)")
    p.add_argument("--noise_type", required=True, choices=["white", "ou"],
                   help="Noise type: 'white' for white noise, 'ou' for Ornstein-Uhlenbeck noise")
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
                run_outdir = run_root / args.noise_type / f"noise{sigma}" / f"cue{amp}"

                cmd = [
                    sys.executable, "-m", "circuit_model", "ring-run",
                    "--params_json", str(params),
                    "--amplitude", str(amp),
                    "--sigma_noise", str(sigma),
                    "--noise_type", args.noise_type,
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

                label_base = f"{args.noise_type}/noise{sigma}/cue{amp}"
                label = f"sigma_som{_fmt_sigma(sigma_som)}/{label_base}" if sigma_som is not None else label_base
                jobs.append((label, cmd))

    return jobs, outdir


def run_job(name, cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    return name, result.returncode, tail


_SIGMA_SOM_KEY_RE = re.compile(r"^sigma_som([\d.]+)")


def _sigma_som_sort_key(group_key: str | None) -> float:
    """Numeric sort key for sigma_som group labels; None sorts last."""
    if group_key is None:
        return float("inf")
    m = _SIGMA_SOM_KEY_RE.match(group_key)
    return float(m.group(1)) if m else float("inf")


def _plot_heatmap(outdir: Path) -> None:
    """Regenerate the state heatmap from all runs completed so far."""
    heatmap_script = Path(__file__).parent / "ring_sweep_heatmap.py"
    if not heatmap_script.exists():
        tqdm.write("  (ring_sweep_heatmap.py not found — skipping heatmap)")
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
    jobs, outdir = build_jobs(args)
    total = len(jobs)

    print(f"Params          : {args.input_params}")
    print(f"Output          : {outdir}")
    print(f"Amplitudes      : {args.amplitudes}")
    print(f"Noise type      : {args.noise_type}")
    print(f"Noise sigmas    : {args.sigmas}")
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

    # Group jobs by sigma_som so the heatmap can be updated after each group.
    sigma_som_groups: dict[str | None, list] = {}
    for label, cmd in jobs:
        key = label.split("/")[0] if "/" in label and label.split("/")[0].startswith("sigma_som") else None
        sigma_som_groups.setdefault(key, []).append((label, cmd))

    group_order = sorted(sigma_som_groups.keys(), key=_sigma_som_sort_key)

    failed = []
    with tqdm(total=total, unit="run") as bar:
        for group_key in group_order:
            group_jobs = sigma_som_groups[group_key]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_job, n, c): n for n, c in group_jobs}
                for future in as_completed(futures):
                    name, rc, tail = future.result()
                    if rc != 0:
                        failed.append(name)
                        tqdm.write(f"FAIL {name}\n  {tail}")
                    bar.set_postfix_str(name[:55])
                    bar.update(1)

            label_display = group_key if group_key else "(no sigma_som)"
            tqdm.write(f"\nCompleted {label_display} — plotting heatmap...")
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
