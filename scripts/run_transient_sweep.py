#!/usr/bin/env python3
"""
Run a parameterizable 2D transient sweep (Duration × Amplitude).

Two independent 1D sweeps per polarity (negative / positive):
  - Duration sweep : vary duration, fix amplitude at median
  - Amplitude sweep: vary amplitude, fix duration at median

Default: 1 + 2*(25 + 25) = 101 runs.

Usage:
    python3 scripts/run_transient_sweep.py [options]

Options:
    --n-duration N          Duration points (default: 25)
    --n-amplitude N         Amplitude points (default: 25)
    --duration-range MIN MAX  ms range (default: 50 500)
    --amplitude-range MIN MAX  fraction-of-I0 range (default: 0.1 0.5)
    --fixed-duration MS     Duration used for amplitude sweep (default: median)
    --fixed-amplitude AMP   Amplitude used for duration sweep (default: median)
    --output-dir DIR        Root output directory (default: figs/ring/run/transient_sweep)
    --params-json FILE      Parameter file (default: figs/optim/bistable_high_fr/best_params.json)
    --workers N             Parallel workers, 1–10 (default: 4)
    --dry-run               Print jobs without running them
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ─── argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-duration",      type=int,   default=25)
    p.add_argument("--n-amplitude",     type=int,   default=25)
    p.add_argument("--duration-range",  type=float, nargs=2, default=[50, 500],
                   metavar=("MIN", "MAX"))
    p.add_argument("--amplitude-range", type=float, nargs=2, default=[0.1, 0.5],
                   metavar=("MIN", "MAX"))
    p.add_argument("--fixed-duration",  type=float, default=None)
    p.add_argument("--fixed-amplitude", type=float, default=None)
    p.add_argument("--output-dir",   default="figs/ring/run/transient_sweep")
    p.add_argument("--params-json",  default="figs/optim/bistable_high_fr/best_params.json")
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--force",        action="store_true",
                   help="Re-run all jobs (ignore cache)")
    return p.parse_args()


# ─── job builder ─────────────────────────────────────────────────────────────

COMMON = [
    "--sigma_pyr_deg",    "15",
    "--w_pyr_pyr_inter",  "0.002514",
    "--w_pv_global",      "0.03",
    "--amplitude",        "0.55",
    "--delay_ms",         "5000",
    "--seed",             "rdm",
    "--sigma_noise",      "0.05",
    "--no_snapshot_mp4",
]


def _has_valid_metrics(run_dir: Path) -> bool:
    """Check if run_dir has valid run_metrics.json with post_transient."""
    import json
    mf = run_dir / "run_metrics.json"
    if not mf.exists():
        return False
    try:
        with open(mf) as f:
            m = json.load(f)
        return m.get("post_transient") is not None
    except Exception:
        return False


def build_jobs(args):
    """Return list of (label, cmd) tuples in submission order (2D grid sweep)."""
    durations  = np.linspace(*args.duration_range,  args.n_duration).astype(int)
    amplitudes = np.linspace(*args.amplitude_range, args.n_amplitude)

    base = Path(args.output_dir)
    params = args.params_json

    def cmd_base(output_dir):
        return [
            sys.executable, "-m", "circuit_model", "ring-run",
            "--params_json", params,
            *COMMON,
            "--output_dir", str(output_dir),
        ]

    def transient_args(duration_ms, factor):
        return [
            "--response_onset_ms",    "0",
            "--response_duration_ms", str(duration_ms),
            "--response_factor",      f"{factor:.4f}",
        ]

    jobs = []
    force = getattr(args, 'force', False)

    # Baseline
    baseline_dir = base / "baseline"
    if force or not _has_valid_metrics(baseline_dir):
        jobs.append((
            "baseline",
            cmd_base(baseline_dir),
        ))

    # Negative 2D grid: all (duration, amplitude) combinations
    for dur in durations:
        for amp in amplitudes:
            label = f"NEG dur {dur:>4} ms  amp={-amp:>5.2f}"
            out_dir = base / "negative" / f"{dur}ms_{amp:.2f}"
            if force or not _has_valid_metrics(out_dir):
                jobs.append((label, cmd_base(out_dir) + transient_args(dur, -amp)))

    # Positive 2D grid: all (duration, amplitude) combinations
    for dur in durations:
        for amp in amplitudes:
            label = f"POS dur {dur:>4} ms  amp=+{amp:>5.2f}"
            out_dir = base / "positive" / f"{dur}ms_{amp:.2f}"
            if force or not _has_valid_metrics(out_dir):
                jobs.append((label, cmd_base(out_dir) + transient_args(dur, +amp)))

    return jobs, durations, amplitudes


# ─── runner ──────────────────────────────────────────────────────────────────

def run_job(label, cmd):
    """Run a single ring-run job; return (label, returncode, last_lines)."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    last_lines = "\n".join(result.stdout.strip().splitlines()[-3:])
    return label, result.returncode, last_lines


def main():
    args = parse_args()

    workers = max(1, min(args.workers, 10))
    jobs, durations, amplitudes = build_jobs(args)
    total = len(jobs)

    dur_min, dur_max = args.duration_range
    amp_min, amp_max = args.amplitude_range

    print("=" * 60)
    print("TRANSIENT SWEEP 2D  (Duration × Amplitude)")
    print("=" * 60)
    print(f"  Duration points : {args.n_duration}  ({dur_min:.0f}–{dur_max:.0f} ms)")
    print(f"  Amplitude points: {args.n_amplitude}  ({amp_min:.2f}–{amp_max:.2f})")
    print(f"  Grid size       : {args.n_duration} × {args.n_amplitude} = {args.n_duration * args.n_amplitude} runs per polarity")
    print(f"  Total runs      : {total} (1 baseline + 2 × {args.n_duration * args.n_amplitude})")
    print(f"  Workers         : {workers}")
    print(f"  Output          : {args.output_dir}")
    if args.dry_run:
        print("  Mode            : DRY-RUN (no execution)")
    print("=" * 60)

    if args.dry_run:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"[{i:>3}/{total}] {label}")
            print("       " + " ".join(cmd))
        return

    # Create output root
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    failed = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, label, cmd): label
                   for label, cmd in jobs}

        with tqdm(total=total, smoothing=0, unit="run", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                label, rc, tail = future.result()
                if rc != 0:
                    failed.append(label)
                    tqdm.write(f"  FAIL {label}")
                    tqdm.write(f"       {tail}")
                bar.set_postfix_str(label[:40])
                bar.update(1)

    elapsed_total = time.monotonic() - t0
    print("=" * 60)
    if failed:
        print(f"DONE with {len(failed)} failure(s) in {elapsed_total:.1f}s:")
        for f in failed:
            print(f"  FAILED: {f}")
        sys.exit(1)
    else:
        print(f"All {total} runs completed successfully in {elapsed_total:.1f}s.")
        print(f"Outputs in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
