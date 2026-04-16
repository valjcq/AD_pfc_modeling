#!/usr/bin/env python3
"""
Fill missing cells in transient sweep grid.

Identifies (duration, amplitude) combinations that exist in the dataset but are
incomplete, and runs only the missing combinations to complete the grid.

Usage:
    python3 scripts/fill_missing_sweep.py [--workers N]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="figs/ring/run/transient_sweep")
    p.add_argument("--params-json", default="figs/optim/bistable_high_fr/best_params.json")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def get_missing_combinations(base: Path):
    """Identify missing (duration, amplitude) pairs."""
    missing = {"negative": [], "positive": []}

    for polarity in ["negative", "positive"]:
        pol_dir = base / polarity

        # Get all existing (dur, amp) with data
        existing = set()
        durs_all = set()
        amps_all = set()

        for run_dir in pol_dir.iterdir():
            if not run_dir.is_dir():
                continue
            mf = run_dir / "run_metrics.json"
            if not mf.exists():
                continue
            try:
                with open(mf) as f:
                    m = json.load(f)
                if m.get("post_transient"):
                    parts = run_dir.name.split('_')
                    dur = int(parts[0].replace('ms', ''))
                    amp = float(parts[1])
                    existing.add((dur, amp))
                    durs_all.add(dur)
                    amps_all.add(amp)
            except:
                pass

        durs = sorted(durs_all)
        amps = sorted(amps_all)

        # Find missing cells
        for dur in durs:
            for amp in amps:
                if (dur, amp) not in existing:
                    missing[polarity].append((dur, amp))

    return missing


def build_missing_jobs(args, missing):
    """Build job list for missing combinations."""
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

    for polarity, sign in [("negative", "-"), ("positive", "+")]:
        for dur, amp in missing[polarity]:
            label = f"{polarity.upper()} dur {dur:>4} ms  amp={sign}{amp:>5.2f}"
            out_dir = base / polarity / f"{dur}ms_{amp:.2f}"
            factor = -amp if sign == "-" else amp
            jobs.append((label, cmd_base(out_dir) + transient_args(dur, factor)))

    return jobs


def run_job(label, cmd):
    """Run a single ring-run job."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_lines = "\n".join(result.stdout.strip().splitlines()[-3:])
    return label, result.returncode, last_lines


def main():
    args = parse_args()
    base = Path(args.output_dir)

    if not base.exists():
        print(f"Error: {base} not found")
        return

    # Find missing
    missing = get_missing_combinations(base)
    total_missing = len(missing["negative"]) + len(missing["positive"])

    if total_missing == 0:
        print("✓ Grid is complete! No missing cells.")
        return

    print("=" * 70)
    print(f"FILL MISSING CELLS ({total_missing} total)")
    print("=" * 70)
    print(f"  Negative: {len(missing['negative'])} missing")
    print(f"  Positive: {len(missing['positive'])} missing")
    print("=" * 70)

    jobs = build_missing_jobs(args, missing)
    workers = max(1, min(args.workers, 10))

    # Run jobs
    failed = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, label, cmd): label for label, cmd in jobs}

        with tqdm(total=len(jobs), smoothing=0, unit="run", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                label, rc, tail = future.result()
                if rc != 0:
                    failed.append(label)
                    tqdm.write(f"  FAIL {label}")
                    tqdm.write(f"       {tail}")
                bar.set_postfix_str(label[:40])
                bar.update(1)

    elapsed = time.monotonic() - t0
    print("=" * 70)
    if failed:
        print(f"Done with {len(failed)} failure(s) in {elapsed:.1f}s")
        sys.exit(1)
    else:
        print(f"✓ All {total_missing} missing cells filled in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
