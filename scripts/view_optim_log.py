#!/usr/bin/env python3
"""
View optimization log file showing loss and firing rates evolution.

Usage:
    python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl
    python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl --tail 20
    python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl --watch
"""

import json
import sys
from pathlib import Path
from typing import Optional
import time

def read_log_file(log_path: str) -> list[dict]:
    """Read JSONL log file and return list of entries."""
    entries = []
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries

def format_log_entry(entry: dict) -> str:
    """Format a single log entry for display."""
    step = entry.get('step', 'N/A')
    loss = entry.get('loss', 'N/A')

    # Extract firing rates
    means = entry.get('means', {})
    pyr = means.get('pyr', 'N/A')
    som = means.get('som', 'N/A')
    pv = means.get('pv', 'N/A')
    vip = means.get('vip', 'N/A')

    # Format numbers
    if isinstance(loss, (int, float)):
        loss_str = f"{loss:.4g}"
    else:
        loss_str = str(loss)

    if isinstance(pyr, (int, float)):
        pyr_str = f"{pyr:.2f}"
    else:
        pyr_str = str(pyr)

    if isinstance(som, (int, float)):
        som_str = f"{som:.2f}"
    else:
        som_str = str(som)

    if isinstance(pv, (int, float)):
        pv_str = f"{pv:.2f}"
    else:
        pv_str = str(pv)

    if isinstance(vip, (int, float)):
        vip_str = f"{vip:.2f}"
    else:
        vip_str = str(vip)

    return f"Step {step:6} | Loss {loss_str:>10} | PYR {pyr_str:>6} SOM {som_str:>6} PV {pv_str:>6} VIP {vip_str:>6}"

def display_log(entries: list[dict], tail: Optional[int] = None):
    """Display log entries."""
    if not entries:
        print("No log entries found.")
        return

    # Show header
    print("\n" + "=" * 90)
    print("Optimization Progress Log")
    print("=" * 90)
    print(f"{'Step':>6} | {'Loss':>10} | {'PYR':>6} {'SOM':>6} {'PV':>6} {'VIP':>6}")
    print("-" * 90)

    # Determine which entries to show
    if tail is not None:
        entries_to_show = entries[-tail:]
    else:
        entries_to_show = entries

    # Show entries
    for entry in entries_to_show:
        print(format_log_entry(entry))

    # Show summary
    if entries:
        last = entries[-1]
        best = min(entries, key=lambda e: e.get('loss', float('inf')))
        print("-" * 90)
        print(f"Latest: Step {last.get('step', 'N/A')}, Loss {last.get('loss', 'N/A'):.4g}")
        print(f"Best:   Step {best.get('step', 'N/A')}, Loss {best.get('loss', 'N/A'):.4g}")
    print("=" * 90 + "\n")

def watch_log(log_path: str, interval: float = 2.0):
    """Watch log file for updates and display."""
    last_count = 0
    print(f"Watching log file: {log_path} (refresh every {interval}s)")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            entries = read_log_file(log_path)
            if len(entries) > last_count:
                # Clear screen and show latest entries
                print("\033[2J\033[H")  # Clear screen
                display_log(entries, tail=10)
                last_count = len(entries)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExiting log viewer.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="View optimization log file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl
  python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl --tail 20
  python view_optim_log.py figs/optim/bistable_physiol_long/log.jsonl --watch
        """
    )
    parser.add_argument("log_file", help="Path to JSONL log file")
    parser.add_argument("--tail", type=int, default=None, help="Show only last N entries")
    parser.add_argument("--watch", action="store_true", help="Watch log file for updates")

    args = parser.parse_args()

    if args.watch:
        watch_log(args.log_file)
    else:
        entries = read_log_file(args.log_file)
        display_log(entries, tail=args.tail)
