#!/usr/bin/env python3
"""
List all available optimization logs and allow quick viewing.

Usage:
    python list_optim_logs.py
    python list_optim_logs.py --view bistable_physiol_long
"""

import json
from pathlib import Path
import sys
import subprocess

def get_log_info(log_path: Path) -> dict:
    """Extract info from a log file."""
    info = {
        'path': log_path,
        'steps': 0,
        'first_loss': None,
        'best_loss': float('inf'),
        'latest_loss': None,
    }

    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        info['steps'] += 1
                        loss = entry.get('loss', float('inf'))
                        if info['first_loss'] is None:
                            info['first_loss'] = loss
                        if loss < info['best_loss']:
                            info['best_loss'] = loss
                        info['latest_loss'] = loss
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        pass

    return info

def list_logs():
    """List all available optimization logs."""
    optim_dir = Path("figs/optim")
    if not optim_dir.exists():
        print("No optimization logs found (figs/optim/ directory does not exist)")
        return []

    logs = []
    for run_dir in sorted(optim_dir.iterdir()):
        if run_dir.is_dir():
            log_file = run_dir / "log.jsonl"
            if log_file.exists():
                info = get_log_info(log_file)
                logs.append((run_dir.name, info))

    return logs

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="List and view optimization logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_optim_logs.py
  python list_optim_logs.py --view bistable_physiol_long
  python list_optim_logs.py --watch bistable_physiol_long
        """
    )
    parser.add_argument("--view", type=str, help="View a specific run's log")
    parser.add_argument("--watch", type=str, help="Watch a specific run's log for updates")
    parser.add_argument("--tail", type=int, default=20, help="Show last N entries (with --view)")

    args = parser.parse_args()

    if args.view or args.watch:
        # View specific log
        run_name = args.view or args.watch
        log_path = Path("figs/optim") / run_name / "log.jsonl"

        if not log_path.exists():
            print(f"Error: Log file not found for run '{run_name}'")
            print(f"Expected: {log_path}")
            sys.exit(1)

        # Call view_optim_log.py (in same directory as this script)
        script_dir = Path(__file__).parent
        view_script = script_dir / "view_optim_log.py"
        cmd = [sys.executable, str(view_script), str(log_path)]
        if args.view:
            cmd.extend(["--tail", str(args.tail)])
        elif args.watch:
            cmd.append("--watch")

        subprocess.run(cmd)
    else:
        # List available logs
        logs = list_logs()

        if not logs:
            print("No optimization logs found in figs/optim/")
            sys.exit(0)

        print("\n" + "=" * 100)
        print("Available Optimization Runs")
        print("=" * 100)
        print(f"{'Run Name':<40} | {'Steps':>6} | {'Best Loss':>12} | {'Latest Loss':>12}")
        print("-" * 100)

        for run_name, info in logs:
            best_loss = info['best_loss'] if info['best_loss'] < float('inf') else 'N/A'
            latest_loss = info['latest_loss']
            best_str = f"{best_loss:.4g}" if isinstance(best_loss, float) else str(best_loss)
            latest_str = f"{latest_loss:.4g}" if isinstance(latest_loss, float) else str(latest_loss)

            print(
                f"{run_name:<40} | {info['steps']:>6} | {best_str:>12} | {latest_str:>12}"
            )

        print("=" * 100)
        print("\nTo view a run:")
        print("  python list_optim_logs.py --view <run_name>")
        print("  python list_optim_logs.py --watch <run_name>  # Monitor in real-time")
        print()

if __name__ == "__main__":
    main()
