"""Run a local demo pipeline for the clinical trial eligibility benchmark.

Default (offline, fast):
    PYTHONPATH=. python scripts/run_local_demo.py

Steps:
    1. scripts/download_trials.py  (offline reuse or live download)
    2. eval/run_sample_benchmark.py  (quick-demo subset or full sample)
"""

import argparse
import subprocess
import sys
from pathlib import Path

DOWNLOAD_SCRIPT = Path("scripts") / "download_trials.py"
BENCHMARK_SCRIPT = Path("eval") / "run_sample_benchmark.py"
DEFAULT_OUTPUT = Path("data") / "raw" / "parkinson_trials_raw.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local demo pipeline (download + sample benchmark)."
    )
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Run eval/run_sample_benchmark.py without --quick-demo (full sample benchmark).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download/offline-reuse step.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        metavar="N",
        help="Pass --max-trials N to download_trials.py when running online.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Download fresh trial data from ClinicalTrials.gov instead of reusing offline file.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    """Print and execute a command; exit non-zero if it fails."""
    print(">>> " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: command exited with status {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def build_download_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(DOWNLOAD_SCRIPT), "--output", str(DEFAULT_OUTPUT)]
    if not args.online:
        cmd.append("--offline")
    elif args.max_trials is not None:
        cmd.extend(["--max-trials", str(args.max_trials)])
    return cmd


def build_benchmark_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(BENCHMARK_SCRIPT)]
    if not args.full_sample:
        cmd.append("--quick-demo")
    return cmd


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        run_command(build_download_cmd(args))

    run_command(build_benchmark_cmd(args))


if __name__ == "__main__":
    main()
