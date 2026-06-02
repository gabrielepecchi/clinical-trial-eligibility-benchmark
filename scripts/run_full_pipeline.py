"""Run the local benchmark pipeline end-to-end.

Usage:
    PYTHONPATH=. python scripts/run_full_pipeline.py
    PYTHONPATH=. python scripts/run_full_pipeline.py --with-tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    ("Run LLM-reviewed benchmark", ["eval/run_llm_reviewed_benchmark.py"]),
    ("Summarize LLM-reviewed errors", ["eval/summarize_llm_reviewed_errors.py"]),
    ("Generate HTML benchmark report", ["eval/generate_benchmark_report.py"]),
    ("Validate processed schemas", ["eval/validate_schema.py"]),
]

TEST_STEP = ("Run pytest", ["-m", "pytest"])


def run_step(index: int, total: int, title: str, args: list[str], env: dict[str, str]) -> None:
    """Run one pipeline step and stop immediately if it fails."""
    print("=" * 72)
    print(f"[{index}/{total}] {title}")
    print("=" * 72)

    command = [sys.executable, *args]
    completed = subprocess.run(command, cwd=ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_env() -> dict[str, str]:
    """Return an environment with the repository root added to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    root_str = str(ROOT)
    env["PYTHONPATH"] = root_str if not existing else os.pathsep.join([root_str, existing])
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local benchmark pipeline end-to-end."
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Run pytest after the benchmark pipeline completes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env()

    steps = list(PIPELINE_STEPS)
    if args.with_tests:
        steps.append(TEST_STEP)

    total = len(steps)
    for index, (title, command_args) in enumerate(steps, start=1):
        run_step(index, total, title, command_args, env)

    print("=" * 72)
    print("Pipeline completed successfully.")
    print("Generated/validated outputs include:")
    print("  data/processed/results_llm_reviewed.json")
    print("  data/processed/results_llm_reviewed.csv")
    print("  data/processed/criterion_level_results.csv")
    print("  data/processed/error_analysis_llm_reviewed.json")
    print("  data/processed/error_analysis_llm_reviewed.csv")
    print("  reports/benchmark_report.html")
    print("=" * 72)


if __name__ == "__main__":
    main()
