"""
Task 58: Pipeline status report.

Checks whether key pipeline input/output files exist and writes a
compact status report to reports/pipeline_status_report.json.

Does not execute the pipeline, does not modify any source files,
and does not touch rule_matcher.py.

Usage:
    PYTHONPATH=. python eval/run_pipeline_status_report.py
    PYTHONPATH=. python eval/run_pipeline_status_report.py \
        --output reports/pipeline_status_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = "reports/pipeline_status_report.json"

PIPELINE_STEPS: list[dict[str, str]] = [
    {
        "step": 1,
        "label": "Raw trial data",
        "path": "data/raw/parkinson_trials_raw.json",
    },
    {
        "step": 2,
        "label": "Eligibility criteria",
        "path": "data/processed/eligibility_criteria.json",
    },
    {
        "step": 3,
        "label": "Trial cases",
        "path": "data/processed/trial_cases.json",
    },
    {
        "step": 4,
        "label": "Patient cases",
        "path": "data/processed/patient_cases.json",
    },
    {
        "step": 5,
        "label": "Reviewed labels",
        "path": "data/processed/labels_llm_reviewed.json",
    },
    {
        "step": 6,
        "label": "Benchmark results",
        "path": "data/processed/results_llm_reviewed.json",
    },
    {
        "step": 7,
        "label": "HTML report",
        "path": "reports/benchmark_report.html",
    },
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def check_step(step: dict[str, str]) -> dict[str, Any]:
    """Return a status record for one pipeline step."""
    path = step["path"]
    exists = os.path.isfile(path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = None
    return {
        "step": int(step["step"]),
        "label": step["label"],
        "path": path,
        "status": "OK" if exists else "MISSING",
        "exists": exists,
        "size_bytes": size_bytes,
    }


def check_all_steps(steps: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return status records for all pipeline steps."""
    return [check_step(s) for s in steps]


def build_report(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the full JSON report from step results."""
    total = len(step_results)
    completed = [s for s in step_results if s["exists"]]
    missing = [s for s in step_results if not s["exists"]]
    pipeline_ready = len(missing) == 0

    if pipeline_ready:
        recommendation = "All pipeline files are present. The benchmark appears complete."
    elif len(completed) == 0:
        recommendation = (
            "No pipeline files found. Run the full pipeline from the beginning: "
            "download_trials → extract_eligibility → select_trial_cases → "
            "generate_patients → generate_labels → run_llm_reviewed_benchmark."
        )
    else:
        missing_labels = [s["label"] for s in missing]
        recommendation = (
            f"Missing steps: {', '.join(missing_labels)}. "
            "Run the pipeline from the first missing step."
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_ready": pipeline_ready,
        "total_steps": total,
        "completed_steps": len(completed),
        "missing_steps": len(missing),
        "steps": step_results,
        "recommendation": recommendation,
    }


def write_json(data: dict[str, Any], path: str) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def print_terminal_summary(step_results: list[dict[str, Any]]) -> None:
    """Print a compact step-style summary to stdout."""
    total = len(step_results)
    for s in step_results:
        print(f"  [{s['step']}/{total}] {s['label']}: {s['status']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline status report (Task 58)."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    step_results = check_all_steps(PIPELINE_STEPS)
    report = build_report(step_results)

    write_json(report, args.output)
    print_terminal_summary(step_results)

    completed = report["completed_steps"]
    total = report["total_steps"]
    ready = report["pipeline_ready"]
    print(
        f"\n  Pipeline ready: {'YES' if ready else 'NO'} "
        f"({completed}/{total} steps complete)\n"
        f"  Report written to: {args.output}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
