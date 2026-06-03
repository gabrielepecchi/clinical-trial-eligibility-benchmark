"""
Task 39 (partial): Dataset size report.

Reads existing label/result files and reports current benchmark pair count
against the 300 and 500 pair targets. Does not expand the dataset.

Usage:
    PYTHONPATH=. python eval/run_dataset_size_report.py
    PYTHONPATH=. python eval/run_dataset_size_report.py \
        --output reports/dataset_size_report.json
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

DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_SPLITS = "data/processed/labels_llm_reviewed_with_splits.json"
DEFAULT_OUTPUT = "reports/dataset_size_report.json"

TARGET_300 = 300
TARGET_500 = 500


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def load_json(path: str) -> Any | None:
    """Return parsed JSON from *path*, or None if missing or malformed."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def count_records(data: Any) -> int:
    """Return the number of top-level records in *data*."""
    if data is None:
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("labels", "predictions", "results", "pairs", "records", "cases"):
            if key in data and isinstance(data[key], list):
                return len(data[key])
        # dict of records
        return sum(1 for v in data.values() if isinstance(v, dict))
    return 0


def build_report(
    labels_count: int,
    results_count: int,
    splits_count: int,
) -> dict[str, Any]:
    """Build the full JSON report."""
    current = labels_count if labels_count > 0 else results_count

    remaining_300 = max(0, TARGET_300 - current)
    remaining_500 = max(0, TARGET_500 - current)
    target_300_reached = current >= TARGET_300
    target_500_reached = current >= TARGET_500

    if target_500_reached:
        recommendation = (
            f"Current pair count ({current}) meets the 500-pair target. "
            "Task 39 dataset expansion is complete."
        )
    elif target_300_reached:
        recommendation = (
            f"Current pair count ({current}) meets the 300-pair target "
            f"but not the 500-pair target. "
            f"Task 39 needs {remaining_500} more pairs to reach 500."
        )
    else:
        recommendation = (
            f"Current pair count ({current}) does not meet the 300-pair target. "
            f"Task 39 needs {remaining_300} more pairs to reach 300, "
            f"and {remaining_500} more pairs to reach 500. "
            "Dataset expansion is required."
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "labels_count": labels_count,
        "results_prediction_count": results_count,
        "split_labeled_count": splits_count,
        "current_pair_count": current,
        "target_300_reached": target_300_reached,
        "target_500_reached": target_500_reached,
        "remaining_to_300": remaining_300,
        "remaining_to_500": remaining_500,
        "recommendation": recommendation,
    }


def write_json(data: dict[str, Any], path: str) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def print_terminal_summary(report: dict[str, Any]) -> None:
    """Print a compact summary to stdout."""
    t300 = "YES" if report["target_300_reached"] else "NO"
    t500 = "YES" if report["target_500_reached"] else "NO"
    print(f"  Labels count          : {report['labels_count']}")
    print(f"  Results count         : {report['results_prediction_count']}")
    print(f"  Split-labeled count   : {report['split_labeled_count']}")
    print(f"  Current pair count    : {report['current_pair_count']}")
    print(f"  300-pair target reached: {t300}  (remaining: {report['remaining_to_300']})")
    print(f"  500-pair target reached: {t500}  (remaining: {report['remaining_to_500']})")
    print(f"  Recommendation: {report['recommendation']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset size report (Task 39 partial)."
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--splits", default=DEFAULT_SPLITS,
        help=f"Path to splits labels JSON (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    labels_data = load_json(args.labels)
    results_data = load_json(args.results)
    splits_data = load_json(args.splits)

    labels_count = count_records(labels_data)
    results_count = count_records(results_data)
    splits_count = count_records(splits_data)

    report = build_report(labels_count, results_count, splits_count)
    write_json(report, args.output)
    print_terminal_summary(report)
    print(f"\n  Report written to: {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
