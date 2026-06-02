"""
run_difficulty_label_audit.py — Task 20 audit: Difficulty label presence check.

Checks whether benchmark records currently contain difficulty labels, and
reports whether Task 20 (metrics by difficulty) can be completed with the
current dataset.

Usage:
    PYTHONPATH=. python eval/run_difficulty_label_audit.py
    PYTHONPATH=. python eval/run_difficulty_label_audit.py --output PATH
"""

import json
import os
import sys
import argparse
from collections import Counter

DEFAULT_OUTPUT = "reports/difficulty_label_audit.json"

DIFFICULTY_FIELDS = ["difficulty", "difficulty_label", "difficulty_level", "case_difficulty"]

INPUT_FILES = [
    "data/processed/labels_llm_reviewed.json",
    "data/processed/results_llm_reviewed.json",
    "examples/golden_cases.json",
]

MIN_COVERAGE_FOR_METRICS = 0.80


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_safe(path: str) -> object:
    """Load JSON if present; return None if missing."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_records(data) -> list:
    """Flatten data to a list of dicts."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("predictions", "labels", "records", "cases"):
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# Difficulty detection
# ---------------------------------------------------------------------------

def get_difficulty(record: dict) -> str:
    """Return the difficulty value from a record, or empty string if absent."""
    for field in DIFFICULTY_FIELDS:
        val = record.get(field)
        if val is not None:
            s = str(val).strip()
            if s:
                return s
        # Check nested metadata
        meta = record.get("metadata")
        if isinstance(meta, dict):
            val = meta.get(field)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
    return ""


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit_records(all_records: list) -> dict:
    total = len(all_records)
    difficulty_values = [get_difficulty(r) for r in all_records]

    with_difficulty = [v for v in difficulty_values if v]
    without_difficulty = [v for v in difficulty_values if not v]

    counts = Counter(with_difficulty)
    coverage = len(with_difficulty) / total if total > 0 else 0.0
    has_enough = coverage >= MIN_COVERAGE_FOR_METRICS

    if total == 0:
        recommendation = (
            "No records found across checked files. "
            "Task 20 cannot be completed until records are present."
        )
    elif not with_difficulty:
        recommendation = (
            "No difficulty labels found in any checked records. "
            "Task 20 (metrics by difficulty) requires difficulty labels to be added to the dataset "
            "before it can be completed."
        )
    elif not has_enough:
        recommendation = (
            f"Only {len(with_difficulty)}/{total} records ({coverage:.0%}) have difficulty labels. "
            f"At least {MIN_COVERAGE_FOR_METRICS:.0%} coverage is required for meaningful metrics. "
            "Add difficulty labels to the remaining records before completing Task 20."
        )
    else:
        recommendation = (
            f"{len(with_difficulty)}/{total} records ({coverage:.0%}) have difficulty labels. "
            "Task 20 can be completed with the current dataset."
        )

    return {
        "total_records_checked": total,
        "records_with_difficulty": len(with_difficulty),
        "records_missing_difficulty": len(without_difficulty),
        "difficulty_counts": dict(counts),
        "coverage": round(coverage, 4),
        "has_enough_difficulty_labels_for_metrics": has_enough,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Difficulty label audit.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    all_records = []
    files_checked = []
    files_missing = []

    for path in INPUT_FILES:
        data = load_json_safe(path)
        if data is None:
            files_missing.append(path)
            continue
        records = extract_records(data)
        all_records.extend(records)
        files_checked.append({"path": path, "records_loaded": len(records)})

    audit = audit_records(all_records)

    report = {
        "files_checked": files_checked,
        "files_missing": files_missing,
        **audit,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Files checked            : {len(files_checked)}")
    if files_missing:
        print(f"Files not found          : {', '.join(files_missing)}")
    print(f"Total records checked    : {audit['total_records_checked']}")
    print(f"Records with difficulty  : {audit['records_with_difficulty']}")
    print(f"Records missing difficulty: {audit['records_missing_difficulty']}")
    if audit["difficulty_counts"]:
        print(f"Difficulty counts        : {audit['difficulty_counts']}")
    print(f"Enough for metrics       : {audit['has_enough_difficulty_labels_for_metrics']}")
    print(f"Recommendation           : {audit['recommendation']}")
    print(f"Report written           : {args.output}")


if __name__ == "__main__":
    main()
