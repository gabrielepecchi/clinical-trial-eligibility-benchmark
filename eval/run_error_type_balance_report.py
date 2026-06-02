"""
run_error_type_balance_report.py — Task 41: Error type balance report.

Analyzes whether error types are balanced in the current benchmark outputs.
This is a descriptive report only — it does not rebalance the dataset.

Usage:
    PYTHONPATH=. python eval/run_error_type_balance_report.py
    PYTHONPATH=. python eval/run_error_type_balance_report.py \\
        --input PATH --output PATH
"""

import json
import os
import sys
import argparse

DEFAULT_INPUT = "data/processed/error_analysis_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/error_type_balance_report.json"

NOTE = (
    "This is a descriptive balance report only. "
    "It documents the distribution of error types in the current benchmark outputs. "
    "It does not automatically rebalance the dataset or modify any labels."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        print(
            f"Run the error analysis first:\n"
            f"    PYTHONPATH=. python eval/summarize_llm_reviewed_errors.py",
            file=sys.stderr,
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def extract_error_type_counts(data: object) -> dict:
    """
    Extract error_type counts from the error analysis JSON.
    Supports multiple common structures:
    - list of error records with an 'error_type' field
    - dict with 'errors' list
    - dict with 'error_type_counts' already computed
    - dict with 'errors_by_type' mapping
    """
    # Already computed
    if isinstance(data, dict):
        if "error_type_counts" in data and isinstance(data["error_type_counts"], dict):
            return dict(data["error_type_counts"])
        if "errors_by_type" in data and isinstance(data["errors_by_type"], dict):
            return {k: len(v) if isinstance(v, list) else int(v)
                    for k, v in data["errors_by_type"].items()}

    # List of error records
    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("errors", "records", "predictions"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break

    counts: dict[str, int] = {}
    for rec in records:
        et = rec.get("error_type", "")
        if not et:
            continue
        counts[et] = counts.get(et, 0) + 1

    return counts


def compute_balance_report(error_type_counts: dict) -> dict:
    """Compute balance statistics from error type counts."""
    if not error_type_counts:
        return {
            "total_errors": 0,
            "error_type_counts": {},
            "most_common_error_type": None,
            "most_common_count": 0,
            "least_common_error_type": None,
            "least_common_count": 0,
            "imbalance_ratio": None,
            "underrepresented_error_types": [],
            "note": NOTE,
        }

    total = sum(error_type_counts.values())
    sorted_types = sorted(error_type_counts.items(), key=lambda x: (-x[1], x[0]))

    most_type, most_count = sorted_types[0]
    least_type, least_count = sorted_types[-1]

    imbalance_ratio = round(most_count / least_count, 4) if least_count > 0 else None
    underrepresented = [t for t, c in error_type_counts.items() if c <= 2]

    return {
        "total_errors": total,
        "error_type_counts": dict(sorted_types),
        "most_common_error_type": most_type,
        "most_common_count": most_count,
        "least_common_error_type": least_type,
        "least_common_count": least_count,
        "imbalance_ratio": imbalance_ratio,
        "underrepresented_error_types": sorted(underrepresented),
        "note": NOTE,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Error type balance report.")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Error analysis JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data = load_json(args.input)
    error_type_counts = extract_error_type_counts(data)
    report = compute_balance_report(error_type_counts)
    write_json(report, args.output)

    print(f"Total errors             : {report['total_errors']}")
    print(f"Error types found        : {len(report['error_type_counts'])}")
    if report["most_common_error_type"]:
        print(f"Most common error type   : {report['most_common_error_type']} ({report['most_common_count']})")
        print(f"Least common error type  : {report['least_common_error_type']} ({report['least_common_count']})")
    if report["imbalance_ratio"] is not None:
        print(f"Imbalance ratio          : {report['imbalance_ratio']:.2f}x")
    if report["underrepresented_error_types"]:
        print(f"Underrepresented types   : {', '.join(report['underrepresented_error_types'])}")
    else:
        print("Underrepresented types   : none")
    print(f"Report written           : {args.output}")
    print(f"Note: {NOTE}")


if __name__ == "__main__":
    main()
