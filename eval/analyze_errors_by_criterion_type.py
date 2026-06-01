"""
analyze_errors_by_criterion_type.py — Task 15: Error pattern analysis by criterion type.

Joins criterion_type_classified.csv with results_llm_reviewed.json to report
where the matcher makes errors grouped by criterion type.

Usage:
    PYTHONPATH=. python eval/analyze_errors_by_criterion_type.py
    PYTHONPATH=. python eval/analyze_errors_by_criterion_type.py \\
        --criteria PATH --results PATH --output PATH
"""

import csv
import json
import os
import sys
import argparse
from collections import defaultdict, Counter

DEFAULT_CRITERIA = "data/processed/criterion_type_classified.csv"
DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/errors_by_criterion_type.md"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv_rows(path: str) -> list:
    """Load and return rows from a CSV file as list of dicts."""
    if not os.path.exists(path):
        print(
            f"ERROR: File not found: {path}\n"
            "To generate it, run:\n"
            "    PYTHONPATH=. python eval/classify_criterion_types.py",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except csv.Error as e:
        print(f"ERROR: CSV error in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return rows


def load_results(path: str) -> dict:
    """Load and return results JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Index and analysis
# ---------------------------------------------------------------------------

def index_predictions(results: dict) -> dict:
    """
    Return a dict keyed by (patient_id, trial_id) ->
    {"gold_label": str, "predicted_label": str, "is_error": bool}.
    """
    index = {}
    predictions = results.get("predictions", [])
    for pred in predictions:
        pid = pred.get("patient_id", "")
        tid = pred.get("trial_id", "")
        gold = pred.get("gold_label", "") or pred.get("label", "")
        predicted = pred.get("predicted_label", "") or pred.get("prediction", "")
        if pid and tid:
            index[(pid, tid)] = {
                "gold_label": gold,
                "predicted_label": predicted,
                "is_error": gold != predicted,
            }
    return index


def group_errors_by_criterion_type(criterion_rows: list, prediction_index: dict) -> dict:
    """
    Group criterion rows by classified_criterion_type and compute per-type
    error statistics based on prediction-level correctness.

    Returns a dict: {criterion_type -> summary_dict}
    """
    # Build per-criterion-type sets of pairs and error pairs
    type_pairs: dict[str, set] = defaultdict(set)
    type_error_pairs: dict[str, set] = defaultdict(set)
    type_row_count: dict[str, int] = defaultdict(int)

    for row in criterion_rows:
        pid = row.get("patient_id", "")
        tid = row.get("trial_id", "")
        ctype = row.get("classified_criterion_type", "other") or "other"
        key = (pid, tid)

        type_row_count[ctype] += 1
        type_pairs[ctype].add(key)

        pred = prediction_index.get(key)
        if pred and pred["is_error"]:
            type_error_pairs[ctype].add(key)

    # Build summary per type
    summary = {}
    for ctype in sorted(type_pairs.keys()):
        pairs = type_pairs[ctype]
        error_pairs = type_error_pairs[ctype]
        total_pairs = len(pairs)
        n_errors = len(error_pairs)
        error_rate = n_errors / total_pairs if total_pairs > 0 else 0.0

        gold_dist: Counter = Counter()
        pred_dist: Counter = Counter()
        examples = []

        for key in error_pairs:
            pred = prediction_index.get(key)
            if pred:
                gold_dist[pred["gold_label"]] += 1
                pred_dist[pred["predicted_label"]] += 1
                examples.append({
                    "patient_id": key[0],
                    "trial_id": key[1],
                    "gold_label": pred["gold_label"],
                    "predicted_label": pred["predicted_label"],
                })

        # Stable top-3 examples: sort by (patient_id, trial_id)
        examples_sorted = sorted(examples, key=lambda x: (x["patient_id"], x["trial_id"]))[:3]

        summary[ctype] = {
            "total_criterion_rows": type_row_count[ctype],
            "total_pairs": total_pairs,
            "error_pairs": n_errors,
            "error_rate": error_rate,
            "gold_distribution": dict(gold_dist),
            "predicted_distribution": dict(pred_dist),
            "top_examples": examples_sorted,
        }

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _dist_str(dist: dict) -> str:
    if not dist:
        return "—"
    return ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))


def format_markdown_report(summary: dict) -> str:
    """Format the per-criterion-type error summary as a Markdown report."""
    lines = [
        "# Error Analysis by Criterion Type",
        "",
        "This report shows matcher prediction errors grouped by criterion type.",
        "An error is a patient–trial pair where `gold_label != predicted_label`.",
        "",
        "---",
        "",
    ]

    # Overview table
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "| Criterion Type | Criterion Rows | Pairs | Error Pairs | Error Rate |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )
    for ctype, m in sorted(summary.items(), key=lambda x: -x[1]["error_pairs"]):
        lines.append(
            f"| {ctype} "
            f"| {m['total_criterion_rows']} "
            f"| {m['total_pairs']} "
            f"| {m['error_pairs']} "
            f"| {m['error_rate']:.1%} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-type detail
    lines.append("## Detail by Criterion Type")
    lines.append("")

    for ctype, m in sorted(summary.items(), key=lambda x: -x[1]["error_pairs"]):
        lines.append(f"### {ctype}")
        lines.append("")
        lines.append(f"- **Criterion rows**: {m['total_criterion_rows']}")
        lines.append(f"- **Unique patient–trial pairs**: {m['total_pairs']}")
        lines.append(f"- **Error pairs**: {m['error_pairs']}")
        lines.append(f"- **Error rate**: {m['error_rate']:.1%}")
        if m["error_pairs"] > 0:
            lines.append(f"- **Gold label distribution (errors)**: {_dist_str(m['gold_distribution'])}")
            lines.append(f"- **Predicted label distribution (errors)**: {_dist_str(m['predicted_distribution'])}")
            lines.append("")
            lines.append("**Top error examples:**")
            lines.append("")
            lines.append("| patient_id | trial_id | gold_label | predicted_label |")
            lines.append("|---|---|---|---|")
            for ex in m["top_examples"]:
                lines.append(
                    f"| {ex['patient_id']} | {ex['trial_id']} "
                    f"| {ex['gold_label']} | {ex['predicted_label']} |"
                )
        lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error pattern analysis by criterion type."
    )
    parser.add_argument(
        "--criteria", default=DEFAULT_CRITERIA,
        help=f"Classified criteria CSV (default: {DEFAULT_CRITERIA})"
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Results JSON (default: {DEFAULT_RESULTS})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    criterion_rows = load_csv_rows(args.criteria)
    results = load_results(args.results)
    prediction_index = index_predictions(results)
    summary = group_errors_by_criterion_type(criterion_rows, prediction_index)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(f"Criterion rows read      : {len(criterion_rows)}")
    print(f"Prediction records read  : {len(prediction_index)}")
    print(f"Criterion types found    : {', '.join(sorted(summary.keys()))}")
    print(f"Report written           : {args.output}")


if __name__ == "__main__":
    main()
