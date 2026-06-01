"""
Task 35: Human-review friendly CSV export.

Reads prediction results and optional supporting files, then writes a
single CSV designed for manual triage and annotation.

Usage:
    PYTHONPATH=. python eval/human_review_csv.py
    PYTHONPATH=. python eval/human_review_csv.py \
        --results   data/processed/results_llm_reviewed.json \
        --errors    data/processed/error_analysis_llm_reviewed.json \
        --labels    data/processed/labels_llm_reviewed.json \
        --output    data/processed/human_review_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_ERRORS = "data/processed/error_analysis_llm_reviewed.json"
DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_OUTPUT = "data/processed/human_review_queue.csv"

CSV_COLUMNS = [
    "patient_id",
    "trial_id",
    "gold_label",
    "predicted_label",
    "correct",
    "confidence",
    "error_type",
    "severity",
    "label_status",
    "rationale_preview",
    "matcher_explanation_preview",
    "blocking_criteria_preview",
    "uncertain_criteria_preview",
    "review_priority",
    "reviewer_notes",
]

# error_type values that indicate an unsafe eligible prediction
UNSAFE_ELIGIBLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "unsafe_eligible",
        "unsafe_eligible_error",
        "unsafe_prediction",
        "false_eligible",
        "critical_eligible_error",
    }
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def load_json(path: str, required: bool = False) -> Any:
    """
    Load JSON from *path*.

    If *required* is True, exit non-zero on missing or malformed file.
    If *required* is False, return None silently on missing file and exit
    non-zero only on malformed JSON.
    """
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(results: Any) -> list[dict[str, Any]]:
    """
    Return a flat list of prediction records from *results*.

    Accepts:
    - a list of records directly
    - a dict with a 'predictions', 'results', or 'pairs' key holding a list
    - a dict mapping pair_id -> record
    """
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in ("predictions", "results", "pairs", "evaluations"):
            if key in results and isinstance(results[key], list):
                return results[key]
        # dict of {pair_id: record}
        candidates = [v for v in results.values() if isinstance(v, dict)]
        if candidates:
            return candidates
    print("ERROR: unrecognised structure in results file.", file=sys.stderr)
    sys.exit(1)


def _pair_key(record: dict[str, Any]) -> str:
    """Return a normalised lookup key for a record."""
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    pair = str(record.get("pair_id", "")).strip()
    if pair:
        return pair
    if pid and tid:
        return f"{pid}__{tid}"
    return pid or tid


def index_error_analysis(error_data: Any) -> dict[str, dict[str, Any]]:
    """
    Return a dict mapping pair key -> error record.

    Accepts the same flexible shapes as extract_predictions.
    Returns an empty dict if *error_data* is None or unrecognised.
    """
    if error_data is None:
        return {}
    records: list[dict[str, Any]] = []
    if isinstance(error_data, list):
        records = error_data
    elif isinstance(error_data, dict):
        for key in ("errors", "error_analysis", "predictions", "pairs"):
            if key in error_data and isinstance(error_data[key], list):
                records = error_data[key]
                break
        else:
            candidates = [v for v in error_data.values() if isinstance(v, dict)]
            records = candidates
    return {_pair_key(r): r for r in records if isinstance(r, dict)}


def index_labels(labels_data: Any) -> dict[str, dict[str, Any]]:
    """
    Return a dict mapping pair key -> label record.

    Returns an empty dict if *labels_data* is None or unrecognised.
    """
    if labels_data is None:
        return {}
    records: list[dict[str, Any]] = []
    if isinstance(labels_data, list):
        records = labels_data
    elif isinstance(labels_data, dict):
        for key in ("labels", "pairs", "records"):
            if key in labels_data and isinstance(labels_data[key], list):
                records = labels_data[key]
                break
        else:
            candidates = [v for v in labels_data.values() if isinstance(v, dict)]
            records = candidates
    return {_pair_key(r): r for r in records if isinstance(r, dict)}


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a short string preview of *value*, safe for a CSV cell."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "; ".join(str(item).strip() for item in value if item)
    elif isinstance(value, dict):
        text = "; ".join(
            f"{k}: {v}" for k, v in value.items() if v is not None
        )
    else:
        text = str(value)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def compute_review_priority(row: dict[str, Any]) -> str:
    """
    Return 'high', 'medium', or 'low' based on prediction fields.

    Rules applied in order — first match wins.
    """
    gold = str(row.get("gold_label", "")).strip().lower()
    predicted = str(row.get("predicted_label", "")).strip().lower()
    severity = str(row.get("severity", "")).strip().lower()
    error_type = str(row.get("error_type", "")).strip().lower()
    correct = row.get("correct")

    # --- high ---
    if gold == "not_eligible" and predicted == "eligible":
        return "high"
    if severity == "critical":
        return "high"
    if error_type in UNSAFE_ELIGIBLE_ERROR_TYPES:
        return "high"

    # --- medium ---
    if correct is False or str(correct).lower() in ("false", "0", "no"):
        return "medium"
    if predicted == "unclear" and gold in ("eligible", "not_eligible"):
        return "medium"
    if gold == "unclear" and predicted in ("eligible", "not_eligible"):
        return "medium"
    if severity == "major":
        return "medium"

    # --- low ---
    return "low"


def build_review_rows(
    predictions: list[dict[str, Any]],
    error_index: dict[str, dict[str, Any]],
    label_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combine prediction, error, and label data into one row per prediction.

    Returns a list of dicts keyed by CSV_COLUMNS.
    """
    rows: list[dict[str, Any]] = []

    for pred in predictions:
        key = _pair_key(pred)
        err = error_index.get(key, {})
        lbl = label_index.get(key, {})

        # core prediction fields
        patient_id = pred.get("patient_id", "")
        trial_id = pred.get("trial_id", "")
        gold_label = pred.get("gold_label", pred.get("gold", ""))
        predicted_label = pred.get(
            "predicted_label", pred.get("prediction", pred.get("predicted", ""))
        )
        correct = pred.get("correct", "")
        confidence = pred.get("confidence", pred.get("confidence_score", ""))

        # error fields (from error_analysis if present, else from prediction)
        error_type = err.get("error_type", pred.get("error_type", ""))
        severity = err.get("severity", pred.get("severity", ""))

        # label fields
        label_status = lbl.get("label_status", pred.get("label_status", ""))
        rationale_raw = (
            lbl.get("rationale")
            or lbl.get("explanation")
            or pred.get("rationale")
            or ""
        )

        # matcher explanation fields from prediction record
        explanation_raw = pred.get("explanation", pred.get("matcher_explanation", ""))
        blocking_raw = pred.get("blocking_criteria", "")
        uncertain_raw = pred.get("uncertain_criteria", "")

        row: dict[str, Any] = {
            "patient_id": patient_id,
            "trial_id": trial_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "correct": correct,
            "confidence": confidence,
            "error_type": error_type,
            "severity": severity,
            "label_status": label_status,
            "rationale_preview": preview_text(rationale_raw),
            "matcher_explanation_preview": preview_text(explanation_raw),
            "blocking_criteria_preview": preview_text(blocking_raw),
            "uncertain_criteria_preview": preview_text(uncertain_raw),
            "review_priority": "",   # filled below
            "reviewer_notes": "",
        }

        row["review_priority"] = compute_review_priority(row)
        rows.append(row)

    # Sort: high -> medium -> low, then by patient_id / trial_id
    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(
        key=lambda r: (
            priority_order.get(str(r["review_priority"]), 9),
            str(r.get("patient_id", "")),
            str(r.get("trial_id", "")),
        )
    )

    return rows


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Write *rows* to a CSV at *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate human-review CSV from benchmark results (Task 35)."
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--errors",
        default=DEFAULT_ERRORS,
        help=f"Path to error analysis JSON (default: {DEFAULT_ERRORS})",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    error_data = load_json(args.errors, required=False)
    labels_data = load_json(args.labels, required=False)

    predictions = extract_predictions(results_data)
    error_index = index_error_analysis(error_data)
    label_index = index_labels(labels_data)

    rows = build_review_rows(predictions, error_index, label_index)
    write_csv(rows, args.output)

    counts = {"high": 0, "medium": 0, "low": 0}
    for r in rows:
        p = str(r.get("review_priority", "low"))
        counts[p] = counts.get(p, 0) + 1

    print(
        f"Human review CSV written to: {args.output}\n"
        f"  Rows written  : {len(rows)}\n"
        f"  High priority : {counts['high']}\n"
        f"  Medium priority: {counts['medium']}\n"
        f"  Low priority  : {counts['low']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
