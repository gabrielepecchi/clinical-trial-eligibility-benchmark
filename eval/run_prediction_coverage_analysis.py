"""
eval/run_prediction_coverage_analysis.py

Task 64 — Prediction coverage / completeness analysis.

Checks which patient_id + trial_id pairs in the gold label set are missing
from predictions, and which prediction pairs have no gold label.

Usage:
    PYTHONPATH=. python eval/run_prediction_coverage_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LABELS_PATH = "data/processed/labels_llm_reviewed.json"
DEFAULT_RESULTS_PATH = "data/processed/results_llm_reviewed.json"
DEFAULT_REPORT_PATH = "reports/prediction_coverage_analysis.md"

LABEL_CANDIDATE_KEYS = ("labels", "records", "pairs")
RESULT_CANDIDATE_KEYS = ("predictions", "results", "records", "pairs")


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def load_json(path: str) -> Any:
    """Load and return JSON from *path*. Exits non-zero on missing or malformed file."""
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_records(data: Any, candidate_keys: Tuple[str, ...]) -> List[dict]:
    """
    Return a flat list of dicts from *data*.

    Handles:
    - list of dicts  → returned as-is
    - dict with a known candidate key whose value is a list → use that list
    - dict of dicts  → values flattened
    """
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in candidate_keys:
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
        # dict of dicts (keyed by some id)
        flat: List[dict] = []
        for v in data.values():
            if isinstance(v, dict):
                flat.append(v)
            elif isinstance(v, list):
                flat.extend(r for r in v if isinstance(r, dict))
        return flat
    return []


def pair_key(record: dict) -> Optional[Tuple[str, str]]:
    """Return (patient_id, trial_id) for a record, or None if either field is missing."""
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    if pid and tid:
        return (pid, tid)
    return None


def count_pairs(records: List[dict]) -> Dict[Tuple[str, str], int]:
    """Return a dict mapping each (patient_id, trial_id) key to its occurrence count."""
    counts: Dict[Tuple[str, str], int] = {}
    for record in records:
        key = pair_key(record)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return counts


def analyze_prediction_coverage(
    label_records: List[dict],
    prediction_records: List[dict],
) -> dict:
    """
    Compare gold label pairs to prediction pairs and return a coverage summary.

    Keys:
        gold_pair_count, prediction_pair_count, shared_pair_count,
        missing_prediction_count, extra_prediction_count,
        duplicate_gold_count, duplicate_prediction_count,
        coverage_pct, prediction_only_pct,
        missing_examples, extra_examples,
        duplicate_gold_examples, duplicate_prediction_examples
    """
    gold_counts = count_pairs(label_records)
    pred_counts = count_pairs(prediction_records)

    gold_keys = set(gold_counts.keys())
    pred_keys = set(pred_counts.keys())

    shared = gold_keys & pred_keys
    missing = gold_keys - pred_keys   # in gold but not in predictions
    extra = pred_keys - gold_keys     # in predictions but not in gold

    duplicate_gold = {k: v for k, v in gold_counts.items() if v > 1}
    duplicate_pred = {k: v for k, v in pred_counts.items() if v > 1}

    gold_count = len(gold_keys)
    pred_count = len(pred_keys)
    coverage_pct = (len(shared) / gold_count * 100) if gold_count > 0 else 0.0
    prediction_only_pct = (len(extra) / pred_count * 100) if pred_count > 0 else 0.0

    def _to_examples(key_set: set, limit: int = 20) -> List[dict]:
        return [
            {"patient_id": k[0], "trial_id": k[1]}
            for k in sorted(key_set)[:limit]
        ]

    def _dup_examples(dup_dict: dict, limit: int = 20) -> List[dict]:
        return [
            {"patient_id": k[0], "trial_id": k[1], "count": v}
            for k, v in sorted(dup_dict.items())[:limit]
        ]

    return {
        "gold_pair_count": gold_count,
        "prediction_pair_count": pred_count,
        "shared_pair_count": len(shared),
        "missing_prediction_count": len(missing),
        "extra_prediction_count": len(extra),
        "duplicate_gold_count": len(duplicate_gold),
        "duplicate_prediction_count": len(duplicate_pred),
        "coverage_pct": coverage_pct,
        "prediction_only_pct": prediction_only_pct,
        "missing_examples": _to_examples(missing),
        "extra_examples": _to_examples(extra),
        "duplicate_gold_examples": _dup_examples(duplicate_gold),
        "duplicate_prediction_examples": _dup_examples(duplicate_pred),
    }


def format_markdown_report(summary: dict) -> str:
    """Render the coverage summary as a Markdown string."""
    lines = [
        "# Prediction Coverage Analysis",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Gold label pairs | {summary['gold_pair_count']} |",
        f"| Prediction pairs | {summary['prediction_pair_count']} |",
        f"| Shared pairs | {summary['shared_pair_count']} |",
        f"| Missing predictions (gold only) | {summary['missing_prediction_count']} |",
        f"| Extra predictions (pred only) | {summary['extra_prediction_count']} |",
        f"| Duplicate gold pairs | {summary['duplicate_gold_count']} |",
        f"| Duplicate prediction pairs | {summary['duplicate_prediction_count']} |",
        f"| Coverage % (shared / gold) | {summary['coverage_pct']:.1f}% |",
        f"| Prediction-only % (extra / pred) | {summary['prediction_only_pct']:.1f}% |",
        "",
    ]

    def _append_examples(title: str, examples: List[dict], cols: List[str]) -> None:
        if not examples:
            lines.append(f"## {title}")
            lines.append("")
            lines.append("_None._")
            lines.append("")
            return
        lines.append(f"## {title} (up to 20)")
        lines.append("")
        header = " | ".join(cols)
        sep = " | ".join("---" for _ in cols)
        lines.append(f"| {header} |")
        lines.append(f"| {sep} |")
        for ex in examples:
            row = " | ".join(str(ex.get(c, "")) for c in cols)
            lines.append(f"| {row} |")
        lines.append("")

    _append_examples(
        "Missing Predictions (gold pair, no prediction)",
        summary["missing_examples"],
        ["patient_id", "trial_id"],
    )
    _append_examples(
        "Extra Predictions (prediction pair, no gold label)",
        summary["extra_examples"],
        ["patient_id", "trial_id"],
    )
    _append_examples(
        "Duplicate Gold Pairs",
        summary["duplicate_gold_examples"],
        ["patient_id", "trial_id", "count"],
    )
    _append_examples(
        "Duplicate Prediction Pairs",
        summary["duplicate_prediction_examples"],
        ["patient_id", "trial_id", "count"],
    )

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    labels_data = load_json(DEFAULT_LABELS_PATH)
    results_data = load_json(DEFAULT_RESULTS_PATH)

    label_records = extract_records(labels_data, LABEL_CANDIDATE_KEYS)
    prediction_records = extract_records(results_data, RESULT_CANDIDATE_KEYS)

    summary = analyze_prediction_coverage(label_records, prediction_records)
    report_text = format_markdown_report(summary)
    write_text(report_text, DEFAULT_REPORT_PATH)

    print(f"Gold pairs        : {summary['gold_pair_count']}")
    print(f"Prediction pairs  : {summary['prediction_pair_count']}")
    print(f"Shared pairs      : {summary['shared_pair_count']}")
    print(f"Missing preds     : {summary['missing_prediction_count']}")
    print(f"Extra preds       : {summary['extra_prediction_count']}")
    print(f"Coverage          : {summary['coverage_pct']:.1f}%")
    print(f"Report written to : {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
