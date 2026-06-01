"""
eval/run_per_patient_summary.py

Task 42 — Per-patient criterion-level summary.

Reads  data/processed/criterion_level_results.csv
Reads  data/processed/results_llm_reviewed.json  (optional)
Writes reports/per_patient_criterion_summary.md

Usage:
    PYTHONPATH=. python eval/run_per_patient_summary.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any

CRITERION_CSV = "data/processed/criterion_level_results.csv"
RESULTS_JSON = "data/processed/results_llm_reviewed.json"
OUTPUT_PATH = "reports/per_patient_criterion_summary.md"

# ---------------------------------------------------------------------------
# Keyword sets for pattern detection
# ---------------------------------------------------------------------------

_EXCLUSION_WORDS = {"exclusion", "excluded", "exclude", "not eligible", "ineligible", "disqualif"}
_UNCERTAIN_WORDS = {"uncertain", "unclear", "unknown", "missing", "not documented", "not reported", "ambiguous"}
_BLOCKING_WORDS = {"blocking", "not_met", "failed", "not met", "fail", "unmet", "block"}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_csv_rows(path: str) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f"WARNING: No rows found in {path}.", file=sys.stderr)
        return rows
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Could not read {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def load_json(path: str, required: bool = False) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    except json.JSONDecodeError as exc:
        if required:
            print(f"ERROR: Malformed JSON in {path}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: Could not parse {path}: {exc}", file=sys.stderr)
        return None


def extract_predictions(data: Any) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("predictions", "results", "cases"):
            if key in data and isinstance(data[key], list):
                return data[key]
        flat = []
        for v in data.values():
            if isinstance(v, list):
                flat.extend(v)
            elif isinstance(v, dict):
                flat.append(v)
        return flat
    return []


def find_first_existing_column(rows: list[dict], candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in any row."""
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def normalize_text_pattern(text: str) -> str:
    """Return a lowercased, stripped version of text for pattern matching."""
    return str(text).strip().lower()


def is_exclusion_like(row: dict) -> bool:
    combined = " ".join(normalize_text_pattern(v) for v in row.values())
    return any(w in combined for w in _EXCLUSION_WORDS)


def is_uncertain_like(row: dict) -> bool:
    combined = " ".join(normalize_text_pattern(v) for v in row.values())
    return any(w in combined for w in _UNCERTAIN_WORDS)


def is_blocking_like(row: dict) -> bool:
    combined = " ".join(normalize_text_pattern(v) for v in row.values())
    return any(w in combined for w in _BLOCKING_WORDS)


def summarize_patient_criteria(
    patient_id: str,
    rows: list[dict],
    prediction_rows: list[dict] | None = None,
) -> dict[str, Any]:
    total = len(rows)
    trials: set[str] = set()
    status_dist: dict[str, int] = defaultdict(int)
    criterion_type_dist: dict[str, int] = defaultdict(int)
    classified_type_dist: dict[str, int] = defaultdict(int)
    exclusion_count = 0
    uncertain_count = 0
    blocking_count = 0
    reason_patterns: dict[str, int] = defaultdict(int)

    status_col = find_first_existing_column(rows, ["decision", "status", "result", "label"])
    ctype_col = find_first_existing_column(rows, ["criterion_type", "type"])
    classified_col = find_first_existing_column(rows, ["classified_criterion_type", "classified_type"])
    trial_col = find_first_existing_column(rows, ["trial_id", "trial"])
    reason_col = find_first_existing_column(rows, ["reason", "explanation", "notes"])

    for row in rows:
        if trial_col and row.get(trial_col):
            trials.add(row[trial_col])
        if status_col and row.get(status_col):
            status_dist[normalize_text_pattern(row[status_col])] += 1
        if ctype_col and row.get(ctype_col):
            criterion_type_dist[normalize_text_pattern(row[ctype_col])] += 1
        if classified_col and row.get(classified_col):
            classified_type_dist[normalize_text_pattern(row[classified_col])] += 1
        if is_exclusion_like(row):
            exclusion_count += 1
        if is_uncertain_like(row):
            uncertain_count += 1
        if is_blocking_like(row):
            blocking_count += 1
        if reason_col and row.get(reason_col):
            snippet = normalize_text_pattern(row[reason_col])[:60]
            reason_patterns[snippet] += 1

    top_reasons = sorted(reason_patterns.items(), key=lambda x: -x[1])[:5]

    pred_summary: dict[str, Any] = {}
    if prediction_rows:
        patient_preds = [
            r for r in prediction_rows
            if str(r.get("patient_id", "")).strip() == patient_id
        ]
        total_preds = len(patient_preds)
        correct = sum(
            1 for r in patient_preds
            if _gold(r) == _pred(r) and _gold(r) != "unknown"
        )
        errors = total_preds - correct
        pred_summary = {
            "prediction_pairs": total_preds,
            "correct": correct,
            "errors": errors,
            "accuracy": round(correct / total_preds, 4) if total_preds else None,
        }

    return {
        "patient_id": patient_id,
        "total_criterion_rows": total,
        "trials_represented": sorted(trials),
        "trial_count": len(trials),
        "status_distribution": dict(status_dist),
        "criterion_type_distribution": dict(criterion_type_dist),
        "classified_criterion_type_distribution": dict(classified_type_dist),
        "exclusion_like_count": exclusion_count,
        "uncertain_like_count": uncertain_count,
        "blocking_like_count": blocking_count,
        "top_reason_patterns": [{"pattern": p, "count": c} for p, c in top_reasons],
        **pred_summary,
    }


def _gold(rec: dict) -> str:
    return str(rec.get("gold_label") or rec.get("label") or rec.get("expected_label") or "unknown").strip().lower()


def _pred(rec: dict) -> str:
    return str(rec.get("predicted_label") or rec.get("prediction") or rec.get("predicted") or "unknown").strip().lower()


def analyze_per_patient_summary(
    criterion_rows: list[dict],
    predictions: list[dict] | None = None,
) -> dict[str, Any]:
    pid_col = find_first_existing_column(criterion_rows, ["patient_id", "patient"])
    if not pid_col:
        print("ERROR: No patient_id column found in criterion_level_results.csv.", file=sys.stderr)
        sys.exit(1)

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in criterion_rows:
        pid = str(row.get(pid_col, "")).strip()
        if pid:
            by_patient[pid].append(row)

    patient_summaries = [
        summarize_patient_criteria(pid, rows, predictions)
        for pid, rows in sorted(by_patient.items())
    ]

    return {
        "total_patients": len(patient_summaries),
        "total_criterion_rows": len(criterion_rows),
        "patient_summaries": patient_summaries,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _top10(summaries: list[dict], key: str, label: str) -> list[str]:
    lines = [f"## Top 10 Patients by {label}\n"]
    ranked = sorted(summaries, key=lambda s: s.get(key, 0), reverse=True)[:10]
    lines.append(f"| patient_id | {label} |")
    lines.append(f"|------------|{'-' * (len(label) + 2)}|")
    for s in ranked:
        lines.append(f"| {s['patient_id']} | {s.get(key, 0)} |")
    lines.append("")
    return lines


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Per-Patient Criterion-Level Summary\n")
    lines.append(f"- **Total patients:** {summary['total_patients']}")
    lines.append(f"- **Total criterion rows:** {summary['total_criterion_rows']}")
    lines.append("")

    ps = summary["patient_summaries"]

    lines.extend(_top10(ps, "total_criterion_rows", "Criterion Row Count"))
    lines.extend(_top10(ps, "uncertain_like_count", "Uncertain-like Criterion Count"))
    lines.extend(_top10(ps, "blocking_like_count", "Blocking/Not-Met Criterion Count"))

    # Full per-patient table
    has_preds = any("prediction_pairs" in s for s in ps)
    lines.append("## Full Per-Patient Summary\n")
    header = "| patient_id | trials | rows | exclusion | uncertain | blocking |"
    sep    = "|------------|-------:|-----:|----------:|----------:|---------:|"
    if has_preds:
        header += " preds | correct | errors | accuracy |"
        sep    += "------:|--------:|-------:|---------:|"
    lines.append(header)
    lines.append(sep)

    for s in ps:
        row = (
            f"| {s['patient_id']} "
            f"| {s['trial_count']} "
            f"| {s['total_criterion_rows']} "
            f"| {s['exclusion_like_count']} "
            f"| {s['uncertain_like_count']} "
            f"| {s['blocking_like_count']} |"
        )
        if has_preds:
            pp = s.get("prediction_pairs", 0)
            cor = s.get("correct", 0)
            err = s.get("errors", 0)
            acc = s.get("accuracy")
            acc_str = f"{acc:.2%}" if acc is not None else "—"
            row += f" {pp} | {cor} | {err} | {acc_str} |"
        lines.append(row)
    lines.append("")

    # Short examples: top 5 highest-risk patients (most blocking + uncertain)
    risk_ranked = sorted(
        ps,
        key=lambda s: s.get("blocking_like_count", 0) + s.get("uncertain_like_count", 0),
        reverse=True,
    )[:5]
    if risk_ranked:
        lines.append("## Highest-Risk Patient Examples\n")
        for s in risk_ranked:
            lines.append(f"### {s['patient_id']}\n")
            lines.append(f"- Trials: {', '.join(s['trials_represented']) or '—'}")
            lines.append(f"- Criterion rows: {s['total_criterion_rows']}")
            lines.append(f"- Blocking-like: {s['blocking_like_count']}")
            lines.append(f"- Uncertain-like: {s['uncertain_like_count']}")
            lines.append(f"- Exclusion-like: {s['exclusion_like_count']}")
            if s.get("top_reason_patterns"):
                lines.append("- Top reason patterns:")
                for rp in s["top_reason_patterns"]:
                    lines.append(f"  - ({rp['count']}x) {rp['pattern']}")
            if "accuracy" in s and s["accuracy"] is not None:
                lines.append(f"- Prediction accuracy: {s['accuracy']:.2%} ({s['correct']}/{s['prediction_pairs']})")
            lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    criterion_rows = load_csv_rows(CRITERION_CSV)
    print(f"Criterion rows read: {len(criterion_rows)}")

    results_data = load_json(RESULTS_JSON, required=False)
    predictions = extract_predictions(results_data) if results_data else []

    summary = analyze_per_patient_summary(criterion_rows, predictions or None)
    report = format_markdown_report(summary)
    write_text(report, OUTPUT_PATH)

    print(f"Patients summarized: {summary['total_patients']}")
    print(f"Report written to  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
