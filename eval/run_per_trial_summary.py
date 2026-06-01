"""
eval/run_per_trial_summary.py

Task 43 — Per-trial criterion-level summary.

Reads  data/processed/criterion_level_results.csv
Reads  data/processed/results_llm_reviewed.json  (optional)
Writes reports/per_trial_criterion_summary.md

Usage:
    PYTHONPATH=. python eval/run_per_trial_summary.py
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
OUTPUT_PATH = "reports/per_trial_criterion_summary.md"

_EXCLUSION_WORDS = {"exclusion", "excluded", "exclude", "not eligible", "ineligible", "disqualif"}
_UNCERTAIN_WORDS = {"uncertain", "unclear", "unknown", "missing", "not documented", "not reported", "ambiguous"}
_BLOCKING_WORDS = {"blocking", "not_met", "failed", "not met", "fail", "unmet", "block"}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_csv_rows(path: str) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
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
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def normalize_text_pattern(text: str) -> str:
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


def _gold(rec: dict) -> str:
    return str(rec.get("gold_label") or rec.get("label") or rec.get("expected_label") or "unknown").strip().lower()


def _pred(rec: dict) -> str:
    return str(rec.get("predicted_label") or rec.get("prediction") or rec.get("predicted") or "unknown").strip().lower()


def summarize_trial_criteria(
    trial_id: str,
    rows: list[dict],
    prediction_rows: list[dict] | None = None,
) -> dict[str, Any]:
    total = len(rows)
    patients: set[str] = set()
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
    patient_col = find_first_existing_column(rows, ["patient_id", "patient"])
    reason_col = find_first_existing_column(rows, ["reason", "explanation", "notes"])

    for row in rows:
        if patient_col and row.get(patient_col):
            patients.add(row[patient_col])
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
        trial_preds = [
            r for r in prediction_rows
            if str(r.get("trial_id", "")).strip() == trial_id
        ]
        total_preds = len(trial_preds)
        correct = sum(
            1 for r in trial_preds
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
        "trial_id": trial_id,
        "total_criterion_rows": total,
        "patients_represented": sorted(patients),
        "patient_count": len(patients),
        "status_distribution": dict(status_dist),
        "criterion_type_distribution": dict(criterion_type_dist),
        "classified_criterion_type_distribution": dict(classified_type_dist),
        "exclusion_like_count": exclusion_count,
        "uncertain_like_count": uncertain_count,
        "blocking_like_count": blocking_count,
        "top_reason_patterns": [{"pattern": p, "count": c} for p, c in top_reasons],
        **pred_summary,
    }


def analyze_per_trial_summary(
    criterion_rows: list[dict],
    predictions: list[dict] | None = None,
) -> dict[str, Any]:
    trial_col = find_first_existing_column(criterion_rows, ["trial_id", "trial"])
    if not trial_col:
        print("ERROR: No trial_id column found in criterion_level_results.csv.", file=sys.stderr)
        sys.exit(1)

    by_trial: dict[str, list[dict]] = defaultdict(list)
    for row in criterion_rows:
        tid = str(row.get(trial_col, "")).strip()
        if tid:
            by_trial[tid].append(row)

    trial_summaries = [
        summarize_trial_criteria(tid, rows, predictions)
        for tid, rows in sorted(by_trial.items())
    ]

    return {
        "total_trials": len(trial_summaries),
        "total_criterion_rows": len(criterion_rows),
        "trial_summaries": trial_summaries,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _top10(summaries: list[dict], key: str, label: str) -> list[str]:
    lines = [f"## Top 10 Trials by {label}\n"]
    ranked = sorted(summaries, key=lambda s: s.get(key, 0), reverse=True)[:10]
    lines.append(f"| trial_id | {label} |")
    lines.append(f"|----------|{'-' * (len(label) + 2)}|")
    for s in ranked:
        lines.append(f"| {s['trial_id']} | {s.get(key, 0)} |")
    lines.append("")
    return lines


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Per-Trial Criterion-Level Summary\n")
    lines.append(f"- **Total trials:** {summary['total_trials']}")
    lines.append(f"- **Total criterion rows:** {summary['total_criterion_rows']}")
    lines.append("")

    ts = summary["trial_summaries"]

    lines.extend(_top10(ts, "total_criterion_rows", "Criterion Row Count"))
    lines.extend(_top10(ts, "uncertain_like_count", "Uncertain-like Criterion Count"))
    lines.extend(_top10(ts, "blocking_like_count", "Blocking/Not-Met Criterion Count"))

    has_preds = any("prediction_pairs" in s for s in ts)
    lines.append("## Full Per-Trial Summary\n")
    header = "| trial_id | patients | rows | exclusion | uncertain | blocking |"
    sep    = "|----------|--------:|-----:|----------:|----------:|---------:|"
    if has_preds:
        header += " preds | correct | errors | accuracy |"
        sep    += "------:|--------:|-------:|---------:|"
    lines.append(header)
    lines.append(sep)

    for s in ts:
        row = (
            f"| {s['trial_id']} "
            f"| {s['patient_count']} "
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

    risk_ranked = sorted(
        ts,
        key=lambda s: s.get("blocking_like_count", 0) + s.get("uncertain_like_count", 0),
        reverse=True,
    )[:5]
    if risk_ranked:
        lines.append("## Highest-Risk Trial Examples\n")
        for s in risk_ranked:
            lines.append(f"### {s['trial_id']}\n")
            lines.append(f"- Patients: {', '.join(s['patients_represented']) or '—'}")
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

    summary = analyze_per_trial_summary(criterion_rows, predictions or None)
    report = format_markdown_report(summary)
    write_text(report, OUTPUT_PATH)

    print(f"Trials summarized  : {summary['total_trials']}")
    print(f"Report written to  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
