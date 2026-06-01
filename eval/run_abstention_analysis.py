"""
eval/run_abstention_analysis.py

Task 53 — Abstention analysis.

Treats predicted_label == "unclear" as abstention.
Reads  data/processed/results_llm_reviewed.json
Writes reports/abstention_analysis.md

Usage:
    PYTHONPATH=. python eval/run_abstention_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any

RESULTS_PATH = "data/processed/results_llm_reviewed.json"
OUTPUT_PATH = "reports/abstention_analysis.md"

ANSWERED_LABELS = {"eligible", "not_eligible"}
ABSTAIN_LABEL = "unclear"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_results(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(results: Any) -> list[dict]:
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in ("predictions", "results", "cases"):
            if key in results and isinstance(results[key], list):
                return results[key]
        flat = []
        for v in results.values():
            if isinstance(v, list):
                flat.extend(v)
            elif isinstance(v, dict):
                flat.append(v)
        return flat
    return []


def is_abstained(record: dict) -> bool:
    pred = record.get("predicted_label") or record.get("prediction") or record.get("predicted") or ""
    return str(pred).strip().lower() == ABSTAIN_LABEL


def compute_abstention_metrics(predictions: list[dict]) -> dict[str, Any]:
    total = len(predictions)
    answered: list[dict] = []
    abstained: list[dict] = []

    for rec in predictions:
        if is_abstained(rec):
            abstained.append(rec)
        else:
            answered.append(rec)

    coverage = len(answered) / total if total else 0.0
    abstention_rate = len(abstained) / total if total else 0.0

    # Accuracy on answered only
    answered_correct = 0
    answered_errors: list[dict] = []
    unsafe_answered_errors: list[dict] = []
    gold_dist_answered: dict[str, int] = defaultdict(int)
    pred_dist_answered: dict[str, int] = defaultdict(int)

    for rec in answered:
        gold = _gold(rec)
        pred = _pred(rec)
        conf = rec.get("confidence", "")
        gold_dist_answered[gold] += 1
        pred_dist_answered[pred] += 1
        if gold == pred:
            answered_correct += 1
        else:
            entry = {
                "patient_id": rec.get("patient_id", ""),
                "trial_id": rec.get("trial_id", ""),
                "gold_label": gold,
                "predicted_label": pred,
                "confidence": conf,
            }
            answered_errors.append(entry)
            if gold == "not_eligible" and pred == "eligible":
                unsafe_answered_errors.append(entry)

    accuracy_on_answered = answered_correct / len(answered) if answered else 0.0
    overall_accuracy = answered_correct / total if total else 0.0

    # Abstention breakdown
    over_conservative: list[dict] = []
    correct_abstentions: list[dict] = []
    gold_dist_abstained: dict[str, int] = defaultdict(int)
    abstention_examples: list[dict] = []

    for rec in abstained:
        gold = _gold(rec)
        conf = rec.get("confidence", "")
        gold_dist_abstained[gold] += 1
        entry = {
            "patient_id": rec.get("patient_id", ""),
            "trial_id": rec.get("trial_id", ""),
            "gold_label": gold,
            "confidence": conf,
        }
        abstention_examples.append(entry)
        if gold in ANSWERED_LABELS:
            over_conservative.append(entry)
        if gold == ABSTAIN_LABEL:
            correct_abstentions.append(entry)

    return {
        "total": total,
        "answered": len(answered),
        "abstained": len(abstained),
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "accuracy_on_answered": round(accuracy_on_answered, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "answered_errors": len(answered_errors),
        "unsafe_answered_errors": len(unsafe_answered_errors),
        "over_conservative_abstentions": len(over_conservative),
        "correct_abstentions": len(correct_abstentions),
        "gold_distribution_answered": dict(gold_dist_answered),
        "predicted_distribution_answered": dict(pred_dist_answered),
        "gold_distribution_abstained": dict(gold_dist_abstained),
        "top_answered_errors": answered_errors[:10],
        "top_abstention_examples": abstention_examples[:10],
    }


def _gold(rec: dict) -> str:
    return str(rec.get("gold_label") or rec.get("label") or rec.get("expected_label") or "unknown").strip().lower()


def _pred(rec: dict) -> str:
    return str(rec.get("predicted_label") or rec.get("prediction") or rec.get("predicted") or "unknown").strip().lower()


def format_markdown_report(s: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Abstention Analysis Report\n")
    lines.append("> Abstention = `predicted_label == \"unclear\"`\n")

    lines.append("## Overall Counts\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total records | {s['total']} |")
    lines.append(f"| Answered records | {s['answered']} |")
    lines.append(f"| Abstained records | {s['abstained']} |")
    lines.append(f"| Coverage | {s['coverage']:.2%} |")
    lines.append(f"| Abstention rate | {s['abstention_rate']:.2%} |")
    lines.append("")

    lines.append("## Accuracy\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Accuracy on answered only | {s['accuracy_on_answered']:.2%} |")
    lines.append(f"| Overall accuracy (abstentions = incorrect) | {s['overall_accuracy']:.2%} |")
    lines.append("")

    lines.append("## Error Breakdown\n")
    lines.append(f"| Type | Count |")
    lines.append(f"|------|------:|")
    lines.append(f"| Answered errors | {s['answered_errors']} |")
    lines.append(f"| Unsafe answered errors (gold=not_eligible, pred=eligible) | {s['unsafe_answered_errors']} |")
    lines.append(f"| Over-conservative abstentions (gold=eligible/not_eligible, pred=unclear) | {s['over_conservative_abstentions']} |")
    lines.append(f"| Correct abstentions (gold=unclear, pred=unclear) | {s['correct_abstentions']} |")
    lines.append("")

    lines.append("## Gold Label Distribution — Answered Records\n")
    for lbl, cnt in sorted(s["gold_distribution_answered"].items()):
        lines.append(f"- {lbl}: {cnt}")
    lines.append("")

    lines.append("## Predicted Label Distribution — Answered Records\n")
    for lbl, cnt in sorted(s["predicted_distribution_answered"].items()):
        lines.append(f"- {lbl}: {cnt}")
    lines.append("")

    lines.append("## Gold Label Distribution — Abstained Records\n")
    for lbl, cnt in sorted(s["gold_distribution_abstained"].items()):
        lines.append(f"- {lbl}: {cnt}")
    lines.append("")

    if s["top_answered_errors"]:
        lines.append("## Top Answered Errors (up to 10)\n")
        lines.append("| patient_id | trial_id | gold | predicted | confidence |")
        lines.append("|------------|----------|------|-----------|------------|")
        for e in s["top_answered_errors"]:
            lines.append(f"| {e['patient_id']} | {e['trial_id']} | {e['gold_label']} | {e['predicted_label']} | {e['confidence']} |")
        lines.append("")

    if s["top_abstention_examples"]:
        lines.append("## Top Abstention Examples (up to 10)\n")
        lines.append("| patient_id | trial_id | gold | confidence |")
        lines.append("|------------|----------|------|------------|")
        for e in s["top_abstention_examples"]:
            lines.append(f"| {e['patient_id']} | {e['trial_id']} | {e['gold_label']} | {e['confidence']} |")
        lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    raw = load_results(RESULTS_PATH)
    predictions = extract_predictions(raw)

    if not predictions:
        print("WARNING: No prediction records found.", file=sys.stderr)

    summary = compute_abstention_metrics(predictions)
    report = format_markdown_report(summary)
    write_text(report, OUTPUT_PATH)

    print(f"Abstention analysis written to: {OUTPUT_PATH}")
    print(f"Total: {summary['total']} | Answered: {summary['answered']} | Abstained: {summary['abstained']}")
    print(f"Coverage: {summary['coverage']:.2%} | Accuracy on answered: {summary['accuracy_on_answered']:.2%}")
    print(f"Unsafe answered errors: {summary['unsafe_answered_errors']}")
    print(f"Over-conservative abstentions: {summary['over_conservative_abstentions']}")
    print(f"Correct abstentions: {summary['correct_abstentions']}")


if __name__ == "__main__":
    main()
