"""
per_trial_report.py — Task 56: per-trial performance report.

Usage:
    PYTHONPATH=. python eval/per_trial_report.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
REPORT_PATH = Path("reports/per_trial_report.md")

VALID_LABELS = ["eligible", "not_eligible", "unclear"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Field accessors
# ---------------------------------------------------------------------------

def extract_predictions(data: dict) -> list:
    return data.get("predictions", []) if isinstance(data, dict) else []


def get_gold_label(record: dict) -> str:
    return record.get("gold_label", "")


def get_predicted_label(record: dict) -> str:
    return record.get("predicted_label", "")


def parse_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
        return float(value)
    return None


def label_distribution(records: list, label_getter) -> dict[str, int]:
    dist: dict[str, int] = {lbl: 0 for lbl in VALID_LABELS}
    for rec in records:
        lbl = label_getter(rec)
        if lbl in dist:
            dist[lbl] += 1
    return dist


# ---------------------------------------------------------------------------
# Per-trial summary
# ---------------------------------------------------------------------------

def summarize_trial(trial_id: str, records: list) -> dict:
    total = len(records)
    correct = sum(get_gold_label(r) == get_predicted_label(r) for r in records)
    errors = total - correct
    accuracy = correct / total if total else 0.0

    gold_dist = label_distribution(records, get_gold_label)
    pred_dist = label_distribution(records, get_predicted_label)

    false_eligible = sum(
        get_gold_label(r) == "not_eligible" and get_predicted_label(r) == "eligible"
        for r in records
    )
    over_conservative = sum(
        get_gold_label(r) in ("eligible", "not_eligible") and get_predicted_label(r) == "unclear"
        for r in records
    )
    under_called_unclear = sum(
        get_gold_label(r) == "unclear" and get_predicted_label(r) in ("eligible", "not_eligible")
        for r in records
    )

    valid_conf = [
        parse_confidence(r.get("confidence"))
        for r in records
        if parse_confidence(r.get("confidence")) is not None
    ]
    mean_conf = sum(valid_conf) / len(valid_conf) if valid_conf else None

    error_examples = [
        {
            "patient_id": r.get("patient_id", ""),
            "gold_label": get_gold_label(r),
            "predicted_label": get_predicted_label(r),
            "confidence": parse_confidence(r.get("confidence")),
        }
        for r in records
        if get_gold_label(r) != get_predicted_label(r)
    ]

    return {
        "trial_id": trial_id,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "gold_distribution": gold_dist,
        "predicted_distribution": pred_dist,
        "false_eligible_count": false_eligible,
        "over_conservative_count": over_conservative,
        "under_called_unclear_count": under_called_unclear,
        "mean_confidence": mean_conf,
        "error_examples": error_examples,
    }


def analyze_per_trial(records: list) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for rec in records:
        if isinstance(rec, dict):
            tid = rec.get("trial_id", "")
            if tid:
                grouped[tid].append(rec)

    trial_summaries = [
        summarize_trial(tid, recs)
        for tid, recs in sorted(grouped.items())
    ]

    total_records = sum(s["total"] for s in trial_summaries)
    total_correct = sum(s["correct"] for s in trial_summaries)
    overall_accuracy = total_correct / total_records if total_records else 0.0

    return {
        "total_trials": len(trial_summaries),
        "total_records": total_records,
        "overall_accuracy": overall_accuracy,
        "trials": trial_summaries,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(value: float | None, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def format_markdown_report(summary: dict) -> str:
    trials: list = summary["trials"]
    lines = [
        "# Per-Trial Performance Report",
        "",
        f"- Total trials: {summary['total_trials']}",
        f"- Total records: {summary['total_records']}",
        f"- Overall accuracy: {_fmt(summary['overall_accuracy'])}",
        "",
    ]

    by_errors = sorted(trials, key=lambda x: x["errors"], reverse=True)[:10]
    lines += [
        "## Top 10 Trials by Error Count",
        "",
        "| trial_id | total | errors | accuracy |",
        "|---|---|---|---|",
    ]
    for t in by_errors:
        lines.append(f"| {t['trial_id']} | {t['total']} | {t['errors']} | {_fmt(t['accuracy'])} |")

    by_acc = sorted(
        [t for t in trials if t["total"] >= 2],
        key=lambda x: x["accuracy"]
    )[:10]
    lines += [
        "",
        "## Top 10 Trials by Lowest Accuracy (≥2 records)",
        "",
        "| trial_id | total | errors | accuracy |",
        "|---|---|---|---|",
    ]
    for t in by_acc:
        lines.append(f"| {t['trial_id']} | {t['total']} | {t['errors']} | {_fmt(t['accuracy'])} |")

    by_fe = sorted(trials, key=lambda x: x["false_eligible_count"], reverse=True)[:10]
    lines += [
        "",
        "## Top 10 Trials by High-Risk False Eligible Count",
        "",
        "| trial_id | false_eligible | total | accuracy |",
        "|---|---|---|---|",
    ]
    for t in by_fe:
        lines.append(
            f"| {t['trial_id']} | {t['false_eligible_count']} | {t['total']} | {_fmt(t['accuracy'])} |"
        )

    lines += [
        "",
        "## Full Per-Trial Summary",
        "",
        "| trial_id | total | correct | errors | accuracy | false_eligible | over_conservative | under_unclear | mean_conf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in trials:
        lines.append(
            f"| {t['trial_id']} | {t['total']} | {t['correct']} | {t['errors']} "
            f"| {_fmt(t['accuracy'])} | {t['false_eligible_count']} "
            f"| {t['over_conservative_count']} | {t['under_called_unclear_count']} "
            f"| {_fmt(t['mean_confidence'])} |"
        )

    lines += ["", "## Per-Trial Error Examples", ""]
    for t in trials:
        if not t["error_examples"]:
            continue
        lines.append(f"### {t['trial_id']}")
        lines += [
            "",
            "| patient_id | gold_label | predicted_label | confidence |",
            "|---|---|---|---|",
        ]
        for ex in t["error_examples"]:
            lines.append(
                f"| {ex['patient_id']} | {ex['gold_label']} | {ex['predicted_label']} "
                f"| {_fmt(ex['confidence'])} |"
            )
        lines.append("")

    lines += ["---", "_Generated by eval/per_trial_report.py_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        data = load_json(RESULTS_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {RESULTS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {RESULTS_PATH}: {exc}")
        sys.exit(1)

    records = extract_predictions(data)
    if not records:
        print("[ERROR] No predictions found in results file.")
        sys.exit(1)

    summary = analyze_per_trial(records)
    report = format_markdown_report(summary)
    write_text(report, REPORT_PATH)

    print(f"Records read:      {summary['total_records']}")
    print(f"Trials summarized: {summary['total_trials']}")
    print(f"Overall accuracy:  {_fmt(summary['overall_accuracy'])}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
