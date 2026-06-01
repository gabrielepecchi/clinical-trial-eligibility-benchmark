"""
error_heatmap.py — Task 57: patient × trial error heatmap.

Usage:
    PYTHONPATH=. python eval/error_heatmap.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
REPORT_PATH = Path("reports/error_heatmap.md")


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


# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------

def classify_cell(record: dict) -> str:
    gold = get_gold_label(record)
    pred = get_predicted_label(record)

    if gold == "not_eligible" and pred == "eligible":
        return "FE"
    if pred == "unclear" and gold == "unclear":
        return "UC"
    if pred == "unclear" and gold != "unclear":
        return "AU"
    if gold == pred:
        return "."
    return "E"


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_heatmap(records: list) -> dict:
    matrix: dict[str, dict[str, str]] = defaultdict(dict)
    for rec in records:
        if not isinstance(rec, dict):
            continue
        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        if pid and tid:
            matrix[pid][tid] = classify_cell(rec)
    return dict(matrix)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_heatmap(records: list, matrix: dict) -> dict:
    patient_ids = sorted(matrix.keys())
    trial_ids = sorted({tid for row in matrix.values() for tid in row})

    total = len([r for r in records if isinstance(r, dict) and r.get("patient_id") and r.get("trial_id")])
    errors = sum(v not in (".", "UC") for row in matrix.values() for v in row.values())
    false_eligible = sum(v == "FE" for row in matrix.values() for v in row.values())
    abstained = sum(v == "AU" for row in matrix.values() for v in row.values())
    correct_unclear = sum(v == "UC" for row in matrix.values() for v in row.values())

    patient_errors = {
        pid: sum(v not in (".", "UC") for v in matrix[pid].values())
        for pid in patient_ids
    }
    trial_errors = {
        tid: sum(
            matrix[pid].get(tid, "") not in (".", "UC", "")
            for pid in patient_ids
            if tid in matrix[pid]
        )
        for tid in trial_ids
    }

    top_patients = sorted(patient_errors.items(), key=lambda x: x[1], reverse=True)[:10]
    top_trials = sorted(trial_errors.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "patient_ids": patient_ids,
        "trial_ids": trial_ids,
        "matrix": matrix,
        "total": total,
        "total_patients": len(patient_ids),
        "total_trials": len(trial_ids),
        "errors": errors,
        "false_eligible": false_eligible,
        "abstained": abstained,
        "correct_unclear": correct_unclear,
        "top_patients": top_patients,
        "top_trials": top_trials,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_markdown_report(summary: dict) -> str:
    patient_ids = summary["patient_ids"]
    trial_ids = summary["trial_ids"]
    matrix = summary["matrix"]

    lines = [
        "# Patient × Trial Error Heatmap",
        "",
        "## Legend",
        "",
        "| Symbol | Meaning |",
        "|---|---|",
        "| `.` | Correct prediction |",
        "| `E` | Incorrect prediction |",
        "| `FE` | High-risk false eligible (gold=not_eligible, pred=eligible) |",
        "| `AU` | Abstained/unclear incorrect (pred=unclear, gold≠unclear) |",
        "| `UC` | Correct unclear (gold=unclear, pred=unclear) |",
        "| _(blank)_ | No record for this pair |",
        "",
        "## Summary",
        "",
        f"- Total records: {summary['total']}",
        f"- Total patients: {summary['total_patients']}",
        f"- Total trials: {summary['total_trials']}",
        f"- Total errors: {summary['errors']}",
        f"- High-risk false eligible (FE): {summary['false_eligible']}",
        f"- Abstained/unclear incorrect (AU): {summary['abstained']}",
        f"- Correct unclear (UC): {summary['correct_unclear']}",
        "",
        "## Top 10 Patients by Error Count",
        "",
        "| patient_id | errors |",
        "|---|---|",
    ]
    for pid, count in summary["top_patients"]:
        lines.append(f"| {pid} | {count} |")

    lines += [
        "",
        "## Top 10 Trials by Error Count",
        "",
        "| trial_id | errors |",
        "|---|---|",
    ]
    for tid, count in summary["top_trials"]:
        lines.append(f"| {tid} | {count} |")

    # Heatmap table
    header = "| patient \\ trial | " + " | ".join(trial_ids) + " |"
    separator = "|---|" + "---|" * len(trial_ids)
    lines += ["", "## Heatmap", "", header, separator]

    for pid in patient_ids:
        cells = [matrix[pid].get(tid, "") for tid in trial_ids]
        row = "| " + pid + " | " + " | ".join(cells) + " |"
        lines.append(row)

    lines += ["", "---", "_Generated by eval/error_heatmap.py_", ""]
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

    matrix = build_heatmap(records)
    summary = summarize_heatmap(records, matrix)
    report = format_markdown_report(summary)
    write_text(report, REPORT_PATH)

    print(f"Records read:      {summary['total']}")
    print(f"Patients:          {summary['total_patients']}")
    print(f"Trials:            {summary['total_trials']}")
    print(f"Errors:            {summary['errors']}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
