"""
Task 92: eligibility rate analysis.

Usage:
    PYTHONPATH=. python eval/analyze_eligibility_rate.py
"""

import json
import os
import sys
from collections import defaultdict


INPUT_PATH = "data/processed/results_llm_reviewed.json"
REPORT_PATH = "reports/eligibility_rate_analysis.md"

VALID_LABELS = {"eligible", "not_eligible", "unclear"}
TOP_N = 10


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def extract_predictions(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("predictions", "results", "records", "cases"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Cannot locate a records list in the JSON.")


def get_gold_label(record: dict) -> str:
    for f in ("gold_label", "gold", "label", "expected"):
        v = record.get(f, "")
        if v:
            return str(v).strip().lower()
    return ""


def get_predicted_label(record: dict) -> str:
    for f in ("predicted_label", "predicted", "prediction", "output"):
        v = record.get(f, "")
        if v:
            return str(v).strip().lower()
    return ""


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# Counting and rates
# ---------------------------------------------------------------------------

def label_counts(records: list[dict], label_getter) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        lbl = label_getter(r)
        if lbl:
            counts[lbl] += 1
    return dict(counts)


def label_rates(counts: dict[str, int], total: int) -> dict[str, float]:
    return {lbl: round(safe_divide(cnt, total), 4) for lbl, cnt in counts.items()}


# ---------------------------------------------------------------------------
# Group summarization
# ---------------------------------------------------------------------------

def summarize_group(group_id: str, records: list[dict]) -> dict:
    total = len(records)
    pred_counts = label_counts(records, get_predicted_label)
    gold_counts = label_counts(records, get_gold_label)
    pred_rates = label_rates(pred_counts, total)
    gold_rates = label_rates(gold_counts, total)
    return {
        "group_id": group_id,
        "total": total,
        "pred_counts": pred_counts,
        "pred_rates": pred_rates,
        "gold_counts": gold_counts,
        "gold_rates": gold_rates,
    }


def group_records(records: list[dict], key_field: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = str(r.get(key_field, "")).strip() or "unknown"
        groups[key].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_eligibility_rates(records: list[dict]) -> dict:
    used = []
    skipped = 0
    for r in records:
        pred = get_predicted_label(r)
        if not pred or pred not in VALID_LABELS:
            skipped += 1
            continue
        used.append(r)

    overall = summarize_group("overall", used)

    by_trial = {
        tid: summarize_group(tid, recs)
        for tid, recs in sorted(group_records(used, "trial_id").items())
    }
    by_patient = {
        pid: summarize_group(pid, recs)
        for pid, recs in sorted(group_records(used, "patient_id").items())
    }

    def top_by_rate(groups: dict[str, dict], label: str, min_records: int = 2) -> list[dict]:
        eligible = [g for g in groups.values() if g["total"] >= min_records]
        return sorted(
            eligible,
            key=lambda g: -g["pred_rates"].get(label, 0.0)
        )[:TOP_N]

    return {
        "total_records": len(records),
        "used_records": len(used),
        "skipped": skipped,
        "overall": overall,
        "by_trial": by_trial,
        "by_patient": by_patient,
        "top_eligible_trials": top_by_rate(by_trial, "eligible"),
        "top_unclear_trials": top_by_rate(by_trial, "unclear"),
        "top_eligible_patients": top_by_rate(by_patient, "eligible"),
        "top_unclear_patients": top_by_rate(by_patient, "unclear"),
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _r(v: float) -> str:
    return f"{v:.4f}"


def _rate_table(title: str, summary: dict, include_gold: bool) -> list[str]:
    lines = [f"### {title}", ""]
    total = summary["total"]
    pred = summary["pred_counts"]
    gold = summary["gold_counts"]
    pred_r = summary["pred_rates"]
    gold_r = summary["gold_rates"]

    lines += [f"**Total records:** {total}", "",
              "| Label | Predicted Count | Predicted Rate |"
              + (" Gold Count | Gold Rate |" if include_gold else " |"),
              "| --- | --- | --- |" + (" --- | --- |" if include_gold else "")]

    for lbl in ("eligible", "not_eligible", "unclear"):
        pc = pred.get(lbl, 0)
        pr = _r(pred_r.get(lbl, 0.0))
        row = f"| {lbl} | {pc} | {pr} |"
        if include_gold:
            gc = gold.get(lbl, 0)
            gr = _r(gold_r.get(lbl, 0.0))
            row += f" {gc} | {gr} |"
        lines.append(row)
    lines.append("")
    return lines


def _group_rate_table(title: str, groups: dict[str, dict]) -> list[str]:
    lines = [f"### {title}", "",
             "| ID | n | eligible | not_eligible | unclear |",
             "| --- | --- | --- | --- | --- |"]
    for gid, g in groups.items():
        r = g["pred_rates"]
        lines.append(
            f"| {gid} | {g['total']} "
            f"| {_r(r.get('eligible', 0))} "
            f"| {_r(r.get('not_eligible', 0))} "
            f"| {_r(r.get('unclear', 0))} |"
        )
    lines.append("")
    return lines


def _top_table(title: str, groups: list[dict], label: str) -> list[str]:
    lines = [f"### {title}", "",
             f"| ID | n | {label} rate |",
             "| --- | --- | --- |"]
    for g in groups:
        rate = _r(g["pred_rates"].get(label, 0.0))
        lines.append(f"| {g['group_id']} | {g['total']} | {rate} |")
    lines.append("")
    return lines


def format_markdown_report(summary: dict) -> str:
    has_gold = bool(summary["overall"]["gold_counts"])

    lines = [
        "# Eligibility Rate Analysis",
        "",
        f"**Total records:** {summary['total_records']}  ",
        f"**Records used:** {summary['used_records']}  ",
        f"**Records skipped (missing/invalid predicted label):** {summary['skipped']}",
        "",
        "---",
        "",
    ]

    lines += _rate_table("Overall Prediction Rates", summary["overall"], has_gold)
    lines += ["---", ""]
    lines += _group_rate_table("Per-Trial Prediction Rates", summary["by_trial"])
    lines += ["---", ""]
    lines += _group_rate_table("Per-Patient Prediction Rates", summary["by_patient"])
    lines += ["---", ""]
    lines += _top_table(
        f"Top {TOP_N} Trials by Highest Predicted Eligible Rate (≥2 records)",
        summary["top_eligible_trials"], "eligible"
    )
    lines += _top_table(
        f"Top {TOP_N} Trials by Highest Predicted Unclear Rate (≥2 records)",
        summary["top_unclear_trials"], "unclear"
    )
    lines += ["---", ""]
    lines += _top_table(
        f"Top {TOP_N} Patients by Highest Predicted Eligible Rate (≥2 records)",
        summary["top_eligible_patients"], "eligible"
    )
    lines += _top_table(
        f"Top {TOP_N} Patients by Highest Predicted Unclear Rate (≥2 records)",
        summary["top_unclear_patients"], "unclear"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        data = load_json(INPUT_PATH)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        records = extract_predictions(data)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_eligibility_rates(records)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Records read  : {summary['total_records']}")
    print(f"Records used  : {summary['used_records']}")
    print(f"Skipped       : {summary['skipped']}")
    print(f"Report        : {REPORT_PATH}")


if __name__ == "__main__":
    main()
