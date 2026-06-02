"""
Task 68: per-trial classification metrics.

Usage:
    PYTHONPATH=. python eval/analyze_per_trial_metrics.py
"""

import json
import os
import sys
from collections import defaultdict


INPUT_PATH = "data/processed/results_llm_reviewed.json"
REPORT_PATH = "reports/per_trial_metrics.md"

LABELS = ["eligible", "not_eligible", "unclear"]
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
# Record extraction
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_class_metrics(records: list[dict], labels: list[str]) -> dict[str, dict]:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for r in records:
        gold = get_gold_label(r)
        pred = get_predicted_label(r)
        for lbl in labels:
            if gold == lbl and pred == lbl:
                tp[lbl] += 1
            elif pred == lbl and gold != lbl:
                fp[lbl] += 1
            elif gold == lbl and pred != lbl:
                fn[lbl] += 1

    result = {}
    for lbl in labels:
        p = safe_divide(tp[lbl], tp[lbl] + fp[lbl])
        r = safe_divide(tp[lbl], tp[lbl] + fn[lbl])
        f1 = safe_divide(2 * p * r, p + r)
        result[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                       "tp": tp[lbl], "fp": fp[lbl], "fn": fn[lbl]}
    return result


def compute_macro_metrics(class_metrics: dict[str, dict]) -> dict:
    keys = list(class_metrics.keys())
    if not keys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    mp = sum(class_metrics[k]["precision"] for k in keys) / len(keys)
    mr = sum(class_metrics[k]["recall"] for k in keys) / len(keys)
    mf = sum(class_metrics[k]["f1"] for k in keys) / len(keys)
    return {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)}


def summarize_trial_metrics(trial_id: str, records: list[dict]) -> dict:
    total = len(records)
    correct = sum(1 for r in records if get_gold_label(r) == get_predicted_label(r))
    accuracy = round(safe_divide(correct, total), 4)

    class_metrics = compute_class_metrics(records, LABELS)
    macro = compute_macro_metrics(class_metrics)

    errors = total - correct
    false_eligible = sum(
        1 for r in records
        if get_gold_label(r) == "not_eligible" and get_predicted_label(r) == "eligible"
    )
    unclear_overcommit = sum(
        1 for r in records
        if get_gold_label(r) == "unclear" and get_predicted_label(r) != "unclear"
    )
    unclear_overuse = sum(
        1 for r in records
        if get_gold_label(r) != "unclear" and get_predicted_label(r) == "unclear"
    )

    # Confusion counts
    confusion: dict[str, int] = defaultdict(int)
    for r in records:
        key = f"{get_gold_label(r)}→{get_predicted_label(r)}"
        confusion[key] += 1

    return {
        "trial_id": trial_id,
        "total": total,
        "accuracy": accuracy,
        "macro": macro,
        "class_metrics": class_metrics,
        "errors": errors,
        "false_eligible": false_eligible,
        "unclear_overcommit": unclear_overcommit,
        "unclear_overuse": unclear_overuse,
        "confusion": dict(confusion),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_per_trial_metrics(records: list[dict]) -> dict:
    by_trial: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for r in records:
        gold = get_gold_label(r)
        pred = get_predicted_label(r)
        if not gold or not pred:
            skipped += 1
            continue
        tid = r.get("trial_id", r.get("trial", "unknown_trial"))
        if isinstance(tid, dict):
            tid = str(tid)
        by_trial[str(tid).strip()].append(r)

    trial_summaries = [
        summarize_trial_metrics(tid, recs)
        for tid, recs in sorted(by_trial.items())
    ]

    all_valid = [r for recs in by_trial.values() for r in recs]
    overall_class = compute_class_metrics(all_valid, LABELS)
    overall_macro = compute_macro_metrics(overall_class)
    overall_correct = sum(1 for r in all_valid if get_gold_label(r) == get_predicted_label(r))
    overall_accuracy = round(safe_divide(overall_correct, len(all_valid)), 4)

    multi = [t for t in trial_summaries if t["total"] >= 2]
    top_low_f1 = sorted(multi, key=lambda t: t["macro"]["f1"])[:TOP_N]
    top_errors = sorted(trial_summaries, key=lambda t: -t["errors"])[:TOP_N]
    top_false_elig = sorted(trial_summaries, key=lambda t: -t["false_eligible"])[:TOP_N]

    return {
        "total_records": len(records),
        "used_records": len(all_valid),
        "skipped": skipped,
        "total_trials": len(trial_summaries),
        "overall_accuracy": overall_accuracy,
        "overall_macro": overall_macro,
        "overall_class": overall_class,
        "trial_summaries": trial_summaries,
        "top_low_f1": top_low_f1,
        "top_errors": top_errors,
        "top_false_elig": top_false_elig,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _r(v: float) -> str:
    return f"{v:.4f}"


def format_markdown_report(summary: dict) -> str:
    lines = [
        "# Per-Trial Classification Metrics",
        "",
        f"**Total records:** {summary['total_records']}  ",
        f"**Records used:** {summary['used_records']}  ",
        f"**Records skipped (missing labels):** {summary['skipped']}  ",
        f"**Total trials:** {summary['total_trials']}",
        "",
        "---",
        "",
        "### Overall Metrics (all records)",
        "",
        f"- Accuracy: {_r(summary['overall_accuracy'])}",
        f"- Macro Precision: {_r(summary['overall_macro']['precision'])}",
        f"- Macro Recall: {_r(summary['overall_macro']['recall'])}",
        f"- Macro F1: {_r(summary['overall_macro']['f1'])}",
        "",
        "| Class | Precision | Recall | F1 |",
        "| --- | --- | --- | --- |",
    ]
    for lbl in LABELS:
        cm = summary["overall_class"].get(lbl, {})
        lines.append(
            f"| {lbl} | {_r(cm.get('precision', 0))} "
            f"| {_r(cm.get('recall', 0))} | {_r(cm.get('f1', 0))} |"
        )
    lines.append("")

    def _top_table(title: str, trials: list[dict], sort_key: str) -> list[str]:
        out = [f"### {title}", "",
               "| trial_id | Records | Accuracy | Macro F1 | Errors | False Eligible |",
               "| --- | --- | --- | --- | --- | --- |"]
        for t in trials:
            out.append(
                f"| {t['trial_id']} | {t['total']} | {_r(t['accuracy'])} "
                f"| {_r(t['macro']['f1'])} | {t['errors']} | {t['false_eligible']} |"
            )
        out.append("")
        return out

    lines += ["---", ""]
    lines += _top_table(f"Top {TOP_N} Trials by Lowest Macro F1 (≥2 records)",
                        summary["top_low_f1"], "f1")
    lines += ["---", ""]
    lines += _top_table(f"Top {TOP_N} Trials by Highest Error Count",
                        summary["top_errors"], "errors")
    lines += ["---", ""]
    lines += _top_table(f"Top {TOP_N} Trials by High-Risk False Eligible Count",
                        summary["top_false_elig"], "false_eligible")

    # Full table
    lines += [
        "---", "",
        "### Full Per-Trial Metrics Table", "",
        "| trial_id | n | Acc | MacroF1 | Err | FalseElig | UnclearOver | UnclearOveruse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in summary["trial_summaries"]:
        lines.append(
            f"| {t['trial_id']} | {t['total']} | {_r(t['accuracy'])} "
            f"| {_r(t['macro']['f1'])} | {t['errors']} | {t['false_eligible']} "
            f"| {t['unclear_overcommit']} | {t['unclear_overuse']} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        data = load_json(INPUT_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        records = extract_predictions(data)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_per_trial_metrics(records)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Records read    : {summary['total_records']}")
    print(f"Records used    : {summary['used_records']}")
    print(f"Trials          : {summary['total_trials']}")
    print(f"Report          : {REPORT_PATH}")


if __name__ == "__main__":
    main()
