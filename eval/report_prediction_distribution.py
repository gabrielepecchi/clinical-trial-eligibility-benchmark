"""
report_prediction_distribution.py — Task 37: Prediction distribution report.

Summarises how predicted labels are distributed across patients, trials, and
confidence levels, and compares predicted vs gold distributions.

Usage:
    PYTHONPATH=. python eval/report_prediction_distribution.py
    PYTHONPATH=. python eval/report_prediction_distribution.py \\
        --results PATH --output PATH
"""

import json
import os
import sys
import argparse
from collections import Counter, defaultdict

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/prediction_distribution_report.md"

VALID_LABELS = ["eligible", "not_eligible", "unclear"]

CONF_BUCKETS = [
    (0.00, 0.25, "0.00–0.25"),
    (0.26, 0.50, "0.26–0.50"),
    (0.51, 0.75, "0.51–0.75"),
    (0.76, 1.00, "0.76–1.00"),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(data: dict) -> list:
    if not isinstance(data, dict) or "predictions" not in data:
        print("ERROR: results JSON missing 'predictions' key.", file=sys.stderr)
        sys.exit(1)
    preds = data["predictions"]
    if not isinstance(preds, list):
        print("ERROR: 'predictions' is not a list.", file=sys.stderr)
        sys.exit(1)
    return preds


# ---------------------------------------------------------------------------
# Field accessors
# ---------------------------------------------------------------------------

def get_predicted_label(record: dict) -> str:
    return record.get("predicted_label", "") or record.get("prediction", "") or ""


def get_gold_label(record: dict) -> str:
    return record.get("gold_label", "") or record.get("label", "") or ""


def parse_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_basic_stats(values: list) -> dict:
    """Compute min, max, mean, median for a list of floats."""
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2
    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": median,
    }


def confidence_bucket(value: float) -> str:
    for low, high, label in CONF_BUCKETS:
        if low <= value <= high:
            return label
    return CONF_BUCKETS[-1][2]


def count_labels(records: list, label_getter) -> Counter:
    return Counter(label_getter(r) for r in records)


# ---------------------------------------------------------------------------
# Group summarisation
# ---------------------------------------------------------------------------

def summarize_by_group(records: list, group_field: str) -> dict:
    """
    Group records by group_field and return per-group stats:
    total, predicted label counts, error count.
    """
    groups: dict[str, list] = defaultdict(list)
    for rec in records:
        key = rec.get(group_field, "")
        groups[key].append(rec)

    summary = {}
    for key, recs in groups.items():
        pred_counts = count_labels(recs, get_predicted_label)
        errors = sum(
            1 for r in recs if get_gold_label(r) != get_predicted_label(r)
        )
        summary[key] = {
            "total": len(recs),
            "eligible": pred_counts.get("eligible", 0),
            "not_eligible": pred_counts.get("not_eligible", 0),
            "unclear": pred_counts.get("unclear", 0),
            "errors": errors,
        }
    return summary


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_prediction_distribution(records: list) -> dict:
    total = len(records)
    gold_dist = count_labels(records, get_gold_label)
    pred_dist = count_labels(records, get_predicted_label)

    correct = sum(
        1 for r in records if get_gold_label(r) == get_predicted_label(r)
    )
    accuracy = correct / total if total > 0 else None

    # Confidence analysis
    conf_values = []
    missing_conf = 0
    for rec in records:
        c = parse_confidence(rec.get("confidence"))
        if c is None:
            missing_conf += 1
        else:
            conf_values.append(c)

    conf_stats = compute_basic_stats(conf_values)

    # Confidence buckets + accuracy per bucket
    bucket_records: dict[str, list] = defaultdict(list)
    for rec in records:
        c = parse_confidence(rec.get("confidence"))
        if c is not None:
            bucket_records[confidence_bucket(c)].append(rec)

    bucket_summary = {}
    for _, _, label in CONF_BUCKETS:
        recs = bucket_records.get(label, [])
        if not recs:
            bucket_summary[label] = {"total": 0, "accuracy": None}
            continue
        correct_b = sum(
            1 for r in recs if get_gold_label(r) == get_predicted_label(r)
        )
        bucket_summary[label] = {
            "total": len(recs),
            "accuracy": correct_b / len(recs),
        }

    # Per-patient and per-trial
    by_patient = summarize_by_group(records, "patient_id")
    by_trial = summarize_by_group(records, "trial_id")

    # Top 10 by errors
    top_patients = sorted(by_patient.items(), key=lambda x: (-x[1]["errors"], x[0]))[:10]
    top_trials = sorted(by_trial.items(), key=lambda x: (-x[1]["errors"], x[0]))[:10]

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "gold_distribution": dict(gold_dist),
        "predicted_distribution": dict(pred_dist),
        "conf_stats": conf_stats,
        "missing_conf": missing_conf,
        "conf_values_count": len(conf_values),
        "bucket_summary": bucket_summary,
        "by_patient": by_patient,
        "by_trial": by_trial,
        "top_patients": top_patients,
        "top_trials": top_trials,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(val, spec=".4f") -> str:
    return f"{val:{spec}}" if val is not None else "—"


def format_markdown_report(summary: dict) -> str:
    total = summary["total"]
    accuracy = summary["accuracy"]

    lines = [
        "# Prediction Distribution Report",
        "",
        f"- **Total prediction records**: {total}",
        f"- **Correct predictions**: {summary['correct']}",
        f"- **Accuracy**: {_fmt(accuracy)}",
        "",
        "---",
        "",
        "## Label Distributions",
        "",
        "| Label | Gold Count | Predicted Count |",
        "|---|---:|---:|",
    ]

    all_labels = sorted(
        set(list(summary["gold_distribution"]) + list(summary["predicted_distribution"]))
    )
    for lbl in all_labels:
        g = summary["gold_distribution"].get(lbl, 0)
        p = summary["predicted_distribution"].get(lbl, 0)
        lines.append(f"| {lbl or '(empty)'} | {g} | {p} |")

    lines += ["", "---", "", "## Confidence Distribution", ""]

    cs = summary["conf_stats"]
    lines += [
        f"- **Records with confidence**: {summary['conf_values_count']}",
        f"- **Missing confidence**: {summary['missing_conf']}",
        f"- **Min**: {_fmt(cs['min'])}",
        f"- **Max**: {_fmt(cs['max'])}",
        f"- **Mean**: {_fmt(cs['mean'])}",
        f"- **Median**: {_fmt(cs['median'])}",
        "",
        "### Accuracy by Confidence Bucket",
        "",
        "| Bucket | Records | Accuracy |",
        "|---|---:|---:|",
    ]
    for _, _, label in CONF_BUCKETS:
        b = summary["bucket_summary"].get(label, {"total": 0, "accuracy": None})
        lines.append(f"| {label} | {b['total']} | {_fmt(b['accuracy'])} |")

    lines += ["", "---", "", "## Top 10 Patients by Error Count", "",
               "| patient_id | Total | Eligible | Not Eligible | Unclear | Errors |",
               "|---|---:|---:|---:|---:|---:|"]
    for pid, m in summary["top_patients"]:
        lines.append(
            f"| {pid} | {m['total']} | {m['eligible']} "
            f"| {m['not_eligible']} | {m['unclear']} | {m['errors']} |"
        )

    lines += ["", "---", "", "## Top 10 Trials by Error Count", "",
               "| trial_id | Total | Eligible | Not Eligible | Unclear | Errors |",
               "|---|---:|---:|---:|---:|---:|"]
    for tid, m in summary["top_trials"]:
        lines.append(
            f"| {tid} | {m['total']} | {m['eligible']} "
            f"| {m['not_eligible']} | {m['unclear']} | {m['errors']} |"
        )

    lines += ["", "---", "", "## Full Patient Summary", "",
               "| patient_id | Total | Eligible | Not Eligible | Unclear | Errors |",
               "|---|---:|---:|---:|---:|---:|"]
    for pid, m in sorted(summary["by_patient"].items()):
        lines.append(
            f"| {pid} | {m['total']} | {m['eligible']} "
            f"| {m['not_eligible']} | {m['unclear']} | {m['errors']} |"
        )

    lines += ["", "## Full Trial Summary", "",
               "| trial_id | Total | Eligible | Not Eligible | Unclear | Errors |",
               "|---|---:|---:|---:|---:|---:|"]
    for tid, m in sorted(summary["by_trial"].items()):
        lines.append(
            f"| {tid} | {m['total']} | {m['eligible']} "
            f"| {m['not_eligible']} | {m['unclear']} | {m['errors']} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction distribution report.")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help=f"Results JSON path (default: {DEFAULT_RESULTS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data = load_json(args.results)
    records = extract_predictions(data)
    summary = analyze_prediction_distribution(records)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    acc_str = f"{summary['accuracy']:.4f}" if summary["accuracy"] is not None else "n/a"
    print(f"Records read : {summary['total']}")
    print(f"Accuracy     : {acc_str}")
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
