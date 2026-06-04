"""
baselines.py — Task 31: simple benchmark baselines.

Usage:
    PYTHONPATH=. python eval/baselines.py
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

LABELS_PATH = Path("data/processed/labels_llm_reviewed.json")
RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
REPORT_PATH = Path("reports/baseline_comparison.md")

VALID_LABELS = ["eligible", "not_eligible", "unclear"]

STRATEGIES = [
    "always_unclear",
    "always_eligible",
    "always_not_eligible",
    "majority_class",
    "strict_missing_unclear",
    "optimistic_missing_eligible",
    "conservative_missing_unclear_or_not_eligible",
]

MISSING_POLICY_STRATEGIES = {
    "strict_missing_unclear",
    "optimistic_missing_eligible",
    "conservative_missing_unclear_or_not_eligible",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_gold_labels(labels: list) -> list[str]:
    return [rec["label"] for rec in labels if isinstance(rec, dict) and rec.get("label")]


def _has_structured_missingness(record: dict) -> bool:
    """Return True if a prediction record signals structured missingness or uncertainty."""
    if record.get("unknown_fields"):
        return True
    if record.get("missing_information"):
        return True
    details = record.get("missing_information_details") or []
    if any(d.get("status") == "unknown" for d in details):
        return True
    if record.get("uncertain_criteria"):
        return True
    if record.get("missing_reason_type"):
        return True
    return False


def _has_blocking_evidence(record: dict) -> bool:
    """Return True if a prediction record has clear blocking evidence."""
    if record.get("blocking_criteria"):
        return True
    if record.get("blocked_by"):
        return True
    return False


def predict_missing_policy(record: dict, strategy: str) -> str:
    """Predict a label for a single prediction record using a missing-information policy.

    Args:
        record:   A prediction record dict (may contain structured missingness fields).
        strategy: One of the MISSING_POLICY_STRATEGIES.

    Returns:
        A label string: 'eligible' | 'not_eligible' | 'unclear'.
    """
    has_missing = _has_structured_missingness(record)
    has_blocking = _has_blocking_evidence(record)

    if strategy == "strict_missing_unclear":
        if has_missing:
            return "unclear"
        return "eligible"

    if strategy == "optimistic_missing_eligible":
        if has_blocking:
            return "not_eligible"
        return "eligible"

    if strategy == "conservative_missing_unclear_or_not_eligible":
        if has_blocking:
            return "not_eligible"
        if has_missing:
            return "unclear"
        return "eligible"

    raise ValueError(f"Unknown missing-policy strategy: {strategy}")


def predict_baseline(records: list, strategy: str, prediction_records: list | None = None) -> list[str]:
    gold = extract_gold_labels(records)
    if strategy == "always_unclear":
        return ["unclear"] * len(gold)
    if strategy == "always_eligible":
        return ["eligible"] * len(gold)
    if strategy == "always_not_eligible":
        return ["not_eligible"] * len(gold)
    if strategy == "majority_class":
        majority = Counter(gold).most_common(1)[0][0]
        return [majority] * len(gold)
    if strategy in MISSING_POLICY_STRATEGIES:
        if prediction_records and len(prediction_records) == len(gold):
            return [predict_missing_policy(r, strategy) for r in prediction_records]
        # Fallback when structured records are unavailable: use gold-length list of eligible
        return ["eligible"] * len(gold)
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(gold: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        g: {p: 0 for p in VALID_LABELS} for g in VALID_LABELS
    }
    for g, p in zip(gold, predicted):
        if g in matrix and p in VALID_LABELS:
            matrix[g][p] += 1
    return matrix


def compute_metrics(gold: list[str], predicted: list[str]) -> dict:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, predicted))
    accuracy = correct / n if n else 0.0

    per_class: dict[str, dict] = {}
    for cls in VALID_LABELS:
        tp = sum(g == cls and p == cls for g, p in zip(gold, predicted))
        fp = sum(g != cls and p == cls for g, p in zip(gold, predicted))
        fn = sum(g == cls and p != cls for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    macro_precision = sum(v["precision"] for v in per_class.values()) / 3
    macro_recall = sum(v["recall"] for v in per_class.values()) / 3
    macro_f1 = sum(v["f1"] for v in per_class.values()) / 3

    return {
        "total": n,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(gold, predicted),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baselines(labels: list, prediction_records: list | None = None) -> dict[str, dict]:
    gold = extract_gold_labels(labels)
    results = {}
    for strategy in STRATEGIES:
        predicted = predict_baseline(labels, strategy, prediction_records)
        results[strategy] = compute_metrics(gold, predicted)
    return results


def load_current_metrics(path: Path) -> tuple[dict | None, list | None]:
    """Return (summary_metrics, prediction_records) from a results JSON file.

    prediction_records will be None if the file is missing or unreadable.
    """
    try:
        data = load_json(path)
        metrics = data.get("metrics", {})
        summary = {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        }
        prediction_records = data.get("predictions") or None
        return summary, prediction_records
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_markdown_report(summary: dict) -> str:
    baselines: dict = summary["baselines"]
    current: dict | None = summary.get("current_metrics")
    gold_dist: dict = summary["gold_distribution"]

    lines = [
        "# Baseline Comparison Report",
        "",
        "## Gold Label Distribution",
        "",
        "| Label | Count | % |",
        "|---|---|---|",
    ]
    total = sum(gold_dist.values())
    for label in VALID_LABELS:
        count = gold_dist.get(label, 0)
        pct = 100 * count / total if total else 0
        lines.append(f"| {label} | {count} | {pct:.1f}% |")

    lines += [
        "",
        "## Summary Comparison",
        "",
        "| Strategy | Accuracy | Macro F1 |",
        "|---|---|---|",
    ]
    for strategy, metrics in baselines.items():
        lines.append(
            f"| {strategy} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['macro_f1'])} |"
        )
    if current:
        lines.append(
            f"| **current_matcher** | {_fmt(current['accuracy'])} | {_fmt(current['macro_f1'])} |"
        )

    lines += [
        "",
        "## Note on Missing-Information Policy Baselines",
        "",
        "> `strict_missing_unclear`, `optimistic_missing_eligible`, and "
        "`conservative_missing_unclear_or_not_eligible` are **diagnostic policy baselines**.  ",
        "> They test how different strategies for handling incomplete clinical information "
        "affect benchmark scores.  ",
        "> They are **not clinical recommendations** and do not represent valid decision rules "
        "for real patient eligibility assessment.  ",
        "> When structured prediction records (with `unknown_fields`, `blocking_criteria`, etc.) "
        "are unavailable, these baselines fall back to predicting `eligible` for all records.  ",
        "",
    ]

    for strategy, metrics in baselines.items():
        lines += [
            "",
            f"## Baseline: {strategy}",
            "",
            f"- Total records: {metrics['total']}",
            f"- Accuracy: {_fmt(metrics['accuracy'])}",
            f"- Macro precision: {_fmt(metrics['macro_precision'])}",
            f"- Macro recall: {_fmt(metrics['macro_recall'])}",
            f"- Macro F1: {_fmt(metrics['macro_f1'])}",
            "",
            "### Per-class metrics",
            "",
            "| Class | Precision | Recall | F1 |",
            "|---|---|---|---|",
        ]
        for cls in VALID_LABELS:
            pc = metrics["per_class"][cls]
            lines.append(
                f"| {cls} | {_fmt(pc['precision'])} | {_fmt(pc['recall'])} | {_fmt(pc['f1'])} |"
            )

        cm = metrics["confusion_matrix"]
        lines += [
            "",
            "### Confusion matrix",
            "",
            "| gold \\ predicted | eligible | not_eligible | unclear |",
            "|---|---|---|---|",
        ]
        for g in VALID_LABELS:
            row = " | ".join(str(cm[g][p]) for p in VALID_LABELS)
            lines.append(f"| {g} | {row} |")

    lines += ["", "---", "_Generated by eval/baselines.py_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        labels_data = load_json(LABELS_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Labels file not found: {LABELS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {LABELS_PATH}: {exc}")
        sys.exit(1)

    if not isinstance(labels_data, list):
        print(f"[ERROR] {LABELS_PATH} must be a JSON array.")
        sys.exit(1)

    gold = extract_gold_labels(labels_data)
    gold_dist = dict(Counter(gold))

    current, prediction_records = load_current_metrics(RESULTS_PATH)
    baselines = evaluate_baselines(labels_data, prediction_records)

    summary = {
        "baselines": baselines,
        "gold_distribution": gold_dist,
        "current_metrics": current,
    }

    report = format_markdown_report(summary)
    write_text(report, REPORT_PATH)
    print(f"Report written to {REPORT_PATH}")

    for strategy, metrics in baselines.items():
        print(f"  {strategy:25s}  acc={_fmt(metrics['accuracy'])}  macro_f1={_fmt(metrics['macro_f1'])}")
    if current:
        print(f"  {'current_matcher':25s}  acc={_fmt(current['accuracy'])}  macro_f1={_fmt(current['macro_f1'])}")


if __name__ == "__main__":
    main()
