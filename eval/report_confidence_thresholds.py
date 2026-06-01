"""
report_confidence_thresholds.py — Task 25: Confidence threshold analysis.

Sweeps confidence thresholds and reports precision/recall/F1 at each cutoff.
Extends Task 18 (report_calibration.py) with threshold-based abstention analysis.

Usage:
    PYTHONPATH=. python eval/report_confidence_thresholds.py
    PYTHONPATH=. python eval/report_confidence_thresholds.py \\
        --results PATH --output PATH
"""

import json
import os
import sys
import argparse
from collections import defaultdict

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/confidence_thresholds.md"

LABEL_ORDER = ["eligible", "not_eligible", "unclear"]
THRESHOLD_STEP = 0.05


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(results: dict) -> list:
    if not isinstance(results, dict) or "predictions" not in results:
        print("ERROR: results JSON missing 'predictions' key.", file=sys.stderr)
        sys.exit(1)
    preds = results["predictions"]
    if not isinstance(preds, list):
        print("ERROR: 'predictions' is not a list.", file=sys.stderr)
        sys.exit(1)
    return preds


def parse_confidence(value) -> float | None:
    """Return float in [0, 1] or None if invalid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


# ---------------------------------------------------------------------------
# Threshold filtering and metrics
# ---------------------------------------------------------------------------

def filter_predictions_by_threshold(predictions: list, threshold: float) -> tuple:
    """
    Split usable predictions into kept (confidence >= threshold) and abstained.
    Returns (kept: list, abstained_count: int).
    Only records with valid confidence, gold_label, and predicted_label are used.
    """
    kept = []
    abstained = 0
    for rec in predictions:
        conf = parse_confidence(rec.get("confidence"))
        if conf is None:
            continue
        gold = rec.get("gold_label", "")
        predicted = rec.get("predicted_label", "") or rec.get("prediction", "")
        if not gold or not predicted:
            continue
        if conf >= threshold:
            kept.append({"gold": gold, "predicted": predicted, "confidence": conf})
        else:
            abstained += 1
    return kept, abstained


def compute_classification_metrics(records: list) -> dict:
    """
    Compute per-label and macro precision, recall, F1 from kept records.
    Uses LABEL_ORDER; labels not present get 0 counts.
    Returns dict with per_label and macro metrics plus accuracy.
    """
    if not records:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
        }

    # Confusion counts per label
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for rec in records:
        g = rec["gold"]
        p = rec["predicted"]
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    # Collect all labels seen
    all_labels = set(LABEL_ORDER)
    for rec in records:
        all_labels.add(rec["gold"])
        all_labels.add(rec["predicted"])

    precisions = []
    recalls = []
    f1s = []

    for label in all_labels:
        t = tp[label]
        f_p = fp[label]
        f_n = fn[label]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec_val = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = (2 * prec * rec_val / (prec + rec_val)) if (prec + rec_val) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec_val)
        f1s.append(f1)

    n = len(all_labels)
    total = len(records)
    correct = sum(1 for r in records if r["gold"] == r["predicted"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else None,
        "macro_precision": sum(precisions) / n if n > 0 else None,
        "macro_recall": sum(recalls) / n if n > 0 else None,
        "macro_f1": sum(f1s) / n if n > 0 else None,
    }


def sweep_thresholds(predictions: list) -> list:
    """
    Sweep thresholds from 0.00 to 1.00 in steps of THRESHOLD_STEP.
    Returns a list of row dicts, one per threshold.
    """
    # Pre-filter to usable records only (valid confidence + labels)
    usable = []
    for rec in predictions:
        conf = parse_confidence(rec.get("confidence"))
        if conf is None:
            continue
        gold = rec.get("gold_label", "")
        predicted = rec.get("predicted_label", "") or rec.get("prediction", "")
        if gold and predicted:
            usable.append(rec)

    total_usable = len(usable)
    rows = []

    # Generate thresholds as integer steps to avoid float precision issues
    steps = round(1.0 / THRESHOLD_STEP)
    thresholds = [round(i * THRESHOLD_STEP, 10) for i in range(steps + 1)]

    for threshold in thresholds:
        kept, abstained = filter_predictions_by_threshold(usable, threshold)
        metrics = compute_classification_metrics(kept)
        coverage = len(kept) / total_usable if total_usable > 0 else 0.0
        rows.append({
            "threshold": threshold,
            "kept": len(kept),
            "abstained": abstained,
            "coverage": coverage,
            "correct": metrics["correct"],
            "errors": len(kept) - metrics["correct"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        })

    return rows


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def choose_recommended_thresholds(rows: list, overall_accuracy: float | None) -> dict:
    """
    Identify:
    - best threshold by kept accuracy
    - best threshold by macro F1
    - highest coverage threshold where kept accuracy >= overall_accuracy
    """
    usable_rows = [r for r in rows if r["accuracy"] is not None]
    f1_rows = [r for r in rows if r["macro_f1"] is not None]

    best_acc = max(usable_rows, key=lambda r: (r["accuracy"], -r["threshold"])) if usable_rows else None
    best_f1 = max(f1_rows, key=lambda r: (r["macro_f1"], -r["threshold"])) if f1_rows else None

    best_coverage = None
    if overall_accuracy is not None:
        candidates = [
            r for r in usable_rows
            if r["accuracy"] >= overall_accuracy
        ]
        if candidates:
            best_coverage = max(candidates, key=lambda r: (r["coverage"], -r["threshold"]))

    return {
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "best_coverage_at_baseline": best_coverage,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "—"


def format_markdown_report(rows: list, recommendations: dict, skipped_count: int) -> str:
    lines = [
        "# Confidence Threshold Analysis",
        "",
        "Thresholds are swept from 0.00 to 1.00 in steps of 0.05.",
        "Predictions with `confidence >= threshold` are **kept**; others are **abstained**.",
        "",
        f"- **Skipped records** (missing/invalid confidence): {skipped_count}",
        "",
    ]

    # Recommendations
    rec = recommendations
    lines += ["## Recommended Thresholds", ""]
    if rec["best_accuracy"] and rec["best_accuracy"]["accuracy"] is not None:
        r = rec["best_accuracy"]
        lines.append(
            f"- **Best kept accuracy**: threshold `{r['threshold']:.2f}` → "
            f"accuracy {r['accuracy']:.4f}, coverage {r['coverage']:.1%}, "
            f"kept {r['kept']}"
        )
    if rec["best_f1"] and rec["best_f1"]["macro_f1"] is not None:
        r = rec["best_f1"]
        lines.append(
            f"- **Best macro F1**: threshold `{r['threshold']:.2f}` → "
            f"macro F1 {r['macro_f1']:.4f}, coverage {r['coverage']:.1%}, "
            f"kept {r['kept']}"
        )
    if rec["best_coverage_at_baseline"]:
        r = rec["best_coverage_at_baseline"]
        lines.append(
            f"- **Highest coverage ≥ baseline accuracy**: threshold `{r['threshold']:.2f}` → "
            f"accuracy {r['accuracy']:.4f}, coverage {r['coverage']:.1%}, "
            f"kept {r['kept']}"
        )
    lines.append("")

    # Full sweep table
    lines += [
        "## Threshold Sweep Table",
        "",
        "| Threshold | Kept | Abstained | Coverage | Accuracy | Macro P | Macro R | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} "
            f"| {r['kept']} "
            f"| {r['abstained']} "
            f"| {r['coverage']:.1%} "
            f"| {_fmt(r['accuracy'])} "
            f"| {_fmt(r['macro_precision'])} "
            f"| {_fmt(r['macro_recall'])} "
            f"| {_fmt(r['macro_f1'])} |"
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
    parser = argparse.ArgumentParser(description="Confidence threshold analysis.")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help=f"Results JSON path (default: {DEFAULT_RESULTS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    results = load_results(args.results)
    predictions = extract_predictions(results)

    # Count skipped
    skipped = sum(
        1 for rec in predictions
        if parse_confidence(rec.get("confidence")) is None
        or not rec.get("gold_label")
        or not (rec.get("predicted_label") or rec.get("prediction"))
    )
    usable = len(predictions) - skipped

    rows = sweep_thresholds(predictions)

    # Overall accuracy = threshold 0.00 row (all kept)
    baseline_row = next((r for r in rows if r["threshold"] == 0.0), None)
    overall_accuracy = baseline_row["accuracy"] if baseline_row else None

    recommendations = choose_recommended_thresholds(rows, overall_accuracy)
    report = format_markdown_report(rows, recommendations, skipped)
    write_text(report, args.output)

    print(f"Prediction records read : {len(predictions)}")
    print(f"Usable confidence records: {usable}")
    print(f"Skipped records          : {skipped}")
    print(f"Report written           : {args.output}")


if __name__ == "__main__":
    main()
