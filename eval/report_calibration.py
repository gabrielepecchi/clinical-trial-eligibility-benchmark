"""
report_calibration.py — Task 18: Confidence calibration report.

Usage:
    PYTHONPATH=. python eval/report_calibration.py
    PYTHONPATH=. python eval/report_calibration.py --results path/to/results.json
    PYTHONPATH=. python eval/report_calibration.py --output path/to/report.md
"""

import json
import os
import sys
import argparse
from collections import defaultdict

DEFAULT_RESULTS_PATH = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT_PATH = "reports/calibration_report.md"

BANDS = [
    (0.00, 0.49, "0.00–0.49"),
    (0.50, 0.59, "0.50–0.59"),
    (0.60, 0.69, "0.60–0.69"),
    (0.70, 0.79, "0.70–0.79"),
    (0.80, 0.89, "0.80–0.89"),
    (0.90, 1.00, "0.90–1.00"),
]


def load_results(path: str) -> dict:
    """Load and return the results JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(results: dict) -> list:
    """Extract the predictions list from results dict."""
    if not isinstance(results, dict) or "predictions" not in results:
        print("ERROR: results JSON missing 'predictions' key.", file=sys.stderr)
        sys.exit(1)
    preds = results["predictions"]
    if not isinstance(preds, list):
        print("ERROR: 'predictions' is not a list.", file=sys.stderr)
        sys.exit(1)
    return preds


def parse_confidence(value) -> float | None:
    """Parse a confidence value; return float in [0,1] or None if invalid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


def assign_confidence_band(confidence: float) -> str:
    """Return the band label for a valid confidence float."""
    for low, high, label in BANDS:
        if low <= confidence <= high:
            return label
    # Clamp edge: confidence == 1.0 is caught by last band; fallback:
    return BANDS[-1][2]


def compute_calibration_by_band(predictions: list) -> dict:
    """
    Group predictions by confidence band and compute per-band metrics.

    Returns a dict with:
        bands: {band_label: {total, correct, errors, accuracy, avg_confidence}}
        total_usable: int
        total_skipped: int
        overall_accuracy: float | None
    """
    band_buckets: dict[str, list] = defaultdict(list)
    skipped = 0

    for rec in predictions:
        gold = rec.get("gold_label")
        predicted = rec.get("predicted_label")
        raw_conf = rec.get("confidence")

        conf = parse_confidence(raw_conf)
        if conf is None or not gold or not predicted:
            skipped += 1
            continue

        band = assign_confidence_band(conf)
        band_buckets[band].append({
            "correct": gold == predicted,
            "confidence": conf,
        })

    band_order = [label for _, _, label in BANDS]
    band_metrics = {}
    total_usable = 0
    total_correct = 0

    for label in band_order:
        records = band_buckets.get(label, [])
        total = len(records)
        correct = sum(1 for r in records if r["correct"])
        errors = total - correct
        accuracy = correct / total if total > 0 else None
        avg_conf = sum(r["confidence"] for r in records) / total if total > 0 else None
        band_metrics[label] = {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
        }
        total_usable += total
        total_correct += correct

    overall_accuracy = total_correct / total_usable if total_usable > 0 else None

    return {
        "bands": band_metrics,
        "total_usable": total_usable,
        "total_skipped": skipped,
        "overall_accuracy": overall_accuracy,
    }


def format_calibration_report(summary: dict) -> str:
    """Format the calibration summary as a readable terminal report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CONFIDENCE CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total usable records : {summary['total_usable']}")
    lines.append(f"Skipped records      : {summary['total_skipped']}")
    oa = summary["overall_accuracy"]
    oa_str = f"{oa:.4f}" if oa is not None else "n/a"
    lines.append(f"Overall accuracy     : {oa_str}")
    lines.append("")
    lines.append(
        f"{'Band':<12} {'Total':>6} {'Correct':>8} {'Errors':>7} "
        f"{'Accuracy':>10} {'Avg Conf':>10}"
    )
    lines.append("-" * 60)

    for band_label, m in summary["bands"].items():
        total = m["total"]
        if total == 0:
            lines.append(f"{band_label:<12} {'0':>6} {'-':>8} {'-':>7} {'-':>10} {'-':>10}")
            continue
        acc_str = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
        avg_str = f"{m['avg_confidence']:.4f}" if m["avg_confidence"] is not None else "n/a"
        lines.append(
            f"{band_label:<12} {total:>6} {m['correct']:>8} {m['errors']:>7} "
            f"{acc_str:>10} {avg_str:>10}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def format_markdown_calibration_report(summary: dict) -> str:
    """Format the calibration summary as a Markdown report."""
    oa = summary["overall_accuracy"]
    oa_str = f"{oa:.4f}" if oa is not None else "n/a"
    lines = [
        "# Confidence Calibration Report",
        "",
        f"**Total usable records:** {summary['total_usable']}  ",
        f"**Skipped records:** {summary['total_skipped']}  ",
        f"**Overall accuracy:** {oa_str}",
        "",
        "---",
        "",
        "## Per-Band Metrics",
        "",
        "| Band | Total | Correct | Errors | Accuracy | Avg Confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for band_label, m in summary["bands"].items():
        total = m["total"]
        if total == 0:
            lines.append(f"| {band_label} | 0 | - | - | - | - |")
            continue
        acc_str = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
        avg_str = f"{m['avg_confidence']:.4f}" if m["avg_confidence"] is not None else "n/a"
        lines.append(
            f"| {band_label} | {total} | {m['correct']} | {m['errors']} | {acc_str} | {avg_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence calibration report.")
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for Markdown report output (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    predictions = extract_predictions(results)
    summary = compute_calibration_by_band(predictions)
    print(format_calibration_report(summary))

    report_md = format_markdown_calibration_report(summary)
    write_report(report_md, args.output)
    print(f"\nReport saved  : {args.output}")


if __name__ == "__main__":
    main()
