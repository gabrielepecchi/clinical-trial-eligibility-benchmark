"""
report_cross_trial_consistency.py — Task 23: Cross-trial patient consistency report.

Groups predictions by patient_id and surfaces consistency patterns:
repeated errors, label distribution mismatches, shared blocking/uncertain criteria.

Usage:
    PYTHONPATH=. python eval/report_cross_trial_consistency.py
    PYTHONPATH=. python eval/report_cross_trial_consistency.py \\
        --results PATH --output PATH
"""

import json
import os
import sys
import argparse
from collections import Counter, defaultdict

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/cross_trial_consistency.md"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict:
    """Load and return results JSON."""
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
    """Extract the predictions list from results dict."""
    if not isinstance(results, dict) or "predictions" not in results:
        print("ERROR: results JSON missing 'predictions' key.", file=sys.stderr)
        sys.exit(1)
    preds = results["predictions"]
    if not isinstance(preds, list):
        print("ERROR: 'predictions' is not a list.", file=sys.stderr)
        sys.exit(1)
    return preds


# ---------------------------------------------------------------------------
# Grouping and summarisation
# ---------------------------------------------------------------------------

def group_predictions_by_patient(predictions: list) -> dict:
    """Return dict: patient_id -> list of prediction records."""
    groups: dict[str, list] = defaultdict(list)
    for pred in predictions:
        pid = pred.get("patient_id", "")
        if pid:
            groups[pid].append(pred)
    return dict(groups)


def _collect_phrases(records: list, field: str) -> Counter:
    """Collect and count non-empty string items from a list field across records."""
    counter: Counter = Counter()
    for rec in records:
        items = rec.get(field) or []
        if isinstance(items, str):
            items = [items]
        for item in items:
            text = str(item).strip()
            if text:
                counter[text] += 1
    return counter


def summarize_patient_predictions(patient_id: str, records: list) -> dict:
    """
    Summarise all trial predictions for one patient.
    Returns a dict with consistency statistics and examples.
    """
    total = len(records)
    gold_dist = Counter(r.get("gold_label", "") for r in records)
    pred_dist = Counter(r.get("predicted_label", "") or r.get("prediction", "") for r in records)

    errors = [
        r for r in records
        if r.get("gold_label", "") != (r.get("predicted_label", "") or r.get("prediction", ""))
    ]
    n_errors = len(errors)

    # Shared blocking/uncertain criteria across records
    blocking_counts = _collect_phrases(records, "blocking_criteria")
    uncertain_counts = _collect_phrases(records, "uncertain_criteria")

    # Top repeated blocking/uncertain phrases (appearing in >1 trial)
    repeated_blocking = [p for p, c in blocking_counts.most_common() if c > 1]
    repeated_uncertain = [p for p, c in uncertain_counts.most_common() if c > 1]

    # Gold vs predicted disagreement: label most common in gold but absent/low in pred
    gold_dominant = gold_dist.most_common(1)[0][0] if gold_dist else ""
    pred_dominant = pred_dist.most_common(1)[0][0] if pred_dist else ""
    high_disagreement = (gold_dominant != pred_dominant) and total >= 2

    # Pattern flags
    patterns = []
    if n_errors > 0 and n_errors < total:
        patterns.append(
            f"Mostly consistent but has {n_errors}/{total} prediction error(s)."
        )
    if n_errors == total and total > 1:
        patterns.append("All predictions are errors for this patient.")
    if repeated_blocking:
        patterns.append(
            "Repeated blocking criteria across trials: "
            + "; ".join(repeated_blocking[:3])
        )
    if repeated_uncertain:
        patterns.append(
            "Repeated uncertain criteria across trials: "
            + "; ".join(repeated_uncertain[:3])
        )
    if high_disagreement:
        patterns.append(
            f"Label distribution mismatch: gold dominant={gold_dominant!r}, "
            f"predicted dominant={pred_dominant!r}."
        )

    # Top error examples (up to 3), sorted stably
    error_examples = sorted(errors, key=lambda r: r.get("trial_id", ""))[:3]
    examples = []
    for r in error_examples:
        examples.append({
            "trial_id": r.get("trial_id", ""),
            "gold_label": r.get("gold_label", ""),
            "predicted_label": r.get("predicted_label", "") or r.get("prediction", ""),
            "confidence": r.get("confidence"),
            "matcher_explanation": (r.get("matcher_explanation", "") or "")[:120],
        })

    return {
        "patient_id": patient_id,
        "total_trials": total,
        "n_errors": n_errors,
        "gold_distribution": dict(gold_dist),
        "predicted_distribution": dict(pred_dist),
        "repeated_blocking": repeated_blocking[:5],
        "repeated_uncertain": repeated_uncertain[:5],
        "patterns": patterns,
        "error_examples": examples,
    }


def build_consistency_summary(predictions: list) -> list:
    """
    Return a list of per-patient summary dicts, sorted by descending error count
    then patient_id.
    """
    grouped = group_predictions_by_patient(predictions)
    summaries = [
        summarize_patient_predictions(pid, records)
        for pid, records in grouped.items()
    ]
    return sorted(summaries, key=lambda s: (-s["n_errors"], s["patient_id"]))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _dist_str(dist: dict) -> str:
    if not dist:
        return "—"
    return ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))


def format_markdown_report(summary: list) -> str:
    """Format the consistency summary as a Markdown report."""
    n_patients = len(summary)
    n_with_errors = sum(1 for s in summary if s["n_errors"] > 0)

    lines = [
        "# Cross-Trial Patient Consistency Report",
        "",
        "This report groups predictions by patient and surfaces consistency patterns.",
        "An **error** is defined as `gold_label != predicted_label`.",
        "",
        f"- **Patients summarised**: {n_patients}",
        f"- **Patients with at least one error**: {n_with_errors}",
        "",
        "---",
        "",
        "## Overview Table",
        "",
        "| patient_id | Trials | Errors | Error Rate | Patterns |",
        "|---|---:|---:|---:|---|",
    ]

    for s in summary:
        rate = s["n_errors"] / s["total_trials"] if s["total_trials"] > 0 else 0.0
        pattern_str = "; ".join(s["patterns"])[:80] if s["patterns"] else "—"
        lines.append(
            f"| {s['patient_id']} "
            f"| {s['total_trials']} "
            f"| {s['n_errors']} "
            f"| {rate:.0%} "
            f"| {pattern_str} |"
        )

    lines += ["", "---", "", "## Patient Detail", ""]

    for s in summary:
        lines.append(f"### {s['patient_id']}")
        lines.append("")
        lines.append(f"- **Trials evaluated**: {s['total_trials']}")
        lines.append(f"- **Prediction errors**: {s['n_errors']}")
        lines.append(f"- **Gold label distribution**: {_dist_str(s['gold_distribution'])}")
        lines.append(f"- **Predicted label distribution**: {_dist_str(s['predicted_distribution'])}")

        if s["repeated_blocking"]:
            lines.append(
                "- **Repeated blocking criteria**: "
                + "; ".join(s["repeated_blocking"])
            )
        if s["repeated_uncertain"]:
            lines.append(
                "- **Repeated uncertain criteria**: "
                + "; ".join(s["repeated_uncertain"])
            )

        if s["patterns"]:
            lines.append("")
            lines.append("**Consistency patterns:**")
            for pat in s["patterns"]:
                lines.append(f"- {pat}")

        if s["error_examples"]:
            lines.append("")
            lines.append("**Error examples:**")
            lines.append("")
            lines.append("| trial_id | gold_label | predicted_label | confidence | explanation |")
            lines.append("|---|---|---|---|---|")
            for ex in s["error_examples"]:
                conf = f"{ex['confidence']:.2f}" if ex["confidence"] is not None else "—"
                expl = (ex["matcher_explanation"] or "—").replace("|", "\\|")
                lines.append(
                    f"| {ex['trial_id']} "
                    f"| {ex['gold_label']} "
                    f"| {ex['predicted_label']} "
                    f"| {conf} "
                    f"| {expl} |"
                )

        lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-trial patient consistency report."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Results JSON path (default: {DEFAULT_RESULTS})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    results = load_results(args.results)
    predictions = extract_predictions(results)
    summary = build_consistency_summary(predictions)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(f"Prediction records read : {len(predictions)}")
    print(f"Patients summarised     : {len(summary)}")
    print(f"Report written          : {args.output}")


if __name__ == "__main__":
    main()
