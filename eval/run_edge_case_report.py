"""
eval/run_edge_case_report.py

Task 61 — Edge case report.

Reads results_llm_reviewed.json (and optionally labels_llm_reviewed.json)
and writes a Markdown report of flagged edge cases.

Usage:
    PYTHONPATH=. python eval/run_edge_case_report.py
    PYTHONPATH=. python eval/run_edge_case_report.py --results path/to/results.json
    PYTHONPATH=. python eval/run_edge_case_report.py --output reports/my_report.md
"""

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_RESULTS_PATH = "data/processed/results_llm_reviewed.json"
DEFAULT_LABELS_PATH = "data/processed/labels_llm_reviewed.json"
DEFAULT_OUTPUT_PATH = "reports/edge_case_report.md"

BOUNDARY_KEYWORDS = [
    "age", "boundary", "threshold", "minimum", "maximum",
    "greater than", "less than", "at least", "no more than",
    "moca", "updrs", "hoehn", "years", "weeks", "months",
]

FLAG_PRIORITY = [
    "high_risk_false_eligible",
    "incorrect_high_confidence",
    "predicted_unclear",
    "gold_unclear",
    "low_confidence",
    "missing_confidence",
    "boundary_text",
]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_json(path: str, required: bool = False) -> Any:
    """Load JSON from path. Returns None if missing and not required."""
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_predictions(data: Any) -> list[dict]:
    """Extract list of prediction records from various JSON shapes."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "predictions", "items", "cases"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Cannot extract predictions: unexpected JSON shape.")


def index_labels(labels_data: Any) -> dict[str, dict]:
    """Return dict keyed by pair_key string from labels data."""
    if labels_data is None:
        return {}
    records = labels_data if isinstance(labels_data, list) else []
    if isinstance(labels_data, dict):
        for key in ("labels", "items", "cases"):
            if key in labels_data and isinstance(labels_data[key], list):
                records = labels_data[key]
                break
    index: dict[str, dict] = {}
    for r in records:
        k = pair_key(r)
        if k:
            index[k] = r
    return index


def pair_key(record: dict) -> str:
    """Return a consistent string key for a patient+trial pair."""
    pid = record.get("patient_id", "")
    tid = record.get("trial_id", "")
    return f"{pid}|{tid}"


def parse_confidence(value: Any) -> "float | None":
    """Return float confidence or None if missing/malformed."""
    if value is None:
        return None
    try:
        f = float(value)
        if 0.0 <= f <= 1.0:
            return f
        return None
    except (TypeError, ValueError):
        return None


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a truncated string preview of a value."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "; ".join(str(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\u2026"
    return text


def collect_text_for_boundary_detection(
    record: dict, label_record: "dict | None" = None
) -> str:
    """Collect all text fields relevant for boundary keyword detection."""
    parts = []
    for key in (
        "explanation", "matcher_explanation", "reason", "reasoning",
        "rationale", "blocking_criteria", "uncertain_criteria",
        "matched_facts", "notes",
    ):
        val = record.get(key)
        if val:
            parts.append(preview_text(val, max_chars=500))
    if label_record:
        for key in ("rationale", "label_status", "notes", "explanation"):
            val = label_record.get(key)
            if val:
                parts.append(preview_text(val, max_chars=500))
    return " ".join(parts).lower()


def detect_edge_case_flags(
    record: dict, label_record: "dict | None" = None
) -> list[str]:
    """Return list of edge case flag strings for this record."""
    flags: list[str] = []

    gold = record.get("gold_label") or record.get("gold") or record.get("label")
    pred = record.get("predicted_label") or record.get("prediction")
    conf_raw = record.get("confidence")
    conf = parse_confidence(conf_raw)

    if gold == "not_eligible" and pred == "eligible":
        flags.append("high_risk_false_eligible")

    if gold and pred and gold != pred and conf is not None and conf >= 0.75:
        flags.append("incorrect_high_confidence")

    if pred == "unclear":
        flags.append("predicted_unclear")

    if gold == "unclear":
        flags.append("gold_unclear")

    if conf_raw is None:
        flags.append("missing_confidence")
    elif conf is None:
        flags.append("missing_confidence")
    elif conf < 0.50:
        flags.append("low_confidence")

    combined = collect_text_for_boundary_detection(record, label_record)
    if any(kw in combined for kw in BOUNDARY_KEYWORDS):
        flags.append("boundary_text")

    return flags


def analyze_edge_cases(
    predictions: list[dict], label_index: dict[str, dict]
) -> dict:
    """Return summary dict with flagged cases and counts."""
    flagged: list[dict] = []
    flag_counts: dict[str, int] = {f: 0 for f in FLAG_PRIORITY}

    for record in predictions:
        key = pair_key(record)
        label_record = label_index.get(key)
        flags = detect_edge_case_flags(record, label_record)
        if not flags:
            continue

        gold = record.get("gold_label") or record.get("gold") or record.get("label")
        pred = record.get("predicted_label") or record.get("prediction")
        conf = parse_confidence(record.get("confidence"))

        explanation = preview_text(
            record.get("explanation") or record.get("matcher_explanation")
        )
        rationale = ""
        if label_record:
            rationale = preview_text(
                label_record.get("rationale") or label_record.get("label_status")
            )
        blocking = preview_text(record.get("blocking_criteria"))
        uncertain = preview_text(record.get("uncertain_criteria"))

        entry = {
            "patient_id": record.get("patient_id", "?"),
            "trial_id": record.get("trial_id", "?"),
            "gold_label": gold,
            "predicted_label": pred,
            "confidence": conf,
            "edge_case_flags": flags,
            "rationale_preview": rationale,
            "explanation_preview": explanation,
            "blocking_criteria_preview": blocking,
            "uncertain_criteria_preview": uncertain,
        }
        flagged.append(entry)

        for f in flags:
            if f in flag_counts:
                flag_counts[f] += 1

    def sort_key(e: dict) -> int:
        for i, f in enumerate(FLAG_PRIORITY):
            if f in e["edge_case_flags"]:
                return i
        return len(FLAG_PRIORITY)

    flagged.sort(key=sort_key)

    return {
        "total_records": len(predictions),
        "total_flagged": len(flagged),
        "flag_counts": flag_counts,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _md_record(entry: dict, idx: int) -> str:
    conf_str = (
        f"{entry['confidence']:.3f}" if entry["confidence"] is not None else "n/a"
    )
    lines = [
        f"#### {idx}. patient=`{entry['patient_id']}` trial=`{entry['trial_id']}`",
        f"- **gold:** {entry['gold_label']}  "
        f"**predicted:** {entry['predicted_label']}  "
        f"**confidence:** {conf_str}",
        f"- **flags:** {', '.join(entry['edge_case_flags'])}",
    ]
    if entry["rationale_preview"]:
        lines.append(f"- **rationale:** {entry['rationale_preview']}")
    if entry["explanation_preview"]:
        lines.append(f"- **explanation:** {entry['explanation_preview']}")
    if entry["blocking_criteria_preview"]:
        lines.append(f"- **blocking criteria:** {entry['blocking_criteria_preview']}")
    if entry["uncertain_criteria_preview"]:
        lines.append(f"- **uncertain criteria:** {entry['uncertain_criteria_preview']}")
    return "\n".join(lines)


def _section(title: str, flag: str, flagged: list[dict], limit: int = 20) -> str:
    subset = [e for e in flagged if flag in e["edge_case_flags"]][:limit]
    if not subset:
        return f"## {title}\n\n_No cases found._\n"
    lines = [f"## {title}\n"]
    for i, entry in enumerate(subset, 1):
        lines.append(_md_record(entry, i))
        lines.append("")
    return "\n".join(lines)


def format_markdown_report(summary: dict) -> str:
    total = summary["total_records"]
    flagged_count = summary["total_flagged"]
    flag_counts = summary["flag_counts"]
    flagged = summary["flagged"]
    top20 = flagged[:20]

    lines = [
        "# Edge Case Report",
        "",
        "## Summary",
        "",
        f"- **Total records:** {total}",
        f"- **Total flagged edge cases:** {flagged_count}",
        "",
        "### Counts by flag",
        "",
    ]
    for flag in FLAG_PRIORITY:
        count = flag_counts.get(flag, 0)
        lines.append(f"- `{flag}`: {count}")
    lines.append("")

    if flagged_count == 0:
        lines.append("_No edge cases found in this results file._\n")
        return "\n".join(lines)

    lines += ["## Top 20 Edge Cases (by priority)", ""]
    for i, entry in enumerate(top20, 1):
        lines.append(_md_record(entry, i))
        lines.append("")

    lines.append("")
    lines.append(_section("High-Risk False Eligible Cases", "high_risk_false_eligible", flagged))
    lines.append(_section("High-Confidence Incorrect Cases", "incorrect_high_confidence", flagged))
    lines.append(_section("Unclear Prediction Cases", "predicted_unclear", flagged))
    lines.append(_section("Gold Unclear Cases", "gold_unclear", flagged))
    lines.append(_section("Low Confidence Cases", "low_confidence", flagged))
    lines.append(_section("Missing Confidence Cases", "missing_confidence", flagged))
    lines.append(_section("Boundary-Like Cases", "boundary_text", flagged))

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Edge case report for benchmark results.")
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        results_data = load_json(args.results, required=True)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Could not load results: {exc}", file=sys.stderr)
        return 1

    try:
        predictions = extract_predictions(results_data)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    labels_data = load_json(args.labels, required=False)
    label_index = index_labels(labels_data)

    summary = analyze_edge_cases(predictions, label_index)
    report_text = format_markdown_report(summary)
    write_text(report_text, args.output)

    print(
        f"\nEdge case report: "
        f"records={summary['total_records']}  "
        f"flagged={summary['total_flagged']}  "
        f"output={args.output}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
