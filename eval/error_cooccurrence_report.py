"""
error_cooccurrence_report.py — Task 60: error co-occurrence report.

Usage:
    PYTHONPATH=. python eval/error_cooccurrence_report.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
ERROR_ANALYSIS_PATH = Path("data/processed/error_analysis_llm_reviewed.json")
REPORT_PATH = Path("reports/error_cooccurrence_report.md")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path, required: bool = True) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if required:
            raise
        return None
    except json.JSONDecodeError:
        if required:
            raise
        return None


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Field accessors
# ---------------------------------------------------------------------------

def extract_predictions(data: dict) -> list:
    return data.get("predictions", []) if isinstance(data, dict) else []


def index_error_analysis(error_data: Any) -> dict[tuple, dict]:
    """Return a dict keyed by (patient_id, trial_id) from error_analysis data."""
    index: dict[tuple, dict] = {}
    if error_data is None:
        return index
    records = error_data if isinstance(error_data, list) else error_data.get("errors", [])
    for rec in records:
        if not isinstance(rec, dict):
            continue
        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        if pid and tid:
            index[(pid, tid)] = rec
    return index


def pair_key(record: dict) -> tuple:
    return (record.get("patient_id", ""), record.get("trial_id", ""))


def get_gold_label(record: dict) -> str:
    return record.get("gold_label", "")


def get_predicted_label(record: dict) -> str:
    return record.get("predicted_label", "")


# ---------------------------------------------------------------------------
# Error tagging
# ---------------------------------------------------------------------------

def derive_error_tags(record: dict, error_record: dict | None = None) -> list[str]:
    gold = get_gold_label(record)
    pred = get_predicted_label(record)
    tags: list[str] = []

    if gold == pred:
        return []

    if gold == "not_eligible" and pred == "eligible":
        tags.append("false_eligible")
    if gold == "eligible" and pred == "not_eligible":
        tags.append("false_not_eligible")
    if pred == "unclear" and gold != "unclear":
        tags.append("false_unclear")
    if gold == "unclear" and pred != "unclear":
        tags.append("missed_unclear")

    if error_record:
        et = error_record.get("error_type", "")
        if et and et not in tags:
            tags.append(et)

    return tags


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_error_cooccurrence(predictions: list, error_index: dict) -> dict:
    # Build per-pair tagged records
    tagged: list[dict] = []
    for rec in predictions:
        if not isinstance(rec, dict):
            continue
        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        if not pid or not tid:
            continue
        gold = get_gold_label(rec)
        pred = get_predicted_label(rec)
        error_rec = error_index.get((pid, tid))
        tags = derive_error_tags(rec, error_rec)
        severity = (error_rec or {}).get("severity", "")
        confidence = rec.get("confidence")
        if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
            conf_val: float | None = float(confidence)
        else:
            conf_val = None

        tagged.append({
            "patient_id": pid,
            "trial_id": tid,
            "gold_label": gold,
            "predicted_label": pred,
            "error_tags": tags,
            "severity": severity,
            "confidence": conf_val,
            "is_error": gold != pred,
        })

    total = len(tagged)
    incorrect = [r for r in tagged if r["is_error"]]

    # Tag distribution
    tag_counts: dict[str, int] = defaultdict(int)
    for r in incorrect:
        for t in r["error_tags"]:
            tag_counts[t] += 1

    # Severity distribution
    severity_counts: dict[str, int] = defaultdict(int)
    for r in incorrect:
        if r["severity"]:
            severity_counts[r["severity"]] += 1

    # Pair-level co-occurrences (multiple tags on one record)
    pair_cooccurrences = [r for r in incorrect if len(r["error_tags"]) > 1]

    # Patient-level: distinct error tags across trials
    patient_tags: dict[str, set] = defaultdict(set)
    for r in incorrect:
        for t in r["error_tags"]:
            patient_tags[r["patient_id"]].add(t)

    # Trial-level: distinct error tags across patients
    trial_tags: dict[str, set] = defaultdict(set)
    for r in incorrect:
        for t in r["error_tags"]:
            trial_tags[r["trial_id"]].add(t)

    top_patients = sorted(
        patient_tags.items(), key=lambda x: len(x[1]), reverse=True
    )[:10]
    top_trials = sorted(
        trial_tags.items(), key=lambda x: len(x[1]), reverse=True
    )[:10]

    # Co-occurrence pattern examples: group incorrect by sorted tag tuple
    pattern_examples: dict[str, list] = defaultdict(list)
    for r in incorrect:
        if r["error_tags"]:
            key = "+".join(sorted(r["error_tags"]))
            pattern_examples[key].append(r)

    return {
        "total": total,
        "total_incorrect": len(incorrect),
        "tag_counts": dict(tag_counts),
        "severity_counts": dict(severity_counts),
        "pair_cooccurrences": pair_cooccurrences,
        "patient_tags": {k: sorted(v) for k, v in patient_tags.items()},
        "trial_tags": {k: sorted(v) for k, v in trial_tags.items()},
        "top_patients": [(pid, sorted(tags)) for pid, tags in top_patients],
        "top_trials": [(tid, sorted(tags)) for tid, tags in top_trials],
        "pattern_examples": {k: v[:3] for k, v in pattern_examples.items()},
        "all_patient_ids": sorted(patient_tags.keys()),
        "all_trial_ids": sorted(trial_tags.keys()),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_conf(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "n/a"


def format_markdown_report(summary: dict) -> str:
    lines = [
        "# Error Co-occurrence Report",
        "",
        "## Summary",
        "",
        f"- Total prediction records: {summary['total']}",
        f"- Total incorrect records: {summary['total_incorrect']}",
        f"- Pair-level co-occurrences (multiple tags): {len(summary['pair_cooccurrences'])}",
        f"- Patients with any error tags: {len(summary['patient_tags'])}",
        f"- Trials with any error tags: {len(summary['trial_tags'])}",
        "",
        "## Error Tag Distribution",
        "",
        "| error_tag | count |",
        "|---|---|",
    ]
    for tag, count in sorted(summary["tag_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {tag} | {count} |")

    if summary["severity_counts"]:
        lines += [
            "",
            "## Severity Distribution",
            "",
            "| severity | count |",
            "|---|---|",
        ]
        for sev, count in sorted(summary["severity_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"| {sev} | {count} |")

    lines += [
        "",
        "## Top 10 Patients with Most Distinct Error Tags",
        "",
        "| patient_id | distinct_tags | tags |",
        "|---|---|---|",
    ]
    for pid, tags in summary["top_patients"]:
        lines.append(f"| {pid} | {len(tags)} | {', '.join(tags)} |")

    lines += [
        "",
        "## Top 10 Trials with Most Distinct Error Tags",
        "",
        "| trial_id | distinct_tags | tags |",
        "|---|---|---|",
    ]
    for tid, tags in summary["top_trials"]:
        lines.append(f"| {tid} | {len(tags)} | {', '.join(tags)} |")

    # Patient × error-tag table
    all_tags = sorted(summary["tag_counts"].keys())
    if all_tags and summary["all_patient_ids"]:
        header = "| patient_id | " + " | ".join(all_tags) + " |"
        sep = "|---|" + "---|" * len(all_tags)
        lines += ["", "## Patient × Error-Tag Table", "", header, sep]
        for pid in summary["all_patient_ids"]:
            ptags = set(summary["patient_tags"].get(pid, []))
            cells = ["✓" if t in ptags else "" for t in all_tags]
            lines.append(f"| {pid} | " + " | ".join(cells) + " |")

    # Trial × error-tag table
    if all_tags and summary["all_trial_ids"]:
        header = "| trial_id | " + " | ".join(all_tags) + " |"
        sep = "|---|" + "---|" * len(all_tags)
        lines += ["", "## Trial × Error-Tag Table", "", header, sep]
        for tid in summary["all_trial_ids"]:
            ttags = set(summary["trial_tags"].get(tid, []))
            cells = ["✓" if t in ttags else "" for t in all_tags]
            lines.append(f"| {tid} | " + " | ".join(cells) + " |")

    # Pair-level co-occurrences
    if summary["pair_cooccurrences"]:
        lines += [
            "",
            "## Pair-Level Co-occurrences",
            "",
            "| patient_id | trial_id | gold_label | predicted_label | error_tags | severity | confidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in summary["pair_cooccurrences"]:
            lines.append(
                f"| {r['patient_id']} | {r['trial_id']} | {r['gold_label']} "
                f"| {r['predicted_label']} | {', '.join(r['error_tags'])} "
                f"| {r['severity'] or 'n/a'} | {_fmt_conf(r['confidence'])} |"
            )
    else:
        lines += ["", "## Pair-Level Co-occurrences", "", "_No single record carries multiple error tags._"]

    # Pattern examples
    if summary["pattern_examples"]:
        lines += ["", "## Co-occurrence Pattern Examples", ""]
        for pattern, examples in sorted(summary["pattern_examples"].items()):
            lines += [
                f"### Pattern: `{pattern}`",
                "",
                "| patient_id | trial_id | gold_label | predicted_label | severity | confidence |",
                "|---|---|---|---|---|---|",
            ]
            for r in examples:
                lines.append(
                    f"| {r['patient_id']} | {r['trial_id']} | {r['gold_label']} "
                    f"| {r['predicted_label']} | {r['severity'] or 'n/a'} "
                    f"| {_fmt_conf(r['confidence'])} |"
                )
            lines.append("")

    lines += ["---", "_Generated by eval/error_cooccurrence_report.py_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        results_data = load_json(RESULTS_PATH, required=True)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {RESULTS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {RESULTS_PATH}: {exc}")
        sys.exit(1)

    error_data = load_json(ERROR_ANALYSIS_PATH, required=False)
    error_index = index_error_analysis(error_data)

    predictions = extract_predictions(results_data)
    if not predictions:
        print("[ERROR] No predictions found in results file.")
        sys.exit(1)

    summary = analyze_error_cooccurrence(predictions, error_index)
    report = format_markdown_report(summary)
    write_text(report, REPORT_PATH)

    multi_error_patients = sum(
        1 for tags in summary["patient_tags"].values() if len(tags) > 1
    )
    multi_error_trials = sum(
        1 for tags in summary["trial_tags"].values() if len(tags) > 1
    )

    print(f"Records read:                  {summary['total']}")
    print(f"Incorrect records:             {summary['total_incorrect']}")
    print(f"Patients with multiple tags:   {multi_error_patients}")
    print(f"Trials with multiple tags:     {multi_error_trials}")
    print(f"Report written to:             {REPORT_PATH}")


if __name__ == "__main__":
    main()
