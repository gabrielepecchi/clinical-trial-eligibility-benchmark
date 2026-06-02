"""
Task 95: Capability tag report.

Assigns deterministic capability tags to each prediction pair based on
trial criteria text and criterion-level text, then writes a tagged CSV
and a Markdown summary report.

Tagging is purely keyword-based. No model calls, no external libraries.

Usage:
    PYTHONPATH=. python eval/run_capability_tag_report.py
    PYTHONPATH=. python eval/run_capability_tag_report.py \
        --results  data/processed/results_llm_reviewed.json \
        --trials   data/processed/trial_cases.json \
        --criteria data/processed/criterion_level_results.csv \
        --csv      data/processed/capability_tagged_predictions.csv \
        --output   reports/capability_tag_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_TRIALS = "data/processed/trial_cases.json"
DEFAULT_CRITERIA = "data/processed/criterion_level_results.csv"
DEFAULT_CSV = "data/processed/capability_tagged_predictions.csv"
DEFAULT_OUTPUT = "reports/capability_tag_report.md"

ALL_TAGS: list[str] = [
    "age_check",
    "diagnosis_check",
    "medication_check",
    "procedure_check",
    "cognitive_check",
    "device_check",
    "severity_check",
    "temporal_check",
    "comorbidity_check",
    "reproductive_check",
    "lab_check",
    "missing_info_check",
    "exclusion_logic",
    "inclusion_logic",
    "uncertainty_handling",
]

# Keywords per tag (matched against normalized text)
TAG_KEYWORDS: dict[str, list[str]] = {
    "age_check": [
        "age", "year", "years old", "aged", "minimum age", "maximum age",
        "at least", "no older", "no younger",
    ],
    "diagnosis_check": [
        "diagnosis", "idiopathic", "parkinson", "parkinsonism", "diagnosed",
        "disease type", "confirmed diagnosis", "atypical",
    ],
    "medication_check": [
        "medication", "drug", "levodopa", "carbidopa", "dopamine agonist",
        "mao-b", "rasagiline", "selegiline", "safinamide", "comt",
        "inhibitor", "stable dose", "current use", "therapy",
    ],
    "procedure_check": [
        "dbs", "deep brain stimulation", "surgery", "procedure",
        "prior surgery", "surgical", "ablation", "implant",
        "prior dbs", "brain stimulation",
    ],
    "cognitive_check": [
        "moca", "mmse", "cognitive", "cognition", "montreal cognitive",
        "mental status", "memory", "dementia", "cognitive impairment",
        "mini-mental",
    ],
    "device_check": [
        "pacemaker", "defibrillator", "implanted device", "cardiac device",
        "icd", "ivcd", "implanted cardioverter", "device",
    ],
    "severity_check": [
        "updrs", "hoehn", "yahr", "h&y", "severity", "stage",
        "disease stage", "motor score", "disability",
    ],
    "temporal_check": [
        "within", "days", "weeks", "months", "washout", "duration",
        "recently", "prior to", "last", "since", "onset",
        "years with", "stable for",
    ],
    "comorbidity_check": [
        "cardiac", "heart", "renal", "kidney", "liver", "hepatic",
        "comorbid", "hypertension", "diabetes", "arrhythmia",
        "depression", "anxiety", "significant disease",
    ],
    "reproductive_check": [
        "pregnant", "pregnancy", "reproductive", "postmenopausal",
        "menopause", "contraception", "childbearing",
    ],
    "lab_check": [
        "lab", "laboratory", "creatinine", "ast", "alt",
        "blood test", "serum", "value", "result",
    ],
    "missing_info_check": [
        "unclear", "unknown", "not documented", "not specified",
        "missing", "insufficient", "not available", "cannot determine",
        "uncertain", "undocumented",
    ],
    "exclusion_logic": [
        "exclusion", "excluded", "exclude", "not eligible", "ineligible",
        "must not", "no history", "contraindication", "prohibited",
    ],
    "inclusion_logic": [
        "inclusion", "must have", "required", "requirement", "meets",
        "satisfies", "confirmed", "eligible",
    ],
    "uncertainty_handling": [
        "unclear", "uncertain", "ambiguous", "unclear", "not documented",
        "cannot determine", "insufficient data", "requires clarification",
    ],
}

TRIAL_CRITERIA_FIELDS: list[str] = [
    "criteria_text", "eligibility_criteria", "inclusion_criteria",
    "exclusion_criteria", "criteria", "inclusion", "exclusion",
    "inclusion_text", "exclusion_text",
]

CSV_COLUMNS: list[str] = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "correct", "confidence", "capability_tags", "tag_count",
    "source_text_preview",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str, required: bool = False) -> Any:
    """Load JSON from *path*. Exits non-zero when *required* and missing or
    malformed; returns None silently when not required."""
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def load_csv_rows(path: str, required: bool = False) -> list[dict[str, str]] | None:
    """Load CSV as list of dicts. Returns None if missing and not required."""
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Write *rows* to CSV at *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_predictions(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of prediction records from *data*."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("predictions", "results", "pairs", "evaluations"):
            if key in data and isinstance(data[key], list):
                return data[key]
        candidates = [v for v in data.values() if isinstance(v, dict)]
        if candidates:
            return candidates
    print("ERROR: unrecognised structure in results file.", file=sys.stderr)
    sys.exit(1)


def extract_trial_records(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of trial records from *data*."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("trials", "cases", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [
            {**v, "trial_id": k} if "trial_id" not in v else v
            for k, v in data.items()
            if isinstance(v, dict)
        ]
    return []


def pair_key(record: dict[str, Any]) -> str:
    """Return a stable lookup key for *record*."""
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    pair = str(record.get("pair_id", "")).strip()
    if pair:
        return pair
    if pid and tid:
        return f"{pid}__{tid}"
    return pid or tid


def index_trials(trial_data: Any) -> dict[str, dict[str, Any]]:
    """Return {trial_id: trial_record}."""
    records = extract_trial_records(trial_data)
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        tid = str(rec.get("trial_id", rec.get("id", ""))).strip()
        if tid:
            index[tid] = rec
    return index


def index_criterion_text(rows: list[dict[str, str]] | None) -> dict[str, str]:
    """Return {trial_id: concatenated_criterion_text} from CSV rows."""
    if not rows:
        return {}
    crit_col: str | None = None
    for candidate in ("criterion", "criterion_text", "text", "eligibility_criterion"):
        if candidate in (rows[0] if rows else {}):
            crit_col = candidate
            break
    if not crit_col:
        return {}
    by_trial: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tid = row.get("trial_id", "").strip()
        val = row.get(crit_col, "").strip()
        if tid and val:
            by_trial[tid].append(val)
    return {tid: " ".join(texts) for tid, texts in by_trial.items()}


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def get_gold_label(record: dict[str, Any]) -> str:
    return str(record.get("gold_label", record.get("gold", ""))).strip().lower()


def get_predicted_label(record: dict[str, Any]) -> str:
    return str(
        record.get("predicted_label", record.get("prediction", record.get("predicted", "")))
    ).strip().lower()


def normalize_text(value: Any) -> str:
    """Return a normalized lowercase text representation."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(i).strip() for i in value if i).lower()
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v is not None).lower()
    return str(value).lower().strip()


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a short preview of *value*."""
    text = normalize_text(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def collect_tagging_text(
    record: dict[str, Any],
    trial_index: dict[str, dict[str, Any]],
    criterion_index: dict[str, str],
) -> str:
    """Collect all text sources for capability tagging."""
    parts: list[str] = []

    # From trial_cases.json
    tid = str(record.get("trial_id", "")).strip()
    trial_rec = trial_index.get(tid, {})
    for field in TRIAL_CRITERIA_FIELDS:
        val = trial_rec.get(field)
        if val:
            parts.append(normalize_text(val))

    # From criterion_level_results.csv
    crit_text = criterion_index.get(tid, "")
    if crit_text:
        parts.append(crit_text.lower())

    # From prediction record itself
    for field in (
        "explanation", "matcher_explanation", "reason",
        "blocking_criteria", "uncertain_criteria",
    ):
        val = record.get(field, "")
        if val:
            parts.append(normalize_text(val))

    return " ".join(parts)


def assign_capability_tags(text: str) -> list[str]:
    """
    Return a sorted list of capability tags matched in *text*.

    Falls back to ['other'] if no tags match.
    """
    if not text:
        return ["other"]

    matched: list[str] = []
    for tag, keywords in TAG_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(tag)
                break  # one match per tag is enough

    return sorted(set(matched)) if matched else ["other"]


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def build_tagged_rows(
    predictions: list[dict[str, Any]],
    trial_index: dict[str, dict[str, Any]],
    criterion_index: dict[str, str],
) -> list[dict[str, Any]]:
    """Build one output row per prediction with capability tags."""
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        gold = get_gold_label(pred)
        predicted = get_predicted_label(pred)
        correct_raw = pred.get("correct", "")
        correct = str(correct_raw).lower() not in ("false", "0", "no") and correct_raw is not False

        tagging_text = collect_tagging_text(pred, trial_index, criterion_index)
        tags = assign_capability_tags(tagging_text)

        rows.append(
            {
                "patient_id": pred.get("patient_id", ""),
                "trial_id": pred.get("trial_id", ""),
                "gold_label": gold,
                "predicted_label": predicted,
                "correct": correct,
                "confidence": pred.get("confidence", pred.get("confidence_score", "")),
                "capability_tags": "|".join(tags),
                "tag_count": len(tags),
                "source_text_preview": preview_text(tagging_text),
                # Keep raw tags for analysis
                "_tags": tags,
                "_incorrect": str(correct_raw).lower() in ("false", "0", "no") or correct_raw is False,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_tagged_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics over tagged rows."""
    total = len(rows)
    non_other = [r for r in rows if r["_tags"] != ["other"]]
    only_other = [r for r in rows if r["_tags"] == ["other"]]

    # Counts by tag
    tag_counts: dict[str, int] = defaultdict(int)
    tag_correct: dict[str, int] = defaultdict(int)
    tag_total: dict[str, int] = defaultdict(int)

    for row in rows:
        for tag in row["_tags"]:
            tag_counts[tag] += 1
            tag_total[tag] += 1
            if not row["_incorrect"]:
                tag_correct[tag] += 1

    # Accuracy by tag
    tag_accuracy: dict[str, float | None] = {}
    for tag in tag_total:
        if tag_total[tag] > 0:
            gold_present = any(
                r["gold_label"] for r in rows if tag in r["_tags"]
            )
            if gold_present:
                tag_accuracy[tag] = tag_correct[tag] / tag_total[tag]
            else:
                tag_accuracy[tag] = None

    # Top 10 tag combinations
    combo_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        combo = "|".join(row["_tags"])
        combo_counts[combo] += 1
    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:10]

    # Examples per tag (up to 3 each)
    examples_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for tag in row["_tags"]:
            if len(examples_by_tag[tag]) < 3:
                examples_by_tag[tag].append(row)

    return {
        "total": total,
        "non_other_count": len(non_other),
        "only_other_count": len(only_other),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "tag_accuracy": tag_accuracy,
        "tag_total": dict(tag_total),
        "top_combos": top_combos,
        "examples_by_tag": dict(examples_by_tag),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _display_id(row: dict[str, Any]) -> str:
    pid = str(row.get("patient_id", ""))
    tid = str(row.get("trial_id", ""))
    if pid and tid:
        return f"{pid} / {tid}"
    return pid or tid or "(no id)"


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Capability Tag Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> Tags are assigned by deterministic keyword matching on trial criteria text,  "
    )
    lines.append(
        "> criterion-level text, and prediction fields. No model calls are used.  "
    )
    lines.append("> Accuracy figures are based on gold/predicted labels in the results file.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total prediction records | {summary['total']} |")
    lines.append(f"| Records with at least one non-other tag | {summary['non_other_count']} |")
    lines.append(f"| Records tagged only 'other' | {summary['only_other_count']} |")
    lines.append("")

    lines.append("## Counts and Accuracy by Capability Tag")
    lines.append("")
    lines.append("| Tag | Count | Accuracy |")
    lines.append("|-----|-------|----------|")
    for tag, count in summary["tag_counts"].items():
        acc = summary["tag_accuracy"].get(tag)
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        lines.append(f"| {tag} | {count} | {acc_str} |")
    lines.append("")

    lines.append("## Top 10 Tag Combinations")
    lines.append("")
    lines.append("| Combination | Count |")
    lines.append("|-------------|-------|")
    for combo, count in summary["top_combos"]:
        lines.append(f"| {combo} | {count} |")
    lines.append("")

    lines.append("## Examples by Capability Tag")
    lines.append("")
    for tag in ALL_TAGS + ["other"]:
        examples = summary["examples_by_tag"].get(tag, [])
        if not examples:
            continue
        lines.append(f"### {tag}")
        lines.append("")
        for row in examples:
            rid = _display_id(row)
            lines.append(
                f"- **{rid}**: gold=`{row['gold_label']}`, "
                f"predicted=`{row['predicted_label']}`, correct=`{row['correct']}`"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capability tag report (Task 95)."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--trials", default=DEFAULT_TRIALS,
        help=f"Path to trial_cases.json (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--criteria", default=DEFAULT_CRITERIA,
        help=f"Path to criterion CSV (default: {DEFAULT_CRITERIA})",
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    trial_data = load_json(args.trials, required=False)
    criterion_rows = load_csv_rows(args.criteria, required=False)

    predictions = extract_predictions(results_data)
    trial_index = index_trials(trial_data)
    criterion_index = index_criterion_text(criterion_rows)

    tagged_rows = build_tagged_rows(predictions, trial_index, criterion_index)
    summary = analyze_tagged_rows(tagged_rows)

    write_csv(tagged_rows, args.csv)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Capability tag CSV written to   : {args.csv}\n"
        f"Capability tag report written to: {args.output}\n"
        f"  Records tagged         : {summary['total']}\n"
        f"  Non-other tagged       : {summary['non_other_count']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
