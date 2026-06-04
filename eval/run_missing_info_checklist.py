"""
Task 71: Missing-information checklist report.

Analyzes cases marked as unclear or with uncertain criteria and generates
a conservative checklist of what patient information would be needed to
resolve eligibility.

Usage:
    PYTHONPATH=. python eval/run_missing_info_checklist.py
    PYTHONPATH=. python eval/run_missing_info_checklist.py \
        --results data/processed/results_llm_reviewed.json \
        --labels  data/processed/labels_llm_reviewed.json \
        --criteria data/processed/criterion_level_results.csv \
        --output  reports/missing_info_checklist.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_CRITERIA = "data/processed/criterion_level_results.csv"
DEFAULT_OUTPUT = "reports/missing_info_checklist.md"

# Keywords that signal missing information needs
MISSING_INFO_KEYWORDS: dict[str, list[str]] = {
    "age": ["age", "year.old"],
    "sex": ["sex", "gender", "male", "female"],
    "diagnosis details": [
        "diagnosis", "idiopathic", "symptomatic", "disease type",
    ],
    "disease duration": [
        "duration", "since onset", "years with", "disease duration",
        "symptom onset", "when diagnosed",
    ],
    "medication list": [
        "medication", "drug", "levodopa", "dopamine agonist",
        "mao-b", "comt inhibitor",
    ],
    "medication stability": [
        "stability", "dose stable", "weeks stable", "months stable",
        "unchanged", "washout",
    ],
    "MAO-B inhibitor status": [
        "mao-b", "rasagiline", "selegiline", "safinamide", "maoi",
    ],
    "DBS / procedure history": [
        "dbs", "deep brain stimulation", "procedure", "surgery",
        "implant", "prior surgery",
    ],
    "implanted device": [
        "pacemaker", "implanted", "cardiac device", "defibrillator",
        "ivcd", "icd", "device",
    ],
    "cognitive status": [
        "cognitive", "moca", "mmse", "mental status", "cognition",
        "memory", "dementia",
    ],
    "MoCA score": ["moca", "montreal cognitive"],
    "UPDRS score": ["updrs", "unified parkinson"],
    "Hoehn-Yahr stage": ["hoehn", "yahr", "h&y"],
    "comorbidities": [
        "comorbid", "comorbidity", "depression", "anxiety",
        "parkinson", "condition",
    ],
    "cardiac history": [
        "cardiac", "heart", "arrhythmia", "hypertension", "infarction",
        "disease", "myocardial",
    ],
    "renal/liver function": [
        "renal", "kidney", "liver", "hepatic", "creatinine",
        "gfr", "ast", "alt",
    ],
    "lab values": [
        "lab", "laboratory", "value", "blood", "serum", "result",
    ],
    "pregnancy / reproductive": [
        "pregnant", "pregnancy", "reproductive", "menopause",
        "postmenopausal",
    ],
    "recent trial participation": [
        "trial", "study", "research", "prior participation",
        "recent study",
    ],
    "administrative / consent": [
        "consent", "able to consent", "willing", "capacity",
        "informed", "enrollment",
    ],
}

# Keywords that signal uncertainty or missing info in text
UNCERTAINTY_TRIGGERS: list[str] = [
    "unclear",
    "uncertain",
    "unknown",
    "not documented",
    "not specified",
    "not provided",
    "missing",
    "insufficient",
    "insufficient data",
    "incomplete",
    "undefined",
    "not available",
    "not reported",
    "cannot determine",
    "cannot assess",
    "not assessed",
    "requires",
    "need",
    "need to",
    "without knowing",
    "without information about",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str, required: bool = False) -> Any:
    """Load JSON from *path*. Exits non-zero when *required* and file is
    missing or malformed; returns None silently when not required."""
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
    """Load CSV as a list of dicts. Returns None if file is missing and not required."""
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


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


def index_labels(labels_data: Any) -> dict[str, dict[str, Any]]:
    """Return {pair_key: label_record} from *labels_data*; empty dict if None."""
    if labels_data is None:
        return {}
    records: list[dict[str, Any]] = []
    if isinstance(labels_data, list):
        records = labels_data
    elif isinstance(labels_data, dict):
        for key in ("labels", "pairs", "records"):
            if key in labels_data and isinstance(labels_data[key], list):
                records = labels_data[key]
                break
        else:
            records = [v for v in labels_data.values() if isinstance(v, dict)]
    return {pair_key(r): r for r in records if isinstance(r, dict)}


def index_criterion_rows(rows: list[dict[str, str]] | None) -> dict[str, list[dict[str, str]]]:
    """Return {trial_id: [criterion_rows]} from CSV rows; empty dict if None."""
    if not rows:
        return {}
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        tid = row.get("trial_id", "").strip()
        if tid:
            index[tid].append(row)
    return dict(index)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def get_gold_label(record: dict[str, Any]) -> str:
    return str(record.get("gold_label", record.get("gold", ""))).strip().lower()


def get_predicted_label(record: dict[str, Any]) -> str:
    return str(
        record.get("predicted_label", record.get("prediction", record.get("predicted", "")))
    ).strip().lower()


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a short plain-text preview of *value*."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "; ".join(str(i).strip() for i in value if i)
    elif isinstance(value, dict):
        text = "; ".join(f"{k}: {v}" for k, v in value.items() if v is not None)
    else:
        text = str(value)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Missing info inference
# ---------------------------------------------------------------------------


def collect_text_for_missing_info(
    record: dict[str, Any],
    label_record: dict[str, Any] | None = None,
    criterion_rows: list[dict[str, str]] | None = None,
) -> str:
    """Collect all text fields that might contain hints about missing info."""
    texts: list[str] = []

    # From prediction record
    for field in (
        "explanation",
        "matcher_explanation",
        "reason",
        "blocked_by",
        "matched_criteria",
    ):
        val = record.get(field, "")
        if val:
            texts.append(str(val))

    # Criteria fields
    for field in ("blocking_criteria", "uncertain_criteria"):
        val = record.get(field, "")
        if val:
            texts.append(str(val))

    # From label record if present
    if label_record:
        for field in ("rationale", "explanation"):
            val = label_record.get(field, "")
            if val:
                texts.append(str(val))

    # From criterion rows if present
    if criterion_rows:
        for row in criterion_rows:
            for field in ("criterion", "criterion_text", "text"):
                val = row.get(field, "")
                if val:
                    texts.append(str(val))

    return " ".join(texts).lower()


def infer_missing_info_items(text: str) -> list[str]:
    """
    Return a list of missing-information items inferred from *text*.

    Uses keyword matching in a conservative manner: only suggests an item
    if multiple related keywords appear in the text.
    """
    if not text:
        return []

    suggested: dict[str, int] = defaultdict(int)

    for item, keywords in MISSING_INFO_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        if count >= 1:  # At least one keyword match
            suggested[item] += count

    # Also check for uncertainty triggers that might hint at missing info
    uncertainty_found = any(trigger in text for trigger in UNCERTAINTY_TRIGGERS)

    result: list[str] = []
    for item, count in sorted(suggested.items(), key=lambda x: -x[1]):
        result.append(item)

    # If uncertainty was mentioned but no specific items matched, add "other"
    if uncertainty_found and not result:
        result.append("other")

    return result


def should_include_record(record: dict[str, Any], collected_text: str) -> bool:
    """Return True if *record* should be included in the missing-info report."""
    predicted = get_predicted_label(record)
    gold = get_gold_label(record)
    uncertain_criteria = record.get("uncertain_criteria")
    blocked_by = record.get("blocked_by")

    # Include if predicted is unclear
    if predicted == "unclear":
        return True

    # Include if gold is unclear
    if gold == "unclear":
        return True

    # Include if uncertain_criteria is present and non-empty
    if uncertain_criteria:
        uc_text = preview_text(uncertain_criteria)
        if uc_text:
            return True

    # Include if blocked_by is present (indicates criterion blocking)
    if blocked_by:
        bb_text = preview_text(blocked_by)
        if bb_text:
            return True

    # Include if collected text contains uncertainty language
    for trigger in UNCERTAINTY_TRIGGERS:
        if trigger in collected_text:
            return True

    return False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_missing_info(
    predictions: list[dict[str, Any]],
    label_index: dict[str, dict[str, Any]],
    criterion_index: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Analyze predictions and return summary of missing-info cases."""
    cases: list[dict[str, Any]] = []
    missing_reason_type_counts: dict[str, int] = defaultdict(int)

    for pred in predictions:
        key = pair_key(pred)
        label_rec = label_index.get(key, {})
        trial_id = str(pred.get("trial_id", "")).strip()
        criterion_rows = criterion_index.get(trial_id, [])

        collected = collect_text_for_missing_info(pred, label_rec, criterion_rows)
        if not should_include_record(pred, collected):
            continue

        # Use structured missingness fields when available
        unknown_fields: list[str] = pred.get("unknown_fields") or []
        absent_evidence: list[str] = pred.get("absent_evidence") or []
        present_evidence_list: list[str] = pred.get("present_evidence") or []
        missing_info_details: list[dict] = pred.get("missing_information_details") or []
        missing_reason_type: str = pred.get("missing_reason_type") or ""
        unclear_reason: str = pred.get("unclear_reason") or ""

        # Count missing_reason_type
        if missing_reason_type:
            missing_reason_type_counts[missing_reason_type] += 1
        for detail in missing_info_details:
            rt = detail.get("missing_reason_type") or ""
            if rt and rt != missing_reason_type:
                missing_reason_type_counts[rt] += 1

        # Fall back to text-based inference when structured fields are absent
        if unknown_fields or missing_info_details:
            # Structured path: use unknown_fields directly as missing items
            missing_items = list(unknown_fields)
            # Also add fields from details that are unknown but not already listed
            for detail in missing_info_details:
                if detail.get("status") == "unknown" and detail["field"] not in missing_items:
                    missing_items.append(detail["field"])
            if not missing_items:
                # All known present/absent — still record for unknown-field breakdown
                missing_items = infer_missing_info_items(collected)
        else:
            missing_items = infer_missing_info_items(collected)

        confidence = pred.get("confidence", pred.get("confidence_score"))

        case = {
            "patient_id": pred.get("patient_id", ""),
            "trial_id": trial_id,
            "pair_id": pred.get("pair_id", ""),
            "gold_label": get_gold_label(pred),
            "predicted_label": get_predicted_label(pred),
            "confidence": confidence,
            "missing_info_items": missing_items,
            "collected_text": collected,
            "explanation_preview": preview_text(
                pred.get("explanation", pred.get("matcher_explanation", ""))
            ),
            "uncertain_preview": preview_text(pred.get("uncertain_criteria", "")),
            "rationale_preview": preview_text(
                label_rec.get("rationale", label_rec.get("explanation", ""))
            ),
            # Structured fields
            "unknown_fields": unknown_fields,
            "absent_evidence": absent_evidence,
            "present_evidence": present_evidence_list,
            "missing_reason_type": missing_reason_type,
            "unclear_reason": unclear_reason,
            "missing_information_details": missing_info_details,
        }
        cases.append(case)

    # Sort by number of missing items (longest list first)
    cases.sort(key=lambda c: (-len(c["missing_info_items"]), str(c["patient_id"])))

    # Count missing info items across all cases
    item_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        for item in case["missing_info_items"]:
            item_counts[item] += 1

    # Group cases by item
    cases_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        for item in case["missing_info_items"]:
            cases_by_item[item].append(case)

    return {
        "total_records": len(predictions),
        "total_cases": len(cases),
        "cases": cases,
        "item_counts": dict(sorted(item_counts.items(), key=lambda x: -x[1])),
        "cases_by_item": dict(cases_by_item),
        "missing_reason_type_counts": dict(missing_reason_type_counts),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _display_id(case: dict[str, Any]) -> str:
    pid = case.get("patient_id", "")
    tid = case.get("trial_id", "")
    pair = case.get("pair_id", "")
    if pid and tid:
        return f"{pid} / {tid}"
    return pair or pid or tid or "(no id)"


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Return a Markdown string for the summary."""
    lines: list[str] = []

    total_records: int = summary["total_records"]
    total_cases: int = summary["total_cases"]
    cases: list[dict[str, Any]] = summary["cases"]
    item_counts: dict[str, int] = summary["item_counts"]
    cases_by_item: dict[str, list[dict[str, Any]]] = summary["cases_by_item"]

    lines.append("# Missing Information Checklist Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> This report identifies structured unknown/missing information fields for cases  "
    )
    lines.append(
        "> marked as unclear or with uncertain criteria. It reports what data is absent  "
    )
    lines.append(
        "> or not documented — **not** proof of ineligibility. When structured missingness  "
    )
    lines.append(
        "> fields are available (unknown_fields, missing_information_details), they are  "
    )
    lines.append(
        "> used directly; otherwise inference falls back to keyword matching on text fields.  "
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total records | {total_records} |")
    lines.append(f"| Cases with unknown/missing information | {total_cases} |")
    lines.append("")

    if total_cases == 0:
        lines.append("**No cases have unknown or missing structured information.**")
        lines.append("")
        return "\n".join(lines)

    # Missing reason type counts if available
    reason_type_counts: dict[str, int] = summary.get("missing_reason_type_counts", {})
    if reason_type_counts:
        lines.append("## Missing Reason Type Counts")
        lines.append("")
        lines.append("| Reason Type | Count |")
        lines.append("|-------------|-------|")
        for rt, cnt in sorted(reason_type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {rt} | {cnt} |")
        lines.append("")

    # Item counts
    lines.append("## Unknown / Missing Information Item Counts")
    lines.append("")
    lines.append("_(Items are unknown fields — not documented and not negated. This is not a list of ineligibility reasons.)_")
    lines.append("")
    lines.append("| Information Type | Case Count |")
    lines.append("|------------------|-----------|")
    for item, count in item_counts.items():
        lines.append(f"| {item} | {count} |")
    lines.append("")

    # Top 25 cases
    lines.append("## Top 25 Cases by Unknown Field Count")
    lines.append("")
    for i, case in enumerate(cases[:25], 1):
        rid = _display_id(case)
        # Prefer structured unknown_fields label if available
        uf = case.get("unknown_fields") or []
        if uf:
            missing_label = ", ".join(f"`{m}`" for m in uf)
            structured_note = " _(structured)_"
        else:
            missing_label = ", ".join(f"`{m}`" for m in case["missing_info_items"])
            structured_note = " _(inferred)_"
        lines.append(
            f"{i}. **{rid}** ({len(case['missing_info_items'])} items{structured_note}) — "
            f"{missing_label}"
        )
    lines.append("")

    # Examples by item
    lines.append("## Unknown / Missing Information Examples by Type")
    lines.append("")
    for item in sorted(item_counts.keys()):
        item_cases = cases_by_item[item][:5]
        lines.append(f"### {item}")
        lines.append("")
        lines.append(
            f"Found in **{item_counts[item]}** cases. Examples:"
        )
        lines.append("")
        for case in item_cases:
            rid = _display_id(case)
            lines.append(f"- **{rid}**: gold=`{case['gold_label']}`, "
                        f"predicted=`{case['predicted_label']}`")
            # Show unclear_reason if available (structured), else fall back to preview
            ur = case.get("unclear_reason") or ""
            if ur:
                lines.append(f"  - unclear reason: _{ur}_")
            elif case["explanation_preview"]:
                lines.append(f"  - _{case['explanation_preview']}_")
            if case["uncertain_preview"]:
                lines.append(f"  - uncertain: _{case['uncertain_preview']}_")
            # Show absent evidence if available
            aev = case.get("absent_evidence") or []
            if aev:
                lines.append(f"  - absent evidence: _{'; '.join(aev[:3])}_")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Missing-information checklist report (Task 71)."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--criteria", default=DEFAULT_CRITERIA,
        help=f"Path to criterion-level CSV (default: {DEFAULT_CRITERIA})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    labels_data = load_json(args.labels, required=False)
    criterion_rows = load_csv_rows(args.criteria, required=False)

    predictions = extract_predictions(results_data)
    label_index = index_labels(labels_data)
    criterion_index = index_criterion_rows(criterion_rows)

    summary = analyze_missing_info(predictions, label_index, criterion_index)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Missing-info checklist written to: {args.output}\n"
        f"  Records read: {summary['total_records']}\n"
        f"  Checklist cases: {summary['total_cases']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
