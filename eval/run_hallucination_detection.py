"""
Task 98: Hallucination detection / explanation audit.

Analyzes matcher explanations to identify unsupported claims —
explanation mentions of patient facts not clearly present in the
patient profile.

This is a heuristic audit, not a proof of hallucination. Results
may include false positives and false negatives.

Usage:
    PYTHONPATH=. python eval/run_hallucination_detection.py
    PYTHONPATH=. python eval/run_hallucination_detection.py \
        --results data/processed/results_llm_reviewed.json \
        --patients data/processed/patient_cases.json \
        --labels  data/processed/labels_llm_reviewed.json \
        --output  reports/hallucination_detection_report.md
"""

from __future__ import annotations

import argparse
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
DEFAULT_PATIENTS = "data/processed/patient_cases.json"
DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/hallucination_detection_report.md"

# Category keywords for candidate extraction
CLAIM_CATEGORIES: dict[str, list[str]] = {
    "medication": [
        "levodopa", "carbidopa", "dopamine", "agonist", "mao-b",
        "inhibitor", "rasagiline", "selegiline", "comt", "medication",
        "drug",
    ],
    "procedure": [
        "dbs", "deep brain", "surgery", "procedure", "implant", "stimulation",
        "ablation", "transplant",
    ],
    "device": [
        "pacemaker", "defibrillator", "device", "cardiac", "ivcd", "icd",
    ],
    "cognitive": [
        "moca", "mmse", "montreal", "cognitive", "dementia", "memory",
    ],
    "updrs": ["updrs", "unified parkinson"],
    "hoehn_yahr": ["hoehn", "yahr", "h&y", "stage"],
    "comorbidity": [
        "cardiac", "heart", "renal", "kidney", "liver", "hepatic",
        "diabetes", "hypertension", "depression", "arrhythmia",
    ],
    "pregnancy": ["pregnant", "pregnancy", "reproductive", "postmenopausal"],
    "trial": ["trial", "study", "research", "participation"],
    "lab": ["lab", "laboratory", "value", "creatinine", "ast", "alt"],
    "demographics": ["age", "year", "old", "male", "female", "sex", "gender"],
    "diagnosis": ["parkinson", "diagnosis", "idiopathic", "symptomatic"],
}

# Synonym map for conservative matching
SYNONYMS: dict[str, set[str]] = {
    "dbs": {"dbs", "deep brain stimulation", "brain stimulation"},
    "mao-b": {"mao-b", "mao-b inhibitor", "rasagiline", "selegiline"},
    "dopamine agonist": {"dopamine agonist", "agonist", "dopamine"},
    "levodopa": {"levodopa", "l-dopa", "carbidopa"},
    "moca": {"moca", "montreal cognitive assessment"},
    "mmse": {"mmse", "mini-mental state"},
    "updrs": {"updrs", "unified parkinson disease rating scale"},
    "hoehn-yahr": {"hoehn", "yahr", "h&y", "hoehn-yahr"},
}


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


def extract_label_records(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of label records from *data*."""
    if data is None:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("labels", "records", "pairs", "cases"):
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
        candidates = [v for v in data.values() if isinstance(v, dict)]
        if candidates:
            return candidates
    return []


def extract_patient_records(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of patient records from *data*."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("patients", "cases", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        candidates = [v for v in data.values() if isinstance(v, dict)]
        if candidates:
            return candidates
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


def index_patients(
    patient_data: Any,
) -> dict[str, dict[str, Any]]:
    """Return {patient_id: patient_record} from *patient_data*."""
    records = extract_patient_records(patient_data)
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        pid = str(rec.get("patient_id", rec.get("id", ""))).strip()
        if pid:
            index[pid] = rec
    return index


def normalize_text(value: Any) -> str:
    """Return a normalized lowercase text representation."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(item).strip() for item in value if item)
    elif isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            if v is not None:
                parts.append(str(v).strip())
        text = " ".join(parts)
    else:
        text = str(value)
    return text.lower()


# ---------------------------------------------------------------------------
# Patient profile extraction
# ---------------------------------------------------------------------------


def get_patient_profile_text(
    record: dict[str, Any],
    patient_index: dict[str, dict[str, Any]],
) -> str:
    """Return normalized text representation of patient profile."""
    pid = str(record.get("patient_id", "")).strip()
    patient_rec = patient_index.get(pid, {})

    if not patient_rec:
        # Fallback: use patient fields from result record itself
        patient_rec = {}
        for field in (
            "age", "sex", "diagnosis", "medications", "dbs_history",
            "cognitive_status", "updrs_score", "hoehn_yahr_stage",
            "moca_score", "comorbidities",
        ):
            if field in record:
                patient_rec[field] = record[field]

    return normalize_text(patient_rec)


def collect_explanation_text(
    record: dict[str, Any],
    label_record: dict[str, Any] | None = None,
) -> str:
    """Collect all explanation-related text from records."""
    texts: list[str] = []

    for field in (
        "explanation",
        "matcher_explanation",
        "reason",
        "reasoning",
        "rationale",
        "reasoning_trace",
    ):
        val = record.get(field, "")
        if val:
            texts.append(str(val))

    for field in ("blocking_criteria", "uncertain_criteria"):
        val = record.get(field, "")
        if val:
            texts.append(str(val))

    if label_record:
        for field in ("rationale", "explanation"):
            val = label_record.get(field, "")
            if val:
                texts.append(str(val))

    return normalize_text(texts)


# ---------------------------------------------------------------------------
# Claim extraction and support checking
# ---------------------------------------------------------------------------


def extract_claim_candidates(text: str) -> dict[str, list[str]]:
    """
    Extract candidate clinical claims from *text*, grouped by category.

    Returns {category: [candidate, ...]} for claims that might not be
    supported by the patient profile.
    """
    if not text:
        return {}

    candidates: dict[str, list[str]] = defaultdict(list)

    for category, keywords in CLAIM_CATEGORIES.items():
        for keyword in keywords:
            # Look for the keyword as a whole word (case-insensitive)
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text):
                candidates[category].append(keyword)

    # Deduplicate
    return {cat: list(set(cands)) for cat, cands in candidates.items()}


def is_supported_by_profile(candidate: str, profile_text: str) -> bool:
    """
    Return True if *candidate* is clearly supported by *profile_text*.

    Uses exact keyword match and synonym map for conservative checking.
    """
    candidate_lower = candidate.lower().strip()

    # Check exact match
    if candidate_lower in profile_text:
        return True

    # Check synonym map
    for base_term, synonyms in SYNONYMS.items():
        if candidate_lower in synonyms:
            # Check if any synonym appears in profile
            if any(syn in profile_text for syn in synonyms):
                return True

    return False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_unsupported_claims(
    predictions: list[dict[str, Any]],
    patient_index: dict[str, dict[str, Any]],
    label_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Analyze predictions for unsupported explanation claims."""
    flagged: list[dict[str, Any]] = []
    total_with_explanation = 0
    total_with_patient_profile = 0

    for pred in predictions:
        pid = str(pred.get("patient_id", "")).strip()
        tid = str(pred.get("trial_id", "")).strip()
        key = pair_key(pred)

        # Collect explanation text
        label_rec = label_index.get(key, {})
        explanation_text = collect_explanation_text(pred, label_rec)
        if explanation_text:
            total_with_explanation += 1

        # Get patient profile
        profile_text = get_patient_profile_text(pred, patient_index)
        if profile_text:
            total_with_patient_profile += 1

        # Skip if no explanation or no profile
        if not explanation_text or not profile_text:
            continue

        # Extract candidates
        candidates = extract_claim_candidates(explanation_text)
        if not candidates:
            continue

        # Check for unsupported claims
        unsupported: dict[str, list[str]] = defaultdict(list)
        for category, claims in candidates.items():
            for claim in claims:
                if not is_supported_by_profile(claim, profile_text):
                    unsupported[category].append(claim)

        # Flag if unsupported claims found
        if unsupported:
            flagged.append(
                {
                    "patient_id": pid,
                    "trial_id": tid,
                    "gold_label": str(pred.get("gold_label", "")).strip().lower(),
                    "predicted_label": str(
                        pred.get("predicted_label", pred.get("prediction", ""))
                    ).strip().lower(),
                    "confidence": pred.get("confidence", pred.get("confidence_score")),
                    "unsupported_claims": dict(unsupported),
                    "explanation_preview": explanation_text[:200].rstrip() + "…",
                    "profile_preview": profile_text[:200].rstrip() + "…",
                }
            )

    # Count by category
    category_counts: dict[str, int] = defaultdict(int)
    for case in flagged:
        for category in case["unsupported_claims"].keys():
            category_counts[category] += 1

    # Group by category
    cases_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in flagged:
        for category in case["unsupported_claims"].keys():
            cases_by_category[category].append(case)

    return {
        "total_records": len(predictions),
        "total_with_explanation": total_with_explanation,
        "total_with_patient_profile": total_with_patient_profile,
        "total_flagged": len(flagged),
        "flagged": flagged,
        "category_counts": dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        "cases_by_category": dict(cases_by_category),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _display_id(case: dict[str, Any]) -> str:
    pid = case.get("patient_id", "")
    tid = case.get("trial_id", "")
    if pid and tid:
        return f"{pid} / {tid}"
    return pid or tid or "(no id)"


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Return a Markdown string for the summary."""
    lines: list[str] = []

    total_records: int = summary["total_records"]
    total_with_explanation: int = summary["total_with_explanation"]
    total_with_patient_profile: int = summary["total_with_patient_profile"]
    total_flagged: int = summary["total_flagged"]
    flagged: list[dict[str, Any]] = summary["flagged"]
    category_counts: dict[str, int] = summary["category_counts"]
    cases_by_category: dict[str, list[dict[str, Any]]] = summary["cases_by_category"]

    lines.append("# Explanation Unsupported-Claim Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> This report identifies records where matcher explanations mention patient  "
    )
    lines.append(
        "> facts that do not appear clearly in the patient profile. This is a heuristic  "
    )
    lines.append(
        "> audit and may include false positives (generic terms, synonyms not in map)  "
    )
    lines.append("> and false negatives (claims expressed indirectly or implied). ")
    lines.append("> This is **not** a proof of hallucination.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total records | {total_records} |")
    lines.append(f"| Records with explanation text | {total_with_explanation} |")
    lines.append(f"| Records with patient profile source | {total_with_patient_profile} |")
    lines.append(f"| Flagged for unsupported claim candidates | {total_flagged} |")
    lines.append("")

    if total_flagged == 0:
        lines.append("**No unsupported claim candidates detected.**")
        lines.append("")
        return "\n".join(lines)

    # Category counts
    lines.append("## Unsupported Claim Candidates by Category")
    lines.append("")
    lines.append("| Category | Case Count |")
    lines.append("|----------|-----------|")
    for cat, count in category_counts.items():
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # Top 25 flagged
    lines.append("## Top 25 Flagged Records")
    lines.append("")
    for i, case in enumerate(flagged[:25], 1):
        rid = _display_id(case)
        categories = ", ".join(f"`{c}`" for c in case["unsupported_claims"].keys())
        lines.append(f"{i}. **{rid}** — categories: {categories}")
        lines.append(
            f"   - gold: `{case['gold_label']}`, predicted: `{case['predicted_label']}`"
        )
    lines.append("")

    # Examples by category
    lines.append("## Examples by Unsupported Claim Category")
    lines.append("")
    for category in sorted(category_counts.keys()):
        cat_cases = cases_by_category[category][:5]
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"Found in **{category_counts[category]}** cases.")
        lines.append("")
        for case in cat_cases:
            rid = _display_id(case)
            claims = ", ".join(case["unsupported_claims"].get(category, []))
            lines.append(f"- **{rid}**: {claims}")
            lines.append(f"  - explanation snippet: _{case['explanation_preview']}_")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **False positives**: Explanations may mention terms that are implied by the "
    )
    lines.append(
        "  patient profile but not explicitly stated (e.g., 'patient taking medication X' "
    )
    lines.append(
        "  when the profile lists X but doesn't use the word 'medication')."
    )
    lines.append("")
    lines.append(
        "- **False negatives**: Explanations may describe patient facts indirectly, using "
    )
    lines.append(
        "  phrasing not captured by keyword matching (e.g., 'requiring device management' "
    )
    lines.append("  instead of 'pacemaker').")
    lines.append("")
    lines.append(
        "- **Synonym coverage**: The synonym map is small and hand-written. Many clinical "
    )
    lines.append("  terms may not be properly matched.")
    lines.append("")
    lines.append(
        "- **Scope**: This audit only flags explicit keyword mismatches, not implicit "
    )
    lines.append(
        "  claims or complex reasoning errors."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explanation unsupported-claim audit (Task 98)."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--patients", default=DEFAULT_PATIENTS,
        help=f"Path to patient records JSON (default: {DEFAULT_PATIENTS})",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    patient_data = load_json(args.patients, required=False)
    labels_data = load_json(args.labels, required=False)

    predictions = extract_predictions(results_data)
    patient_index = index_patients(patient_data)

    label_index: dict[str, dict[str, Any]] = {}
    if labels_data:
        for rec in extract_label_records(labels_data):
            key = pair_key(rec)
            label_index[key] = rec

    summary = analyze_unsupported_claims(predictions, patient_index, label_index)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Hallucination audit report written to: {args.output}\n"
        f"  Records analyzed: {summary['total_records']}\n"
        f"  Records with explanations: {summary['total_with_explanation']}\n"
        f"  Flagged for unsupported claims: {summary['total_flagged']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
