"""
eval/run_sensitivity_report.py

Task 62 — Patient-field sensitivity report.

For each synthetic base case, removes or nulls one patient field at a time
and checks whether the matcher prediction changes.

Usage:
    PYTHONPATH=. python eval/run_sensitivity_report.py
    PYTHONPATH=. python eval/run_sensitivity_report.py --output reports/my_report.md
"""

import argparse
import json
import os
import sys
import traceback
from typing import Any

from app.eligibility.rule_matcher import match_patient_to_trial

DEFAULT_OUTPUT_PATH = "reports/sensitivity_report.md"
VALID_LABELS = {"eligible", "not_eligible", "unclear"}


# ---------------------------------------------------------------------------
# Base cases
# ---------------------------------------------------------------------------

def make_base_cases() -> list[dict]:
    """Return synthetic base patient/trial cases. No real benchmark IDs used."""
    return [
        {
            "case_id": "SEN_001",
            "capability_area": "age_threshold",
            "patient": {
                "patient_id": "SEN_PAT_001",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": {
                "trial_id": "SEN_TRIAL_001",
                "inclusion_criteria": [
                    "Diagnosis of idiopathic Parkinson disease",
                    "Age between 40 and 80",
                ],
                "exclusion_criteria": [],
            },
            "fields_to_remove": ["age"],
        },
        {
            "case_id": "SEN_002",
            "capability_area": "diagnosis_requirement",
            "patient": {
                "patient_id": "SEN_PAT_002",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": {
                "trial_id": "SEN_TRIAL_002",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": [],
            },
            "fields_to_remove": ["diagnosis"],
        },
        {
            "case_id": "SEN_003",
            "capability_area": "medication_exclusion",
            "patient": {
                "patient_id": "SEN_PAT_003",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa", "rasagiline"],
            },
            "trial": {
                "trial_id": "SEN_TRIAL_003",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": ["Current MAO-B inhibitor use"],
            },
            "fields_to_remove": ["medications"],
        },
        {
            "case_id": "SEN_004",
            "capability_area": "dbs_exclusion",
            "patient": {
                "patient_id": "SEN_PAT_004",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "dbs_history": True,
                "procedure_history": ["deep brain stimulation"],
            },
            "trial": {
                "trial_id": "SEN_TRIAL_004",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": ["Prior deep brain stimulation"],
            },
            "fields_to_remove": ["dbs_history", "procedure_history"],
        },
        {
            "case_id": "SEN_005",
            "capability_area": "cognitive_requirement",
            "patient": {
                "patient_id": "SEN_PAT_005",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "cognitive_status": "normal",
                "moca_score": 27,
            },
            "trial": {
                "trial_id": "SEN_TRIAL_005",
                "inclusion_criteria": [
                    "Diagnosis of idiopathic Parkinson disease",
                    "MoCA score >= 24",
                ],
                "exclusion_criteria": [],
            },
            "fields_to_remove": ["cognitive_status", "moca_score"],
        },
        {
            "case_id": "SEN_006",
            "capability_area": "device_exclusion",
            "patient": {
                "patient_id": "SEN_PAT_006",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "implanted_device": "cardiac pacemaker",
            },
            "trial": {
                "trial_id": "SEN_TRIAL_006",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": ["Implanted cardiac pacemaker or device"],
            },
            "fields_to_remove": ["implanted_device"],
        },
        {
            "case_id": "SEN_007",
            "capability_area": "comorbidity_exclusion",
            "patient": {
                "patient_id": "SEN_PAT_007",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "comorbidities": ["active malignancy"],
            },
            "trial": {
                "trial_id": "SEN_TRIAL_007",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": ["Active malignancy or cancer"],
            },
            "fields_to_remove": ["comorbidities"],
        },
        {
            "case_id": "SEN_008",
            "capability_area": "recent_trial_exclusion",
            "patient": {
                "patient_id": "SEN_PAT_008",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "recent_trial_participation": True,
            },
            "trial": {
                "trial_id": "SEN_TRIAL_008",
                "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
                "exclusion_criteria": [
                    "Participation in another clinical trial within 30 days"
                ],
            },
            "fields_to_remove": ["recent_trial_participation"],
        },
    ]


# ---------------------------------------------------------------------------
# Field variants
# ---------------------------------------------------------------------------

def make_field_variants(case: dict) -> list[dict]:
    """
    Return one variant per field in fields_to_remove.
    Each variant is a copy of the base patient with that field removed.
    """
    variants = []
    for field in case["fields_to_remove"]:
        variant_patient = {k: v for k, v in case["patient"].items() if k != field}
        variants.append(
            {
                "case_id": case["case_id"],
                "capability_area": case["capability_area"],
                "removed_or_unknown_field": field,
                "base_patient": case["patient"],
                "variant_patient": variant_patient,
                "trial": case["trial"],
            }
        )
    return variants


# ---------------------------------------------------------------------------
# Matcher helpers
# ---------------------------------------------------------------------------

def validate_matcher_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    label = (
        result.get("prediction")
        or result.get("label")
        or result.get("eligibility")
        or result.get("decision")
    )
    return label in VALID_LABELS


def extract_prediction(result: dict) -> str:
    return (
        result.get("prediction")
        or result.get("label")
        or result.get("eligibility")
        or result.get("decision")
        or "unknown"
    )


def extract_explanation(result: dict) -> str:
    return (
        result.get("explanation")
        or result.get("matcher_explanation")
        or result.get("reason")
        or result.get("reasoning")
        or ""
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sensitivity_case(variant: dict) -> dict:
    """
    Run base and variant through the matcher.
    Raises if matcher crashes or returns malformed output.
    """
    base_result = match_patient_to_trial(variant["base_patient"], variant["trial"])
    variant_result = match_patient_to_trial(variant["variant_patient"], variant["trial"])

    if not validate_matcher_result(base_result):
        raise ValueError(
            f"[{variant['case_id']}] Malformed base result: {base_result!r}"
        )
    if not validate_matcher_result(variant_result):
        raise ValueError(
            f"[{variant['case_id']}] Malformed variant result: {variant_result!r}"
        )

    base_pred = extract_prediction(base_result)
    variant_pred = extract_prediction(variant_result)

    return {
        "case_id": variant["case_id"],
        "capability_area": variant["capability_area"],
        "removed_or_unknown_field": variant["removed_or_unknown_field"],
        "trial_id": variant["trial"]["trial_id"],
        "base_prediction": base_pred,
        "variant_prediction": variant_pred,
        "changed_prediction": base_pred != variant_pred,
        "base_explanation": extract_explanation(base_result),
        "variant_explanation": extract_explanation(variant_result),
    }


def run_sensitivity_cases(cases: list[dict]) -> list[dict]:
    """Expand all cases into variants and run each. Raises on matcher error."""
    results = []
    for case in cases:
        for variant in make_field_variants(case):
            result = run_sensitivity_case(variant)
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------

def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    changed = sum(1 for r in results if r["changed_prediction"])
    by_area: dict[str, dict] = {}
    for r in results:
        area = r["capability_area"]
        if area not in by_area:
            by_area[area] = {"total": 0, "changed": 0}
        by_area[area]["total"] += 1
        if r["changed_prediction"]:
            by_area[area]["changed"] += 1
    return {
        "total_variants": total,
        "changed_prediction": changed,
        "unchanged_prediction": total - changed,
        "by_capability_area": by_area,
        "results": results,
    }


def _preview(text: str, max_chars: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "\u2026" if len(text) > max_chars else text


def format_markdown_report(summary: dict) -> str:
    lines = [
        "# Patient-Field Sensitivity Report",
        "",
        "## Summary",
        "",
        f"- **Total variants tested:** {summary['total_variants']}",
        f"- **Prediction changed:** {summary['changed_prediction']}",
        f"- **Prediction unchanged:** {summary['unchanged_prediction']}",
        "",
        "### By capability area",
        "",
        "| Capability area | Variants | Changed |",
        "|---|---|---|",
    ]
    for area, counts in sorted(summary["by_capability_area"].items()):
        lines.append(f"| {area} | {counts['total']} | {counts['changed']} |")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")

    for r in summary["results"]:
        changed_str = "\u2713 changed" if r["changed_prediction"] else "\u2013 unchanged"
        lines += [
            f"### [{r['case_id']}] `{r['capability_area']}` — field: `{r['removed_or_unknown_field']}`",
            "",
            f"- **trial:** `{r['trial_id']}`",
            f"- **base prediction:** {r['base_prediction']}",
            f"- **variant prediction:** {r['variant_prediction']}  ({changed_str})",
        ]
        if r["base_explanation"]:
            lines.append(f"- **base explanation:** {_preview(r['base_explanation'])}")
        if r["variant_explanation"]:
            lines.append(f"- **variant explanation:** {_preview(r['variant_explanation'])}")
        lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Patient-field sensitivity report.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    cases = make_base_cases()

    try:
        results = run_sensitivity_cases(cases)
    except Exception:
        print("\n[ERROR] Matcher crashed or returned malformed output:\n", file=sys.stderr)
        traceback.print_exc()
        return 1

    summary = summarize_results(results)
    report_text = format_markdown_report(summary)
    write_text(report_text, args.output)

    changed = summary["changed_prediction"]
    total = summary["total_variants"]
    print(
        f"\nSensitivity report: variants={total}  changed={changed}  "
        f"unchanged={total - changed}  output={args.output}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
