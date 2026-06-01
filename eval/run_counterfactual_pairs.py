"""
eval/run_counterfactual_pairs.py

Task 47 — Counterfactual pair checks.

For each case: run original and counterfactual patient through
match_patient_to_trial, then check whether the prediction
flips (or stays the same) as expected.

Exit 0 if matcher returns valid output.
Exit 1 only if matcher crashes or returns malformed output.
"""

import json
import os
import sys
import traceback
from typing import Any

from app.eligibility.rule_matcher import match_patient_to_trial


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

def make_counterfactual_cases() -> list[dict]:
    """Return synthetic counterfactual pairs. No real benchmark IDs used."""

    base_trial_pd = {
        "trial_id": "CF_TRIAL_PD_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
            "Age 40 to 80",
        ],
        "exclusion_criteria": [],
    }

    trial_no_dbs = {
        "trial_id": "CF_TRIAL_NO_DBS_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
        ],
        "exclusion_criteria": [
            "Prior deep brain stimulation",
        ],
    }

    trial_no_maob = {
        "trial_id": "CF_TRIAL_NO_MAOB_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
        ],
        "exclusion_criteria": [
            "Current MAO-B inhibitor use",
        ],
    }

    trial_no_dementia = {
        "trial_id": "CF_TRIAL_NO_DEMENTIA_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
        ],
        "exclusion_criteria": [
            "Dementia or significant cognitive impairment",
        ],
    }

    trial_pd_only = {
        "trial_id": "CF_TRIAL_PD_ONLY_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
        ],
        "exclusion_criteria": [],
    }

    trial_no_pacemaker = {
        "trial_id": "CF_TRIAL_MRI_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
        ],
        "exclusion_criteria": [
            "Implanted pacemaker or cardiac device",
        ],
    }

    cases = [
        {
            "case_id": "CF_001",
            "description": "Age inside vs outside explicit age range (40-80)",
            "changed_variable": "age",
            "original_patient": {
                "patient_id": "CF_PAT_001A",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_001B",
                "age": 85,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": base_trial_pd,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_002",
            "description": "Age at lower boundary (40) vs just below (39)",
            "changed_variable": "age",
            "original_patient": {
                "patient_id": "CF_PAT_002A",
                "age": 40,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_002B",
                "age": 39,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": base_trial_pd,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_003",
            "description": "No DBS history vs prior DBS when trial excludes DBS",
            "changed_variable": "dbs_history",
            "original_patient": {
                "patient_id": "CF_PAT_003A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "dbs_history": False,
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_003B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "dbs_history": True,
                "prior_deep_brain_stimulation": True,
            },
            "trial": trial_no_dbs,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_004",
            "description": "No MAO-B inhibitor vs rasagiline when trial excludes MAO-B inhibitors",
            "changed_variable": "current_medications",
            "original_patient": {
                "patient_id": "CF_PAT_004A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_004B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa", "rasagiline"],
            },
            "trial": trial_no_maob,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_005",
            "description": "Normal cognition vs dementia when trial excludes cognitive impairment",
            "changed_variable": "cognitive_status",
            "original_patient": {
                "patient_id": "CF_PAT_005A",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "cognitive_status": "normal",
                "moca_score": 27,
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_005B",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "cognitive_status": "dementia",
                "moca_score": 18,
            },
            "trial": trial_no_dementia,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_006",
            "description": "PD patient vs healthy control for PD-only trial",
            "changed_variable": "diagnosis",
            "original_patient": {
                "patient_id": "CF_PAT_006A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_006B",
                "age": 60,
                "diagnosis": "healthy control",
            },
            "trial": trial_pd_only,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_007",
            "description": "No pacemaker vs pacemaker for MRI/device-exclusion trial",
            "changed_variable": "pacemaker",
            "original_patient": {
                "patient_id": "CF_PAT_007A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "pacemaker": False,
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_007B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "pacemaker": True,
                "implanted_cardiac_device": True,
            },
            "trial": trial_no_pacemaker,
            "expected_relation": "should_flip",
        },
        {
            "case_id": "CF_008",
            "description": "Two identical patients — prediction should stay the same",
            "changed_variable": "none",
            "original_patient": {
                "patient_id": "CF_PAT_008A",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "counterfactual_patient": {
                "patient_id": "CF_PAT_008B",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": base_trial_pd,
            "expected_relation": "should_stay_same",
        },
    ]

    return cases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LABELS = {"eligible", "not_eligible", "unclear"}


def validate_matcher_result(result: Any) -> bool:
    """Return True if result is a dict with a valid prediction field."""
    if not isinstance(result, dict):
        return False
    label = (
        result.get("prediction")
        or result.get("label")
        or result.get("eligibility")
        or result.get("decision")
    )
    return label in VALID_LABELS


def _extract_label(result: dict) -> str:
    return (
        result.get("prediction")
        or result.get("label")
        or result.get("eligibility")
        or result.get("decision")
        or "unknown"
    )


def _extract_explanation(result: dict) -> str:
    return (
        result.get("explanation")
        or result.get("matcher_explanation")
        or result.get("reason")
        or result.get("reasoning")
        or ""
    )


def run_case(case: dict) -> dict:
    """
    Run one counterfactual case. Return enriched dict with predictions and status.
    Raises on matcher crash or malformed output.
    """
    orig_result = match_patient_to_trial(
        case["original_patient"], case["trial"]
    )
    cf_result = match_patient_to_trial(
        case["counterfactual_patient"], case["trial"]
    )

    if not validate_matcher_result(orig_result):
        raise ValueError(
            f"[{case['case_id']}] Malformed original result: {orig_result!r}"
        )
    if not validate_matcher_result(cf_result):
        raise ValueError(
            f"[{case['case_id']}] Malformed counterfactual result: {cf_result!r}"
        )

    orig_label = _extract_label(orig_result)
    cf_label = _extract_label(cf_result)
    flipped = orig_label != cf_label

    expected_relation = case["expected_relation"]
    if expected_relation == "should_flip":
        status = "passed_expected_flip" if flipped else "failed_expected_flip"
    else:
        status = "passed_expected_same" if not flipped else "failed_expected_same"

    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "changed_variable": case["changed_variable"],
        "trial_id": case["trial"]["trial_id"],
        "expected_relation": expected_relation,
        "original_label": orig_label,
        "counterfactual_label": cf_label,
        "flipped": flipped,
        "status": status,
        "original_explanation": _extract_explanation(orig_result),
        "counterfactual_explanation": _extract_explanation(cf_result),
    }


def run_counterfactual_cases(cases: list[dict]) -> list[dict]:
    """Run all cases. Raises on any matcher crash or malformed output."""
    results = []
    for case in cases:
        result = run_case(case)
        results.append(result)
    return results


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["status"].startswith("passed"))
    failed = sum(1 for r in results if r["status"].startswith("failed"))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "status_counts": counts,
    }


def write_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

STATUS_SYMBOL = {
    "passed_expected_flip": "✓",
    "failed_expected_flip": "✗",
    "passed_expected_same": "✓",
    "failed_expected_same": "✗",
}


def print_summary(results: list[dict], summary: dict) -> None:
    print("\n=== Counterfactual Pair Results ===\n")
    for r in results:
        sym = STATUS_SYMBOL.get(r["status"], "?")
        flip_str = "flipped" if r["flipped"] else "no flip"
        print(
            f"  {sym} [{r['case_id']}] {r['description']}\n"
            f"      original={r['original_label']}  "
            f"counterfactual={r['counterfactual_label']}  "
            f"({flip_str})  → {r['status']}"
        )
    print(
        f"\n  Total: {summary['total']}  "
        f"Passed: {summary['passed']}  "
        f"Failed: {summary['failed']}"
    )
    for status, count in sorted(summary["status_counts"].items()):
        print(f"    {status}: {count}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_REPORT_PATH = "reports/counterfactual_pairs_report.json"


def main() -> int:
    report_path = DEFAULT_REPORT_PATH
    if len(sys.argv) > 1:
        report_path = sys.argv[1]

    cases = make_counterfactual_cases()

    try:
        results = run_counterfactual_cases(cases)
    except Exception:
        print("\n[ERROR] Matcher crashed or returned malformed output:\n", file=sys.stderr)
        traceback.print_exc()
        return 1

    summary = summarize_results(results)
    print_summary(results, summary)

    report = {"summary": summary, "results": results}
    write_json(report, report_path)
    print(f"  Report written to: {report_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
