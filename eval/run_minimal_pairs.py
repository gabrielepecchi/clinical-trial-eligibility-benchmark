"""
eval/run_minimal_pairs.py

Task 51 — Minimal pair checks.

Each pair of patients differs by one minimal clinical phrase or field.
Checks whether the matcher reacts consistently to small clinical changes.

Exit 0 if matcher returns valid output for all pairs.
Exit 1 only if matcher crashes or returns malformed output.
"""

import json
import os
import sys
import traceback
from typing import Any

from app.eligibility.rule_matcher import match_patient_to_trial


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

def make_minimal_pairs() -> list[dict]:
    """Return synthetic minimal pairs. No real benchmark IDs used."""

    trial_no_maob = {
        "trial_id": "MP_TRIAL_NO_MAOB_001",
        "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
        "exclusion_criteria": ["Current MAO-B inhibitor use"],
    }

    trial_no_dbs = {
        "trial_id": "MP_TRIAL_NO_DBS_001",
        "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
        "exclusion_criteria": ["Prior deep brain stimulation"],
    }

    trial_moca = {
        "trial_id": "MP_TRIAL_MOCA_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
            "MoCA score >= 24",
        ],
        "exclusion_criteria": [],
    }

    trial_age = {
        "trial_id": "MP_TRIAL_AGE_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
            "Age >= 40",
        ],
        "exclusion_criteria": [],
    }

    trial_pd_only = {
        "trial_id": "MP_TRIAL_PD_001",
        "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
        "exclusion_criteria": [],
    }

    trial_no_pacemaker = {
        "trial_id": "MP_TRIAL_MRI_001",
        "inclusion_criteria": ["Diagnosis of idiopathic Parkinson disease"],
        "exclusion_criteria": ["Implanted cardiac pacemaker or device"],
    }

    trial_stable_med = {
        "trial_id": "MP_TRIAL_STABLE_MED_001",
        "inclusion_criteria": [
            "Diagnosis of idiopathic Parkinson disease",
            "Stable levodopa dose for at least 4 weeks",
        ],
        "exclusion_criteria": [],
    }

    pairs = [
        {
            "pair_id": "MP_001",
            "description": "No MAO-B inhibitor vs current rasagiline",
            "changed_phrase_or_field": "current_medications",
            "case_a_patient": {
                "patient_id": "MP_PAT_001A",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_001B",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa", "rasagiline"],
            },
            "trial": trial_no_maob,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_002",
            "description": "No prior DBS vs prior DBS implantation",
            "changed_phrase_or_field": "dbs_history",
            "case_a_patient": {
                "patient_id": "MP_PAT_002A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "dbs_history": False,
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_002B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "dbs_history": True,
                "prior_deep_brain_stimulation": True,
            },
            "trial": trial_no_dbs,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_003",
            "description": "MoCA 27 (normal) vs MoCA 18 (impaired)",
            "changed_phrase_or_field": "moca_score",
            "case_a_patient": {
                "patient_id": "MP_PAT_003A",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "moca_score": 27,
                "cognitive_status": "normal",
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_003B",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "moca_score": 18,
                "cognitive_status": "impaired",
            },
            "trial": trial_moca,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_004",
            "description": "Age 40 (at boundary) vs age 39 (below boundary)",
            "changed_phrase_or_field": "age",
            "case_a_patient": {
                "patient_id": "MP_PAT_004A",
                "age": 40,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_004B",
                "age": 39,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "trial": trial_age,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_005",
            "description": "Idiopathic Parkinson disease vs healthy control",
            "changed_phrase_or_field": "diagnosis",
            "case_a_patient": {
                "patient_id": "MP_PAT_005A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_005B",
                "age": 60,
                "diagnosis": "healthy control",
            },
            "trial": trial_pd_only,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_006",
            "description": "No pacemaker vs implanted cardiac pacemaker",
            "changed_phrase_or_field": "pacemaker",
            "case_a_patient": {
                "patient_id": "MP_PAT_006A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "pacemaker": False,
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_006B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "pacemaker": True,
                "implanted_cardiac_device": True,
            },
            "trial": trial_no_pacemaker,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_007",
            "description": "Stable levodopa dose vs medication history unclear",
            "changed_phrase_or_field": "medication_stability",
            "case_a_patient": {
                "patient_id": "MP_PAT_007A",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
                "medication_stable": True,
                "medication_stable_weeks": 6,
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_007B",
                "age": 60,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
                "medication_stable": None,
                "medication_history_unclear": True,
            },
            "trial": trial_stable_med,
            "expected_relation": "should_differ",
        },
        {
            "pair_id": "MP_008",
            "description": "Same patient twice — prediction should stay the same",
            "changed_phrase_or_field": "none",
            "case_a_patient": {
                "patient_id": "MP_PAT_008A",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
            },
            "case_b_patient": {
                "patient_id": "MP_PAT_008B",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
                "current_medications": ["levodopa"],
            },
            "trial": trial_no_maob,
            "expected_relation": "should_stay_same",
        },
    ]

    return pairs


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


def run_pair(pair: dict) -> dict:
    """
    Run one minimal pair. Return enriched dict with predictions and status.
    Raises on matcher crash or malformed output.
    """
    result_a = match_patient_to_trial(pair["case_a_patient"], pair["trial"])
    result_b = match_patient_to_trial(pair["case_b_patient"], pair["trial"])

    if not validate_matcher_result(result_a):
        raise ValueError(
            f"[{pair['pair_id']}] Malformed case_a result: {result_a!r}"
        )
    if not validate_matcher_result(result_b):
        raise ValueError(
            f"[{pair['pair_id']}] Malformed case_b result: {result_b!r}"
        )

    label_a = _extract_label(result_a)
    label_b = _extract_label(result_b)
    differs = label_a != label_b

    expected_relation = pair["expected_relation"]
    if expected_relation == "should_differ":
        status = "passed_expected_difference" if differs else "failed_expected_difference"
    else:
        status = "passed_expected_same" if not differs else "failed_expected_same"

    return {
        "pair_id": pair["pair_id"],
        "description": pair["description"],
        "changed_phrase_or_field": pair["changed_phrase_or_field"],
        "trial_id": pair["trial"]["trial_id"],
        "expected_relation": expected_relation,
        "case_a_prediction": label_a,
        "case_b_prediction": label_b,
        "differs": differs,
        "status": status,
        "case_a_explanation": _extract_explanation(result_a),
        "case_b_explanation": _extract_explanation(result_b),
    }


def run_minimal_pairs(pairs: list[dict]) -> list[dict]:
    """Run all pairs. Raises on any matcher crash or malformed output."""
    results = []
    for pair in pairs:
        result = run_pair(pair)
        results.append(result)
    return results


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["status"].startswith("passed"))
    failed = sum(1 for r in results if r["status"].startswith("failed"))
    counts: dict[str, int] = {}
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
    "passed_expected_difference": "✓",
    "failed_expected_difference": "✗",
    "passed_expected_same": "✓",
    "failed_expected_same": "✗",
}


def print_summary(results: list[dict], summary: dict) -> None:
    print("\n=== Minimal Pair Results ===\n")
    for r in results:
        sym = STATUS_SYMBOL.get(r["status"], "?")
        diff_str = "differs" if r["differs"] else "same"
        print(
            f"  {sym} [{r['pair_id']}] {r['description']}\n"
            f"      case_a={r['case_a_prediction']}  "
            f"case_b={r['case_b_prediction']}  "
            f"({diff_str})  → {r['status']}"
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

DEFAULT_REPORT_PATH = "reports/minimal_pairs_report.json"


def main() -> int:
    report_path = DEFAULT_REPORT_PATH
    if len(sys.argv) > 1:
        report_path = sys.argv[1]

    pairs = make_minimal_pairs()

    try:
        results = run_minimal_pairs(pairs)
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
