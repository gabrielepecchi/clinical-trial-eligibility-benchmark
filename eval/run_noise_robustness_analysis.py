"""
eval/run_noise_robustness_analysis.py

Task 54 — Noise robustness analysis.

Creates synthetic patient/trial base cases, generates input variants
(narrative, noisy casing, extra text, reordered fields, synonyms),
runs each variant through the matcher, and reports prediction stability.

Usage:
    PYTHONPATH=. python eval/run_noise_robustness_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

OUTPUT_PATH = "reports/noise_robustness_report.json"
VALID_PREDICTIONS = {"eligible", "not_eligible", "unclear"}


# ---------------------------------------------------------------------------
# Synthetic base cases
# ---------------------------------------------------------------------------

def make_base_cases() -> list[dict]:
    """Return a list of synthetic base case dicts with patient and trial."""
    return [
        {
            "case_id": "age_threshold_01",
            "capability_area": "age_threshold",
            "patient": {
                "patient_id": "synthetic_age_01",
                "age": 65,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa"],
                "dbs_history": False,
                "cognitive_status": "normal",
            },
            "trial": {
                "trial_id": "synthetic_trial_age",
                "inclusion_criteria": "Age between 40 and 80 years.",
                "exclusion_criteria": "Prior deep brain stimulation.",
            },
        },
        {
            "case_id": "pd_diagnosis_01",
            "capability_area": "diagnosis",
            "patient": {
                "patient_id": "synthetic_dx_01",
                "age": 58,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa/carbidopa"],
                "dbs_history": False,
                "cognitive_status": "normal",
            },
            "trial": {
                "trial_id": "synthetic_trial_dx",
                "inclusion_criteria": "Diagnosis of idiopathic Parkinson disease.",
                "exclusion_criteria": "Atypical Parkinsonism or secondary Parkinsonism.",
            },
        },
        {
            "case_id": "medication_exclusion_01",
            "capability_area": "medication",
            "patient": {
                "patient_id": "synthetic_med_01",
                "age": 62,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["rasagiline"],
                "dbs_history": False,
                "cognitive_status": "normal",
            },
            "trial": {
                "trial_id": "synthetic_trial_med",
                "inclusion_criteria": "Diagnosis of idiopathic Parkinson disease.",
                "exclusion_criteria": "Current use of MAO-B inhibitors.",
            },
        },
        {
            "case_id": "dbs_exclusion_01",
            "capability_area": "procedure",
            "patient": {
                "patient_id": "synthetic_dbs_01",
                "age": 70,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa"],
                "dbs_history": True,
                "cognitive_status": "normal",
            },
            "trial": {
                "trial_id": "synthetic_trial_dbs",
                "inclusion_criteria": "Diagnosis of idiopathic Parkinson disease.",
                "exclusion_criteria": "Prior deep brain stimulation surgery.",
            },
        },
        {
            "case_id": "cognitive_impairment_01",
            "capability_area": "cognitive",
            "patient": {
                "patient_id": "synthetic_cog_01",
                "age": 72,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa"],
                "dbs_history": False,
                "cognitive_status": "dementia",
                "moca_score": 18,
            },
            "trial": {
                "trial_id": "synthetic_trial_cog",
                "inclusion_criteria": "MoCA score >= 24. No dementia.",
                "exclusion_criteria": "Diagnosis of dementia or significant cognitive impairment.",
            },
        },
        {
            "case_id": "device_pacemaker_01",
            "capability_area": "device",
            "patient": {
                "patient_id": "synthetic_dev_01",
                "age": 66,
                "diagnosis": "idiopathic Parkinson disease",
                "medications": ["levodopa"],
                "dbs_history": False,
                "cognitive_status": "normal",
                "implanted_device": "pacemaker",
            },
            "trial": {
                "trial_id": "synthetic_trial_dev",
                "inclusion_criteria": "Diagnosis of idiopathic Parkinson disease.",
                "exclusion_criteria": "Presence of any implanted electronic device, including pacemakers.",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------

def _patient_to_narrative(patient: dict) -> str:
    """Convert a structured patient dict into a narrative-style string."""
    age = patient.get("age", "unknown")
    dx = patient.get("diagnosis", "unknown condition")
    meds = patient.get("medications", [])
    meds_str = ", ".join(meds) if meds else "no current medications"
    dbs = "has a history of deep brain stimulation" if patient.get("dbs_history") else "no history of deep brain stimulation"
    cog = patient.get("cognitive_status", "unknown")
    moca = patient.get("moca_score")
    moca_str = f" MoCA score {moca}." if moca is not None else ""
    device = patient.get("implanted_device")
    device_str = f" Patient has an implanted {device}." if device else ""
    return (
        f"A {age}-year-old patient with {dx}. "
        f"Current medications include {meds_str}. "
        f"The patient {dbs}. "
        f"Cognitive status: {cog}.{moca_str}{device_str}"
    )


def _noisy_casing(text: str) -> str:
    """Apply mixed casing noise."""
    result = []
    for i, ch in enumerate(text):
        result.append(ch.upper() if i % 3 == 0 else ch.lower())
    return "".join(result)


def _add_extra_sentence(text: str) -> str:
    return text + " The patient is generally cooperative and lives independently."


def _reorder_fields(patient: dict) -> dict:
    """Return a copy with keys in reversed order."""
    return dict(reversed(list(patient.items())))


_SYNONYMS = {
    "idiopathic Parkinson disease": "idiopathic Parkinson's disease",
    "deep brain stimulation": "DBS",
    "MAO-B inhibitors": "monoamine oxidase type B inhibitors",
    "pacemaker": "cardiac pacemaker device",
    "dementia": "major neurocognitive disorder",
    "levodopa": "L-DOPA",
    "rasagiline": "Azilect",
}


def _apply_synonyms(patient: dict) -> dict:
    """Return a shallow copy of patient with synonym substitutions in string values."""
    result = {}
    for k, v in patient.items():
        if isinstance(v, str):
            for src, dst in _SYNONYMS.items():
                v = v.replace(src, dst)
            result[k] = v
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, str):
                    for src, dst in _SYNONYMS.items():
                        item = item.replace(src, dst)
                new_list.append(item)
            result[k] = new_list
        else:
            result[k] = v
    return result


def make_variants(case: dict) -> list[dict]:
    """
    Return a list of variant dicts: {variant_name, patient, trial}.
    Base case patient is structured dict; variants alter presentation.
    """
    base_patient = case["patient"]
    trial = case["trial"]

    return [
        {
            "variant_name": "structured",
            "patient": dict(base_patient),
            "trial": trial,
        },
        {
            "variant_name": "narrative",
            "patient": {"patient_id": base_patient.get("patient_id", ""), "narrative": _patient_to_narrative(base_patient)},
            "trial": trial,
        },
        {
            "variant_name": "noisy_casing",
            "patient": {k: (_noisy_casing(v) if isinstance(v, str) else v) for k, v in base_patient.items()},
            "trial": trial,
        },
        {
            "variant_name": "extra_sentence",
            "patient": {**base_patient, "notes": _add_extra_sentence("Patient referred for clinical trial evaluation.")},
            "trial": trial,
        },
        {
            "variant_name": "reordered_fields",
            "patient": _reorder_fields(base_patient),
            "trial": trial,
        },
        {
            "variant_name": "synonym_wording",
            "patient": _apply_synonyms(base_patient),
            "trial": trial,
        },
    ]


# ---------------------------------------------------------------------------
# Matcher interface
# ---------------------------------------------------------------------------

def validate_matcher_result(result: Any) -> bool:
    """Return True if result is a dict with a valid prediction field."""
    if not isinstance(result, dict):
        return False
    pred = result.get("prediction") or result.get("predicted_label") or result.get("label")
    return str(pred).strip().lower() in VALID_PREDICTIONS


def extract_prediction(result: dict) -> str:
    """Extract the prediction string from a matcher result dict."""
    for key in ("prediction", "predicted_label", "label"):
        val = result.get(key)
        if val and str(val).strip().lower() in VALID_PREDICTIONS:
            return str(val).strip().lower()
    return "unknown"


def run_variant(case: dict, variant: dict) -> dict:
    """
    Run one variant through the matcher and return a result record.
    Catches all exceptions so the script never exits mid-run.
    """
    from app.eligibility.rule_matcher import match_patient_to_trial  # noqa: PLC0415

    patient = variant["patient"]
    trial = variant["trial"]
    error: str | None = None
    raw_result: dict = {}
    prediction = "unknown"
    explanation = ""
    crashed = False

    try:
        raw_result = match_patient_to_trial(patient, trial)
        if not validate_matcher_result(raw_result):
            error = f"Malformed matcher result: {raw_result!r}"
            crashed = True
        else:
            prediction = extract_prediction(raw_result)
            explanation = (
                raw_result.get("explanation")
                or raw_result.get("matcher_explanation")
                or ""
            )
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        crashed = True

    return {
        "case_id": case["case_id"],
        "capability_area": case["capability_area"],
        "variant_name": variant["variant_name"],
        "prediction": prediction,
        "explanation": explanation,
        "crashed": crashed,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_noise_robustness_cases(cases: list[dict]) -> list[dict]:
    """
    For each case, run all variants and annotate each with base_prediction
    and stable_prediction.
    """
    all_results: list[dict] = []

    for case in cases:
        variants = make_variants(case)
        variant_results: list[dict] = []

        for variant in variants:
            result = run_variant(case, variant)
            variant_results.append(result)

        # Base prediction = structured variant
        base_pred = "unknown"
        for r in variant_results:
            if r["variant_name"] == "structured":
                base_pred = r["prediction"]
                break

        for r in variant_results:
            r["base_prediction"] = base_pred
            r["variant_prediction"] = r.pop("prediction")
            r["stable_prediction"] = (
                r["variant_prediction"] == base_pred
                and r["variant_prediction"] != "unknown"
            )

        all_results.extend(variant_results)

    return all_results


def summarize_results(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    crashed = [r for r in results if r["crashed"]]
    stable = [r for r in results if r["stable_prediction"] and not r["crashed"]]
    unstable = [r for r in results if not r["stable_prediction"] and not r["crashed"]]

    by_area: dict[str, dict] = {}
    for r in results:
        area = r["capability_area"]
        if area not in by_area:
            by_area[area] = {"total": 0, "stable": 0, "unstable": 0, "crashed": 0}
        by_area[area]["total"] += 1
        if r["crashed"]:
            by_area[area]["crashed"] += 1
        elif r["stable_prediction"]:
            by_area[area]["stable"] += 1
        else:
            by_area[area]["unstable"] += 1

    return {
        "total_variants": total,
        "stable_count": len(stable),
        "unstable_count": len(unstable),
        "crashed_count": len(crashed),
        "stability_rate": round(len(stable) / total, 4) if total else 0.0,
        "by_capability_area": by_area,
        "results": results,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    try:
        from app.eligibility import rule_matcher as _  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: Cannot import rule_matcher: {exc}", file=sys.stderr)
        sys.exit(1)

    cases = make_base_cases()
    results = run_noise_robustness_cases(cases)
    summary = summarize_results(results)
    write_json(summary, OUTPUT_PATH)

    print(f"Noise robustness report written to: {OUTPUT_PATH}")
    print(f"Total variants run : {summary['total_variants']}")
    print(f"Stable predictions : {summary['stable_count']}")
    print(f"Unstable predictions: {summary['unstable_count']}")
    print(f"Crashed             : {summary['crashed_count']}")
    print(f"Stability rate      : {summary['stability_rate']:.2%}")
    print()
    print("By capability area:")
    for area, counts in summary["by_capability_area"].items():
        print(
            f"  {area:<30} total={counts['total']}  stable={counts['stable']}  "
            f"unstable={counts['unstable']}  crashed={counts['crashed']}"
        )

    if summary["crashed_count"] > 0:
        print("\nERROR: matcher crashed on one or more variants.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
