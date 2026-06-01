"""Stress tests for the rule-based eligibility matcher.

Verifies the matcher does not crash and returns valid structured output
for extreme, conflicting, or unusual patient/trial inputs.

Usage:
    PYTHONPATH=. python eval/run_stress_tests.py
"""

import sys
from app.eligibility.rule_matcher import match_patient_to_trial

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

# ---------------------------------------------------------------------------
# Stress-test cases
# ---------------------------------------------------------------------------

STRESS_CASES = [
    {
        "name": "extremely long eligibility criteria",
        "patient": {
            "patient_id": "stress_001",
            "age": 65,
            "diagnosis": "idiopathic Parkinson disease",
            "medications": ["levodopa/carbidopa"],
        },
        "trial": {
            "trial_id": "stress_T001",
            "eligibility_criteria": (
                "Inclusion: Diagnosis of idiopathic Parkinson disease. "
                "Age 40 to 80. Stable medication for at least 4 weeks. "
                "No prior deep brain stimulation. "
                "Able to provide informed consent. "
                "Willing to attend follow-up visits. "
                "No significant cardiovascular disease. "
                "No history of stroke. "
                "No active malignancy. "
                "No severe renal impairment. "
                "No severe hepatic impairment. "
                "No concurrent participation in another interventional trial. "
                "No known hypersensitivity to study drug components. "
                "No pregnancy or breastfeeding. "
                "No significant psychiatric disorder other than depression. "
                "No cognitive impairment precluding informed consent. "
                "No prior neurosurgical procedure. "
                "No implanted electrical device. "
                "No severe dyskinesia at baseline. "
                "No history of neuroleptic malignant syndrome. "
                "No clinically significant abnormal laboratory values. "
                + ("No additional exclusion. " * 100)
            ),
        },
    },
    {
        "name": "patient with many medications",
        "patient": {
            "patient_id": "stress_002",
            "age": 68,
            "diagnosis": "Parkinson disease",
            "medications": [
                "levodopa/carbidopa",
                "pramipexole",
                "rasagiline",
                "amantadine",
                "rivastigmine",
                "donepezil",
                "quetiapine",
                "sertraline",
                "atorvastatin",
                "amlodipine",
                "metformin",
                "omeprazole",
                "aspirin",
                "vitamin D",
                "melatonin",
                "clonazepam",
                "gabapentin",
                "lisinopril",
                "furosemide",
                "warfarin",
            ],
        },
        "trial": {
            "trial_id": "stress_T002",
            "eligibility_criteria": (
                "Inclusion: Parkinson disease diagnosis. Age 50 to 80. "
                "Exclusion: Current MAO-B inhibitor use. No concurrent anticoagulant therapy."
            ),
        },
    },
    {
        "name": "conflicting patient facts",
        "patient": {
            "patient_id": "stress_003",
            "age": 72,
            "diagnosis": "idiopathic Parkinson disease",
            "dbs_history": True,
            "dbs_history_notes": "no prior DBS",
            "cognitive_status": "normal",
            "moca_score": 14,
            "medications": ["levodopa/carbidopa"],
        },
        "trial": {
            "trial_id": "stress_T003",
            "eligibility_criteria": (
                "Inclusion: Idiopathic Parkinson disease. Age >= 40. "
                "Exclusion: Prior deep brain stimulation. "
                "Exclusion: Cognitive impairment (MoCA < 24)."
            ),
        },
    },
    {
        "name": "ambiguous comorbidity and safety signal",
        "patient": {
            "patient_id": "stress_004",
            "age": 60,
            "diagnosis": "Parkinson disease",
            "comorbidities": [
                "arrhythmia",
                "orthostatic hypotension",
                "mild renal insufficiency",
            ],
            "medications": ["levodopa/carbidopa"],
        },
        "trial": {
            "trial_id": "stress_T004",
            "eligibility_criteria": (
                "Inclusion: Parkinson disease. Age 50-75. "
                "Exclusion: Clinically significant cardiovascular disease. "
                "Exclusion: Severe renal impairment."
            ),
        },
    },
    {
        "name": "nested and complex inclusion/exclusion wording",
        "patient": {
            "patient_id": "stress_005",
            "age": 55,
            "diagnosis": "Parkinson disease",
            "medications": ["levodopa/carbidopa", "pramipexole"],
        },
        "trial": {
            "trial_id": "stress_T005",
            "eligibility_criteria": (
                "Inclusion: Patients with idiopathic Parkinson disease who either "
                "(a) have motor fluctuations despite optimised levodopa therapy, or "
                "(b) have dyskinesia affecting quality of life, or "
                "(c) are candidates for advanced therapy as determined by a movement disorder specialist. "
                "Exclusion: Patients who have (1) prior DBS, or (2) prior ablative surgery, or "
                "(3) are currently enrolled in another trial, unless a washout of 30 days has elapsed. "
                "Patients with atypical parkinsonism, vascular parkinsonism, or drug-induced parkinsonism "
                "are excluded unless confirmed idiopathic PD is additionally documented."
            ),
        },
    },
    {
        "name": "missing most optional patient fields",
        "patient": {
            "patient_id": "stress_006",
            "diagnosis": "Parkinson disease",
        },
        "trial": {
            "trial_id": "stress_T006",
            "eligibility_criteria": (
                "Inclusion: Parkinson disease. Age 40 to 80. "
                "Stable levodopa dose for 4 weeks. "
                "Exclusion: Prior DBS."
            ),
        },
    },
    {
        "name": "device and procedure heavy trial criteria",
        "patient": {
            "patient_id": "stress_007",
            "age": 63,
            "diagnosis": "idiopathic Parkinson disease",
            "dbs_history": False,
            "medications": ["levodopa/carbidopa"],
            "comorbidities": ["cardiac pacemaker"],
        },
        "trial": {
            "trial_id": "stress_T007",
            "eligibility_criteria": (
                "Inclusion: Idiopathic Parkinson disease. DBS candidacy confirmed. "
                "Exclusion: Implanted cardiac pacemaker or defibrillator. "
                "Exclusion: Active implanted neurostimulator other than DBS. "
                "Exclusion: Prior ablative brain surgery. "
                "Exclusion: MRI-incompatible implants. "
                "Exclusion: Cochlear implant. "
                "Exclusion: Deep brain stimulation already implanted if enrolling as naive candidate."
            ),
        },
    },
    {
        "name": "entirely empty trial criteria",
        "patient": {
            "patient_id": "stress_008",
            "age": 67,
            "diagnosis": "Parkinson disease",
            "medications": ["levodopa/carbidopa"],
        },
        "trial": {
            "trial_id": "stress_T008",
            "eligibility_criteria": "",
        },
    },
    {
        "name": "patient with no diagnosis field",
        "patient": {
            "patient_id": "stress_009",
            "age": 58,
            "medications": ["levodopa/carbidopa"],
        },
        "trial": {
            "trial_id": "stress_T009",
            "eligibility_criteria": (
                "Inclusion: Parkinson disease diagnosis required. Age 40-80."
            ),
        },
    },
    {
        "name": "healthy control against PD-only trial",
        "patient": {
            "patient_id": "stress_010",
            "age": 55,
            "diagnosis": "healthy control",
            "medications": [],
        },
        "trial": {
            "trial_id": "stress_T010",
            "eligibility_criteria": (
                "Inclusion: Idiopathic Parkinson disease. Age 40-75. "
                "Exclusion: Any neurological disorder other than PD."
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(result: object) -> list[str]:
    """Return a list of failure messages; empty list means valid."""
    failures = []
    if not isinstance(result, dict):
        failures.append(f"result is not a dict (got {type(result).__name__})")
        return failures
    if "prediction" not in result:
        failures.append("missing key: prediction")
    elif result["prediction"] not in VALID_LABELS:
        failures.append(f"invalid prediction value: {result['prediction']!r}")
    if "confidence" not in result:
        failures.append("missing key: confidence")
    if "explanation" not in result:
        failures.append("missing key: explanation")
    for key in ("matched_facts", "blocking_criteria", "uncertain_criteria"):
        if key in result and not isinstance(result[key], list):
            failures.append(f"{key} is not a list")
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    total = len(STRESS_CASES)
    passed = 0
    failed = 0

    print(f"\n=== Stress Test Report ===")
    print(f"Total cases: {total}\n")

    for case in STRESS_CASES:
        name = case["name"]
        patient = case["patient"]
        trial = case["trial"]

        try:
            result = match_patient_to_trial(patient, trial)
            failures = validate_result(result)
        except Exception as exc:
            failures = [f"exception: {exc}"]
            result = {}

        if failures:
            failed += 1
            status = "FAIL"
            detail = "; ".join(failures)
            prediction = result.get("prediction", "—") if isinstance(result, dict) else "—"
            print(f"  [{status}] {name}")
            print(f"           prediction: {prediction}")
            print(f"           reason    : {detail}")
        else:
            passed += 1
            status = "PASS"
            print(f"  [{status}] {name}  →  {result['prediction']}")

    print(f"\n--- Summary ---")
    print(f"Passed: {passed} / {total}")
    print(f"Failed: {failed} / {total}")

    if failed:
        print(f"\nFAIL: {failed} stress case(s) failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nOK: All stress cases passed.")


if __name__ == "__main__":
    main()
