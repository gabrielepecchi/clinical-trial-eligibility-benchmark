"""Generate candidate patient-trial pairs for manual eligibility labeling."""

import json
from pathlib import Path

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
OUTPUT_FILE = Path("data/processed/label_candidates.json")

MAX_PAIRS = 150
PAIRS_PER_TRIAL = 3


CATEGORY_TO_PATIENT_FOCUS = {
    "rehabilitation": [
        "gait_and_falls",
        "older_frail_patient",
        "early_parkinson",
        "insufficient_patient_detail",
        "device_or_imaging_exclusion",
        "autonomic_symptoms",
    ],
    "device": [
        "prior_dbs",
        "device_or_imaging_exclusion",
        "gait_and_falls",
        "biomarker_imaging",
        "cognitive_impairment",
        "healthy_control",
    ],
    "non_motor_symptoms": [
        "mood_symptoms",
        "sleep_symptoms",
        "cognitive_impairment",
        "autonomic_symptoms",
        "insufficient_patient_detail",
        "advanced_therapy",
    ],
    "biomarker": [
        "biomarker_imaging",
        "healthy_control",
        "unclear_diagnosis",
        "device_or_imaging_exclusion",
        "atypical_parkinsonism",
        "early_parkinson",
    ],
    "advanced_therapy": [
        "advanced_motor_fluctuations",
        "advanced_therapy",
        "prior_dbs",
        "older_frail_patient",
        "medical_comorbidity",
        "recent_trial_participation",
    ],
    "drug_treatment": [
        "early_parkinson",
        "advanced_motor_fluctuations",
        "unclear_medication_history",
        "atypical_parkinsonism",
        "medical_comorbidity",
        "recent_trial_participation",
        "early_onset",
        "older_frail_patient",
    ],
}


def load_json(path: Path) -> list[dict]:
    """Load a JSON list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def patient_matches_category(patient: dict, category: str) -> bool:
    """Return True if a patient is useful for labeling a trial category."""
    focus = patient.get("category_focus", "")
    return focus in CATEGORY_TO_PATIENT_FOCUS.get(category, [])


def build_candidate(patient: dict, trial: dict) -> dict:
    """Build one candidate labeling record without assigning a gold label."""
    return {
        "patient_id": patient["patient_id"],
        "trial_id": trial["trial_id"],
        "trial_category": trial.get("category", ""),
        "suggested_label": "",
        "rationale": "",
        "evidence": {
            "patient_facts": [],
            "trial_criteria": [],
        },
        "label_status": "needs_manual_review",
    }


def choose_patients_for_trial(
    patients: list[dict],
    category: str,
    patient_counts: dict[str, int],
) -> list[dict]:
    """Choose a balanced set of patients for one trial category."""
    matched = [
        patient
        for patient in patients
        if patient_matches_category(patient, category)
    ]

    if len(matched) < PAIRS_PER_TRIAL:
        matched_ids = {patient["patient_id"] for patient in matched}
        matched.extend(
            patient for patient in patients if patient["patient_id"] not in matched_ids
        )

    return sorted(
        matched,
        key=lambda patient: (patient_counts.get(patient["patient_id"], 0), patient["patient_id"]),
    )[:PAIRS_PER_TRIAL]


def generate_candidates(patients: list[dict], trials: list[dict]) -> list[dict]:
    """Generate deterministic and balanced patient-trial pairs for manual review."""
    candidates: list[dict] = []
    patient_counts = {patient["patient_id"]: 0 for patient in patients}

    for trial in trials:
        if len(candidates) >= MAX_PAIRS:
            break

        selected_patients = choose_patients_for_trial(
            patients,
            trial.get("category", ""),
            patient_counts,
        )

        for patient in selected_patients:
            candidates.append(build_candidate(patient, trial))
            patient_counts[patient["patient_id"]] += 1
            if len(candidates) >= MAX_PAIRS:
                return candidates

    return candidates


def main() -> None:
    patients = load_json(PATIENTS_FILE)
    trials = load_json(TRIALS_FILE)

    candidates = generate_candidates(patients, trials)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    print(f"Generated {len(candidates)} label candidates at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
