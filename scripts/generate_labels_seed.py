"""Generate a first-draft labels_seed.json using conservative heuristics only.

Labels are benchmark-style annotations for review. No real clinical eligibility
is inferred. Never use this output to advise patient enrollment.
"""

import json
import re
from pathlib import Path

CANDIDATES_PATH = Path("data/processed/label_candidates.json")
PATIENTS_PATH = Path("data/processed/patient_cases.json")
TRIALS_PATH = Path("data/processed/trial_cases.json")
OUTPUT_PATH = Path("data/processed/labels_seed.json")


def parse_age_years(age_str: str | None) -> float | None:
    """Convert an age string like '18 Years' or '65 Months' to years."""
    if not age_str:
        return None

    match = re.match(
        r"(\d+(?:\.\d+)?)\s*(year|month|week|day)?s?",
        str(age_str).strip(),
        re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = (match.group(2) or "year").lower()

    if unit.startswith("month"):
        return value / 12
    if unit.startswith("week"):
        return value / 52
    if unit.startswith("day"):
        return value / 365
    return value


def patient_age_years(patient: dict) -> float | None:
    """Return patient age as years, or None if unavailable."""
    raw_age = patient.get("age")
    if raw_age is None:
        return None
    if isinstance(raw_age, (int, float)):
        return float(raw_age)
    return parse_age_years(str(raw_age))


def flatten_text(value) -> str:
    """Flatten strings, lists, and dict values into lowercase searchable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value).lower()
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values()).lower()
    return str(value).lower()


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if text contains any keyword."""
    return any(keyword.lower() in text for keyword in keywords)


def patient_text(patient: dict) -> str:
    """Combine relevant patient fields for heuristic checks."""
    return flatten_text(
        [
            patient.get("summary", ""),
            patient.get("diagnosis", ""),
            patient.get("disease_stage", ""),
            patient.get("key_features", []),
            patient.get("exclusions", []),
            patient.get("medications", []),
            patient.get("labs", {}),
            patient.get("category_focus", ""),
        ]
    )


def inclusion_text(trial: dict) -> str:
    """Return flattened trial inclusion criteria text."""
    return flatten_text(trial.get("inclusion_criteria", []))


def exclusion_text(trial: dict) -> str:
    """Return flattened trial exclusion criteria text."""
    return flatten_text(trial.get("exclusion_criteria", []))


def make_label(
    patient_id: str,
    trial_id: str,
    label: str,
    rationale: str,
    patient_facts: list[str],
    trial_criteria: list[str],
) -> dict:
    """Build one seed label record."""
    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "label": label,
        "rationale": rationale,
        "evidence": {
            "patient_facts": patient_facts,
            "trial_criteria": trial_criteria,
        },
        "label_status": "seed_needs_review",
    }


def check_age(patient: dict, trial: dict) -> tuple | None:
    """Flag age outside trial age limits."""
    age = patient_age_years(patient)
    min_age = parse_age_years(trial.get("minimum_age"))
    max_age = parse_age_years(trial.get("maximum_age"))

    if age is None:
        return None

    if min_age is not None and age < min_age:
        return (
            "not_eligible",
            "Patient age is below the trial minimum age.",
            [f"patient age {age:.1f} years"],
            [f"minimum age {trial.get('minimum_age')}"],
        )

    if max_age is not None and age > max_age:
        return (
            "not_eligible",
            "Patient age exceeds the trial maximum age.",
            [f"patient age {age:.1f} years"],
            [f"maximum age {trial.get('maximum_age')}"],
        )

    return None


def check_dbs(patient: dict, trial: dict) -> tuple | None:
    """Flag prior DBS when trial excludes DBS."""
    if not contains_any(exclusion_text(trial), ["dbs", "deep brain stimulation"]):
        return None

    if contains_any(patient_text(patient), ["dbs", "deep brain stimulation", "subthalamic nucleus"]):
        return (
            "not_eligible",
            "Patient has prior DBS and the trial excludes DBS/deep brain stimulation.",
            ["patient history mentions DBS/deep brain stimulation"],
            ["trial exclusion criteria mention DBS/deep brain stimulation"],
        )

    return None


def check_diagnosis_healthy(patient: dict, trial: dict) -> tuple | None:
    """Flag healthy control/no Parkinson when trial requires Parkinson disease."""
    if "parkinson" not in inclusion_text(trial):
        return None

    text = patient_text(patient)
    if contains_any(text, ["healthy control", "no parkinson disease", "no neurological disease"]):
        return (
            "not_eligible",
            "Patient does not have Parkinson disease while the trial requires Parkinson disease.",
            ["patient diagnosis indicates healthy control/no Parkinson disease"],
            ["trial inclusion criteria require Parkinson disease"],
        )

    return None


def check_atypical_parkinsonism(patient: dict, trial: dict) -> tuple | None:
    """Flag atypical/secondary parkinsonism when excluded."""
    if not contains_any(exclusion_text(trial), ["atypical", "secondary parkinsonism"]):
        return None

    if contains_any(patient_text(patient), ["atypical parkinsonism", "secondary parkinsonism", "multiple system atrophy"]):
        return (
            "not_eligible",
            "Patient has atypical/secondary parkinsonism and the trial excludes it.",
            ["patient diagnosis/history mentions atypical or secondary parkinsonism"],
            ["trial exclusion criteria mention atypical or secondary parkinsonism"],
        )

    return None


def check_cognitive_impairment(patient: dict, trial: dict) -> tuple | None:
    """Flag cognitive impairment when excluded by trial criteria."""
    if not contains_any(exclusion_text(trial), ["dementia", "cognitive impairment", "mmse", "moca"]):
        return None

    text = patient_text(patient)
    if contains_any(text, ["cognitive impairment", "moca score 19", "mmse score 22", "dementia"]):
        return (
            "not_eligible",
            "Patient has cognitive impairment and the trial excludes dementia/cognitive impairment.",
            ["patient facts mention cognitive impairment or low cognitive score"],
            ["trial exclusion criteria mention dementia/cognitive impairment/MMSE/MoCA"],
        )

    return None


def check_medication_unclear(patient: dict, trial: dict) -> tuple | None:
    """Flag unclear medication history when stable medication is required."""
    incl = inclusion_text(trial)
    requires_stability = contains_any(
        incl,
        [
            "stable medication",
            "stable levodopa",
            "stable dopaminergic",
            "stable regimen",
            "levodopa regimen",
        ],
    )
    if not requires_stability:
        return None

    text = patient_text(patient)
    if contains_any(text, ["unclear", "not documented", "unavailable", "no recent pharmacy records"]):
        return (
            "unclear",
            "Patient medication history is unclear and the trial requires stable medication.",
            ["patient medication history is unclear or unavailable"],
            ["trial inclusion criteria require stable medication/levodopa regimen"],
        )

    return None


def check_missing_disease_info(patient: dict, trial: dict) -> tuple | None:
    """Flag missing disease severity when criteria require severity details."""
    incl = inclusion_text(trial)
    requires_stage = contains_any(incl, ["hoehn", "updrs", "disease severity", "h&y", "h & y"])

    if not requires_stage:
        return None

    text = patient_text(patient)
    if contains_any(text, ["stage not available", "disease_stage unclear", "disease duration not available"]):
        return (
            "unclear",
            "Patient is missing disease severity or duration details required by the trial.",
            ["patient disease severity or duration is missing/unclear"],
            ["trial inclusion criteria require disease severity or staging information"],
        )

    return None


CHECKS = [
    check_age,
    check_dbs,
    check_diagnosis_healthy,
    check_atypical_parkinsonism,
    check_cognitive_impairment,
    check_medication_unclear,
    check_missing_disease_info,
]


def label_candidate(candidate: dict, patients: dict[str, dict], trials: dict[str, dict]) -> dict:
    """Label one candidate with conservative seed heuristics."""
    patient_id = candidate.get("patient_id", "")
    trial_id = candidate.get("trial_id", "")
    patient = patients.get(patient_id, {})
    trial = trials.get(trial_id, {})

    for check in CHECKS:
        result = check(patient, trial)
        if result:
            label, rationale, patient_facts, trial_criteria = result
            return make_label(patient_id, trial_id, label, rationale, patient_facts, trial_criteria)

    return make_label(
        patient_id,
        trial_id,
        "unclear",
        "No conservative heuristic rule fired; manual review is required.",
        ["no disqualifying patient fact identified by seed heuristics"],
        ["no matching exclusion or required criterion triggered by seed heuristics"],
    )


def main() -> None:
    candidates: list[dict] = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    patient_list: list[dict] = json.loads(PATIENTS_PATH.read_text(encoding="utf-8"))
    trial_list: list[dict] = json.loads(TRIALS_PATH.read_text(encoding="utf-8"))

    patients = {patient["patient_id"]: patient for patient in patient_list if patient.get("patient_id")}
    trials = {trial["trial_id"]: trial for trial in trial_list if trial.get("trial_id")}

    labels = [label_candidate(candidate, patients, trials) for candidate in candidates]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for label in labels:
        counts[label["label"]] = counts.get(label["label"], 0) + 1

    print(f"Written {len(labels)} records to {OUTPUT_PATH}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
