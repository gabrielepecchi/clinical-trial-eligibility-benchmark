"""Export labels_seed.json to a review-friendly CSV file."""

import csv
import json
from pathlib import Path

LABELS_FILE = Path("data/processed/labels_seed.json")
PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
OUTPUT_FILE = Path("data/processed/labels_seed_review.csv")


CSV_FIELDS = [
    "patient_id",
    "trial_id",
    "trial_category",
    "label",
    "label_status",
    "patient_summary",
    "trial_title",
    "rationale",
    "patient_facts",
    "trial_criteria",
]


def load_json(path: Path) -> list[dict]:
    """Load a JSON list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def join_list(value) -> str:
    """Convert a list-like value to a readable string."""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value or "")


def build_review_row(label: dict, patients: dict[str, dict], trials: dict[str, dict]) -> dict:
    """Build one CSV row from one seed label."""
    patient = patients.get(label["patient_id"], {})
    trial = trials.get(label["trial_id"], {})
    evidence = label.get("evidence", {})

    return {
        "patient_id": label["patient_id"],
        "trial_id": label["trial_id"],
        "trial_category": trial.get("category", ""),
        "label": label.get("label", ""),
        "label_status": label.get("label_status", ""),
        "patient_summary": patient.get("summary", ""),
        "trial_title": trial.get("title", ""),
        "rationale": label.get("rationale", ""),
        "patient_facts": join_list(evidence.get("patient_facts", [])),
        "trial_criteria": join_list(evidence.get("trial_criteria", [])),
    }


def main() -> None:
    labels = load_json(LABELS_FILE)
    patient_list = load_json(PATIENTS_FILE)
    trial_list = load_json(TRIALS_FILE)

    patients = {patient["patient_id"]: patient for patient in patient_list}
    trials = {trial["trial_id"]: trial for trial in trial_list}

    rows = [build_review_row(label, patients, trials) for label in labels]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} review rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
