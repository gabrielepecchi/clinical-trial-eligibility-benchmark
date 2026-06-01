"""Export unified benchmark records merging patients, trials, and labels.

Reads:
    data/processed/patient_cases.json
    data/processed/trial_cases.json
    data/processed/labels_llm_reviewed.json

Writes:
    data/processed/unified_benchmark.json

Usage:
    PYTHONPATH=. python eval/export_unified_benchmark.py
"""

import json
import sys
from pathlib import Path

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")
OUTPUT_FILE = Path("data/processed/unified_benchmark.json")

BENCHMARK_VERSION = "v0.1"

_CRITERIA_FIELDS = [
    "criteria_text", "eligibility_criteria", "inclusion_criteria",
    "exclusion_criteria", "criteria", "inclusion", "exclusion",
    "inclusion_text", "exclusion_text",
]


def load_json_list(path: Path) -> list[dict]:
    """Load a JSON list from disk, exiting on error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"ERROR: Expected a JSON list in {path}", file=sys.stderr)
        sys.exit(1)
    return data


def index_by_id(records: list[dict], id_field: str) -> dict[str, dict]:
    """Return a dict keyed by the value of id_field for each record."""
    return {
        r[id_field]: r
        for r in records
        if id_field in r and r[id_field]
    }


def extract_trial_criteria(trial: dict) -> dict:
    """Return a dict of available criteria fields from a trial record."""
    return {
        field: trial[field]
        for field in _CRITERIA_FIELDS
        if field in trial and trial[field]
    }


def build_unified_record(
    label_record: dict,
    patient_index: dict[str, dict],
    trial_index: dict[str, dict],
) -> dict | None:
    """Build a unified benchmark record from a label entry and the indexes.

    Returns None if the patient_id or trial_id is missing from the indexes.
    """
    patient_id = label_record.get("patient_id", "")
    trial_id = label_record.get("trial_id", "")

    patient = patient_index.get(patient_id)
    trial = trial_index.get(trial_id)

    if patient is None or trial is None:
        return None

    benchmark_id = f"{patient_id}__{trial_id}"

    criteria = extract_trial_criteria(trial)

    label = {
        "gold_label": label_record.get("label", ""),
        "label_status": label_record.get("label_status", ""),
        "rationale": label_record.get("rationale", ""),
        "evidence": label_record.get("evidence", {}),
        "source": "llm_reviewed",
    }

    metadata = {
        "synthetic_patient": True,
        "label_source": str(LABELS_FILE),
        "trial_source": "ClinicalTrials.gov",
        "benchmark_version": label_record.get("benchmark_version", BENCHMARK_VERSION),
    }

    return {
        "benchmark_id": benchmark_id,
        "patient": patient,
        "trial": trial,
        "criteria": criteria,
        "label": label,
        "metadata": metadata,
    }


def build_unified_benchmark(
    patients: list[dict],
    trials: list[dict],
    labels: list[dict],
) -> tuple[list[dict], int]:
    """Build the full list of unified records and return (records, skipped_count)."""
    patient_index = index_by_id(patients, "patient_id")
    trial_index = index_by_id(trials, "trial_id")

    records: list[dict] = []
    skipped = 0

    for label_record in labels:
        record = build_unified_record(label_record, patient_index, trial_index)
        if record is None:
            skipped += 1
        else:
            records.append(record)

    return records, skipped


def write_json(data: object, path: Path) -> None:
    """Write data as indented JSON to path."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    patients = load_json_list(PATIENTS_FILE)
    trials = load_json_list(TRIALS_FILE)
    labels = load_json_list(LABELS_FILE)

    records, skipped = build_unified_benchmark(patients, trials, labels)

    write_json(records, OUTPUT_FILE)

    print(f"Patients read          : {len(patients)}")
    print(f"Trials read            : {len(trials)}")
    print(f"Labels read            : {len(labels)}")
    print(f"Unified records written: {len(records)}")
    print(f"Skipped records        : {skipped}")
    print(f"Output path            : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
