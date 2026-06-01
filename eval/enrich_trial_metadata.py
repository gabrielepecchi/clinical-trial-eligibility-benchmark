"""Enrich trial cases with normalised metadata fields.

Reads data/processed/trial_cases.json and writes an enriched copy to
data/processed/trial_cases_enriched.json. The original file is not modified.

Usage:
    PYTHONPATH=. python eval/enrich_trial_metadata.py
"""

import json
import sys
from pathlib import Path

INPUT_FILE = Path("data/processed/trial_cases.json")
OUTPUT_FILE = Path("data/processed/trial_cases_enriched.json")

_METADATA_FIELDS = ["nct_id", "title", "phase", "status", "intervention_type", "condition"]

# Candidate source field names for each metadata field, in priority order.
_FIELD_CANDIDATES: dict[str, list[str]] = {
    "nct_id": ["nct_id", "trial_id", "id", "nctid"],
    "title": ["title", "official_title", "brief_title", "name"],
    "phase": ["phase", "study_phase", "trial_phase"],
    "status": ["status", "overall_status", "recruitment_status", "study_status"],
    "intervention_type": [
        "intervention_type", "intervention_types", "primary_intervention_type",
        "interventions", "intervention",
    ],
    "condition": ["condition", "conditions", "disease", "indication", "therapeutic_area"],
}


def load_trial_cases(path: Path) -> list[dict]:
    """Load trial cases from a JSON file, exiting on error."""
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


def extract_trial_metadata(trial: dict) -> dict[str, str]:
    """Extract normalised metadata from a trial record.

    Returns a dict with exactly the keys in _METADATA_FIELDS.
    Values are strings; missing values are empty strings.
    No data is invented — only existing fields are used.
    """
    metadata: dict[str, str] = {}
    for field, candidates in _FIELD_CANDIDATES.items():
        value = ""
        for candidate in candidates:
            raw = trial.get(candidate)
            if raw is None:
                continue
            if isinstance(raw, list):
                raw = ", ".join(str(v) for v in raw if v)
            if isinstance(raw, str) and raw.strip():
                value = raw.strip()
                break
            if not isinstance(raw, str) and raw:
                value = str(raw).strip()
                break
        metadata[field] = value
    return metadata


def enrich_trial_case(trial: dict) -> dict:
    """Return a copy of the trial record with normalised metadata fields added.

    Existing fields are preserved. Metadata fields are set only if not already
    present, so the function is safe to run on already-enriched records.
    """
    enriched = dict(trial)
    metadata = extract_trial_metadata(trial)
    for field, value in metadata.items():
        if field not in enriched:
            enriched[field] = value
    return enriched


def enrich_trial_cases(trials: list[dict]) -> list[dict]:
    """Enrich a list of trial records."""
    return [enrich_trial_case(t) for t in trials]


def write_trial_cases(trials: list[dict], path: Path) -> None:
    """Write trial records to a JSON file."""
    path.write_text(json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    trials = load_trial_cases(INPUT_FILE)
    enriched = enrich_trial_cases(trials)
    write_trial_cases(enriched, OUTPUT_FILE)

    print(f"Records read   : {len(trials)}")
    print(f"Records written: {len(enriched)}")
    print(f"Output path    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
