"""Read raw ClinicalTrials.gov study JSON and save simplified eligibility records."""

import json
from pathlib import Path

from app.eligibility.criteria_parser import parse_eligibility_criteria

INPUT_FILE = Path("data/raw/parkinson_trials_raw.json")
OUTPUT_FILE = Path("data/processed/eligibility_criteria.json")


def extract_trial(study: dict) -> dict | None:
    """Extract eligibility fields from a single study dict.

    Returns None if no NCT ID is found.
    """
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId")
    if not nct_id:
        return None

    title = id_module.get("briefTitle", "")
    official_title = id_module.get("officialTitle", "")

    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus", "")

    design_module = protocol.get("designModule", {})
    phases = design_module.get("phases", [])
    phase = phases[0] if phases else ""
    study_type = design_module.get("studyType", "")

    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])

    arms_module = protocol.get("armsInterventionsModule", {})
    interventions = [
        i.get("name", "") for i in arms_module.get("interventions", [])
    ]

    eligibility_module = protocol.get("eligibilityModule", {})
    eligibility_text = eligibility_module.get("eligibilityCriteria", "")
    minimum_age = eligibility_module.get("minimumAge", "")
    maximum_age = eligibility_module.get("maximumAge", "")
    sex = eligibility_module.get("sex", "")
    healthy_volunteers = eligibility_module.get("healthyVolunteers", "")

    parsed = parse_eligibility_criteria(eligibility_text)

    return {
        "nct_id": nct_id,
        "title": title,
        "official_title": official_title,
        "overall_status": overall_status,
        "phase": phase,
        "study_type": study_type,
        "conditions": conditions,
        "interventions": interventions,
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "sex": sex,
        "healthy_volunteers": healthy_volunteers,
        "eligibility_text": eligibility_text,
        "inclusion_criteria": parsed["inclusion_criteria"],
        "exclusion_criteria": parsed["exclusion_criteria"],
    }


def main() -> None:
    raw_studies: list[dict] = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    records: list[dict] = []
    skipped = 0
    for study in raw_studies:
        record = extract_trial(study)
        if record is None:
            skipped += 1
        else:
            records.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} records to {OUTPUT_FILE} ({skipped} skipped)")


if __name__ == "__main__":
    main()
