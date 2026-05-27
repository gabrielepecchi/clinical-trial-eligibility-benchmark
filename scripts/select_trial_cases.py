"""Select up to 60 benchmark trial cases from extracted eligibility records."""

import json
from pathlib import Path

INPUT_FILE = Path("data/processed/eligibility_criteria.json")
OUTPUT_FILE = Path("data/processed/trial_cases.json")
MAX_TRIALS = 60

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "device": [
        "dbs",
        "deep brain stimulation",
        "device",
        "implant",
        "ultrasound",
        "tms",
        "transcranial",
        "stimulation",
        "lfp sensing",
        "directional leads",
        "virtual reality",
    ],
    "rehabilitation": [
        "exercise",
        "rehabilitation",
        "physical therapy",
        "physiotherapy",
        "home physiotherapy",
        "gait",
        "freezing of gait",
        "fog",
        "balance",
        "falls",
        "fall prevention",
        "treadmill",
        "agility training",
        "yoga",
        "tai chi",
        "alexander technique",
        "feldenkrais",
        "somatosensory",
        "motor skills",
        "quality of life",
    ],
    "non_motor_symptoms": [
        "sleep",
        "depression",
        "anxiety",
        "worry",
        "psychosocial",
        "neuropsychiatric",
        "psychiatric",
        "fluctuations scale",
        "scale",
        "validation",
        "cognition",
        "cognitive",
        "dementia",
        "pain",
        "fatigue",
        "threat interpretation bias",
        "behavioral physiology",
        "non-motor",
        "non motor",
    ],
    "advanced_therapy": [
        "gene therapy",
        "stem cell",
        "infusion",
        "intestinal gel",
        "lcig",
        "duodopa",
        "apomorphine",
        "pump",
        "brt-da01",
    ],
    "biomarker": [
        "biomarker",
        "biopsy",
        "biopsies",
        "enteric nervous system",
        "imaging",
        "pet scan",
        "pet",
        "mri",
        "fmri",
        "blood",
        "csf",
        "alpha-synuclein",
        "substantia nigra",
        "iron",
    ],
    "drug_treatment": [],  # default fallback
}


def assign_category(title: str, interventions: list[str]) -> str:
    """Assign a category based on title and intervention keywords."""
    text = " ".join([title] + interventions).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "drug_treatment":
            continue
        for kw in keywords:
            if kw in text:
                return category
    return "drug_treatment"


def is_valid(record: dict) -> bool:
    """Return True if the record has required fields and at least one criterion."""
    return bool(
        record.get("nct_id")
        and record.get("title")
        and record.get("eligibility_text")
        and (record.get("inclusion_criteria") or record.get("exclusion_criteria"))
    )


def build_trial_case(record: dict, index: int) -> dict:
    """Build a trial case dict from an eligibility record."""
    nct_id = record["nct_id"]
    interventions = record.get("interventions") or []
    return {
        "trial_id": f"T{index:03d}",
        "nct_id": nct_id,
        "category": assign_category(record.get("title", ""), interventions),
        "title": record.get("title", ""),
        "official_title": record.get("official_title", ""),
        "conditions": record.get("conditions") or [],
        "interventions": interventions,
        "overall_status": record.get("overall_status", ""),
        "phase": record.get("phase", ""),
        "study_type": record.get("study_type", ""),
        "minimum_age": record.get("minimum_age", ""),
        "maximum_age": record.get("maximum_age", ""),
        "sex": record.get("sex", ""),
        "healthy_volunteers": record.get("healthy_volunteers", ""),
        "inclusion_criteria": record.get("inclusion_criteria") or [],
        "exclusion_criteria": record.get("exclusion_criteria") or [],
        "raw_eligibility": record.get("eligibility_text", ""),
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def main() -> None:
    records: list[dict] = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    selected = []
    for record in records:
        if len(selected) >= MAX_TRIALS:
            break
        if is_valid(record):
            selected.append(build_trial_case(record, len(selected) + 1))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} trial cases to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
