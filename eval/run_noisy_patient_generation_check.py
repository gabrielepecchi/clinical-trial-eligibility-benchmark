"""
run_noisy_patient_generation_check.py — readiness check for noisy patient input generation.
Prepares Task 12 (eligibility-focused trial enrichment) and Task 92 (clean/noisy comparison).

Usage:
    PYTHONPATH=. python eval/run_noisy_patient_generation_check.py
"""

import json
from pathlib import Path

STRUCTURED_PATH = Path("data/processed/patient_cases.json")
NARRATIVE_PATH = Path("data/processed/patient_cases_narrative.json")
NOISY_PATH = Path("data/processed/patient_cases_noisy.json")
REPORT_PATH = Path("reports/noisy_patient_generation_check.json")

RECOMMENDED_NOISE_TYPES = [
    "missing fields",
    "ambiguous values",
    "incomplete medication history",
    "uncertain DBS/device history",
    "temporal ambiguity",
    "negation-heavy narrative phrasing",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_records(path: Path) -> int | None:
    try:
        data = load_json(path)
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def build_report() -> dict:
    structured_exists = STRUCTURED_PATH.exists()
    narrative_exists = NARRATIVE_PATH.exists()
    noisy_exists = NOISY_PATH.exists()

    structured_count = count_records(STRUCTURED_PATH) if structured_exists else None
    narrative_count = count_records(NARRATIVE_PATH) if narrative_exists else None
    noisy_count = count_records(NOISY_PATH) if noisy_exists else None

    missing: list[str] = []
    if not structured_exists:
        missing.append(str(STRUCTURED_PATH))
    if not narrative_exists:
        missing.append(str(NARRATIVE_PATH))
    if not noisy_exists:
        missing.append(str(NOISY_PATH))

    noisy_generation_needed = not noisy_exists

    parts: list[str] = []
    if not structured_exists:
        parts.append(
            "patient_cases.json is missing — the structured patient dataset must exist "
            "before noisy variants can be generated."
        )
    if not narrative_exists:
        parts.append(
            "patient_cases_narrative.json is missing — narrative patient profiles should "
            "be generated first (e.g. via eval/generate_narrative_patients.py) as they "
            "serve as the natural base for noisy variants."
        )
    if noisy_generation_needed:
        parts.append(
            "patient_cases_noisy.json is missing — a dedicated noisy patient generation "
            "script must be created and run. It should take patient_cases.json or "
            "patient_cases_narrative.json as input and introduce realistic clinical noise "
            "per the recommended_noise_types listed in this report. Until this file exists, "
            "Task 12 (eligibility-focused enrichment with noisy inputs) and Task 92 "
            "(clean vs noisy comparison) cannot be completed."
        )

    if not noisy_generation_needed:
        recommendation = (
            "patient_cases_noisy.json already exists. Task 12 and Task 92 can proceed "
            "with the current noisy patient dataset."
        )
    elif not parts:
        recommendation = "Noisy patient generation is needed. Run the generation script."
    else:
        recommendation = " ".join(parts)

    return {
        "structured_patient_file_exists": structured_exists,
        "narrative_patient_file_exists": narrative_exists,
        "noisy_patient_file_exists": noisy_exists,
        "structured_patient_count": structured_count,
        "narrative_patient_count": narrative_count,
        "noisy_patient_count": noisy_count,
        "noisy_generation_needed": noisy_generation_needed,
        "missing_files": missing,
        "recommended_noise_types": RECOMMENDED_NOISE_TYPES,
        "recommendation": recommendation,
    }


def main() -> None:
    report = build_report()
    write_json(report, REPORT_PATH)

    print(f"structured_patient_file_exists: {report['structured_patient_file_exists']}")
    print(f"narrative_patient_file_exists:  {report['narrative_patient_file_exists']}")
    print(f"noisy_patient_file_exists:      {report['noisy_patient_file_exists']}")
    print(f"structured_patient_count:       {report['structured_patient_count']}")
    print(f"narrative_patient_count:        {report['narrative_patient_count']}")
    print(f"noisy_patient_count:            {report['noisy_patient_count']}")
    print(f"noisy_generation_needed:        {report['noisy_generation_needed']}")
    if report["missing_files"]:
        print("missing_files:")
        for f in report["missing_files"]:
            print(f"  - {f}")
    print("recommended_noise_types:")
    for t in report["recommended_noise_types"]:
        print(f"  - {t}")
    print(f"recommendation: {report['recommendation']}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
