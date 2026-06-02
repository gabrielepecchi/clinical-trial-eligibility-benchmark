"""
run_clean_vs_noisy_input_report.py — Task 92 (partial): clean/narrative/noisy input comparison framework.

Usage:
    PYTHONPATH=. python eval/run_clean_vs_noisy_input_report.py
"""

import json
from pathlib import Path

CLEAN_PATH = Path("data/processed/patient_cases.json")
NARRATIVE_PATH = Path("data/processed/patient_cases_narrative.json")
NOISY_PATH = Path("data/processed/patient_cases_noisy.json")
RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
REPORT_PATH = Path("reports/clean_vs_noisy_input_report.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_patients(path: Path) -> int | None:
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
    clean_exists = CLEAN_PATH.exists()
    narrative_exists = NARRATIVE_PATH.exists()
    noisy_exists = NOISY_PATH.exists()
    results_exists = RESULTS_PATH.exists()

    clean_count = count_patients(CLEAN_PATH) if clean_exists else None
    narrative_count = count_patients(NARRATIVE_PATH) if narrative_exists else None
    noisy_count = count_patients(NOISY_PATH) if noisy_exists else None

    missing: list[str] = []
    if not clean_exists:
        missing.append(str(CLEAN_PATH))
    if not narrative_exists:
        missing.append(str(NARRATIVE_PATH))
    if not noisy_exists:
        missing.append(str(NOISY_PATH))
    if not results_exists:
        missing.append(str(RESULTS_PATH))

    comparison_ready = clean_exists and narrative_exists and noisy_exists and results_exists

    if comparison_ready:
        recommendation = (
            "All required files are present. Task 92 can be fully implemented: "
            "run the matcher against clean, narrative, and noisy patient inputs "
            "and compare accuracy, macro F1, and error rates across input types."
        )
    else:
        parts = []
        if not noisy_exists:
            parts.append(
                "patient_cases_noisy.json is missing — generate noisy patient inputs "
                "first (e.g. via a data augmentation script that introduces realistic "
                "clinical noise: missing fields, ambiguous values, inconsistent units)."
            )
        if not narrative_exists:
            parts.append(
                "patient_cases_narrative.json is missing — generate narrative patient "
                "profiles first (e.g. via eval/generate_narrative_patients.py)."
            )
        if not results_exists:
            parts.append("results_llm_reviewed.json is missing — run the benchmark first.")
        recommendation = (
            "Task 92 cannot be completed yet. " + " ".join(parts)
        )

    return {
        "clean_patient_file_exists": clean_exists,
        "narrative_patient_file_exists": narrative_exists,
        "noisy_patient_file_exists": noisy_exists,
        "results_file_exists": results_exists,
        "clean_patient_count": clean_count,
        "narrative_patient_count": narrative_count,
        "noisy_patient_count": noisy_count,
        "comparison_ready": comparison_ready,
        "missing_files": missing,
        "recommendation": recommendation,
    }


def main() -> None:
    report = build_report()
    write_json(report, REPORT_PATH)

    print(f"clean_patient_file_exists:     {report['clean_patient_file_exists']}")
    print(f"narrative_patient_file_exists: {report['narrative_patient_file_exists']}")
    print(f"noisy_patient_file_exists:     {report['noisy_patient_file_exists']}")
    print(f"results_file_exists:           {report['results_file_exists']}")
    print(f"clean_patient_count:           {report['clean_patient_count']}")
    print(f"narrative_patient_count:       {report['narrative_patient_count']}")
    print(f"noisy_patient_count:           {report['noisy_patient_count']}")
    print(f"comparison_ready:              {report['comparison_ready']}")
    if report["missing_files"]:
        print(f"missing_files:")
        for f in report["missing_files"]:
            print(f"  - {f}")
    print(f"recommendation: {report['recommendation']}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
