"""
run_no_internet_required_check.py — Task 59: Offline/local readiness check.

Documents that after ClinicalTrials.gov data has been downloaded, the full
benchmark evaluation and reporting pipeline can run locally without internet.
No network calls are made by this script.

Usage:
    PYTHONPATH=. python eval/run_no_internet_required_check.py
    PYTHONPATH=. python eval/run_no_internet_required_check.py --output PATH
"""

import json
import os
import sys
import argparse

DEFAULT_OUTPUT = "reports/no_internet_required_check.json"

REQUIRED_FILES = [
    "data/raw/parkinson_trials_raw.json",
    "data/processed/patient_cases.json",
    "data/processed/trial_cases.json",
    "data/processed/labels_llm_reviewed.json",
]

OPTIONAL_OUTPUTS = [
    "data/processed/results_llm_reviewed.json",
    "data/processed/error_analysis_llm_reviewed.json",
]

NOTE = (
    "After ClinicalTrials.gov data has been downloaded and patient/trial cases and "
    "labels have been generated, the full benchmark evaluation and reporting pipeline "
    "can run locally without internet access. This script checks local file availability "
    "only and makes no network calls."
)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_files(paths: list) -> tuple:
    """Return (present: list, missing: list)."""
    present = [p for p in paths if os.path.isfile(p)]
    missing = [p for p in paths if not os.path.isfile(p)]
    return present, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline/local readiness check.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    _, missing_required = check_files(REQUIRED_FILES)
    _, missing_optional = check_files(OPTIONAL_OUTPUTS)

    offline_ready = len(missing_required) == 0

    report = {
        "offline_ready": offline_ready,
        "missing_required_files": missing_required,
        "missing_optional_outputs": missing_optional,
        "checked_required_files": REQUIRED_FILES,
        "checked_optional_outputs": OPTIONAL_OUTPUTS,
        "note": NOTE,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Offline ready            : {offline_ready}")
    print(f"Required files checked   : {len(REQUIRED_FILES)}")
    print(f"Missing required         : {len(missing_required)}")
    if missing_required:
        for p in missing_required:
            print(f"  MISSING: {p}")
    print(f"Optional outputs checked : {len(OPTIONAL_OUTPUTS)}")
    print(f"Missing optional         : {len(missing_optional)}")
    if missing_optional:
        for p in missing_optional:
            print(f"  NOT YET GENERATED: {p}")
    print(f"Report written           : {args.output}")


if __name__ == "__main__":
    main()
