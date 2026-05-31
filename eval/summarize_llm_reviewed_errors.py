"""Summarize errors from the LLM-reviewed draft benchmark results."""

import csv
import json
from collections import Counter
from pathlib import Path

RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")
OUTPUT_FILE = Path("data/processed/error_analysis_llm_reviewed.json")
ERROR_CSV_FILE = Path("data/processed/error_analysis_llm_reviewed.csv")


_CSV_FIELDNAMES = [
    "case_id", "patient_id", "trial_id", "gold_label", "predicted_label",
    "error_type", "severity", "explanation", "possible_fix",
    "blocking_criteria", "uncertain_criteria",
]


def format_severity_breakdown(errors: list[dict]) -> str:
    """Format severity breakdown as a printable string."""
    counts = Counter(e["severity"] for e in errors)
    if not counts:
        return "Errors by severity:"
    lines = ["Errors by severity:"]
    for severity, count in counts.most_common():
        lines.append(f"  {severity:<20} {count}")
    return "\n".join(lines)


def build_error_csv_rows(error_records: list[dict]) -> list[dict]:
    rows = []
    for i, r in enumerate(error_records, start=1):
        rows.append({
            "case_id": i,
            "patient_id": r.get("patient_id", ""),
            "trial_id": r.get("trial_id", ""),
            "gold_label": r.get("gold_label", ""),
            "predicted_label": r.get("predicted_label", ""),
            "error_type": r.get("error_type", ""),
            "severity": r.get("severity", ""),
            "explanation": r.get("matcher_explanation", ""),
            "possible_fix": "",
            "blocking_criteria": "; ".join(r.get("blocking_criteria") or []),
            "uncertain_criteria": "; ".join(r.get("uncertain_criteria") or []),
        })
    return rows


def write_error_csv_rows(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_error_severity(record: dict) -> str:
    """Assign a severity level based on gold/predicted labels."""
    gold = record.get("gold_label", "")
    pred = record.get("predicted_label", "")
    if gold == pred:
        return "none"
    if gold == "not_eligible" and pred == "eligible":
        return "critical"
    if gold == "not_eligible" and pred == "unclear":
        return "major_minor"
    if (gold == "unclear" and pred in {"eligible", "not_eligible"}) or \
       (gold in {"eligible", "not_eligible"} and pred == "unclear"):
        return "major"
    if gold == "eligible" and pred == "not_eligible":
        return "minor"
    return "other"


def classify_error(record: dict) -> str:
    """Assign a simple error type from gold/predicted labels and evidence."""
    gold = record.get("gold_label", "")
    pred = record.get("predicted_label", "")
    text = " ".join(
        [
            record.get("gold_rationale", ""),
            " ".join(record.get("gold_evidence", {}).get("patient_facts", [])),
            " ".join(record.get("gold_evidence", {}).get("trial_criteria", [])),
            " ".join(record.get("blocking_criteria", [])),
            " ".join(record.get("uncertain_criteria", [])),
        ]
    ).lower()

    if gold == pred:
        return "correct"

    if gold == "unclear" and pred == "eligible":
        if any(k in text for k in ["missing", "not documented", "cannot confirm", "uncertain", "unclear"]):
            return "missed_uncertainty_missing_detail"
        if any(k in text for k in ["pacemaker", "dbs", "device", "stimulation", "mri"]):
            return "missed_device_uncertainty"
        if any(k in text for k in ["frailty", "falls", "orthostatic", "safety"]):
            return "missed_safety_uncertainty"
        return "overcalled_eligible"

    if gold == "not_eligible" and pred == "eligible":
        if any(k in text for k in ["healthy control", "no parkinson", "diagnosis required", "unconfirmed diagnosis"]):
            return "missed_diagnosis_exclusion"
        if any(k in text for k in ["cognitive", "moca", "mmse", "dementia"]):
            return "missed_cognitive_exclusion"
        if any(k in text for k in ["depression", "psychiatric", "neuropsychiatric"]):
            return "missed_psychiatric_exclusion"
        if any(k in text for k in ["pacemaker", "rtms", "tacs", "stimulation", "dbs", "mri"]):
            return "missed_device_exclusion"
        if any(k in text for k in ["atypical", "secondary parkinsonism"]):
            return "missed_atypical_parkinsonism"
        if any(k in text for k in ["advanced", "early", "prior enrollment", "open-label extension"]):
            return "missed_specific_inclusion_requirement"
        return "missed_exclusion"

    if gold == "eligible" and pred == "not_eligible":
        if any(k in text for k in ["age", "stage", "1-3", "3"]):
            return "overstrict_age_or_stage_rule"
        if any(k in text for k in ["dbs", "deep brain stimulation"]):
            return "overstrict_dbs_rule"
        return "overcalled_not_eligible"

    if gold == "unclear" and pred == "not_eligible":
        return "overcalled_not_eligible_instead_of_unclear"

    if gold == "eligible" and pred == "unclear":
        return "overcalled_unclear"

    if gold == "not_eligible" and pred == "unclear":
        return "undercalled_not_eligible_as_unclear"

    return "other_error"


def build_error_record(record: dict) -> dict:
    """Build one compact error-analysis record."""
    return {
        "patient_id": record.get("patient_id", ""),
        "trial_id": record.get("trial_id", ""),
        "gold_label": record.get("gold_label", ""),
        "predicted_label": record.get("predicted_label", ""),
        "error_type": classify_error(record),
        "severity": classify_error_severity(record),
        "gold_rationale": record.get("gold_rationale", ""),
        "matcher_explanation": record.get("matcher_explanation", ""),
        "blocking_criteria": record.get("blocking_criteria", []),
        "uncertain_criteria": record.get("uncertain_criteria", []),
    }


def main() -> None:
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    predictions = results.get("predictions", [])

    errors = [
        build_error_record(record)
        for record in predictions
        if record.get("gold_label") != record.get("predicted_label")
    ]

    OUTPUT_FILE.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    csv_rows = build_error_csv_rows(errors)
    write_error_csv_rows(csv_rows, ERROR_CSV_FILE)

    print("=== LLM-Reviewed Benchmark Error Analysis ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Errors:            {len(errors)}")
    print(f"Correct:           {len(predictions) - len(errors)}")

    print("\nErrors by type:")
    for error_type, count in Counter(e["error_type"] for e in errors).most_common():
        print(f"  {error_type:<40} {count}")

    print(format_severity_breakdown(errors))

    print("\nErrors by gold/predicted pair:")
    pair_counts = Counter((e["gold_label"], e["predicted_label"]) for e in errors)
    for (gold, pred), count in pair_counts.most_common():
        print(f"  {gold:<12} -> {pred:<12} {count}")

    print("\nFirst 15 errors:")
    for error in errors[:15]:
        print(
            f"  {error['patient_id']} -> {error['trial_id']} | "
            f"{error['gold_label']} vs {error['predicted_label']} | "
            f"{error['error_type']}"
        )

    print(f"\nSaved detailed errors to {OUTPUT_FILE}")
    print(f"Error CSV saved to {ERROR_CSV_FILE}")


if __name__ == "__main__":
    main()
