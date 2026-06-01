"""
validate_schema.py — Task 28: processed dataset schema validation.

Usage:
    PYTHONPATH=. python eval/validate_schema.py
"""

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/processed")

FILES = {
    "patient_cases":    DATA_DIR / "patient_cases.json",
    "trial_cases":      DATA_DIR / "trial_cases.json",
    "labels":           DATA_DIR / "labels_llm_reviewed.json",
    "results":          DATA_DIR / "results_llm_reviewed.json",
    "error_analysis":   DATA_DIR / "error_analysis_llm_reviewed.json",
}

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

CRITERIA_FIELDS = {
    "criteria_text", "eligibility_criteria", "inclusion_criteria",
    "exclusion_criteria", "criteria", "inclusion", "exclusion",
}

PATIENT_OPTIONAL = [
    "age", "sex", "diagnosis", "disease_stage", "medications",
    "key_features", "labs", "category_focus",
]

TRIAL_OPTIONAL = [
    "title", "nct_id", "phase", "overall_status", "study_type",
    "minimum_age", "maximum_age",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def add_issue(
    issues: list,
    severity: str,
    file_name: str,
    record_id: str,
    field: str,
    message: str,
) -> None:
    issues.append({
        "severity": severity,
        "file": file_name,
        "record_id": record_id,
        "field": field,
        "message": message,
    })


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------

def validate_patient_cases(data: Any) -> list:
    issues: list = []
    fname = "patient_cases.json"

    if not isinstance(data, list):
        add_issue(issues, "ERROR", fname, "", "root", "Must be a JSON array.")
        return issues

    seen_ids: set = set()
    for i, rec in enumerate(data):
        rid = f"index {i}"
        if not isinstance(rec, dict):
            add_issue(issues, "ERROR", fname, rid, "root", "Each record must be a dict.")
            continue

        pid = rec.get("patient_id", "")
        if not pid:
            add_issue(issues, "ERROR", fname, rid, "patient_id", "Missing or empty patient_id.")
        else:
            rid = pid
            if pid in seen_ids:
                add_issue(issues, "ERROR", fname, pid, "patient_id", "Duplicate patient_id.")
            seen_ids.add(pid)

        for field in PATIENT_OPTIONAL:
            if field not in rec:
                add_issue(issues, "WARN", fname, rid, field, f"Optional field '{field}' missing.")

    return issues


def validate_trial_cases(data: Any) -> list:
    issues: list = []
    fname = "trial_cases.json"

    if not isinstance(data, list):
        add_issue(issues, "ERROR", fname, "", "root", "Must be a JSON array.")
        return issues

    seen_ids: set = set()
    for i, rec in enumerate(data):
        rid = f"index {i}"
        if not isinstance(rec, dict):
            add_issue(issues, "ERROR", fname, rid, "root", "Each record must be a dict.")
            continue

        tid = rec.get("trial_id", "")
        if not tid:
            add_issue(issues, "ERROR", fname, rid, "trial_id", "Missing or empty trial_id.")
        else:
            rid = tid
            if tid in seen_ids:
                add_issue(issues, "ERROR", fname, tid, "trial_id", "Duplicate trial_id.")
            seen_ids.add(tid)

        has_criteria = any(
            rec.get(f) for f in CRITERIA_FIELDS
        )
        if not has_criteria:
            add_issue(issues, "ERROR", fname, rid, "criteria", "No criteria text field found.")

        for field in TRIAL_OPTIONAL:
            if field not in rec:
                add_issue(issues, "WARN", fname, rid, field, f"Optional field '{field}' missing.")

    return issues


def validate_labels(data: Any) -> list:
    issues: list = []
    fname = "labels_llm_reviewed.json"

    if not isinstance(data, list):
        add_issue(issues, "ERROR", fname, "", "root", "Must be a JSON array.")
        return issues

    seen_pairs: set = set()
    for i, rec in enumerate(data):
        rid = f"index {i}"
        if not isinstance(rec, dict):
            add_issue(issues, "ERROR", fname, rid, "root", "Each record must be a dict.")
            continue

        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        label = rec.get("label", "")
        rid = f"{pid}_{tid}" if pid and tid else rid

        if not pid:
            add_issue(issues, "ERROR", fname, rid, "patient_id", "Missing or empty patient_id.")
        if not tid:
            add_issue(issues, "ERROR", fname, rid, "trial_id", "Missing or empty trial_id.")
        if not label:
            add_issue(issues, "ERROR", fname, rid, "label", "Missing label.")
        elif label not in VALID_LABELS:
            add_issue(issues, "ERROR", fname, rid, "label",
                      f"Invalid label '{label}'. Must be one of {sorted(VALID_LABELS)}.")

        if pid and tid:
            pair = (pid, tid)
            if pair in seen_pairs:
                add_issue(issues, "ERROR", fname, rid, "pair", "Duplicate patient_id + trial_id pair.")
            seen_pairs.add(pair)

        if not rec.get("rationale"):
            add_issue(issues, "WARN", fname, rid, "rationale", "Optional field 'rationale' missing.")
        if not rec.get("evidence"):
            add_issue(issues, "WARN", fname, rid, "evidence", "Optional field 'evidence' missing.")

    return issues


def validate_results(data: Any) -> list:
    issues: list = []
    fname = "results_llm_reviewed.json"

    if not isinstance(data, dict):
        add_issue(issues, "ERROR", fname, "", "root", "Must be a JSON object.")
        return issues

    if "predictions" not in data:
        add_issue(issues, "ERROR", fname, "", "predictions", "Missing 'predictions' key.")
        return issues

    if not isinstance(data["predictions"], list):
        add_issue(issues, "ERROR", fname, "", "predictions", "'predictions' must be a list.")
        return issues

    if "metrics" in data and not isinstance(data["metrics"], dict):
        add_issue(issues, "ERROR", fname, "", "metrics", "'metrics' must be a dict if present.")

    seen_pairs: set = set()
    for i, rec in enumerate(data["predictions"]):
        rid = f"index {i}"
        if not isinstance(rec, dict):
            add_issue(issues, "ERROR", fname, rid, "root", "Each prediction must be a dict.")
            continue

        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        rid = f"{pid}_{tid}" if pid and tid else rid

        if not pid:
            add_issue(issues, "ERROR", fname, rid, "patient_id", "Missing patient_id.")
        if not tid:
            add_issue(issues, "ERROR", fname, rid, "trial_id", "Missing trial_id.")

        for label_field in ("gold_label", "predicted_label"):
            val = rec.get(label_field, "")
            if not val:
                add_issue(issues, "ERROR", fname, rid, label_field, f"Missing '{label_field}'.")
            elif val not in VALID_LABELS:
                add_issue(issues, "ERROR", fname, rid, label_field,
                          f"Invalid value '{val}' for '{label_field}'.")

        conf = rec.get("confidence")
        if conf is not None:
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                add_issue(issues, "ERROR", fname, rid, "confidence",
                          f"'confidence' must be a number between 0 and 1, got {conf!r}.")

        if pid and tid:
            pair = (pid, tid)
            if pair in seen_pairs:
                add_issue(issues, "ERROR", fname, rid, "pair", "Duplicate patient_id + trial_id pair.")
            seen_pairs.add(pair)

    return issues


def validate_error_analysis(data: Any) -> list:
    issues: list = []
    fname = "error_analysis_llm_reviewed.json"

    if isinstance(data, dict):
        records = data.get("errors", [data])
        if not isinstance(records, list):
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        add_issue(issues, "ERROR", fname, "", "root", "Must be a JSON array or object.")
        return issues

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            add_issue(issues, "WARN", fname, f"index {i}", "root",
                      "Expected a dict record; skipping.")
            continue

        pid = rec.get("patient_id", "")
        tid = rec.get("trial_id", "")
        rid = f"{pid}_{tid}" if pid and tid else f"index {i}"

        if not rec.get("severity"):
            add_issue(issues, "WARN", fname, rid, "severity",
                      "Optional field 'severity' missing.")
        if not rec.get("error_type"):
            add_issue(issues, "WARN", fname, rid, "error_type",
                      "Optional field 'error_type' missing.")

    return issues


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------

def build_indexes(
    patients: Any,
    trials: Any,
    labels: Any,
    results: Any,
) -> dict:
    patient_ids: set = set()
    trial_ids: set = set()
    label_pairs: set = set()

    if isinstance(patients, list):
        for rec in patients:
            if isinstance(rec, dict) and rec.get("patient_id"):
                patient_ids.add(rec["patient_id"])

    if isinstance(trials, list):
        for rec in trials:
            if isinstance(rec, dict) and rec.get("trial_id"):
                trial_ids.add(rec["trial_id"])

    if isinstance(labels, list):
        for rec in labels:
            if isinstance(rec, dict):
                pid = rec.get("patient_id", "")
                tid = rec.get("trial_id", "")
                if pid and tid:
                    label_pairs.add((pid, tid))

    prediction_pairs: set = set()
    if isinstance(results, dict):
        for rec in results.get("predictions", []):
            if isinstance(rec, dict):
                pid = rec.get("patient_id", "")
                tid = rec.get("trial_id", "")
                if pid and tid:
                    prediction_pairs.add((pid, tid))

    return {
        "patient_ids": patient_ids,
        "trial_ids": trial_ids,
        "label_pairs": label_pairs,
        "prediction_pairs": prediction_pairs,
    }


def validate_cross_file_consistency(
    indexes: dict,
    labels: Any,
    results: Any,
) -> list:
    issues: list = []
    patient_ids = indexes["patient_ids"]
    trial_ids = indexes["trial_ids"]
    label_pairs = indexes["label_pairs"]

    if isinstance(labels, list):
        for rec in labels:
            if not isinstance(rec, dict):
                continue
            pid = rec.get("patient_id", "")
            tid = rec.get("trial_id", "")
            rid = f"{pid}_{tid}"
            if pid and pid not in patient_ids:
                add_issue(issues, "ERROR", "labels_llm_reviewed.json", rid,
                          "patient_id", f"patient_id '{pid}' not found in patient_cases.json.")
            if tid and tid not in trial_ids:
                add_issue(issues, "ERROR", "labels_llm_reviewed.json", rid,
                          "trial_id", f"trial_id '{tid}' not found in trial_cases.json.")

    if isinstance(results, dict):
        for rec in results.get("predictions", []):
            if not isinstance(rec, dict):
                continue
            pid = rec.get("patient_id", "")
            tid = rec.get("trial_id", "")
            rid = f"{pid}_{tid}"
            if pid and pid not in patient_ids:
                add_issue(issues, "ERROR", "results_llm_reviewed.json", rid,
                          "patient_id", f"patient_id '{pid}' not found in patient_cases.json.")
            if tid and tid not in trial_ids:
                add_issue(issues, "ERROR", "results_llm_reviewed.json", rid,
                          "trial_id", f"trial_id '{tid}' not found in trial_cases.json.")
            if pid and tid and (pid, tid) not in label_pairs:
                add_issue(issues, "WARN", "results_llm_reviewed.json", rid,
                          "pair", f"Prediction pair ({pid}, {tid}) not found in labels_llm_reviewed.json.")

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(issues: list) -> str:
    errors = [x for x in issues if x["severity"] == "ERROR"]
    warnings = [x for x in issues if x["severity"] == "WARN"]

    lines = ["=" * 60, "SCHEMA VALIDATION REPORT", "=" * 60]

    if errors:
        lines.append(f"\nERRORS ({len(errors)}):")
        for iss in errors:
            lines.append(
                f"  [ERROR] {iss['file']} | {iss['record_id']} | {iss['field']}: {iss['message']}"
            )

    if warnings:
        lines.append(f"\nWARNINGS ({len(warnings)}):")
        for iss in warnings:
            lines.append(
                f"  [WARN]  {iss['file']} | {iss['record_id']} | {iss['field']}: {iss['message']}"
            )

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total errors:   {len(errors)}")
    lines.append(f"Total warnings: {len(warnings)}")
    if not errors:
        lines.append("Result: PASS (no errors)")
    else:
        lines.append("Result: FAIL")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data: dict = {}
    load_errors = False

    for key, path in FILES.items():
        try:
            data[key] = load_json(path)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {path}")
            load_errors = True
            data[key] = None
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Invalid JSON in {path}: {exc}")
            load_errors = True
            data[key] = None

    if load_errors:
        sys.exit(1)

    all_issues: list = []

    all_issues += validate_patient_cases(data["patient_cases"])
    all_issues += validate_trial_cases(data["trial_cases"])
    all_issues += validate_labels(data["labels"])
    all_issues += validate_results(data["results"])
    all_issues += validate_error_analysis(data["error_analysis"])

    indexes = build_indexes(
        data["patient_cases"],
        data["trial_cases"],
        data["labels"],
        data["results"],
    )
    all_issues += validate_cross_file_consistency(
        indexes, data["labels"], data["results"]
    )

    print(format_report(all_issues))

    errors = [x for x in all_issues if x["severity"] == "ERROR"]
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
