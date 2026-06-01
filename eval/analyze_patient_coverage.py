"""
Task 30: patient field coverage analysis.

Usage:
    PYTHONPATH=. python eval/analyze_patient_coverage.py
"""

import json
import os
import sys
from typing import Any


PREFERRED_PATH = "data/processed/patient_cases_enriched.json"
FALLBACK_PATH = "data/processed/patient_cases.json"
REPORT_PATH = "reports/patient_coverage_analysis.md"

# Fields to check, each entry is (display_name, [candidate_keys])
COVERAGE_FIELDS: list[tuple[str, list[str]]] = [
    ("patient_id",               ["patient_id"]),
    ("age",                      ["age"]),
    ("sex / gender",             ["sex", "gender"]),
    ("diagnosis",                ["diagnosis"]),
    ("diagnosis_subtype",        ["diagnosis_subtype"]),
    ("disease_duration_years",   ["disease_duration_years"]),
    ("hoehn_yahr_stage",         ["hoehn_yahr_stage"]),
    ("updrs_iii",                ["updrs_iii"]),
    ("dbs_history",              ["dbs_history"]),
    ("cognitive_status",         ["cognitive_status"]),
    ("moca_score",               ["moca_score", "moca"]),
    ("mmse_score",               ["mmse_score", "mmse"]),
    ("medications",              ["medication_summary", "medications"]),
    ("procedure_history",        ["procedure_history"]),
    ("comorbidities",            ["comorbidities"]),
    ("recent_trial_participation", ["recent_trial_participation"]),
    ("narrative_profile",        ["narrative_profile"]),
]

LIST_FIELDS: list[tuple[str, list[str]]] = [
    ("medications",      ["medication_summary", "medications"]),
    ("comorbidities",    ["comorbidities"]),
    ("procedure_history", ["procedure_history"]),
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def choose_input_path(preferred: str, fallback: str) -> str:
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Neither '{preferred}' nor '{fallback}' found."
    )


def load_patient_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in '{path}'.")
    return data


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def value_present(value: Any) -> bool:
    """Return True if value is non-None and non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True  # int, float, bool


def get_field_value(record: dict, field_names: list[str]) -> Any:
    """Return value from first matching key; None if absent."""
    for name in field_names:
        if name in record:
            return record[name]
    return None


def compute_basic_stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    return {
        "min": int(s[0]),
        "max": int(s[-1]),
        "mean": round(sum(s) / n, 1),
        "median": round(median, 1),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_patient_coverage(patients: list[dict]) -> dict:
    total = len(patients)

    coverage: dict[str, dict] = {}
    for display, keys in COVERAGE_FIELDS:
        present_ids: list[str] = []
        missing_ids: list[str] = []
        for p in patients:
            pid = p.get("patient_id", "?")
            val = get_field_value(p, keys)
            if value_present(val):
                present_ids.append(pid)
            else:
                missing_ids.append(pid)
        coverage[display] = {
            "present": len(present_ids),
            "missing": len(missing_ids),
            "missing_examples": missing_ids[:10],
        }

    list_stats: dict[str, dict] = {}
    for display, keys in LIST_FIELDS:
        sizes: list[float] = []
        for p in patients:
            val = get_field_value(p, keys)
            if isinstance(val, list):
                sizes.append(float(len(val)))
        list_stats[display] = compute_basic_stats(sizes)

    return {
        "total": total,
        "coverage": coverage,
        "list_stats": list_stats,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_markdown_report(summary: dict) -> str:
    total = summary["total"]
    lines = [
        "# Patient Field Coverage Analysis",
        "",
        f"**Total patients:** {total}",
        "",
        "---",
        "",
        "### Field Coverage",
        "",
        "| Field | Present | Missing | Coverage % |",
        "| --- | --- | --- | --- |",
    ]

    for field, stats in summary["coverage"].items():
        pct = round(100 * stats["present"] / total, 1) if total else 0
        lines.append(
            f"| {field} | {stats['present']} | {stats['missing']} | {pct}% |"
        )

    lines += ["", "---", "", "### Missing Patient IDs per Field (first 10)", ""]

    for field, stats in summary["coverage"].items():
        if stats["missing_examples"]:
            examples = ", ".join(stats["missing_examples"])
            lines.append(f"**{field}:** {examples}  ")

    lines += ["", "---", "", "### List Field Size Statistics", "",
              "| Field | Min | Max | Mean | Median |",
              "| --- | --- | --- | --- | --- |"]

    for field, s in summary["list_stats"].items():
        lines.append(
            f"| {field} | {s['min']} | {s['max']} | {s['mean']} | {s['median']} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        input_path = choose_input_path(PREFERRED_PATH, FALLBACK_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        patients = load_patient_cases(input_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_patient_coverage(patients)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Input   : {input_path}")
    print(f"Patients: {summary['total']}")
    print(f"Report  : {REPORT_PATH}")


if __name__ == "__main__":
    main()
