"""Print dataset coverage report for patient and trial cases.

Usage:
    PYTHONPATH=. python eval/print_coverage_report.py
"""

import json
import os
import sys
from pathlib import Path

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
REPORT_PATH = Path("reports/coverage_report.md")


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


def field_present(record: dict, field_names: list[str]) -> bool:
    """Return True if any of the given field names exist and are non-empty."""
    for name in field_names:
        val = record.get(name)
        if val is not None and val != "" and val != [] and val != {}:
            return True
    return False


def keyword_present(record: dict, keywords: list[str]) -> bool:
    """Return True if any keyword appears in any string value of the record."""
    text = " ".join(
        str(v).lower()
        for v in record.values()
        if isinstance(v, (str, list))
        for v in ([v] if isinstance(v, str) else v)
        if isinstance(v, str)
    )
    return any(kw.lower() in text for kw in keywords)


def compute_patient_coverage(patients: list[dict]) -> list[tuple[str, int, int]]:
    """Return list of (label, count_present, total) for patient fields."""
    total = len(patients)
    checks: list[tuple[str, list[dict]]] = [
        ("age", [p for p in patients if field_present(p, ["age"])]),
        ("sex / gender", [p for p in patients if field_present(p, ["sex", "gender"])]),
        ("diagnosis", [p for p in patients if field_present(p, ["diagnosis", "condition", "conditions"])]),
        ("clinical summary", [p for p in patients if field_present(p, ["clinical_summary", "summary", "narrative", "profile"])]),
        ("medications", [p for p in patients if field_present(p, ["medications", "medication_history", "current_medications", "drugs"])]),
        ("procedures", [p for p in patients if field_present(p, ["procedures", "procedure_history", "surgical_history"])]),
        ("DBS history (field)", [p for p in patients if field_present(p, ["dbs_history", "deep_brain_stimulation", "dbs"])]),
        ("DBS history (keyword)", [p for p in patients if keyword_present(p, ["deep brain stimulation", "dbs"])]),
        ("pacemaker / device (keyword)", [p for p in patients if keyword_present(p, ["pacemaker", "implantable device", "cardiac device"])]),
        ("MoCA / MMSE (keyword)", [p for p in patients if keyword_present(p, ["moca", "mmse", "mini-mental", "montreal cognitive"])]),
        ("UPDRS (keyword)", [p for p in patients if keyword_present(p, ["updrs", "unified parkinson"])]),
    ]
    return [(label, len(matched), total) for label, matched in checks]


def compute_trial_coverage(trials: list[dict]) -> list[tuple[str, int, int]]:
    """Return list of (label, count_present, total) for trial fields."""
    total = len(trials)

    _criteria_fields = [
        "criteria_text", "eligibility_criteria", "inclusion_criteria",
        "exclusion_criteria", "criteria", "inclusion", "exclusion",
        "eligibility", "inclusion_text", "exclusion_text",
    ]

    checks: list[tuple[str, list[dict]]] = [
        ("trial_id", [t for t in trials if field_present(t, ["trial_id", "nct_id", "id"])]),
        ("title / summary", [t for t in trials if field_present(t, ["title", "brief_title", "official_title", "summary", "brief_summary"])]),
        ("criteria text (any field)", [t for t in trials if field_present(t, _criteria_fields)]),
        ("inclusion language (keyword)", [t for t in trials if keyword_present(t, ["inclusion criteria", "must have", "required to have", "patients with"])]),
        ("exclusion language (keyword)", [t for t in trials if keyword_present(t, ["exclusion criteria", "must not", "excluded if", "not eligible"])]),
        ("age criteria (keyword)", [t for t in trials if keyword_present(t, ["age", "years old", "years of age"])]),
        ("diagnosis criteria (keyword)", [t for t in trials if keyword_present(t, ["parkinson", "diagnosis", "idiopathic"])]),
        ("DBS / device exclusion (keyword)", [t for t in trials if keyword_present(t, ["deep brain stimulation", "dbs", "implant", "pacemaker"])]),
        ("medication criteria (keyword)", [t for t in trials if keyword_present(t, ["medication", "drug", "levodopa", "inhibitor", "washout", "treatment"])]),
    ]
    return [(label, len(matched), total) for label, matched in checks]


def format_coverage_table(title: str, rows: list[tuple[str, int, int]]) -> str:
    """Format a coverage table as a terminal string."""
    col = max(len(label) for label, _, _ in rows) + 2
    lines = [
        f"\n=== {title} ===",
        f"  {'field / signal':<{col}}  {'present':>8}  {'total':>6}  {'coverage':>9}",
        "  " + "-" * (col + 28),
    ]
    for label, present, total in rows:
        pct = present / total * 100 if total else 0.0
        lines.append(f"  {label:<{col}}  {present:>8}  {total:>6}  {pct:>8.1f}%")
    return "\n".join(lines)


def format_markdown_coverage_table(title: str, rows: list[tuple[str, int, int]]) -> str:
    """Format a coverage table as Markdown."""
    lines = [
        f"## {title}",
        "",
        "| Field / Signal | Present | Total | Coverage |",
        "| --- | --- | --- | --- |",
    ]
    for label, present, total in rows:
        pct = present / total * 100 if total else 0.0
        lines.append(f"| {label} | {present} | {total} | {pct:.1f}% |")
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(
    patients_file: Path,
    trials_file: Path,
    patient_rows: list[tuple[str, int, int]],
    trial_rows: list[tuple[str, int, int]],
) -> str:
    n_patients = patient_rows[0][2] if patient_rows else 0
    n_trials = trial_rows[0][2] if trial_rows else 0
    parts = [
        "# Coverage Report",
        "",
        f"**Patients file:** `{patients_file}` ({n_patients} records)  ",
        f"**Trials file:** `{trials_file}` ({n_trials} records)",
        "",
        "---",
        "",
        format_markdown_coverage_table("Patient Coverage", patient_rows),
        format_markdown_coverage_table("Trial Coverage", trial_rows),
    ]
    return "\n".join(parts)


def write_report(text: str, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patients = load_json_list(PATIENTS_FILE)
    trials = load_json_list(TRIALS_FILE)

    print(f"\nPatients file : {PATIENTS_FILE}  ({len(patients)} records)")
    print(f"Trials file   : {TRIALS_FILE}  ({len(trials)} records)")

    patient_rows = compute_patient_coverage(patients)
    trial_rows = compute_trial_coverage(trials)

    print(format_coverage_table("Patient Coverage", patient_rows))
    print(format_coverage_table("Trial Coverage", trial_rows))

    report_md = format_markdown_report(PATIENTS_FILE, TRIALS_FILE, patient_rows, trial_rows)
    write_report(report_md, REPORT_PATH)
    print(f"\nReport saved  : {REPORT_PATH}")


if __name__ == "__main__":
    main()
