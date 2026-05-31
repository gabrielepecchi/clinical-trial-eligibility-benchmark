"""Validate data/processed/patient_cases.json for structural quality.

Usage:
    PYTHONPATH=. python eval/validate_patient_cases.py
    PYTHONPATH=. python eval/validate_patient_cases.py --path data/processed/patient_cases.json
"""

import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/processed/patient_cases.json")

# Keywords that signal a DBS procedure when dbs_history is explicitly False.
_DBS_KEYWORDS = {"dbs", "deep brain stimulation"}

# Keywords that signal a pacemaker when pacemaker is explicitly False.
_PACEMAKER_KEYWORDS = {"pacemaker", "cardiac pacemaker"}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_patient_cases(path: Path) -> list[dict]:
    """Load and return a list of patient case dicts from *path*."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


# ── Contradiction helpers ─────────────────────────────────────────────────────

def _text_contains_any(text: str, keywords: set[str]) -> bool:
    """Return True if *text* (lowercased) contains any of *keywords*."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _collect_text_values(obj: object) -> list[str]:
    """Recursively collect all string leaf values from a nested dict/list."""
    results: list[str] = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_collect_text_values(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_text_values(item))
    return results


def _flag_contradiction(
    case: dict,
    bool_field: str,
    search_fields: list[str],
    keywords: set[str],
    label: str,
) -> dict | None:
    """Return an issue dict if *bool_field* is False but *search_fields* mention *keywords*."""
    raw = case.get(bool_field)
    if raw is not True and raw is not False:
        return None  # field absent or not a plain bool — skip
    if raw is not False:
        return None  # only flag when explicitly False

    pid = case.get("patient_id", "")
    for field_name in search_fields:
        value = case.get(field_name)
        if value is None:
            continue
        texts = _collect_text_values(value)
        for text in texts:
            if _text_contains_any(text, keywords):
                return {
                    "patient_id": pid,
                    "severity": "error",
                    "field": bool_field,
                    "message": (
                        f"{bool_field} is False but {field_name!r} mentions "
                        f"{label} ({text!r:.80})"
                    ),
                }
    return None


# ── Single-case validator ─────────────────────────────────────────────────────

def validate_patient_case(case: dict) -> list[dict]:
    """Validate a single patient case dict.

    Returns a list of issue dicts.  Each issue has:
        patient_id, severity, field, message
    An empty list means no issues found.
    """
    issues: list[dict] = []
    pid = case.get("patient_id", "")

    def issue(severity: str, field: str, message: str) -> dict:
        return {"patient_id": pid, "severity": severity, "field": field, "message": message}

    # ── Required fields ──────────────────────────────────────────────────────

    # patient_id
    if not isinstance(pid, str) or not pid.strip():
        issues.append(issue("error", "patient_id", "patient_id must be a non-empty string"))

    # demographics — accept a demographics dict OR top-level age + sex/gender fields
    demographics = case.get("demographics")
    has_top_level_demographics = (
        "age" in case and ("sex" in case or "gender" in case)
    )
    if "demographics" not in case:
        if not has_top_level_demographics:
            issues.append(issue("error", "demographics",
                                "demographics field is missing and no top-level age/sex fields found"))
    elif not isinstance(demographics, dict):
        issues.append(issue("error", "demographics", "demographics must be a dict"))

    # diagnosis — accept top-level key or nested under demographics/clinical data
    diagnosis = case.get("diagnosis") or (
        demographics.get("diagnosis") if isinstance(demographics, dict) else None
    )
    if not diagnosis:
        issues.append(issue("error", "diagnosis", "diagnosis is missing or empty"))

    # clinical_summary / summary — warn if neither present
    has_summary = "clinical_summary" in case or "summary" in case
    if not has_summary:
        issues.append(issue(
            "warning", "clinical_summary",
            "neither clinical_summary nor summary field is present",
        ))

    # ── Contradiction checks (only when relevant fields exist) ───────────────

    # dbs_history False + procedures/history mentions DBS
    dbs_issue = _flag_contradiction(
        case,
        bool_field="dbs_history",
        search_fields=["procedures", "history", "medical_history", "clinical_summary", "summary"],
        keywords=_DBS_KEYWORDS,
        label="DBS",
    )
    if dbs_issue:
        issues.append(dbs_issue)

    # pacemaker False + devices/history mentions pacemaker
    pm_issue = _flag_contradiction(
        case,
        bool_field="pacemaker",
        search_fields=["devices", "history", "medical_history", "clinical_summary", "summary"],
        keywords=_PACEMAKER_KEYWORDS,
        label="pacemaker",
    )
    if pm_issue:
        issues.append(pm_issue)

    return issues


# ── Multi-case validator ──────────────────────────────────────────────────────

def validate_patient_cases(cases: list[dict]) -> list[dict]:
    """Validate a list of patient case dicts and return all issues."""
    all_issues: list[dict] = []
    for case in cases:
        all_issues.extend(validate_patient_case(case))
    return all_issues


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic patient cases JSON.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to patient_cases.json (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_patient_cases(args.path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    issues = validate_patient_cases(cases)

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    print(f"Validated {len(cases)} patient cases.")
    print(f"  errors   : {len(errors)}")
    print(f"  warnings : {len(warnings)}")

    if issues:
        print()
        for iss in issues:
            prefix = "ERROR  " if iss["severity"] == "error" else "WARNING"
            print(f"  [{prefix}] patient={iss['patient_id']} field={iss['field']}: {iss['message']}")

    if errors:
        print(f"\nValidation failed: {len(errors)} error(s) found.")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
