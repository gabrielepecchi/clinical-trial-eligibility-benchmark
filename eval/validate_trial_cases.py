"""Validate data/processed/trial_cases.json for structural quality.

Usage:
    PYTHONPATH=. python eval/validate_trial_cases.py
    PYTHONPATH=. python eval/validate_trial_cases.py --path data/processed/trial_cases.json
"""

import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/processed/trial_cases.json")

# Criteria text shorter than this (characters) is flagged as suspiciously short.
_MIN_CRITERIA_LENGTH = 20

# Keywords that suggest criteria contain inclusion/exclusion structure.
_ELIGIBILITY_KEYWORDS = {
    "inclusion", "exclusion", "eligible", "criteria",
    "must", "should", "required", "age", "diagnosis",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_trial_cases(path: Path) -> list[dict]:
    """Load and return a list of trial case dicts from *path*."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


# ── Criteria text helper ──────────────────────────────────────────────────────

_CRITERIA_FIELDS = [
    "criteria_text", "eligibility_criteria",
    "inclusion_criteria", "exclusion_criteria",
    "criteria", "inclusion", "exclusion",
]


def _collect_strings(obj: object) -> list[str]:
    """Recursively collect all non-empty string leaf values from a nested structure."""
    results: list[str] = []
    if isinstance(obj, str):
        if obj.strip():
            results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_strings(item))
    return results


def _extract_criteria_text(case: dict) -> tuple[str | None, bool]:
    """Return (combined_criteria_text, has_any_criteria_field).

    has_any_criteria_field is True if at least one recognised criteria field is
    present in the case, regardless of whether it contains text.
    """
    has_field = any(f in case for f in _CRITERIA_FIELDS)
    parts: list[str] = []
    for field in _CRITERIA_FIELDS:
        if field in case:
            parts.extend(_collect_strings(case[field]))
    combined = " ".join(parts).strip() or None
    return combined, has_field


# ── Single-case validator ─────────────────────────────────────────────────────

def validate_trial_case(case: dict) -> list[dict]:
    """Validate a single trial case dict.

    Returns a list of issue dicts. Each issue has:
        trial_id, severity, field, message
    An empty list means no issues found.
    """
    issues: list[dict] = []
    tid = case.get("trial_id", "")

    def issue(severity: str, field: str, message: str) -> dict:
        return {"trial_id": tid, "severity": severity, "field": field, "message": message}

    # ── Required fields ───────────────────────────────────────────────────────

    # trial_id
    if not isinstance(tid, str) or not tid.strip():
        issues.append(issue("error", "trial_id", "trial_id must be a non-empty string"))

    # criteria text — accept any recognised criteria field
    criteria_text, has_criteria_field = _extract_criteria_text(case)

    if not has_criteria_field:
        issues.append(issue(
            "error", "criteria_text",
            "no criteria field found (checked: " + ", ".join(_CRITERIA_FIELDS) + ")",
        ))
    elif not criteria_text:
        issues.append(issue(
            "error", "criteria_text",
            "criteria fields are present but contain no non-empty text",
        ))
    else:
        # ── Structural checks on criteria content ─────────────────────────────

        stripped = criteria_text.strip()

        # Extremely short criteria
        if len(stripped) < _MIN_CRITERIA_LENGTH:
            issues.append(issue(
                "warning", "criteria_text",
                f"criteria text is suspiciously short ({len(stripped)} characters)",
            ))

        # No eligibility-related language
        lower = stripped.lower()
        if not any(kw in lower for kw in _ELIGIBILITY_KEYWORDS):
            issues.append(issue(
                "warning", "criteria_text",
                "criteria text contains no recognisable inclusion/exclusion language",
            ))

    return issues


# ── Multi-case validator ──────────────────────────────────────────────────────

def validate_trial_cases(cases: list[dict]) -> list[dict]:
    """Validate a list of trial case dicts and return all issues.

    Also checks for duplicate trial_id values across the full list.
    """
    all_issues: list[dict] = []
    for case in cases:
        all_issues.extend(validate_trial_case(case))

    # Duplicate trial_id check
    seen: dict[str, int] = {}
    for case in cases:
        tid = case.get("trial_id", "")
        if not isinstance(tid, str) or not tid.strip():
            continue
        seen[tid] = seen.get(tid, 0) + 1

    for tid, count in seen.items():
        if count > 1:
            all_issues.append({
                "trial_id": tid,
                "severity": "error",
                "field": "trial_id",
                "message": f"duplicate trial_id '{tid}' appears {count} times",
            })

    return all_issues


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate trial cases JSON.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to trial_cases.json (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_trial_cases(args.path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    issues = validate_trial_cases(cases)

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    print(f"Validated {len(cases)} trial cases.")
    print(f"  errors   : {len(errors)}")
    print(f"  warnings : {len(warnings)}")

    if issues:
        print()
        for iss in issues:
            prefix = "ERROR  " if iss["severity"] == "error" else "WARNING"
            print(f"  [{prefix}] trial={iss['trial_id']} field={iss['field']}: {iss['message']}")

    if errors:
        print(f"\nValidation failed: {len(errors)} error(s) found.")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
