"""Medication-related eligibility rule helpers."""

import re

from app.eligibility.clinical_terms import (
    _any_match,
    has_contradiction,
    _MAOB_CRITERION_PATTERN,
    _MAOB_DRUGS,
    _has_maob_inhibitor,
    _patient_has_med_class,
    _STABLE_MED_PATTERNS,
    _UNCLEAR_MED_PATTERNS,
    _TRIAL_MED_SPECIFIC_PATTERNS,
    _PATIENT_UNCLEAR_MED_PATTERNS,
)

from app.eligibility.clinical_units import (
    _to_weeks,
    _required_weeks,
    _patient_stable_weeks,
    _patient_changed_weeks_ago,
)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


_STABLE_REGIMEN_DURATION_PATTERN = re.compile(
    r"stable\s+\w+(?:\s+\w+)?\s+regimen\s+for\s+at\s+least\s+(\d+)\s+(weeks?|months?)",
    re.IGNORECASE,
)


def _required_weeks_extended(criterion: str) -> int | None:
    """Like _required_weeks but also matches 'stable <drug> regimen for at least N weeks'."""
    result = _required_weeks(criterion)
    if result is not None:
        return result
    m = _STABLE_REGIMEN_DURATION_PATTERN.search(criterion)
    if not m:
        return None
    return _to_weeks(int(m.group(1)), m.group(2))


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------

def _collect_patient_med_text(patient: dict) -> str:
    """Return a single lowercased string covering all medication-bearing patient fields."""
    parts = []
    for field in ("medications", "current_medications", "medication_history", "key_features", "exclusions"):
        val = patient.get(field, [])
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
        elif val:
            parts.append(str(val))
    summary = patient.get("summary", "")
    if summary:
        parts.append(str(summary))
    return " ".join(parts).lower()


def _check_maob(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if MAO-B inhibitor exclusion applies."""
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    if not _MAOB_CRITERION_PATTERN.search(exclusion_text):
        if not _any_match(_MAOB_DRUGS, exclusion_text):
            return None, None
    patient_med_text = _collect_patient_med_text(patient)
    if has_contradiction(patient_med_text, "maob_inhibitor"):
        return (
            "__unclear__:contradictory MAO-B inhibitor records: both negation and affirmation found — eligibility cannot be determined",
            "contradiction in MAO-B inhibitor records",
        )
    if _has_maob_inhibitor(patient_med_text) or _patient_has_med_class(patient_med_text, "maob_inhibitor"):
        return "MAO-B inhibitor use is an exclusion criterion", "MAO-B inhibitor medication present"
    return None, None


def _check_medication_stability(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (uncertain_criterion, matched_fact) if medication stability is unclear or insufficient."""
    inclusion_list = trial.get("inclusion_criteria", [])
    inclusion_text = _text(inclusion_list)
    if not _any_match(_STABLE_MED_PATTERNS, inclusion_text):
        return None, None

    patient_med_text = _collect_patient_med_text(patient)
    if _any_match(_UNCLEAR_MED_PATTERNS, patient_med_text):
        return (
            "stable medication regimen required but cannot be confirmed",
            "medication dose, frequency, or compliance unclear",
        )

    # Numeric duration check
    for criterion in inclusion_list:
        req = _required_weeks_extended(criterion)
        if req is None:
            continue
        changed_ago = _patient_changed_weeks_ago(patient_med_text)
        if changed_ago is not None and changed_ago < req:
            return (
                f"stable medication regimen for at least {req} week(s) required; "
                f"medication changed {changed_ago} week(s) ago",
                f"medication changed {changed_ago} week(s) ago (required: {req} weeks stable)",
            )
        patient_weeks = _patient_stable_weeks(patient_med_text)
        if patient_weeks is not None and patient_weeks < req:
            return (
                f"stable medication regimen for at least {req} week(s) required; "
                f"patient stable for only {patient_weeks} week(s)",
                f"medication stable {patient_weeks} week(s) (required: {req} weeks)",
            )
        if patient_weeks is None and changed_ago is None:
            return (
                f"stable medication regimen for at least {req} week(s) required but duration not documented",
                "medication stability duration not documented",
            )

    return None, None


def _check_medication_details_unclear(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if trial requires specific drug details but patient data is unclear."""
    inclusion_text = _text(trial.get("inclusion_criteria", []))
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    trial_text = inclusion_text + " " + exclusion_text

    if not _any_match(_TRIAL_MED_SPECIFIC_PATTERNS, trial_text):
        return None, None

    patient_med_text = _collect_patient_med_text(patient)

    if _any_match(_PATIENT_UNCLEAR_MED_PATTERNS, patient_med_text):
        return (
            "trial requires specific medication details but patient medication data is unclear or missing",
            "medication details unclear or missing",
        )

    return None, None
