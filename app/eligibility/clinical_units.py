"""
clinical_units.py — Unit and duration conversion helpers for clinical trial matching.

This module holds measurement-related parsing and conversion utilities extracted
from rule_matcher.py. Currently covers medication stability duration arithmetic.
Future unit conversion work (weight, BMI, creatinine, hemoglobin) should be added here.
"""

from app.eligibility.clinical_terms import (
    _STABILITY_CRITERION_PATTERN,
    _PATIENT_STABLE_DURATION_PATTERN,
    _PATIENT_CHANGED_PATTERN,
)


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def _to_weeks(amount: int, unit: str) -> int:
    """Convert a duration amount+unit to whole weeks (months = 4 weeks)."""
    return amount * 4 if unit.lower().startswith("month") else amount


def _required_weeks(criterion: str) -> int | None:
    """Return the required stability duration in weeks, or None if not specified."""
    m = _STABILITY_CRITERION_PATTERN.search(criterion)
    if not m:
        return None
    return _to_weeks(int(m.group(1)), m.group(2))


def _patient_stable_weeks(patient_med_text: str) -> int | None:
    """Return how many weeks the patient's medication has been stable, or None."""
    m = _PATIENT_STABLE_DURATION_PATTERN.search(patient_med_text)
    if not m:
        return None
    if m.group(1) is not None:
        return _to_weeks(int(m.group(1)), m.group(2))
    return _to_weeks(int(m.group(3)), m.group(4))


def _patient_changed_weeks_ago(patient_med_text: str) -> int | None:
    """Return how many weeks ago the medication was changed, or None."""
    m = _PATIENT_CHANGED_PATTERN.search(patient_med_text)
    if not m:
        return None
    if m.group(1) is not None:
        return _to_weeks(int(m.group(1)), m.group(2))
    return _to_weeks(int(m.group(3)), m.group(4))
