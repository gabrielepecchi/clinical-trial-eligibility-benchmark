"""
clinical_units.py — Unit and duration conversion helpers for clinical trial matching.

This module holds measurement-related parsing and conversion utilities extracted
from rule_matcher.py. Covers medication stability duration arithmetic and general
clinical unit conversion (weight, hemoglobin, creatinine, BMI).
"""

import re
from typing import Optional

from app.eligibility.clinical_terms import (
    _STABILITY_CRITERION_PATTERN,
    _PATIENT_STABLE_DURATION_PATTERN,
    _PATIENT_CHANGED_PATTERN,
)


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

_LB_TO_KG = 0.453592
_KG_TO_LB = 2.20462
_G_PER_L_TO_G_PER_DL = 0.1      # 1 g/L = 0.1 g/dL
_G_PER_DL_TO_G_PER_L = 10.0
_CM_TO_M = 0.01
_M_TO_CM = 100.0


# ---------------------------------------------------------------------------
# Weight conversions
# ---------------------------------------------------------------------------

def lb_to_kg(lb: float) -> float:
    return lb * _LB_TO_KG


def kg_to_lb(kg: float) -> float:
    return kg * _KG_TO_LB


def get_weight_kg(patient: dict) -> Optional[float]:
    """Return patient weight in kg, converting from lb if needed."""
    if "weight_kg" in patient and patient["weight_kg"] is not None:
        return float(patient["weight_kg"])
    for field in ("weight_lb", "weight_lbs", "weight_pounds"):
        if field in patient and patient[field] is not None:
            return lb_to_kg(float(patient[field]))
    return None


# ---------------------------------------------------------------------------
# Height / BMI
# ---------------------------------------------------------------------------

def get_height_m(patient: dict) -> Optional[float]:
    """Return patient height in metres."""
    if "height_m" in patient and patient["height_m"] is not None:
        return float(patient["height_m"])
    if "height_cm" in patient and patient["height_cm"] is not None:
        return float(patient["height_cm"]) * _CM_TO_M
    return None


def compute_bmi(patient: dict) -> Optional[float]:
    """Compute BMI from weight_kg / height fields if both available."""
    weight_kg = get_weight_kg(patient)
    height_m = get_height_m(patient)
    if weight_kg is None or height_m is None or height_m == 0:
        return None
    return weight_kg / (height_m ** 2)


def get_bmi(patient: dict) -> Optional[float]:
    """Return explicit BMI or derive it from height/weight."""
    if "bmi" in patient and patient["bmi"] is not None:
        return float(patient["bmi"])
    return compute_bmi(patient)


# ---------------------------------------------------------------------------
# Hemoglobin
# ---------------------------------------------------------------------------

def get_hemoglobin_g_dl(patient: dict) -> Optional[float]:
    """Return hemoglobin in g/dL, converting from g/L if needed."""
    if "hemoglobin_g_dl" in patient and patient["hemoglobin_g_dl"] is not None:
        return float(patient["hemoglobin_g_dl"])
    if "hemoglobin_g_l" in patient and patient["hemoglobin_g_l"] is not None:
        return float(patient["hemoglobin_g_l"]) * _G_PER_L_TO_G_PER_DL
    return None


# ---------------------------------------------------------------------------
# Creatinine / renal
# ---------------------------------------------------------------------------

def get_creatinine_mg_dl(patient: dict) -> Optional[float]:
    """Return serum creatinine in mg/dL if available."""
    if "creatinine_mg_dl" in patient and patient["creatinine_mg_dl"] is not None:
        return float(patient["creatinine_mg_dl"])
    return None


def get_creatinine_clearance(patient: dict) -> Optional[float]:
    """Return creatinine clearance in mL/min if available."""
    for field in ("creatinine_clearance_ml_min", "creatinine_clearance", "egfr"):
        if field in patient and patient[field] is not None:
            return float(patient[field])
    return None


# ---------------------------------------------------------------------------
# Threshold parsing helpers
# ---------------------------------------------------------------------------

_BETWEEN_PATTERN = re.compile(
    r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)
_LESS_THAN_PATTERN = re.compile(
    r"(?:less\s+than|<|below|under)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)
_GREATER_THAN_PATTERN = re.compile(
    r"(?:greater\s+than|>|above|over|more\s+than|exceed[s]?)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_AT_LEAST_PATTERN = re.compile(
    r"(?:at\s+least|>=|≥)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)
_AT_MOST_PATTERN = re.compile(
    r"(?:at\s+most|no\s+more\s+than|<=|≤)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)


def parse_numeric_range(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (min_val, max_val) from a criterion text; None where not specified."""
    text_lower = text.lower()
    m = _BETWEEN_PATTERN.search(text_lower)
    if m:
        return float(m.group(1)), float(m.group(2))

    lo: Optional[float] = None
    hi: Optional[float] = None

    m = _LESS_THAN_PATTERN.search(text_lower)
    if m:
        hi = float(m.group(1))

    m = _AT_MOST_PATTERN.search(text_lower)
    if m:
        hi = float(m.group(1))

    m = _GREATER_THAN_PATTERN.search(text_lower)
    if m:
        lo = float(m.group(1))

    m = _AT_LEAST_PATTERN.search(text_lower)
    if m:
        lo = float(m.group(1))

    return lo, hi


# ---------------------------------------------------------------------------
# Lab/measurement threshold check helpers
# ---------------------------------------------------------------------------

def check_weight_criterion(
    patient: dict, criterion_text: str
) -> Optional[tuple[str, str]]:
    """Return (blocking_criterion, matched_fact) if weight is out of range, else None."""
    text = criterion_text.lower()
    if "bmi" in text or "kg/m" in text:
        return None
    if "kg" not in text and "weight" not in text:
        return None
    weight_kg = get_weight_kg(patient)
    if weight_kg is None:
        return None
    lo, hi = parse_numeric_range(text)
    if lo is None and hi is None:
        return None
    too_low = lo is not None and weight_kg < lo
    too_high = hi is not None and weight_kg > hi
    if too_low or too_high:
        return (
            f"weight criterion not met (patient {weight_kg:.1f} kg)",
            f"patient weight {weight_kg:.1f} kg out of required range",
        )
    return None


def check_bmi_criterion(
    patient: dict, criterion_text: str
) -> Optional[tuple[str, str]]:
    """Return (blocking_criterion, matched_fact) if BMI is out of range, else None."""
    text = criterion_text.lower()
    if "bmi" not in text:
        return None
    bmi = get_bmi(patient)
    if bmi is None:
        return None
    lo, hi = parse_numeric_range(text)
    if lo is None and hi is None:
        return None
    too_low = lo is not None and bmi < lo
    too_high = hi is not None and bmi > hi
    if too_low or too_high:
        return (
            f"BMI criterion not met (patient BMI {bmi:.1f})",
            f"patient BMI {bmi:.1f} out of required range",
        )
    return None


def check_creatinine_criterion(
    patient: dict, criterion_text: str
) -> Optional[tuple[str, str]]:
    """Return (blocking_criterion, matched_fact) if creatinine is out of range, else None."""
    text = criterion_text.lower()
    if "creatinine" not in text:
        return None
    # Creatinine clearance criterion
    if "clearance" in text or "ml/min" in text or "egfr" in text:
        cc = get_creatinine_clearance(patient)
        if cc is None:
            return None
        lo, hi = parse_numeric_range(text)
        if lo is not None and cc < lo:
            return (
                f"creatinine clearance below required minimum (patient {cc:.1f} mL/min)",
                f"creatinine clearance {cc:.1f} mL/min below threshold",
            )
        if hi is not None and cc > hi:
            return (
                f"creatinine clearance above required maximum (patient {cc:.1f} mL/min)",
                f"creatinine clearance {cc:.1f} mL/min above threshold",
            )
        return None
    # Serum creatinine
    cr = get_creatinine_mg_dl(patient)
    if cr is None:
        return None
    lo, hi = parse_numeric_range(text)
    if hi is not None and cr >= hi:
        return (
            f"serum creatinine above required threshold (patient {cr:.2f} mg/dL)",
            f"creatinine {cr:.2f} mg/dL at or above limit {hi}",
        )
    if lo is not None and cr < lo:
        return (
            f"serum creatinine below required minimum (patient {cr:.2f} mg/dL)",
            f"creatinine {cr:.2f} mg/dL below minimum {lo}",
        )
    return None


def check_hemoglobin_criterion(
    patient: dict, criterion_text: str
) -> Optional[tuple[str, str]]:
    """Return (blocking_criterion, matched_fact) if hemoglobin is out of range, else None."""
    text = criterion_text.lower()
    if "hemoglobin" not in text and "haemoglobin" not in text:
        return None
    hgb = get_hemoglobin_g_dl(patient)
    if hgb is None:
        return None
    lo, hi = parse_numeric_range(text)
    if lo is not None and hgb <= lo:
        return (
            f"hemoglobin below required minimum (patient {hgb:.1f} g/dL)",
            f"hemoglobin {hgb:.1f} g/dL at or below threshold {lo}",
        )
    if hi is not None and hgb >= hi:
        return (
            f"hemoglobin above required maximum (patient {hgb:.1f} g/dL)",
            f"hemoglobin {hgb:.1f} g/dL at or above threshold {hi}",
        )
    return None


def check_lab_thresholds(
    patient: dict, criteria_list: list[str]
) -> list[tuple[str, str]]:
    """Return list of (blocking_criterion, matched_fact) for all lab/measurement violations."""
    blocks: list[tuple[str, str]] = []
    for criterion in criteria_list:
        for checker in (
            check_weight_criterion,
            check_bmi_criterion,
            check_creatinine_criterion,
            check_hemoglobin_criterion,
        ):
            result = checker(patient, criterion)
            if result is not None:
                blocks.append(result)
    return blocks


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
