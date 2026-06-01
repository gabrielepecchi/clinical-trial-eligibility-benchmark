"""
tests/test_unit_conversion.py — Task 6: Unit conversion regression tests.

Tests unit-aware numeric thresholds for weight, BMI, creatinine, hemoglobin.
Passing tests assert stable current matcher behavior.
Xfail tests document desired future behavior not yet implemented.
"""

import pytest
from app.eligibility.rule_matcher import match_patient_to_trial


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trial(criteria_text: str) -> dict:
    return {
        "trial_id": "T_UNIT_TEST",
        "eligibility_criteria": criteria_text,
    }


def _patient(**fields) -> dict:
    base = {
        "patient_id": "P_UNIT_TEST",
        "condition": "Parkinson disease",
        "age": 60,
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# Weight — kg (native units)
# ---------------------------------------------------------------------------

class TestWeightKg:

    def test_weight_within_range_kg_returns_not_blocking(self):
        """Patient weight 70 kg is within 45–100 kg; criterion should not block."""
        patient = _patient(weight_kg=70)
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear"), (
            "A patient within the weight range should not be not_eligible solely on weight."
        )

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce weight_kg lower bound yet; returns eligible.")
    def test_weight_below_range_kg(self):
        """Patient weight 40 kg is below 45 kg minimum; expect not_eligible or unclear."""
        patient = _patient(weight_kg=40)
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        # Current matcher may return unclear if it cannot evaluate; not_eligible also acceptable.
        assert result["prediction"] in ("not_eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce weight_kg upper bound yet; returns eligible.")
    def test_weight_above_range_kg(self):
        """Patient weight 110 kg exceeds 100 kg maximum; expect not_eligible or unclear."""
        patient = _patient(weight_kg=110)
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("not_eligible", "unclear")

    def test_weight_missing_returns_unclear_or_unknown(self):
        """No weight field in patient; criterion cannot be evaluated; expect unclear."""
        patient = _patient()  # no weight field
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        # Matcher marks unknown criteria as uncertain → overall unclear or eligible
        assert result["prediction"] in ("unclear", "eligible")


# ---------------------------------------------------------------------------
# Weight — lb → kg conversion (future behavior)
# ---------------------------------------------------------------------------

class TestWeightLbConversion:

    @pytest.mark.xfail(strict=False, reason="Matcher does not convert lb to kg yet.")
    def test_weight_154_lb_within_45_100_kg(self):
        """154 lb ≈ 70 kg; should be within 45–100 kg range after conversion."""
        patient = _patient(weight_lb=154)
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not convert lb to kg yet.")
    def test_weight_88_lb_below_45_kg(self):
        """88 lb ≈ 40 kg; below 45 kg minimum."""
        patient = _patient(weight_lb=88)
        trial = _trial("Body weight must be between 45 and 100 kg.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] == "not_eligible"


# ---------------------------------------------------------------------------
# BMI
# ---------------------------------------------------------------------------

class TestBMI:

    def test_bmi_within_range(self):
        """BMI 24 is within 18–32; criterion should not block."""
        patient = _patient(bmi=24)
        trial = _trial("BMI must be between 18 and 32 kg/m2.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce BMI lower bound yet; returns eligible.")
    def test_bmi_below_range(self):
        """BMI 16 is below minimum of 18; expect not_eligible or unclear."""
        patient = _patient(bmi=16)
        trial = _trial("BMI must be between 18 and 32 kg/m2.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("not_eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce BMI upper bound yet; returns eligible.")
    def test_bmi_above_range(self):
        """BMI 35 exceeds maximum of 32; expect not_eligible or unclear."""
        patient = _patient(bmi=35)
        trial = _trial("BMI must be between 18 and 32 kg/m2.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("not_eligible", "unclear")

    def test_bmi_missing_returns_unclear_or_eligible(self):
        """No BMI field; criterion cannot be evaluated."""
        patient = _patient()
        trial = _trial("BMI must be between 18 and 32 kg/m2.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("unclear", "eligible")

    @pytest.mark.xfail(strict=False, reason="Matcher does not compute BMI from height/weight yet.")
    def test_bmi_derived_from_height_weight(self):
        """BMI derivable from height 170 cm / weight 70 kg ≈ 24.2; within 18–32."""
        patient = _patient(height_cm=170, weight_kg=70)
        trial = _trial("BMI must be between 18 and 32 kg/m2.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")


# ---------------------------------------------------------------------------
# Creatinine / renal function
# ---------------------------------------------------------------------------

class TestCreatinine:

    def test_creatinine_within_normal_range(self):
        """Creatinine 0.9 mg/dL is within acceptable range (<1.5); should not block."""
        patient = _patient(creatinine_mg_dl=0.9)
        trial = _trial("Serum creatinine must be less than 1.5 mg/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce creatinine_mg_dl upper bound yet; returns eligible.")
    def test_creatinine_above_threshold(self):
        """Creatinine 2.0 mg/dL exceeds 1.5 mg/dL threshold; expect not_eligible or unclear."""
        patient = _patient(creatinine_mg_dl=2.0)
        trial = _trial("Serum creatinine must be less than 1.5 mg/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("not_eligible", "unclear")

    def test_creatinine_missing_returns_unclear_or_eligible(self):
        """No creatinine field; cannot evaluate lab criterion."""
        patient = _patient()
        trial = _trial("Serum creatinine must be less than 1.5 mg/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("unclear", "eligible")

    @pytest.mark.xfail(strict=False, reason="Matcher does not convert creatinine clearance to eGFR yet.")
    def test_creatinine_clearance_above_minimum(self):
        """Creatinine clearance 75 mL/min exceeds required >60 mL/min."""
        patient = _patient(creatinine_clearance_ml_min=75)
        trial = _trial("Creatinine clearance must be greater than 60 mL/min.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not convert creatinine clearance to eGFR yet.")
    def test_creatinine_clearance_below_minimum(self):
        """Creatinine clearance 45 mL/min is below required >60 mL/min."""
        patient = _patient(creatinine_clearance_ml_min=45)
        trial = _trial("Creatinine clearance must be greater than 60 mL/min.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] == "not_eligible"


# ---------------------------------------------------------------------------
# Hemoglobin
# ---------------------------------------------------------------------------

class TestHemoglobin:

    def test_hemoglobin_above_threshold(self):
        """Hemoglobin 12.5 g/dL exceeds minimum of 10 g/dL; should not block."""
        patient = _patient(hemoglobin_g_dl=12.5)
        trial = _trial("Hemoglobin must be greater than 10 g/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")

    @pytest.mark.xfail(strict=False, reason="Matcher does not enforce hemoglobin_g_dl lower bound yet; returns eligible.")
    def test_hemoglobin_below_threshold(self):
        """Hemoglobin 8.0 g/dL is below minimum of 10 g/dL; expect not_eligible or unclear."""
        patient = _patient(hemoglobin_g_dl=8.0)
        trial = _trial("Hemoglobin must be greater than 10 g/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("not_eligible", "unclear")

    def test_hemoglobin_missing_returns_unclear_or_eligible(self):
        """No hemoglobin field; lab criterion cannot be evaluated."""
        patient = _patient()
        trial = _trial("Hemoglobin must be greater than 10 g/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("unclear", "eligible")

    @pytest.mark.xfail(strict=False, reason="Matcher does not convert g/L to g/dL yet.")
    def test_hemoglobin_unit_conversion_g_per_l(self):
        """Hemoglobin 125 g/L = 12.5 g/dL; should satisfy >10 g/dL criterion after conversion."""
        patient = _patient(hemoglobin_g_l=125)
        trial = _trial("Hemoglobin must be greater than 10 g/dL.")
        result = match_patient_to_trial(patient, trial)
        assert result["prediction"] in ("eligible", "unclear")
