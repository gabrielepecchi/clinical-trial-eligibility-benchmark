"""Medication-specific tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision
from tests.helpers import make_patient, incl_trial, excl_trial


def test_stable_6_weeks_meets_4_week_requirement():
    patient = make_patient(key_features=["medication regimen stable for 6 weeks"])
    trial = incl_trial("Stable medication regimen for at least 4 weeks")
    assert match_patient_to_trial(patient, trial)["prediction"] == "eligible"


def test_changed_2_weeks_ago_fails_4_week_requirement():
    patient = make_patient(key_features=["medication regimen changed 2 weeks ago"])
    trial = incl_trial("Stable medication regimen for at least 4 weeks")
    assert match_patient_to_trial(patient, trial)["prediction"] in {"not_eligible", "unclear"}


def test_stable_1_month_fails_3_month_requirement():
    patient = make_patient(key_features=["medication regimen stable for 1 month"])
    trial = incl_trial("Stable medication regimen for at least 3 months")
    assert match_patient_to_trial(patient, trial)["prediction"] in {"not_eligible", "unclear"}


def test_stable_6_weeks_criterion_met_for_4_week_requirement():
    patient = make_patient(key_features=["medication regimen stable for 6 weeks"])
    results = match_patient_to_trial_criteria(patient, incl_trial("Stable medication regimen for at least 4 weeks"))
    assert results[0].decision == CriterionDecision.met


def test_stable_1_month_criterion_not_met_or_unknown_for_3_month_requirement():
    patient = make_patient(key_features=["medication regimen stable for 1 month"])
    results = match_patient_to_trial_criteria(patient, incl_trial("Stable medication regimen for at least 3 months"))
    assert results[0].decision in {CriterionDecision.not_met, CriterionDecision.unknown}


def test_maob_rasagiline_predicts_not_eligible():
    patient = make_patient(medications=["rasagiline 1 mg daily"])
    result = match_patient_to_trial(patient, excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_selegiline_predicts_not_eligible():
    patient = make_patient(medications=["selegiline 5 mg daily"])
    result = match_patient_to_trial(patient, excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_safinamide_predicts_not_eligible():
    patient = make_patient(medications=["safinamide 50 mg daily"])
    result = match_patient_to_trial(patient, excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_rasagiline_criterion_level_met():
    patient = make_patient(medications=["rasagiline 1 mg daily"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Current MAO-B inhibitor use"))
    assert results[0].decision == CriterionDecision.met


def test_maob_no_inhibitor_not_not_eligible():
    patient = make_patient(medications=["levodopa/carbidopa 100/25 mg three times daily"])
    result = match_patient_to_trial(patient, excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] != "not_eligible"


def test_no_maob_documented_not_not_eligible():
    patient = make_patient(medications=["no MAO-B inhibitor use documented"])
    result = match_patient_to_trial(patient, excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] != "not_eligible"


def test_no_maob_documented_criterion_level_not_met():
    patient = make_patient(medications=["no MAO-B inhibitor use documented"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Current MAO-B inhibitor use"))
    assert results[0].decision == CriterionDecision.not_met


def test_no_stability_duration_missing_information_includes_key():
    """No stability info documented → missing_information flags medication_stability_duration."""
    patient = make_patient(key_features=["Hoehn and Yahr stage 2"])
    trial = incl_trial("Stable medication regimen for at least 4 weeks")
    result = match_patient_to_trial(patient, trial)
    assert "medication_stability_duration" in result.get("missing_information", [])


def test_stable_1_month_vs_3_month_requirement_missing_information():
    """Patient stable only 1 month but trial requires 3 months → missing_information includes medication_stability_duration."""
    patient = make_patient(key_features=["medication regimen stable for 1 month"])
    trial = incl_trial("Stable medication regimen for at least 3 months")
    result = match_patient_to_trial(patient, trial)
    assert "medication_stability_duration" in result.get("missing_information", [])


def test_stable_6_weeks_meets_4_week_no_missing_information():
    """Patient stable 6 weeks satisfies a 4-week requirement → missing_information does not include medication_stability_duration."""
    patient = make_patient(key_features=["medication regimen stable for 6 weeks"])
    trial = incl_trial("Stable medication regimen for at least 4 weeks")
    result = match_patient_to_trial(patient, trial)
    assert "medication_stability_duration" not in result.get("missing_information", [])
