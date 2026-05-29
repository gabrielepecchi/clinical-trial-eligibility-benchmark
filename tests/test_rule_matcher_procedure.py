"""Procedure-specific tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision


def make_patient(**kwargs) -> dict:
    base = {
        "patient_id": "P_TEST",
        "age": 60,
        "sex": "male",
        "diagnosis": ["Parkinson disease"],
        "key_features": ["Hoehn and Yahr stage 2"],
        "exclusions": [],
        "medications": ["levodopa/carbidopa 100/25 mg three times daily"],
    }
    base.update(kwargs)
    return base


def _excl_trial(criterion: str) -> dict:
    return {"trial_id": "T_EXCL", "inclusion_criteria": [], "exclusion_criteria": [criterion]}


# ---------------------------------------------------------------------------
# Procedure exclusion logic
# ---------------------------------------------------------------------------

# Procedure synonym exclusions

def test_dbs_synonym_history_predicts_not_eligible():
    patient = make_patient(key_features=["history of DBS surgery"])
    result = match_patient_to_trial(patient, _excl_trial("Prior deep brain stimulation"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_full_name_predicts_not_eligible():
    patient = make_patient(key_features=["deep brain stimulation implanted previously"])
    result = match_patient_to_trial(patient, _excl_trial("Prior DBS"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_criterion_level_met():
    patient = make_patient(key_features=["history of DBS surgery"])
    results = match_patient_to_trial_criteria(patient, _excl_trial("Prior deep brain stimulation"))
    assert results[0].decision == CriterionDecision.met


# Negation / absence handling

def test_no_dbs_history_not_not_eligible():
    patient = make_patient(key_features=["no history of DBS"])
    result = match_patient_to_trial(patient, _excl_trial("Prior DBS"))
    assert result["prediction"] != "not_eligible"


def test_no_dbs_history_criterion_level_not_met():
    patient = make_patient(key_features=["no history of DBS"])
    results = match_patient_to_trial_criteria(patient, _excl_trial("Prior DBS"))
    assert results[0].decision == CriterionDecision.not_met
