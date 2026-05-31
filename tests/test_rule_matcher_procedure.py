"""Procedure-specific tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision
from tests.helpers import make_patient, excl_trial


def test_dbs_synonym_history_predicts_not_eligible():
    patient = make_patient(key_features=["history of DBS surgery"])
    result = match_patient_to_trial(patient, excl_trial("Prior deep brain stimulation"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_full_name_predicts_not_eligible():
    patient = make_patient(key_features=["deep brain stimulation implanted previously"])
    result = match_patient_to_trial(patient, excl_trial("Prior DBS"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_criterion_level_met():
    patient = make_patient(key_features=["history of DBS surgery"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Prior deep brain stimulation"))
    assert results[0].decision == CriterionDecision.met


def test_no_dbs_history_not_not_eligible():
    patient = make_patient(key_features=["no history of DBS"])
    result = match_patient_to_trial(patient, excl_trial("Prior DBS"))
    assert result["prediction"] != "not_eligible"


def test_no_dbs_history_criterion_level_not_met():
    patient = make_patient(key_features=["no history of DBS"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Prior DBS"))
    assert results[0].decision == CriterionDecision.not_met
