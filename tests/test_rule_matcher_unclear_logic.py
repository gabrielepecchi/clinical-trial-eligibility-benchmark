"""Focused tests for the extended unclear-detection logic in rule_matcher.py."""

import pytest
from app.eligibility.rule_matcher import match_patient_to_trial


# ---------------------------------------------------------------------------
# 1. Unclear medication history + medication-specific trial criteria
# ---------------------------------------------------------------------------

def test_unclear_medication_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 65,
        "key_features": ["dose and frequency unclear"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["stable levodopa regimen for at least 4 weeks"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert any("medication" in c for c in result["uncertain_criteria"])


def test_missing_medication_records_with_levodopa_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 60,
        "key_features": ["no pharmacy records available"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["patients on levodopa for minimum 6 months"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_rotigotine_trial_unclear_medication():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 58,
        "key_features": ["medication details unavailable"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["patients receiving rotigotine patch"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 2. Missing disease severity/duration + FoG or disease-stage trial criteria
# ---------------------------------------------------------------------------

def test_missing_duration_fog_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 70,
        "disease_duration": None,
        "key_features": ["duration unknown"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["freezing of gait present", "disease duration > 3 years"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert any("stage" in c or "severity" in c or "duration" in c for c in result["uncertain_criteria"])


def test_unclear_disease_stage_field_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 68,
        "disease_stage": "unclear",
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Hoehn and Yahr stage 2 to 4"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_missing_updrs_for_updrs_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 63,
        "key_features": ["UPDRS unknown"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["UPDRS motor score >= 20", "motor fluctuations present"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 3. Atypical parkinsonism + generic PD trial returns unclear
# ---------------------------------------------------------------------------

def test_atypical_parkinsonism_generic_pd_trial_returns_unclear():
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 67,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["diagnosis of Parkinson disease"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] in ("unclear", "not_eligible")


def test_suspected_parkinsonism_generic_pd_trial_returns_unclear():
    patient = {
        "diagnosis": "suspected parkinsonism",
        "age": 72,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["PD diagnosis confirmed by neurologist"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_msa_generic_pd_trial_returns_unclear():
    patient = {
        "diagnosis": "multiple system atrophy",
        "age": 65,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease diagnosis"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 4. Atypical parkinsonism + idiopathic/confirmed PD criteria returns not_eligible
# ---------------------------------------------------------------------------

def test_atypical_parkinsonism_idiopathic_pd_required_returns_not_eligible():
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 64,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["idiopathic Parkinson disease per UK Brain Bank criteria"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"
    assert any("idiopathic" in c for c in result["blocking_criteria"])


def test_poor_levodopa_response_confirmed_pd_required_returns_not_eligible():
    patient = {
        "diagnosis": "parkinsonism with poor levodopa response",
        "age": 69,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["confirmed idiopathic Parkinson disease"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


# ---------------------------------------------------------------------------
# 5. Active cancer treatment + safety-sensitive non-oncology trial returns unclear
# ---------------------------------------------------------------------------

def test_active_cancer_safety_sensitive_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 66,
        "key_features": ["active cancer treatment ongoing"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "stable cardiovascular status"],
        "exclusion_criteria": ["no serious comorbidities"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert any("cancer" in c for c in result["uncertain_criteria"])


def test_ongoing_chemotherapy_safety_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 61,
        "key_features": [],
        "medications": ["current chemotherapy"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["no hepatic impairment", "tolerability must be assessed"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 6. Recent interventional trial participation + washout/prior trial criteria
# ---------------------------------------------------------------------------

def test_recent_trial_participation_washout_required_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": ["enrolled in recent interventional trial"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["washout period of 30 days required", "no prior trial participation"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert any("washout" in c or "trial" in c for c in result["uncertain_criteria"])


def test_concurrent_study_participation_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 58,
        "key_features": ["currently enrolled in another study"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["no concurrent interventional study participation"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 7. Frailty/falls + gait/exercise/rehabilitation trial returns unclear
# ---------------------------------------------------------------------------

def test_frailty_gait_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 75,
        "key_features": ["frailty noted", "recurrent falls"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "ability to participate in gait study"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert any("comorbidity" in c or "frail" in c or "protocol" in c for c in result["uncertain_criteria"])


def test_orthostatic_hypotension_rehabilitation_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 71,
        "key_features": ["orthostatic hypotension"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["rehabilitation exercise program", "fall prevention protocol"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_pacemaker_stimulation_trial_returns_unclear():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 68,
        "key_features": ["pacemaker implanted"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["TMS stimulation study", "MRI compatible required"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


# ---------------------------------------------------------------------------
# 8. Clear ordinary PD patient + broad PD trial returns eligible
# ---------------------------------------------------------------------------

def test_clear_pd_patient_broad_trial_returns_eligible():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": ["Hoehn and Yahr stage 2", "stable on levodopa"],
        "medications": ["levodopa 100mg three times daily"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "age 40 to 80 years"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"


def test_clear_pd_patient_no_comorbidities_eligible():
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 55,
        "key_features": [],
        "medications": ["carbidopa-levodopa 25-100mg"],
    }
    trial = {
        "inclusion_criteria": ["diagnosis of Parkinson disease", "age 18 to 75 years"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"
