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
# 4. Atypical parkinsonism + idiopathic/confirmed PD criteria
# ---------------------------------------------------------------------------

def test_atypical_parkinsonism_idiopathic_pd_treatment_trial_returns_not_eligible():
    """Treatment/intervention trial with idiopathic PD requirement → not_eligible."""
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
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("atypical", "idiopathic", "parkinsonism")
    )


def test_atypical_parkinsonism_idiopathic_pd_neuroprotection_trial_returns_not_eligible():
    """Neuroprotection trial with idiopathic PD inclusion → not_eligible."""
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 64,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["idiopathic Parkinson disease per UK Brain Bank criteria"],
        "exclusion_criteria": [],
        "title": "neuroprotection study in Parkinson disease",
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_atypical_parkinsonism_idiopathic_pd_stimulation_trial_returns_not_eligible():
    """Stimulation trial with idiopathic PD inclusion → not_eligible."""
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 60,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": [
            "idiopathic Parkinson disease",
            "scheduled for DBS surgery",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_atypical_parkinsonism_idiopathic_pd_diagnostic_study_returns_unclear():
    """Differential diagnosis study with idiopathic PD inclusion → unclear."""
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 64,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["idiopathic Parkinson disease or suspected parkinsonism"],
        "exclusion_criteria": [],
        "title": "differential diagnosis study for parkinsonism",
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] in ("unclear", "eligible")


def test_poor_levodopa_response_confirmed_pd_treatment_returns_not_eligible():
    """Treatment trial with confirmed idiopathic PD requirement → not_eligible."""
    patient = {
        "diagnosis": "parkinsonism with poor levodopa response",
        "age": 69,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["confirmed idiopathic Parkinson disease"],
        "exclusion_criteria": [],
        "title": "randomized placebo-controlled treatment trial",
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_atypical_parkinsonism_explicit_atypical_exclusion_returns_not_eligible():
    patient = {
        "diagnosis": "atypical parkinsonism",
        "age": 64,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["idiopathic Parkinson disease diagnosis"],
        "exclusion_criteria": ["atypical parkinsonism or secondary parkinsonism excluded"],
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


def test_pacemaker_stimulation_trial_returns_not_eligible():
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
    assert result["prediction"] == "not_eligible"
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("pacemaker", "cardiac", "contraindication", "transcranial", "stimulation")
    )


# ---------------------------------------------------------------------------
# 8. Healthy control patient
# ---------------------------------------------------------------------------

def test_healthy_control_explicit_control_group_trial_returns_unclear():
    """Healthy control + trial with explicit healthy/control/comparator wording → unclear."""
    patient = {
        "diagnosis": "healthy control",
        "age": 60,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": [
            "Parkinson disease diagnosis",
            "age-matched healthy control group",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_healthy_control_explicit_comparator_group_returns_unclear():
    """Healthy control + comparator group wording → unclear."""
    patient = {
        "diagnosis": "healthy control",
        "age": 55,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": [
            "diagnosis of Parkinson disease",
            "comparator group of healthy volunteers",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_healthy_control_pd_stimulation_trial_no_explicit_controls_returns_not_eligible():
    """Healthy control + PD stimulation trial with no explicit control group → not_eligible."""
    patient = {
        "diagnosis": "healthy control",
        "age": 62,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease diagnosis", "scheduled for DBS stimulation"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_healthy_control_pd_treadmill_trial_no_explicit_controls_returns_not_eligible():
    """Healthy control + PD treadmill/training trial without control group wording → not_eligible."""
    patient = {
        "diagnosis": "healthy control",
        "age": 58,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "treadmill training intervention"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


# ---------------------------------------------------------------------------
# 9. DBS: no DBS patient + DBS candidacy/evaluation wording → unclear
# ---------------------------------------------------------------------------

def test_no_dbs_dbs_candidacy_trial_returns_unclear():
    """Patient without DBS + DBS candidacy trial → unclear, not not_eligible."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 60,
        "key_features": ["no history of DBS"],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "meets criteria for DBS candidacy"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_no_dbs_scheduled_to_undergo_dbs_returns_unclear():
    """Patient without DBS + scheduled-to-undergo DBS wording → unclear."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 63,
        "key_features": [],
        "medications": ["levodopa", "pramipexole"],
    }
    trial = {
        "inclusion_criteria": [
            "Parkinson disease",
            "scheduled to undergo DBS surgery",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_no_dbs_lfp_sensing_directional_leads_returns_not_eligible():
    """Patient without DBS + LFP sensing from directional leads (existing hardware required) → not_eligible."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 65,
        "key_features": ["no DBS implant"],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": [
            "Parkinson disease",
            "LFP sensing from directional lead hardware required",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_no_dbs_existing_dbs_hardware_required_returns_not_eligible():
    """Patient without DBS + existing DBS hardware explicitly required → not_eligible."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 66,
        "key_features": [],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": [
            "Parkinson disease",
            "existing DBS hardware implanted",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


# ---------------------------------------------------------------------------
# 10. Clear ordinary PD patient + broad PD trial returns eligible
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


# ---------------------------------------------------------------------------
# Task 99: Unknown/unclear propagation — targeted general cases
# ---------------------------------------------------------------------------

def test_stable_med_trial_no_duration_in_patient_gives_unclear_and_missing_duration():
    """Trial requires stable medication duration; patient has medication but no duration → unclear."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": ["on levodopa"],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": ["stable levodopa regimen for at least 4 weeks"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"
    assert "medication_stability_duration" in result["missing_information"]


def test_disease_stage_missing_hy_trial_gives_unclear():
    """Trial requires H&Y stage; patient disease_stage is missing → unclear."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 66,
        "disease_stage": None,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Hoehn and Yahr stage 2 to 4"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_medication_list_unknown_specific_med_trial_gives_unclear():
    """Trial requires specific medication; patient medication list is empty/unknown → unclear."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 60,
        "key_features": ["medication history not available"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["patients receiving rotigotine patch for at least 3 months"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_hard_exclusion_plus_missing_info_gives_not_eligible():
    """Hard exclusion criterion present together with missing info → not_eligible takes precedence."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 45,  # below 50 minimum
        "key_features": ["dose and frequency unclear"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": [
            "age 50 to 80 years",
            "stable levodopa regimen for at least 4 weeks",
        ],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"

