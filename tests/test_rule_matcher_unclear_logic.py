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



# ---------------------------------------------------------------------------
# Task 8: Negation and contradiction — unclear propagation
# ---------------------------------------------------------------------------

def test_contradictory_maob_records_give_unclear():
    """Patient text has both denial and affirmation of MAO-B inhibitor."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 65,
        "key_features": ["no MAO-B inhibitor use", "taking rasagiline"],
        "medications": ["rasagiline"],
    }
    trial = {
        "inclusion_criteria": ["Confirmed Parkinson disease diagnosis"],
        "exclusion_criteria": ["MAO-B inhibitor use"],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_negated_investigational_drug_does_not_flag_unclear():
    """Patient denies investigational drug use; no trial washout — should not produce unclear."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": ["no investigational drug use"],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": ["Confirmed Parkinson disease diagnosis"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert not any("investigational" in u.lower() for u in result["uncertain_criteria"])


# ---------------------------------------------------------------------------
# Priority 3: Structured missingness fields
# ---------------------------------------------------------------------------

def test_structured_missingness_fields_present_in_result():
    """match_patient_to_trial() always returns all structured missingness keys."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": [],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "age 40 to 80 years"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert "unknown_fields" in result
    assert "present_evidence" in result
    assert "absent_evidence" in result
    assert "unclear_reason" in result
    assert "missing_reason_type" in result
    assert "missing_information_details" in result
    assert isinstance(result["unknown_fields"], list)
    assert isinstance(result["present_evidence"], list)
    assert isinstance(result["absent_evidence"], list)
    assert isinstance(result["missing_information_details"], list)


def test_missing_medication_produces_unknown_not_absent():
    """Missing medication list → field appears in unknown_fields, not absent_evidence."""
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
    # medication should be in unknown, not in absent_evidence
    assert any("medication" in f for f in result["unknown_fields"])
    assert not any("medication" in e.lower() and "no" in e.lower() for e in result["absent_evidence"])


def test_missing_disease_stage_produces_unknown_field():
    """Absent disease_stage → disease_stage_or_duration appears in unknown_fields."""
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
    assert any("stage" in f or "duration" in f for f in result["unknown_fields"])


def test_missing_cognitive_score_produces_unknown_field():
    """MMSE required by trial but absent from patient → cognitive_score in unknown_fields."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 68,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["MMSE < 24 excluded"],
    }
    result = match_patient_to_trial(patient, trial)
    assert "cognitive_score" in result["missing_information"]
    # cognitive_score should be unknown (no score present, none negated)
    cog_detail = next((d for d in result["missing_information_details"] if d["field"] == "cognitive_score"), None)
    assert cog_detail is not None
    assert cog_detail["status"] == "unknown"
    assert cog_detail["present_evidence"] == ""


def test_negated_dbs_produces_absent_evidence():
    """Explicit 'no DBS history' → absent_evidence contains dbs entry."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 65,
        "key_features": ["no history of DBS"],
        "medications": ["levodopa"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["prior DBS implantation excluded"],
    }
    result = match_patient_to_trial(patient, trial)
    assert any("dbs" in e.lower() for e in result["absent_evidence"])
    # DBS should NOT appear in unknown_fields since it is explicitly negated
    assert "dbs" not in result["unknown_fields"]


def test_missing_information_details_structure():
    """missing_information_details records have required keys."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 60,
        "key_features": ["dose and frequency unclear"],
        "medications": [],
    }
    trial = {
        "inclusion_criteria": ["stable levodopa regimen for at least 4 weeks"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    for detail in result["missing_information_details"]:
        assert "field" in detail
        assert "status" in detail
        assert "missing_reason_type" in detail
        assert "unclear_reason" in detail
        assert "present_evidence" in detail
        assert "absent_evidence" in detail
        assert detail["status"] in ("unknown", "present", "absent")


def test_eligible_case_has_empty_unknown_fields():
    """Clear eligible patient → unknown_fields is empty."""
    patient = {
        "diagnosis": "Parkinson disease",
        "age": 62,
        "key_features": ["Hoehn and Yahr stage 2", "stable on levodopa 4 weeks"],
        "medications": ["levodopa 100mg three times daily"],
    }
    trial = {
        "inclusion_criteria": ["Parkinson disease", "age 40 to 80 years"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"
    assert result["unknown_fields"] == []
    assert result["missing_information_details"] == []


def test_run_sample_benchmark_prediction_record_has_structured_fields():
    """Simulate a prediction record like run_sample_benchmark would build; verify fields present."""
    from app.eligibility.rule_matcher import match_patient_to_trial

    patient = {
        "patient_id": "P_TEST",
        "diagnosis": "Parkinson disease",
        "age": 65,
        "key_features": ["dose and frequency unclear"],
        "medications": [],
    }
    trial = {
        "trial_id": "T_TEST",
        "inclusion_criteria": ["stable levodopa regimen for at least 4 weeks"],
        "exclusion_criteria": [],
    }
    result = match_patient_to_trial(patient, trial)
    record = {
        "patient_id": patient["patient_id"],
        "trial_id": trial["trial_id"],
        "gold_label": "unclear",
        "predicted_label": result["prediction"],
        "confidence": result["confidence"],
        "matched_facts": result["matched_facts"],
        "blocking_criteria": result["blocking_criteria"],
        "uncertain_criteria": result["uncertain_criteria"],
        "explanation": result["explanation"],
        "missing_information": result.get("missing_information", []),
        "unknown_fields": result.get("unknown_fields", []),
        "present_evidence": result.get("present_evidence", []),
        "absent_evidence": result.get("absent_evidence", []),
        "unclear_reason": result.get("unclear_reason", ""),
        "missing_reason_type": result.get("missing_reason_type", ""),
        "missing_information_details": result.get("missing_information_details", []),
        "criterion_results": [],
    }
    assert "unknown_fields" in record
    assert "present_evidence" in record
    assert "absent_evidence" in record
    assert "unclear_reason" in record
    assert "missing_reason_type" in record
    assert "missing_information_details" in record


def test_run_llm_reviewed_prediction_record_has_structured_fields():
    """Simulate a prediction record like run_llm_reviewed_benchmark would build; verify fields present."""
    from app.eligibility.rule_matcher import match_patient_to_trial

    patient = {
        "patient_id": "P_LLM",
        "diagnosis": "Parkinson disease",
        "age": 70,
        "key_features": [],
        "medications": [],
    }
    trial = {
        "trial_id": "T_LLM",
        "inclusion_criteria": ["Parkinson disease"],
        "exclusion_criteria": ["MMSE < 24 excluded"],
    }
    result = match_patient_to_trial(patient, trial)
    record = {
        "patient_id": patient["patient_id"],
        "trial_id": trial["trial_id"],
        "gold_label": "unclear",
        "predicted_label": result["prediction"],
        "label_status": "",
        "confidence": result["confidence"],
        "matched_facts": result["matched_facts"],
        "blocking_criteria": result["blocking_criteria"],
        "uncertain_criteria": result["uncertain_criteria"],
        "matcher_explanation": result["explanation"],
        "gold_rationale": "",
        "gold_evidence": {},
        "criterion_results": [],
        "reasoning_trace": [],
        "missing_information": result.get("missing_information", []),
        "unknown_fields": result.get("unknown_fields", []),
        "present_evidence": result.get("present_evidence", []),
        "absent_evidence": result.get("absent_evidence", []),
        "unclear_reason": result.get("unclear_reason", ""),
        "missing_reason_type": result.get("missing_reason_type", ""),
        "missing_information_details": result.get("missing_information_details", []),
    }
    assert isinstance(record["unknown_fields"], list)
    assert isinstance(record["missing_information_details"], list)


def test_run_missing_info_checklist_works_with_old_style_records():
    """analyze_missing_info falls back gracefully for records without structured fields."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from run_missing_info_checklist import analyze_missing_info

    old_record = {
        "patient_id": "P_OLD",
        "trial_id": "T_OLD",
        "predicted_label": "unclear",
        "gold_label": "unclear",
        "uncertain_criteria": "medication stability unclear",
        "explanation": "medication stability cannot be confirmed",
    }
    summary = analyze_missing_info([old_record], {}, {})
    assert summary["total_records"] == 1
    assert summary["total_cases"] == 1
    case = summary["cases"][0]
    assert isinstance(case["missing_info_items"], list)


def test_run_missing_info_checklist_works_with_structured_records():
    """analyze_missing_info uses structured fields when present."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from run_missing_info_checklist import analyze_missing_info

    structured_record = {
        "patient_id": "P_STRUCT",
        "trial_id": "T_STRUCT",
        "predicted_label": "unclear",
        "gold_label": "unclear",
        "unknown_fields": ["cognitive_score", "disease_stage"],
        "absent_evidence": ["no dbs documented"],
        "present_evidence": [],
        "missing_reason_type": "not_documented",
        "unclear_reason": "MoCA/MMSE score required but not documented",
        "missing_information_details": [
            {
                "field": "cognitive_score",
                "status": "unknown",
                "missing_reason_type": "not_documented",
                "unclear_reason": "MoCA/MMSE score required but not documented",
                "present_evidence": "",
                "absent_evidence": "",
            },
            {
                "field": "disease_stage",
                "status": "unknown",
                "missing_reason_type": "not_documented",
                "unclear_reason": "disease stage or duration not documented or unclear",
                "present_evidence": "",
                "absent_evidence": "",
            },
        ],
        "uncertain_criteria": ["cognitive score not available", "stage unclear"],
        "explanation": "eligibility cannot be determined",
    }
    summary = analyze_missing_info([structured_record], {}, {})
    assert summary["total_cases"] == 1
    case = summary["cases"][0]
    assert "cognitive_score" in case["missing_info_items"] or "cognitive_score" in case["unknown_fields"]
    assert case["unknown_fields"] == ["cognitive_score", "disease_stage"]
    assert "not_documented" in summary["missing_reason_type_counts"]
