"""Unit tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision, CriterionType

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "prediction",
    "confidence",
    "matched_facts",
    "blocking_criteria",
    "uncertain_criteria",
    "explanation",
}

VALID_PREDICTIONS = {"eligible", "not_eligible", "unclear"}


# ---------------------------------------------------------------------------
# Inline patient and trial helpers
# ---------------------------------------------------------------------------

def make_patient(**kwargs) -> dict:
    """Return a minimal patient dict, overridable via kwargs."""
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


def make_trial(**kwargs) -> dict:
    """Return a minimal trial dict, overridable via kwargs."""
    base = {
        "trial_id": "T_TEST",
        "inclusion_criteria": [
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        "exclusion_criteria": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_output_has_all_required_keys():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert REQUIRED_KEYS == set(result.keys()) & REQUIRED_KEYS


def test_prediction_is_valid_string():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert result["prediction"] in VALID_PREDICTIONS


def test_confidence_is_float():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["confidence"], float)


def test_confidence_between_zero_and_one():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert 0.0 <= result["confidence"] <= 1.0


def test_matched_facts_is_list():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["matched_facts"], list)


def test_blocking_criteria_is_list():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["blocking_criteria"], list)


def test_uncertain_criteria_is_list():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["uncertain_criteria"], list)


def test_explanation_is_non_empty_string():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["explanation"], str)
    assert result["explanation"].strip() != ""


# ---------------------------------------------------------------------------
# Eligible
# ---------------------------------------------------------------------------

def test_eligible_when_age_and_diagnosis_match():
    patient = make_patient(age=60, diagnosis=["Parkinson disease"])
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"


def test_eligible_confidence_is_correct_with_matched_facts():
    patient = make_patient(age=60, diagnosis=["Parkinson disease"])
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"
    assert result["confidence"] == 0.75


def test_eligible_blocking_criteria_is_empty():
    patient = make_patient(age=60, diagnosis=["Parkinson disease"])
    trial = make_trial()
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible"
    assert result["blocking_criteria"] == []


# ---------------------------------------------------------------------------
# Not eligible — age out of range
# ---------------------------------------------------------------------------

def test_not_eligible_when_patient_too_young():
    patient = make_patient(age=30)
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_when_patient_too_old():
    patient = make_patient(age=90)
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_age_confidence_is_correct():
    patient = make_patient(age=30)
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["confidence"] == 0.90


def test_not_eligible_age_has_blocking_criterion():
    patient = make_patient(age=30)
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert len(result["blocking_criteria"]) >= 1


# ---------------------------------------------------------------------------
# Not eligible — prior DBS
# ---------------------------------------------------------------------------

def test_not_eligible_when_patient_has_dbs():
    patient = make_patient(
        key_features=["bilateral STN DBS implanted 3 years ago"],
        exclusions=["prior DBS surgery"],
    )
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_dbs_has_blocking_criterion():
    patient = make_patient(
        key_features=["bilateral STN DBS implanted 3 years ago"],
        exclusions=["prior DBS surgery"],
    )
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    result = match_patient_to_trial(patient, trial)
    assert any("dbs" in c.lower() or "deep brain" in c.lower() for c in result["blocking_criteria"])


# ---------------------------------------------------------------------------
# Not eligible — cognitive impairment (MMSE)
# ---------------------------------------------------------------------------

def test_not_eligible_when_mmse_below_threshold():
    patient = make_patient(
        key_features=["MMSE score 21"],
        exclusions=["cognitive impairment"],
    )
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Severe cognitive impairment (MMSE < 24)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_mmse_has_blocking_criterion():
    patient = make_patient(
        key_features=["MMSE score 21"],
        exclusions=["cognitive impairment"],
    )
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Severe cognitive impairment (MMSE < 24)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert any("mmse" in c.lower() for c in result["blocking_criteria"])


# ---------------------------------------------------------------------------
# Not eligible — cognitive impairment (MoCA)
# ---------------------------------------------------------------------------

def test_not_eligible_when_moca_below_threshold():
    patient = make_patient(key_features=["MoCA score 19"])
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Dementia diagnosis (MoCA < 21)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_moca_has_blocking_criterion():
    patient = make_patient(key_features=["MoCA score 19"])
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Dementia diagnosis (MoCA < 21)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert any("moca" in c.lower() for c in result["blocking_criteria"])


# ---------------------------------------------------------------------------
# Unclear — medication history unclear
# ---------------------------------------------------------------------------

def test_unclear_when_medication_history_unclear():
    patient = make_patient(
        medications=["levodopa/carbidopa — dose and frequency unclear"],
        key_features=["self-reported medication use", "no pharmacy records available"],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Currently on stable levodopa therapy for at least 3 months",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear"


def test_unclear_confidence_is_correct():
    patient = make_patient(
        medications=["levodopa/carbidopa — dose and frequency unclear"],
        key_features=["self-reported medication use", "no pharmacy records available"],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Currently on stable levodopa therapy for at least 3 months",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["confidence"] == 0.40


def test_unclear_has_uncertain_criterion():
    patient = make_patient(
        medications=["levodopa/carbidopa — dose and frequency unclear"],
        key_features=["self-reported medication use", "no pharmacy records available"],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Currently on stable levodopa therapy for at least 3 months",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert len(result["uncertain_criteria"]) >= 1


# ---------------------------------------------------------------------------
# Criterion-level matcher
# ---------------------------------------------------------------------------

VALID_DECISIONS = {CriterionDecision.met, CriterionDecision.not_met, CriterionDecision.unknown}


def test_criteria_count_matches_trial_criteria():
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years", "Confirmed Parkinson disease diagnosis"],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    results = match_patient_to_trial_criteria(make_patient(), trial)
    assert len(results) == 3


def test_criteria_types_match_trial_sections():
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    results = match_patient_to_trial_criteria(make_patient(), trial)
    types = [r.criterion_type for r in results]
    assert types[0] == CriterionType.inclusion
    assert types[1] == CriterionType.exclusion


def test_all_decisions_are_valid():
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years", "Confirmed Parkinson disease diagnosis"],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    results = match_patient_to_trial_criteria(make_patient(), trial)
    for r in results:
        assert r.decision in VALID_DECISIONS


def test_age_inclusion_met_when_in_range():
    patient = make_patient(age=60)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    results = match_patient_to_trial_criteria(patient, trial)
    age_result = results[0]
    assert age_result.decision == CriterionDecision.met


def test_age_inclusion_not_met_when_out_of_range():
    patient = make_patient(age=30)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    results = match_patient_to_trial_criteria(patient, trial)
    age_result = results[0]
    assert age_result.decision == CriterionDecision.not_met


def test_dbs_exclusion_met_when_patient_has_dbs():
    patient = make_patient(
        key_features=["bilateral STN DBS implanted 3 years ago"],
        exclusions=["prior DBS surgery"],
    )
    trial = make_trial(
        inclusion_criteria=[],
        exclusion_criteria=["Deep brain stimulation (DBS) implant"],
    )
    results = match_patient_to_trial_criteria(patient, trial)
    dbs_result = results[0]
    assert dbs_result.decision == CriterionDecision.met


def test_missing_moca_score_returns_unknown():
    patient = make_patient(key_features=[])
    trial = make_trial(
        inclusion_criteria=[],
        exclusion_criteria=["Dementia diagnosis (MoCA < 21)"],
    )
    results = match_patient_to_trial_criteria(patient, trial)
    assert results[0].decision == CriterionDecision.unknown


def test_missing_mmse_score_returns_unknown():
    patient = make_patient(key_features=[])
    trial = make_trial(
        inclusion_criteria=[],
        exclusion_criteria=["Severe cognitive impairment (MMSE < 24)"],
    )
    results = match_patient_to_trial_criteria(patient, trial)
    assert results[0].decision == CriterionDecision.unknown


# ---------------------------------------------------------------------------
# missing_information
# ---------------------------------------------------------------------------

def test_missing_information_key_present():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert "missing_information" in result


def test_missing_information_is_list():
    result = match_patient_to_trial(make_patient(), make_trial())
    assert isinstance(result["missing_information"], list)


def test_missing_age_adds_age_label():
    patient = make_patient(age=None)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    result = match_patient_to_trial(patient, trial)
    assert "age" in result["missing_information"]


def test_unclear_medication_adds_medication_details_label():
    patient = make_patient(
        medications=["levodopa/carbidopa — dose and frequency unclear"],
        key_features=["self-reported medication use", "no pharmacy records available"],
    )
    trial = make_trial(
        inclusion_criteria=["Currently on stable levodopa therapy for at least 3 months"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert "medication_details" in result["missing_information"]


def test_unclear_disease_stage_adds_label():
    patient = make_patient(key_features=["disease stage unclear"])
    trial = make_trial(
        inclusion_criteria=["Hoehn and Yahr stage 1 to 3"],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert "disease_stage_or_duration" in result["missing_information"]


def test_missing_moca_score_adds_cognitive_score_label():
    patient = make_patient(key_features=[])
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Dementia diagnosis (MoCA < 21)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert "cognitive_score" in result["missing_information"]


def test_missing_mmse_score_adds_cognitive_score_label():
    patient = make_patient(key_features=[])
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years"],
        exclusion_criteria=["Severe cognitive impairment (MMSE < 24)"],
    )
    result = match_patient_to_trial(patient, trial)
    assert "cognitive_score" in result["missing_information"]


def test_no_duplicate_cognitive_score_label():
    patient = make_patient(key_features=[])
    trial = make_trial(
        inclusion_criteria=[],
        exclusion_criteria=[
            "Severe cognitive impairment (MMSE < 24)",
            "Dementia diagnosis (MoCA < 21)",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["missing_information"].count("cognitive_score") == 1


# ---------------------------------------------------------------------------
# Temporal medication stability
# ---------------------------------------------------------------------------

def _trial(criterion: str) -> dict:
    return {"trial_id": "T_FMT", "inclusion_criteria": [criterion], "exclusion_criteria": []}


def test_stable_6_weeks_meets_4_week_requirement():
    patient = make_patient(key_features=["medication regimen stable for 6 weeks"])
    trial = _trial("Stable medication regimen for at least 4 weeks")
    assert match_patient_to_trial(patient, trial)["prediction"] == "eligible"


def test_changed_2_weeks_ago_fails_4_week_requirement():
    patient = make_patient(key_features=["medication regimen changed 2 weeks ago"])
    trial = _trial("Stable medication regimen for at least 4 weeks")
    assert match_patient_to_trial(patient, trial)["prediction"] in {"not_eligible", "unclear"}


def test_stable_1_month_fails_3_month_requirement():
    patient = make_patient(key_features=["medication regimen stable for 1 month"])
    trial = _trial("Stable medication regimen for at least 3 months")
    assert match_patient_to_trial(patient, trial)["prediction"] in {"not_eligible", "unclear"}


def test_stable_6_weeks_criterion_met_for_4_week_requirement():
    patient = make_patient(key_features=["medication regimen stable for 6 weeks"])
    results = match_patient_to_trial_criteria(patient, _trial("Stable medication regimen for at least 4 weeks"))
    assert results[0].decision == CriterionDecision.met


def test_stable_1_month_criterion_not_met_or_unknown_for_3_month_requirement():
    patient = make_patient(key_features=["medication regimen stable for 1 month"])
    results = match_patient_to_trial_criteria(patient, _trial("Stable medication regimen for at least 3 months"))
    assert results[0].decision in {CriterionDecision.not_met, CriterionDecision.unknown}


# ---------------------------------------------------------------------------
# Medication and procedure exclusion logic
# ---------------------------------------------------------------------------

def _excl_trial(criterion: str) -> dict:
    return {"trial_id": "T_EXCL", "inclusion_criteria": [], "exclusion_criteria": [criterion]}


# Medication class exclusions

def test_maob_rasagiline_predicts_not_eligible():
    patient = make_patient(medications=["rasagiline 1 mg daily"])
    result = match_patient_to_trial(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_selegiline_predicts_not_eligible():
    patient = make_patient(medications=["selegiline 5 mg daily"])
    result = match_patient_to_trial(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_safinamide_predicts_not_eligible():
    patient = make_patient(medications=["safinamide 50 mg daily"])
    result = match_patient_to_trial(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] == "not_eligible"


def test_maob_rasagiline_criterion_level_met():
    patient = make_patient(medications=["rasagiline 1 mg daily"])
    results = match_patient_to_trial_criteria(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert results[0].decision == CriterionDecision.met


def test_maob_no_inhibitor_not_not_eligible():
    patient = make_patient(medications=["levodopa/carbidopa 100/25 mg three times daily"])
    result = match_patient_to_trial(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] != "not_eligible"


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


def test_no_maob_documented_not_not_eligible():
    patient = make_patient(medications=["no MAO-B inhibitor use documented"])
    result = match_patient_to_trial(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert result["prediction"] != "not_eligible"


def test_no_maob_documented_criterion_level_not_met():
    patient = make_patient(medications=["no MAO-B inhibitor use documented"])
    results = match_patient_to_trial_criteria(patient, _excl_trial("Current MAO-B inhibitor use"))
    assert results[0].decision == CriterionDecision.not_met