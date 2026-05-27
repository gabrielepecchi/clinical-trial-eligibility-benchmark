"""Unit tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial

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