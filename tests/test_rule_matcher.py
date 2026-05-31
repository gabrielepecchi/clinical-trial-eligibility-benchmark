"""Unit tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision, CriterionType
from tests.helpers import make_patient

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
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_when_patient_too_old():
    patient = make_patient(age=90)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"


def test_not_eligible_age_confidence_is_correct():
    patient = make_patient(age=30)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    result = match_patient_to_trial(patient, trial)
    assert result["confidence"] == 0.90


def test_not_eligible_age_has_blocking_criterion():
    patient = make_patient(age=30)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
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
    assert results[0].decision == CriterionDecision.met


def test_age_inclusion_not_met_when_out_of_range():
    patient = make_patient(age=30)
    trial = make_trial(inclusion_criteria=["Age 40 to 80 years"], exclusion_criteria=[])
    results = match_patient_to_trial_criteria(patient, trial)
    assert results[0].decision == CriterionDecision.not_met


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
    assert results[0].decision == CriterionDecision.met


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
# Unclear — unverifiable inclusion criteria burden
# ---------------------------------------------------------------------------

def test_unclear_when_multiple_unverifiable_inclusion_criteria():
    """Predicts eligible today; should predict unclear once the burden check is implemented."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease, stable"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Confirmed diagnosis of idiopathic Parkinson disease",
            "Ability to safely operate and use the study device independently",
            "Access to home wireless internet (WiFi) required for data transmission",
            "No concurrent participation in another interventional clinical trial",
            "Medical clearance from physician prior to study participation",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    # The patient has no blocking criteria, but 4 inclusion criteria cannot be
    # verified from the patient profile. The matcher should flag this as unclear.
    assert result["prediction"] == "unclear", (
        f"Expected 'unclear' due to unverifiable inclusion criteria burden, "
        f"got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}, "
        f"missing_information={result.get('missing_information', [])}"
    )
    has_uncertainty_signal = (
        any(
            "unverifiable" in c.lower() or "inclusion" in c.lower()
            for c in result["uncertain_criteria"]
        )
        or "unverifiable_inclusion_criteria" in result.get("missing_information", [])
    )
    assert has_uncertainty_signal, (
        "Expected an uncertainty signal mentioning unverifiable or missing inclusion criteria. "
        f"uncertain_criteria={result['uncertain_criteria']}, "
        f"missing_information={result.get('missing_information', [])}"
    )


# ---------------------------------------------------------------------------
# Comorbidity / protocol-risk — target-population exemption and contraindication escalation
# ---------------------------------------------------------------------------

def test_eligible_when_fog_patient_in_fog_device_trial():
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["freezing of gait episodes documented"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Documented freezing of gait episodes",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible", (
        f"Expected 'eligible' for FoG patient in FoG trial, got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )


def test_eligible_when_dbs_patient_in_dbs_effects_study():
    patient = make_patient(
        age=62,
        diagnosis=["Parkinson disease"],
        key_features=["bilateral STN DBS implanted 2 years ago"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "DBS-implanted patients with directional lead hardware",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible", (
        f"Expected 'eligible' for DBS patient in DBS effects study, got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )


def test_eligible_when_frail_patient_in_frailty_physiotherapy_trial():
    patient = make_patient(
        age=78,
        diagnosis=["Parkinson disease"],
        key_features=["frailty noted, recurrent falls"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Frailty present as defined by Fried criteria",
            "Home physiotherapy frailty rehabilitation study",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible", (
        f"Expected 'eligible' for frail patient in frailty trial, got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )


def test_not_eligible_when_pacemaker_in_rtms_trial():
    patient = make_patient(
        age=68,
        diagnosis=["Parkinson disease"],
        key_features=["implanted cardiac pacemaker"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[
            "Repetitive transcranial magnetic stimulation (rTMS) contraindications",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker patient in rTMS trial, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}, "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )

