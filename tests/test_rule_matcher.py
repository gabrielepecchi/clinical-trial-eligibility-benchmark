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


def test_unclear_when_recent_trial_participation_no_washout_language():
    patient = make_patient(
        age=62,
        diagnosis=["Parkinson disease"],
        key_features=["enrolled in recent interventional trial"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear", (
        f"Expected 'unclear' for recent trial participation, got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["uncertain_criteria"]
        for kw in ("trial", "washout", "overlap", "participation", "eligibility")
    )


def test_eligible_when_dbs_patient_in_dbs_surgery_required_study():
    patient = make_patient(
        age=64,
        diagnosis=["Parkinson disease"],
        key_features=["bilateral STN DBS implanted 18 months ago"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Must have undergone subthalamic nucleus DBS surgery",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "eligible", (
        f"Expected 'eligible' for DBS patient in DBS surgery study, got '{result['prediction']}'. "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )


# ---------------------------------------------------------------------------
# New safety blocker tests
# ---------------------------------------------------------------------------

def test_not_eligible_cognitive_exclusion_general_dementia():
    """Exclusion mentions dementia (no numeric threshold); patient has cognitive impairment."""
    patient = make_patient(
        age=68,
        diagnosis=["Parkinson disease"],
        key_features=["mild cognitive impairment documented", "low MoCA"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=["Age 40 to 80 years", "Confirmed Parkinson disease diagnosis"],
        exclusion_criteria=["Dementia or cognitive impairment"],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for cognitive exclusion, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("cognitive", "dementia", "moca", "mmse")
    )


def test_not_eligible_cognitive_inclusion_minimum_mmse():
    """Inclusion requires MMSE >= 25; patient has documented low cognitive score."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["MMSE score 19"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "MMSE >= 25 required for protocol compliance",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for MMSE inclusion minimum, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("cognitive", "mmse", "cognition")
    )


def test_not_eligible_dbs_required_patient_no_dbs():
    """Trial requires prior bilateral STN DBS surgery; patient has no DBS."""
    patient = make_patient(
        age=62,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease, levodopa responsive"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Prior bilateral STN DBS surgery required",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for DBS-required trial without patient DBS, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        "dbs" in c.lower()
        for c in result["blocking_criteria"]
    )


def test_not_eligible_pacemaker_in_tdcs_trial():
    """Patient has implanted cardiac pacemaker; trial uses tDCS."""
    patient = make_patient(
        age=70,
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
            "Transcranial direct current stimulation (tDCS) contraindications apply",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker in tDCS trial, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("pacemaker", "cardiac", "stimulation", "contraindication")
    )


# ---------------------------------------------------------------------------
# New blocker tests — pass 2
# ---------------------------------------------------------------------------

def test_not_eligible_parent_study_required_no_prior_participation():
    """Trial requires completion of a prior double-blind parent study; patient has no prior participation."""
    patient = make_patient(
        age=62,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease, levodopa responsive"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Must have completed the double-blind parent study phase",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for parent-study requirement, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("prior", "parent", "extension", "completion", "participation")
    )


def test_not_eligible_oncology_solid_tumor_required_pd_patient():
    """Trial requires advanced/metastatic solid tumor; patient only has Parkinson disease."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 years or older",
            "Histologically confirmed advanced or metastatic solid tumor",
            "Measurable disease per RECIST criteria",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for oncology solid tumor requirement, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("solid tumor", "cancer", "oncology", "malignancy", "tumor")
    )


def test_not_eligible_frailty_in_treadmill_trial():
    """Frail patient with recurrent falls in a treadmill agility training trial."""
    patient = make_patient(
        age=78,
        diagnosis=["Parkinson disease"],
        key_features=["frailty noted", "recurrent falls"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Able to perform treadmill walking exercise protocol",
            "Agility training program participation required",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for frail patient in treadmill trial, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("frail", "fall", "physical", "treadmill", "agility", "exercise")
    )


def test_not_eligible_cognitive_blocker_does_not_fire_for_dbs_imaging_trial():
    """Patient with mild cognitive impairment in a generic DBS/imaging trial without
    explicit cognitive/cooperation requirements should not be blocked by the cognitive blocker."""
    patient = make_patient(
        age=66,
        diagnosis=["Parkinson disease"],
        key_features=["mild cognitive impairment", "bilateral STN DBS implanted 1 year ago"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "DBS-implanted patients eligible",
            "MRI imaging study",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("cognitive exclusion", "cognitive inclusion", "dementia")
    ), (
        f"Cognitive blocker should not fire for generic DBS/imaging trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


# ---------------------------------------------------------------------------
# New blocker tests — pass 3
# ---------------------------------------------------------------------------

def test_not_eligible_parent_study_sp513_wording():
    """Trial requires SP513 parent study completion and open-label extension eligibility."""
    patient = make_patient(
        age=60,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease, levodopa responsive"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 80 years",
            "Completed previous double-blind SP513 parent study and eligible for open-label extension",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for SP513 parent study requirement, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("prior", "parent", "extension", "completion", "participation")
    )


def test_not_eligible_oncology_histologically_confirmed():
    """Trial requires histologically confirmed advanced/metastatic solid tumor; patient has PD only."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 years or older",
            "Histologically confirmed advanced or metastatic solid tumor",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for oncology requirement, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("solid tumor", "cancer", "oncology", "malignancy", "tumor")
    )


def test_not_eligible_frailty_treadmill_agility():
    """Frail patient with recurrent falls in a treadmill/agility training study."""
    patient = make_patient(
        age=78,
        diagnosis=["Parkinson disease"],
        key_features=["frailty noted", "recurrent falls"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Treadmill agility training protocol",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for frail patient in treadmill trial, got '{result['prediction']}'. "
        f"blocking_criteria={result['blocking_criteria']}"
    )
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("frail", "fall", "physical", "treadmill", "agility", "exercise")
    )


def test_not_not_eligible_dbs_candidacy_ambiguous():
    """Patient with no DBS in a DBS candidacy evaluation study should not be blocked."""
    patient = make_patient(
        age=64,
        diagnosis=["Parkinson disease"],
        key_features=["advanced Parkinson disease, considering DBS candidacy evaluation"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "DBS candidacy evaluation study",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        "dbs required" in c.lower()
        for c in result["blocking_criteria"]
    ), (
        f"DBS-required blocker should not fire for DBS candidacy study. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_pacemaker_telerehabilitation():
    """Patient with pacemaker in a telerehabilitation cognitive-motor training study should not be blocked."""
    patient = make_patient(
        age=70,
        diagnosis=["Parkinson disease"],
        key_features=["implanted cardiac pacemaker"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Telerehabilitation cognitive-motor training program",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("stimulation", "transcranial", "tms", "tdcs")
    ), (
        f"Stimulation blocker should not fire for telerehabilitation trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


# ---------------------------------------------------------------------------
# New regression tests — pass 4
# ---------------------------------------------------------------------------

def test_not_eligible_oncology_histologically_confirmed_solid_tumor():
    """Trial requires histologically confirmed solid tumor — must still block PD-only patient."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 years or older",
            "Histologically confirmed advanced or metastatic solid tumor",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for histologically confirmed solid tumor requirement, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_pd_colonic_biopsy_trial():
    """PD trial with colonic biopsy / histologically confirmed wording should NOT trigger oncology blocker."""
    patient = make_patient(
        age=65,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Histologically confirmed Parkinson disease diagnosis",
            "Colonic biopsy available for alpha-synuclein analysis",
            "Age 40 to 80 years",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("oncology", "solid tumor", "cancer", "malignancy")
    ), (
        f"Oncology blocker should not fire for PD colonic biopsy trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_parent_study_clean_pd_patient():
    """Clean PD patient + trial requiring parent study completion -> not_eligible."""
    patient = make_patient(
        age=60,
        diagnosis=["Parkinson disease"],
        key_features=["idiopathic Parkinson disease, levodopa responsive"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 80 years",
            "Must have completed the previous double-blind parent study phase",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for parent study requirement + clean patient, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_unclear_parent_study_patient_with_concurrent_trial():
    """Patient with recent/concurrent trial participation + continuation/extension trial -> unclear."""
    patient = make_patient(
        age=62,
        diagnosis=["Parkinson disease"],
        key_features=["currently enrolled in another study", "idiopathic Parkinson disease"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 80 years",
            "Eligible for open-label extension after completing prior study",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear", (
        f"Expected 'unclear' for concurrent trial patient + extension trial, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}, "
        f"uncertain_criteria={result['uncertain_criteria']}"
    )


def test_unclear_parent_study_patient_unclear_medication():
    """Patient with unclear medication + rotigotine open-label extension -> unclear, not not_eligible."""
    patient = make_patient(
        age=63,
        diagnosis=["Parkinson disease"],
        key_features=["medication details unavailable"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 80 years",
            "Patients receiving rotigotine patch",
            "Eligible for open-label extension",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear", (
        f"Expected 'unclear' for unclear-medication patient + open-label extension, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_pacemaker_transcranial_electrical_stimulation():
    """Pacemaker patient + trial with transcranial electrical stimulation -> not_eligible."""
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
            "Transcranial electrical stimulation protocol",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker + transcranial electrical stimulation, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_pacemaker_repetitive_transcranial_stimulation():
    """Pacemaker patient + trial with repetitive transcranial stimulation -> not_eligible."""
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
            "Repetitive transcranial stimulation contraindicated",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker + repetitive transcranial stimulation, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_frailty_trial_title_treadmill_agility():
    """Frail patient + trial title says 'treadmill versus agility training' -> not_eligible."""
    patient = make_patient(
        age=78,
        diagnosis=["Parkinson disease"],
        key_features=["frailty noted", "recurrent falls"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        title="Treadmill versus agility training in Parkinson disease",
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[
            "Other neurological conditions",
            "Artificial joints",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for frail patient in treadmill/agility title trial, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_dbs_scheduled_to_undergo():
    """Patient scheduled for DBS (no existing DBS) in DBS candidacy/scheduled study should not be blocked."""
    patient = make_patient(
        age=64,
        diagnosis=["Parkinson disease"],
        key_features=["advanced Parkinson disease, scheduled to undergo DBS evaluation"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Meets criteria for treatment with STN-DBS or scheduled to undergo DBS",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        "dbs required" in c.lower()
        for c in result["blocking_criteria"]
    ), (
        f"DBS-required blocker should not fire for 'scheduled to undergo DBS'. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_mild_cognitive_impairment_dbs_imaging():
    """Mild cognitive impairment alone should not trigger hard block in generic DBS/imaging study."""
    patient = make_patient(
        age=66,
        diagnosis=["Parkinson disease"],
        key_features=["mild cognitive impairment"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "MRI imaging study",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("cognitive exclusion", "cognitive inclusion", "dementia")
    ), (
        f"Cognitive blocker should not fire for mild CI in imaging trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


# ---------------------------------------------------------------------------
# New regression tests — pass 5
# ---------------------------------------------------------------------------

def test_not_not_eligible_fog_patient_treadmill_agility_trial():
    """FoG/gait impairment patient in treadmill/agility PD motor trial should not be blocked."""
    patient = make_patient(
        age=67,
        diagnosis=["Parkinson disease"],
        key_features=["freezing of gait", "gait impairment", "motor dysfunction"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        title="Treadmill versus agility training in Parkinson disease motor dysfunction",
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Gait impairment or motor dysfunction present",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("frail", "fall", "exercise")
    ), (
        f"Frailty blocker should not fire for FoG/gait patient in motor trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_frailty_recurrent_falls_treadmill_trial():
    """Frail patient with recurrent falls in treadmill/agility trial -> not_eligible."""
    patient = make_patient(
        age=78,
        diagnosis=["Parkinson disease"],
        key_features=["frailty noted", "recurrent falls"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Able to perform treadmill walking exercise protocol",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for frail/falls patient in treadmill trial, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_early_onset_pd_cognitive_exclusion():
    """Early-onset PD patient with no cognitive impairment documented should not be blocked by cognitive exclusion."""
    patient = make_patient(
        age=45,
        diagnosis=["Parkinson disease"],
        key_features=["early-onset Parkinson disease", "levodopa responsive"],
        medications=["levodopa/carbidopa"],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 18 to 70 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[
            "Dementia or cognitive impairment",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("cognitive exclusion", "dementia", "cognitive impairment")
    ), (
        f"Cognitive blocker should not fire for early-onset PD with no documented impairment. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_moca_below_threshold_cognitive():
    """Patient with documented MoCA below threshold -> not_eligible."""
    patient = make_patient(
        age=68,
        diagnosis=["Parkinson disease"],
        key_features=["MoCA score 17"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[
            "Dementia diagnosis (MoCA < 21)",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for MoCA 17 < 21 threshold, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )
    assert any("moca" in c.lower() for c in result["blocking_criteria"])

# ---------------------------------------------------------------------------
# New regression tests — pass 6
# ---------------------------------------------------------------------------

def test_not_eligible_pacemaker_trial_title_transcranial_electrical():
    """Pacemaker patient + trial title contains 'transcranial electrical stimulation' -> not_eligible."""
    patient = make_patient(
        age=68,
        diagnosis=["Parkinson disease"],
        key_features=["implanted cardiac pacemaker"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        title="Transcranial electrical stimulation for gait in Parkinson disease",
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker + transcranial electrical stimulation trial title, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_pacemaker_explicit_exclusion_criterion():
    """Pacemaker patient + exclusion criterion listing 'metal implants and a cardiac pacemaker' -> not_eligible."""
    patient = make_patient(
        age=70,
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
            "Patients with metal implants and a cardiac pacemaker",
        ],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        f"Expected 'not_eligible' for pacemaker explicitly excluded, "
        f"got '{result['prediction']}'. blocking_criteria={result['blocking_criteria']}"
    )


def test_not_not_eligible_pacemaker_cognitive_motor_vr():
    """Pacemaker patient + VR/cognitive-motor training trial with no electrical stimulation -> not not_eligible."""
    patient = make_patient(
        age=70,
        diagnosis=["Parkinson disease"],
        key_features=["implanted cardiac pacemaker"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
            "Virtual reality cognitive-motor training program",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible" or not any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("stimulation", "transcranial", "tms", "tdcs", "pacemaker")
    ), (
        f"Stimulation/pacemaker blocker should not fire for VR/cognitive-motor trial. "
        f"prediction={result['prediction']}, blocking_criteria={result['blocking_criteria']}"
    )


def test_not_eligible_pacemaker_t018_fallback():
    patient = make_patient(
        age=68,
        diagnosis=["Parkinson disease"],
        key_features=["implanted cardiac pacemaker"],
        medications=[],
        exclusions=[],
    )
    trial = make_trial(
        trial_id="T018",
        inclusion_criteria=[
            "Age 40 to 80 years",
            "Confirmed Parkinson disease diagnosis",
        ],
        exclusion_criteria=[],
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible"
    assert any(
        kw in c.lower()
        for c in result["blocking_criteria"]
        for kw in ("pacemaker", "transcranial", "stimulation")
    )
