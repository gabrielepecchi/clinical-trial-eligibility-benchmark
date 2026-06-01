"""Clinical regression tests for the rule-based eligibility matcher.

These tests lock in correct matcher behaviour for specific known clinical
patterns. They use small synthetic fixtures and do not depend on benchmark
data files.

Run with:
    PYTHONPATH=. python -m pytest tests/test_clinical_regressions.py -v
"""

import pytest
from app.eligibility.rule_matcher import match_patient_to_trial

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _patient(**kwargs) -> dict:
    base = {
        "patient_id": "reg_patient",
        "age": 65,
        "diagnosis": "idiopathic Parkinson disease",
        "medications": ["levodopa/carbidopa"],
    }
    base.update(kwargs)
    return base


def _trial(criteria: str, **kwargs) -> dict:
    base = {
        "trial_id": "reg_trial",
        "eligibility_criteria": criteria,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# DBS history
# ---------------------------------------------------------------------------

def test_no_dbs_history_does_not_block_eligibility():
    """Patient with no DBS history should not be blocked by a DBS exclusion rule."""
    patient = _patient(dbs_history=False)
    trial = _trial(
        "Inclusion: Idiopathic Parkinson disease. Age >= 40. "
        "Exclusion: Prior deep brain stimulation."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible", (
        "Patient with no DBS history should not be blocked by DBS exclusion"
    )


@pytest.mark.xfail(
    reason=(
        "Matcher does not yet detect prior DBS history from the patient record "
        "and apply it against a DBS exclusion criterion in the trial. "
        "Implement DBS history detection before removing xfail."
    ),
    strict=False,
)
def test_prior_dbs_blocks_eligibility_for_drug_trial():
    """Patient with prior DBS should be blocked or flagged for a drug trial that excludes DBS."""
    patient = _patient(dbs_history=True)
    trial = _trial(
        "Inclusion: Idiopathic Parkinson disease. Age >= 40. "
        "Exclusion: Prior deep brain stimulation implant."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] in {"not_eligible", "unclear"}, (
        "Patient with prior DBS should be not_eligible or unclear for a DBS-excluding drug trial"
    )


# ---------------------------------------------------------------------------
# MAO-B inhibitor exclusion
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Matcher does not yet cross-reference patient medication list against "
        "MAO-B inhibitor exclusion criteria in the trial text. "
        "Implement medication exclusion detection before removing xfail."
    ),
    strict=False,
)
def test_maob_inhibitor_triggers_blocking_for_excluding_trial():
    """Current MAO-B inhibitor use should produce not_eligible when the trial excludes it."""
    patient = _patient(medications=["levodopa/carbidopa", "rasagiline"])
    trial = _trial(
        "Inclusion: Parkinson disease. Age >= 40. "
        "Exclusion: Current use of MAO-B inhibitors including rasagiline or selegiline."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] in {"not_eligible", "unclear"}, (
        "Current MAO-B inhibitor use should be not_eligible or flagged for a trial excluding MAO-B inhibitors"
    )


def test_no_maob_inhibitor_does_not_block():
    """Patient not using MAO-B inhibitors should not be blocked by a MAO-B exclusion."""
    patient = _patient(medications=["levodopa/carbidopa", "pramipexole"])
    trial = _trial(
        "Inclusion: Parkinson disease. Age >= 40. "
        "Exclusion: Current use of MAO-B inhibitors."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible", (
        "Patient not using MAO-B inhibitors should not be blocked by MAO-B exclusion"
    )


# ---------------------------------------------------------------------------
# Age boundary
# ---------------------------------------------------------------------------

def test_age_at_lower_boundary_is_accepted():
    """Patient aged exactly 40 should be accepted when criteria require age 40 to 80."""
    patient = _patient(age=40)
    trial = _trial(
        "Inclusion: Parkinson disease. Age 40 to 80."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] != "not_eligible", (
        "Age 40 should be accepted when the lower boundary is 40 (inclusive)"
    )


@pytest.mark.xfail(
    reason=(
        "Matcher does not yet parse age range criteria from free-text eligibility "
        "and enforce a lower boundary block when patient age is below it. "
        "Implement age boundary enforcement before removing xfail."
    ),
    strict=False,
)
def test_age_below_lower_boundary_is_not_eligible():
    """Patient aged 39 should be not_eligible when criteria require age 40 to 80."""
    patient = _patient(age=39)
    trial = _trial(
        "Inclusion: Parkinson disease. Age 40 to 80."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        "Age 39 should be not_eligible when the lower boundary is 40"
    )


@pytest.mark.xfail(
    reason=(
        "Matcher does not yet parse age range criteria from free-text eligibility "
        "and enforce an upper boundary block when patient age exceeds it. "
        "Implement age boundary enforcement before removing xfail."
    ),
    strict=False,
)
def test_age_above_upper_boundary_is_not_eligible():
    """Patient aged 81 should be not_eligible when criteria require age 40 to 80."""
    patient = _patient(age=81)
    trial = _trial(
        "Inclusion: Parkinson disease. Age 40 to 80."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        "Age 81 should be not_eligible when the upper boundary is 80"
    )


# ---------------------------------------------------------------------------
# Cognitive score / MoCA
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Matcher does not yet flag missing MoCA score as unclear "
        "when the trial explicitly requires a minimum MoCA. "
        "Implement missing-cognitive-score uncertainty detection before removing xfail."
    ),
    strict=False,
)
def test_missing_moca_score_returns_unclear_for_moca_required_trial():
    """Missing MoCA score should produce unclear when the trial requires MoCA >= 24."""
    patient = _patient()  # no moca_score field
    trial = _trial(
        "Inclusion: Parkinson disease. Age >= 40. MoCA score >= 24 required."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "unclear", (
        "Missing MoCA score should produce unclear when MoCA is an explicit inclusion threshold"
    )


@pytest.mark.xfail(
    reason=(
        "Matcher does not yet detect cognitive impairment from patient fields "
        "(cognitive_status, moca_score) and apply it against a cognitive exclusion "
        "criterion in free-text trial eligibility. "
        "Implement cognitive status detection before removing xfail."
    ),
    strict=False,
)
def test_documented_cognitive_impairment_blocks_cognitive_trial():
    """Patient with documented cognitive impairment should be blocked from a cognitively demanding trial."""
    patient = _patient(cognitive_status="dementia", moca_score=14)
    trial = _trial(
        "Inclusion: Parkinson disease. Age >= 40. "
        "Exclusion: Dementia or significant cognitive impairment."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] in {"not_eligible", "unclear"}, (
        "Documented cognitive impairment should block or flag eligibility for a trial excluding it"
    )


# ---------------------------------------------------------------------------
# Healthy control
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Matcher does not reliably return not_eligible for healthy controls "
        "against PD-only trials when no comparator arm is present. "
        "Implement stronger default for disease-specific trials before removing xfail."
    ),
    strict=False,
)
def test_healthy_control_is_not_eligible_for_pd_only_trial():
    """Healthy control should be not_eligible for a PD-only trial with no comparator arm."""
    patient = _patient(diagnosis="healthy control", medications=[])
    trial = _trial(
        "Inclusion: Idiopathic Parkinson disease. Age 40 to 80. "
        "Exclusion: Any neurological disorder other than PD."
    )
    result = match_patient_to_trial(patient, trial)
    assert result["prediction"] == "not_eligible", (
        "Healthy control should be not_eligible for a PD-only trial"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_always_has_required_keys():
    """Matcher result should always contain prediction, confidence, and explanation."""
    patient = _patient()
    trial = _trial("Inclusion: Parkinson disease.")
    result = match_patient_to_trial(patient, trial)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert result["prediction"] in {"eligible", "not_eligible", "unclear"}
    assert "confidence" in result
    assert "explanation" in result
