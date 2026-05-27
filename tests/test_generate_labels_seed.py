"""Unit tests for generate_labels_seed.py."""

from generate_labels_seed import (
    check_age,
    check_dbs,
    check_diagnosis_healthy,
    label_candidate,
    parse_age_years,
)


def test_parse_age_years_years():
    assert parse_age_years("60 Years") == 60.0


def test_parse_age_years_months():
    assert parse_age_years("6 Months") == 0.5


def test_check_age_below_minimum():
    patient = {"age": 30}
    trial = {"minimum_age": "40 Years", "maximum_age": "80 Years"}

    result = check_age(patient, trial)

    assert result is not None
    assert result[0] == "not_eligible"


def test_check_age_above_maximum():
    patient = {"age": 90}
    trial = {"minimum_age": "40 Years", "maximum_age": "80 Years"}

    result = check_age(patient, trial)

    assert result is not None
    assert result[0] == "not_eligible"


def test_check_dbs_exclusion():
    patient = {
        "summary": "Prior DBS implant",
        "key_features": ["deep brain stimulation implanted"],
        "exclusions": ["DBS implant present"],
    }
    trial = {
        "exclusion_criteria": ["Prior deep brain stimulation is excluded"],
    }

    result = check_dbs(patient, trial)

    assert result is not None
    assert result[0] == "not_eligible"


def test_check_healthy_control_when_pd_required():
    patient = {
        "diagnosis": "healthy control",
        "summary": "No Parkinson disease diagnosis",
    }
    trial = {
        "inclusion_criteria": ["Diagnosis of Parkinson disease required"],
    }

    result = check_diagnosis_healthy(patient, trial)

    assert result is not None
    assert result[0] == "not_eligible"


def test_label_candidate_default_unclear():
    candidate = {"patient_id": "P001", "trial_id": "T001"}
    patients = {"P001": {"patient_id": "P001", "age": 60, "diagnosis": "Parkinson disease"}}
    trials = {"T001": {"trial_id": "T001", "inclusion_criteria": [], "exclusion_criteria": []}}

    result = label_candidate(candidate, patients, trials)

    assert result["label"] == "unclear"
    assert result["label_status"] == "seed_needs_review"
    assert isinstance(result["evidence"]["patient_facts"], list)
    assert isinstance(result["evidence"]["trial_criteria"], list)
