"""Unit tests for models.py."""

import pytest
from pydantic import ValidationError

from models import EligibilityLabel, EligibilityLabelValue, Patient, Trial


def test_trial_basic():
    trial = Trial(nct_id="NCT00000001", title="Test Trial")
    assert trial.nct_id == "NCT00000001"
    assert trial.title == "Test Trial"
    assert trial.eligibility_text is None
    assert trial.inclusion_criteria == []
    assert trial.exclusion_criteria == []


def test_trial_full():
    trial = Trial(
        nct_id="NCT00000002",
        title="Full Trial",
        eligibility_text="Age 18-80, Parkinson diagnosis required.",
        inclusion_criteria=["Age 18-80", "Parkinson diagnosis"],
        exclusion_criteria=["Prior DBS surgery"],
    )
    assert trial.eligibility_text == "Age 18-80, Parkinson diagnosis required."
    assert len(trial.inclusion_criteria) == 2
    assert len(trial.exclusion_criteria) == 1


def test_patient_basic():
    patient = Patient(patient_id="P001", age=65, sex="male")
    assert patient.patient_id == "P001"
    assert patient.age == 65
    assert patient.sex == "male"
    assert patient.diagnosis == []
    assert patient.comorbidities == []
    assert patient.medications == []
    assert patient.labs == {}


def test_patient_full():
    patient = Patient(
        patient_id="P002",
        age=72,
        sex="female",
        diagnosis=["Parkinson disease"],
        comorbidities=["hypertension"],
        medications=["levodopa"],
        labs={"hba1c": 5.4},
    )
    assert "Parkinson disease" in patient.diagnosis
    assert "hypertension" in patient.comorbidities
    assert "levodopa" in patient.medications
    assert patient.labs["hba1c"] == 5.4


@pytest.mark.parametrize("label", ["eligible", "not_eligible", "unclear"])
def test_eligibility_label_valid(label):
    el = EligibilityLabel(trial_id="NCT00000001", patient_id="P001", label=label)
    assert el.label == EligibilityLabelValue(label)


def test_eligibility_label_with_notes():
    el = EligibilityLabel(
        trial_id="NCT00000001",
        patient_id="P001",
        label="unclear",
        notes="Missing lab values.",
    )
    assert el.notes == "Missing lab values."


def test_eligibility_label_invalid():
    with pytest.raises(ValidationError):
        EligibilityLabel(trial_id="NCT00000001", patient_id="P001", label="maybe")
