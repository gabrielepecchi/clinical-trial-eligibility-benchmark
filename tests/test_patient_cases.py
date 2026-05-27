"""Unit tests for patient_cases.json."""

import json
from pathlib import Path

import pytest

PATIENTS_FILE = Path("data/processed/patient_cases.json")

REQUIRED_FIELDS = {
    "patient_id",
    "summary",
    "age",
    "sex",
    "diagnosis",
    "disease_stage",
    "key_features",
    "exclusions",
    "medications",
    "labs",
    "category_focus",
}

LIST_FIELDS = {"key_features", "exclusions", "medications"}
VALID_PATIENT_IDS = {f"P{i:03d}" for i in range(1, 21)}


@pytest.fixture(scope="module")
def patients() -> list[dict]:
    return json.loads(PATIENTS_FILE.read_text(encoding="utf-8"))


def test_exactly_twenty_patients(patients):
    assert len(patients) == 20


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_patients_have_required_fields(patients, field):
    for patient in patients:
        assert field in patient


def test_patient_ids_are_expected_and_unique(patients):
    ids = [patient["patient_id"] for patient in patients]
    assert set(ids) == VALID_PATIENT_IDS
    assert len(ids) == len(set(ids))


def test_summaries_are_not_empty(patients):
    for patient in patients:
        assert patient["summary"].strip() != ""


def test_age_is_integer(patients):
    for patient in patients:
        assert isinstance(patient["age"], int)


@pytest.mark.parametrize("field", sorted(LIST_FIELDS))
def test_list_fields_are_lists(patients, field):
    for patient in patients:
        assert isinstance(patient[field], list)


def test_labs_is_dictionary(patients):
    for patient in patients:
        assert isinstance(patient["labs"], dict)


def test_no_patient_summary_claims_real_patient_data(patients):
    for patient in patients:
        assert "real patient" not in patient["summary"].lower()
