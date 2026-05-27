"""Unit tests for patient_cases_sample.json."""

import json
from pathlib import Path

import pytest

SAMPLE_FILE = Path("data/processed/patient_cases_sample.json")

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
    "category_focus",
}

LIST_FIELDS = {"diagnosis", "key_features", "exclusions", "medications"}


@pytest.fixture(scope="module")
def patients() -> list[dict]:
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))


def test_exactly_five_records(patients):
    assert len(patients) == 5


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_records_have_field(patients, field):
    for patient in patients:
        assert field in patient


def test_patient_ids_are_unique(patients):
    ids = [p["patient_id"] for p in patients]
    assert len(ids) == len(set(ids))


def test_patient_ids_start_with_P(patients):
    for patient in patients:
        assert patient["patient_id"].startswith("P")


def test_age_is_integer(patients):
    for patient in patients:
        assert isinstance(patient["age"], int)


@pytest.mark.parametrize("field", sorted(LIST_FIELDS))
def test_list_fields_are_lists(patients, field):
    for patient in patients:
        assert isinstance(patient[field], list)


def test_summary_is_not_empty(patients):
    for patient in patients:
        assert patient["summary"].strip() != ""
