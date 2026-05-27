"""Unit tests for trial_cases_sample.json."""

import json
from pathlib import Path

import pytest

SAMPLE_FILE = Path("data/processed/trial_cases_sample.json")

REQUIRED_FIELDS = {
    "trial_id",
    "nct_id",
    "title",
    "category",
    "inclusion_criteria",
    "exclusion_criteria",
    "raw_eligibility",
    "url",
}

VALID_CATEGORIES = {
    "drug_treatment",
    "device",
    "rehabilitation",
    "non_motor_symptoms",
    "advanced_therapy",
    "biomarker",
}


@pytest.fixture(scope="module")
def trials() -> list[dict]:
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))


def test_exactly_three_records(trials):
    assert len(trials) == 3


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_records_have_field(trials, field):
    for trial in trials:
        assert field in trial


def test_trial_ids_are_unique(trials):
    ids = [t["trial_id"] for t in trials]
    assert len(ids) == len(set(ids))


def test_nct_ids_start_with_NCT(trials):
    for trial in trials:
        assert trial["nct_id"].startswith("NCT")


def test_categories_are_valid(trials):
    for trial in trials:
        assert trial["category"] in VALID_CATEGORIES


def test_inclusion_criteria_are_lists(trials):
    for trial in trials:
        assert isinstance(trial["inclusion_criteria"], list)


def test_exclusion_criteria_are_lists(trials):
    for trial in trials:
        assert isinstance(trial["exclusion_criteria"], list)
