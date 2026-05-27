"""Unit tests for error_analysis_sample.json."""

import json
from pathlib import Path

import pytest

SAMPLE_FILE = Path("data/processed/error_analysis_sample.json")

REQUIRED_FIELDS = {
    "case_id",
    "patient_id",
    "trial_id",
    "gold_label",
    "predicted_label",
    "error_type",
    "explanation",
    "possible_fix",
}

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

VALID_PATIENT_IDS = {"P001", "P002", "P003", "P004", "P005"}

VALID_TRIAL_IDS = {"T001", "T002", "T003"}

VALID_ERROR_TYPES = {
    "missed_exclusion",
    "age_rule_failure",
    "synonym_mismatch",
    "insufficient_patient_detail",
    "overmatched_keyword",
    "criterion_requires_clinical_judgment",
    "negation_error",
    "ambiguous_trial_criteria",
}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def test_exactly_five_records(records):
    assert len(records) == 5


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_records_have_field(records, field):
    for record in records:
        assert field in record


# ---------------------------------------------------------------------------
# case_id
# ---------------------------------------------------------------------------

def test_case_ids_are_unique(records):
    ids = [r["case_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_case_ids_start_with_err(records):
    for record in records:
        assert record["case_id"].startswith("ERR")


# ---------------------------------------------------------------------------
# Label values
# ---------------------------------------------------------------------------

def test_gold_label_values_are_valid(records):
    for record in records:
        assert record["gold_label"] in VALID_LABELS


def test_predicted_label_values_are_valid(records):
    for record in records:
        assert record["predicted_label"] in VALID_LABELS


# ---------------------------------------------------------------------------
# patient_id and trial_id
# ---------------------------------------------------------------------------

def test_patient_ids_are_valid(records):
    for record in records:
        assert record["patient_id"] in VALID_PATIENT_IDS


def test_trial_ids_are_valid(records):
    for record in records:
        assert record["trial_id"] in VALID_TRIAL_IDS


# ---------------------------------------------------------------------------
# error_type
# ---------------------------------------------------------------------------

def test_error_types_are_valid(records):
    for record in records:
        assert record["error_type"] in VALID_ERROR_TYPES


# ---------------------------------------------------------------------------
# explanation and possible_fix
# ---------------------------------------------------------------------------

def test_explanation_is_not_empty(records):
    for record in records:
        assert record["explanation"].strip() != ""


def test_possible_fix_is_not_empty(records):
    for record in records:
        assert record["possible_fix"].strip() != ""
