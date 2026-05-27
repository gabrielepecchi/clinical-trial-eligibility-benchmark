"""Unit tests for labels_llm_reviewed.json."""

import json
from pathlib import Path

import pytest

LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")

REQUIRED_FIELDS = {
    "patient_id",
    "trial_id",
    "label",
    "rationale",
    "evidence",
    "label_status",
}

VALID_LABELS = {"eligible", "not_eligible", "unclear"}
EXPECTED_STATUS = "llm_reviewed_needs_spotcheck"


@pytest.fixture(scope="module")
def labels() -> list[dict]:
    return json.loads(LABELS_FILE.read_text(encoding="utf-8"))


def test_exactly_150_records(labels):
    assert len(labels) == 150


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_records_have_required_fields(labels, field):
    for record in labels:
        assert field in record


def test_labels_are_valid(labels):
    for record in labels:
        assert record["label"] in VALID_LABELS


def test_label_status_is_expected(labels):
    for record in labels:
        assert record["label_status"] == EXPECTED_STATUS


def test_rationale_is_not_empty(labels):
    for record in labels:
        assert record["rationale"].strip() != ""


def test_evidence_has_required_keys(labels):
    for record in labels:
        assert "patient_facts" in record["evidence"]
        assert "trial_criteria" in record["evidence"]


def test_evidence_values_are_lists(labels):
    for record in labels:
        assert isinstance(record["evidence"]["patient_facts"], list)
        assert isinstance(record["evidence"]["trial_criteria"], list)


def test_patient_ids_are_not_empty(labels):
    for record in labels:
        assert record["patient_id"].strip() != ""


def test_trial_ids_are_not_empty(labels):
    for record in labels:
        assert record["trial_id"].strip() != ""


def test_contains_at_least_one_record_per_label(labels):
    present_labels = {record["label"] for record in labels}
    assert VALID_LABELS <= present_labels
