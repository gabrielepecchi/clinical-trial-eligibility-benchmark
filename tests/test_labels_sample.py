"""Unit tests for labels_sample.json."""

import json
from pathlib import Path

import pytest

SAMPLE_FILE = Path("data/processed/labels_sample.json")

VALID_LABELS = {"eligible", "not_eligible", "unclear"}
VALID_PATIENT_IDS = {"P001", "P002", "P003", "P004", "P005"}
VALID_TRIAL_IDS = {"T001", "T002", "T003"}


@pytest.fixture(scope="module")
def labels() -> list[dict]:
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))


def test_exactly_ten_records(labels):
    assert len(labels) == 10


@pytest.mark.parametrize("field", ["patient_id", "trial_id", "label", "rationale", "evidence"])
def test_all_records_have_field(labels, field):
    for record in labels:
        assert field in record


def test_label_values_are_valid(labels):
    for record in labels:
        assert record["label"] in VALID_LABELS


def test_each_label_appears_at_least_twice(labels):
    counts = {label: 0 for label in VALID_LABELS}
    for record in labels:
        counts[record["label"]] += 1
    for label, count in counts.items():
        assert count >= 2, f"Label '{label}' appears only {count} time(s)"


def test_patient_ids_are_valid(labels):
    for record in labels:
        assert record["patient_id"] in VALID_PATIENT_IDS


def test_trial_ids_are_valid(labels):
    for record in labels:
        assert record["trial_id"] in VALID_TRIAL_IDS


def test_rationale_is_not_empty(labels):
    for record in labels:
        assert record["rationale"].strip() != ""


def test_evidence_has_patient_facts(labels):
    for record in labels:
        assert "patient_facts" in record["evidence"]


def test_evidence_has_trial_criteria(labels):
    for record in labels:
        assert "trial_criteria" in record["evidence"]


def test_patient_facts_is_list(labels):
    for record in labels:
        assert isinstance(record["evidence"]["patient_facts"], list)


def test_trial_criteria_is_list(labels):
    for record in labels:
        assert isinstance(record["evidence"]["trial_criteria"], list)
