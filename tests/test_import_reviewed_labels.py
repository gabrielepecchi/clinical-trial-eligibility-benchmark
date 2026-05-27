"""Unit tests for import_reviewed_labels.py."""

import csv
import json

import pytest

from import_reviewed_labels import build_label_record, main, split_pipe_text


def test_split_pipe_text():
    assert split_pipe_text("fact one | fact two") == ["fact one", "fact two"]


def test_split_pipe_text_empty():
    assert split_pipe_text("") == []


def test_build_label_record_valid():
    row = {
        "patient_id": "P001",
        "trial_id": "T001",
        "label": "not_eligible",
        "rationale": "Age outside range.",
        "patient_facts": "patient age 83",
        "trial_criteria": "maximum age 80 Years",
    }

    record = build_label_record(row)

    assert record["patient_id"] == "P001"
    assert record["trial_id"] == "T001"
    assert record["label"] == "not_eligible"
    assert record["rationale"] == "Age outside range."
    assert record["evidence"]["patient_facts"] == ["patient age 83"]
    assert record["evidence"]["trial_criteria"] == ["maximum age 80 Years"]


def test_build_label_record_invalid_label():
    row = {
        "patient_id": "P001",
        "trial_id": "T001",
        "label": "maybe",
        "rationale": "",
        "patient_facts": "",
        "trial_criteria": "",
    }

    with pytest.raises(ValueError):
        build_label_record(row)


def test_main_exports_only_reviewed_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    rows = [
        {
            "patient_id": "P001",
            "trial_id": "T001",
            "trial_category": "device",
            "label": "not_eligible",
            "label_status": "reviewed",
            "patient_summary": "Patient summary",
            "trial_title": "Trial title",
            "rationale": "Reviewed rationale.",
            "patient_facts": "fact",
            "trial_criteria": "criterion",
        },
        {
            "patient_id": "P002",
            "trial_id": "T002",
            "trial_category": "drug_treatment",
            "label": "unclear",
            "label_status": "seed_needs_review",
            "patient_summary": "Patient summary",
            "trial_title": "Trial title",
            "rationale": "Unreviewed rationale.",
            "patient_facts": "fact",
            "trial_criteria": "criterion",
        },
    ]

    input_file = processed / "labels_seed_review.csv"
    with input_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    main()

    output_file = processed / "labels.json"
    labels = json.loads(output_file.read_text(encoding="utf-8"))

    assert len(labels) == 1
    assert labels[0]["patient_id"] == "P001"
    assert labels[0]["label"] == "not_eligible"
