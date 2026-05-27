"""Unit tests for export_labels_seed_review.py."""

import csv
import json

from export_labels_seed_review import build_review_row, join_list, main


def test_join_list_with_list():
    assert join_list(["a", "b"]) == "a | b"


def test_join_list_with_empty_value():
    assert join_list(None) == ""


def test_build_review_row():
    label = {
        "patient_id": "P001",
        "trial_id": "T001",
        "label": "unclear",
        "label_status": "seed_needs_review",
        "rationale": "Needs review.",
        "evidence": {
            "patient_facts": ["fact one"],
            "trial_criteria": ["criterion one"],
        },
    }
    patients = {"P001": {"summary": "Synthetic patient summary."}}
    trials = {"T001": {"category": "rehabilitation", "title": "Trial title"}}

    row = build_review_row(label, patients, trials)

    assert row["patient_id"] == "P001"
    assert row["trial_id"] == "T001"
    assert row["trial_category"] == "rehabilitation"
    assert row["patient_summary"] == "Synthetic patient summary."
    assert row["trial_title"] == "Trial title"
    assert row["patient_facts"] == "fact one"
    assert row["trial_criteria"] == "criterion one"


def test_main_writes_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    labels = [
        {
            "patient_id": "P001",
            "trial_id": "T001",
            "label": "unclear",
            "label_status": "seed_needs_review",
            "rationale": "Needs review.",
            "evidence": {"patient_facts": ["fact"], "trial_criteria": ["criterion"]},
        }
    ]
    patients = [{"patient_id": "P001", "summary": "Patient summary"}]
    trials = [{"trial_id": "T001", "category": "device", "title": "Trial title"}]

    (processed / "labels_seed.json").write_text(json.dumps(labels), encoding="utf-8")
    (processed / "patient_cases.json").write_text(json.dumps(patients), encoding="utf-8")
    (processed / "trial_cases.json").write_text(json.dumps(trials), encoding="utf-8")

    main()

    output_file = processed / "labels_seed_review.csv"
    assert output_file.exists()

    rows = list(csv.DictReader(output_file.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "P001"
    assert rows[0]["trial_id"] == "T001"
    assert rows[0]["trial_category"] == "device"
