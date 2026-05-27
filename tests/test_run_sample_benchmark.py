"""Unit tests for run_sample_benchmark.py."""

import json

from run_sample_benchmark import load_json


def test_load_json_returns_list(tmp_path):
    data = [{"patient_id": "P001", "label": "eligible"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert isinstance(result, list)


def test_load_json_returns_correct_length(tmp_path):
    data = [
        {"patient_id": "P001", "label": "eligible"},
        {"patient_id": "P002", "label": "not_eligible"},
    ]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert len(result) == 2


def test_load_json_preserves_string_value(tmp_path):
    data = [{"patient_id": "P001", "label": "eligible"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["patient_id"] == "P001"


def test_load_json_preserves_label_value(tmp_path):
    data = [{"patient_id": "P001", "label": "unclear"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["label"] == "unclear"


def test_load_json_preserves_integer_value(tmp_path):
    data = [{"patient_id": "P001", "age": 62}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["age"] == 62


def test_load_json_preserves_list_value(tmp_path):
    data = [{"patient_id": "P001", "diagnosis": ["Parkinson disease"]}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["diagnosis"] == ["Parkinson disease"]


def test_load_json_empty_list(tmp_path):
    file = tmp_path / "empty.json"
    file.write_text(json.dumps([]), encoding="utf-8")

    result = load_json(file)

    assert result == []
