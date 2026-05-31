"""Unit tests for run_sample_benchmark.py."""

import json

from run_sample_benchmark import load_json, build_coverage_summary


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


# ---------------------------------------------------------------------------
# Coverage summary tests
# ---------------------------------------------------------------------------

def _make_prediction(missing_information=None, criterion_results=None) -> dict:
    return {
        "patient_id": "P001",
        "trial_id": "T001",
        "gold_label": "eligible",
        "predicted_label": "eligible",
        "missing_information": missing_information or [],
        "criterion_results": criterion_results or [],
    }


def test_build_coverage_summary_returns_dict():
    assert isinstance(build_coverage_summary([_make_prediction()]), dict)


def test_build_coverage_summary_all_fields_present():
    result = build_coverage_summary([_make_prediction()])
    assert {"total_predictions", "with_missing_information", "with_criterion_results"} <= set(result)


def test_build_coverage_summary_total_predictions():
    predictions = [_make_prediction(), _make_prediction(), _make_prediction()]
    assert build_coverage_summary(predictions)["total_predictions"] == 3


def test_build_coverage_summary_empty_list():
    result = build_coverage_summary([])
    assert result["total_predictions"] == 0
    assert result["with_missing_information"] == 0
    assert result["with_criterion_results"] == 0


def test_build_coverage_summary_with_missing_information_counts_non_empty():
    predictions = [
        _make_prediction(missing_information=["age"]),
        _make_prediction(missing_information=[]),
        _make_prediction(missing_information=["cognitive_score", "medication_details"]),
    ]
    assert build_coverage_summary(predictions)["with_missing_information"] == 2


def test_build_coverage_summary_with_missing_information_all_empty():
    predictions = [_make_prediction(), _make_prediction()]
    assert build_coverage_summary(predictions)["with_missing_information"] == 0


def test_build_coverage_summary_with_criterion_results_counts_non_empty():
    predictions = [
        _make_prediction(criterion_results=[{"criterion_text": "Age 40 to 80 years"}]),
        _make_prediction(criterion_results=[]),
    ]
    assert build_coverage_summary(predictions)["with_criterion_results"] == 1


def test_build_coverage_summary_with_criterion_results_all_empty():
    predictions = [_make_prediction(), _make_prediction()]
    assert build_coverage_summary(predictions)["with_criterion_results"] == 0
