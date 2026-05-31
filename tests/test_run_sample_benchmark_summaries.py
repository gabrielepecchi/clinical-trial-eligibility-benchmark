"""Tests for benchmark summary helpers in run_sample_benchmark.py."""

from run_sample_benchmark import (
    build_coverage_summary,
    format_coverage_summary,
    build_label_distribution,
    format_label_distribution,
    build_confusion_matrix,
    format_confusion_matrix,
    build_benchmark_metadata,
    format_benchmark_metadata,
)


# ---------------------------------------------------------------------------
# Shared helpers
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


# ---------------------------------------------------------------------------
# Coverage summary tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Coverage summary formatting tests
# ---------------------------------------------------------------------------

_SAMPLE_COVERAGE = {
    "total_predictions": 10,
    "with_missing_information": 4,
    "with_criterion_results": 7,
}


def test_format_coverage_summary_returns_string():
    assert isinstance(format_coverage_summary(_SAMPLE_COVERAGE), str)


def test_format_coverage_summary_contains_header():
    assert "Coverage" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_total_predictions_label():
    assert "Total predictions" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_with_missing_information_label():
    assert "With missing information" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_with_criterion_results_label():
    assert "With criterion results" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_total_value():
    assert "10" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_missing_information_value():
    assert "4" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_criterion_results_value():
    assert "7" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_all_zeros():
    coverage = {"total_predictions": 0, "with_missing_information": 0, "with_criterion_results": 0}
    result = format_coverage_summary(coverage)
    assert isinstance(result, str)
    assert "0" in result


def test_format_coverage_summary_contains_missing_information_percentage():
    assert "40.0%" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_contains_criterion_results_percentage():
    assert "70.0%" in format_coverage_summary(_SAMPLE_COVERAGE)


def test_format_coverage_summary_all_zeros_does_not_crash():
    coverage = {"total_predictions": 0, "with_missing_information": 0, "with_criterion_results": 0}
    result = format_coverage_summary(coverage)
    assert "0.0%" in result


# ---------------------------------------------------------------------------
# Label distribution tests
# ---------------------------------------------------------------------------

_GOLD = ["eligible", "eligible", "not_eligible", "unclear"]
_PREDICTED = ["eligible", "not_eligible", "not_eligible", "unclear"]
_LABEL_SET = {"eligible", "not_eligible", "unclear"}


def test_build_label_distribution_returns_dict():
    assert isinstance(build_label_distribution(_GOLD, _PREDICTED), dict)


def test_build_label_distribution_has_gold_key():
    assert "gold" in build_label_distribution(_GOLD, _PREDICTED)


def test_build_label_distribution_has_predicted_key():
    assert "predicted" in build_label_distribution(_GOLD, _PREDICTED)


def test_build_label_distribution_gold_all_labels_present():
    result = build_label_distribution(_GOLD, _PREDICTED)
    assert _LABEL_SET <= set(result["gold"])


def test_build_label_distribution_predicted_all_labels_present():
    result = build_label_distribution(_GOLD, _PREDICTED)
    assert _LABEL_SET <= set(result["predicted"])


def test_build_label_distribution_gold_counts():
    result = build_label_distribution(_GOLD, _PREDICTED)
    assert result["gold"]["eligible"] == 2
    assert result["gold"]["not_eligible"] == 1
    assert result["gold"]["unclear"] == 1


def test_build_label_distribution_predicted_counts():
    result = build_label_distribution(_GOLD, _PREDICTED)
    assert result["predicted"]["eligible"] == 1
    assert result["predicted"]["not_eligible"] == 2
    assert result["predicted"]["unclear"] == 1


def test_build_label_distribution_missing_label_is_zero():
    result = build_label_distribution(["eligible"], ["eligible"])
    assert result["gold"]["not_eligible"] == 0
    assert result["gold"]["unclear"] == 0


def test_build_label_distribution_empty_inputs():
    result = build_label_distribution([], [])
    assert result["gold"] == {"eligible": 0, "not_eligible": 0, "unclear": 0}
    assert result["predicted"] == {"eligible": 0, "not_eligible": 0, "unclear": 0}


# ---------------------------------------------------------------------------
# Label distribution formatting tests
# ---------------------------------------------------------------------------

_SAMPLE_DISTRIBUTION = {
    "gold": {"eligible": 3, "not_eligible": 2, "unclear": 1},
    "predicted": {"eligible": 2, "not_eligible": 3, "unclear": 1},
}

_ZERO_DISTRIBUTION = {
    "gold": {"eligible": 0, "not_eligible": 0, "unclear": 0},
    "predicted": {"eligible": 0, "not_eligible": 0, "unclear": 0},
}


def test_format_label_distribution_returns_string():
    assert isinstance(format_label_distribution(_SAMPLE_DISTRIBUTION), str)


def test_format_label_distribution_contains_header():
    assert "Label distribution" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_gold():
    assert "Gold" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_predicted():
    assert "Predicted" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_eligible():
    assert "eligible" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_not_eligible():
    assert "not_eligible" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_unclear():
    assert "unclear" in format_label_distribution(_SAMPLE_DISTRIBUTION)


def test_format_label_distribution_contains_gold_counts():
    result = format_label_distribution(_SAMPLE_DISTRIBUTION)
    assert "3" in result
    assert "2" in result
    assert "1" in result


def test_format_label_distribution_all_zeros_does_not_crash():
    result = format_label_distribution(_ZERO_DISTRIBUTION)
    assert isinstance(result, str)
    assert "0" in result


# ---------------------------------------------------------------------------
# Confusion matrix tests
# ---------------------------------------------------------------------------

_LABEL_LIST = ["eligible", "not_eligible", "unclear"]

_CM_GOLD = ["eligible", "eligible", "not_eligible", "unclear"]
_CM_PRED = ["eligible", "not_eligible", "not_eligible", "unclear"]


def test_build_confusion_matrix_returns_dict():
    assert isinstance(build_confusion_matrix(_CM_GOLD, _CM_PRED), dict)


def test_build_confusion_matrix_top_level_keys():
    result = build_confusion_matrix(_CM_GOLD, _CM_PRED)
    assert set(_LABEL_LIST) <= set(result)


def test_build_confusion_matrix_inner_keys():
    result = build_confusion_matrix(_CM_GOLD, _CM_PRED)
    for label in _LABEL_LIST:
        assert set(_LABEL_LIST) <= set(result[label])


def test_build_confusion_matrix_correct_counts():
    result = build_confusion_matrix(_CM_GOLD, _CM_PRED)
    assert result["eligible"]["eligible"] == 1
    assert result["eligible"]["not_eligible"] == 1
    assert result["not_eligible"]["not_eligible"] == 1
    assert result["unclear"]["unclear"] == 1


def test_build_confusion_matrix_missing_combinations_are_zero():
    result = build_confusion_matrix(_CM_GOLD, _CM_PRED)
    assert result["eligible"]["unclear"] == 0
    assert result["not_eligible"]["eligible"] == 0
    assert result["unclear"]["eligible"] == 0


def test_build_confusion_matrix_empty_inputs():
    result = build_confusion_matrix([], [])
    for gold_label in _LABEL_LIST:
        for pred_label in _LABEL_LIST:
            assert result[gold_label][pred_label] == 0


# ---------------------------------------------------------------------------
# Confusion matrix formatting tests
# ---------------------------------------------------------------------------

_SAMPLE_CM = {
    "eligible":     {"eligible": 3, "not_eligible": 1, "unclear": 0},
    "not_eligible": {"eligible": 0, "not_eligible": 4, "unclear": 1},
    "unclear":      {"eligible": 0, "not_eligible": 0, "unclear": 2},
}

_ZERO_CM = {
    "eligible":     {"eligible": 0, "not_eligible": 0, "unclear": 0},
    "not_eligible": {"eligible": 0, "not_eligible": 0, "unclear": 0},
    "unclear":      {"eligible": 0, "not_eligible": 0, "unclear": 0},
}


def test_format_confusion_matrix_returns_string():
    assert isinstance(format_confusion_matrix(_SAMPLE_CM), str)


def test_format_confusion_matrix_contains_header():
    assert "Confusion matrix" in format_confusion_matrix(_SAMPLE_CM)


def test_format_confusion_matrix_contains_axis_label():
    assert "Gold \\ Predicted" in format_confusion_matrix(_SAMPLE_CM)


def test_format_confusion_matrix_contains_eligible():
    assert "eligible" in format_confusion_matrix(_SAMPLE_CM)


def test_format_confusion_matrix_contains_not_eligible():
    assert "not_eligible" in format_confusion_matrix(_SAMPLE_CM)


def test_format_confusion_matrix_contains_unclear():
    assert "unclear" in format_confusion_matrix(_SAMPLE_CM)


def test_format_confusion_matrix_contains_counts():
    result = format_confusion_matrix(_SAMPLE_CM)
    assert "3" in result
    assert "4" in result
    assert "2" in result


def test_format_confusion_matrix_all_zeros_does_not_crash():
    result = format_confusion_matrix(_ZERO_CM)
    assert isinstance(result, str)
    assert "0" in result


# ---------------------------------------------------------------------------
# Benchmark metadata tests
# ---------------------------------------------------------------------------

_PATIENTS = [{"patient_id": "P001"}, {"patient_id": "P002"}]
_TRIALS = [{"trial_id": "T001"}]
_LABELS = [{"patient_id": "P001", "trial_id": "T001", "label": "eligible"}]
_PRED_RECORDS = [{"patient_id": "P001", "trial_id": "T001", "predicted_label": "eligible"}]


def test_build_benchmark_metadata_returns_dict():
    assert isinstance(build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS), dict)


def test_build_benchmark_metadata_benchmark_name():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert result["benchmark_name"] == "sample_benchmark"


def test_build_benchmark_metadata_num_patients():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert result["num_patients"] == 2


def test_build_benchmark_metadata_num_trials():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert result["num_trials"] == 1


def test_build_benchmark_metadata_num_label_records():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert result["num_label_records"] == 1


def test_build_benchmark_metadata_num_evaluated_pairs():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert result["num_evaluated_pairs"] == 1


def test_build_benchmark_metadata_all_keys_present():
    result = build_benchmark_metadata(_PATIENTS, _TRIALS, _LABELS, _PRED_RECORDS)
    assert {"benchmark_name", "num_patients", "num_trials", "num_label_records", "num_evaluated_pairs"} <= set(result)


def test_build_benchmark_metadata_empty_inputs():
    result = build_benchmark_metadata([], [], [], [])
    assert result["benchmark_name"] == "sample_benchmark"
    assert result["num_patients"] == 0
    assert result["num_trials"] == 0
    assert result["num_label_records"] == 0
    assert result["num_evaluated_pairs"] == 0


# ---------------------------------------------------------------------------
# Benchmark metadata formatting tests
# ---------------------------------------------------------------------------

_SAMPLE_METADATA = {
    "benchmark_name": "sample_benchmark",
    "num_patients": 50,
    "num_trials": 10,
    "num_label_records": 200,
    "num_evaluated_pairs": 195,
}

_ZERO_METADATA = {
    "benchmark_name": "sample_benchmark",
    "num_patients": 0,
    "num_trials": 0,
    "num_label_records": 0,
    "num_evaluated_pairs": 0,
}


def test_format_benchmark_metadata_returns_string():
    assert isinstance(format_benchmark_metadata(_SAMPLE_METADATA), str)


def test_format_benchmark_metadata_contains_header():
    assert "Benchmark metadata" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_benchmark_name():
    assert "sample_benchmark" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_patients_label():
    assert "Patients" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_trials_label():
    assert "Trials" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_label_records_label():
    assert "Label records" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_evaluated_pairs_label():
    assert "Evaluated pairs" in format_benchmark_metadata(_SAMPLE_METADATA)


def test_format_benchmark_metadata_contains_numeric_values():
    result = format_benchmark_metadata(_SAMPLE_METADATA)
    assert "50" in result
    assert "10" in result
    assert "200" in result
    assert "195" in result


def test_format_benchmark_metadata_all_zeros_does_not_crash():
    result = format_benchmark_metadata(_ZERO_METADATA)
    assert isinstance(result, str)
    assert "0" in result
