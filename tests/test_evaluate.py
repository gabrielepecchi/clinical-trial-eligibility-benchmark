"""Unit tests for evaluate.py."""

import pytest

from evaluate import compute_metrics

LABELS = ["eligible", "not_eligible", "unclear"]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect():
    gold = ["eligible", "not_eligible", "unclear"]
    pred = ["eligible", "not_eligible", "unclear"]
    return compute_metrics(gold, pred)


@pytest.fixture
def partial():
    gold = ["eligible", "eligible", "not_eligible", "unclear"]
    pred = ["eligible", "not_eligible", "not_eligible", "unclear"]
    return compute_metrics(gold, pred)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_perfect_accuracy(perfect):
    assert perfect["accuracy"] == 1.0


def test_partial_accuracy(partial):
    # 3 out of 4 correct
    assert partial["accuracy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Top-level keys
# ---------------------------------------------------------------------------

def test_macro_precision_key_exists(perfect):
    assert "macro_precision" in perfect


def test_macro_recall_key_exists(perfect):
    assert "macro_recall" in perfect


def test_macro_f1_key_exists(perfect):
    assert "macro_f1" in perfect


# ---------------------------------------------------------------------------
# per_class structure
# ---------------------------------------------------------------------------

def test_per_class_contains_eligible(perfect):
    assert "eligible" in perfect["per_class"]


def test_per_class_contains_not_eligible(perfect):
    assert "not_eligible" in perfect["per_class"]


def test_per_class_contains_unclear(perfect):
    assert "unclear" in perfect["per_class"]


def test_per_class_has_exactly_three_labels(perfect):
    assert set(perfect["per_class"].keys()) == {"eligible", "not_eligible", "unclear"}


@pytest.mark.parametrize("label", LABELS)
def test_per_class_label_has_precision(perfect, label):
    assert "precision" in perfect["per_class"][label]


@pytest.mark.parametrize("label", LABELS)
def test_per_class_label_has_recall(perfect, label):
    assert "recall" in perfect["per_class"][label]


@pytest.mark.parametrize("label", LABELS)
def test_per_class_label_has_f1(perfect, label):
    assert "f1" in perfect["per_class"][label]


# ---------------------------------------------------------------------------
# per_class values — perfect predictions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", LABELS)
def test_per_class_perfect_precision(perfect, label):
    assert perfect["per_class"][label]["precision"] == pytest.approx(1.0)


@pytest.mark.parametrize("label", LABELS)
def test_per_class_perfect_recall(perfect, label):
    assert perfect["per_class"][label]["recall"] == pytest.approx(1.0)


@pytest.mark.parametrize("label", LABELS)
def test_per_class_perfect_f1(perfect, label):
    assert perfect["per_class"][label]["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_correct_count(perfect):
    # Each gold label predicted correctly — diagonal counts should be 1
    for label in LABELS:
        assert perfect["confusion_matrix"][label][label] == 1


def test_confusion_matrix_off_diagonal_zero(perfect):
    for gold in LABELS:
        for pred in LABELS:
            if gold != pred:
                assert perfect["confusion_matrix"][gold][pred] == 0


def test_confusion_matrix_records_misprediction(partial):
    # gold=eligible, pred=not_eligible should be 1
    assert partial["confusion_matrix"]["eligible"]["not_eligible"] == 1


def test_confusion_matrix_gold_rows_exist(perfect):
    for label in LABELS:
        assert label in perfect["confusion_matrix"]


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_input_no_crash():
    result = compute_metrics([], [])
    assert result["accuracy"] == 0.0


def test_empty_input_macro_precision_zero():
    result = compute_metrics([], [])
    assert result["macro_precision"] == 0.0


def test_empty_input_macro_recall_zero():
    result = compute_metrics([], [])
    assert result["macro_recall"] == 0.0


def test_empty_input_macro_f1_zero():
    result = compute_metrics([], [])
    assert result["macro_f1"] == 0.0


def test_empty_input_per_class_has_all_labels():
    result = compute_metrics([], [])
    assert set(result["per_class"].keys()) == {"eligible", "not_eligible", "unclear"}


# ---------------------------------------------------------------------------
# Mismatched lengths
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises_value_error():
    with pytest.raises(ValueError):
        compute_metrics(["eligible", "unclear"], ["eligible"])


def test_mismatched_lengths_gold_longer_raises():
    with pytest.raises(ValueError):
        compute_metrics(["eligible", "not_eligible", "unclear"], ["eligible"])


def test_mismatched_lengths_pred_longer_raises():
    with pytest.raises(ValueError):
        compute_metrics(["eligible"], ["eligible", "not_eligible"])


# ---------------------------------------------------------------------------
# Missing-policy baseline helpers
# ---------------------------------------------------------------------------

from baselines import (
    predict_missing_policy,
    predict_baseline,
    _has_structured_missingness,
    _has_blocking_evidence,
    extract_gold_labels,
)


def test_strict_missing_unclear_with_unknown_fields():
    """strict_missing_unclear → unclear when unknown_fields is non-empty."""
    record = {"unknown_fields": ["cognitive_score"], "blocking_criteria": []}
    assert predict_missing_policy(record, "strict_missing_unclear") == "unclear"


def test_strict_missing_unclear_with_missing_information():
    """strict_missing_unclear → unclear when missing_information is non-empty."""
    record = {"missing_information": ["medication_details"], "unknown_fields": []}
    assert predict_missing_policy(record, "strict_missing_unclear") == "unclear"


def test_strict_missing_unclear_with_uncertain_criteria():
    """strict_missing_unclear → unclear when uncertain_criteria is non-empty."""
    record = {"uncertain_criteria": ["medication stability unclear"], "unknown_fields": []}
    assert predict_missing_policy(record, "strict_missing_unclear") == "unclear"


def test_strict_missing_unclear_with_missing_reason_type():
    """strict_missing_unclear → unclear when missing_reason_type is set."""
    record = {"missing_reason_type": "not_documented", "unknown_fields": []}
    assert predict_missing_policy(record, "strict_missing_unclear") == "unclear"


def test_strict_missing_unclear_with_unknown_detail_status():
    """strict_missing_unclear → unclear when missing_information_details has unknown status."""
    record = {
        "missing_information_details": [{"field": "age", "status": "unknown"}],
        "unknown_fields": [],
    }
    assert predict_missing_policy(record, "strict_missing_unclear") == "unclear"


def test_strict_missing_unclear_no_missingness_returns_eligible():
    """strict_missing_unclear → eligible when no missingness signals present."""
    record = {"unknown_fields": [], "missing_information": [], "uncertain_criteria": []}
    assert predict_missing_policy(record, "strict_missing_unclear") == "eligible"


def test_optimistic_missing_eligible_no_blocking():
    """optimistic_missing_eligible → eligible when missingness exists but no blocking criteria."""
    record = {
        "unknown_fields": ["cognitive_score"],
        "blocking_criteria": [],
        "blocked_by": None,
    }
    assert predict_missing_policy(record, "optimistic_missing_eligible") == "eligible"


def test_optimistic_missing_eligible_with_blocking():
    """optimistic_missing_eligible → not_eligible when blocking_criteria is non-empty."""
    record = {"blocking_criteria": ["DBS implant present"], "unknown_fields": ["age"]}
    assert predict_missing_policy(record, "optimistic_missing_eligible") == "not_eligible"


def test_optimistic_missing_eligible_blocked_by():
    """optimistic_missing_eligible → not_eligible when blocked_by is set."""
    record = {"blocked_by": "age out of range", "unknown_fields": []}
    assert predict_missing_policy(record, "optimistic_missing_eligible") == "not_eligible"


def test_conservative_not_eligible_when_blocking():
    """conservative_missing_unclear_or_not_eligible → not_eligible when blocking_criteria present."""
    record = {
        "blocking_criteria": ["patient age 45 out of range"],
        "unknown_fields": ["medication_details"],
    }
    assert predict_missing_policy(record, "conservative_missing_unclear_or_not_eligible") == "not_eligible"


def test_conservative_unclear_when_missing_no_blocking():
    """conservative_missing_unclear_or_not_eligible → unclear when missingness present but no blocking."""
    record = {
        "blocking_criteria": [],
        "unknown_fields": ["disease_stage"],
    }
    assert predict_missing_policy(record, "conservative_missing_unclear_or_not_eligible") == "unclear"


def test_conservative_eligible_when_no_missingness_no_blocking():
    """conservative_missing_unclear_or_not_eligible → eligible when nothing missing and no blocking."""
    record = {
        "blocking_criteria": [],
        "unknown_fields": [],
        "missing_information": [],
        "uncertain_criteria": [],
        "missing_reason_type": "",
        "missing_information_details": [],
    }
    assert predict_missing_policy(record, "conservative_missing_unclear_or_not_eligible") == "eligible"


def test_has_structured_missingness_true_for_unknown_fields():
    assert _has_structured_missingness({"unknown_fields": ["age"]}) is True


def test_has_structured_missingness_false_for_empty():
    assert _has_structured_missingness({
        "unknown_fields": [],
        "missing_information": [],
        "missing_information_details": [],
        "uncertain_criteria": [],
        "missing_reason_type": "",
    }) is False


def test_has_blocking_evidence_true():
    assert _has_blocking_evidence({"blocking_criteria": ["age out of range"]}) is True


def test_has_blocking_evidence_false():
    assert _has_blocking_evidence({"blocking_criteria": [], "blocked_by": None}) is False


def test_existing_always_unclear_still_works():
    """Existing always_unclear strategy unchanged."""
    labels = [{"label": "eligible"}, {"label": "not_eligible"}]
    result = predict_baseline(labels, "always_unclear")
    assert result == ["unclear", "unclear"]


def test_existing_majority_class_still_works():
    """Existing majority_class strategy unchanged."""
    labels = [{"label": "eligible"}, {"label": "eligible"}, {"label": "not_eligible"}]
    result = predict_baseline(labels, "majority_class")
    assert result == ["eligible", "eligible", "eligible"]


def test_missing_policy_fallback_when_no_prediction_records():
    """Missing-policy strategy falls back to eligible list when no prediction_records provided."""
    labels = [{"label": "eligible"}, {"label": "unclear"}]
    result = predict_baseline(labels, "strict_missing_unclear", prediction_records=None)
    assert result == ["eligible", "eligible"]


def test_missing_policy_uses_prediction_records_when_provided():
    """Missing-policy strategy uses structured records when provided."""
    labels = [{"label": "eligible"}, {"label": "unclear"}]
    pred_records = [
        {"unknown_fields": ["age"], "blocking_criteria": []},
        {"unknown_fields": [], "blocking_criteria": [], "missing_information": []},
    ]
    result = predict_baseline(labels, "strict_missing_unclear", prediction_records=pred_records)
    assert result[0] == "unclear"
    assert result[1] == "eligible"
