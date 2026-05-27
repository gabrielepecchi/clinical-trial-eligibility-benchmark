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
