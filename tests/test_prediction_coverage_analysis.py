"""
tests/test_prediction_coverage_analysis.py

Unit tests for eval/run_prediction_coverage_analysis.py pure functions.
No real data files are required.
"""

import pytest

from eval.run_prediction_coverage_analysis import (
    analyze_prediction_coverage,
    count_pairs,
    extract_records,
    pair_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rec(patient_id, trial_id, label="eligible"):
    return {"patient_id": patient_id, "trial_id": trial_id, "label": label}


# ---------------------------------------------------------------------------
# pair_key
# ---------------------------------------------------------------------------


def test_pair_key_returns_tuple():
    assert pair_key(rec("p1", "t1")) == ("p1", "t1")


def test_pair_key_stringifies():
    assert pair_key({"patient_id": 1, "trial_id": 2, "label": "eligible"}) == ("1", "2")


def test_pair_key_missing_patient_id():
    assert pair_key({"trial_id": "t1", "label": "eligible"}) is None


def test_pair_key_missing_trial_id():
    assert pair_key({"patient_id": "p1", "label": "eligible"}) is None


def test_pair_key_empty_strings():
    assert pair_key({"patient_id": "", "trial_id": "t1"}) is None
    assert pair_key({"patient_id": "p1", "trial_id": ""}) is None


def test_pair_key_empty_dict():
    assert pair_key({}) is None


# ---------------------------------------------------------------------------
# count_pairs
# ---------------------------------------------------------------------------


def test_count_pairs_no_duplicates():
    records = [rec("p1", "t1"), rec("p2", "t2"), rec("p3", "t3")]
    counts = count_pairs(records)
    assert counts == {("p1", "t1"): 1, ("p2", "t2"): 1, ("p3", "t3"): 1}


def test_count_pairs_detects_duplicates():
    records = [rec("p1", "t1"), rec("p1", "t1"), rec("p2", "t2")]
    counts = count_pairs(records)
    assert counts[("p1", "t1")] == 2
    assert counts[("p2", "t2")] == 1


def test_count_pairs_skips_incomplete_records():
    records = [{"trial_id": "t1"}, rec("p1", "t1")]
    counts = count_pairs(records)
    assert list(counts.keys()) == [("p1", "t1")]


def test_count_pairs_empty():
    assert count_pairs([]) == {}


# ---------------------------------------------------------------------------
# extract_records
# ---------------------------------------------------------------------------


def test_extract_records_with_list():
    data = [rec("p1", "t1"), rec("p2", "t2")]
    result = extract_records(data, ("labels", "records"))
    assert len(result) == 2
    assert result[0]["patient_id"] == "p1"


def test_extract_records_with_dict_candidate_key():
    data = {"labels": [rec("p1", "t1"), rec("p2", "t2")]}
    result = extract_records(data, ("labels", "records"))
    assert len(result) == 2


def test_extract_records_with_dict_second_candidate_key():
    data = {"records": [rec("p1", "t1")]}
    result = extract_records(data, ("labels", "records"))
    assert len(result) == 1


def test_extract_records_with_dict_of_dicts():
    data = {
        "key1": rec("p1", "t1"),
        "key2": rec("p2", "t2"),
    }
    result = extract_records(data, ("labels",))
    assert len(result) == 2


def test_extract_records_filters_non_dicts_in_list():
    data = [rec("p1", "t1"), "not_a_dict", 42]
    result = extract_records(data, ())
    assert len(result) == 1


def test_extract_records_empty_list():
    assert extract_records([], ("labels",)) == []


def test_extract_records_non_matching_dict():
    data = {"other_key": [rec("p1", "t1")]}
    # no candidate key matches → fallback to dict-of-values
    result = extract_records(data, ("labels", "records"))
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# analyze_prediction_coverage
# ---------------------------------------------------------------------------


def test_analyze_full_coverage():
    gold = [rec("p1", "t1"), rec("p2", "t2"), rec("p3", "t3")]
    preds = [rec("p1", "t1"), rec("p2", "t2"), rec("p3", "t3")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["gold_pair_count"] == 3
    assert summary["prediction_pair_count"] == 3
    assert summary["shared_pair_count"] == 3
    assert summary["missing_prediction_count"] == 0
    assert summary["extra_prediction_count"] == 0
    assert summary["coverage_pct"] == pytest.approx(100.0)
    assert summary["prediction_only_pct"] == pytest.approx(0.0)


def test_analyze_missing_predictions():
    gold = [rec("p1", "t1"), rec("p2", "t2"), rec("p3", "t3")]
    preds = [rec("p1", "t1")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["missing_prediction_count"] == 2
    assert summary["shared_pair_count"] == 1
    assert summary["coverage_pct"] == pytest.approx(100 / 3)
    assert len(summary["missing_examples"]) == 2


def test_analyze_extra_predictions():
    gold = [rec("p1", "t1")]
    preds = [rec("p1", "t1"), rec("p9", "t9"), rec("p8", "t8")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["extra_prediction_count"] == 2
    assert summary["shared_pair_count"] == 1
    assert len(summary["extra_examples"]) == 2


def test_analyze_duplicate_gold():
    gold = [rec("p1", "t1"), rec("p1", "t1"), rec("p2", "t2")]
    preds = [rec("p1", "t1"), rec("p2", "t2")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["duplicate_gold_count"] == 1
    assert summary["duplicate_gold_examples"][0]["patient_id"] == "p1"
    assert summary["duplicate_gold_examples"][0]["count"] == 2


def test_analyze_duplicate_predictions():
    gold = [rec("p1", "t1"), rec("p2", "t2")]
    preds = [rec("p1", "t1"), rec("p1", "t1"), rec("p2", "t2")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["duplicate_prediction_count"] == 1
    assert summary["duplicate_prediction_examples"][0]["patient_id"] == "p1"


def test_analyze_empty_inputs():
    summary = analyze_prediction_coverage([], [])
    assert summary["gold_pair_count"] == 0
    assert summary["prediction_pair_count"] == 0
    assert summary["coverage_pct"] == pytest.approx(0.0)
    assert summary["prediction_only_pct"] == pytest.approx(0.0)


def test_analyze_no_overlap():
    gold = [rec("p1", "t1"), rec("p2", "t2")]
    preds = [rec("p3", "t3"), rec("p4", "t4")]
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["shared_pair_count"] == 0
    assert summary["missing_prediction_count"] == 2
    assert summary["extra_prediction_count"] == 2
    assert summary["coverage_pct"] == pytest.approx(0.0)


def test_analyze_missing_examples_capped_at_20():
    gold = [rec(f"p{i}", "t1") for i in range(30)]
    preds = []
    summary = analyze_prediction_coverage(gold, preds)
    assert summary["missing_prediction_count"] == 30
    assert len(summary["missing_examples"]) == 20
