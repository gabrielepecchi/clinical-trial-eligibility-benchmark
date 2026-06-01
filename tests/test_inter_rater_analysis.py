"""
tests/test_inter_rater_analysis.py

Unit tests for eval/run_inter_rater_analysis.py pure functions.
No real data files are required.
"""

import pytest

from eval.run_inter_rater_analysis import (
    cohen_kappa,
    compare_label_sources,
    index_labels,
    percent_agreement,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_record(patient_id, trial_id, label):
    return {"patient_id": patient_id, "trial_id": trial_id, "label": label}


# ---------------------------------------------------------------------------
# percent_agreement
# ---------------------------------------------------------------------------


def test_percent_agreement_perfect():
    a = ["eligible", "not_eligible", "unclear"]
    b = ["eligible", "not_eligible", "unclear"]
    assert percent_agreement(a, b) == pytest.approx(1.0)


def test_percent_agreement_zero():
    a = ["eligible", "eligible", "eligible"]
    b = ["not_eligible", "not_eligible", "not_eligible"]
    assert percent_agreement(a, b) == pytest.approx(0.0)


def test_percent_agreement_partial():
    a = ["eligible", "not_eligible", "unclear", "eligible"]
    b = ["eligible", "unclear", "unclear", "not_eligible"]
    # 2 matches out of 4
    assert percent_agreement(a, b) == pytest.approx(0.5)


def test_percent_agreement_empty():
    assert percent_agreement([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    labels = ["eligible", "not_eligible", "unclear", "eligible", "not_eligible"]
    kappa = cohen_kappa(labels, labels)
    assert kappa == pytest.approx(1.0, abs=1e-9)


def test_cohen_kappa_empty():
    assert cohen_kappa([], []) == pytest.approx(0.0)


def test_cohen_kappa_no_crash_on_disagreement():
    a = ["eligible", "eligible", "not_eligible", "unclear"]
    b = ["not_eligible", "unclear", "eligible", "eligible"]
    kappa = cohen_kappa(a, b)
    # Just check it returns a finite float without crashing
    assert isinstance(kappa, float)
    assert -1.0 <= kappa <= 1.0


def test_cohen_kappa_all_same_label():
    # Degenerate: both raters always say 'eligible'
    a = ["eligible", "eligible", "eligible"]
    b = ["eligible", "eligible", "eligible"]
    # p_e == 1.0 -> kappa defined as 0.0 by convention
    kappa = cohen_kappa(a, b)
    assert isinstance(kappa, float)


def test_cohen_kappa_partial():
    a = ["eligible", "not_eligible", "unclear", "eligible"]
    b = ["eligible", "not_eligible", "eligible", "eligible"]
    # 3 out of 4 agree
    kappa = cohen_kappa(a, b)
    assert kappa > 0.0


# ---------------------------------------------------------------------------
# index_labels
# ---------------------------------------------------------------------------


def test_index_labels_uses_patient_and_trial_id():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p2", "t1", "not_eligible"),
        make_record("p1", "t2", "unclear"),
    ]
    idx = index_labels(records)
    assert idx[("p1", "t1")] == "eligible"
    assert idx[("p2", "t1")] == "not_eligible"
    assert idx[("p1", "t2")] == "unclear"


def test_index_labels_skips_invalid_labels():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p2", "t2", "unknown_label"),
        make_record("p3", "t3", ""),
    ]
    idx = index_labels(records)
    assert ("p1", "t1") in idx
    assert ("p2", "t2") not in idx
    assert ("p3", "t3") not in idx


def test_index_labels_empty():
    assert index_labels([]) == {}


# ---------------------------------------------------------------------------
# compare_label_sources
# ---------------------------------------------------------------------------


def _make_source(path, patient_trial_label_triples):
    records = [
        make_record(pid, tid, lbl)
        for pid, tid, lbl in patient_trial_label_triples
    ]
    return (path, records)


def test_compare_label_sources_shared_pairs():
    src_a = _make_source(
        "a.json",
        [("p1", "t1", "eligible"), ("p2", "t2", "not_eligible"), ("p3", "t3", "unclear")],
    )
    src_b = _make_source(
        "b.json",
        [("p1", "t1", "eligible"), ("p2", "t2", "eligible")],
    )
    result = compare_label_sources(src_a, src_b)
    assert result["shared_pairs"] == 2


def test_compare_label_sources_perfect_agreement():
    src_a = _make_source(
        "a.json",
        [("p1", "t1", "eligible"), ("p2", "t2", "not_eligible")],
    )
    src_b = _make_source(
        "b.json",
        [("p1", "t1", "eligible"), ("p2", "t2", "not_eligible")],
    )
    result = compare_label_sources(src_a, src_b)
    assert result["percent_agreement"] == pytest.approx(1.0)
    assert result["disagreement_counts"] == {}
    assert result["top_disagreements"] == []


def test_compare_label_sources_disagreements():
    src_a = _make_source(
        "a.json",
        [("p1", "t1", "eligible"), ("p2", "t2", "not_eligible"), ("p3", "t3", "unclear")],
    )
    src_b = _make_source(
        "b.json",
        [("p1", "t1", "not_eligible"), ("p2", "t2", "not_eligible"), ("p3", "t3", "eligible")],
    )
    result = compare_label_sources(src_a, src_b)
    assert result["shared_pairs"] == 3
    assert result["percent_agreement"] == pytest.approx(1 / 3)
    assert len(result["disagreement_counts"]) >= 1
    assert len(result["top_disagreements"]) == 2


def test_compare_label_sources_no_shared_pairs():
    src_a = _make_source("a.json", [("p1", "t1", "eligible")])
    src_b = _make_source("b.json", [("p9", "t9", "not_eligible")])
    result = compare_label_sources(src_a, src_b)
    assert result["shared_pairs"] == 0
    assert result["percent_agreement"] == pytest.approx(0.0)
    assert result["disagreement_counts"] == {}


def test_compare_label_sources_stores_paths():
    src_a = _make_source("path/a.json", [("p1", "t1", "eligible")])
    src_b = _make_source("path/b.json", [("p1", "t1", "unclear")])
    result = compare_label_sources(src_a, src_b)
    assert result["source_a_path"] == "path/a.json"
    assert result["source_b_path"] == "path/b.json"
