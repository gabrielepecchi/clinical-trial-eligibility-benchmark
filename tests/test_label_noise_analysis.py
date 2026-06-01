"""
tests/test_label_noise_analysis.py

Unit tests for eval/run_label_noise_analysis.py pure functions.
No real data files are required.
"""

import pytest

from eval.run_label_noise_analysis import (
    analyze_label_source,
    build_label_noise_summary,
    label_pair_key,
    validate_label_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(patient_id, trial_id, label):
    return {"patient_id": patient_id, "trial_id": trial_id, "label": label}


# ---------------------------------------------------------------------------
# label_pair_key
# ---------------------------------------------------------------------------


def test_label_pair_key_returns_tuple():
    record = make_record("p1", "t1", "eligible")
    assert label_pair_key(record) == ("p1", "t1")


def test_label_pair_key_stringifies_values():
    record = {"patient_id": 42, "trial_id": 7, "label": "eligible"}
    assert label_pair_key(record) == ("42", "7")


def test_label_pair_key_missing_fields():
    record = {}
    key = label_pair_key(record)
    assert key == ("", "")


# ---------------------------------------------------------------------------
# validate_label_record
# ---------------------------------------------------------------------------


def test_validate_label_record_valid():
    record = make_record("p1", "t1", "eligible")
    assert validate_label_record(record) == []


def test_validate_label_record_valid_all_labels():
    for label in ("eligible", "not_eligible", "unclear"):
        record = make_record("p1", "t1", label)
        assert validate_label_record(record) == []


def test_validate_label_record_missing_patient_id():
    record = {"trial_id": "t1", "label": "eligible"}
    issues = validate_label_record(record)
    assert any("missing_field:patient_id" in i for i in issues)


def test_validate_label_record_missing_trial_id():
    record = {"patient_id": "p1", "label": "eligible"}
    issues = validate_label_record(record)
    assert any("missing_field:trial_id" in i for i in issues)


def test_validate_label_record_missing_label():
    record = {"patient_id": "p1", "trial_id": "t1"}
    issues = validate_label_record(record)
    assert any("missing_field:label" in i for i in issues)


def test_validate_label_record_invalid_label():
    record = make_record("p1", "t1", "unknown")
    issues = validate_label_record(record)
    assert any("invalid_label:" in i for i in issues)


def test_validate_label_record_empty_label():
    record = make_record("p1", "t1", "")
    issues = validate_label_record(record)
    assert any("missing_field:label" in i or "invalid_label:" in i for i in issues)


def test_validate_label_record_none_label():
    record = {"patient_id": "p1", "trial_id": "t1", "label": None}
    issues = validate_label_record(record)
    assert len(issues) > 0


# ---------------------------------------------------------------------------
# analyze_label_source — basic counts
# ---------------------------------------------------------------------------


def test_analyze_label_source_all_valid():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p2", "t2", "not_eligible"),
        make_record("p3", "t3", "unclear"),
    ]
    result = analyze_label_source("fake.json", records)
    assert result["total_records"] == 3
    assert result["valid_records"] == 3
    assert result["invalid_records"] == 0
    assert result["duplicate_pair_count"] == 0
    assert result["conflicting_duplicate_pair_count"] == 0
    assert result["top_conflicts"] == []


def test_analyze_label_source_invalid_label():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p2", "t2", "WRONG"),
    ]
    result = analyze_label_source("fake.json", records)
    assert result["invalid_label_count"] >= 1
    assert result["invalid_records"] >= 1


def test_analyze_label_source_missing_field():
    records = [
        {"trial_id": "t1", "label": "eligible"},  # missing patient_id
    ]
    result = analyze_label_source("fake.json", records)
    assert result["missing_field_count"] >= 1
    assert result["invalid_records"] >= 1


# ---------------------------------------------------------------------------
# analyze_label_source — duplicates
# ---------------------------------------------------------------------------


def test_analyze_label_source_detects_same_label_duplicates():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p1", "t1", "eligible"),  # exact duplicate
        make_record("p2", "t2", "not_eligible"),
    ]
    result = analyze_label_source("fake.json", records)
    assert result["duplicate_pair_count"] == 1
    assert result["conflicting_duplicate_pair_count"] == 0


def test_analyze_label_source_detects_conflicting_duplicate_labels():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p1", "t1", "not_eligible"),  # conflict!
        make_record("p2", "t2", "unclear"),
    ]
    result = analyze_label_source("fake.json", records)
    assert result["duplicate_pair_count"] == 1
    assert result["conflicting_duplicate_pair_count"] == 1
    assert len(result["top_conflicts"]) == 1
    conflict = result["top_conflicts"][0]
    assert conflict["patient_id"] == "p1"
    assert conflict["trial_id"] == "t1"
    assert "eligible" in conflict["labels_seen"]
    assert "not_eligible" in conflict["labels_seen"]


def test_analyze_label_source_multiple_conflicts():
    records = [
        make_record("p1", "t1", "eligible"),
        make_record("p1", "t1", "not_eligible"),
        make_record("p2", "t2", "unclear"),
        make_record("p2", "t2", "eligible"),
        make_record("p3", "t3", "not_eligible"),  # no duplicate
    ]
    result = analyze_label_source("fake.json", records)
    assert result["duplicate_pair_count"] == 2
    assert result["conflicting_duplicate_pair_count"] == 2
    assert len(result["top_conflicts"]) == 2


def test_analyze_label_source_top_conflicts_capped_at_10():
    records = []
    for i in range(15):
        records.append(make_record(f"p{i}", "t1", "eligible"))
        records.append(make_record(f"p{i}", "t1", "not_eligible"))
    result = analyze_label_source("fake.json", records)
    assert result["conflicting_duplicate_pair_count"] == 15
    assert len(result["top_conflicts"]) == 10


# ---------------------------------------------------------------------------
# build_label_noise_summary
# ---------------------------------------------------------------------------


def test_build_label_noise_summary_multiple_sources():
    src_a = ("a.json", [make_record("p1", "t1", "eligible")])
    src_b = (
        "b.json",
        [
            make_record("p2", "t2", "not_eligible"),
            make_record("p2", "t2", "eligible"),  # conflict
        ],
    )
    summary = build_label_noise_summary([src_a, src_b])
    analyses = summary["source_analyses"]
    assert len(analyses) == 2
    assert analyses[0]["path"] == "a.json"
    assert analyses[1]["path"] == "b.json"
    assert analyses[1]["conflicting_duplicate_pair_count"] == 1


def test_build_label_noise_summary_empty_sources():
    summary = build_label_noise_summary([])
    assert summary["source_analyses"] == []


def test_build_label_noise_summary_single_clean_source():
    src = ("only.json", [make_record("p1", "t1", "eligible")])
    summary = build_label_noise_summary([src])
    assert len(summary["source_analyses"]) == 1
    assert summary["source_analyses"][0]["duplicate_pair_count"] == 0
