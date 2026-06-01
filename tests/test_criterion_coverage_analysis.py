"""
Tests for eval/run_criterion_coverage_analysis.py (Task 65).

Run with:
    PYTHONPATH=. python -m pytest tests/test_criterion_coverage_analysis.py -v
"""

from __future__ import annotations

import pytest

from eval.run_criterion_coverage_analysis import (
    analyze_criterion_coverage,
    criterion_matches,
    extract_trial_criteria,
    index_trial_criteria,
    normalize_criterion_text,
    split_criteria,
)


# ---------------------------------------------------------------------------
# normalize_criterion_text
# ---------------------------------------------------------------------------


def test_normalize_lowercase():
    assert normalize_criterion_text("Age >= 18 Years") == "age >= 18 years"


def test_normalize_strips_bullets():
    assert normalize_criterion_text("- Must be an adult") == "must be an adult"
    assert normalize_criterion_text("• No prior treatment") == "no prior treatment"
    assert normalize_criterion_text("* Active smoker") == "active smoker"


def test_normalize_collapses_whitespace():
    assert normalize_criterion_text("  age   >=  18  ") == "age >= 18"


def test_normalize_empty():
    assert normalize_criterion_text("") == ""


def test_normalize_mixed():
    result = normalize_criterion_text("  - Prior  DBS  Implant  ")
    assert result == "prior dbs implant"


# ---------------------------------------------------------------------------
# split_criteria
# ---------------------------------------------------------------------------


def test_split_criteria_list():
    result = split_criteria(["Age >= 18", "No prior DBS", ""])
    assert result == ["Age >= 18", "No prior DBS"]


def test_split_criteria_string_newlines():
    result = split_criteria("Age >= 18\nNo prior DBS\nDiagnosis confirmed")
    assert len(result) == 3
    assert "Age >= 18" in result


def test_split_criteria_string_semicolons():
    result = split_criteria("Age >= 18; No prior DBS; MMSE >= 24")
    assert len(result) == 3
    assert "MMSE >= 24" in result


def test_split_criteria_empty_string():
    assert split_criteria("") == []


def test_split_criteria_empty_list():
    assert split_criteria([]) == []


def test_split_criteria_non_string_non_list():
    assert split_criteria(None) == []
    assert split_criteria(42) == []


# ---------------------------------------------------------------------------
# extract_trial_criteria
# ---------------------------------------------------------------------------


def test_extract_from_inclusion_exclusion():
    record = {
        "trial_id": "T1",
        "inclusion_criteria": "Age >= 18\nMMSE >= 20",
        "exclusion_criteria": "Prior DBS",
    }
    result = extract_trial_criteria(record)
    assert "Age >= 18" in result
    assert "MMSE >= 20" in result
    assert "Prior DBS" in result


def test_extract_from_criteria_text():
    record = {
        "trial_id": "T2",
        "criteria_text": "Diagnosis confirmed; No prior treatment",
    }
    result = extract_trial_criteria(record)
    assert "Diagnosis confirmed" in result
    assert "No prior treatment" in result


def test_extract_deduplicates():
    record = {
        "inclusion_criteria": "Age >= 18",
        "eligibility_criteria": "Age >= 18",
    }
    result = extract_trial_criteria(record)
    assert result.count("Age >= 18") == 1


def test_extract_empty_record():
    assert extract_trial_criteria({}) == []


# ---------------------------------------------------------------------------
# criterion_matches
# ---------------------------------------------------------------------------


def test_criterion_matches_exact():
    obs = {"age >= 18 years", "no prior dbs implant"}
    assert criterion_matches("age >= 18 years", obs) is True


def test_criterion_matches_no_match():
    obs = {"age >= 18 years"}
    assert criterion_matches("mmse >= 24", obs) is False


def test_criterion_matches_substring_long_enough():
    long_expected = "patient must have a confirmed diagnosis of parkinson disease"
    long_observed = (
        "patient must have a confirmed diagnosis of parkinson disease and "
        "have been symptomatic for at least two years"
    )
    obs = {long_observed}
    assert criterion_matches(long_expected, obs) is True


def test_criterion_matches_substring_too_short():
    # Both strings are shorter than SUBSTRING_MATCH_MIN_LEN (20)
    obs = {"age >= 18"}
    # "age >= 18" is 9 chars — substring match should not fire
    assert criterion_matches("age >= 18", obs) is True  # exact match fires first


def test_criterion_matches_short_no_exact():
    obs = {"age >= 65"}
    assert criterion_matches("age >= 18", obs) is False


# ---------------------------------------------------------------------------
# analyze_criterion_coverage — full coverage
# ---------------------------------------------------------------------------


def test_analyze_full_coverage():
    trial_criteria = {
        "T1": ["age >= 18 years", "no prior dbs implant"],
    }
    result_criteria = {
        "T1": ["age >= 18 years", "no prior dbs implant"],
    }
    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    assert "T1" in summary["full_coverage"]
    assert "T1" not in summary["partial_coverage"]
    assert "T1" not in summary["zero_coverage"]
    assert summary["total_missing_criteria"] == 0


# ---------------------------------------------------------------------------
# analyze_criterion_coverage — partial coverage
# ---------------------------------------------------------------------------


def test_analyze_partial_coverage():
    trial_criteria = {
        "T2": ["age >= 18 years", "no prior dbs implant", "mmse >= 24"],
    }
    result_criteria = {
        "T2": ["age >= 18 years"],
    }
    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    assert "T2" in summary["partial_coverage"]
    assert "T2" not in summary["full_coverage"]
    assert "T2" not in summary["zero_coverage"]
    assert summary["total_missing_criteria"] == 2
    assert "T2" in summary["missing_by_trial"]
    assert len(summary["missing_by_trial"]["T2"]) == 2


# ---------------------------------------------------------------------------
# analyze_criterion_coverage — zero coverage
# ---------------------------------------------------------------------------


def test_analyze_zero_coverage():
    trial_criteria = {
        "T3": ["age >= 18 years", "diagnosis confirmed"],
    }
    result_criteria = {}
    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    assert "T3" in summary["zero_coverage"]
    assert "T3" not in summary["full_coverage"]
    assert "T3" not in summary["partial_coverage"]


# ---------------------------------------------------------------------------
# Orphan criterion rows
# ---------------------------------------------------------------------------


def test_analyze_orphan_rows():
    trial_criteria = {
        "T1": ["age >= 18 years"],
    }
    result_criteria = {
        "T1": ["age >= 18 years"],
        "T_UNKNOWN": ["some criterion not in any trial"],
    }
    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    orphan_ids = {row["trial_id"] for row in summary["orphan_rows"]}
    assert "T_UNKNOWN" in orphan_ids
    assert "T1" not in orphan_ids


def test_analyze_no_orphan_rows():
    trial_criteria = {"T1": ["age >= 18 years"]}
    result_criteria = {"T1": ["age >= 18 years"]}
    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    assert summary["orphan_rows"] == []


# ---------------------------------------------------------------------------
# index_trial_criteria (smoke test for list input)
# ---------------------------------------------------------------------------


def test_index_trial_criteria_list():
    trials = [
        {
            "trial_id": "T10",
            "inclusion_criteria": "Age >= 18\nMMSE >= 20",
        },
        {
            "trial_id": "T11",
            "exclusion_criteria": "Prior DBS",
        },
    ]
    index = index_trial_criteria(trials)
    assert "T10" in index
    assert "T11" in index
    assert any("age >= 18" in c for c in index["T10"])
    assert any("prior dbs" in c for c in index["T11"])
