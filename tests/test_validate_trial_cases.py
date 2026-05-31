"""Tests for validate_trial_cases.py."""

import json
import pytest

from eval.validate_trial_cases import (
    load_trial_cases,
    validate_trial_case,
    validate_trial_cases,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _minimal_valid() -> dict:
    """Return a minimal trial case that should pass all checks."""
    return {
        "trial_id": "NCT001",
        "criteria_text": (
            "Inclusion criteria: Age 40 to 80. "
            "Diagnosis of idiopathic Parkinson disease. "
            "Exclusion criteria: Prior deep brain stimulation."
        ),
    }


# ── Valid cases ───────────────────────────────────────────────────────────────

def test_valid_minimal_case_has_no_issues():
    issues = validate_trial_case(_minimal_valid())
    assert issues == []


def test_valid_case_with_optional_metadata_has_no_issues():
    case = {
        **_minimal_valid(),
        "title": "A Parkinson Study",
        "phase": "2",
        "status": "recruiting",
    }
    issues = validate_trial_case(case)
    assert issues == []


def test_eligibility_criteria_accepted_instead_of_criteria_text():
    case = {
        "trial_id": "NCT002",
        "eligibility_criteria": (
            "Inclusion: Age >= 18. Diagnosis of Parkinson disease. "
            "Exclusion: Prior DBS. Current MAO-B inhibitor use."
        ),
    }
    issues = validate_trial_case(case)
    assert issues == []


def test_eligibility_criteria_accepted_no_criteria_text_error():
    case = {
        "trial_id": "NCT003",
        "eligibility_criteria": "Inclusion: Age 40-80. Exclusion: Prior DBS.",
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


# ── Required field checks ─────────────────────────────────────────────────────

def test_missing_trial_id_reported():
    case = _minimal_valid()
    del case["trial_id"]
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "trial_id" in fields


def test_empty_trial_id_reported():
    case = {**_minimal_valid(), "trial_id": ""}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "trial_id" in fields


def test_whitespace_trial_id_reported():
    case = {**_minimal_valid(), "trial_id": "   "}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "trial_id" in fields


def test_missing_criteria_text_reported():
    case = {"trial_id": "NCT004"}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "criteria_text" in fields


def test_empty_criteria_text_reported():
    case = {**_minimal_valid(), "criteria_text": ""}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "criteria_text" in fields


def test_whitespace_only_criteria_text_reported():
    case = {**_minimal_valid(), "criteria_text": "   "}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "criteria_text" in fields


def test_empty_criteria_text_is_error_severity():
    case = {**_minimal_valid(), "criteria_text": ""}
    issues = validate_trial_case(case)
    crit_issues = [i for i in issues if i["field"] == "criteria_text"]
    assert all(i["severity"] == "error" for i in crit_issues)


def test_inclusion_criteria_list_accepted():
    case = {
        "trial_id": "NCT010",
        "inclusion_criteria": ["Age 40 to 80", "Diagnosis of Parkinson disease"],
        "exclusion_criteria": ["Prior deep brain stimulation"],
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


def test_exclusion_criteria_only_accepted():
    case = {
        "trial_id": "NCT011",
        "exclusion_criteria": ["Prior DBS", "Current MAO-B inhibitor use"],
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


def test_inclusion_field_string_accepted():
    case = {
        "trial_id": "NCT012",
        "inclusion": "Age 40 to 80. Diagnosis of idiopathic Parkinson disease.",
        "exclusion": "Prior DBS.",
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


def test_criteria_dict_accepted():
    case = {
        "trial_id": "NCT013",
        "criteria": {
            "inclusion": ["Age >= 40", "Diagnosis of Parkinson disease"],
            "exclusion": ["Prior deep brain stimulation"],
        },
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


def test_criteria_list_accepted():
    case = {
        "trial_id": "NCT014",
        "criteria": ["Age 40-80", "Diagnosis of Parkinson disease", "No prior DBS"],
    }
    issues = validate_trial_case(case)
    error_fields = [i["field"] for i in issues if i["severity"] == "error"]
    assert "criteria_text" not in error_fields


def test_all_criteria_fields_empty_strings_reported():
    case = {
        "trial_id": "NCT015",
        "inclusion_criteria": [],
        "exclusion_criteria": [],
    }
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "criteria_text" in fields


# ── Short criteria warning ────────────────────────────────────────────────────

def test_extremely_short_criteria_reported():
    case = {**_minimal_valid(), "criteria_text": "Age > 18"}
    issues = validate_trial_case(case)
    fields = [i["field"] for i in issues]
    assert "criteria_text" in fields


def test_extremely_short_criteria_is_warning():
    case = {**_minimal_valid(), "criteria_text": "Age > 18"}
    issues = validate_trial_case(case)
    crit_issues = [i for i in issues if i["field"] == "criteria_text"]
    severities = {i["severity"] for i in crit_issues}
    assert "warning" in severities
    assert "error" not in severities


def test_adequate_length_criteria_no_length_warning():
    issues = validate_trial_case(_minimal_valid())
    length_warnings = [
        i for i in issues
        if i["field"] == "criteria_text" and "short" in i["message"]
    ]
    assert length_warnings == []


# ── No eligibility language warning ──────────────────────────────────────────

def test_no_eligibility_language_reported_as_warning():
    case = {
        **_minimal_valid(),
        "criteria_text": "This is a long enough string without any relevant medical words in it at all.",
    }
    issues = validate_trial_case(case)
    warn_fields = [i["field"] for i in issues if i["severity"] == "warning"]
    assert "criteria_text" in warn_fields


def test_criteria_with_eligibility_language_no_language_warning():
    issues = validate_trial_case(_minimal_valid())
    language_warnings = [
        i for i in issues
        if i["field"] == "criteria_text" and "inclusion" in i["message"]
    ]
    assert language_warnings == []


# ── Issue dict structure ──────────────────────────────────────────────────────

def test_issue_dict_has_required_keys():
    case = {**_minimal_valid(), "trial_id": ""}
    issues = validate_trial_case(case)
    assert issues
    for iss in issues:
        assert "trial_id" in iss
        assert "severity" in iss
        assert "field" in iss
        assert "message" in iss


def test_issue_trial_id_matches_case():
    case = {**_minimal_valid(), "criteria_text": ""}
    issues = validate_trial_case(case)
    for iss in issues:
        assert iss["trial_id"] == "NCT001"


# ── Duplicate trial_id detection ──────────────────────────────────────────────

def test_duplicate_trial_id_reported():
    cases = [
        _minimal_valid(),
        _minimal_valid(),  # same trial_id "NCT001"
    ]
    issues = validate_trial_cases(cases)
    dup_issues = [i for i in issues if "duplicate" in i["message"]]
    assert dup_issues


def test_duplicate_trial_id_is_error_severity():
    cases = [_minimal_valid(), _minimal_valid()]
    issues = validate_trial_cases(cases)
    dup_issues = [i for i in issues if "duplicate" in i["message"]]
    assert all(i["severity"] == "error" for i in dup_issues)


def test_unique_trial_ids_no_duplicate_issue():
    cases = [
        _minimal_valid(),
        {**_minimal_valid(), "trial_id": "NCT999"},
    ]
    issues = validate_trial_cases(cases)
    dup_issues = [i for i in issues if "duplicate" in i["message"]]
    assert dup_issues == []


def test_three_duplicates_reported_once():
    cases = [_minimal_valid(), _minimal_valid(), _minimal_valid()]
    issues = validate_trial_cases(cases)
    dup_issues = [i for i in issues if "duplicate" in i["message"]]
    assert len(dup_issues) == 1
    assert "3" in dup_issues[0]["message"]


# ── Multi-case aggregation ────────────────────────────────────────────────────

def test_validate_trial_cases_empty_list():
    assert validate_trial_cases([]) == []


def test_validate_trial_cases_all_valid():
    cases = [
        _minimal_valid(),
        {**_minimal_valid(), "trial_id": "NCT002"},
    ]
    assert validate_trial_cases(cases) == []


def test_validate_trial_cases_aggregates_across_cases():
    cases = [
        {**_minimal_valid(), "trial_id": ""},
        {**_minimal_valid(), "trial_id": "NCT002", "criteria_text": ""},
    ]
    issues = validate_trial_cases(cases)
    assert len(issues) >= 2


def test_validate_trial_cases_reports_correct_trial_ids():
    cases = [
        {**_minimal_valid(), "trial_id": "NCT010", "criteria_text": ""},
        {**_minimal_valid(), "trial_id": "NCT011", "criteria_text": ""},
    ]
    issues = validate_trial_cases(cases)
    tids = {i["trial_id"] for i in issues}
    assert "NCT010" in tids
    assert "NCT011" in tids


def test_validate_trial_cases_mixed_valid_and_invalid():
    cases = [
        _minimal_valid(),
        {**_minimal_valid(), "trial_id": "NCT020", "criteria_text": ""},
    ]
    issues = validate_trial_cases(cases)
    assert any(i["trial_id"] == "NCT020" for i in issues)
    assert not any(i["trial_id"] == "NCT001" for i in issues)


# ── load_trial_cases ──────────────────────────────────────────────────────────

def test_load_trial_cases_valid_file(tmp_path):
    data = [_minimal_valid()]
    p = tmp_path / "trial_cases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_trial_cases(p)
    assert result == data


def test_load_trial_cases_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trial_cases(tmp_path / "nonexistent.json")


def test_load_trial_cases_not_a_list_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"trial_id": "NCT001"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_trial_cases(p)


def test_load_trial_cases_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        load_trial_cases(p)
