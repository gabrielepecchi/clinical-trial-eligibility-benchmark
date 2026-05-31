"""Tests for validate_patient_cases.py."""

import json
import pytest

from eval.validate_patient_cases import (
    load_patient_cases,
    validate_patient_case,
    validate_patient_cases,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _minimal_valid() -> dict:
    """Return a minimal patient case that should pass all checks."""
    return {
        "patient_id": "P001",
        "demographics": {"age": 62, "sex": "male"},
        "diagnosis": "idiopathic Parkinson disease",
        "clinical_summary": "62-year-old male with Parkinson disease.",
    }


# ── Valid case ────────────────────────────────────────────────────────────────

def test_valid_minimal_case_has_no_issues():
    issues = validate_patient_case(_minimal_valid())
    assert issues == []


def test_valid_case_with_optional_fields_has_no_issues():
    case = {
        **_minimal_valid(),
        "dbs_history": False,
        "pacemaker": False,
        "procedures": [],
        "devices": [],
    }
    issues = validate_patient_case(case)
    assert issues == []


def test_valid_case_dbs_true_with_dbs_in_procedures_no_contradiction():
    case = {
        **_minimal_valid(),
        "dbs_history": True,
        "procedures": ["deep brain stimulation"],
    }
    issues = validate_patient_case(case)
    assert issues == []


# ── Required field checks ─────────────────────────────────────────────────────

def test_missing_patient_id_reported():
    case = _minimal_valid()
    del case["patient_id"]
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "patient_id" in fields


def test_empty_patient_id_reported():
    case = {**_minimal_valid(), "patient_id": ""}
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "patient_id" in fields


def test_whitespace_patient_id_reported():
    case = {**_minimal_valid(), "patient_id": "   "}
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "patient_id" in fields


def test_valid_case_with_top_level_age_and_sex_no_demographics_passes():
    case = {
        "patient_id": "P010",
        "age": 62,
        "sex": "male",
        "diagnosis": "idiopathic Parkinson disease",
        "clinical_summary": "62-year-old male with PD.",
    }
    issues = validate_patient_case(case)
    assert issues == []


def test_valid_case_with_top_level_age_and_gender_no_demographics_passes():
    case = {
        "patient_id": "P011",
        "age": 55,
        "gender": "female",
        "diagnosis": "idiopathic Parkinson disease",
        "clinical_summary": "55-year-old female with PD.",
    }
    issues = validate_patient_case(case)
    assert issues == []


def test_missing_demographics_reported_only_when_no_top_level_fields():
    case = {
        "patient_id": "P012",
        "diagnosis": "PD",
        "clinical_summary": "ok",
        # no demographics dict, no age, no sex/gender
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "demographics" in fields


def test_top_level_age_without_sex_still_reports_missing_demographics():
    case = {
        "patient_id": "P013",
        "age": 60,
        # no sex or gender, no demographics dict
        "diagnosis": "PD",
        "clinical_summary": "ok",
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "demographics" in fields





def test_demographics_not_dict_reported():
    case = {**_minimal_valid(), "demographics": "old man"}
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "demographics" in fields


def test_missing_diagnosis_reported():
    case = _minimal_valid()
    del case["diagnosis"]
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "diagnosis" in fields


def test_empty_diagnosis_reported():
    case = {**_minimal_valid(), "diagnosis": ""}
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "diagnosis" in fields


def test_none_diagnosis_reported():
    case = {**_minimal_valid(), "diagnosis": None}
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "diagnosis" in fields


def test_no_summary_field_reported_as_warning():
    case = _minimal_valid()
    del case["clinical_summary"]
    issues = validate_patient_case(case)
    warn_fields = [i["field"] for i in issues if i["severity"] == "warning"]
    assert "clinical_summary" in warn_fields


def test_summary_field_accepted_instead_of_clinical_summary():
    case = _minimal_valid()
    del case["clinical_summary"]
    case["summary"] = "Some narrative."
    issues = validate_patient_case(case)
    warn_fields = [i["field"] for i in issues if i["severity"] == "warning"]
    assert "clinical_summary" not in warn_fields


# ── Severity ──────────────────────────────────────────────────────────────────

def test_missing_patient_id_is_error_severity():
    case = {**_minimal_valid(), "patient_id": ""}
    issues = validate_patient_case(case)
    pid_issues = [i for i in issues if i["field"] == "patient_id"]
    assert all(i["severity"] == "error" for i in pid_issues)


def test_missing_diagnosis_is_error_severity():
    case = {**_minimal_valid(), "diagnosis": ""}
    issues = validate_patient_case(case)
    diag_issues = [i for i in issues if i["field"] == "diagnosis"]
    assert all(i["severity"] == "error" for i in diag_issues)


# ── DBS contradiction ─────────────────────────────────────────────────────────

def test_dbs_contradiction_in_procedures_reported():
    case = {
        **_minimal_valid(),
        "dbs_history": False,
        "procedures": ["deep brain stimulation"],
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "dbs_history" in fields


def test_dbs_contradiction_in_history_reported():
    case = {
        **_minimal_valid(),
        "dbs_history": False,
        "history": "Patient underwent DBS in 2019.",
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "dbs_history" in fields


def test_dbs_contradiction_case_insensitive():
    case = {
        **_minimal_valid(),
        "dbs_history": False,
        "procedures": ["Deep Brain Stimulation"],
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "dbs_history" in fields


def test_dbs_absent_field_no_contradiction():
    case = _minimal_valid()
    # dbs_history not present at all — should not flag
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "dbs_history" not in fields


def test_dbs_contradiction_is_error_severity():
    case = {
        **_minimal_valid(),
        "dbs_history": False,
        "procedures": ["DBS"],
    }
    issues = validate_patient_case(case)
    dbs_issues = [i for i in issues if i["field"] == "dbs_history"]
    assert all(i["severity"] == "error" for i in dbs_issues)


# ── Pacemaker contradiction ───────────────────────────────────────────────────

def test_pacemaker_contradiction_in_devices_reported():
    case = {
        **_minimal_valid(),
        "pacemaker": False,
        "devices": ["cardiac pacemaker"],
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "pacemaker" in fields


def test_pacemaker_contradiction_in_history_reported():
    case = {
        **_minimal_valid(),
        "pacemaker": False,
        "history": "Patient has a pacemaker implanted.",
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "pacemaker" in fields


def test_pacemaker_absent_field_no_contradiction():
    case = _minimal_valid()
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "pacemaker" not in fields


def test_pacemaker_true_with_pacemaker_in_devices_no_contradiction():
    case = {
        **_minimal_valid(),
        "pacemaker": True,
        "devices": ["pacemaker"],
    }
    issues = validate_patient_case(case)
    fields = [i["field"] for i in issues]
    assert "pacemaker" not in fields


# ── Issue dict structure ──────────────────────────────────────────────────────

def test_issue_dict_has_required_keys():
    case = {**_minimal_valid(), "patient_id": ""}
    issues = validate_patient_case(case)
    assert issues
    for iss in issues:
        assert "patient_id" in iss
        assert "severity" in iss
        assert "field" in iss
        assert "message" in iss


def test_issue_patient_id_matches_case():
    case = {**_minimal_valid(), "diagnosis": ""}
    issues = validate_patient_case(case)
    for iss in issues:
        assert iss["patient_id"] == "P001"


# ── Multi-case aggregation ────────────────────────────────────────────────────

def test_validate_patient_cases_empty_list():
    assert validate_patient_cases([]) == []


def test_validate_patient_cases_all_valid():
    cases = [_minimal_valid(), {**_minimal_valid(), "patient_id": "P002"}]
    assert validate_patient_cases(cases) == []


def test_validate_patient_cases_aggregates_across_cases():
    cases = [
        {**_minimal_valid(), "patient_id": ""},          # error in case 1
        {**_minimal_valid(), "patient_id": "P002", "diagnosis": ""},  # error in case 2
    ]
    issues = validate_patient_cases(cases)
    assert len(issues) >= 2


def test_validate_patient_cases_reports_correct_patient_ids():
    cases = [
        {**_minimal_valid(), "patient_id": "P001", "diagnosis": ""},
        {**_minimal_valid(), "patient_id": "P002",
         "dbs_history": False, "procedures": ["DBS"]},
    ]
    issues = validate_patient_cases(cases)
    pids = {i["patient_id"] for i in issues}
    assert "P001" in pids
    assert "P002" in pids


def test_validate_patient_cases_mixed_valid_and_invalid():
    cases = [
        _minimal_valid(),
        {**_minimal_valid(), "patient_id": "P002", "diagnosis": None},
    ]
    issues = validate_patient_cases(cases)
    assert any(i["patient_id"] == "P002" for i in issues)
    assert not any(i["patient_id"] == "P001" for i in issues)


# ── load_patient_cases ────────────────────────────────────────────────────────

def test_load_patient_cases_valid_file(tmp_path):
    data = [_minimal_valid()]
    p = tmp_path / "patient_cases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_patient_cases(p)
    assert result == data


def test_load_patient_cases_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_patient_cases(tmp_path / "nonexistent.json")


def test_load_patient_cases_not_a_list_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"patient_id": "P001"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_patient_cases(p)


def test_load_patient_cases_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        load_patient_cases(p)
