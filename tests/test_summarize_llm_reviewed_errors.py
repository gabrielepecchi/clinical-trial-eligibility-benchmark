"""Tests for LLM-reviewed error analysis helpers."""

from summarize_llm_reviewed_errors import (
    build_error_csv_rows,
    write_error_csv_rows,
)

_ERROR_RECORD = {
    "patient_id": "P001",
    "trial_id": "T001",
    "gold_label": "not_eligible",
    "predicted_label": "eligible",
    "error_type": "missed_cognitive_exclusion",
    "matcher_explanation": "No blocking criteria found.",
    "blocking_criteria": ["cognitive impairment", "moca < 24"],
    "uncertain_criteria": ["mri compatibility"],
}

_ERROR_RECORD_MINIMAL = {
    "patient_id": "P002",
    "trial_id": "T002",
    "gold_label": "eligible",
    "predicted_label": "not_eligible",
    "error_type": "overcalled_not_eligible",
}


def test_build_error_csv_rows_empty_input():
    assert build_error_csv_rows([]) == []


def test_build_error_csv_rows_returns_list():
    assert isinstance(build_error_csv_rows([_ERROR_RECORD]), list)


def test_build_error_csv_rows_length():
    assert len(build_error_csv_rows([_ERROR_RECORD, _ERROR_RECORD_MINIMAL])) == 2


def test_build_error_csv_rows_row_is_dict():
    assert isinstance(build_error_csv_rows([_ERROR_RECORD])[0], dict)


def test_build_error_csv_rows_patient_id():
    assert build_error_csv_rows([_ERROR_RECORD])[0]["patient_id"] == "P001"


def test_build_error_csv_rows_error_type():
    assert build_error_csv_rows([_ERROR_RECORD])[0]["error_type"] == "missed_cognitive_exclusion"


def test_build_error_csv_rows_blocking_criteria_joined():
    assert build_error_csv_rows([_ERROR_RECORD])[0]["blocking_criteria"] == "cognitive impairment; moca < 24"


def test_build_error_csv_rows_uncertain_criteria_joined():
    assert build_error_csv_rows([_ERROR_RECORD])[0]["uncertain_criteria"] == "mri compatibility"


def test_build_error_csv_rows_empty_list_fields_are_strings():
    row = build_error_csv_rows([_ERROR_RECORD_MINIMAL])[0]
    assert isinstance(row["blocking_criteria"], str)
    assert isinstance(row["uncertain_criteria"], str)


def test_build_error_csv_rows_missing_optional_fields_no_crash():
    assert build_error_csv_rows([_ERROR_RECORD_MINIMAL])[0]["explanation"] == ""


def test_build_error_csv_rows_case_id_increments():
    rows = build_error_csv_rows([_ERROR_RECORD, _ERROR_RECORD_MINIMAL])
    assert rows[0]["case_id"] == 1
    assert rows[1]["case_id"] == 2


def test_write_error_csv_rows_creates_file(tmp_path):
    out = tmp_path / "errors.csv"
    write_error_csv_rows(build_error_csv_rows([_ERROR_RECORD]), out)
    assert out.exists()


def test_write_error_csv_rows_header_present(tmp_path):
    out = tmp_path / "errors.csv"
    write_error_csv_rows(build_error_csv_rows([_ERROR_RECORD]), out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert "patient_id" in first_line
    assert "error_type" in first_line


def test_write_error_csv_rows_empty_creates_header_only(tmp_path):
    out = tmp_path / "empty.csv"
    write_error_csv_rows([], out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "patient_id" in lines[0]


# --- classify_error_severity ---

from summarize_llm_reviewed_errors import classify_error_severity


def test_classify_error_severity_critical():
    assert classify_error_severity({"gold_label": "not_eligible", "predicted_label": "eligible"}) == "critical"


def test_classify_error_severity_major_unclear_eligible():
    assert classify_error_severity({"gold_label": "unclear", "predicted_label": "eligible"}) == "major"


def test_classify_error_severity_major_unclear_not_eligible():
    assert classify_error_severity({"gold_label": "unclear", "predicted_label": "not_eligible"}) == "major"


def test_classify_error_severity_major_eligible_unclear():
    assert classify_error_severity({"gold_label": "eligible", "predicted_label": "unclear"}) == "major"


def test_classify_error_severity_major_not_eligible_unclear():
    assert classify_error_severity({"gold_label": "not_eligible", "predicted_label": "unclear"}) == "major_minor"


def test_classify_error_severity_minor():
    assert classify_error_severity({"gold_label": "eligible", "predicted_label": "not_eligible"}) == "minor"


def test_classify_error_severity_none():
    assert classify_error_severity({"gold_label": "eligible", "predicted_label": "eligible"}) == "none"


def test_build_error_csv_rows_includes_severity():
    row = build_error_csv_rows([_ERROR_RECORD])[0]
    assert "severity" in row


def test_write_error_csv_rows_header_includes_severity(tmp_path):
    out = tmp_path / "errors_sev.csv"
    write_error_csv_rows(build_error_csv_rows([_ERROR_RECORD]), out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert "severity" in first_line


# --- build_error_record ---

from summarize_llm_reviewed_errors import build_error_record

_PRED_RECORD = {
    "patient_id": "P001",
    "trial_id": "T001",
    "gold_label": "not_eligible",
    "predicted_label": "eligible",
    "gold_rationale": "Cognitive exclusion applies.",
    "matcher_explanation": "No blocking criteria found.",
    "blocking_criteria": ["moca < 24"],
    "uncertain_criteria": [],
    "gold_evidence": {},
}

_PRED_RECORD_MINIMAL = {
    "patient_id": "P002",
    "trial_id": "T002",
    "gold_label": "eligible",
    "predicted_label": "not_eligible",
}


def test_build_error_record_returns_dict():
    assert isinstance(build_error_record(_PRED_RECORD), dict)


def test_build_error_record_keys():
    result = build_error_record(_PRED_RECORD)
    for key in [
        "patient_id", "trial_id", "gold_label", "predicted_label",
        "error_type", "severity", "gold_rationale", "matcher_explanation",
        "blocking_criteria", "uncertain_criteria",
    ]:
        assert key in result


def test_build_error_record_patient_id():
    assert build_error_record(_PRED_RECORD)["patient_id"] == "P001"


def test_build_error_record_gold_label():
    assert build_error_record(_PRED_RECORD)["gold_label"] == "not_eligible"


def test_build_error_record_predicted_label():
    assert build_error_record(_PRED_RECORD)["predicted_label"] == "eligible"


def test_build_error_record_severity_populated():
    assert build_error_record(_PRED_RECORD)["severity"] == "critical"


def test_build_error_record_error_type_populated():
    assert build_error_record(_PRED_RECORD)["error_type"] != ""


def test_build_error_record_minimal_no_crash():
    result = build_error_record(_PRED_RECORD_MINIMAL)
    assert result["patient_id"] == "P002"


def test_build_error_record_minimal_severity_populated():
    assert build_error_record(_PRED_RECORD_MINIMAL)["severity"] == "minor"


def test_build_error_record_blocking_criteria_preserved():
    assert build_error_record(_PRED_RECORD)["blocking_criteria"] == ["moca < 24"]


# --- format_severity_breakdown ---

from summarize_llm_reviewed_errors import format_severity_breakdown

_SEV_ERRORS = [
    {"severity": "critical"},
    {"severity": "major"},
    {"severity": "major"},
    {"severity": "major_minor"},
    {"severity": "minor"},
]


def test_format_severity_breakdown_returns_string():
    assert isinstance(format_severity_breakdown(_SEV_ERRORS), str)


def test_format_severity_breakdown_header():
    assert "Errors by severity:" in format_severity_breakdown(_SEV_ERRORS)


def test_format_severity_breakdown_contains_critical():
    assert "critical" in format_severity_breakdown(_SEV_ERRORS)


def test_format_severity_breakdown_contains_major():
    assert "major" in format_severity_breakdown(_SEV_ERRORS)


def test_format_severity_breakdown_contains_major_minor():
    assert "major_minor" in format_severity_breakdown(_SEV_ERRORS)


def test_format_severity_breakdown_contains_minor():
    assert "minor" in format_severity_breakdown(_SEV_ERRORS)


def test_format_severity_breakdown_contains_counts():
    result = format_severity_breakdown(_SEV_ERRORS)
    assert "2" in result  # major appears twice
    assert "1" in result  # critical, major_minor, minor each once


def test_format_severity_breakdown_empty_no_crash():
    assert isinstance(format_severity_breakdown([]), str)


def test_format_severity_breakdown_empty_no_absent_severities():
    result = format_severity_breakdown([])
    assert "critical" not in result
    assert "major" not in result


def test_format_severity_breakdown_only_present_severities():
    errors = [{"severity": "critical"}, {"severity": "critical"}]
    result = format_severity_breakdown(errors)
    assert "critical" in result
    assert "minor" not in result
