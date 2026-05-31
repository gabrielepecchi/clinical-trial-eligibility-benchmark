"""Tests for CSV helpers in summarize_llm_reviewed_errors.py."""

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
