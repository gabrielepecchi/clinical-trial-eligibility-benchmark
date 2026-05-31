"""Tests for CSV helpers in run_llm_reviewed_benchmark.py."""

from run_llm_reviewed_benchmark import (
    build_llm_reviewed_csv_rows,
    write_llm_reviewed_csv_rows,
)

_RECORD = {
    "patient_id": "P001",
    "trial_id": "T001",
    "gold_label": "eligible",
    "predicted_label": "eligible",
    "label_status": "llm_reviewed_needs_spotcheck",
    "confidence": 0.9,
    "matched_facts": ["age >= 18", "diagnosis confirmed"],
    "blocking_criteria": [],
    "uncertain_criteria": ["mri compatibility"],
    "matcher_explanation": "Patient meets all criteria.",
    "gold_rationale": "Patient is eligible based on review.",
}

_RECORD_WRONG = {
    "patient_id": "P002",
    "trial_id": "T001",
    "gold_label": "eligible",
    "predicted_label": "not_eligible",
    "label_status": "llm_reviewed_needs_spotcheck",
    "confidence": 0.5,
    "matched_facts": [],
    "blocking_criteria": ["age < 18"],
    "uncertain_criteria": [],
    "matcher_explanation": "Blocked by age criterion.",
    "gold_rationale": "Patient is actually eligible.",
}


def test_build_llm_reviewed_csv_rows_empty_input():
    assert build_llm_reviewed_csv_rows([]) == []


def test_build_llm_reviewed_csv_rows_returns_list():
    assert isinstance(build_llm_reviewed_csv_rows([_RECORD]), list)


def test_build_llm_reviewed_csv_rows_length():
    assert len(build_llm_reviewed_csv_rows([_RECORD, _RECORD_WRONG])) == 2


def test_build_llm_reviewed_csv_rows_row_is_dict():
    assert isinstance(build_llm_reviewed_csv_rows([_RECORD])[0], dict)


def test_build_llm_reviewed_csv_rows_correct_true():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["correct"] is True


def test_build_llm_reviewed_csv_rows_correct_false():
    assert build_llm_reviewed_csv_rows([_RECORD_WRONG])[0]["correct"] is False


def test_build_llm_reviewed_csv_rows_matched_facts_joined():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["matched_facts"] == "age >= 18; diagnosis confirmed"


def test_build_llm_reviewed_csv_rows_blocking_criteria_joined():
    assert build_llm_reviewed_csv_rows([_RECORD_WRONG])[0]["blocking_criteria"] == "age < 18"


def test_build_llm_reviewed_csv_rows_uncertain_criteria_joined():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["uncertain_criteria"] == "mri compatibility"


def test_build_llm_reviewed_csv_rows_empty_list_fields_are_strings():
    row = build_llm_reviewed_csv_rows([_RECORD_WRONG])[0]
    assert isinstance(row["matched_facts"], str)
    assert isinstance(row["uncertain_criteria"], str)


def test_build_llm_reviewed_csv_rows_patient_id():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["patient_id"] == "P001"


def test_build_llm_reviewed_csv_rows_gold_label():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["gold_label"] == "eligible"


def test_build_llm_reviewed_csv_rows_confidence():
    assert build_llm_reviewed_csv_rows([_RECORD])[0]["confidence"] == 0.9


def test_write_llm_reviewed_csv_rows_creates_file(tmp_path):
    out = tmp_path / "results.csv"
    write_llm_reviewed_csv_rows(build_llm_reviewed_csv_rows([_RECORD]), out)
    assert out.exists()


def test_write_llm_reviewed_csv_rows_header_present(tmp_path):
    out = tmp_path / "results.csv"
    write_llm_reviewed_csv_rows(build_llm_reviewed_csv_rows([_RECORD]), out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert "patient_id" in first_line
    assert "gold_label" in first_line


def test_write_llm_reviewed_csv_rows_empty_creates_header_only(tmp_path):
    out = tmp_path / "empty.csv"
    write_llm_reviewed_csv_rows([], out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "patient_id" in lines[0]


# --- criterion-level CSV helpers ---

from run_llm_reviewed_benchmark import (
    build_criterion_level_csv_rows,
    write_criterion_level_csv_rows,
)

_CR1 = {"criterion_text": "Age >= 18", "criterion_type": "inclusion", "decision": "met", "reason": "Patient is 25."}
_CR2 = {"criterion_text": "No prior chemo", "criterion_type": "exclusion", "decision": "not_met", "reason": "No chemo history."}

_PRED_WITH_CRITERIA = {
    "patient_id": "P001",
    "trial_id": "T001",
    "gold_label": "eligible",
    "predicted_label": "eligible",
    "criterion_results": [_CR1, _CR2],
}

_PRED_NO_CRITERIA = {
    "patient_id": "P002",
    "trial_id": "T002",
    "gold_label": "not_eligible",
    "predicted_label": "not_eligible",
    "criterion_results": [],
}


def test_build_criterion_level_csv_rows_empty_input():
    assert build_criterion_level_csv_rows([]) == []


def test_build_criterion_level_csv_rows_two_criteria_two_rows():
    assert len(build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])) == 2


def test_build_criterion_level_csv_rows_no_criteria_no_rows():
    assert build_criterion_level_csv_rows([_PRED_NO_CRITERIA]) == []


def test_build_criterion_level_csv_rows_row_is_dict():
    assert isinstance(build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0], dict)


def test_build_criterion_level_csv_rows_patient_id():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["patient_id"] == "P001"


def test_build_criterion_level_csv_rows_trial_id():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["trial_id"] == "T001"


def test_build_criterion_level_csv_rows_gold_label():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["gold_label"] == "eligible"


def test_build_criterion_level_csv_rows_predicted_label():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["predicted_label"] == "eligible"


def test_build_criterion_level_csv_rows_criterion():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["criterion"] == "Age >= 18"


def test_build_criterion_level_csv_rows_criterion_type():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["criterion_type"] == "inclusion"


def test_build_criterion_level_csv_rows_decision():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["decision"] == "met"


def test_build_criterion_level_csv_rows_reason():
    assert build_criterion_level_csv_rows([_PRED_WITH_CRITERIA])[0]["reason"] == "Patient is 25."


def test_write_criterion_level_csv_rows_creates_file(tmp_path):
    out = tmp_path / "criterion.csv"
    write_criterion_level_csv_rows(build_criterion_level_csv_rows([_PRED_WITH_CRITERIA]), out)
    assert out.exists()


def test_write_criterion_level_csv_rows_header_present(tmp_path):
    out = tmp_path / "criterion.csv"
    write_criterion_level_csv_rows(build_criterion_level_csv_rows([_PRED_WITH_CRITERIA]), out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert "patient_id" in first_line
    assert "criterion" in first_line


def test_write_criterion_level_csv_rows_empty_creates_header_only(tmp_path):
    out = tmp_path / "criterion_empty.csv"
    write_criterion_level_csv_rows([], out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "patient_id" in lines[0]
