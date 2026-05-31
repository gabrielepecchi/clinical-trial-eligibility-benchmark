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


# --- build_safety_uncertainty_summary ---

from run_llm_reviewed_benchmark import build_safety_uncertainty_summary

_SU_RECORDS = [
    # correct eligible
    {"gold_label": "eligible",     "predicted_label": "eligible"},
    # overly_conservative: gold eligible, pred not_eligible
    {"gold_label": "eligible",     "predicted_label": "not_eligible"},
    # unsafe_eligible: gold not_eligible, pred eligible
    {"gold_label": "not_eligible", "predicted_label": "eligible"},
    # correct not_eligible
    {"gold_label": "not_eligible", "predicted_label": "not_eligible"},
    # uncertainty_error: gold unclear, pred eligible
    {"gold_label": "unclear",      "predicted_label": "eligible"},
    # uncertainty_error: gold unclear, pred not_eligible
    {"gold_label": "unclear",      "predicted_label": "not_eligible"},
    # correct unclear
    {"gold_label": "unclear",      "predicted_label": "unclear"},
    # predicted unclear, gold not_eligible (precision miss)
    {"gold_label": "not_eligible", "predicted_label": "unclear"},
]


def test_build_safety_uncertainty_summary_empty():
    result = build_safety_uncertainty_summary([])
    assert result["total_predictions"] == 0
    assert result["unsafe_eligible_errors"] == 0
    assert result["uncertainty_errors"] == 0
    assert result["overly_conservative_errors"] == 0


def test_build_safety_uncertainty_summary_returns_dict():
    assert isinstance(build_safety_uncertainty_summary(_SU_RECORDS), dict)


def test_build_safety_uncertainty_summary_keys():
    result = build_safety_uncertainty_summary(_SU_RECORDS)
    for key in [
        "total_predictions", "unsafe_eligible_errors", "uncertainty_errors",
        "overly_conservative_errors", "unclear_recall", "unclear_precision",
        "overcommitment_rate",
    ]:
        assert key in result


def test_build_safety_uncertainty_summary_total():
    assert build_safety_uncertainty_summary(_SU_RECORDS)["total_predictions"] == 8


def test_build_safety_uncertainty_summary_unsafe_eligible_errors():
    assert build_safety_uncertainty_summary(_SU_RECORDS)["unsafe_eligible_errors"] == 1


def test_build_safety_uncertainty_summary_uncertainty_errors():
    assert build_safety_uncertainty_summary(_SU_RECORDS)["uncertainty_errors"] == 2


def test_build_safety_uncertainty_summary_overly_conservative_errors():
    assert build_safety_uncertainty_summary(_SU_RECORDS)["overly_conservative_errors"] == 1


def test_build_safety_uncertainty_summary_unclear_recall():
    # gold unclear = 3 (indices 4,5,6); predicted unclear among those = 1
    result = build_safety_uncertainty_summary(_SU_RECORDS)
    assert result["unclear_recall"] == 1 / 3


def test_build_safety_uncertainty_summary_unclear_precision():
    # predicted unclear = 2 (indices 6,7); gold unclear among those = 1
    result = build_safety_uncertainty_summary(_SU_RECORDS)
    assert result["unclear_precision"] == 1 / 2


def test_build_safety_uncertainty_summary_overcommitment_rate():
    # gold unclear = 3; predicted eligible/not_eligible among those = 2
    result = build_safety_uncertainty_summary(_SU_RECORDS)
    assert result["overcommitment_rate"] == 2 / 3


def test_build_safety_uncertainty_summary_no_unclear_gold_rates_are_zero():
    records = [
        {"gold_label": "eligible",     "predicted_label": "eligible"},
        {"gold_label": "not_eligible", "predicted_label": "not_eligible"},
    ]
    result = build_safety_uncertainty_summary(records)
    assert result["unclear_recall"] == 0
    assert result["overcommitment_rate"] == 0


def test_build_safety_uncertainty_summary_no_unclear_predicted_precision_is_zero():
    records = [
        {"gold_label": "eligible",     "predicted_label": "eligible"},
        {"gold_label": "unclear",      "predicted_label": "eligible"},
    ]
    result = build_safety_uncertainty_summary(records)
    assert result["unclear_precision"] == 0


# --- build_benchmark_output ---

from run_llm_reviewed_benchmark import build_benchmark_output

_META = {"label_source": "labels.json", "evaluated_pairs": 10, "skipped_pairs": 0}
_METRICS = {"accuracy": 0.9, "macro_f1": 0.88}
_SU_SUMMARY = {"total_predictions": 10, "unsafe_eligible_errors": 1}
_PREDS = [{"patient_id": "P001", "trial_id": "T001"}]


def test_build_benchmark_output_returns_dict():
    assert isinstance(build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS), dict)


def test_build_benchmark_output_keys():
    result = build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS)
    assert set(result.keys()) == {"metadata", "metrics", "safety_uncertainty_summary", "predictions"}


def test_build_benchmark_output_metadata():
    assert build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS)["metadata"] is _META


def test_build_benchmark_output_metrics():
    assert build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS)["metrics"] is _METRICS


def test_build_benchmark_output_safety_uncertainty_summary():
    assert build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS)["safety_uncertainty_summary"] is _SU_SUMMARY


def test_build_benchmark_output_predictions():
    assert build_benchmark_output(_META, _METRICS, _SU_SUMMARY, _PREDS)["predictions"] is _PREDS


def test_build_benchmark_output_summary_preserved_exactly():
    summary = {"total_predictions": 5, "unsafe_eligible_errors": 2, "unclear_recall": 0.5}
    result = build_benchmark_output(_META, _METRICS, summary, _PREDS)
    assert result["safety_uncertainty_summary"] == summary
