"""Tests for generate_benchmark_report.py."""

import pytest

from eval.generate_benchmark_report import (
    esc,
    fmt,
    generic_table,
    pct,
    render_confusion_matrix,
    render_criterion_type_summary,
    render_error_examples,
    render_errors_by_type,
    render_generated_files,
    render_title_and_disclaimer,
)


# ── esc ──────────────────────────────────────────────────────────────────────

def test_esc_passthrough():
    assert esc("hello") == "hello"


def test_esc_ampersand():
    assert esc("a & b") == "a &amp; b"


def test_esc_less_than():
    assert esc("a < b") == "a &lt; b"


def test_esc_greater_than():
    assert esc("a > b") == "a &gt; b"


def test_esc_combined():
    assert esc("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"


def test_esc_non_string():
    assert esc(42) == "42"


# ── fmt ──────────────────────────────────────────────────────────────────────

def test_fmt_default_decimals():
    assert fmt(0.12345) == "0.123"


def test_fmt_custom_decimals():
    assert fmt(0.5, 1) == "0.5"


def test_fmt_zero():
    assert fmt(0.0) == "0.000"


def test_fmt_one():
    assert fmt(1.0) == "1.000"


# ── pct ──────────────────────────────────────────────────────────────────────

def test_pct_half():
    assert pct(0.5) == "50.0%"


def test_pct_zero():
    assert pct(0.0) == "0.0%"


def test_pct_one():
    assert pct(1.0) == "100.0%"


def test_pct_decimal():
    assert pct(0.753) == "75.3%"


# ── generic_table ─────────────────────────────────────────────────────────────

def test_generic_table_headers_present():
    html = generic_table(["col_a", "col_b"], [["x", "y"]])
    assert "<th>col_a</th>" in html
    assert "<th>col_b</th>" in html


def test_generic_table_cell_values():
    html = generic_table(["h1", "h2"], [["foo", "bar"], ["baz", "qux"]])
    assert "<td>foo</td>" in html
    assert "<td>qux</td>" in html


def test_generic_table_escapes_html():
    html = generic_table(["h"], [["<script>"]])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_generic_table_empty_rows():
    html = generic_table(["h1"], [])
    assert "<th>h1</th>" in html
    assert "<tbody></tbody>" in html


# ── render_confusion_matrix ───────────────────────────────────────────────────

def test_render_confusion_matrix_empty():
    assert render_confusion_matrix({}) == ""


def test_render_confusion_matrix_no_key():
    assert render_confusion_matrix({"accuracy": 0.9}) == ""


def test_render_confusion_matrix_contains_labels():
    metrics = {
        "confusion_matrix": {
            "eligible": {"eligible": 5, "not_eligible": 1},
            "not_eligible": {"eligible": 0, "not_eligible": 4},
        }
    }
    html = render_confusion_matrix(metrics)
    assert "eligible" in html
    assert "not_eligible" in html
    assert "Confusion Matrix" in html


def test_render_confusion_matrix_counts():
    metrics = {
        "confusion_matrix": {
            "eligible": {"eligible": 7, "not_eligible": 2},
            "not_eligible": {"eligible": 1, "not_eligible": 3},
        }
    }
    html = render_confusion_matrix(metrics)
    assert ">7<" in html
    assert ">2<" in html
    assert ">1<" in html
    assert ">3<" in html


# ── render_errors_by_type ─────────────────────────────────────────────────────

def test_render_errors_by_type_counts_error_type():
    error_analysis = [
        {"error_type": "false_positive"},
        {"error_type": "false_positive"},
        {"error_type": "missed_exclusion"},
    ]
    predictions = []
    html = render_errors_by_type(error_analysis, predictions)
    assert "false_positive" in html
    assert ">2<" in html
    assert "missed_exclusion" in html
    assert ">1<" in html


def test_render_errors_by_type_unknown_fallback():
    error_analysis = [{"error_type": None}, {"other_field": "x"}]
    html = render_errors_by_type(error_analysis, [])
    assert "unknown" in html


def test_render_errors_by_type_none_error_analysis():
    html = render_errors_by_type(None, [])
    assert "not available" in html


def test_render_errors_by_type_gold_predicted_pair():
    predictions = [
        {"gold_label": "eligible", "predicted_label": "not_eligible"},
        {"gold_label": "eligible", "predicted_label": "not_eligible"},
        {"gold_label": "not_eligible", "predicted_label": "eligible"},
    ]
    html = render_errors_by_type(None, predictions)
    assert "Errors by Gold/Predicted Pair" in html
    assert ">2<" in html
    assert ">1<" in html


def test_render_errors_by_type_correct_pairs_excluded():
    predictions = [
        {"gold_label": "eligible", "predicted_label": "eligible"},  # correct, not an error
        {"gold_label": "eligible", "predicted_label": "not_eligible"},
    ]
    html = render_errors_by_type(None, predictions)
    # Only 1 error pair should appear
    assert "Errors by Gold/Predicted Pair" in html
    assert ">1<" in html


def test_render_errors_by_type_no_predictions_no_pairs():
    html = render_errors_by_type(None, [])
    assert "Errors by Gold/Predicted Pair" not in html


# ── render_error_examples ─────────────────────────────────────────────────────

def test_render_error_examples_no_errors():
    html = render_error_examples([])
    assert "No errors" in html


def test_render_error_examples_only_correct():
    predictions = [{"gold_label": "eligible", "predicted_label": "eligible"}]
    html = render_error_examples(predictions)
    assert "No errors" in html


def test_render_error_examples_shows_errors():
    predictions = [
        {"gold_label": "eligible", "predicted_label": "not_eligible",
         "patient_id": "p1", "trial_id": "t1",
         "matcher_explanation": "reason", "gold_rationale": "rationale",
         "blocking_criteria": [], "uncertain_criteria": []},
    ]
    html = render_error_examples(predictions)
    assert "p1" in html
    assert "t1" in html
    assert "eligible" in html
    assert "not_eligible" in html


def test_render_error_examples_caps_at_n():
    predictions = [
        {"gold_label": "eligible", "predicted_label": "not_eligible",
         "patient_id": f"p{i}", "trial_id": "t1",
         "blocking_criteria": [], "uncertain_criteria": []}
        for i in range(20)
    ]
    html = render_error_examples(predictions, n=5)
    assert "p0" in html
    assert "p4" in html
    assert "p5" not in html


def test_render_error_examples_escapes_content():
    predictions = [
        {"gold_label": "eligible", "predicted_label": "not_eligible",
         "patient_id": "<evil>", "trial_id": "t1",
         "blocking_criteria": [], "uncertain_criteria": []}
    ]
    html = render_error_examples(predictions)
    assert "<evil>" not in html
    assert "&lt;evil&gt;" in html


# ── render_title_and_disclaimer ───────────────────────────────────────────────

def test_disclaimer_synthetic_patients():
    html = render_title_and_disclaimer({})
    assert "synthetic patient cases" in html


def test_disclaimer_draft_labels():
    html = render_title_and_disclaimer({})
    assert "draft LLM-reviewed labels" in html


def test_disclaimer_not_for_clinical_use():
    html = render_title_and_disclaimer({})
    assert "Not for clinical use" in html


def test_disclaimer_label_source_shown():
    html = render_title_and_disclaimer({"label_source": "data/processed/labels.json"})
    assert "data/processed/labels.json" in html


# ── render_criterion_type_summary ─────────────────────────────────────────────

def test_criterion_type_summary_rounds_floats():
    rows = [{"criterion_type": "inclusion", "pair_accuracy": 0.66666666}]
    html = render_criterion_type_summary(rows)
    assert "0.667" in html
    assert "0.66666666" not in html


def test_criterion_type_summary_integer_not_changed():
    rows = [{"criterion_type": "exclusion", "total_criteria": 10}]
    html = render_criterion_type_summary(rows)
    assert ">10<" in html


def test_criterion_type_summary_empty():
    html = render_criterion_type_summary([])
    assert "No data" in html


def test_criterion_type_summary_multiple_float_columns():
    rows = [{"criterion_type": "inclusion", "pair_accuracy": 0.1, "decision_met": 5}]
    html = render_criterion_type_summary(rows)
    assert "0.100" in html
    assert ">5<" in html


# ── render_generated_files ────────────────────────────────────────────────────

def test_generated_files_results_json():
    html = render_generated_files()
    assert "data/processed/results_llm_reviewed.json" in html


def test_generated_files_criterion_type_csv():
    html = render_generated_files()
    assert "data/processed/criterion_type_summary.csv" in html


def test_generated_files_report_html():
    html = render_generated_files()
    assert "reports/benchmark_report.html" in html


def test_generated_files_uses_code_tags():
    html = render_generated_files()
    assert "<code>" in html
    assert "</code>" in html


def test_generated_files_paths_inside_code_tags():
    html = render_generated_files()
    assert "<code>data/processed/results_llm_reviewed.json</code>" in html
    assert "<code>reports/benchmark_report.html</code>" in html
