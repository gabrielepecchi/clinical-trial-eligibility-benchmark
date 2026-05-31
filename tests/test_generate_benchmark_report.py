"""Tests for generate_benchmark_report.py."""

import json
import pytest

import eval.generate_benchmark_report as gbr
from eval.generate_benchmark_report import (
    esc,
    fmt,
    generic_table,
    pct,
    render_confusion_matrix,
    render_criterion_level_examples,
    render_criterion_type_summary,
    render_error_examples,
    render_errors_by_type,
    render_generated_files,
    render_key_takeaways,
    render_label_distribution,
    render_model_behavior_summary,
    render_quick_metrics_cards,
    render_report_footer,
    render_report_navigation,
    render_safety_critical_summary,
    render_title_and_disclaimer,
    render_top_error_types,
    render_uncertainty_signals,
    render_worst_error_examples,
    section,
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


# ── render_top_error_types ────────────────────────────────────────────────────

def test_top_error_types_counts():
    error_analysis = [
        {"error_type": "false_positive"},
        {"error_type": "false_positive"},
        {"error_type": "missed_exclusion"},
    ]
    html = render_top_error_types(error_analysis)
    assert "false_positive" in html
    assert ">2<" in html
    assert "missed_exclusion" in html
    assert ">1<" in html


def test_top_error_types_limits_to_five():
    error_analysis = [{"error_type": f"type_{i}"} for i in range(10)]
    html = render_top_error_types(error_analysis)
    # With one record each, any 5 should appear; type_5..9 may be excluded
    # Count <td> cells for error_type column — each row has 2 cells
    row_count = html.count("<tr>") - 1  # subtract header row
    assert row_count <= 5


def test_top_error_types_top_ranked_appear():
    error_analysis = (
        [{"error_type": "common"}] * 10
        + [{"error_type": "rare_a"}] * 2
        + [{"error_type": "rare_b"}] * 1
        + [{"error_type": f"singleton_{i}"} for i in range(10)]
    )
    html = render_top_error_types(error_analysis)
    assert "common" in html
    assert ">10<" in html


def test_top_error_types_none_input():
    html = render_top_error_types(None)
    assert "No error analysis data available" in html


def test_top_error_types_empty_list():
    html = render_top_error_types([])
    assert "No error analysis data available" in html


def test_top_error_types_missing_error_type_field():
    error_analysis = [{"other_field": "x"}, {"other_field": "y"}]
    html = render_top_error_types(error_analysis)
    assert "unknown" in html
    assert ">2<" in html


# ── render_key_takeaways ──────────────────────────────────────────────────────

_METRICS = {"accuracy": 0.75, "macro_f1": 0.72}
_SAFETY = {"unsafe_eligible_errors": 3, "total_predictions": 100, "overcommitment_rate": 0.25}


def test_key_takeaways_accuracy_and_f1():
    html = render_key_takeaways(_METRICS, {}, None)
    assert "75.0%" in html
    assert "0.720" in html


def test_key_takeaways_unsafe_eligible_errors():
    html = render_key_takeaways({}, _SAFETY, None)
    assert "3" in html
    assert "not_eligible" in html


def test_key_takeaways_overcommitment_rate():
    html = render_key_takeaways({}, _SAFETY, None)
    assert "0.250" in html
    assert "overcommitment" in html.lower()


def test_key_takeaways_largest_error_type():
    error_analysis = [
        {"error_type": "false_positive"},
        {"error_type": "false_positive"},
        {"error_type": "missed_exclusion"},
    ]
    html = render_key_takeaways({}, {}, error_analysis)
    assert "false_positive" in html


def test_key_takeaways_empty_returns_empty_string():
    html = render_key_takeaways({}, {}, None)
    assert html == ""


def test_key_takeaways_partial_metrics_skips_missing():
    html = render_key_takeaways({"accuracy": 0.8}, {}, None)
    # macro_f1 missing so accuracy+f1 bullet should be skipped
    assert "80.0%" not in html


# ── render_label_distribution ─────────────────────────────────────────────────

_PREDICTIONS = [
    {"gold_label": "eligible", "predicted_label": "eligible"},
    {"gold_label": "eligible", "predicted_label": "not_eligible"},
    {"gold_label": "not_eligible", "predicted_label": "not_eligible"},
    {"gold_label": "unclear", "predicted_label": "unclear"},
]


def test_label_distribution_gold_counts():
    html = render_label_distribution(_PREDICTIONS)
    # eligible gold count = 2
    assert ">2<" in html


def test_label_distribution_predicted_counts():
    html = render_label_distribution(_PREDICTIONS)
    # not_eligible predicted count = 2 (one correct + one error)
    assert ">2<" in html


def test_label_distribution_all_three_labels():
    html = render_label_distribution(_PREDICTIONS)
    assert "eligible" in html
    assert "not_eligible" in html
    assert "unclear" in html


def test_label_distribution_columns():
    html = render_label_distribution(_PREDICTIONS)
    assert "gold_count" in html
    assert "predicted_count" in html


def test_label_distribution_empty_predictions():
    html = render_label_distribution([])
    assert "No prediction data available" in html


def test_label_distribution_zero_counts_included():
    predictions = [{"gold_label": "eligible", "predicted_label": "eligible"}]
    html = render_label_distribution(predictions)
    # not_eligible and unclear should show 0
    assert ">0<" in html


# ── render_worst_error_examples ───────────────────────────────────────────────

def _make_pred(gold: str, pred: str, pid: str = "p1", tid: str = "t1") -> dict:
    return {
        "gold_label": gold, "predicted_label": pred,
        "patient_id": pid, "trial_id": tid,
        "matcher_explanation": "", "gold_rationale": "",
        "blocking_criteria": [], "uncertain_criteria": [],
    }


def test_worst_errors_no_errors():
    html = render_worst_error_examples([])
    assert "No errors" in html


def test_worst_errors_correct_only():
    predictions = [_make_pred("eligible", "eligible")]
    html = render_worst_error_examples(predictions)
    assert "No errors" in html


def test_worst_errors_critical_first():
    predictions = [
        _make_pred("eligible", "not_eligible", pid="p_minor"),
        _make_pred("not_eligible", "eligible", pid="p_critical"),
    ]
    html = render_worst_error_examples(predictions)
    assert html.index("p_critical") < html.index("p_minor")


def test_worst_errors_severity_label_shown():
    predictions = [_make_pred("not_eligible", "eligible")]
    html = render_worst_error_examples(predictions)
    assert "critical" in html


def test_worst_errors_major_before_minor():
    predictions = [
        _make_pred("eligible", "not_eligible", pid="p_minor"),
        _make_pred("eligible", "unclear", pid="p_major"),
    ]
    html = render_worst_error_examples(predictions)
    assert html.index("p_major") < html.index("p_minor")


def test_worst_errors_caps_at_n():
    predictions = [_make_pred("not_eligible", "eligible", pid=f"p{i}") for i in range(20)]
    html = render_worst_error_examples(predictions, n=5)
    assert "p0" in html
    assert "p4" in html
    assert "p5" not in html


def test_worst_errors_shows_patient_and_trial():
    predictions = [_make_pred("not_eligible", "eligible", pid="pat42", tid="trial7")]
    html = render_worst_error_examples(predictions)
    assert "pat42" in html
    assert "trial7" in html


# ── render_safety_critical_summary ────────────────────────────────────────────

_SAFETY_DATA = {
    "unsafe_eligible_errors": 4,
    "overcommitment_rate": 0.15,
}
_ERROR_SEV_DATA = {
    "critical_errors": 4,
    "critical_error_rate": 0.04,
    "major_errors": 7,
    "major_error_rate": 0.07,
}


def test_safety_critical_unsafe_eligible_errors():
    html = render_safety_critical_summary(_SAFETY_DATA, _ERROR_SEV_DATA)
    assert ">4<" in html
    assert "not_eligible" in html


def test_safety_critical_critical_errors_and_rate():
    html = render_safety_critical_summary(_SAFETY_DATA, _ERROR_SEV_DATA)
    assert "Critical errors" in html
    assert "4.0%" in html


def test_safety_critical_major_errors():
    html = render_safety_critical_summary(_SAFETY_DATA, _ERROR_SEV_DATA)
    assert "Major errors" in html
    assert ">7<" in html


def test_safety_critical_overcommitment_rate():
    html = render_safety_critical_summary(_SAFETY_DATA, _ERROR_SEV_DATA)
    assert "15.0%" in html
    assert "overcommitment" in html.lower()


def test_safety_critical_empty_inputs():
    html = render_safety_critical_summary({}, {})
    assert "No safety-critical data available" in html


# ── render_uncertainty_signals ────────────────────────────────────────────────

_UC_PREDICTIONS = [
    {"gold_label": "eligible", "predicted_label": "eligible",
     "uncertain_criteria": ["age unknown", "prior tx unclear"]},
    {"gold_label": "unclear", "predicted_label": "eligible",
     "uncertain_criteria": ["diagnosis unclear"]},
    {"gold_label": "unclear", "predicted_label": "unclear",
     "uncertain_criteria": []},
    {"gold_label": "not_eligible", "predicted_label": "not_eligible",
     "uncertain_criteria": None},
]


def test_uncertainty_signals_total_uncertain_criteria():
    html = render_uncertainty_signals(_UC_PREDICTIONS)
    # 2 + 1 + 0 + 0 = 3
    assert ">3<" in html


def test_uncertainty_signals_predictions_with_uncertain():
    html = render_uncertainty_signals(_UC_PREDICTIONS)
    # 2 records have at least one uncertain criterion
    assert "Predictions with uncertain criteria" in html
    assert ">2<" in html


def test_uncertainty_signals_overcommitted_unclear():
    html = render_uncertainty_signals(_UC_PREDICTIONS)
    # 1 record: gold unclear, predicted eligible
    assert "Overcommitted" in html or "overcommitted" in html.lower()
    assert ">1<" in html


def test_uncertainty_signals_empty_predictions():
    html = render_uncertainty_signals([])
    assert "No prediction data available" in html


def test_uncertainty_signals_unclear_gold_count():
    html = render_uncertainty_signals(_UC_PREDICTIONS)
    assert "Unclear gold" in html or "unclear gold" in html.lower()
    # 2 records have gold_label unclear
    assert ">2<" in html


# ── render_model_behavior_summary ─────────────────────────────────────────────

_MB_PREDICTIONS = [
    {"predicted_label": "eligible"},
    {"predicted_label": "eligible"},
    {"predicted_label": "not_eligible"},
    {"predicted_label": "unclear"},
]


def test_model_behavior_total_predictions():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Total predictions" in html
    assert ">4<" in html


def test_model_behavior_predicted_eligible_count():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Predicted eligible" in html
    assert ">2<" in html


def test_model_behavior_predicted_not_eligible_count():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Predicted not_eligible" in html
    assert ">1<" in html


def test_model_behavior_predicted_unclear_count():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Predicted unclear" in html


def test_model_behavior_eligible_rate():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Eligible prediction rate" in html
    assert "50.0%" in html


def test_model_behavior_unclear_rate():
    html = render_model_behavior_summary(_MB_PREDICTIONS)
    assert "Unclear prediction rate" in html
    assert "25.0%" in html


def test_model_behavior_empty_predictions():
    html = render_model_behavior_summary([])
    assert "No prediction data available" in html


# ── render_report_navigation ──────────────────────────────────────────────────

def test_report_navigation_is_list():
    html = render_report_navigation()
    assert "<ul>" in html
    assert "<li>" in html


@pytest.mark.parametrize("title,anchor", [
    ("Global Metrics", "global-metrics"),
    ("Label Distribution", "label-distribution"),
    ("Safety-Critical Summary", "safety-critical-summary"),
    ("Worst Error Examples", "worst-error-examples"),
    ("Criterion-Level Examples", "criterion-level-examples"),
])
def test_report_navigation_sections_and_anchors(title, anchor):
    html = render_report_navigation()
    assert title in html
    assert f"href='#{anchor}'" in html or f'href="#{anchor}"' in html


# ── render_quick_metrics_cards ────────────────────────────────────────────────

_QM_METRICS = {"accuracy": 0.82, "macro_f1": 0.79}
_QM_SAFETY = {"unsafe_eligible_errors": 5, "overcommitment_rate": 0.12}
_QM_ERROR_SEV = {"critical_errors": 5, "major_errors": 9}


def test_quick_metrics_accuracy():
    html = render_quick_metrics_cards(_QM_METRICS, {}, {})
    assert "Accuracy" in html
    assert "82.0%" in html


def test_quick_metrics_macro_f1():
    html = render_quick_metrics_cards(_QM_METRICS, {}, {})
    assert "Macro F1" in html
    assert "0.790" in html


def test_quick_metrics_unsafe_eligible_errors():
    html = render_quick_metrics_cards({}, _QM_SAFETY, {})
    assert "Unsafe Eligible Errors" in html
    assert ">5<" in html


def test_quick_metrics_critical_errors():
    html = render_quick_metrics_cards({}, {}, _QM_ERROR_SEV)
    assert "Critical Errors" in html
    assert ">5<" in html


def test_quick_metrics_major_errors():
    html = render_quick_metrics_cards({}, {}, _QM_ERROR_SEV)
    assert "Major Errors" in html
    assert ">9<" in html


def test_quick_metrics_overcommitment_rate():
    html = render_quick_metrics_cards({}, _QM_SAFETY, {})
    assert "Overcommitment Rate" in html
    assert "12.0%" in html


def test_quick_metrics_empty_inputs():
    html = render_quick_metrics_cards({}, {}, {})
    assert "No quick metrics available" in html


def test_quick_metrics_cards_html_structure():
    html = render_quick_metrics_cards(_QM_METRICS, _QM_SAFETY, _QM_ERROR_SEV)
    assert "class='cards'" in html
    assert "class='card'" in html


# ── section() anchor ids ──────────────────────────────────────────────────────

def test_section_global_metrics_has_anchor_id():
    html = section("Global Metrics", "body text")
    assert 'id="global-metrics"' in html or "id='global-metrics'" in html


def test_section_safety_critical_summary_anchor_id():
    html = section("Safety-Critical Summary", "body text")
    assert 'id="safety-critical-summary"' in html or "id='safety-critical-summary'" in html


def test_section_slash_in_title_becomes_hyphen():
    html = section("Missing Information / Uncertainty Signals", "body")
    assert "id='missing-information-uncertainty-signals'" in html or \
           'id="missing-information-uncertainty-signals"' in html


# ── render_report_footer ──────────────────────────────────────────────────────

def test_report_footer_generated_locally():
    html = render_report_footer()
    assert "generated locally" in html


def test_report_footer_synthetic_patients():
    html = render_report_footer()
    assert "synthetic patients" in html


def test_report_footer_draft_llm_reviewed_labels():
    html = render_report_footer()
    assert "draft LLM-reviewed labels" in html


def test_report_footer_not_for_clinical_use():
    html = render_report_footer()
    assert "Not for clinical use" in html or "not for clinical use" in html.lower()


# ── render_criterion_level_examples ──────────────────────────────────────────

def _make_crit_pred(pid: str, tid: str, gold: str = "eligible", pred: str = "eligible",
                    criteria: list | None = None) -> dict:
    return {
        "patient_id": pid,
        "trial_id": tid,
        "gold_label": gold,
        "predicted_label": pred,
        "criterion_results": criteria if criteria is not None else [],
    }


def _crit(ctype: str = "inclusion", decision: str = "met",
          text: str = "Age >= 18", evidence: str = "", confidence: float | None = None) -> dict:
    c: dict = {"criterion_type": ctype, "decision": decision, "criterion_text": text}
    if evidence:
        c["evidence"] = evidence
    if confidence is not None:
        c["confidence"] = confidence
    return c


_CRIT_PREDS = [
    _make_crit_pred("p1", "t1", criteria=[_crit("inclusion", "met", "Age >= 18")]),
    _make_crit_pred("p2", "t2", criteria=[_crit("exclusion", "not_met", "No prior surgery")]),
]


def test_criterion_level_examples_patient_and_trial():
    html = render_criterion_level_examples(_CRIT_PREDS)
    assert "p1" in html
    assert "t1" in html


def test_criterion_level_examples_criterion_type_and_decision():
    html = render_criterion_level_examples(_CRIT_PREDS)
    assert "inclusion" in html
    assert "met" in html


def test_criterion_level_examples_limits_criteria_rows_to_5():
    criteria = [_crit("inclusion", "met", f"Criterion {i}") for i in range(10)]
    preds = [_make_crit_pred("p1", "t1", criteria=criteria)]
    html = render_criterion_level_examples(preds)
    assert "Criterion 4" in html
    assert "Criterion 5" not in html


def test_criterion_level_examples_limits_predictions_to_n():
    preds = [
        _make_crit_pred(f"p{i}", "t1", criteria=[_crit()])
        for i in range(15)
    ]
    html = render_criterion_level_examples(preds, n=5)
    assert "p0" in html
    assert "p4" in html
    assert "p5" not in html


def test_criterion_level_examples_no_criterion_results():
    preds = [{"patient_id": "p1", "trial_id": "t1", "gold_label": "eligible",
              "predicted_label": "eligible"}]
    html = render_criterion_level_examples(preds)
    assert "No criterion-level results available" in html


def test_criterion_level_examples_empty_predictions():
    html = render_criterion_level_examples([])
    assert "No criterion-level results available" in html


def test_criterion_level_examples_escapes_criterion_text_and_evidence():
    preds = [_make_crit_pred("p1", "t1", criteria=[
        {"criterion_type": "inclusion", "decision": "met",
         "criterion_text": "<b>Age >= 18</b>", "evidence": "<script>alert(1)</script>"},
    ])]
    html = render_criterion_level_examples(preds)
    assert "<b>Age >= 18</b>" not in html
    assert "&lt;b&gt;" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_worst_error_examples_escapes_explanation_and_rationale():
    predictions = [_make_pred(
        "not_eligible", "eligible",
        pid="<evil>", tid="t1",
    )]
    predictions[0]["matcher_explanation"] = "<script>xss()</script>"
    predictions[0]["gold_rationale"] = "<b>bad</b>"
    html = render_worst_error_examples(predictions)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>bad</b>" not in html
    assert "&lt;b&gt;" in html


# ── main() smoke test ─────────────────────────────────────────────────────────

def test_main_generates_html_report(tmp_path, monkeypatch):
    results_data = {
        "metadata": {"label_source": "test", "model": "test-model"},
        "metrics": {
            "accuracy": 0.8,
            "macro_precision": 0.79,
            "macro_recall": 0.78,
            "macro_f1": 0.78,
            "per_class": {
                "eligible": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                "not_eligible": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
                "unclear": {"precision": 0.75, "recall": 0.75, "f1": 0.75},
            },
        },
        "safety_uncertainty_summary": {
            "total_predictions": 3,
            "unsafe_eligible_errors": 0,
            "overly_conservative_errors": 0,
            "uncertainty_errors": 0,
            "unclear_recall": 1.0,
            "unclear_precision": 1.0,
            "overcommitment_rate": 0.0,
        },
        "error_severity_summary": {
            "total_predictions": 3,
            "total_errors": 0,
            "critical_errors": 0,
            "major_errors": 0,
            "minor_errors": 0,
            "critical_error_rate": 0.0,
            "major_error_rate": 0.0,
            "minor_error_rate": 0.0,
        },
        "predictions": [
            {"patient_id": "p1", "trial_id": "t1",
             "gold_label": "eligible", "predicted_label": "eligible",
             "blocking_criteria": [], "uncertain_criteria": [], "criterion_results": []},
            {"patient_id": "p2", "trial_id": "t2",
             "gold_label": "not_eligible", "predicted_label": "not_eligible",
             "blocking_criteria": [], "uncertain_criteria": [], "criterion_results": []},
            {"patient_id": "p3", "trial_id": "t3",
             "gold_label": "unclear", "predicted_label": "unclear",
             "blocking_criteria": [], "uncertain_criteria": [], "criterion_results": []},
        ],
        "criterion_type_summary": [
            {"criterion_type": "inclusion", "pair_accuracy": 0.9, "total_criteria": 10},
        ],
    }
    error_analysis_data = [
        {"error_type": "false_positive", "patient_id": "p1", "trial_id": "t1"},
    ]
    criterion_type_data = [
        {"criterion_type": "inclusion", "pair_accuracy": 0.9, "total_criteria": 10},
    ]

    results_file = tmp_path / "results_llm_reviewed.json"
    error_file = tmp_path / "error_analysis_llm_reviewed.json"
    criterion_file = tmp_path / "criterion_type_summary.json"
    report_file = tmp_path / "benchmark_report.html"

    results_file.write_text(json.dumps(results_data), encoding="utf-8")
    error_file.write_text(json.dumps(error_analysis_data), encoding="utf-8")
    criterion_file.write_text(json.dumps(criterion_type_data), encoding="utf-8")

    monkeypatch.setattr(gbr, "RESULTS_FILE", results_file)
    monkeypatch.setattr(gbr, "ERROR_ANALYSIS_FILE", error_file)
    monkeypatch.setattr(gbr, "CRITERION_TYPE_FILE", criterion_file)
    monkeypatch.setattr(gbr, "REPORT_FILE", report_file)

    gbr.main()

    assert report_file.exists()
    html = report_file.read_text(encoding="utf-8")
    assert "LLM-Reviewed Benchmark Report" in html
    assert "Global Metrics" in html
    assert "Criterion Type Summary" in html
    assert "Not for clinical use" in html
    assert "Report Navigation" in html
    assert "Quick Metrics" in html
    assert "Criterion-Level Examples" in html
    assert "href='#global-metrics'" in html or 'href="#global-metrics"' in html
    assert "generated locally" in html


# ── aggregate_criterion_type_summary ──────────────────────────────────────────

import csv as _csv
from eval.summarize_llm_reviewed_errors import aggregate_criterion_type_summary


def _write_criterion_csv(path, rows):
    """Helper: write a minimal criterion_level_results.csv for testing."""
    if not rows:
        path.write_text("criterion_type,gold_decision,decision\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_criterion_type_summary_basic(tmp_path):
    csv_path = tmp_path / "criterion_level_results.csv"
    _write_criterion_csv(csv_path, [
        {"criterion_type": "inclusion", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "inclusion", "gold_decision": "met", "decision": "not_met"},
        {"criterion_type": "exclusion", "gold_decision": "not_met", "decision": "not_met"},
    ])
    result = aggregate_criterion_type_summary(csv_path)
    types = {r["criterion_type"]: r for r in result}
    assert "inclusion" in types
    assert "exclusion" in types
    assert types["inclusion"]["total_criteria"] == 2
    assert types["exclusion"]["total_criteria"] == 1


def test_aggregate_criterion_type_summary_correct_count(tmp_path):
    csv_path = tmp_path / "criterion_level_results.csv"
    _write_criterion_csv(csv_path, [
        {"criterion_type": "inclusion", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "inclusion", "gold_decision": "met", "decision": "not_met"},
        {"criterion_type": "inclusion", "gold_decision": "not_met", "decision": "not_met"},
    ])
    result = aggregate_criterion_type_summary(csv_path)
    inc = next(r for r in result if r["criterion_type"] == "inclusion")
    assert inc["correct_criteria"] == 2
    assert inc["total_criteria"] == 3


def test_aggregate_criterion_type_summary_accuracy(tmp_path):
    csv_path = tmp_path / "criterion_level_results.csv"
    _write_criterion_csv(csv_path, [
        {"criterion_type": "age", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "age", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "age", "gold_decision": "met", "decision": "not_met"},
        {"criterion_type": "age", "gold_decision": "met", "decision": "not_met"},
    ])
    result = aggregate_criterion_type_summary(csv_path)
    age = next(r for r in result if r["criterion_type"] == "age")
    assert age["criterion_accuracy"] == 0.5


def test_aggregate_criterion_type_summary_decision_counts(tmp_path):
    csv_path = tmp_path / "criterion_level_results.csv"
    _write_criterion_csv(csv_path, [
        {"criterion_type": "medication", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "medication", "gold_decision": "not_met", "decision": "not_met"},
        {"criterion_type": "medication", "gold_decision": "met", "decision": "unknown"},
    ])
    result = aggregate_criterion_type_summary(csv_path)
    med = next(r for r in result if r["criterion_type"] == "medication")
    assert med["decision_met"] == 1
    assert med["decision_not_met"] == 1
    assert med["decision_unknown"] == 1


def test_aggregate_criterion_type_summary_missing_file(tmp_path):
    result = aggregate_criterion_type_summary(tmp_path / "nonexistent.csv")
    assert result == []


def test_aggregate_criterion_type_summary_sorted(tmp_path):
    csv_path = tmp_path / "criterion_level_results.csv"
    _write_criterion_csv(csv_path, [
        {"criterion_type": "temporal", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "age", "gold_decision": "met", "decision": "met"},
        {"criterion_type": "medication", "gold_decision": "not_met", "decision": "not_met"},
    ])
    result = aggregate_criterion_type_summary(csv_path)
    types = [r["criterion_type"] for r in result]
    assert types == sorted(types)


def test_criterion_type_summary_renders_real_fields():
    rows = [{
        "criterion_type": "inclusion",
        "total_criteria": 10,
        "correct_criteria": 8,
        "criterion_accuracy": 0.8,
        "decision_met": 6,
        "decision_not_met": 3,
        "decision_unknown": 1,
    }]
    html = render_criterion_type_summary(rows)
    assert "criterion_accuracy" in html
    assert "total_criteria" in html
    assert "correct_criteria" in html
    assert "decision_met" in html
    assert "0.800" in html
    assert ">10<" in html
