"""Tests for app.eligibility.evidence_span (Task 4)."""

import pytest
from app.eligibility.evidence_span import (
    normalize_text,
    find_keyword_span,
    find_any_keyword_span,
    extract_evidence_spans,
    extract_criterion_evidence,
)

TEXT = (
    "Patient is a 62-year-old male with idiopathic Parkinson Disease. "
    "Current medications include Levodopa and Rasagiline. "
    "No history of Deep Brain Stimulation."
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_lowercases():
    assert normalize_text("Parkinson Disease") == "parkinson disease"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_text_empty():
    assert normalize_text("") == ""


# ---------------------------------------------------------------------------
# find_keyword_span
# ---------------------------------------------------------------------------

def test_find_keyword_span_case_insensitive():
    span = find_keyword_span(TEXT, "levodopa")
    assert span is not None
    assert span["keyword"] == "levodopa"


def test_find_keyword_span_offsets_correct():
    span = find_keyword_span(TEXT, "Rasagiline")
    assert span is not None
    assert TEXT[span["start"]:span["end"]].lower() == "rasagiline"


def test_find_keyword_span_snippet_preserves_casing():
    span = find_keyword_span(TEXT, "parkinson disease")
    assert span is not None
    assert "Parkinson Disease" in span["snippet"]


def test_find_keyword_span_snippet_includes_context():
    span = find_keyword_span(TEXT, "Rasagiline", window=20)
    assert span is not None
    assert len(span["snippet"]) > len("Rasagiline")


def test_find_keyword_span_missing_returns_none():
    assert find_keyword_span(TEXT, "chemotherapy") is None


def test_find_keyword_span_empty_text():
    assert find_keyword_span("", "levodopa") is None


def test_find_keyword_span_empty_keyword():
    assert find_keyword_span(TEXT, "") is None


# ---------------------------------------------------------------------------
# find_any_keyword_span
# ---------------------------------------------------------------------------

def test_find_any_keyword_span_returns_first_match():
    span = find_any_keyword_span(TEXT, ["chemotherapy", "rasagiline", "levodopa"])
    assert span is not None
    assert span["keyword"].lower() == "rasagiline"


def test_find_any_keyword_span_none_found():
    assert find_any_keyword_span(TEXT, ["chemotherapy", "insulin"]) is None


def test_find_any_keyword_span_empty_list():
    assert find_any_keyword_span(TEXT, []) is None


# ---------------------------------------------------------------------------
# extract_evidence_spans
# ---------------------------------------------------------------------------

def test_extract_evidence_spans_basic():
    spans = extract_evidence_spans(TEXT, ["levodopa", "rasagiline"])
    assert len(spans) == 2


def test_extract_evidence_spans_max_spans():
    keywords = ["levodopa", "rasagiline", "parkinson", "male", "62"]
    spans = extract_evidence_spans(TEXT, keywords, max_spans=2)
    assert len(spans) <= 2


def test_extract_evidence_spans_no_duplicates():
    spans = extract_evidence_spans(TEXT, ["levodopa", "Levodopa", "LEVODOPA"])
    keywords_found = [s["keyword"].lower() for s in spans]
    assert keywords_found.count("levodopa") == 1


def test_extract_evidence_spans_sorted_by_position():
    spans = extract_evidence_spans(TEXT, ["rasagiline", "levodopa"])
    starts = [s["start"] for s in spans]
    assert starts == sorted(starts)


def test_extract_evidence_spans_missing_keywords():
    spans = extract_evidence_spans(TEXT, ["insulin", "chemotherapy"])
    assert spans == []


def test_extract_evidence_spans_empty_text():
    assert extract_evidence_spans("", ["levodopa"]) == []


def test_extract_evidence_spans_empty_keywords():
    assert extract_evidence_spans(TEXT, []) == []


# ---------------------------------------------------------------------------
# extract_criterion_evidence
# ---------------------------------------------------------------------------

_PATIENT = (
    "Patient is a 62-year-old male with idiopathic Parkinson Disease. "
    "Current medications include Levodopa and Rasagiline. "
    "No history of Deep Brain Stimulation."
)

_TRIAL = (
    "Inclusion criteria: Diagnosis of Parkinson Disease confirmed. "
    "Age between 30 and 80 years. "
    "Exclusion criteria: Prior Deep Brain Stimulation surgery."
)


def test_extract_criterion_evidence_returns_dict():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease diagnosis")
    assert isinstance(result, dict)


def test_extract_criterion_evidence_has_all_keys():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert {"patient_evidence", "trial_evidence",
            "patient_span_start", "patient_span_end",
            "trial_span_start", "trial_span_end"} <= set(result)


def test_extract_criterion_evidence_patient_evidence_string():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert isinstance(result["patient_evidence"], str)


def test_extract_criterion_evidence_trial_evidence_string():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert isinstance(result["trial_evidence"], str)


def test_extract_criterion_evidence_patient_evidence_nonempty_on_match():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert result["patient_evidence"] != ""


def test_extract_criterion_evidence_trial_evidence_nonempty_on_match():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert result["trial_evidence"] != ""


def test_extract_criterion_evidence_patient_span_offsets_int_on_match():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert isinstance(result["patient_span_start"], int)
    assert isinstance(result["patient_span_end"], int)


def test_extract_criterion_evidence_trial_span_offsets_int_on_match():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert isinstance(result["trial_span_start"], int)
    assert isinstance(result["trial_span_end"], int)


def test_extract_criterion_evidence_offsets_ordered():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Parkinson Disease")
    assert result["patient_span_start"] < result["patient_span_end"]
    assert result["trial_span_start"] < result["trial_span_end"]


def test_extract_criterion_evidence_missing_patient_returns_empty_evidence():
    result = extract_criterion_evidence("", _TRIAL, "Parkinson Disease")
    assert result["patient_evidence"] == ""


def test_extract_criterion_evidence_missing_patient_returns_empty_offsets():
    result = extract_criterion_evidence("", _TRIAL, "Parkinson Disease")
    assert result["patient_span_start"] == ""
    assert result["patient_span_end"] == ""


def test_extract_criterion_evidence_missing_trial_returns_empty_evidence():
    result = extract_criterion_evidence(_PATIENT, "", "Parkinson Disease")
    assert result["trial_evidence"] == ""


def test_extract_criterion_evidence_missing_trial_returns_empty_offsets():
    result = extract_criterion_evidence(_PATIENT, "", "Parkinson Disease")
    assert result["trial_span_start"] == ""
    assert result["trial_span_end"] == ""


def test_extract_criterion_evidence_empty_criterion_text():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "")
    assert result["patient_evidence"] == ""
    assert result["trial_evidence"] == ""
    assert result["patient_span_start"] == ""
    assert result["trial_span_start"] == ""


def test_extract_criterion_evidence_no_keyword_match_returns_empty_evidence():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "chemotherapy insulin")
    assert result["patient_evidence"] == ""
    assert result["trial_evidence"] == ""


def test_extract_criterion_evidence_no_keyword_match_returns_empty_offsets():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "chemotherapy insulin")
    assert result["patient_span_start"] == ""
    assert result["patient_span_end"] == ""
    assert result["trial_span_start"] == ""
    assert result["trial_span_end"] == ""


def test_extract_criterion_evidence_reason_keywords_used():
    # keyword only in reason, not criterion_text
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "eligibility check", reason="Levodopa requirement")
    assert result["patient_evidence"] != ""


def test_extract_criterion_evidence_snippet_contains_context():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "Levodopa", window=30)
    assert len(result["patient_evidence"]) > len("Levodopa")


def test_extract_criterion_evidence_patient_text_preserved_casing():
    result = extract_criterion_evidence(_PATIENT, _TRIAL, "levodopa")
    assert "Levodopa" in result["patient_evidence"]


def test_extract_criterion_evidence_all_empty_inputs():
    result = extract_criterion_evidence("", "", "")
    assert result == {
        "patient_evidence": "",
        "trial_evidence": "",
        "patient_span_start": "",
        "patient_span_end": "",
        "trial_span_start": "",
        "trial_span_end": "",
    }
