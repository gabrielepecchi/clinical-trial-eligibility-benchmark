"""Tests for app.eligibility.evidence_span (Task 4)."""

import pytest
from app.eligibility.evidence_span import (
    normalize_text,
    find_keyword_span,
    find_any_keyword_span,
    extract_evidence_spans,
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
