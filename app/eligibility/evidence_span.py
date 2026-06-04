"""
Task 4: Evidence span extraction helpers.

Extracts text snippets (spans) around clinical keywords from patient or
trial text, preserving original casing and recording character offsets.
"""

import re
from typing import Optional


def normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()


def find_keyword_span(
    text: str,
    keyword: str,
    window: int = 80,
) -> Optional[dict]:
    """
    Find the first occurrence of keyword (case-insensitive) in text.

    Returns a dict with keyword, start, end, snippet or None if not found.
    start/end are character offsets in the original text.
    snippet preserves original casing.
    """
    if not text or not keyword:
        return None
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return None
    start = match.start()
    end = match.end()
    snip_start = max(0, start - window)
    snip_end = min(len(text), end + window)
    return {
        "keyword": keyword,
        "start": start,
        "end": end,
        "snippet": text[snip_start:snip_end],
    }


def find_any_keyword_span(
    text: str,
    keywords: list[str],
    window: int = 80,
) -> Optional[dict]:
    """
    Return the first span found for any keyword in the list.

    Tries keywords in order; returns on the first match.
    """
    for kw in keywords:
        span = find_keyword_span(text, kw, window=window)
        if span is not None:
            return span
    return None


def extract_evidence_spans(
    text: str,
    keywords: list[str],
    window: int = 80,
    max_spans: int = 5,
) -> list[dict]:
    """
    Extract up to max_spans evidence spans, one per unique keyword found.

    Skips duplicate keywords and keywords not present in text.
    Returns a list of span dicts ordered by position in text.
    """
    seen_keywords: set[str] = set()
    spans: list[dict] = []
    for kw in keywords:
        key = kw.lower()
        if key in seen_keywords:
            continue
        seen_keywords.add(key)
        span = find_keyword_span(text, kw, window=window)
        if span is not None:
            spans.append(span)
        if len(spans) >= max_spans:
            break
    spans.sort(key=lambda s: s["start"])
    return spans


def _keywords_from_text(text: str, min_len: int = 4) -> list[str]:
    """
    Extract unique word-like tokens of at least min_len characters from text.

    Returns tokens in order of first appearance, preserving original casing.
    Deduplication is case-insensitive.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        key = tok.lower()
        if len(key) >= min_len and key not in seen:
            seen.add(key)
            result.append(tok)
    return result


def extract_criterion_evidence(
    patient_text: str,
    trial_text: str,
    criterion_text: str,
    reason: str = "",
    window: int = 80,
) -> dict:
    """
    Extract evidence spans for one criterion from patient and trial text.

    Keywords are derived deterministically from criterion_text and reason.
    Searches patient_text for patient evidence and trial_text for trial evidence.

    Returns a dict with:
        patient_evidence  — snippet from patient_text around the first keyword match
        trial_evidence    — snippet from trial_text around the first keyword match
        patient_span_start — int offset in patient_text, or "" when not found
        patient_span_end   — int offset in patient_text, or "" when not found
        trial_span_start   — int offset in trial_text, or "" when not found
        trial_span_end     — int offset in trial_text, or "" when not found

    All evidence strings are empty and all offsets are "" when no span is found.
    No model calls are made; extraction is fully deterministic.
    """
    keywords = _keywords_from_text(criterion_text + " " + reason)

    patient_span = (
        find_any_keyword_span(patient_text, keywords, window=window)
        if keywords and patient_text
        else None
    )
    trial_span = (
        find_any_keyword_span(trial_text, keywords, window=window)
        if keywords and trial_text
        else None
    )

    return {
        "patient_evidence": patient_span["snippet"] if patient_span else "",
        "trial_evidence": trial_span["snippet"] if trial_span else "",
        "patient_span_start": patient_span["start"] if patient_span else "",
        "patient_span_end": patient_span["end"] if patient_span else "",
        "trial_span_start": trial_span["start"] if trial_span else "",
        "trial_span_end": trial_span["end"] if trial_span else "",
    }
