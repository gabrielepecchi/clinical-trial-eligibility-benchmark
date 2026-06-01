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
