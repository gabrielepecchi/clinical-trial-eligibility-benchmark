"""Unit tests for criteria_parser.py."""

from criteria_parser import parse_eligibility_criteria


TEXT_FULL = """\
Inclusion Criteria
- Age 18 or older
- Parkinson disease diagnosis

Exclusion Criteria
- Prior DBS surgery
* Active cancer
"""

TEXT_SHORT_HEADINGS = """\
Inclusion
Age 18 or older

Exclusion
Active cancer
"""

TEXT_INCLUSION_ONLY = """\
Inclusion Criteria
- Age 18 or older
- Signed consent
"""

TEXT_NO_HEADINGS = """\
Age 18 or older
Parkinson disease diagnosis
No prior DBS surgery
"""

TEXT_BULLETS = """\
Inclusion Criteria
- dash item
* star item
• bullet item
"""


def test_full_headings_inclusion():
    result = parse_eligibility_criteria(TEXT_FULL)
    assert "Age 18 or older" in result["inclusion_criteria"]
    assert "Parkinson disease diagnosis" in result["inclusion_criteria"]


def test_full_headings_exclusion():
    result = parse_eligibility_criteria(TEXT_FULL)
    assert "Prior DBS surgery" in result["exclusion_criteria"]
    assert "Active cancer" in result["exclusion_criteria"]


def test_short_headings_inclusion():
    result = parse_eligibility_criteria(TEXT_SHORT_HEADINGS)
    assert "Age 18 or older" in result["inclusion_criteria"]


def test_short_headings_exclusion():
    result = parse_eligibility_criteria(TEXT_SHORT_HEADINGS)
    assert "Active cancer" in result["exclusion_criteria"]


def test_inclusion_only():
    result = parse_eligibility_criteria(TEXT_INCLUSION_ONLY)
    assert "Age 18 or older" in result["inclusion_criteria"]
    assert result["exclusion_criteria"] == []


def test_no_headings_goes_to_inclusion():
    result = parse_eligibility_criteria(TEXT_NO_HEADINGS)
    assert "Age 18 or older" in result["inclusion_criteria"]
    assert "Parkinson disease diagnosis" in result["inclusion_criteria"]
    assert "No prior DBS surgery" in result["inclusion_criteria"]


def test_no_headings_exclusion_empty():
    result = parse_eligibility_criteria(TEXT_NO_HEADINGS)
    assert result["exclusion_criteria"] == []


def test_bullet_prefixes_removed():
    result = parse_eligibility_criteria(TEXT_BULLETS)
    assert "dash item" in result["inclusion_criteria"]
    assert "star item" in result["inclusion_criteria"]
    assert "bullet item" in result["inclusion_criteria"]


def test_empty_lines_removed():
    result = parse_eligibility_criteria(TEXT_FULL)
    for item in result["inclusion_criteria"] + result["exclusion_criteria"]:
        assert item.strip() != ""


def test_raw_eligibility_preserved():
    result = parse_eligibility_criteria(TEXT_FULL)
    assert result["raw_eligibility"] == TEXT_FULL
