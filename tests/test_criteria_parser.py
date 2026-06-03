"""Unit tests for criteria_parser.py."""

from app.eligibility.criteria_parser import parse_eligibility_criteria, parse_numeric_range, parse_numeric_comparator


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


# ---------------------------------------------------------------------------
# Numeric range parsing tests
# ---------------------------------------------------------------------------


def test_range_between_and():
    r = parse_numeric_range("between 18 and 80")
    assert r["lower"] == 18.0
    assert r["upper"] == 80.0


def test_range_n_to_m():
    r = parse_numeric_range("18 to 80")
    assert r["lower"] == 18.0
    assert r["upper"] == 80.0


def test_range_hyphen():
    r = parse_numeric_range("18-80")
    assert r["lower"] == 18.0
    assert r["upper"] == 80.0


def test_range_en_dash():
    r = parse_numeric_range("18 – 80")
    assert r["lower"] == 18.0
    assert r["upper"] == 80.0


def test_range_from_to():
    r = parse_numeric_range("from 18 to 80")
    assert r["lower"] == 18.0
    assert r["upper"] == 80.0


def test_range_bmi_with_units():
    r = parse_numeric_range("BMI between 18 and 32 kg/m2")
    assert r["lower"] == 18.0
    assert r["upper"] == 32.0


def test_range_hy_stage():
    r = parse_numeric_range("Hoehn and Yahr stage 1 to 3")
    assert r["lower"] == 1.0
    assert r["upper"] == 3.0


def test_range_or_older():
    r = parse_numeric_range("age 40 years or older")
    assert r["lower"] == 40.0
    assert r["upper"] is None


def test_range_or_younger():
    r = parse_numeric_range("age 75 years or younger")
    assert r["lower"] is None
    assert r["upper"] == 75.0


def test_range_at_least():
    r = parse_numeric_range("at least 18 years of age")
    assert r["lower"] == 18.0
    assert r["upper"] is None


def test_range_at_most():
    r = parse_numeric_range("at most 80 years of age")
    assert r["lower"] is None
    assert r["upper"] == 80.0


def test_range_gte_symbol():
    r = parse_numeric_range("age >= 18")
    assert r["lower"] == 18.0
    assert r["upper"] is None


def test_range_lte_symbol():
    r = parse_numeric_range("age <= 75")
    assert r["lower"] is None
    assert r["upper"] == 75.0


def test_range_unicode_gte():
    r = parse_numeric_range("age ≥ 18")
    assert r["lower"] == 18.0
    assert r["upper"] is None


def test_range_unicode_lte():
    r = parse_numeric_range("age ≤ 75")
    assert r["lower"] is None
    assert r["upper"] == 75.0


def test_range_no_numbers_returns_none():
    r = parse_numeric_range("Parkinson disease diagnosis required")
    assert r["lower"] is None
    assert r["upper"] is None


# ---------------------------------------------------------------------------
# Comparator parsing tests
# ---------------------------------------------------------------------------

def test_comparator_gte_symbol():
    r = parse_numeric_comparator("age >= 18")
    assert r["operator"] == ">="
    assert r["value"] == 18.0


def test_comparator_gt_symbol():
    r = parse_numeric_comparator("age > 18")
    assert r["operator"] == ">"
    assert r["value"] == 18.0


def test_comparator_lte_symbol():
    r = parse_numeric_comparator("age ≤ 75")
    assert r["operator"] == "<="
    assert r["value"] == 75.0


def test_comparator_lt_symbol():
    r = parse_numeric_comparator("age < 75")
    assert r["operator"] == "<"
    assert r["value"] == 75.0


def test_comparator_at_least():
    r = parse_numeric_comparator("at least 18 years")
    assert r["operator"] == ">="
    assert r["value"] == 18.0


def test_comparator_greater_than():
    r = parse_numeric_comparator("greater than 18 years")
    assert r["operator"] == ">"
    assert r["value"] == 18.0


def test_comparator_more_than():
    r = parse_numeric_comparator("more than 18 years")
    assert r["operator"] == ">"
    assert r["value"] == 18.0


def test_comparator_minimum_age():
    r = parse_numeric_comparator("minimum age 18")
    assert r["operator"] == ">="
    assert r["value"] == 18.0


def test_comparator_no_more_than():
    r = parse_numeric_comparator("no more than 75 years")
    assert r["operator"] == "<="
    assert r["value"] == 75.0


def test_comparator_less_than():
    r = parse_numeric_comparator("less than 75 years")
    assert r["operator"] == "<"
    assert r["value"] == 75.0


def test_comparator_maximum_age():
    r = parse_numeric_comparator("maximum age 75")
    assert r["operator"] == "<="
    assert r["value"] == 75.0


def test_comparator_creatinine_lt():
    r = parse_numeric_comparator("creatinine < 1.5 mg/dL")
    assert r["operator"] == "<"
    assert r["value"] == 1.5


def test_comparator_hemoglobin_greater_than():
    r = parse_numeric_comparator("hemoglobin greater than 10 g/dL")
    assert r["operator"] == ">"
    assert r["value"] == 10.0


def test_comparator_lte_ascii():
    r = parse_numeric_comparator("age <= 75")
    assert r["operator"] == "<="
    assert r["value"] == 75.0


def test_comparator_gte_unicode():
    r = parse_numeric_comparator("age ≥ 18")
    assert r["operator"] == ">="
    assert r["value"] == 18.0


def test_comparator_no_match_returns_none():
    r = parse_numeric_comparator("Parkinson disease diagnosis required")
    assert r["operator"] is None
    assert r["value"] is None


