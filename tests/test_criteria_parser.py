"""Unit tests for criteria_parser.py."""

from app.eligibility.criteria_parser import parse_eligibility_criteria, parse_numeric_range, parse_numeric_comparator, parse_duration


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


# ---------------------------------------------------------------------------
# Duration parsing tests
# ---------------------------------------------------------------------------

def test_duration_for_at_least_weeks():
    r = parse_duration("for at least 4 weeks")
    assert r["operator"] == "at_least"
    assert r["value"] == 4.0
    assert r["unit"] == "weeks"
    assert r["value_days"] == 28.0


def test_duration_stable_for_months():
    r = parse_duration("stable for 3 months")
    assert r["operator"] == "at_least"
    assert r["value"] == 3.0
    assert r["unit"] == "months"
    assert r["value_days"] == 90.0


def test_duration_within_days():
    r = parse_duration("within 30 days")
    assert r["operator"] == "within"
    assert r["value"] == 30.0
    assert r["unit"] == "days"
    assert r["value_days"] == 30.0


def test_duration_in_the_last_months():
    r = parse_duration("in the last 6 months")
    assert r["operator"] == "within"
    assert r["value"] == 6.0
    assert r["unit"] == "months"
    assert r["value_days"] == 180.0


def test_duration_during_the_past_weeks():
    r = parse_duration("during the past 12 weeks")
    assert r["operator"] == "within"
    assert r["value"] == 12.0
    assert r["unit"] == "weeks"
    assert r["value_days"] == 84.0


def test_duration_no_medication_change_weeks():
    r = parse_duration("no medication change for 8 weeks")
    assert r["operator"] == "at_least"
    assert r["value"] == 8.0
    assert r["unit"] == "weeks"
    assert r["value_days"] == 56.0


def test_duration_washout_period_days():
    r = parse_duration("washout period of 14 days")
    assert r["operator"] == "exact"
    assert r["value"] == 14.0
    assert r["unit"] == "days"
    assert r["value_days"] == 14.0


def test_duration_disease_duration_years():
    r = parse_duration("disease duration of at least 2 years")
    assert r["operator"] == "at_least"
    assert r["value"] == 2.0
    assert r["unit"] == "years"
    assert r["value_days"] == 730.0


def test_duration_diagnosed_more_than_years():
    r = parse_duration("diagnosed for more than 5 years")
    assert r["operator"] == "at_least"
    assert r["value"] == 5.0
    assert r["unit"] == "years"
    assert r["value_days"] == 1825.0


def test_duration_symptoms_less_than_year():
    r = parse_duration("symptoms for less than 1 year")
    assert r["operator"] == "less_than"
    assert r["value"] == 1.0
    assert r["unit"] == "years"
    assert r["value_days"] == 365.0


def test_duration_no_match_returns_none():
    r = parse_duration("Parkinson disease diagnosis required")
    assert r["operator"] is None
    assert r["value"] is None
    assert r["unit"] is None
    assert r["value_days"] is None


def test_duration_value_days_weeks():
    r = parse_duration("for at least 2 weeks")
    assert r["value_days"] == 14.0


def test_duration_unit_normalized_plural():
    r = parse_duration("stable for 1 month")
    assert r["unit"] == "months"



