"""Age-specific tests for rule_matcher.py."""

from rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from models import CriterionDecision


def make_patient(**kwargs) -> dict:
    base = {
        "patient_id": "P_TEST",
        "age": 60,
        "sex": "male",
        "diagnosis": ["Parkinson disease"],
        "key_features": ["Hoehn and Yahr stage 2"],
        "exclusions": [],
        "medications": ["levodopa/carbidopa 100/25 mg three times daily"],
    }
    base.update(kwargs)
    return base


def _trial(criterion: str) -> dict:
    return {"trial_id": "T_FMT", "inclusion_criteria": [criterion], "exclusion_criteria": []}


# ---------------------------------------------------------------------------
# Age boundary tests — "Age 40 to 80 years"
# ---------------------------------------------------------------------------

AGE_ONLY_TRIAL = {
    "trial_id": "T_AGE_BOUNDARY",
    "inclusion_criteria": ["Age 40 to 80 years"],
    "exclusion_criteria": [],
}


def test_age_40_is_eligible():
    result = match_patient_to_trial(make_patient(age=40), AGE_ONLY_TRIAL)
    assert result["prediction"] == "eligible"


def test_age_80_is_eligible():
    result = match_patient_to_trial(make_patient(age=80), AGE_ONLY_TRIAL)
    assert result["prediction"] == "eligible"


def test_age_39_is_not_eligible():
    result = match_patient_to_trial(make_patient(age=39), AGE_ONLY_TRIAL)
    assert result["prediction"] == "not_eligible"


def test_age_81_is_not_eligible():
    result = match_patient_to_trial(make_patient(age=81), AGE_ONLY_TRIAL)
    assert result["prediction"] == "not_eligible"


def test_age_40_criterion_is_met():
    results = match_patient_to_trial_criteria(make_patient(age=40), AGE_ONLY_TRIAL)
    assert results[0].decision == CriterionDecision.met


def test_age_39_criterion_is_not_met():
    results = match_patient_to_trial_criteria(make_patient(age=39), AGE_ONLY_TRIAL)
    assert results[0].decision == CriterionDecision.not_met


# ---------------------------------------------------------------------------
# Age criterion format variants
# ---------------------------------------------------------------------------

def test_ages_hyphen_accepts_lower_bound():
    assert match_patient_to_trial(make_patient(age=40), _trial("Ages 40-80 years"))["prediction"] == "eligible"


def test_ages_hyphen_rejects_below_lower_bound():
    assert match_patient_to_trial(make_patient(age=39), _trial("Ages 40-80 years"))["prediction"] == "not_eligible"


def test_ages_endash_accepts_upper_bound():
    assert match_patient_to_trial(make_patient(age=80), _trial("40\u201380 years of age"))["prediction"] == "eligible"


def test_ages_endash_rejects_above_upper_bound():
    assert match_patient_to_trial(make_patient(age=81), _trial("40\u201380 years of age"))["prediction"] == "not_eligible"


def test_age_gte_accepts_lower_bound():
    assert match_patient_to_trial(make_patient(age=40), _trial("Age >= 40"))["prediction"] == "eligible"


def test_age_gte_rejects_below_lower_bound():
    assert match_patient_to_trial(make_patient(age=39), _trial("Age >= 40"))["prediction"] == "not_eligible"


def test_age_lte_accepts_upper_bound():
    assert match_patient_to_trial(make_patient(age=80), _trial("Age <= 80"))["prediction"] == "eligible"


def test_age_lte_rejects_above_upper_bound():
    assert match_patient_to_trial(make_patient(age=81), _trial("Age <= 80"))["prediction"] == "not_eligible"


def test_age_or_older_accepts_lower_bound():
    assert match_patient_to_trial(make_patient(age=40), _trial("Age 40 years or older"))["prediction"] == "eligible"


def test_age_or_older_rejects_below_lower_bound():
    assert match_patient_to_trial(make_patient(age=39), _trial("Age 40 years or older"))["prediction"] == "not_eligible"


def test_age_or_younger_accepts_upper_bound():
    assert match_patient_to_trial(make_patient(age=80), _trial("Age 80 years or younger"))["prediction"] == "eligible"


def test_age_or_younger_rejects_above_upper_bound():
    assert match_patient_to_trial(make_patient(age=81), _trial("Age 80 years or younger"))["prediction"] == "not_eligible"


def test_age_gte_criterion_met_at_lower_bound():
    results = match_patient_to_trial_criteria(make_patient(age=40), _trial("Age >= 40"))
    assert results[0].decision == CriterionDecision.met


def test_age_lte_criterion_not_met_above_upper_bound():
    results = match_patient_to_trial_criteria(make_patient(age=81), _trial("Age <= 80"))
    assert results[0].decision == CriterionDecision.not_met
