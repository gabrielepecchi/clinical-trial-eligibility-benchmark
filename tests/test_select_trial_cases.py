"""Unit tests for select_trial_cases.py."""

from select_trial_cases import assign_category, build_trial_case, is_valid

# --- assign_category ---

def test_assign_category_device_dbs():
    assert assign_category("A DBS trial", []) == "device"


def test_assign_category_device_deep_brain():
    assert assign_category("Deep Brain Stimulation Study", []) == "device"


def test_assign_category_rehabilitation_gait():
    assert assign_category("Gait Training in PD", []) == "rehabilitation"


def test_assign_category_rehabilitation_exercise():
    assert assign_category("Exercise Program", ["exercise"]) == "rehabilitation"


def test_assign_category_non_motor_sleep():
    assert assign_category("Sleep Disorders in Parkinson", []) == "non_motor_symptoms"


def test_assign_category_non_motor_cognitive():
    assert assign_category("Cognitive Decline Study", []) == "non_motor_symptoms"


def test_assign_category_biomarker():
    assert assign_category("Biomarker Detection Trial", []) == "biomarker"


def test_assign_category_biomarker_mri():
    assert assign_category("MRI Study in PD", []) == "biomarker"


def test_assign_category_drug_fallback():
    assert assign_category("A Generic Parkinson Trial", ["placebo"]) == "drug_treatment"


# --- is_valid ---

VALID_RECORD = {
    "nct_id": "NCT00000001",
    "title": "A Trial",
    "eligibility_text": "Inclusion Criteria\n- Age 30+",
    "inclusion_criteria": ["Age 30+"],
    "exclusion_criteria": [],
}


def test_is_valid_returns_true():
    assert is_valid(VALID_RECORD) is True


def test_is_valid_missing_nct_id():
    record = {**VALID_RECORD, "nct_id": ""}
    assert is_valid(record) is False


def test_is_valid_missing_eligibility_text():
    record = {**VALID_RECORD, "eligibility_text": ""}
    assert is_valid(record) is False


def test_is_valid_empty_criteria():
    record = {**VALID_RECORD, "inclusion_criteria": [], "exclusion_criteria": []}
    assert is_valid(record) is False


# --- build_trial_case ---

FULL_RECORD = {
    "nct_id": "NCT00000001",
    "title": "A Parkinson Trial",
    "official_title": "Full Official Title",
    "overall_status": "RECRUITING",
    "phase": "PHASE2",
    "study_type": "INTERVENTIONAL",
    "conditions": ["Parkinson Disease"],
    "interventions": ["Levodopa"],
    "minimum_age": "30 Years",
    "maximum_age": "80 Years",
    "sex": "ALL",
    "healthy_volunteers": "No",
    "eligibility_text": "Inclusion Criteria\n- Age 30+\n\nExclusion Criteria\n- Prior DBS",
    "inclusion_criteria": ["Age 30+"],
    "exclusion_criteria": ["Prior DBS"],
}


def test_build_trial_case_trial_id():
    result = build_trial_case(FULL_RECORD, 1)
    assert result["trial_id"] == "T001"


def test_build_trial_case_url():
    result = build_trial_case(FULL_RECORD, 1)
    assert result["url"] == "https://clinicaltrials.gov/study/NCT00000001"


def test_build_trial_case_raw_eligibility():
    result = build_trial_case(FULL_RECORD, 1)
    assert result["raw_eligibility"] == FULL_RECORD["eligibility_text"]


def test_build_trial_case_criteria_lists():
    result = build_trial_case(FULL_RECORD, 1)
    assert result["inclusion_criteria"] == ["Age 30+"]
    assert result["exclusion_criteria"] == ["Prior DBS"]
