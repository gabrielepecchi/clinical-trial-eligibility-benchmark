"""Unit tests for extract_eligibility.py."""

from extract_eligibility import extract_trial

FULL_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "A Parkinson Trial",
            "officialTitle": "A Full Official Parkinson Trial Title",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
        },
        "designModule": {
            "phases": ["PHASE2"],
            "studyType": "INTERVENTIONAL",
        },
        "conditionsModule": {
            "conditions": ["Parkinson Disease"],
        },
        "armsInterventionsModule": {
            "interventions": [{"name": "Levodopa"}],
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria\n- Age 30 or older\n\nExclusion Criteria\n- Prior DBS\n"
            ),
            "minimumAge": "30 Years",
            "maximumAge": "80 Years",
            "sex": "ALL",
            "healthyVolunteers": "No",
        },
    }
}

NO_NCT_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "briefTitle": "No ID Trial",
        }
    }
}

NO_TITLE_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000002",
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Inclusion Criteria\n- Age 18 or older\n"
        },
    }
}

NO_ELIGIBILITY_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000003",
            "briefTitle": "No Eligibility Trial",
        }
    }
}


def test_extract_trial_returns_all_fields():
    result = extract_trial(FULL_STUDY)
    assert result is not None
    assert result["nct_id"] == "NCT00000001"
    assert result["title"] == "A Parkinson Trial"
    assert result["official_title"] == "A Full Official Parkinson Trial Title"
    assert result["overall_status"] == "RECRUITING"
    assert result["phase"] == "PHASE2"
    assert result["study_type"] == "INTERVENTIONAL"
    assert result["conditions"] == ["Parkinson Disease"]
    assert result["interventions"] == ["Levodopa"]
    assert result["minimum_age"] == "30 Years"
    assert result["maximum_age"] == "80 Years"
    assert result["sex"] == "ALL"
    assert result["healthy_volunteers"] == "No"
    assert "Age 30 or older" in result["eligibility_text"]
    assert "Age 30 or older" in result["inclusion_criteria"]
    assert "Prior DBS" in result["exclusion_criteria"]


def test_extract_trial_no_nct_id_returns_none():
    result = extract_trial(NO_NCT_STUDY)
    assert result is None


def test_extract_trial_missing_title_is_empty_string():
    result = extract_trial(NO_TITLE_STUDY)
    assert result is not None
    assert result["title"] == ""


def test_extract_trial_missing_eligibility_text_is_empty():
    result = extract_trial(NO_ELIGIBILITY_STUDY)
    assert result is not None
    assert result["eligibility_text"] == ""
    assert result["inclusion_criteria"] == []
    assert result["exclusion_criteria"] == []
