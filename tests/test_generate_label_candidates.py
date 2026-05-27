"""Unit tests for generate_label_candidates.py."""

from generate_label_candidates import (
    build_candidate,
    generate_candidates,
    patient_matches_category,
)


def test_patient_matches_category_true():
    patient = {"category_focus": "gait_and_falls"}
    assert patient_matches_category(patient, "rehabilitation") is True


def test_patient_matches_category_false():
    patient = {"category_focus": "gait_and_falls"}
    assert patient_matches_category(patient, "biomarker") is False


def test_build_candidate_schema():
    patient = {"patient_id": "P001"}
    trial = {"trial_id": "T001", "category": "rehabilitation"}

    result = build_candidate(patient, trial)

    assert result["patient_id"] == "P001"
    assert result["trial_id"] == "T001"
    assert result["trial_category"] == "rehabilitation"
    assert result["suggested_label"] == ""
    assert result["rationale"] == ""
    assert result["evidence"] == {"patient_facts": [], "trial_criteria": []}
    assert result["label_status"] == "needs_manual_review"


def test_generate_candidates_returns_pairs():
    patients = [
        {"patient_id": "P001", "category_focus": "gait_and_falls"},
        {"patient_id": "P002", "category_focus": "early_parkinson"},
    ]
    trials = [{"trial_id": "T001", "category": "rehabilitation"}]

    result = generate_candidates(patients, trials)

    assert len(result) == 2
    assert result[0]["trial_id"] == "T001"


def test_generate_candidates_limits_to_three_patients_per_trial():
    patients = [
        {"patient_id": "P001", "category_focus": "gait_and_falls"},
        {"patient_id": "P002", "category_focus": "older_frail_patient"},
        {"patient_id": "P003", "category_focus": "early_parkinson"},
        {"patient_id": "P004", "category_focus": "insufficient_patient_detail"},
    ]
    trials = [{"trial_id": "T001", "category": "rehabilitation"}]

    result = generate_candidates(patients, trials)

    assert len(result) == 3


def test_generate_candidates_uses_fallback_patients():
    patients = [
        {"patient_id": "P001", "category_focus": "unknown_focus"},
        {"patient_id": "P002", "category_focus": "another_unknown_focus"},
        {"patient_id": "P003", "category_focus": "third_unknown_focus"},
        {"patient_id": "P004", "category_focus": "fourth_unknown_focus"},
    ]
    trials = [{"trial_id": "T001", "category": "unknown_category"}]

    result = generate_candidates(patients, trials)

    assert len(result) == 3
    assert [r["patient_id"] for r in result] == ["P001", "P002", "P003"]
