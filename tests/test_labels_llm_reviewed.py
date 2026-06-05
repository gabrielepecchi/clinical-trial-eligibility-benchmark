"""Unit tests for labels_llm_reviewed.json."""

import json
from pathlib import Path

import pytest

LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")

REQUIRED_FIELDS = {
    "patient_id",
    "trial_id",
    "label",
    "rationale",
    "evidence",
    "label_status",
}

VALID_LABELS = {"eligible", "not_eligible", "unclear"}
EXPECTED_STATUS = "llm_reviewed_needs_spotcheck"


@pytest.fixture(scope="module")
def labels() -> list[dict]:
    return json.loads(LABELS_FILE.read_text(encoding="utf-8"))


def test_exactly_150_records(labels):
    assert len(labels) == 150


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_all_records_have_required_fields(labels, field):
    for record in labels:
        assert field in record


def test_labels_are_valid(labels):
    for record in labels:
        assert record["label"] in VALID_LABELS


def test_label_status_is_expected(labels):
    for record in labels:
        assert record["label_status"] == EXPECTED_STATUS


def test_rationale_is_not_empty(labels):
    for record in labels:
        assert record["rationale"].strip() != ""


def test_evidence_has_required_keys(labels):
    for record in labels:
        assert "patient_facts" in record["evidence"]
        assert "trial_criteria" in record["evidence"]


def test_evidence_values_are_lists(labels):
    for record in labels:
        assert isinstance(record["evidence"]["patient_facts"], list)
        assert isinstance(record["evidence"]["trial_criteria"], list)


def test_patient_ids_are_not_empty(labels):
    for record in labels:
        assert record["patient_id"].strip() != ""


def test_trial_ids_are_not_empty(labels):
    for record in labels:
        assert record["trial_id"].strip() != ""


def test_contains_at_least_one_record_per_label(labels):
    present_labels = {record["label"] for record in labels}
    assert VALID_LABELS <= present_labels


# ---------------------------------------------------------------------------
# Hard-case tagger unit tests (synthetic data only, no dataset required)
# ---------------------------------------------------------------------------

from eval.tag_hard_cases import (
    assign_hard_case_tags,
    build_hard_case_records,
    build_patient_index,
    build_result_index,
    build_summary,
    build_trial_index,
)


def _make_label(patient_id="P001", trial_id="T001", label="not_eligible", rationale="", evidence=None):
    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "label": label,
        "rationale": rationale,
        "evidence": evidence or {"patient_facts": [], "trial_criteria": []},
        "label_status": "llm_reviewed_needs_spotcheck",
    }


def _make_patient(patient_id="P001", **kwargs):
    return {"patient_id": patient_id, **kwargs}


def _make_trial(trial_id="T001", **kwargs):
    return {"trial_id": trial_id, **kwargs}


# --- hard_negative ---

def test_hard_negative_assigned_for_exclusion_signal():
    lr = _make_label(label="not_eligible", rationale="Patient has active DBS device, excluded by criterion.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_negative" in tags


def test_hard_negative_assigned_for_maob_signal():
    lr = _make_label(label="not_eligible", rationale="Current use of MAO-B inhibitor contraindicated.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_negative" in tags


def test_hard_negative_not_assigned_for_eligible():
    lr = _make_label(label="eligible", rationale="Patient has active DBS device, excluded by criterion.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_negative" not in tags


def test_hard_negative_not_assigned_without_signals():
    lr = _make_label(label="not_eligible", rationale="Patient does not qualify.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_negative" not in tags


def test_hard_negative_picks_up_signal_from_trial():
    lr = _make_label(label="not_eligible", rationale="see trial")
    trial = _make_trial(criteria_text="Exclusion: prior pacemaker implantation.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), trial)
    assert "hard_negative" in tags


# --- hard_positive ---

def test_hard_positive_assigned_for_complex_eligible():
    lr = _make_label(label="eligible", rationale="Multiple criteria met: UPDRS score within threshold, no exclusion, medication history reviewed.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_positive" in tags


def test_hard_positive_not_assigned_for_not_eligible():
    lr = _make_label(label="not_eligible", rationale="Multiple criteria met: UPDRS score within threshold, no exclusion, medication history reviewed.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_positive" not in tags


def test_hard_positive_not_assigned_for_trivial_eligible():
    lr = _make_label(label="eligible", rationale="Patient meets criteria.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_positive" not in tags


def test_hard_positive_not_assigned_for_generic_meets_language():
    lr = _make_label(label="eligible", rationale="Patient meets all eligibility criteria for the trial.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "hard_positive" not in tags


def test_hard_positive_picks_up_signal_from_patient():
    lr = _make_label(label="eligible", rationale="see patient")
    patient = _make_patient(notes="Age 72, Hoehn-Yahr stage 2, no prior DBS, medications reviewed.")
    tags, _ = assign_hard_case_tags(lr, patient, _make_trial())
    assert "hard_positive" in tags


# --- ambiguous_clinical_severity ---

def test_ambiguous_assigned_for_missing_info():
    lr = _make_label(label="unclear", rationale="UPDRS score not documented; MoCA value missing. Cannot determine eligibility.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "ambiguous_clinical_severity" in tags


def test_ambiguous_assigned_for_severity_signal():
    lr = _make_label(label="unclear", rationale="Disease stage and Hoehn-Yahr score ambiguous from available data.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "ambiguous_clinical_severity" in tags


def test_ambiguous_not_assigned_for_eligible():
    lr = _make_label(label="eligible", rationale="UPDRS score not documented; MoCA value missing.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "ambiguous_clinical_severity" not in tags


def test_ambiguous_not_assigned_without_signals():
    lr = _make_label(label="unclear", rationale="Cannot determine eligibility.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert "ambiguous_clinical_severity" not in tags


# --- multiple tags ---

def test_tags_are_sorted():
    lr = _make_label(label="not_eligible", rationale="Exclusion: DBS device.")
    tags, _ = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert tags == sorted(tags)


def test_tag_reasons_match_tag_count():
    lr = _make_label(label="not_eligible", rationale="Exclusion: DBS device.")
    tags, reasons = assign_hard_case_tags(lr, _make_patient(), _make_trial())
    assert len(tags) == len(reasons)


# --- build_summary ---

def test_build_summary_total_records():
    records = [
        {"gold_label": "not_eligible", "hard_case_tags": ["hard_negative"], "tag_reasons": []},
        {"gold_label": "eligible", "hard_case_tags": ["hard_positive"], "tag_reasons": []},
        {"gold_label": "unclear", "hard_case_tags": [], "tag_reasons": []},
    ]
    summary = build_summary(records)
    assert summary["total_records"] == 3


def test_build_summary_tag_counts():
    records = [
        {"gold_label": "not_eligible", "hard_case_tags": ["hard_negative"], "tag_reasons": []},
        {"gold_label": "not_eligible", "hard_case_tags": ["hard_negative"], "tag_reasons": []},
        {"gold_label": "eligible", "hard_case_tags": ["hard_positive"], "tag_reasons": []},
        {"gold_label": "unclear", "hard_case_tags": [], "tag_reasons": []},
    ]
    summary = build_summary(records)
    assert summary["tag_counts"]["hard_negative"] == 2
    assert summary["tag_counts"]["hard_positive"] == 1
    assert summary["tag_counts"]["ambiguous_clinical_severity"] == 0


def test_build_summary_label_distribution_by_tag():
    records = [
        {"gold_label": "not_eligible", "hard_case_tags": ["hard_negative"], "tag_reasons": []},
        {"gold_label": "not_eligible", "hard_case_tags": ["hard_negative"], "tag_reasons": []},
    ]
    summary = build_summary(records)
    assert summary["label_distribution_by_tag"]["hard_negative"]["not_eligible"] == 2


def test_build_summary_empty():
    summary = build_summary([])
    assert summary["total_records"] == 0
    for tag in ["hard_negative", "hard_positive", "ambiguous_clinical_severity"]:
        assert summary["tag_counts"][tag] == 0


# --- build_hard_case_records ---

def test_build_hard_case_records_returns_list():
    labels = [_make_label()]
    patients = [_make_patient()]
    trials = [_make_trial()]
    records = build_hard_case_records(labels, patients, trials)
    assert isinstance(records, list)
    assert len(records) == 1


def test_build_hard_case_records_schema():
    labels = [_make_label()]
    patients = [_make_patient()]
    trials = [_make_trial()]
    rec = build_hard_case_records(labels, patients, trials)[0]
    for key in ("patient_id", "trial_id", "gold_label", "predicted_label", "hard_case_tags", "tag_reasons"):
        assert key in rec


def test_build_result_index_from_list():
    results = [{"patient_id": "P001", "trial_id": "T001", "predicted_label": "eligible"}]
    idx = build_result_index(results)
    assert ("P001", "T001") in idx


def test_build_result_index_from_dict():
    payload = {"predictions": [{"patient_id": "P001", "trial_id": "T001", "predicted_label": "eligible"}]}
    idx = build_result_index(payload)
    assert ("P001", "T001") in idx


def test_build_hard_case_records_preserves_label_order():
    labels = [
        _make_label(patient_id="P003", trial_id="T001", label="unclear"),
        _make_label(patient_id="P001", trial_id="T001", label="eligible"),
        _make_label(patient_id="P002", trial_id="T001", label="not_eligible"),
    ]
    patients = [_make_patient("P001"), _make_patient("P002"), _make_patient("P003")]
    trials = [_make_trial("T001")]
    records = build_hard_case_records(labels, patients, trials)
    assert [r["patient_id"] for r in records] == ["P003", "P001", "P002"]


def test_build_result_index_none():
    assert build_result_index(None) == {}


def test_build_patient_index():
    patients = [{"patient_id": "P001"}, {"patient_id": "P002"}]
    idx = build_patient_index(patients)
    assert "P001" in idx and "P002" in idx


def test_build_trial_index():
    trials = [{"trial_id": "T001"}, {"trial_id": "T002"}]
    idx = build_trial_index(trials)
    assert "T001" in idx and "T002" in idx

