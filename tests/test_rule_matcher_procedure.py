"""Procedure-specific tests for rule_matcher.py."""

from app.eligibility.rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from app.models import CriterionDecision
from tests.helpers import make_patient, excl_trial


def test_dbs_synonym_history_predicts_not_eligible():
    patient = make_patient(key_features=["history of DBS surgery"])
    result = match_patient_to_trial(patient, excl_trial("Prior deep brain stimulation"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_full_name_predicts_not_eligible():
    patient = make_patient(key_features=["deep brain stimulation implanted previously"])
    result = match_patient_to_trial(patient, excl_trial("Prior DBS"))
    assert result["prediction"] == "not_eligible"


def test_dbs_synonym_criterion_level_met():
    patient = make_patient(key_features=["history of DBS surgery"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Prior deep brain stimulation"))
    assert results[0].decision == CriterionDecision.met


def test_no_dbs_history_not_not_eligible():
    patient = make_patient(key_features=["no history of DBS"])
    result = match_patient_to_trial(patient, excl_trial("Prior DBS"))
    assert result["prediction"] != "not_eligible"


def test_no_dbs_history_criterion_level_not_met():
    patient = make_patient(key_features=["no history of DBS"])
    results = match_patient_to_trial_criteria(patient, excl_trial("Prior DBS"))
    assert results[0].decision == CriterionDecision.not_met


# ---------------------------------------------------------------------------
# New general procedure/device synonym tests
# ---------------------------------------------------------------------------

def test_stn_dbs_synonym_predicts_not_eligible():
    """STN DBS phrasing in patient should match DBS exclusion."""
    patient = make_patient(key_features=["STN DBS implanted 2 years ago"])
    result = match_patient_to_trial(patient, excl_trial("Prior deep brain stimulation"))
    assert result["prediction"] == "not_eligible"


def test_subthalamic_stimulation_synonym_predicts_not_eligible():
    patient = make_patient(key_features=["subthalamic stimulation surgery"])
    result = match_patient_to_trial(patient, excl_trial("Prior DBS"))
    assert result["prediction"] == "not_eligible"


def test_dbs_criterion_synonym_full_name_criterion():
    """Exclusion phrased as 'deep brain stimulation' should match patient with DBS."""
    patient = make_patient(key_features=["DBS implanted"])
    results = match_patient_to_trial_criteria(patient, excl_trial("deep brain stimulation"))
    assert results[0].decision == CriterionDecision.met


def test_pacemaker_synonym_icd_criterion_level():
    """Patient with ICD should match 'implanted cardiac device' exclusion criterion."""
    from app.eligibility.clinical_terms import _patient_has_procedure
    assert _patient_has_procedure("patient has an ICD", "pacemaker")


def test_pacemaker_synonym_implanted_cardiac_device():
    from app.eligibility.clinical_terms import _patient_has_procedure
    assert _patient_has_procedure("implanted cardiac device present", "pacemaker")


def test_lcig_synonym_intestinal_gel_infusion():
    from app.eligibility.clinical_terms import _patient_has_procedure, _trial_involves_procedure
    assert _patient_has_procedure("on intestinal gel infusion", "lcig")
    assert _trial_involves_procedure("levodopa-carbidopa intestinal gel", "lcig")


def test_lcig_synonym_duodopa():
    from app.eligibility.clinical_terms import _patient_has_procedure
    assert _patient_has_procedure("currently receiving Duodopa", "lcig")


def test_tms_synonym_rtms():
    from app.eligibility.clinical_terms import _trial_involves_procedure
    assert _trial_involves_procedure("repetitive transcranial magnetic stimulation", "tms")
    assert _trial_involves_procedure("rTMS protocol", "tms")


def test_tdcs_synonym_transcranial_direct_current():
    from app.eligibility.clinical_terms import _trial_involves_procedure
    assert _trial_involves_procedure("transcranial direct current stimulation", "tdcs")
    assert _trial_involves_procedure("tACS trial", "tdcs")


def test_mri_synonym_fmri():
    from app.eligibility.clinical_terms import _trial_involves_procedure
    assert _trial_involves_procedure("fMRI session", "mri")
    assert _trial_involves_procedure("functional magnetic resonance imaging", "mri")


def test_negated_dbs_patient_procedure_helper_returns_false():
    from app.eligibility.clinical_terms import _patient_has_procedure
    assert not _patient_has_procedure("no history of DBS", "dbs")
    assert not _patient_has_procedure("no deep brain stimulation", "dbs")


def test_positive_dbs_patient_procedure_helper_returns_true():
    from app.eligibility.clinical_terms import _patient_has_procedure
    assert _patient_has_procedure("STN DBS implanted", "dbs")
    assert _patient_has_procedure("deep brain stimulation surgery", "dbs")

