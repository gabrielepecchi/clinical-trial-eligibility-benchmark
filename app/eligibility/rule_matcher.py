"""Simple rule-based baseline matcher for patient-trial eligibility."""

import re

from app.models import CriterionDecision, CriterionMatchResult, CriterionType

from app.eligibility.clinical_terms import (
    _any_match,
    _has_negated_dbs,
    _has_maob_inhibitor,
    _has_negated_maob,
    _DBS_PATTERNS,
    _MAOB_CRITERION_PATTERN,
    _MAOB_DRUGS,
    _COGNITIVE_EXCLUSION_PATTERNS,
    _UNCLEAR_MED_PATTERNS,
    _PARKINSON_PATTERNS,
    _STABLE_MED_PATTERNS,
    _MMSE_THRESHOLD_PATTERN,
    _MOCA_THRESHOLD_PATTERN,
    _MMSE_VALUE_PATTERN,
    _MOCA_VALUE_PATTERN,
    _HY_RANGE_PATTERN,
    _HY_VALUE_PATTERN,
    _UNVERIFIABLE_INCLUSION_PATTERNS,
    _patient_has_med_class,
    _patient_has_procedure,
    _trial_involves_procedure,
)

from app.eligibility.cognitive_rules import (
    _check_cognitive,
    _check_cognitive_exclusion_general,
    _check_cognitive_inclusion_minimum,
)

from app.eligibility.device_rules import (
    _check_dbs,
    _check_dbs_mri_compatibility,
    _check_dbs_required,
    _check_device_contraindication_stimulation,
)

from app.eligibility.clinical_units import (
    _patient_stable_weeks,
    _patient_changed_weeks_ago,
    check_lab_thresholds,
)

from app.eligibility.medication_rules import (
    _check_maob,
    _check_medication_stability,
    _check_medication_details_unclear,
    _required_weeks_extended,
)

from app.eligibility.diagnosis_rules import (
    _check_parkinson_diagnosis,
    _check_atypical_parkinsonism,
    _check_advanced_pd_required,
    _check_advanced_pd_requirement,
)

from app.eligibility.safety_rules import (
    _check_active_cancer,
    _check_active_cancer_hard_block,
    _check_oncology_required,
    _check_comorbidity_protocol_risk,
    _check_frailty_high_demand_exercise,
    _check_nonmotor_comorbidity_uncertainty,
)

from app.eligibility.temporal_rules import (
    _check_temporal_criteria,
)

from app.eligibility.unclear_rules import (
    _check_contradictions,
    _check_disease_stage_unclear,
    _check_recent_trial_participation,
    _check_parent_study_required,
    _check_missing_specific_inclusion_details,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()



def _looks_like_stage_not_age(text: str) -> bool:
    """Return True when a criterion likely describes disease stage, not age."""
    stage_terms = [
        "hoehn",
        "yahr",
        "stage",
        "stages",
        "hn y",
        "h&y",
        "h & y",
        "updrs",
        "disease stage",
        "severity stage",
    ]
    return any(term in text for term in stage_terms)


def _extract_age_range(criteria_list: list[str]) -> tuple[int | None, int | None]:
    """Return (min_age, max_age) parsed from inclusion criteria, or (None, None)."""
    min_age: int | None = None
    max_age: int | None = None

    for criterion in criteria_list:
        c = criterion.lower()

        if _looks_like_stage_not_age(c):
            continue

        m = re.search(
            r"\b(?:age|ages|aged)\s+(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})"
            r"(?:\s*(?:years?|yrs?|y/o|old|of age))?\b",
            c,
        )
        if m:
            low = int(m.group(1))
            high = int(m.group(2))
            if low >= 10 or high >= 10:
                min_age = low
                max_age = high
                break

        m = re.search(
            r"\b(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})\s*(?:years?|yrs?)\s*(?:of age)?\b",
            c,
        )
        if m:
            low = int(m.group(1))
            high = int(m.group(2))
            if low >= 10 or high >= 10:
                min_age = low
                max_age = high
                break

        m = re.search(r"\b(?:age|ages|aged)\s+(\d{1,3})(?:\s+years?)?\s+or\s+older\b", c)
        if m:
            value = int(m.group(1))
            if value >= 10:
                min_age = value
                break

        m = re.search(r"\b(?:age|ages|aged)\s*[≥>=]+\s*(\d{1,3})\b", c)
        if m:
            value = int(m.group(1))
            if value >= 10:
                min_age = value
                break

        m = re.search(r"\b(?:age|ages|aged)\s+(\d{1,3})(?:\s+years?)?\s+or\s+younger\b", c)
        if m:
            value = int(m.group(1))
            if value >= 10:
                max_age = value
                break

        m = re.search(r"\b(?:age|ages|aged)\s*[≤<=]+\s*(\d{1,3})\b", c)
        if m:
            value = int(m.group(1))
            if value >= 10:
                max_age = value
                break

    return min_age, max_age


def _score_from_features(text: str, patterns: list[str]) -> list[str]:
    """Return the subset of patterns that match text."""
    return [p for p in patterns if re.search(p, text)]


# ---------------------------------------------------------------------------
# Rule sets
# ---------------------------------------------------------------------------



def _count_unverifiable_inclusion_criteria(trial: dict) -> int:
    """Return the number of inclusion criteria that are logistical/external and cannot be verified from a patient profile."""
    count = 0
    for criterion in trial.get("inclusion_criteria", []):
        if _any_match(_UNVERIFIABLE_INCLUSION_PATTERNS, criterion.lower()):
            count += 1
    return count





# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------

def _check_age(patient: dict, trial: dict) -> tuple[str | None, str | None, str | None]:
    """Return (status, matched_fact, blocking_criterion)."""
    patient_age = patient.get("age")
    inclusion = trial.get("inclusion_criteria", [])
    min_age, max_age = _extract_age_range(inclusion)

    if min_age is None and max_age is None:
        return None, None, None

    if patient_age is None:
        return "unclear", None, "age criterion present but patient age unknown"

    age_range_str = (
        f"age {min_age}-{max_age}" if min_age is not None and max_age is not None
        else f"age >= {min_age}" if min_age is not None
        else f"age <= {max_age}"
    )

    too_young = min_age is not None and patient_age < min_age
    too_old = max_age is not None and patient_age > max_age

    if too_young or too_old:
        return (
            "not_eligible",
            f"patient age {patient_age}",
            f"trial requires {age_range_str}",
        )

    return "ok", f"patient age {patient_age} within {age_range_str}", None


def _age_miss_by_one(patient: dict, trial: dict) -> bool:
    """Return True if the patient misses the age boundary by exactly 1 year."""
    patient_age = patient.get("age")
    if patient_age is None:
        return False
    inclusion = trial.get("inclusion_criteria", [])
    min_age, max_age = _extract_age_range(inclusion)
    if min_age is not None and patient_age == min_age - 1:
        return True
    if max_age is not None and patient_age == max_age + 1:
        return True
    return False






_DBS_EXCLUSION_SIGNAL_PATTERNS = [
    r"\bprior\b.{0,30}(?:dbs|deep brain stimulation)",
    r"\bprevious\b.{0,30}(?:dbs|deep brain stimulation)",
    r"\bhistory\s+of\b.{0,30}(?:dbs|deep brain stimulation)",
    r"(?:dbs|deep brain stimulation).{0,30}\bimplant",
    r"(?:dbs|deep brain stimulation).{0,30}\bsurgery",
    r"(?:dbs|deep brain stimulation).{0,30}\bprocedure",
    r"exclude.{0,40}(?:dbs|deep brain stimulation)",
    r"(?:dbs|deep brain stimulation).{0,40}exclu",
    r"no\s+(?:prior|previous|history\s+of).{0,30}(?:dbs|deep brain stimulation)",
]

_DBS_STUDY_SIGNAL_PATTERNS = [
    r"dbs.{0,30}(?:effects?|outcome|programming|parameter|setting|stimulation\s+parameter)",
    r"(?:effects?|outcome|programming|parameter).{0,30}dbs",
    r"deep brain stimulation.{0,30}(?:effects?|outcome|programming|parameter)",
    r"(?:effects?|outcome|programming|parameter).{0,30}deep brain stimulation",
    r"dbs.{0,30}(?:implanted|recipient|patient|subject)",
    r"(?:implanted|recipient).{0,30}dbs",
    r"currently\s+(?:undergoing|receiving).{0,30}dbs",
    r"dbs\s+(?:on|off)\b",
    r"stimulation\s+(?:on|off)\b",
    r"dbs\s+(?:cohort|arm|group|population)",
    r"with\s+(?:existing|active|current)\s+dbs",
]

_DBS_REQUIRED_SIGNAL_PATTERNS = [
    r"(?:must|should|required?|eligible|inclusion).{0,40}(?:dbs|deep brain stimulation)",
    r"(?:dbs|deep brain stimulation).{0,40}(?:required|implanted|in\s+situ|present|implantation)",
    r"existing\s+dbs",
    r"with\s+dbs\b",
    r"prior\s+dbs\s+(?:required|implant)",
    r"\bprior\b.{0,30}(?:dbs|deep brain stimulation)",
    r"\bprevious\b.{0,30}(?:dbs|deep brain stimulation)\b",
    r"dbs\s+implantation",
    r"deep brain stimulation\s+implantation",
]


def _trial_excludes_dbs(trial: dict) -> bool:
    """Return True when the trial's exclusion criteria explicitly exclude prior DBS."""
    excl_list = trial.get("exclusion_criteria", [])
    for criterion in excl_list:
        c = criterion.lower()
        if _any_match(_DBS_EXCLUSION_SIGNAL_PATTERNS, c):
            # Check it's not actually a DBS-study or DBS-required trial
            if not _any_match(_DBS_STUDY_SIGNAL_PATTERNS, c):
                return True
    return False


def _trial_is_dbs_study(trial: dict) -> bool:
    """Return True if the trial is specifically studying DBS effects/outcomes/programming."""
    all_criteria = (
        trial.get("inclusion_criteria", [])
        + trial.get("exclusion_criteria", [])
    )
    trial_text = " ".join(all_criteria).lower()
    return _any_match(_DBS_STUDY_SIGNAL_PATTERNS, trial_text)


def _inclusion_section_only(text: str) -> str:
    """Return only the text before any Exclusion section marker.

    When a free-text eligibility_criteria string is copied verbatim into
    inclusion_criteria, any phrase after "exclusion:" / "exclusions:" must
    not be treated as an inclusion requirement.
    """
    # Split on the first occurrence of an exclusion header
    m = re.search(r"\bexclusion(?:s)?\s*:", text, re.IGNORECASE)
    if m:
        return text[: m.start()]
    return text


def _trial_requires_dbs(trial: dict) -> bool:
    """Return True when DBS is a genuine inclusion requirement.

    Critically, if the inclusion_criteria list contains a mixed free-text string
    that has an Exclusion section (e.g. from eligibility_criteria normalisation),
    only the text *before* that section is considered.
    """
    incl_list = trial.get("inclusion_criteria", [])
    for criterion in incl_list:
        c = _inclusion_section_only(criterion.lower())
        if _any_match(_DBS_REQUIRED_SIGNAL_PATTERNS, c):
            return True
    return False


def _patient_has_dbs_history(patient: dict) -> bool:
    """Return True if the patient has documented prior DBS."""
    if patient.get("dbs_history") is True:
        return True
    parts: list[str] = []
    for field in ("key_features", "medications", "exclusions", "procedures",
                  "procedure_history", "surgical_history"):
        v = patient.get(field, [])
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    summary = patient.get("summary", "")
    if summary:
        parts.append(str(summary))
    patient_text = " ".join(parts).lower()
    if _has_negated_dbs(patient_text):
        return False
    if _patient_has_procedure(patient_text, "dbs"):
        return True
    return False


def _check_dbs_exclusion_from_history(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block patient if they have prior DBS and the trial excludes prior DBS.

    Returns (blocking_criterion, matched_fact) or (None, None).
    Does NOT fire for DBS-study or DBS-required trials.
    """
    if _trial_is_dbs_study(trial) or _trial_requires_dbs(trial):
        return None, None
    if not _trial_excludes_dbs(trial):
        return None, None
    if _patient_has_dbs_history(patient):
        return (
            "prior deep brain stimulation — excluded by trial criteria",
            "patient has prior DBS",
        )
    return None, None


def _check_dbs_required_inclusion(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when trial requires prior DBS implantation but patient explicitly has no DBS history.

    Returns (blocking_criterion, matched_fact) or (None, None).
    """
    if not _trial_requires_dbs(trial):
        return None, None
    parts: list[str] = []
    for field in ("key_features", "medications", "exclusions", "procedures",
                  "procedure_history", "surgical_history"):
        v = patient.get(field, [])
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    summary = patient.get("summary", "")
    if summary:
        parts.append(str(summary))
    if patient.get("dbs_history") is True:
        parts.append("dbs history of dbs")
    patient_text = " ".join(parts).lower()
    if patient.get("dbs_history") is True or _patient_has_procedure(patient_text, "dbs"):
        return None, None
    if _has_negated_dbs(patient_text) or patient.get("dbs_history") is False:
        return (
            "trial requires prior DBS implantation; patient has no DBS history",
            "no DBS history documented",
        )
    return None, None


def _check_hy_stage(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if H&Y stage is out of range."""
    inclusion_list = trial.get("inclusion_criteria", [])
    patient_text = _text(patient.get("key_features", []))

    pvm = _HY_VALUE_PATTERN.search(patient_text)
    patient_stage = int(pvm.group(1)) if pvm else None

    for criterion in inclusion_list:
        m = _HY_RANGE_PATTERN.search(criterion)
        if m:
            hy_min = int(m.group(1))
            hy_max = int(m.group(2))
            if patient_stage is None:
                return None, None
            if not (hy_min <= patient_stage <= hy_max):
                return (
                    f"Hoehn and Yahr stage {hy_min} to {hy_max} required",
                    f"patient Hoehn and Yahr stage {patient_stage}",
                )

    return None, None



# ---------------------------------------------------------------------------
# Extended unclear checks
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Missing specific inclusion details (uncertainty only)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Non-motor / safety comorbidity uncertainty helper
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Structured missingness layer
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bdenies\b", r"\bdenied\b",
    r"\bnegative\b", r"\bno history of\b", r"\bwithout\b", r"\babsent\b",
]

_FIELD_NEGATION_MARKERS: dict[str, list[str]] = {
    "dbs": [r"no\s+(?:history\s+of\s+)?dbs", r"no\s+deep\s+brain", r"dbs.*(?:absent|not\s+present|not\s+implanted)"],
    "maob_inhibitor": [r"no\s+mao.b", r"not\s+taking.*(?:rasagiline|azilect|selegiline|deprenyl|eldepryl|safinamide|xadago)", r"mao.b.*(?:not\s+used|not\s+taken|absent)", r"no\s+(?:rasagiline|azilect|selegiline|deprenyl|eldepryl|safinamide|xadago)"],
    "cognitive_impairment": [r"no\s+cognitive\s+impairment", r"cognition\s+(?:intact|normal)", r"no\s+dementia"],
    "active_cancer": [r"no\s+(?:active\s+)?cancer", r"no\s+malignancy", r"cancer.*(?:absent|none|not\s+present)"],
    "investigational_drug": [r"no\s+investigational\s+drug", r"not\s+enrolled.*(?:trial|study)"],
    "trial_participation": [r"not\s+(?:enrolled|participating)\s+in.*(?:trial|study)", r"no\s+(?:prior|recent|concurrent)\s+(?:trial|study)"],
}

_FIELD_PRESENT_MARKERS: dict[str, list[str]] = {
    "dbs": [r"\bdbs\b", r"deep\s+brain\s+stimulation", r"dbs\s+implant"],
    "maob_inhibitor": [r"\brasagiline\b", r"\bazilect\b", r"\bselegiline\b", r"\bdeprenyl\b", r"\beldepryl\b", r"\bzelapar\b", r"\bsafinamide\b", r"\bxadago\b", r"mao.b\s+inhibitor"],
    "cognitive_impairment": [r"\bmmse\b", r"\bmoca\b", r"cognitive\s+impairment", r"\bdementia\b"],
    "active_cancer": [r"active\s+cancer", r"current\s+chemotherapy", r"ongoing.*malignancy"],
    "medication_details": [r"levodopa", r"dopamine\s+agonist", r"carbidopa"],
    "disease_stage": [r"hoehn\s+and\s+yahr", r"\bh&y\b", r"\bupdrs\b", r"stage\s+\d"],
    "cognitive_score": [r"mmse\s*(?:score\s*)?\d+", r"moca\s*(?:score\s*)?\d+"],
}

_MISSING_REASON_TYPE_MAP: dict[str, str] = {
    "age": "not_documented",
    "medication_details": "not_documented",
    "medication_stability_duration": "not_documented",
    "disease_stage_or_duration": "not_documented",
    "cognitive_score": "not_documented",
    "trial_participation_history": "ambiguous_documentation",
    "unverifiable_inclusion_criteria": "unverifiable",
}

_FIELD_UNCLEAR_REASON_MAP: dict[str, str] = {
    "age": "patient age not documented",
    "medication_details": "medication list absent or details unclear",
    "medication_stability_duration": "medication stability duration not documented",
    "disease_stage_or_duration": "disease stage or duration not documented or unclear",
    "cognitive_score": "MoCA/MMSE score required but not documented",
    "trial_participation_history": "recent or concurrent trial participation noted; washout eligibility ambiguous",
    "unverifiable_inclusion_criteria": "multiple inclusion criteria cannot be verified from the patient profile",
}


def _build_structured_missingness(
    patient: dict,
    trial: dict,
    missing_information: list[str],
    matched_facts: list[str],
    blocking_criteria: list[str],
    uncertain_criteria: list[str],
) -> dict:
    """Build structured missingness fields from existing rule outputs."""
    unknown_fields: list[str] = []
    present_evidence: list[str] = []
    absent_evidence: list[str] = []
    missing_information_details: list[dict] = []

    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list)
           else [str(patient.get("diagnosis", ""))])
    )

    # Collect present evidence from matched_facts
    for fact in matched_facts:
        f = fact.lower()
        for field, patterns in _FIELD_PRESENT_MARKERS.items():
            if _any_match(patterns, f):
                entry = f"{field}: {fact}"
                if entry not in present_evidence:
                    present_evidence.append(entry)

    # Collect absent evidence from negation patterns in patient text or blocking criteria
    for field, patterns in _FIELD_NEGATION_MARKERS.items():
        if _any_match(patterns, patient_text):
            entry = f"no {field.replace('_', ' ')} documented"
            if entry not in absent_evidence:
                absent_evidence.append(entry)

    # Also collect from blocking criteria that signal explicit negation/absence
    for bc in blocking_criteria:
        bc_lower = bc.lower()
        for field, patterns in _FIELD_NEGATION_MARKERS.items():
            if _any_match(patterns, bc_lower):
                entry = f"no {field.replace('_', ' ')} (from blocking: {bc})"
                if entry not in absent_evidence:
                    absent_evidence.append(entry)

    # Build missing_information_details and unknown_fields from missing_information list
    for field in missing_information:
        status = "unknown"
        reason_type = _MISSING_REASON_TYPE_MAP.get(field, "not_documented")
        unclear_reason = _FIELD_UNCLEAR_REASON_MAP.get(field, f"{field} not documented")

        # Check if there is present or absent evidence for this field
        field_norm = field.replace("_", " ")
        pev = next((e for e in present_evidence if field_norm in e.lower()), "")
        aev = next((e for e in absent_evidence if field_norm in e.lower()), "")

        if pev:
            status = "present"
        elif aev:
            status = "absent"
        else:
            status = "unknown"
            if field not in unknown_fields:
                unknown_fields.append(field)

        missing_information_details.append({
            "field": field,
            "status": status,
            "missing_reason_type": reason_type,
            "unclear_reason": unclear_reason,
            "present_evidence": pev,
            "absent_evidence": aev,
        })

    # Also flag unknown from uncertain_criteria that imply data absence without negation
    _UNCERTAIN_UNKNOWN_SIGNALS = [
        ("cognitive_score", [r"mmse.*not\s+available", r"moca.*not\s+available", r"cognitive\s+score.*not", r"score.*not\s+documented"]),
        ("disease_stage", [r"stage.*unclear", r"severity.*unclear", r"stage.*missing", r"stage.*not\s+recorded"]),
        ("medication_details", [r"medication.*not\s+documented", r"medication.*unclear", r"medication.*unavailable"]),
    ]
    for field, patterns in _UNCERTAIN_UNKNOWN_SIGNALS:
        uc_text = " ".join(uncertain_criteria).lower()
        if _any_match(patterns, uc_text):
            if field not in unknown_fields:
                unknown_fields.append(field)

    # Determine overall unclear_reason and missing_reason_type (summary of the most significant)
    if missing_information_details:
        first_unknown = next((d for d in missing_information_details if d["status"] == "unknown"), None)
        if first_unknown:
            unclear_reason_summary = first_unknown["unclear_reason"]
            missing_reason_type_summary = first_unknown["missing_reason_type"]
        else:
            unclear_reason_summary = missing_information_details[0]["unclear_reason"]
            missing_reason_type_summary = missing_information_details[0]["missing_reason_type"]
    elif uncertain_criteria:
        unclear_reason_summary = uncertain_criteria[0][:200]
        missing_reason_type_summary = "ambiguous_documentation"
    else:
        unclear_reason_summary = ""
        missing_reason_type_summary = ""

    return {
        "unknown_fields": unknown_fields,
        "present_evidence": present_evidence,
        "absent_evidence": absent_evidence,
        "unclear_reason": unclear_reason_summary,
        "missing_reason_type": missing_reason_type_summary,
        "missing_information_details": missing_information_details,
    }


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

def match_patient_to_trial(patient: dict, trial: dict) -> dict:
    """Match a patient dict to a trial dict using simple deterministic rules.

    Args:
        patient: A patient profile dictionary.
        trial:   A trial dictionary.

    Returns:
        A dictionary with keys:
            prediction         – 'eligible' | 'not_eligible' | 'unclear'
            confidence         – numeric score between 0.0 and 1.0
            matched_facts      – list of patient facts that satisfy inclusion criteria
            blocking_criteria  – list of criteria that disqualify the patient
            uncertain_criteria – list of criteria that could not be assessed
            explanation        – human-readable summary string
            missing_information – list of missing data categories (e.g. 'age', 'cognitive_score')
    """
    # Normalise free-text eligibility_criteria so downstream rules can use it.
    # Work on a shallow copy to avoid mutating the caller's dict.
    _ec_raw = trial.get("eligibility_criteria")
    if _ec_raw is not None:
        trial = dict(trial)
        _ec_text = _ec_raw if isinstance(_ec_raw, str) else " ".join(str(x) for x in _ec_raw)
        if not trial.get("inclusion_criteria"):
            trial["inclusion_criteria"] = [_ec_text]
        if not trial.get("exclusion_criteria"):
            trial["exclusion_criteria"] = [_ec_text]

    matched_facts: list[str] = []
    blocking_criteria: list[str] = []
    uncertain_criteria: list[str] = []

    # --- Age ---
    age_status, age_fact, age_block = _check_age(patient, trial)
    if age_status == "not_eligible":
        blocking_criteria.append(age_block)
    elif age_status == "unclear":
        uncertain_criteria.append(age_block)
    elif age_status == "ok" and age_fact:
        matched_facts.append(age_fact)

    # --- Lab / measurement thresholds (weight, BMI, creatinine, hemoglobin) ---
    _ec = trial.get("eligibility_criteria", [])
    if isinstance(_ec, str):
        _ec = [_ec]
    all_criteria = (
        list(_ec)
        + trial.get("inclusion_criteria", [])
        + trial.get("exclusion_criteria", [])
    )
    for lab_block, lab_fact in check_lab_thresholds(patient, all_criteria):
        if lab_block not in blocking_criteria:
            blocking_criteria.append(lab_block)
        if lab_fact not in matched_facts:
            matched_facts.append(lab_fact)

    # --- DBS ---
    dbs_block, dbs_fact = _check_dbs(patient, trial)
    if dbs_block:
        if dbs_block.startswith("__unclear__:"):
            uncertain_criteria.append(dbs_block[len("__unclear__:"):])
            if dbs_fact:
                matched_facts.append(dbs_fact)
        else:
            blocking_criteria.append(dbs_block)
            matched_facts.append(dbs_fact)

    # --- DBS history exclusion (dbs_history field + explicit exclusion in trial) ---
    dbs_hist_block, dbs_hist_fact = _check_dbs_exclusion_from_history(patient, trial)
    if dbs_hist_block and dbs_hist_block not in blocking_criteria:
        blocking_criteria.append(dbs_hist_block)
        if dbs_hist_fact:
            matched_facts.append(dbs_hist_fact)

    # --- DBS required by inclusion: block if patient explicitly has no DBS ---
    dbs_req_incl_block, dbs_req_incl_fact = _check_dbs_required_inclusion(patient, trial)
    if dbs_req_incl_block and dbs_req_incl_block not in blocking_criteria:
        blocking_criteria.append(dbs_req_incl_block)
        if dbs_req_incl_fact:
            matched_facts.append(dbs_req_incl_fact)

    # --- DBS + MRI/fMRI compatibility ---
    dbs_mri_block, dbs_mri_fact = _check_dbs_mri_compatibility(patient, trial)
    if dbs_mri_block and dbs_mri_block not in blocking_criteria:
        blocking_criteria.append(dbs_mri_block)
        if dbs_mri_fact:
            matched_facts.append(dbs_mri_fact)

    # --- MAO-B inhibitor ---
    maob_block, maob_fact = _check_maob(patient, trial)
    if maob_block:
        if maob_block.startswith("__unclear__:"):
            uncertain_criteria.append(maob_block[len("__unclear__:"):])
            if maob_fact:
                matched_facts.append(maob_fact)
        else:
            blocking_criteria.append(maob_block)
            if maob_fact:
                matched_facts.append(maob_fact)

    # --- Cognitive / MMSE / MoCA (numeric threshold) ---
    cog_block, cog_fact = _check_cognitive(patient, trial)
    if cog_block:
        blocking_criteria.append(cog_block)
        if cog_fact:
            matched_facts.append(cog_fact)

    # --- Cognitive exclusion — general (no numeric threshold) ---
    cog_gen_block, cog_gen_fact = _check_cognitive_exclusion_general(patient, trial)
    if cog_gen_block and cog_gen_block not in blocking_criteria:
        blocking_criteria.append(cog_gen_block)
        if cog_gen_fact:
            matched_facts.append(cog_gen_fact)

    # --- Cognitive inclusion minimum ---
    cog_min_block, cog_min_fact = _check_cognitive_inclusion_minimum(patient, trial)
    if cog_min_block:
        if cog_min_block.startswith("__unclear__:"):
            uncertain_criteria.append(cog_min_block[len("__unclear__:"):])
            if cog_min_fact:
                matched_facts.append(cog_min_fact)
        elif cog_min_block not in blocking_criteria:
            blocking_criteria.append(cog_min_block)
            if cog_min_fact:
                matched_facts.append(cog_min_fact)

    # --- MCI-only + DBS/neuropsychiatric/facial-expression/imaging outcome trial:
    #     prefer unclear over not_eligible when no numeric cutoff or explicit dementia exclusion ---
    if not blocking_criteria:
        _MCI_ONLY_PAT = [r"\bmci\b", r"mild cognitive impairment", r"mild\s+cognitive"]
        _DBS_NEURO_IMAGING_TRIAL_PAT = [
            r"\bdbs\b", r"deep brain stimulation", r"neuropsychiatric", r"neuropsychological",
            r"\bmri\b", r"imaging.*outcome", r"neuroimaging", r"cognitive.*outcome",
            r"facial.*expression", r"expression.*facial",
        ]
        patient_feat_text = _text(patient.get("key_features", []) + patient.get("exclusions", []))
        _HARD_COG_PAT = [r"\bdementia\b", r"(?:significant|moderate|severe).*cognitive", r"low moca", r"low mmse", r"impaired cognition"]
        trial_all_text = _text(trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", []))
        _EXPLICIT_NUM_CUTOFF = bool(_MMSE_THRESHOLD_PATTERN.search(trial_all_text) or _MOCA_THRESHOLD_PATTERN.search(trial_all_text))
        _EXPLICIT_DEMENTIA_EXCL = _any_match([r"\bdementia\b", r"cognitive impairment.*exclud", r"exclud.*cognitive impairment"], trial_all_text)
        if (
            _any_match(_MCI_ONLY_PAT, patient_feat_text)
            and not _any_match(_HARD_COG_PAT, patient_feat_text)
            and _any_match(_DBS_NEURO_IMAGING_TRIAL_PAT, trial_all_text)
            and not _EXPLICIT_NUM_CUTOFF
            and not _EXPLICIT_DEMENTIA_EXCL
        ):
            _unc = (
                "patient has mild cognitive impairment; trial involves DBS/neuropsychiatric/facial-expression/"
                "imaging outcomes without explicit numeric cognitive cutoff or dementia exclusion — eligibility uncertain"
            )
            if _unc not in uncertain_criteria:
                uncertain_criteria.append(_unc)

    # --- DBS required by inclusion ---
    dbs_req_block, dbs_req_fact = _check_dbs_required(patient, trial)
    if dbs_req_block:
        if dbs_req_block.startswith("__unclear__:"):
            uncertain_criteria.append(dbs_req_block[len("__unclear__:"):])
            if dbs_req_fact:
                matched_facts.append(dbs_req_fact)
        else:
            blocking_criteria.append(dbs_req_block)
            if dbs_req_fact:
                matched_facts.append(dbs_req_fact)

    # --- Device contraindication: pacemaker + stimulation (broad) ---
    dev_block, dev_fact = _check_device_contraindication_stimulation(patient, trial)
    if dev_block and dev_block not in blocking_criteria:
        blocking_criteria.append(dev_block)
        if dev_fact:
            matched_facts.append(dev_fact)

    # --- Parent/extension study required ---
    parent_status, parent_uncertain, parent_block = _check_parent_study_required(patient, trial)
    if parent_status == "not_eligible" and parent_block:
        blocking_criteria.append(parent_block)
    elif parent_status == "unclear" and parent_uncertain:
        uncertain_criteria.append(parent_uncertain)

    # --- Oncology diagnosis required ---
    onco_block, onco_fact = _check_oncology_required(patient, trial)
    if onco_block:
        blocking_criteria.append(onco_block)
        if onco_fact:
            matched_facts.append(onco_fact)

    # --- Active cancer hard block (invasive/surgical/implant trials only) ---
    cancer_hard_block, cancer_hard_fact = _check_active_cancer_hard_block(patient, trial)
    if cancer_hard_block and cancer_hard_block not in blocking_criteria:
        blocking_criteria.append(cancer_hard_block)
        if cancer_hard_fact:
            matched_facts.append(cancer_hard_fact)

    # --- Advanced PD required ---
    adv_pd_block, adv_pd_fact = _check_advanced_pd_requirement(patient, trial)
    if adv_pd_block:
        blocking_criteria.append(adv_pd_block)
        if adv_pd_fact:
            matched_facts.append(adv_pd_fact)

    # --- Frailty in high-demand exercise trial ---
    frailty_ex_block, frailty_ex_fact = _check_frailty_high_demand_exercise(patient, trial)
    if frailty_ex_block and frailty_ex_block not in blocking_criteria:
        blocking_criteria.append(frailty_ex_block)
        if frailty_ex_fact:
            matched_facts.append(frailty_ex_fact)

    # --- Hoehn and Yahr stage ---
    hy_block, hy_fact = _check_hy_stage(patient, trial)
    if hy_block:
        blocking_criteria.append(hy_block)
        if hy_fact:
            matched_facts.append(hy_fact)

    # --- Atypical parkinsonism (before general PD check) ---
    atyp_status, atyp_uncertain, atyp_block = _check_atypical_parkinsonism(patient, trial)
    if atyp_status == "not_eligible" and atyp_block:
        blocking_criteria.append(atyp_block)
    elif atyp_status == "unclear" and atyp_uncertain:
        uncertain_criteria.append(atyp_uncertain)
    else:
        # --- Parkinson diagnosis (standard check, skipped if atypical already flagged) ---
        pd_block, _ = _check_parkinson_diagnosis(patient, trial)
        if pd_block:
            if pd_block.startswith("__unclear__:"):
                uncertain_criteria.append(pd_block[len("__unclear__:"):])
            else:
                blocking_criteria.append(pd_block)
        else:
            patient_diag_text = _text(patient.get("diagnosis", []))
            if _any_match(_PARKINSON_PATTERNS, patient_diag_text):
                matched_facts.append("Parkinson disease diagnosis confirmed")

    # --- Medication stability ---
    med_uncertain, med_fact = _check_medication_stability(patient, trial)
    if med_uncertain:
        uncertain_criteria.append(med_uncertain)
        if med_fact:
            matched_facts.append(med_fact)

    # --- Extended: medication details unclear ---
    med_detail_uncertain, med_detail_fact = _check_medication_details_unclear(patient, trial)
    if med_detail_uncertain and med_detail_uncertain not in uncertain_criteria:
        uncertain_criteria.append(med_detail_uncertain)
        if med_detail_fact:
            matched_facts.append(med_detail_fact)

    # --- Extended: disease stage/severity unclear ---
    stage_uncertain, stage_fact = _check_disease_stage_unclear(patient, trial)
    if stage_uncertain:
        uncertain_criteria.append(stage_uncertain)
        if stage_fact:
            matched_facts.append(stage_fact)

    # --- Extended: active cancer in non-oncology trial ---
    cancer_uncertain, cancer_fact = _check_active_cancer(patient, trial)
    if cancer_uncertain:
        uncertain_criteria.append(cancer_uncertain)
        if cancer_fact:
            matched_facts.append(cancer_fact)

    # --- Extended: recent trial participation with washout requirements ---
    trial_part_uncertain, trial_part_fact = _check_recent_trial_participation(patient, trial)
    if trial_part_uncertain:
        uncertain_criteria.append(trial_part_uncertain)
        if trial_part_fact:
            matched_facts.append(trial_part_fact)

    # --- Contradictions (Task 8) ---
    contra_uncertainties, contra_missing = _check_contradictions(patient, trial)
    for cu, cf in contra_uncertainties:
        if cu not in uncertain_criteria:
            uncertain_criteria.append(cu)
            matched_facts.append(cf)

    # --- Temporal criteria (Task 7) ---
    temp_blocks, temp_uncertainties, temp_missing = _check_temporal_criteria(patient, trial)
    for tb, tf in temp_blocks:
        if tb not in blocking_criteria:
            blocking_criteria.append(tb)
            matched_facts.append(tf)
    for tu, tf in temp_uncertainties:
        if tu not in uncertain_criteria:
            uncertain_criteria.append(tu)
            matched_facts.append(tf)

    # --- Extended: comorbidity risk in protocol-sensitive trial ---
    comorbid_block, comorbid_uncertain, comorbid_fact = _check_comorbidity_protocol_risk(patient, trial)
    if comorbid_block:
        blocking_criteria.append(comorbid_block)
        if comorbid_fact:
            matched_facts.append(comorbid_fact)
    elif comorbid_uncertain:
        uncertain_criteria.append(comorbid_uncertain)
        if comorbid_fact:
            matched_facts.append(comorbid_fact)

    # --- Extended: non-motor/safety comorbidity uncertainty (uncertainty only) ---
    if not blocking_criteria:
        for nm_unc in _check_nonmotor_comorbidity_uncertainty(patient, trial, existing_uncertain=uncertain_criteria):
            if nm_unc not in uncertain_criteria:
                uncertain_criteria.append(nm_unc)

    # --- Extended: unverifiable inclusion criteria burden ---
    unverifiable_count = _count_unverifiable_inclusion_criteria(trial)
    if unverifiable_count >= 3 and not blocking_criteria:
        uncertain_criteria.append(
            f"unverifiable inclusion criteria: {unverifiable_count} inclusion criteria"
            " cannot be verified from the patient profile"
            " (e.g. device operation ability, home internet access,"
            " concurrent trial participation, physician clearance)"
        )

    # --- Missing specific inclusion details (uncertainty only, runs when no blocking) ---
    if not blocking_criteria:
        missing_detail_uncertainties = _check_missing_specific_inclusion_details(patient, trial)
        for unc in missing_detail_uncertainties:
            if unc not in uncertain_criteria:
                uncertain_criteria.append(unc)

    # --- Borderline age: downgrade not_eligible → unclear when miss is exactly 1 year
    #     and there are already protocol/safety uncertainties ---
    if (
        blocking_criteria
        and len(blocking_criteria) == 1
        and age_block is not None
        and blocking_criteria[0] == age_block
        and _age_miss_by_one(patient, trial)
        and uncertain_criteria
    ):
        blocking_criteria.clear()
        uncertain_criteria.insert(
            0,
            f"borderline age: patient age is within 1 year of the trial age boundary "
            f"({age_block}); eligibility uncertain given protocol uncertainties",
        )

    # --- Determine prediction ---
    if blocking_criteria:
        prediction = "not_eligible"
        confidence = 0.90
        explanation = (
            "Patient does not meet eligibility requirements. "
            "Blocking criteria: " + "; ".join(blocking_criteria) + "."
        )
    elif uncertain_criteria:
        prediction = "unclear"
        confidence = 0.40
        explanation = (
            "Eligibility cannot be determined due to missing or unverifiable information. "
            "Uncertain criteria: " + "; ".join(uncertain_criteria) + "."
        )
    else:
        prediction = "eligible"
        confidence = 0.75 if matched_facts else 0.60
        explanation = (
            "No blocking or uncertain criteria identified. "
            + (
                "Matched facts: " + "; ".join(matched_facts) + "."
                if matched_facts
                else "No specific matched facts recorded."
            )
        )

    # --- Build missing_information checklist ---
    missing_information: list[str] = []

    if age_status == "unclear":
        missing_information.append("age")

    if med_uncertain or med_detail_uncertain:
        missing_information.append("medication_details")

    # medication_stability_duration: trial requires duration, patient doesn't satisfy it
    inclusion_list = trial.get("inclusion_criteria", [])
    _med_parts = []
    for _f in ("medications", "current_medications", "medication_history", "key_features"):
        _v = patient.get(_f, [])
        if isinstance(_v, list):
            _med_parts.extend(str(x) for x in _v)
        elif _v:
            _med_parts.append(str(_v))
    patient_med_text = " ".join(_med_parts).lower()
    for criterion in inclusion_list:
        req = _required_weeks_extended(criterion)
        if req is None:
            continue
        patient_weeks = _patient_stable_weeks(patient_med_text)
        changed_ago = _patient_changed_weeks_ago(patient_med_text)
        satisfied = (
            patient_weeks is not None and patient_weeks >= req
            and (changed_ago is None or changed_ago >= req)
        )
        if not satisfied and "medication_stability_duration" not in missing_information:
            missing_information.append("medication_stability_duration")
        break

    if stage_uncertain:
        missing_information.append("disease_stage_or_duration")

    if unverifiable_count >= 3 and not blocking_criteria:
        missing_information.append("unverifiable_inclusion_criteria")

    if trial_part_uncertain:
        missing_information.append("trial_participation_history")

    for mk in temp_missing:
        if mk not in missing_information:
            missing_information.append(mk)

    for mk in contra_missing:
        if mk not in missing_information:
            missing_information.append(mk)

    # cognitive_score: MMSE/MoCA required but score absent
    exclusion_list = trial.get("exclusion_criteria", [])
    patient_features = _text(patient.get("key_features", []))
    for criterion in exclusion_list:
        if _MMSE_THRESHOLD_PATTERN.search(criterion):
            if not _MMSE_VALUE_PATTERN.search(patient_features):
                if "cognitive_score" not in missing_information:
                    missing_information.append("cognitive_score")
        if _MOCA_THRESHOLD_PATTERN.search(criterion):
            if not _MOCA_VALUE_PATTERN.search(patient_features):
                if "cognitive_score" not in missing_information:
                    missing_information.append("cognitive_score")

    # --- Build structured missingness layer ---
    structured = _build_structured_missingness(
        patient, trial, missing_information, matched_facts, blocking_criteria, uncertain_criteria
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "matched_facts": matched_facts,
        "blocking_criteria": blocking_criteria,
        "uncertain_criteria": uncertain_criteria,
        "explanation": explanation,
        "missing_information": missing_information,
        # Structured missingness fields
        "unknown_fields": structured["unknown_fields"],
        "present_evidence": structured["present_evidence"],
        "absent_evidence": structured["absent_evidence"],
        "unclear_reason": structured["unclear_reason"],
        "missing_reason_type": structured["missing_reason_type"],
        "missing_information_details": structured["missing_information_details"],
    }


def match_patient_to_trial_criteria(
    patient: dict, trial: dict
) -> list[CriterionMatchResult]:
    """Evaluate each trial criterion individually against a patient.

    Args:
        patient: A patient profile dictionary.
        trial:   A trial dictionary.

    Returns:
        One CriterionMatchResult per inclusion and exclusion criterion.
    """
    summary = match_patient_to_trial(patient, trial)
    blocking = [b.lower() for b in summary["blocking_criteria"]]
    uncertain = [u.lower() for u in summary["uncertain_criteria"]]

    results: list[CriterionMatchResult] = []

    for criterion in trial.get("inclusion_criteria", []):
        c_lower = criterion.lower()
        decision, reason = _evaluate_inclusion_criterion(
            c_lower, patient, blocking, uncertain
        )
        results.append(
            CriterionMatchResult(
                criterion_text=criterion,
                criterion_type=CriterionType.inclusion,
                decision=decision,
                reason=reason,
            )
        )

    for criterion in trial.get("exclusion_criteria", []):
        c_lower = criterion.lower()
        decision, reason = _evaluate_exclusion_criterion(
            c_lower, patient, blocking, uncertain
        )
        results.append(
            CriterionMatchResult(
                criterion_text=criterion,
                criterion_type=CriterionType.exclusion,
                decision=decision,
                reason=reason,
            )
        )

    return results


def _evaluate_inclusion_criterion(
    c_lower: str, patient: dict, blocking: list[str], uncertain: list[str]
) -> tuple[CriterionDecision, str]:
    """Return (decision, reason) for a single inclusion criterion."""
    # Age criterion
    if re.search(r"\bage\b", c_lower) and not _looks_like_stage_not_age(c_lower):
        min_age, max_age = _extract_age_range([c_lower])
        patient_age = patient.get("age")
        if min_age is not None or max_age is not None:
            if patient_age is None:
                return CriterionDecision.unknown, "patient age not available"
            too_young = min_age is not None and patient_age < min_age
            too_old = max_age is not None and patient_age > max_age
            if too_young or too_old:
                return CriterionDecision.not_met, f"patient age {patient_age} out of range"
            return CriterionDecision.met, f"patient age {patient_age} in range"

    # Parkinson diagnosis
    if _any_match(_PARKINSON_PATTERNS, c_lower):
        diag_text = _text(patient.get("diagnosis", []))
        if _any_match(_PARKINSON_PATTERNS, diag_text):
            return CriterionDecision.met, "Parkinson disease diagnosis confirmed"
        if any("parkinson" in b for b in blocking):
            return CriterionDecision.not_met, "Parkinson disease diagnosis not found"
        return CriterionDecision.unknown, "diagnosis status unclear"

    # Stable medication
    if _any_match(_STABLE_MED_PATTERNS, c_lower):
        if any("stable medication" in u or "medication" in u for u in uncertain):
            return CriterionDecision.unknown, "medication stability cannot be confirmed"
        med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
        if _any_match(_UNCLEAR_MED_PATTERNS, med_text):
            return CriterionDecision.unknown, "medication details unclear"
        # Numeric duration check
        req = _required_weeks_extended(c_lower)
        if req is not None:
            changed_ago = _patient_changed_weeks_ago(med_text)
            if changed_ago is not None and changed_ago < req:
                return CriterionDecision.not_met, f"medication changed {changed_ago} week(s) ago; {req} weeks stable required"
            patient_weeks = _patient_stable_weeks(med_text)
            if patient_weeks is not None:
                if patient_weeks >= req:
                    return CriterionDecision.met, f"medication stable {patient_weeks} week(s) (required: {req})"
                return CriterionDecision.not_met, f"medication stable only {patient_weeks} week(s); {req} weeks required"
            return CriterionDecision.unknown, "medication stability duration not documented"
        if _any_match([r"levodopa", r"medication"], med_text):
            return CriterionDecision.met, "medication recorded"
        return CriterionDecision.unknown, "cannot confirm medication stability"

    # H&Y stage
    if _any_match([r"hoehn\s+and\s+yahr", r"\bh&y\b", r"\bhy\b\s*stage"], c_lower):
        if any("hoehn and yahr" in b or "h&y" in b for b in blocking):
            return CriterionDecision.not_met, "H&Y stage out of required range"
        patient_text = _text(patient.get("key_features", []))
        if _HY_VALUE_PATTERN.search(patient_text):
            return CriterionDecision.met, "H&Y stage within range"
        return CriterionDecision.unknown, "H&Y stage not available"

    # DBS required (inclusion)
    _DBS_INCL_DETECT = [
        r"\bprior\b.{0,30}(?:dbs|deep brain stimulation)",
        r"\bprevious\b.{0,30}(?:dbs|deep brain stimulation)",
        r"(?:must|should|required?|eligible|inclusion).{0,40}(?:dbs|deep brain stimulation)",
        r"(?:dbs|deep brain stimulation).{0,40}(?:required|implanted|in\s+situ|present|implantation)",
        r"existing\s+dbs",
        r"dbs\s+implantation",
        r"deep brain stimulation\s+implantation",
    ]
    if _any_match(_DBS_INCL_DETECT, c_lower):
        _parts: list[str] = []
        for _f in ("key_features", "medications", "exclusions", "procedures",
                   "procedure_history", "surgical_history"):
            _v = patient.get(_f, [])
            if isinstance(_v, list):
                _parts.extend(str(x) for x in _v)
            elif _v:
                _parts.append(str(_v))
        _s = patient.get("summary", "")
        if _s:
            _parts.append(str(_s))
        if patient.get("dbs_history") is True:
            _parts.append("dbs history of dbs")
        _pt = " ".join(_parts).lower()
        if patient.get("dbs_history") is True or _patient_has_procedure(_pt, "dbs"):
            return CriterionDecision.met, "DBS history confirmed — inclusion criterion satisfied"
        if _has_negated_dbs(_pt) or patient.get("dbs_history") is False:
            return CriterionDecision.not_met, "no DBS history — inclusion criterion not satisfied"
        return CriterionDecision.unknown, "DBS history not documented"

    return CriterionDecision.unknown, "cannot evaluate from available data"


def _evaluate_exclusion_criterion(
    c_lower: str, patient: dict, blocking: list[str], uncertain: list[str]
) -> tuple[CriterionDecision, str]:
    """Return (decision, reason) for a single exclusion criterion.

    For exclusions: met = criterion applies (patient IS excluded), not_met = criterion does not apply.
    """
    # DBS
    if _any_match(_DBS_PATTERNS, c_lower) or _trial_involves_procedure(c_lower, "dbs"):
        _dbs_parts = []
        for _f in ("key_features", "medications", "exclusions", "procedures",
                   "procedure_history", "surgical_history"):
            _v = patient.get(_f, [])
            if isinstance(_v, list):
                _dbs_parts.extend(str(x) for x in _v)
            elif _v:
                _dbs_parts.append(str(_v))
        _summary = patient.get("summary", "")
        if _summary:
            _dbs_parts.append(str(_summary))
        if patient.get("dbs_history") is True:
            _dbs_parts.append("dbs history of dbs")
        patient_text = " ".join(_dbs_parts).lower()
        if _has_negated_dbs(patient_text):
            return CriterionDecision.not_met, "no DBS history documented"
        if any("dbs" in b or "deep brain" in b for b in blocking):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        if _patient_has_procedure(patient_text, "dbs"):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        return CriterionDecision.not_met, "no DBS implant found"

    # MAO-B inhibitor
    if _MAOB_CRITERION_PATTERN.search(c_lower) or _any_match(_MAOB_DRUGS, c_lower):
        patient_med_text = _text(
            patient.get("medications", [])
            + patient.get("current_medications", [])
            + patient.get("medication_history", [])
            + patient.get("key_features", [])
            + patient.get("exclusions", [])
        )
        summary = patient.get("summary", "")
        if summary:
            patient_med_text = patient_med_text + " " + str(summary).lower()
        if _has_negated_maob(patient_med_text):
            return CriterionDecision.not_met, "no MAO-B inhibitor use documented"
        if _has_maob_inhibitor(patient_med_text) or _patient_has_med_class(patient_med_text, "maob_inhibitor"):
            return CriterionDecision.met, "MAO-B inhibitor present — patient excluded"
        return CriterionDecision.not_met, "no MAO-B inhibitor found"

    # MMSE
    m = _MMSE_THRESHOLD_PATTERN.search(c_lower)
    if m:
        threshold = int(m.group(1))
        patient_features = _text(patient.get("key_features", []))
        vm = _MMSE_VALUE_PATTERN.search(patient_features)
        if vm:
            score = int(vm.group(1))
            if score < threshold:
                return CriterionDecision.met, f"MMSE {score} below threshold {threshold} — excluded"
            return CriterionDecision.not_met, f"MMSE {score} meets threshold"
        return CriterionDecision.unknown, "MMSE score not available"

    # MoCA
    m = _MOCA_THRESHOLD_PATTERN.search(c_lower)
    if m:
        threshold = int(m.group(1))
        patient_features = _text(patient.get("key_features", []))
        vm = _MOCA_VALUE_PATTERN.search(patient_features)
        if vm:
            score = int(vm.group(1))
            if score < threshold:
                return CriterionDecision.met, f"MoCA {score} below threshold {threshold} — excluded"
            return CriterionDecision.not_met, f"MoCA {score} meets threshold"
        return CriterionDecision.unknown, "MoCA score not available"

    # Cognitive impairment (general)
    if _any_match(_COGNITIVE_EXCLUSION_PATTERNS, c_lower):
        if any("mmse" in b or "moca" in b or "cognitive" in b for b in blocking):
            return CriterionDecision.met, "cognitive impairment noted — excluded"
        return CriterionDecision.unknown, "cognitive status unclear"

    return CriterionDecision.unknown, "cannot evaluate from available data"
