"""Simple rule-based baseline matcher for patient-trial eligibility."""

import re

from app.models import CriterionDecision, CriterionMatchResult, CriterionType

from app.eligibility.clinical_terms import (
    _any_match,
    is_negated,
    is_affirmed,
    has_contradiction,
    _has_negated_dbs,
    _has_maob_inhibitor,
    _has_negated_maob,
    _DBS_PATTERNS, _DBS_NEGATION_PATTERN,
    _MAOB_CRITERION_PATTERN, _MAOB_DRUGS, _MAOB_NEGATION_PATTERN,
    _COGNITIVE_EXCLUSION_PATTERNS, _UNCLEAR_MED_PATTERNS,
    _PARKINSON_PATTERNS, _STABLE_MED_PATTERNS,
    _MMSE_THRESHOLD_PATTERN, _MOCA_THRESHOLD_PATTERN,
    _MMSE_VALUE_PATTERN, _MOCA_VALUE_PATTERN,
    _STABILITY_CRITERION_PATTERN, _PATIENT_STABLE_DURATION_PATTERN,
    _PATIENT_CHANGED_PATTERN, _HY_RANGE_PATTERN, _HY_VALUE_PATTERN,
    _TRIAL_MED_SPECIFIC_PATTERNS, _PATIENT_UNCLEAR_MED_PATTERNS,
    _TRIAL_STAGE_SEVERITY_PATTERNS, _PATIENT_UNCLEAR_STAGE_PATTERNS,
    _ATYPICAL_PARKINSON_PATTERNS, _IDIOPATHIC_PD_REQUIRED_PATTERNS,
    _ACTIVE_CANCER_PATTERNS, _TRIAL_SAFETY_SENSITIVE_PATTERNS,
    _RECENT_TRIAL_PATTERNS, _TRIAL_WASHOUT_PATTERNS,
    _PATIENT_COMPLEX_COMORBIDITY_PATTERNS, _TRIAL_COMPLEX_FOCUS_PATTERNS,
    _COMORBIDITY_TARGET_PAIRS, _HARD_CONTRAINDICATION_PAIRS,
    _PATIENT_COGNITIVE_IMPAIRMENT_PATTERNS,
    _TRIAL_COGNITIVE_EXCLUSION_GENERAL_PATTERNS,
    _TRIAL_COGNITIVE_INCLUSION_MIN_PATTERNS,
    _MMSE_INCLUSION_MIN_PATTERN, _MOCA_INCLUSION_MIN_PATTERN,
    _TRIAL_DBS_REQUIRED_PATTERNS, _AMBIGUOUS_DBS_INCLUSION_PATTERNS,
    _TRIAL_STIMULATION_PATTERNS,
    _TRIAL_PARENT_STUDY_REQUIRED_PATTERNS, _PATIENT_PRIOR_STUDY_PATTERNS,
    _TRIAL_ONCOLOGY_REQUIRED_PATTERNS, _PATIENT_CANCER_PATTERNS,
    _TRIAL_HIGH_DEMAND_EXERCISE_PATTERNS, _PATIENT_FRAILTY_FALL_PATTERNS,
    _UNVERIFIABLE_INCLUSION_PATTERNS,
    _HEALTHY_CONTROL_TRIAL_PATTERNS, _HEALTHY_CONTROL_AMBIGUITY_SIGNALS,
    _PATIENT_HEALTHY_CONTROL_PATTERNS, _INTERVENTIONAL_PD_ONLY_PATTERNS,
    _FOG_GAIT_TRIAL_PATTERNS, _FOG_GAIT_PATIENT_PATTERNS,
    _COG_MCI_TRIAL_PATTERNS, _COG_MCI_PATIENT_PATTERNS,
    _SEVERITY_TRIAL_PATTERNS, _SEVERITY_PATIENT_PATTERNS,
    _MED_SPECIFIC_TRIAL_PATTERNS, _MED_DOCUMENTED_PATIENT_PATTERNS,
    _LANG_SCALE_TRIAL_PATTERNS, _LANG_PATIENT_PATTERNS,
    _FRAILTY_TARGET_SUPPRESSION_PATTERNS, _RBD_TARGET_SUPPRESSION_PATTERNS,
    _RBD_AMBIGUITY_TRIGGER_PATTERNS, _DEPRESSION_IMAGING_BIOMARKER_PATTERNS,
    _ACTIVE_CANCER_PATIENT_PATTERNS, _ACTIVE_CANCER_TRIAL_PATTERNS,
    _MED_SYNONYMS, _patient_has_med_class, _trial_requires_med_class,
    _normalize_med_text,
    _PROCEDURE_SYNONYMS, _patient_has_procedure, _trial_involves_procedure,
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
    _to_weeks,
    _required_weeks,
    _patient_stable_weeks,
    _patient_changed_weeks_ago,
    check_lab_thresholds,
    parse_temporal_exclusion,
    parse_temporal_inclusion,
    get_patient_elapsed_days,
)

# ---------------------------------------------------------------------------
# Local patterns
# ---------------------------------------------------------------------------

_STABLE_REGIMEN_DURATION_PATTERN = re.compile(
    r"stable\s+\w+(?:\s+\w+)?\s+regimen\s+for\s+at\s+least\s+(\d+)\s+(weeks?|months?)",
    re.IGNORECASE,
)


def _required_weeks_extended(criterion: str) -> int | None:
    """Like _required_weeks but also matches 'stable <drug> regimen for at least N weeks'."""
    result = _required_weeks(criterion)
    if result is not None:
        return result
    m = _STABLE_REGIMEN_DURATION_PATTERN.search(criterion)
    if not m:
        return None
    return _to_weeks(int(m.group(1)), m.group(2))


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


def _check_maob(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if MAO-B inhibitor exclusion applies."""
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    if not _MAOB_CRITERION_PATTERN.search(exclusion_text):
        return None, None
    patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
    if has_contradiction(patient_med_text, "maob_inhibitor"):
        return (
            "__unclear__:contradictory MAO-B inhibitor records: both negation and affirmation found — eligibility cannot be determined",
            "contradiction in MAO-B inhibitor records",
        )
    if _has_maob_inhibitor(patient_med_text) or _patient_has_med_class(patient_med_text, "maob_inhibitor"):
        return "MAO-B inhibitor use is an exclusion criterion", "MAO-B inhibitor medication present"
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




def _check_parkinson_diagnosis(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, None) if Parkinson diagnosis is required but missing."""
    inclusion_text = _text(trial.get("inclusion_criteria", []))
    if not _any_match(_PARKINSON_PATTERNS, inclusion_text):
        return None, None

    patient_diagnosis_text = _text(patient.get("diagnosis", []))
    if _any_match(_PARKINSON_PATTERNS, patient_diagnosis_text):
        return None, None

    _TRIAL_META_FIELDS = [
        "title", "brief_title", "official_title", "summary", "brief_summary",
        "description", "detailed_description",
    ]
    meta_text = " ".join(
        _text(trial.get(f, "") or "") for f in _TRIAL_META_FIELDS
    )
    trial_full = inclusion_text + " " + _text(trial.get("exclusion_criteria", [])) + " " + meta_text

    is_interventional = _any_match(_INTERVENTIONAL_PD_ONLY_PATTERNS, trial_full)

    # Patient is healthy control / no neurological diagnosis
    if _any_match(_PATIENT_HEALTHY_CONTROL_PATTERNS, patient_diagnosis_text):
        if _any_match(_HEALTHY_CONTROL_AMBIGUITY_SIGNALS, trial_full) and not is_interventional:
            return (
                "__unclear__:patient is a healthy control/volunteer; trial mentions Parkinson disease "
                "but also has comparator/control group language — eligibility as control participant is unclear",
                None,
            )
        if is_interventional:
            return "Parkinson disease diagnosis required", None
        return (
            "__unclear__:patient is a healthy control/volunteer; Parkinson disease may be required "
            "but trial scope is ambiguous",
            None,
        )

    if _any_match(_HEALTHY_CONTROL_TRIAL_PATTERNS, trial_full):
        if not is_interventional:
            return (
                "__unclear__:trial may include healthy/control comparator participants; "
                "Parkinson diagnosis requirement cannot be interpreted as a hard exclusion from available text",
                None,
            )

    return "Parkinson disease diagnosis required", None


def _check_medication_stability(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (uncertain_criterion, matched_fact) if medication stability is unclear or insufficient."""
    inclusion_list = trial.get("inclusion_criteria", [])
    inclusion_text = _text(inclusion_list)
    if not _any_match(_STABLE_MED_PATTERNS, inclusion_text):
        return None, None

    patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
    if _any_match(_UNCLEAR_MED_PATTERNS, patient_med_text):
        return (
            "stable medication regimen required but cannot be confirmed",
            "medication dose, frequency, or compliance unclear",
        )

    # Numeric duration check
    for criterion in inclusion_list:
        req = _required_weeks_extended(criterion)
        if req is None:
            continue
        changed_ago = _patient_changed_weeks_ago(patient_med_text)
        if changed_ago is not None and changed_ago < req:
            return (
                f"stable medication regimen for at least {req} week(s) required; "
                f"medication changed {changed_ago} week(s) ago",
                f"medication changed {changed_ago} week(s) ago (required: {req} weeks stable)",
            )
        patient_weeks = _patient_stable_weeks(patient_med_text)
        if patient_weeks is not None and patient_weeks < req:
            return (
                f"stable medication regimen for at least {req} week(s) required; "
                f"patient stable for only {patient_weeks} week(s)",
                f"medication stable {patient_weeks} week(s) (required: {req} weeks)",
            )
        if patient_weeks is None and changed_ago is None:
            return (
                f"stable medication regimen for at least {req} week(s) required but duration not documented",
                "medication stability duration not documented",
            )

    return None, None




def _check_temporal_criteria(
    patient: dict, trial: dict
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Check temporal inclusion/exclusion criteria against patient data.

    Returns:
        (blocks, uncertainties, missing_keys)
        blocks: list of (blocking_criterion, matched_fact)
        uncertainties: list of (uncertain_criterion, matched_fact)
        missing_keys: list of missing_information keys
    """
    blocks: list[tuple[str, str]] = []
    uncertainties: list[tuple[str, str]] = []
    missing_keys: list[str] = []

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    patient_diag_text = _text(
        [str(patient.get("disease_duration", "") or "")]
        + patient.get("key_features", [])
        + [patient.get("summary", "")]
    )

    # --- Temporal exclusions ---
    for criterion in trial.get("exclusion_criteria", []):
        parsed = parse_temporal_exclusion(criterion)
        if parsed is None:
            continue
        topic, max_days = parsed
        # Pick relevant patient text for the topic
        if topic in ("medication_change",):
            p_text = _text(patient.get("medications", []) + patient.get("key_features", []))
        elif topic in ("dbs_surgery", "surgery"):
            p_text = _text(patient.get("key_features", []) + patient.get("exclusions", []))
        else:
            p_text = patient_all_text

        elapsed = get_patient_elapsed_days(p_text)
        if elapsed is None:
            # Temporal info missing — unclear only if patient has any signal for the topic
            _TOPIC_PATIENT_SIGNALS = {
                "medication_change": [r"medication.*changed", r"adjusted.*dose", r"new.*medication"],
                "investigational_drug": [r"investigational", r"experimental.*drug", r"study.*drug"],
                "trial_participation": [r"clinical.*trial", r"study.*participation", r"enrolled.*trial"],
                "dbs_surgery": [r"dbs", r"deep.*brain.*stimulation", r"dbs.*surgery"],
                "surgery": [r"surgery", r"surgical", r"operation"],
            }
            signals = _TOPIC_PATIENT_SIGNALS.get(topic, [])
            if signals and _any_match(signals, p_text):
                unc_msg = f"temporal exclusion: {criterion.strip()} — patient history present but timing not documented"
                uncertainties.append((unc_msg, f"{topic} noted but timing unknown"))
                if f"{topic}_timing" not in missing_keys:
                    missing_keys.append(f"{topic}_timing")
        elif elapsed <= max_days:
            # Violation — within exclusion window
            blocks.append((
                f"temporal exclusion violated: {criterion.strip()}",
                f"{topic} occurred {elapsed} day(s) ago (must be > {max_days} days ago)",
            ))
        # else: elapsed > max_days → satisfies exclusion, no flag

    # --- Temporal inclusions (disease duration / symptom duration) ---
    for criterion in trial.get("inclusion_criteria", []):
        parsed = parse_temporal_inclusion(criterion)
        if parsed is None:
            continue
        topic, threshold_days, direction = parsed

        elapsed = get_patient_elapsed_days(patient_diag_text)
        if elapsed is None:
            # Check if any duration field is present but unknown
            raw = patient.get("disease_duration")
            if raw is None or str(raw).lower() in ("none", "unknown", "unclear", ""):
                unc_msg = f"temporal inclusion: {criterion.strip()} — {topic} not documented"
                uncertainties.append((unc_msg, f"{topic} not documented"))
                if topic not in missing_keys:
                    missing_keys.append(topic)
        else:
            if direction == "at_least" and elapsed < threshold_days:
                blocks.append((
                    f"temporal inclusion not met: {criterion.strip()}",
                    f"{topic} is {elapsed} day(s); required >= {threshold_days} days",
                ))
            elif direction == "less_than" and elapsed >= threshold_days:
                blocks.append((
                    f"temporal inclusion not met: {criterion.strip()}",
                    f"{topic} is {elapsed} day(s); required < {threshold_days} days",
                ))

    return blocks, uncertainties, missing_keys


def _check_contradictions(
    patient: dict, trial: dict
) -> tuple[list[tuple[str, str]], list[str]]:
    """Detect contradictory patient facts for trial-relevant topics.

    Returns (uncertainties, missing_keys) where each uncertainty is (criterion_text, fact_text).
    Contradictory facts (both negated and affirmed) produce unclear rather than a hard block.
    """
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list) else [str(patient.get("diagnosis", ""))])
    )

    # Topics relevant to hard exclusions checked by rule_matcher
    _EXCLUSION_TOPICS = ["dbs", "maob_inhibitor", "cognitive_impairment", "active_cancer",
                         "investigational_drug", "trial_participation"]

    uncertainties: list[tuple[str, str]] = []
    missing_keys: list[str] = []

    for topic in _EXCLUSION_TOPICS:
        if has_contradiction(patient_all_text, topic):
            msg = (
                f"contradictory patient records for {topic.replace("_", " ")}: "                f"both negation and affirmation found — eligibility cannot be determined"
            )
            uncertainties.append((msg, f"contradiction in {topic} records"))
            key = f"{topic}_contradiction"
            if key not in missing_keys:
                missing_keys.append(key)

    return uncertainties, missing_keys

# ---------------------------------------------------------------------------
# Extended unclear checks
# ---------------------------------------------------------------------------

def _check_medication_details_unclear(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if trial requires specific drug details but patient data is unclear."""
    inclusion_text = _text(trial.get("inclusion_criteria", []))
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    trial_text = inclusion_text + " " + exclusion_text

    if not _any_match(_TRIAL_MED_SPECIFIC_PATTERNS, trial_text):
        return None, None

    patient_med_text = _text(
        patient.get("medications", [])
        + patient.get("key_features", [])
        + [patient.get("summary", "")]
    )

    if _any_match(_PATIENT_UNCLEAR_MED_PATTERNS, patient_med_text):
        return (
            "trial requires specific medication details but patient medication data is unclear or missing",
            "medication details unclear or missing",
        )

    return None, None


def _check_disease_stage_unclear(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if trial requires stage/severity info but patient data is unclear."""
    inclusion_text = _text(trial.get("inclusion_criteria", []))
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    trial_text = inclusion_text + " " + exclusion_text

    if not _any_match(_TRIAL_STAGE_SEVERITY_PATTERNS, trial_text):
        return None, None

    patient_all_text = _text(
        patient.get("key_features", [])
        + [patient.get("summary", "")]
        + [str(patient.get("disease_stage", ""))]
        + [str(patient.get("disease_duration", ""))]
    )

    # FoG exemption: if the trial requires FoG and the patient documents FoG,
    # the only matched stage-severity pattern is FoG itself — don't flag as unclear.
    _FOG_PATTERNS = [r"freezing.*gait", r"\bfog\b"]
    _NON_FOG_STAGE_PATTERNS = [
        p for p in _TRIAL_STAGE_SEVERITY_PATTERNS
        if p not in (r"freezing of gait", r"\bfog\b", r"\bfog\s")
    ]
    if (
        _any_match(_FOG_PATTERNS, trial_text)
        and not _any_match(_NON_FOG_STAGE_PATTERNS, trial_text)
        and _any_match(_FOG_PATTERNS, patient_all_text)
    ):
        return None, None

    if _any_match(_PATIENT_UNCLEAR_STAGE_PATTERNS, patient_all_text):
        return (
            "trial requires disease stage or severity information but patient data is unclear or missing",
            "disease stage, severity, or duration unclear or missing",
        )

    # Also check if disease_stage field is explicitly "unclear" or None (key present but no data)
    if "disease_stage" in patient:
        _raw_stage = patient.get("disease_stage")
        disease_stage = "" if _raw_stage is None else str(_raw_stage).lower()
        if disease_stage in ("unclear", "unknown", "missing", "not recorded", "none", ""):
            # Only flag if stage/severity info is genuinely relevant to the trial
            return (
                "trial requires disease stage or severity information but patient data is unclear or missing",
                "disease stage unclear or not recorded",
            )

    return None, None


def _check_atypical_parkinsonism(
    patient: dict, trial: dict
) -> tuple[str | None, str | None, str | None]:
    """Return (status, uncertain_criterion, blocking_criterion) for atypical/unclear parkinsonism.

    status: 'not_eligible' | 'unclear' | None
    """
    patient_diagnosis_text = _text(patient.get("diagnosis", []))

    if not _any_match(_ATYPICAL_PARKINSON_PATTERNS, patient_diagnosis_text):
        return None, None, None

    inclusion_text = _text(trial.get("inclusion_criteria", []))
    exclusion_text = _text(trial.get("exclusion_criteria", []))

    _EXPLICIT_ATYPICAL_EXCLUSION_PATTERNS = [
        r"atypical.*parkinsonism", r"parkinsonism.*atypical",
        r"secondary.*parkinsonism", r"parkinsonism.*secondary",
        r"non.idiopathic", r"vascular.*parkinsonism", r"drug.induced.*parkinsonism",
        r"multiple system atrophy", r"\bmsa\b", r"progressive supranuclear",
        r"\bpsp\b", r"corticobasal", r"\bcbd\b", r"dementia with lewy", r"\bdlb\b",
        r"parkinson.*plus",
    ]

    _DIAGNOSTIC_STUDY_PATTERNS = [
        r"diagnostic\s+(?:imaging|study|trial|validation)",
        r"imaging\s+diagnosis",
        r"differential\s+diagnosis",
        r"differential\s+parkinsonism",
        r"pd\s+vs\.?\s+essential\s+tremor",
        r"parkinson(?:'s)?\s+disease\s+vs\.?\s+essential\s+tremor",
        r"essential\s+tremor\s+vs\.?\s+(?:pd|parkinson)",
        r"biomarker\s+(?:diagnosis|diagnostic)",
        r"diagnostic\s+biomarker",
        r"suspected\s+parkinsonism",
        r"prodromal",
        r"early\s+diagnostic",
    ]

    _HARD_DIAGNOSTIC_PATTERNS = [
        r"differential\s+diagnosis",
        r"differential\s+parkinsonism",
        r"pd\s+vs\.?\s+essential\s+tremor",
        r"parkinson(?:'s)?\s+disease\s+vs\.?\s+essential\s+tremor",
        r"essential\s+tremor\s+vs\.?\s+(?:pd|parkinson)",
    ]

    _TREATMENT_INTERVENTION_PATTERNS = [
        r"neuroprotection", r"neuroprotective", r"disease.modifying",
        r"\btreatment\b", r"\btherapy\b",
        r"\bintervention\b", r"\bstimulation\b", r"\brehabilitation\b",
        r"\bexercise\b", r"\btraining\b", r"\bsurgery\b", r"\bdbs\b",
        r"deep brain stimulation", r"\btreadmill\b",
        r"randomized", r"randomised", r"placebo",
        r"double.blind",
    ]

    _TRIAL_SCOPE_META_FIELDS = [
        "title", "brief_title", "official_title",
        "summary", "brief_summary", "description", "detailed_description",
        "intervention", "intervention_name", "intervention_type", "interventions",
        "keywords", "conditions",
    ]
    _scope_parts = [inclusion_text, exclusion_text]
    for _f in _TRIAL_SCOPE_META_FIELDS:
        _v = trial.get(_f)
        if _v:
            _scope_parts.append(_text(_v))
    trial_full = " ".join(_scope_parts)

    _BROAD_PD_COHORT_PATTERNS = [
        r"scale.*validation", r"validation.*scale", r"questionnaire.*validation",
        r"non.motor.*symptom", r"non.motor.*pd", r"quality.*of.*life",
        r"\bqol\b", r"pd.*phenotype", r"parkinson.*phenotype",
        r"biomarker.*cohort", r"imaging.*cohort", r"observational.*cohort",
        r"\bobservational\b", r"\bregistry\b", r"natural.*history",
        r"cross.sectional", r"longitudinal.*cohort",
    ]

    if _any_match(_IDIOPATHIC_PD_REQUIRED_PATTERNS, inclusion_text):
        if _any_match(_EXPLICIT_ATYPICAL_EXCLUSION_PATTERNS, exclusion_text):
            return (
                "not_eligible",
                None,
                "trial requires idiopathic Parkinson disease; patient has atypical or unclear parkinsonism",
            )
        is_treatment = _any_match(_TREATMENT_INTERVENTION_PATTERNS, trial_full)
        is_diagnostic = _any_match(_DIAGNOSTIC_STUDY_PATTERNS, trial_full)
        is_hard_diagnostic = _any_match(_HARD_DIAGNOSTIC_PATTERNS, trial_full)
        is_broad_cohort = _any_match(_BROAD_PD_COHORT_PATTERNS, trial_full)

        if is_diagnostic and not is_treatment and not is_hard_diagnostic:
            return (
                "unclear",
                "patient has atypical or unclear parkinsonism; trial requires idiopathic Parkinson disease but appears to be a diagnostic/differential study",
                None,
            )
        if is_hard_diagnostic and not is_treatment:
            return (
                "unclear",
                "patient has atypical or unclear parkinsonism; trial requires idiopathic Parkinson disease but appears to be a diagnostic/differential study",
                None,
            )
        if is_broad_cohort and not is_treatment:
            return (
                "unclear",
                "patient has atypical or unclear parkinsonism; trial requires idiopathic Parkinson disease but appears to be a broad cohort/observational/scale-validation study",
                None,
            )
        if is_treatment:
            return (
                "not_eligible",
                None,
                "trial requires idiopathic Parkinson disease for treatment/intervention; patient has atypical or unclear parkinsonism",
            )
        return (
            "not_eligible",
            None,
            "trial requires idiopathic Parkinson disease; patient has atypical or unclear parkinsonism",
        )

    if _any_match(_PARKINSON_PATTERNS, inclusion_text):
        return (
            "unclear",
            "patient has atypical or unclear parkinsonism; trial may require confirmed idiopathic Parkinson disease",
            None,
        )

    return None, None, None


def _check_active_cancer(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if patient has active cancer treatment and trial is non-oncology."""
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + [patient.get("diagnosis", "")]
    )

    if not _any_match(_ACTIVE_CANCER_PATTERNS, patient_all_text):
        return None, None

    # Negation: patient explicitly denies active cancer
    if is_negated(patient_all_text, "active_cancer") and not has_contradiction(patient_all_text, "active_cancer"):
        return None, None

    # Check if trial itself is oncology-focused (then cancer is expected and not a red flag)
    inclusion_text = _text(trial.get("inclusion_criteria", []))
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    trial_text = inclusion_text + " " + exclusion_text

    oncology_patterns = [r"oncology", r"cancer.*trial", r"tumor.*trial", r"chemotherapy.*eligible"]
    if _any_match(oncology_patterns, trial_text):
        return None, None

    # Check if cancer is explicitly excluded (then existing blocking rule handles it)
    cancer_exclusion_patterns = [r"no.*active.*cancer", r"cancer.*exclusion", r"malignancy.*exclusion"]
    if _any_match(cancer_exclusion_patterns, exclusion_text):
        return None, None

    # If safety-sensitive criteria are present, flag as unclear
    if _any_match(_TRIAL_SAFETY_SENSITIVE_PATTERNS, trial_text):
        return (
            "patient has active cancer treatment; eligibility for non-oncology trial with safety-sensitive criteria is unclear",
            "active cancer treatment noted",
        )

    return None, None


def _check_active_cancer_hard_block(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block (not_eligible) when active cancer + clearly invasive/surgical/implant/procedure-based trial.

    Non-invasive gait/rehab/neuroprotection remains unclear via _check_active_cancer.
    Returns (blocking_criterion, matched_fact) or (None, None).
    """
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + [patient.get("diagnosis", "")]
    )

    if not _any_match(_ACTIVE_CANCER_PATTERNS, patient_all_text):
        return None, None

    # Gather full trial text
    _TRIAL_META_FIELDS = [
        "title", "brief_title", "official_title", "summary", "brief_summary",
        "description", "detailed_description", "intervention", "intervention_name",
        "intervention_type", "interventions", "keywords", "conditions",
    ]
    collected: list[str] = []
    for f in ["inclusion_criteria", "exclusion_criteria"]:
        v = trial.get(f, [])
        if isinstance(v, list):
            collected.extend(v)
        elif v:
            collected.append(str(v))
    for f in _TRIAL_META_FIELDS:
        v = trial.get(f, "")
        if v:
            collected.append(str(v))
    all_trial_text = _text(collected)

    # Skip oncology trials
    oncology_patterns = [r"oncology", r"cancer.*trial", r"tumor.*trial", r"chemotherapy.*eligible"]
    if _any_match(oncology_patterns, all_trial_text):
        return None, None

    # Only hard-block for clearly invasive/surgical/implant/procedure-based trials
    _INVASIVE_TRIAL_PATTERNS = [
        r"\bsurgery\b",
        r"\bsurgical\b",
        r"\bimplant\b",
        r"\bdbs\b",
        r"deep brain stimulation",
        r"device.*implant",
        r"implant.*device",
        r"neurosurgical",
        r"stereotactic",
        r"intracranial",
        r"lumbar.*puncture",
        r"spinal.*cord.*stimulation",
        r"\bstenting\b",
        r"\bcatheter\b",
        r"infusion.*pump",
        r"subcutaneous.*pump",
        r"\blcig\b",
        r"intestinal.*gel.*infusion",
        r"levodopa.*infusion.*pump",
    ]
    if _any_match(_INVASIVE_TRIAL_PATTERNS, all_trial_text):
        return (
            "active cancer treatment: patient has active cancer which is incompatible with invasive/surgical/implant-based trial",
            "active cancer treatment present; invasive/surgical trial",
        )

    return None, None


def _check_recent_trial_participation(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if patient has recent trial participation."""
    trial_text = _text(
        trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", [])
    )

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )

    if not _any_match(_RECENT_TRIAL_PATTERNS, patient_all_text):
        return None, None

    # Existing path: trial explicitly states washout/prior-study requirements
    if _any_match(_TRIAL_WASHOUT_PATTERNS, trial_text):
        return (
            "trial has washout or prior study requirements; patient has recent or concurrent trial participation",
            "recent or concurrent trial participation noted",
        )

    # Patient-side-only path: patient documents recent/concurrent participation
    # but trial has no explicit washout language.
    # Suppress only for explicitly observational / registry / scale-validation trials.
    _OBSERVATIONAL_TRIAL_PATTERNS = [
        r"\bobservational\b", r"\bregistry\b", r"natural history",
        r"\bsurvey\b", r"\bquestionnaire\b", r"scale validation",
        r"validation study", r"non.interventional",
    ]
    if _any_match(_OBSERVATIONAL_TRIAL_PATTERNS, trial_text):
        return None, None

    return (
        "recent or concurrent interventional trial participation noted; washout or overlap eligibility cannot be confirmed",
        "recent or concurrent trial participation noted",
    )


def _check_comorbidity_protocol_risk(
    patient: dict, trial: dict
) -> tuple[str | None, str | None, str | None]:
    """Return (blocking_criterion, uncertain_criterion, matched_fact) for comorbidity/protocol risk.

    Escalates to blocking when a hard safety contraindication applies.
    Suppresses the uncertain signal when the comorbidity is the trial's target population.
    Falls back to uncertain for genuinely ambiguous cases.
    """
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + [patient.get("diagnosis", "")]
    )

    if not _any_match(_PATIENT_COMPLEX_COMORBIDITY_PATTERNS, patient_all_text):
        return None, None, None

    inclusion_text = _text(trial.get("inclusion_criteria", []))
    _TRIAL_META_FIELDS = [
        "title", "brief_title", "official_title", "summary", "brief_summary",
        "description", "detailed_description", "intervention", "intervention_name",
        "intervention_type", "interventions", "keywords", "conditions",
    ]
    meta_parts = []
    for field in _TRIAL_META_FIELDS:
        val = trial.get(field)
        if val is not None:
            meta_parts.append(_text(val))
    trial_text = inclusion_text + " " + _text(trial.get("exclusion_criteria", [])) + " " + " ".join(meta_parts)

    if not _any_match(_TRIAL_COMPLEX_FOCUS_PATTERNS, trial_text):
        return None, None, None

    # Hard safety contraindication → blocking
    for patient_patterns, trial_patterns in _HARD_CONTRAINDICATION_PAIRS:
        if _any_match(patient_patterns, patient_all_text) and _any_match(trial_patterns, trial_text):
            return (
                "hard safety contraindication: implanted cardiac device is incompatible with transcranial stimulation",
                None,
                "implanted cardiac device present; transcranial stimulation trial",
            )

    # Target-population exemption → suppress uncertain
    for patient_patterns, trial_inclusion_patterns in _COMORBIDITY_TARGET_PAIRS:
        if _any_match(patient_patterns, patient_all_text) and _any_match(trial_inclusion_patterns, trial_text):
            return None, None, None

    # Cognitive/MCI scope guard: if the only comorbidity trigger is mild cognitive uncertainty,
    # only flag when the trial has explicit cognitive, neuropsychological, or compliance requirements.
    _COGNITIVE_ONLY_PATTERNS = [r"cognitive.*impairment", r"mild.*cognitive", r"\bmci\b"]
    _NON_COGNITIVE_COMORBIDITY_PATTERNS = [
        p for p in _PATIENT_COMPLEX_COMORBIDITY_PATTERNS
        if p not in (r"cognitive.*impairment", r"mild.*cognitive", r"\bmci\b")
    ]
    _COGNITIVE_TRIAL_REQUIREMENT_PATTERNS = [
        r"cognitive.*assessment", r"cognitive.*trial", r"cognitive.*study",
        r"neuropsychological", r"protocol.*compliance", r"compliance.*protocol",
        r"adherence", r"cognitive.*task", r"informed consent capacity",
    ]
    if (
        _any_match(_COGNITIVE_ONLY_PATTERNS, patient_all_text)
        and not _any_match(_NON_COGNITIVE_COMORBIDITY_PATTERNS, patient_all_text)
        and not _any_match(_COGNITIVE_TRIAL_REQUIREMENT_PATTERNS, trial_text)
    ):
        return None, None, None

    # Genuine ambiguity → uncertain
    return (
        None,
        "patient has comorbidity or condition that may affect protocol compliance or safety in this trial type",
        "complex comorbidity noted in context of device/stimulation/imaging/rehabilitation/cognitive/gait-focused trial",
    )


def _check_parent_study_required(patient: dict, trial: dict) -> tuple[str | None, str | None, str | None]:
    """Check if trial requires prior parent/extension participation.

    Returns (status, uncertain_criterion, blocking_criterion).
    status: 'not_eligible' | 'unclear' | None
    """
    # Patterns that look like prior-participation language but are actually exclusions/washout — skip these.
    _EXCLUSION_LIKE_PATTERNS = [
        r"no concurrent",
        r"not.*(?:enrolled|participating|enrolled).*(?:another|other)",
        r"concurrent.*(?:trial|study).*(?:exclusion|prohibited|not permitted)",
        r"(?:exclusion|excluded).*(?:concurrent|prior|previous).*(?:trial|study)",
        r"washout",
        r"not currently enrolled",
        r"must not.*(?:enrolled|participat)",
        # "prior to" as a timing phrase (before study/enrollment) — NOT prior study participation
        r"prior\s+to\s+(?:study\s+)?(?:participation|enrollment|enrolment|entry|screening)",
        r"before\s+(?:study\s+)?(?:participation|enrollment|enrolment|entry|screening)",
        r"(?:medical|physician|clinician|doctor).*clearance.*(?:prior\s+to|before).*(?:study|participation|enrollment)",
        r"(?:prior\s+to|before).*(?:study\s+)?(?:participation|enrollment).*(?:clearance|approval|consent)",
    ]
    inclusion_list = trial.get("inclusion_criteria", [])
    has_requirement = any(
        _any_match(_TRIAL_PARENT_STUDY_REQUIRED_PATTERNS, c.lower())
        and not _any_match(_EXCLUSION_LIKE_PATTERNS, c.lower())
        for c in inclusion_list
    )
    if not has_requirement:
        return None, None, None

    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    if _any_match(_PATIENT_PRIOR_STUDY_PATTERNS, patient_text):
        return None, None, None

    # Check for ambiguity signals — if present, downgrade to unclear instead of hard block
    patient_med_text = _text(
        patient.get("medications", [])
        + patient.get("key_features", [])
        + [patient.get("summary", "")]
    )
    patient_all_text = patient_text + " " + patient_med_text + " " + _text(
        (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list) else [patient.get("diagnosis", "")])
        + patient.get("exclusions", [])
    )

    _AMBIGUITY_SIGNALS = [
        # Unclear medication history
        r"dose.*unclear", r"frequency.*unclear", r"unclear.*dose", r"unclear.*frequency",
        r"no.*pharmacy records", r"medication.*unclear", r"medication.*details.*unavailable",
        r"medication.*details.*unavailable", r"medication.*not.*recorded",
        # Active cancer / major competing safety issue
        r"active.*cancer", r"current.*chemotherapy", r"ongoing.*chemotherapy",
        r"active.*malignancy", r"cancer.*treatment.*ongoing",
        # Recent/concurrent trial participation
        r"recent.*interventional.*trial", r"enrolled.*in.*(?:another|recent).*(?:trial|study)",
        r"currently.*enrolled.*(?:trial|study)", r"concurrent.*(?:trial|study)",
        r"participated.*in.*(?:recent|another).*(?:trial|study)",
        # Advanced PD / LCIG context — ambiguous continuation eligibility
        r"\blcig\b", r"intestinal.*gel", r"levodopa.*intestinal",
        r"advanced.*parkinson", r"advanced.*pd",
        r"continuous.*(?:infusion|delivery).*(?:levodopa|dopamine)",
    ]
    if _any_match(_AMBIGUITY_SIGNALS, patient_all_text):
        return (
            "unclear",
            "trial requires prior parent/extension study participation; patient eligibility cannot be confirmed due to ambiguous context",
            None,
        )

    return (
        "not_eligible",
        None,
        "prior parent/extension study participation required; patient has no documented prior participation",
    )


def _check_oncology_required(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when trial requires advanced/metastatic solid tumor or specific cancer diagnosis
    and patient has no cancer documented."""
    inclusion_list = trial.get("inclusion_criteria", [])

    # Patterns that look like oncology but are actually screening/biopsy/colonoscopy context — skip.
    _ONCOLOGY_EXCLUSION_CONTEXT_PATTERNS = [
        r"colonoscop",
        r"colonic.*biopsy",
        r"biopsy.*colon",
        r"rectosigmoidoscop",
        r"colorectal.*screening",
        r"colorectal.*risk",
        r"at risk.*(?:colorectal|colon|rectal).*cancer",
        r"bowel.*screening",
        r"stool.*sample",
        r"alpha.synuclein.*biopsy",
        r"biopsy.*alpha.synuclein",
        r"tissue.*biopsy",
        r"biopsy.*parkinson",
        r"parkinson.*biopsy",
    ]

    has_requirement = False
    for c in inclusion_list:
        cl = c.lower()
        if _any_match(_ONCOLOGY_EXCLUSION_CONTEXT_PATTERNS, cl):
            continue
        if _any_match(_TRIAL_ONCOLOGY_REQUIRED_PATTERNS, cl):
            has_requirement = True
            break

    if not has_requirement:
        return None, None

    patient_text = _text(
        (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list) else [patient.get("diagnosis", "")])
        + (patient.get("key_features", []) if isinstance(patient.get("key_features"), list) else [patient.get("key_features", "")])
        + (patient.get("medications", []) if isinstance(patient.get("medications"), list) else [patient.get("medications", "")])
        + (patient.get("exclusions", []) if isinstance(patient.get("exclusions"), list) else [patient.get("exclusions", "")])
        + [patient.get("summary", "")]
    )
    if _any_match(_PATIENT_CANCER_PATTERNS, patient_text):
        return None, None  # Cancer documented

    return (
        "oncology diagnosis required: trial requires advanced/metastatic solid tumor or confirmed cancer diagnosis; patient has no documented cancer",
        "no cancer or solid tumor diagnosis documented",
    )


def _check_frailty_high_demand_exercise(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when patient has explicit frailty/fall risk and trial demands high physical exercise.
    FoG/gait impairment/motor dysfunction alone does NOT count as frailty.
    Very elderly patients with frailty/recurrent falls in high-demand exercise protocols → not_eligible.
    """
    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )

    # Only fire for explicit frailty/fall risk — NOT for FoG or gait impairment
    _STRICT_FRAILTY_PATTERNS = [
        r"\bfrail\b",
        r"\bfrailty\b",
        r"recurrent.*falls",
        r"frequent.*falls",
        r"high.*fall.*risk",
        r"wheelchair.*(?:bound|restricted|dependent)",
        r"unable.*to.*walk",
        r"cannot.*walk",
    ]
    _FOG_GAIT_ONLY_PATTERNS = [
        r"freezing.*(?:of\s+)?gait",
        r"\bfog\b",
        r"gait.*(?:impairment|disturbance|dysfunction|disorder)",
        r"motor.*dysfunction",
        r"gait.*freezing",
    ]

    if not _any_match(_STRICT_FRAILTY_PATTERNS, patient_text):
        return None, None

    # If gait/FoG is present but no actual frailty word, do not block
    has_explicit_frailty = _any_match(
        [r"\bfrail\b", r"\bfrailty\b", r"recurrent.*falls", r"frequent.*falls",
         r"high.*fall.*risk", r"wheelchair.*(?:bound|restricted|dependent)",
         r"unable.*to.*walk", r"cannot.*walk"],
        patient_text,
    )
    if not has_explicit_frailty:
        return None, None

    inclusion_text = _text(trial.get("inclusion_criteria", []))
    extra_trial_text = _text([
        trial.get("title", ""),
        trial.get("summary", ""),
        trial.get("description", ""),
    ])
    trial_text = inclusion_text + " " + _text(trial.get("exclusion_criteria", [])) + " " + extra_trial_text

    # Exempt frailty-targeted physiotherapy trials
    _FRAILTY_TARGET_PATTERNS = [
        r"frailty.*trial", r"frailty.*study", r"frail.*patient",
        r"home.*physiotherapy", r"home.*physical.*therapy",
        r"frailty.*intervention", r"frailty.*rehabilitation",
    ]
    if _any_match(_FRAILTY_TARGET_PATTERNS, inclusion_text):
        return None, None

    if _any_match(_TRIAL_HIGH_DEMAND_EXERCISE_PATTERNS, trial_text):
        return (
            "frailty/fall risk incompatible with high-demand treadmill or agility exercise protocol",
            "frailty or recurrent falls documented; high-demand physical exercise trial",
        )

    return None, None


def _trial_requires_advanced_pd(inclusion_text: str) -> bool:
    """Return True if inclusion criteria explicitly require advanced PD or composite severity criteria."""
    _EXPLICIT_ADVANCED = [
        r"advanced\s+(?:parkinson(?:'s)?(?:\s+disease)?|pd)\b",
        r"advanced.stage\s+(?:parkinson|pd)\b",
        r"advanced\s+motor\s+(?:fluctuation|complication)",
        r"advanced\s+disease\s+stage.*parkinson",
        r"parkinson.*advanced\s+disease",
        r"advanced\s+parkinson",
    ]
    if _any_match(_EXPLICIT_ADVANCED, inclusion_text):
        return True

    # Composite: PD required + at least 2 severity sub-criteria
    if not _any_match(_PARKINSON_PATTERNS, inclusion_text):
        return False

    _SEVERITY_SUBCRITERIA = [
        r"hoehn.*yahr.*[>=≥]\s*3",
        r"h&y\s*[>=≥]\s*3",
        r"modified\s+hoehn.*yahr.*[>=≥]\s*3",
        r"(?:mds.)?updrs.*(?:part\s+)?iii?\s*[>=≥]\s*\d+",
        r"updrs.*part.*3\s*[>=≥]",
        r"motor\s+fluctuation",
        r"wearing.off",
        r"\bdyskinesia\b",
        r"off\s+(?:time|period|state)",
        r"off.time",
        r"hours?\s+(?:of\s+)?off",
        r"disease\s+(?:course|duration).*(?:at\s+least\s+)?\d+\s+years?",
        r"\d+\s+years?\s+(?:of\s+)?disease",
        r"advanced\s+motor\s+complication",
        r"levodopa.induced\s+dyskinesia",
    ]
    count = sum(1 for p in _SEVERITY_SUBCRITERIA if re.search(p, inclusion_text))
    return count >= 2


def _check_advanced_pd_required(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when trial requires advanced PD (explicit or composite) but patient has early-onset/early-stage PD."""
    inclusion_list = trial.get("inclusion_criteria", [])
    inclusion_text = _text(inclusion_list)

    if not _trial_requires_advanced_pd(inclusion_text):
        return None, None

    # Do not fire for trials that target early/young-onset PD, bone density, gait cueing, imaging, observational
    _TRIAL_EXEMPT_PATTERNS = [
        r"early.onset\s+(?:parkinson|pd)",
        r"very\s+early\s+(?:parkinson|pd)",
        r"young.onset\s+(?:parkinson|pd)",
        r"early\s+(?:parkinson|pd)\b",
        r"bone\s+(?:density|mineral)",
        r"\bbmd\b",
        r"gait\s+cue",
        r"auditory\s+cue",
        r"\bobservational\b",
        r"\bregistry\b",
        r"natural\s+history",
        r"neuroimaging",
        r"\bpet\b",
        r"\bfmri\b",
        r"\bmri\s+imaging\b",
    ]
    if _any_match(_TRIAL_EXEMPT_PATTERNS, inclusion_text):
        return None, None

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list) else [patient.get("diagnosis", "")])
    )

    _PATIENT_EARLY_PD_PATTERNS = [
        r"early.onset\s+(?:parkinson|pd)",
        r"parkinson.*early.onset",
        r"early.onset.*parkinson",
        r"very\s+early\s+(?:parkinson|pd)",
        r"young.onset\s+(?:parkinson|pd)",
        r"early.stage\s+(?:parkinson|pd)",
        r"(?:parkinson|pd).*early.stage",
    ]
    if not _any_match(_PATIENT_EARLY_PD_PATTERNS, patient_all_text):
        return None, None

    # Do not block if patient already has advanced disease markers
    _PATIENT_ADVANCED_MARKERS = [
        r"advanced\s+(?:parkinson|pd)",
        r"motor\s+fluctuation",
        r"wearing.off",
        r"\blcig\b",
        r"intestinal\s+gel",
        r"\bdbs\b",
        r"deep\s+brain\s+stimulation",
        r"hoehn\s*(?:and|&)?\s*yahr\s*(?:stage)?\s*(?:3|4|5)\b",
        r"h\s*[&\-]?\s*y\s*(?:stage)?\s*(?:3|4|5)\b",
        r"\bhy\s*(?:stage)?\s*(?:3|4|5)\b",
        r"severe\s+motor",
        r"\bdyskinesia\b",
        r"off\s+(?:time|period|state)",
        r"off.time",
        r"updrs\s*(?:iii|3|part\s*(?:iii|3))\s*(?:score\s*)?(?:of\s*)?\d{2,}",
    ]
    if _any_match(_PATIENT_ADVANCED_MARKERS, patient_all_text):
        return None, None

    return (
        "advanced Parkinson disease required: trial requires advanced PD or composite severity criteria; patient has early-onset/early-stage PD without advanced disease markers",
        "early-onset or early-stage PD documented; no advanced disease markers present",
    )


def _check_advanced_pd_requirement(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when trial requires Parkinson disease plus composite advanced/severe PD signals
    but patient is clearly early-onset/early-stage PD without advanced disease evidence."""

    def _norm(text: str) -> str:
        text = text.lower()
        text = text.replace("\\&", "&")
        text = text.replace("\u2265", ">=")
        text = text.replace("\u2264", "<=")
        return text

    inclusion_list = trial.get("inclusion_criteria", [])
    norm_text = _norm(" ".join(str(c) for c in inclusion_list))

    # Must require Parkinson disease
    if not _any_match(_PARKINSON_PATTERNS, norm_text):
        return None, None

    # Explicit advanced PD language
    _EXPLICIT_ADVANCED = [
        r"advanced\s+(?:parkinson(?:'s)?(?:\s+disease)?|pd)\b",
        r"advanced.stage\s+(?:parkinson|pd)\b",
        r"advanced\s+motor\s+(?:fluctuation|complication)",
        r"advanced\s+parkinson",
    ]
    explicit = _any_match(_EXPLICIT_ADVANCED, norm_text)

    # Count composite severity signals (deduplicated by signal group)
    _SEVERITY_SIGNAL_GROUPS = [
        # Disease course/duration >= 5 years
        [
            r"course\s+of\s+disease\s+for\s+at\s+least\s+\d+\s+years?",
            r"disease\s+course\s+(?:for|of)\s+(?:at\s+least\s+)?\d+\s+years?",
            r"disease\s+duration\s+(?:for|of|at)\s+(?:at\s+least\s+)?\d+\s+years?",
            r"disease\s+(?:course|duration).*(?:at\s+least\s+)?\d+\s+years?",
        ],
        # H&Y >= 3
        [
            r"(?:modified\s+)?hoehn\s*(?:and|&)\s*yahr\s+stage\s*>=\s*3",
            r"(?:modified\s+)?hoehn\s*(?:and|&)\s*yahr\s*>=\s*3",
            r"hoehn.*yahr.*stage\s*>=\s*3",
            r"h\s*&\s*y\s+stage\s*>=\s*3",
            r"h\s*&\s*y\s*>=\s*3",
        ],
        # UPDRS III >= threshold
        [
            r"(?:mds[.\s-])?updrs\s*(?:part\s*)?iii\s*>=\s*\d+",
            r"(?:mds[.\s-])?updrs\s*(?:part\s*)?3\s*>=\s*\d+",
            r"updrs\s*iii\s*>=\s*\d+",
        ],
        # Off period / off time
        [
            r"\boff\s+period\b",
            r"\boff\s+time\b",
            r"\boff.time\b",
            r"3\s*-\s*h(?:our)?\s+off",
            r"3\s+hour\s+off",
            r"\d+\s*-?\s*h(?:ours?)?\s+off\s+time",
        ],
        # Motor fluctuations
        [
            r"fluctuation\s+of\s+motor",
            r"motor\s+fluctuation",
        ],
        # Wearing off
        [
            r"wearing.off",
        ],
    ]
    severity_count = sum(
        1 for group in _SEVERITY_SIGNAL_GROUPS
        if any(re.search(p, norm_text, re.IGNORECASE) for p in group)
    )

    if not explicit and severity_count < 2:
        return None, None

    # Do not fire for trials explicitly targeting early/young-onset PD or exempt study types
    _TRIAL_EXEMPT = [
        r"early.onset\s+(?:parkinson|pd)",
        r"very\s+early\s+(?:parkinson|pd)",
        r"young.onset\s+(?:parkinson|pd)",
        r"early\s+(?:parkinson|pd)\b",
        r"bone\s+(?:density|mineral)",
        r"\bbmd\b",
        r"gait\s+cue",
        r"auditory\s+cue",
        r"\bobservational\b",
        r"\bregistry\b",
        r"natural\s+history",
        r"neuroimaging",
    ]
    if _any_match(_TRIAL_EXEMPT, norm_text):
        return None, None

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list)
           else [patient.get("diagnosis", "")])
    )

    _PATIENT_EARLY_PD = [
        r"early.onset\s+(?:parkinson|pd)",
        r"early.onset.*parkinson",
        r"parkinson.*early.onset",
        r"very\s+early\s+(?:parkinson|pd)",
        r"young.onset\s+(?:parkinson|pd)",
        r"early.stage\s+(?:parkinson|pd)",
        r"(?:parkinson|pd).*early.stage",
        r"hoehn\s+and\s+yahr\s+stage\s+1\b",
        r"\bh\s*&\s*y\s+stage\s+1\b",
        r"\bhy\s+stage\s+1\b",
    ]
    if not _any_match(_PATIENT_EARLY_PD, patient_all_text):
        return None, None

    # Do not block if patient already has advanced disease evidence (bounded stage parsing only)
    _PATIENT_ADVANCED = [
        r"advanced\s+(?:parkinson|pd)",
        r"advanced\s+parkinson",
        r"motor\s+fluctuation",
        r"\boff\s+time\b",
        r"\boff.time\b",
        r"\bdyskinesia\b",
        r"\blcig\b",
        r"intestinal\s+gel",
        r"\bdbs\b",
        r"deep\s+brain\s+stimulation",
        r"severe\s+motor",
        r"updrs\s*(?:iii|3|part\s*(?:iii|3))\s*(?:score\s*)?(?:of\s*)?\d{2,}",
        r"hoehn\s+(?:and|&)\s+yahr\s+stage\s+[345]\b",
        r"\bh\s*&\s*y\s+stage\s+[345]\b",
        r"\bhy\s+stage\s+[345]\b",
        r"hoehn\s+and\s+yahr\s+[345]\b",
    ]
    if _any_match(_PATIENT_ADVANCED, patient_all_text):
        return None, None

    return (
        "advanced/severe Parkinson disease required: trial requires composite advanced PD severity criteria; patient has early-onset/early-stage PD without advanced disease evidence",
        "early-onset or early-stage PD documented; no advanced disease markers present",
    )


# ---------------------------------------------------------------------------
# Missing specific inclusion details (uncertainty only)
# ---------------------------------------------------------------------------



def _check_missing_specific_inclusion_details(
    patient: dict, trial: dict
) -> list[str]:
    """Return uncertain criteria when trial requires specific details absent from patient profile."""
    uncertain: list[str] = []

    inclusion_text = _text(trial.get("inclusion_criteria", []))
    title_text = _text(trial.get("title", "") or "")
    official_title_text = _text(trial.get("official_title", "") or "")
    trial_full = inclusion_text + " " + title_text + " " + official_title_text

    patient_all = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "") or ""]
        + [patient.get("diagnosis", "") or ""]
    )

    _BROAD_PD_TRIAL_SUPPRESS = [
        r"scale.*validation", r"validation.*scale", r"questionnaire.*validation",
        r"non.motor.*symptom", r"non.motor.*pd", r"quality.*of.*life",
        r"\bqol\b", r"pd.*phenotype", r"parkinson.*phenotype",
        r"\bobservational\b", r"\bregistry\b", r"natural.*history",
        r"cross.sectional", r"longitudinal.*cohort",
        r"broad.*pd", r"all.*(?:stage|severity)", r"across.*stage",
    ]

    # 1. FoG/gait-specific requirement
    if _any_match(_FOG_GAIT_TRIAL_PATTERNS, inclusion_text):
        if not _any_match(_FOG_GAIT_PATIENT_PATTERNS, patient_all):
            uncertain.append(
                "trial requires specific gait/FoG/balance features not documented in patient profile"
            )

    # 2. Cognitive/MCI requirement
    if _any_match(_COG_MCI_TRIAL_PATTERNS, inclusion_text):
        if not _any_match(_COG_MCI_PATIENT_PATTERNS, patient_all):
            uncertain.append(
                "trial requires cognitive/MCI status not documented in patient profile"
            )

    # 3. Disease severity/stage/duration requirement
    # Suppress for broad PD cohort / scale validation / non-motor phenotype studies
    if _any_match(_SEVERITY_TRIAL_PATTERNS, inclusion_text):
        if not _any_match(_SEVERITY_PATIENT_PATTERNS, patient_all):
            if not _any_match(_BROAD_PD_TRIAL_SUPPRESS, trial_full):
                uncertain.append(
                    "trial requires disease severity/stage/duration not documented in patient profile"
                )

    # 4. Medication-specific requirement
    # Suppress for broad PD cohort / observational studies where med history is background info
    if _any_match(_MED_SPECIFIC_TRIAL_PATTERNS, inclusion_text):
        patient_meds_empty = not patient.get("medications")
        if patient_meds_empty or not _any_match(_MED_DOCUMENTED_PATIENT_PATTERNS, patient_all):
            if not _any_match(_BROAD_PD_TRIAL_SUPPRESS, trial_full):
                uncertain.append(
                    "trial requires specific medication details not documented in patient profile"
                )

    # 5. Language/scale-validation requirement
    if _any_match(_LANG_SCALE_TRIAL_PATTERNS, trial_full):
        if not _any_match(_LANG_PATIENT_PATTERNS, patient_all):
            uncertain.append(
                "trial appears to be a language-specific or scale-validation study; patient language ability not documented"
            )

    return uncertain


# ---------------------------------------------------------------------------
# Non-motor / safety comorbidity uncertainty helper
# ---------------------------------------------------------------------------



def _check_nonmotor_comorbidity_uncertainty(
    patient: dict, trial: dict, existing_uncertain: list[str] | None = None
) -> list[str]:
    """Return uncertain criteria for non-motor/safety comorbidity + relevant trial focus.

    Only adds uncertainty; never blocks. Only runs when blocking_criteria is empty.
    """
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "") or ""]
        + (patient.get("diagnosis", []) if isinstance(patient.get("diagnosis"), list)
           else [patient.get("diagnosis", "") or ""])
    )

    _TRIAL_SCOPE_FIELDS = [
        "inclusion_criteria", "exclusion_criteria",
        "title", "brief_title", "official_title",
        "summary", "brief_summary", "description", "detailed_description",
        "intervention", "intervention_name", "intervention_type", "interventions",
        "keywords", "conditions",
    ]
    scope_parts: list[str] = []
    for f in _TRIAL_SCOPE_FIELDS:
        v = trial.get(f)
        if v:
            scope_parts.append(_text(v))
    trial_scope_text = " ".join(scope_parts)

    uncertain: list[str] = []

    # RBD: only trigger on explicit protocol/exclusion ambiguity, suppressed for target-population
    _RBD_PATIENT = [r"\brbd\b", r"rem.*sleep.*behavior", r"rem.*behavior.*disorder"]
    if _any_match(_RBD_PATIENT, patient_all_text):
        if (
            _any_match(_RBD_AMBIGUITY_TRIGGER_PATTERNS, trial_scope_text)
            and not _any_match(_RBD_TARGET_SUPPRESSION_PATTERNS, trial_scope_text)
        ):
            msg = "patient has REM sleep behavior disorder; protocol/exclusion criteria may affect eligibility"
            if msg not in uncertain:
                uncertain.append(msg)

    # Autonomic dysfunction / orthostatic hypotension
    _AUTONOMIC_PATIENT = [
        r"autonomic.*dysfunction", r"autonomic.*failure",
        r"orthostatic.*hypotension", r"postural.*hypotension",
    ]
    _AUTONOMIC_TRIAL = [
        r"rehabilitation", r"physiotherapy", r"physical.*therapy",
        r"exercise", r"gait", r"balance", r"home.*therap",
        r"fall.*risk", r"fall.*prevention",
    ]
    _AUTONOMIC_NON_MOTOR_SUPPRESS = [
        r"non.motor.*symptom", r"non.motor.*pd", r"pd.*phenotype",
        r"parkinson.*phenotype", r"quality.*of.*life", r"\bqol\b",
        r"\bobservational\b", r"\bregistry\b", r"natural.*history",
        r"autonomic.*study", r"autonomic.*trial",
    ]
    if _any_match(_AUTONOMIC_PATIENT, patient_all_text) and _any_match(_AUTONOMIC_TRIAL, trial_scope_text):
        if not _any_match(_AUTONOMIC_NON_MOTOR_SUPPRESS, trial_scope_text):
            msg = "patient has autonomic dysfunction/orthostatic hypotension; eligibility for rehabilitation/gait/exercise trial is uncertain"
            if msg not in uncertain:
                uncertain.append(msg)

    # Depression: only imaging/PET/biomarker studies
    _DEPRESSION_PATIENT = [r"\bdepression\b", r"\bdepressed\b"]
    if _any_match(_DEPRESSION_PATIENT, patient_all_text) and _any_match(_DEPRESSION_IMAGING_BIOMARKER_PATTERNS, trial_scope_text):
        msg = "patient has depression; depression may confound imaging/biomarker outcomes"
        if msg not in uncertain:
            uncertain.append(msg)

    # Frailty / recurrent falls: mindfulness/adherence only, suppressed for frailty-targeted trials
    _FRAILTY_PATIENT = [r"\bfrail\b", r"\bfrailty\b", r"recurrent.*falls", r"frequent.*falls"]
    _FRAILTY_AMBIGUITY_TRIAL = [
        r"mindfulness", r"meditation", r"adherence", r"protocol.*adherence",
        r"sustained.*participation", r"cognitive.*engagement",
        r"sustained.*engagement", r"home.*based.*program",
    ]
    if _any_match(_FRAILTY_PATIENT, patient_all_text):
        if (
            _any_match(_FRAILTY_AMBIGUITY_TRIAL, trial_scope_text)
            and not _any_match(_FRAILTY_TARGET_SUPPRESSION_PATTERNS, trial_scope_text)
        ):
            msg = "patient has frailty/recurrent falls; eligibility for mindfulness/adherence/sustained-participation trial is uncertain"
            if msg not in uncertain:
                uncertain.append(msg)

    # Active cancer: only when _check_active_cancer hasn't already flagged it
    already_has_cancer = existing_uncertain is not None and any(
        "cancer" in c for c in existing_uncertain
    )
    if not already_has_cancer:
        if _any_match(_ACTIVE_CANCER_PATIENT_PATTERNS, patient_all_text) and _any_match(_ACTIVE_CANCER_TRIAL_PATTERNS, trial_scope_text):
            msg = "patient has active cancer treatment; eligibility for gait/neuroprotection/safety-sensitive intervention trial is uncertain"
            if msg not in uncertain:
                uncertain.append(msg)

    return uncertain


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
    if cog_min_block and cog_min_block not in blocking_criteria:
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
    patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
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

    return {
        "prediction": prediction,
        "confidence": confidence,
        "matched_facts": matched_facts,
        "blocking_criteria": blocking_criteria,
        "uncertain_criteria": uncertain_criteria,
        "explanation": explanation,
        "missing_information": missing_information,
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

    return CriterionDecision.unknown, "cannot evaluate from available data"


def _evaluate_exclusion_criterion(
    c_lower: str, patient: dict, blocking: list[str], uncertain: list[str]
) -> tuple[CriterionDecision, str]:
    """Return (decision, reason) for a single exclusion criterion.

    For exclusions: met = criterion applies (patient IS excluded), not_met = criterion does not apply.
    """
    # DBS
    if _any_match(_DBS_PATTERNS, c_lower) or _trial_involves_procedure(c_lower, "dbs"):
        patient_text = _text(
            patient.get("key_features", [])
            + patient.get("medications", [])
            + patient.get("exclusions", [])
        )
        if _has_negated_dbs(patient_text):
            return CriterionDecision.not_met, "no DBS history documented"
        if any("dbs" in b or "deep brain" in b for b in blocking):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        if _patient_has_procedure(patient_text, "dbs"):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        return CriterionDecision.not_met, "no DBS implant found"

    # MAO-B inhibitor
    if _MAOB_CRITERION_PATTERN.search(c_lower):
        patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
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
