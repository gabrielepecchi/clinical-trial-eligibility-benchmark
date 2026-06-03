"""Generic unclear / missing-information eligibility helpers."""

from app.eligibility.clinical_terms import (
    _any_match,
    has_contradiction,
    _TRIAL_STAGE_SEVERITY_PATTERNS,
    _PATIENT_UNCLEAR_STAGE_PATTERNS,
    _RECENT_TRIAL_PATTERNS,
    _TRIAL_WASHOUT_PATTERNS,
    _TRIAL_PARENT_STUDY_REQUIRED_PATTERNS,
    _PATIENT_PRIOR_STUDY_PATTERNS,
    _FOG_GAIT_TRIAL_PATTERNS,
    _FOG_GAIT_PATIENT_PATTERNS,
    _COG_MCI_TRIAL_PATTERNS,
    _COG_MCI_PATIENT_PATTERNS,
    _SEVERITY_TRIAL_PATTERNS,
    _SEVERITY_PATIENT_PATTERNS,
    _MED_SPECIFIC_TRIAL_PATTERNS,
    _MED_DOCUMENTED_PATIENT_PATTERNS,
    _LANG_SCALE_TRIAL_PATTERNS,
    _LANG_PATIENT_PATTERNS,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


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




# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------
