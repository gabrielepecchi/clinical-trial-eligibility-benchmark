"""Safety, cancer, and comorbidity helper rules extracted from rule_matcher."""

from app.eligibility.clinical_terms import (
    _any_match,
    is_negated,
    has_contradiction,
    _ACTIVE_CANCER_PATTERNS,
    _TRIAL_SAFETY_SENSITIVE_PATTERNS,
    _PATIENT_COMPLEX_COMORBIDITY_PATTERNS,
    _TRIAL_COMPLEX_FOCUS_PATTERNS,
    _COMORBIDITY_TARGET_PAIRS,
    _HARD_CONTRAINDICATION_PAIRS,
    _TRIAL_ONCOLOGY_REQUIRED_PATTERNS,
    _PATIENT_CANCER_PATTERNS,
    _TRIAL_HIGH_DEMAND_EXERCISE_PATTERNS,
    _ACTIVE_CANCER_PATIENT_PATTERNS,
    _ACTIVE_CANCER_TRIAL_PATTERNS,
    _RBD_AMBIGUITY_TRIGGER_PATTERNS,
    _RBD_TARGET_SUPPRESSION_PATTERNS,
    _FRAILTY_TARGET_SUPPRESSION_PATTERNS,
    _DEPRESSION_IMAGING_BIOMARKER_PATTERNS,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


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
