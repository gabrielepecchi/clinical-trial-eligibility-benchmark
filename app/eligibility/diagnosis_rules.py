"""Diagnosis/Parkinson disease eligibility rule helpers."""

import re

from app.eligibility.clinical_terms import (
    _any_match,
    _PARKINSON_PATTERNS,
    _ATYPICAL_PARKINSON_PATTERNS,
    _IDIOPATHIC_PD_REQUIRED_PATTERNS,
    _HEALTHY_CONTROL_TRIAL_PATTERNS,
    _HEALTHY_CONTROL_AMBIGUITY_SIGNALS,
    _PATIENT_HEALTHY_CONTROL_PATTERNS,
    _INTERVENTIONAL_PD_ONLY_PATTERNS,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


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
