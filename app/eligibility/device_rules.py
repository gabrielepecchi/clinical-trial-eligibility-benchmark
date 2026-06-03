"""DBS and device eligibility helpers extracted from rule_matcher.py.

Contains _check_dbs, _check_dbs_mri_compatibility, _check_dbs_required,
and _check_device_contraindication_stimulation.  All behavior is identical
to the original implementations in rule_matcher.py.
"""

from app.eligibility.clinical_terms import (
    _any_match,
    _has_negated_dbs,
    has_contradiction,
    _patient_has_procedure,
    _DBS_PATTERNS,
    _MMSE_THRESHOLD_PATTERN,
    _MOCA_THRESHOLD_PATTERN,
    _TRIAL_DBS_REQUIRED_PATTERNS,
    _AMBIGUOUS_DBS_INCLUSION_PATTERNS,
    _TRIAL_STIMULATION_PATTERNS,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


# ---------------------------------------------------------------------------
# Local DBS/device patterns (only used by functions in this module)
# ---------------------------------------------------------------------------

_DBS_NEGATION_PHRASES = [
    r"\bno\b.{0,40}(?:history\s+of\s+)?(?:dbs|deep\s+brain\s+stimulation)",
    r"\bdenies?\b.{0,40}(?:dbs|deep\s+brain\s+stimulation)",
    r"\bno\s+prior\b.{0,40}(?:dbs|deep\s+brain\s+stimulation)",
    r"\bno\s+previous\b.{0,40}(?:dbs|deep\s+brain\s+stimulation)",
    r"\bwithout\b.{0,40}(?:dbs|deep\s+brain\s+stimulation)",
]

_DBS_IMPLANT_PHRASES = [
    r"(?:dbs|deep\s+brain\s+stimulation)\s+(?:implanted|implant|surgery|procedure|placed|insertion)",
    r"(?:implanted|underwent|undergone|received|has)\s+(?:a\s+)?(?:dbs|deep\s+brain\s+stimulation)",
    r"(?:stn|subthalamic).{0,20}(?:dbs|stimulation).{0,20}(?:implanted|implant|surgery)",
    r"(?:dbs|deep\s+brain\s+stimulation).{0,40}(?:years?\s+ago|months?\s+ago)",
]

_DBS_CANDIDACY_TRIAL_PATTERNS = [
    r"dbs\s+candidacy",
    r"candidacy.*dbs",
    r"indication.*dbs",
    r"dbs.*indication",
    r"scheduled\s+to\s+undergo\s+dbs",
    r"dbs.*scheduled",
    r"meets\s+criteria\s+for.*dbs",
    r"criteria\s+for\s+(?:treatment\s+with\s+)?(?:stn.)?dbs",
    r"dbs\s+(?:neuropsychiatric|effects|programming|optimization)",
]

_DBS_IMPLANTED_TARGET_PATTERNS = [
    r"dbs\s+(?:effects?|outcomes?|programming|optimization|facial|parameters?|follow.up)",
    r"(?:effects?|outcomes?|programming|optimization|follow.up).*dbs",
    r"lfp\s+sensing",
    r"directional\s+lead",
    r"dbs.*(?:implanted|patient|surgery|undergone)",
    r"(?:undergone|implanted|completed).*dbs",
    r"subthalamic.*steering",
    r"stimulation.*parameter",
    r"dbs.*optimization",
]

_MRI_IMAGING_TRIAL_PATTERNS_DBS = [
    r"\bfmri\b", r"\bmri\b", r"neuroimaging", r"magnetic resonance imaging",
    r"magnetic resonance", r"mri.*compatible", r"mri.*safety",
    r"imaging.*protocol", r"brain.*imaging",
]

_EXPLICIT_PRIOR_DBS_EXCLUSION = [
    r"prior.*dbs.*(?:implant|surgery|procedure)",
    r"previous.*dbs.*(?:implant|surgery|procedure)",
    r"existing.*dbs.*(?:implant|hardware)",
    r"dbs.*(?:implant|surgery|procedure).*(?:prior|previous|existing|already)",
    r"already.*(?:implanted|undergone).*dbs",
]

_PACEMAKER_PATTERNS = [
    r"\bpacemaker\b",
    r"cardiac.*pacemaker",
    r"implanted.*cardiac",
    r"cardiac.*device",
    r"implanted.*pacemaker",
    r"implantable.*cardioverter",
    r"\bicd\b",
]

_EXPLICIT_PACEMAKER_EXCLUSION_PATTERNS = [
    r"(?:metal.*implants?.*and.*)?cardiac\s+pacemaker",
    r"pacemaker.*(?:exclusion|excluded|contraindicated|not permitted)",
    r"(?:exclusion|excluded|contraindicated).*pacemaker",
    r"metal.*implants?.*pacemaker",
    r"pacemaker.*metal.*implants?",
]

_MRI_CONDITIONAL_PATTERNS = [
    r"mri.conditional.*dbs",
    r"dbs.*mri.conditional",
    r"mri.compatible.*dbs",
    r"dbs.*mri.compatible",
    r"mri.safe.*dbs",
    r"dbs.*mri.safe",
    r"medtronic.*mri",
    r"mri.*medtronic",
]

_MRI_AS_PROCEDURE_PATTERNS = [
    r"\bfmri\b",
    r"magnetic resonance imaging.*(?:session|scan|protocol|visit)",
    r"(?:session|scan|protocol|visit).*magnetic resonance imaging",
    r"neuroimaging.*(?:session|protocol|study)",
    r"mri.*(?:session|scan|protocol|acquisition)",
    r"(?:session|scan|protocol|acquisition).*mri",
    r"fmri.*dbs",
    r"dbs.*fmri",
]

_TRIAL_META_FIELDS = [
    "title", "brief_title", "official_title", "summary", "brief_summary",
    "description", "detailed_description",
]

_TRIAL_META_FIELDS_FULL = [
    "title", "brief_title", "official_title", "summary", "brief_summary",
    "description", "detailed_description", "intervention", "intervention_name",
    "intervention_type", "interventions", "keywords", "conditions",
]


def _check_dbs(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if DBS is a problem, else (None, None)."""
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    has_dbs_exclusion = _any_match(_DBS_PATTERNS, exclusion_text)
    if not has_dbs_exclusion:
        return None, None

    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
    )
    _has_dbs_negation_phrase = _any_match(_DBS_NEGATION_PHRASES, patient_text)
    _has_dbs_implant_phrase = _any_match(_DBS_IMPLANT_PHRASES, patient_text)
    if _has_dbs_negation_phrase and _has_dbs_implant_phrase:
        return (
            "__unclear__:contradictory DBS records: both no DBS history and DBS implant/procedure documented",
            "contradiction in DBS records",
        )
    if has_contradiction(patient_text, "dbs"):
        return (
            "__unclear__:contradictory DBS records: both negation and affirmation found — eligibility cannot be determined",
            "contradiction in DBS records",
        )
    if _has_negated_dbs(patient_text):
        return None, None
    patient_has_dbs = _patient_has_procedure(patient_text, "dbs")
    if not patient_has_dbs:
        return None, None

    inclusion_text = _text(trial.get("inclusion_criteria", []))
    meta_text = " ".join(_text(trial.get(f, "") or "") for f in _TRIAL_META_FIELDS)
    trial_full = inclusion_text + " " + exclusion_text + " " + meta_text

    is_candidacy = _any_match(_DBS_CANDIDACY_TRIAL_PATTERNS, inclusion_text)
    is_implanted_target = _any_match(_DBS_IMPLANTED_TARGET_PATTERNS, trial_full)
    is_mri_context = _any_match(_MRI_IMAGING_TRIAL_PATTERNS_DBS, trial_full)

    if is_mri_context and not is_implanted_target:
        return "deep brain stimulation (DBS) implant is an exclusion criterion", "DBS implant present"

    is_outcomes_study = is_implanted_target or is_candidacy

    if is_outcomes_study:
        if not _any_match(_EXPLICIT_PRIOR_DBS_EXCLUSION, exclusion_text):
            return None, None

    return "deep brain stimulation (DBS) implant is an exclusion criterion", "DBS implant present"


def _check_dbs_mri_compatibility(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when patient has DBS implant and trial involves MRI/fMRI/neuroimaging without confirmed MRI-conditional compatibility.

    Returns (blocking_criterion, matched_fact) or (None, None).
    """
    _MRI_IMAGING_TRIAL_PATTERNS_LOCAL = [
        r"\bfmri\b",
        r"\bmri\b",
        r"neuroimaging",
        r"magnetic resonance imaging",
        r"magnetic resonance",
        r"imaging.*protocol",
        r"brain.*imaging",
        r"fmri.*dbs",
        r"dbs.*fmri",
    ]
    collected: list[str] = []
    for f in ["inclusion_criteria", "exclusion_criteria"]:
        v = trial.get(f, [])
        if isinstance(v, list):
            collected.extend(v)
        elif v:
            collected.append(str(v))
    for f in _TRIAL_META_FIELDS_FULL:
        v = trial.get(f, "")
        if v:
            collected.append(str(v))
    all_trial_text = _text(collected)

    if not _any_match(_MRI_IMAGING_TRIAL_PATTERNS_LOCAL, all_trial_text):
        return None, None

    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    if _has_negated_dbs(patient_text):
        return None, None
    if not _patient_has_procedure(patient_text, "dbs"):
        return None, None

    if (
        _any_match(_DBS_IMPLANTED_TARGET_PATTERNS, all_trial_text)
        and _any_match(_MRI_CONDITIONAL_PATTERNS, all_trial_text)
    ):
        return None, None

    excl_text = _text(trial.get("exclusion_criteria", []))
    has_mri_as_procedure = _any_match(_MRI_AS_PROCEDURE_PATTERNS, all_trial_text)
    _DBS_EXPLICITLY_EXCLUDED = _any_match(_DBS_PATTERNS, excl_text)

    if not has_mri_as_procedure and not _DBS_EXPLICITLY_EXCLUDED:
        return None, None

    return (
        "DBS hardware and MRI/fMRI incompatibility: patient has existing DBS implant; "
        "MRI/fMRI-based trial without confirmed MRI-conditional DBS compatibility",
        "DBS implant present; MRI/fMRI trial",
    )


def _check_dbs_required(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when inclusion criteria require DBS but patient has no documented DBS.

    Returns (blocking_criterion, matched_fact) for hard block,
    or ('__unclear__:...', reason) for ambiguous cases.

    Hard block: clear prior/active DBS requirements (e.g. 'prior bilateral STN DBS surgery required').
    Unclear: ambiguous DBS protocol wording (candidacy, effects, LFP, directional leads,
             electrophysiology, STN recording, compatible hardware, neuropsychiatric, etc.)
    """
    inclusion_list = trial.get("inclusion_criteria", [])

    has_hard_requirement = any(
        _any_match(_TRIAL_DBS_REQUIRED_PATTERNS, c.lower()) for c in inclusion_list
    )
    has_ambiguous = any(
        _any_match(_AMBIGUOUS_DBS_INCLUSION_PATTERNS, c.lower()) for c in inclusion_list
    )

    if not has_hard_requirement and not has_ambiguous:
        return None, None

    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    if _any_match(_DBS_PATTERNS, patient_text) and not _has_negated_dbs(patient_text):
        return None, None  # Patient has DBS — fine

    if has_hard_requirement:
        return (
            "DBS required: trial requires prior or active DBS implant; patient has no documented DBS",
            "no DBS documented",
        )

    # Ambiguous only — return unclear signal
    return (
        "__unclear__:DBS eligibility unclear: trial involves DBS effects/candidacy/LFP/electrophysiology "
        "but patient has no confirmed DBS or hardware compatibility not established",
        "no confirmed DBS; ambiguous DBS-related study",
    )


def _check_device_contraindication_stimulation(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when patient has implanted cardiac device and trial involves transcranial/electrical stimulation."""
    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    if not _any_match(_PACEMAKER_PATTERNS, patient_text):
        return None, None

    _list_fields = ["inclusion_criteria", "exclusion_criteria", "interventions", "keywords", "conditions"]
    _str_fields = [
        "title", "brief_title", "official_title", "summary", "brief_summary",
        "description", "detailed_description", "intervention", "intervention_name",
        "intervention_type",
    ]
    collected: list[str] = []
    for f in _list_fields:
        v = trial.get(f, [])
        if isinstance(v, list):
            collected.extend(v)
        elif v:
            collected.append(str(v))
    for f in _str_fields:
        v = trial.get(f, "")
        if v:
            collected.append(str(v))
    all_trial_fields = _text(collected)
    if _any_match(_TRIAL_STIMULATION_PATTERNS, all_trial_fields):
        return (
            "hard safety contraindication: implanted cardiac device is incompatible with transcranial/electrical stimulation",
            "implanted cardiac device present; stimulation trial",
        )

    # Also block if exclusion criteria explicitly list pacemaker as an exclusion
    excl_text = _text(trial.get("exclusion_criteria", []))
    if _any_match(_EXPLICIT_PACEMAKER_EXCLUSION_PATTERNS, excl_text):
        return (
            "hard safety contraindication: patient has implanted cardiac pacemaker which is explicitly excluded",
            "implanted cardiac device present; pacemaker explicitly excluded",
        )

    return None, None
