"""Cognitive eligibility helpers extracted from rule_matcher.py.

Contains _check_cognitive, _check_cognitive_exclusion_general, and
_check_cognitive_inclusion_minimum.  All behavior is identical to the
original implementations in rule_matcher.py.
"""

from app.eligibility.clinical_terms import (
    _any_match,
    is_negated,
    has_contradiction,
    _MMSE_THRESHOLD_PATTERN,
    _MOCA_THRESHOLD_PATTERN,
    _MMSE_VALUE_PATTERN,
    _MOCA_VALUE_PATTERN,
    _TRIAL_COGNITIVE_EXCLUSION_GENERAL_PATTERNS,
    _TRIAL_COGNITIVE_INCLUSION_MIN_PATTERNS,
    _MMSE_INCLUSION_MIN_PATTERN,
    _MOCA_INCLUSION_MIN_PATTERN,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


def _check_cognitive(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if cognitive score disqualifies patient."""
    exclusion_list = trial.get("exclusion_criteria", [])
    patient_features = _text(patient.get("key_features", []))

    for criterion in exclusion_list:
        m = _MMSE_THRESHOLD_PATTERN.search(criterion)
        if m:
            threshold = int(m.group(1))
            vm = _MMSE_VALUE_PATTERN.search(patient_features)
            if vm:
                patient_score = int(vm.group(1))
                if patient_score < threshold:
                    return (
                        f"cognitive exclusion: MMSE < {threshold}",
                        f"patient MMSE score {patient_score}",
                    )
        m = _MOCA_THRESHOLD_PATTERN.search(criterion)
        if m:
            threshold = int(m.group(1))
            vm = _MOCA_VALUE_PATTERN.search(patient_features)
            if vm:
                patient_score = int(vm.group(1))
                if patient_score < threshold:
                    return (
                        f"cognitive exclusion: MoCA < {threshold}",
                        f"patient MoCA score {patient_score}",
                    )

    return None, None


def _check_cognitive_exclusion_general(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when exclusion criteria explicitly exclude dementia/cognitive impairment (no numeric threshold)
    and patient clearly documents dementia or significant cognitive impairment.
    MCI/mild cognitive alone is not sufficient — requires explicit dementia or cognitive impairment.

    In DBS/neuropsychiatric/facial-expression/imaging outcome studies without a numeric cutoff
    or explicit dementia exclusion, downgrade cognitive impairment/low MoCA/MCI to unclear.
    """
    exclusion_list = trial.get("exclusion_criteria", [])
    patient_features = _text(
        patient.get("key_features", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )

    # Use stricter patient evidence — mild cognitive / MCI / early-onset PD alone is not enough
    _STRICT_COGNITIVE_IMPAIRMENT_PATTERNS = [
        r"\bdementia\b",
        r"(?:significant|moderate|severe|major|clear).*cognitive(?:\s+impairment)?",
        r"(?<!mild\s)(?<!early\s)(?<!possible\s)(?<!suspected\s)(?<!mci\s)\bcognitive impairment\b(?!\s+(?:mild|early|possible|suspected))",
        r"low moca",
        r"low mmse",
        r"impaired cognition",
        r"cognitive decline",
        r"neuropsychological impairment",
    ]
    _MCI_ONLY_PATTERNS = [
        r"\bmci\b",
        r"mild cognitive impairment",
        r"mild\s+cognitive",
    ]
    _HARD_COGNITIVE_PATTERNS = [
        r"\bdementia\b",
        r"(?:significant|moderate|severe).*cognitive",
        r"low moca",
        r"low mmse",
        r"impaired cognition",
    ]
    # If patient only has MCI/mild cognitive and nothing harder, do not hard-block
    if _any_match(_MCI_ONLY_PATTERNS, patient_features) and not _any_match(
        _HARD_COGNITIVE_PATTERNS, patient_features
    ):
        return None, None
    _EARLY_PD_EXEMPTION_PATTERNS = [
        r"early.onset.*parkinson",
        r"parkinson.*early.onset",
        r"very early.*parkinson",
        r"parkinson.*very early",
        r"early.*onset.*pd",
        r"young.onset.*parkinson",
        r"juvenile.*parkinson",
    ]
    if _any_match(_EARLY_PD_EXEMPTION_PATTERNS, patient_features) and not _any_match(
        _HARD_COGNITIVE_PATTERNS, patient_features
    ):
        return None, None
    if not _any_match(_STRICT_COGNITIVE_IMPAIRMENT_PATTERNS, patient_features):
        return None, None

    # Pure negation: patient explicitly denies cognitive impairment/dementia — do not block
    if is_negated(patient_features, "cognitive_impairment") and not has_contradiction(
        patient_features, "cognitive_impairment"
    ):
        return None, None

    # For DBS/neuropsychiatric/facial/imaging outcome trials without explicit numeric cutoff
    # or explicit dementia exclusion: downgrade to unclear instead of hard blocking.
    _DBS_NEURO_IMAGING_OUTCOME_PATTERNS = [
        r"\bdbs\b",
        r"deep brain stimulation",
        r"neuropsychiatric",
        r"neuropsychological",
        r"facial.*expression",
        r"expression.*facial",
        r"\bfmri\b",
        r"\bmri\b.*outcome",
        r"imaging.*outcome",
        r"neuroimaging.*outcome",
        r"cognitive.*outcome",
    ]
    trial_all_text = _text(
        trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", [])
    )
    has_numeric_cutoff = bool(
        _MMSE_THRESHOLD_PATTERN.search(trial_all_text)
        or _MOCA_THRESHOLD_PATTERN.search(trial_all_text)
    )
    has_explicit_dementia_excl = _any_match(
        [r"\bdementia\b", r"cognitive impairment.*exclud", r"exclud.*cognitive impairment"],
        trial_all_text,
    )
    if (
        _any_match(_DBS_NEURO_IMAGING_OUTCOME_PATTERNS, trial_all_text)
        and not has_numeric_cutoff
        and not has_explicit_dementia_excl
    ):
        return None, None  # Will be picked up as unclear by MCI/DBS block below

    for criterion in exclusion_list:
        c = criterion.lower()
        # Skip criteria that already have a numeric threshold (handled by _check_cognitive)
        if _MMSE_THRESHOLD_PATTERN.search(c) or _MOCA_THRESHOLD_PATTERN.search(c):
            continue
        if _any_match(_TRIAL_COGNITIVE_EXCLUSION_GENERAL_PATTERNS, c):
            return (
                "cognitive exclusion: trial excludes patients with dementia or cognitive impairment",
                "cognitive impairment or dementia documented in patient",
            )

    return None, None


def _check_cognitive_inclusion_minimum(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when inclusion criteria require a minimum cognitive score or intact cognition
    and patient data clearly indicates failure."""
    inclusion_list = trial.get("inclusion_criteria", [])
    patient_features = _text(patient.get("key_features", []))

    _HARD_COG_PATTERNS = [
        r"\bdementia\b",
        r"(?:significant|moderate|severe).*cognitive",
        r"low mmse",
        r"low moca",
        r"impaired cognition",
    ]
    _EARLY_PD_EXEMPT = [
        r"early.onset.*parkinson", r"parkinson.*early.onset",
        r"very early.*parkinson", r"young.onset.*parkinson",
        r"juvenile.*parkinson", r"early.*onset.*pd",
    ]

    for criterion in inclusion_list:
        c = criterion.lower()

        # Numeric MMSE minimum
        m = _MMSE_INCLUSION_MIN_PATTERN.search(c)
        if m:
            required = int(m.group(1))
            vm = _MMSE_VALUE_PATTERN.search(patient_features)
            if vm:
                score = int(vm.group(1))
                if score < required:
                    return (
                        f"cognitive inclusion minimum: MMSE >= {required} required; patient MMSE {score}",
                        f"patient MMSE {score} below required {required}",
                    )
            elif (
                _any_match(_HARD_COG_PATTERNS, patient_features)
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not (
                    is_negated(patient_features, "cognitive_impairment")
                    and not has_contradiction(patient_features, "cognitive_impairment")
                )
            ):
                return (
                    f"cognitive inclusion minimum: MMSE >= {required} required; patient has documented cognitive impairment",
                    "cognitive impairment documented; MMSE score not available",
                )
            continue

        # Numeric MoCA minimum
        m = _MOCA_INCLUSION_MIN_PATTERN.search(c)
        if m:
            required = int(m.group(1))
            vm = _MOCA_VALUE_PATTERN.search(patient_features)
            if vm:
                score = int(vm.group(1))
                if score < required:
                    return (
                        f"cognitive inclusion minimum: MoCA >= {required} required; patient MoCA {score}",
                        f"patient MoCA {score} below required {required}",
                    )
            elif (
                _any_match(_HARD_COG_PATTERNS, patient_features)
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not (
                    is_negated(patient_features, "cognitive_impairment")
                    and not has_contradiction(patient_features, "cognitive_impairment")
                )
            ):
                return (
                    f"cognitive inclusion minimum: MoCA >= {required} required; patient has documented cognitive impairment",
                    "cognitive impairment documented; MoCA score not available",
                )
            continue

        # Non-numeric intact-cognition requirement — require clear impairment, not just MCI
        if _any_match(_TRIAL_COGNITIVE_INCLUSION_MIN_PATTERNS, c):
            _CLEAR_IMPAIRMENT_PATTERNS = [
                r"\bdementia\b",
                r"(?:significant|moderate|severe|major|clear).*cognitive(?:\s+impairment)?",
                r"cognitive impairment(?!\s+(?:mild|early|possible|suspected))",
                r"low moca",
                r"low mmse",
                r"impaired cognition",
                r"cognitive decline",
                r"neuropsychological impairment",
            ]
            if (
                _any_match(_CLEAR_IMPAIRMENT_PATTERNS, patient_features)
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not (
                    is_negated(patient_features, "cognitive_impairment")
                    and not has_contradiction(patient_features, "cognitive_impairment")
                )
            ):
                return (
                    "cognitive inclusion requirement: intact cognition or consent capacity required; patient has documented cognitive impairment",
                    "cognitive impairment documented",
                )

    return None, None
