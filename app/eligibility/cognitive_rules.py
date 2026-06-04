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

import re


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


def _extract_patient_score(patient: dict, key: str) -> int | None:
    """Extract a numeric cognitive score from a dedicated patient field, or None."""
    val = patient.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _patient_cognitive_normal(patient: dict) -> bool:
    """Return True if cognitive_status field clearly indicates normal/intact cognition."""
    status = str(patient.get("cognitive_status", "")).lower().strip()
    _NORMAL_PATTERNS = [
        r"\bnormal\b",
        r"\bintact\b",
        r"\bnot impaired\b",
        r"\bno impairment\b",
        r"\bno cognitive impairment\b",
        r"\bcognitively intact\b",
        r"\bcognitively normal\b",
        r"\bunimpaired\b",
    ]
    return bool(status) and _any_match(_NORMAL_PATTERNS, status)


def _check_cognitive(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if cognitive score disqualifies patient."""
    exclusion_list = trial.get("exclusion_criteria", [])

    # Resolve MMSE and MoCA scores: prefer dedicated fields, fall back to key_features text
    patient_features = _text(patient.get("key_features", []))

    mmse_score: int | None = _extract_patient_score(patient, "mmse_score")
    moca_score: int | None = _extract_patient_score(patient, "moca_score")

    # If cognitive_status is explicitly normal/intact, do not block on score thresholds
    if _patient_cognitive_normal(patient):
        return None, None

    for criterion in exclusion_list:
        m = _MMSE_THRESHOLD_PATTERN.search(criterion)
        if m:
            threshold = int(m.group(1))
            # Use dedicated field first, then fall back to text extraction
            if mmse_score is None:
                vm = _MMSE_VALUE_PATTERN.search(patient_features)
                if vm:
                    mmse_score = int(vm.group(1))
            if mmse_score is not None:
                if mmse_score < threshold:
                    return (
                        f"cognitive exclusion: MMSE < {threshold}",
                        f"patient MMSE score {mmse_score}",
                    )
                # Score meets threshold — do not block
                continue

        m = _MOCA_THRESHOLD_PATTERN.search(criterion)
        if m:
            threshold = int(m.group(1))
            if moca_score is None:
                vm = _MOCA_VALUE_PATTERN.search(patient_features)
                if vm:
                    moca_score = int(vm.group(1))
            if moca_score is not None:
                if moca_score < threshold:
                    return (
                        f"cognitive exclusion: MoCA < {threshold}",
                        f"patient MoCA score {moca_score}",
                    )
                # Score meets threshold — do not block
                continue

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

    # If cognitive_status is explicitly normal/intact, do not block
    if _patient_cognitive_normal(patient):
        return None, None

    # If dedicated score fields are present and meet any relevant threshold, do not block
    mmse_score = _extract_patient_score(patient, "mmse_score")
    moca_score = _extract_patient_score(patient, "moca_score")
    trial_all_text = _text(
        trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", [])
    )
    has_numeric_cutoff = bool(
        _MMSE_THRESHOLD_PATTERN.search(trial_all_text)
        or _MOCA_THRESHOLD_PATTERN.search(trial_all_text)
    )
    if has_numeric_cutoff:
        # If score fields are present and meet thresholds, do not block here
        # (numeric threshold already handled by _check_cognitive)
        if mmse_score is not None:
            m = _MMSE_THRESHOLD_PATTERN.search(trial_all_text)
            if m and mmse_score >= int(m.group(1)):
                return None, None
        if moca_score is not None:
            m = _MOCA_THRESHOLD_PATTERN.search(trial_all_text)
            if m and moca_score >= int(m.group(1)):
                return None, None

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

    # Supplement text-based patterns with dedicated score fields
    # If moca_score or mmse_score is explicitly documented as low (e.g., below common thresholds),
    # treat as hard cognitive pattern evidence.  We use a conservative threshold of 24 (MMSE) / 21 (MoCA).
    _COG_SCORE_IMPLIES_IMPAIRMENT = False
    if mmse_score is not None and mmse_score < 24:
        _COG_SCORE_IMPLIES_IMPAIRMENT = True
    if moca_score is not None and moca_score < 21:
        _COG_SCORE_IMPLIES_IMPAIRMENT = True

    # If patient only has MCI/mild cognitive and nothing harder, do not hard-block
    if _any_match(_MCI_ONLY_PATTERNS, patient_features) and not _any_match(
        _HARD_COGNITIVE_PATTERNS, patient_features
    ) and not _COG_SCORE_IMPLIES_IMPAIRMENT:
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
    ) and not _COG_SCORE_IMPLIES_IMPAIRMENT:
        return None, None

    if not _any_match(_STRICT_COGNITIVE_IMPAIRMENT_PATTERNS, patient_features) and not _COG_SCORE_IMPLIES_IMPAIRMENT:
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

    # If cognitive_status is explicitly normal/intact, do not block
    if _patient_cognitive_normal(patient):
        return None, None

    # Resolve dedicated score fields
    mmse_score = _extract_patient_score(patient, "mmse_score")
    moca_score = _extract_patient_score(patient, "moca_score")

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

    # Supplement text patterns with numeric score evidence
    def _patient_has_hard_cognitive_evidence() -> bool:
        if _any_match(_HARD_COG_PATTERNS, patient_features):
            return True
        if mmse_score is not None and mmse_score < 24:
            return True
        if moca_score is not None and moca_score < 21:
            return True
        return False

    def _patient_negated_cognitive() -> bool:
        return (
            is_negated(patient_features, "cognitive_impairment")
            and not has_contradiction(patient_features, "cognitive_impairment")
        )

    for criterion in inclusion_list:
        c = criterion.lower()

        # Numeric MMSE minimum
        m = _MMSE_INCLUSION_MIN_PATTERN.search(c)
        if m:
            required = int(m.group(1))
            # Use dedicated field first
            effective_mmse = mmse_score
            if effective_mmse is None:
                vm = _MMSE_VALUE_PATTERN.search(patient_features)
                if vm:
                    effective_mmse = int(vm.group(1))
            if effective_mmse is not None:
                if effective_mmse >= required:
                    # Score meets requirement — do not block
                    continue
                return (
                    f"cognitive inclusion minimum: MMSE >= {required} required; patient MMSE {effective_mmse}",
                    f"patient MMSE {effective_mmse} below required {required}",
                )
            elif (
                _patient_has_hard_cognitive_evidence()
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not _patient_negated_cognitive()
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
            effective_moca = moca_score
            if effective_moca is None:
                vm = _MOCA_VALUE_PATTERN.search(patient_features)
                if vm:
                    effective_moca = int(vm.group(1))
            if effective_moca is not None:
                if effective_moca >= required:
                    # Score meets requirement — do not block
                    continue
                return (
                    f"cognitive inclusion minimum: MoCA >= {required} required; patient MoCA {effective_moca}",
                    f"patient MoCA {effective_moca} below required {required}",
                )
            elif (
                _patient_has_hard_cognitive_evidence()
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not _patient_negated_cognitive()
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
            has_clear_impairment = (
                _any_match(_CLEAR_IMPAIRMENT_PATTERNS, patient_features)
                or (mmse_score is not None and mmse_score < 24)
                or (moca_score is not None and moca_score < 21)
            )
            if (
                has_clear_impairment
                and not _any_match(_EARLY_PD_EXEMPT, patient_features)
                and not _patient_negated_cognitive()
            ):
                return (
                    "cognitive inclusion requirement: intact cognition or consent capacity required; patient has documented cognitive impairment",
                    "cognitive impairment documented",
                )

    return None, None
