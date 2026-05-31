"""Simple rule-based baseline matcher for patient-trial eligibility."""

import re

from app.models import CriterionDecision, CriterionMatchResult, CriterionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


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

_DBS_PATTERNS = [
    r"\bdbs\b",
    r"deep brain stimulation",
    r"subthalamic nucleus",
    r"stn\b",
]

_DBS_NEGATION_PATTERN = re.compile(
    r"\bno\b.{0,30}(?:history of|prior|previous)?\s*(?:dbs|deep brain stimulation)",
    re.IGNORECASE,
)

_MAOB_CRITERION_PATTERN = re.compile(r"mao.?b inhibitor", re.IGNORECASE)
_MAOB_DRUGS = [r"\brasagiline\b", r"\bselegiline\b", r"\bsafinamide\b"]
_MAOB_NEGATION_PATTERN = re.compile(r"\bno\b.{0,40}mao.?b", re.IGNORECASE)


def _has_negated_dbs(text: str) -> bool:
    return bool(_DBS_NEGATION_PATTERN.search(text))


def _has_maob_inhibitor(text: str) -> bool:
    if _has_negated_maob(text):
        return False
    return _any_match(_MAOB_DRUGS, text)


def _has_negated_maob(text: str) -> bool:
    return bool(_MAOB_NEGATION_PATTERN.search(text))

_COGNITIVE_EXCLUSION_PATTERNS = [
    r"mmse",
    r"moca",
    r"cognitive impairment",
    r"dementia",
    r"memory",
]

_UNCLEAR_MED_PATTERNS = [
    r"dose.*unclear",
    r"frequency.*unclear",
    r"unclear.*dose",
    r"unclear.*frequency",
    r"self.reported.*medication",
    r"no.*pharmacy records",
    r"uncertain.*levodopa",
    r"uncertain.*compliance",
    r"dose and frequency unclear",
]

_PARKINSON_PATTERNS = [
    r"parkinson",
    r"\bpd\b",
]

_STABLE_MED_PATTERNS = [
    r"stable.*levodopa",
    r"stable.*medication",
    r"stable.*regimen",
]

_MMSE_THRESHOLD_PATTERN = re.compile(r"mmse\s*[<≤]\s*(\d+)", re.IGNORECASE)
_MOCA_THRESHOLD_PATTERN = re.compile(r"moca\s*[<≤]\s*(\d+)", re.IGNORECASE)
_MMSE_VALUE_PATTERN = re.compile(r"mmse\s*(?:score)?\s*(\d+)", re.IGNORECASE)
_MOCA_VALUE_PATTERN = re.compile(r"moca\s*(?:score)?\s*(\d+)", re.IGNORECASE)

_STABILITY_CRITERION_PATTERN = re.compile(
    r"stable\s+medication\s+(?:regimen|therapy)\s+for\s+at\s+least\s+(\d+)\s+(weeks?|months?)",
    re.IGNORECASE,
)
_PATIENT_STABLE_DURATION_PATTERN = re.compile(
    r"(?:stable|unchanged|consistent).*?(\d+)\s+(weeks?|months?)"
    r"|(\d+)\s+(weeks?|months?).*?(?:stable|unchanged|consistent)",
    re.IGNORECASE,
)
_PATIENT_CHANGED_PATTERN = re.compile(
    r"(?:changed|adjusted|modified|switched|altered).*?(\d+)\s+(weeks?|months?)\s+ago"
    r"|(\d+)\s+(weeks?|months?)\s+ago.*?(?:changed|adjusted|modified|switched|altered)",
    re.IGNORECASE,
)

_HY_RANGE_PATTERN = re.compile(
    r"hoehn\s+and\s+yahr\s+stage\s+(\d+)\s*(?:to|-|–)\s*(\d+)", re.IGNORECASE
)
_HY_VALUE_PATTERN = re.compile(r"hoehn\s+and\s+yahr\s+stage\s+(\d+)", re.IGNORECASE)

# New pattern sets for extended unclear logic

_TRIAL_MED_SPECIFIC_PATTERNS = [
    r"levodopa",
    r"drug.*exposure",
    r"medication.*regimen",
    r"stable.*med",
    r"rotigotine",
    r"botulinum.*toxin",
    r"\bcomt\b",
    r"comt inhibitor",
    r"dopamine.*agonist",
    r"amantadine",
    r"rasagiline",
    r"entacapone",
    r"opicapone",
    r"apomorphine",
    r"drug.*naive",
    r"medication.*free",
    r"on.*levodopa",
    r"prior.*medication",
]

_PATIENT_UNCLEAR_MED_PATTERNS = [
    r"dose.*unclear",
    r"frequency.*unclear",
    r"unclear.*dose",
    r"unclear.*frequency",
    r"self.reported.*medication",
    r"no.*pharmacy records",
    r"uncertain.*levodopa",
    r"uncertain.*compliance",
    r"dose and frequency unclear",
    r"medication.*unclear",
    r"unclear.*medication",
    r"unknown.*medication",
    r"medication.*unknown",
    r"missing.*medication",
    r"medication.*not.*recorded",
    r"no.*medication.*record",
    r"medication.*details.*unavailable",
    r"incomplete.*medication",
]

_TRIAL_STAGE_SEVERITY_PATTERNS = [
    r"hoehn\s+and\s+yahr",
    r"\bh&y\b",
    r"\bhy\b\s*stage",
    r"\bupdrs\b",
    r"disease stage",
    r"disease.*severity",
    r"severity.*stage",
    r"freezing of gait",
    r"\bfog\b",
    r"\bfog\s",
    r"advanced\s+pd",
    r"advanced\s+parkinson",
    r"early\s+pd",
    r"early\s+parkinson",
    r"motor fluctuation",
    r"wearing.off",
    r"\blcig\b",
    r"intestinal gel",
    r"dbs candidacy",
    r"deep brain stimulation candidacy",
    r"disease duration",
]

_PATIENT_UNCLEAR_STAGE_PATTERNS = [
    r"disease_stage.*unclear",
    r"unclear.*disease_stage",
    r"disease stage.*unclear",
    r"unclear.*disease stage",
    r"stage.*unclear",
    r"unclear.*stage",
    r"severity.*unclear",
    r"unclear.*severity",
    r"unknown.*stage",
    r"stage.*unknown",
    r"missing.*duration",
    r"duration.*missing",
    r"duration.*unknown",
    r"unknown.*duration",
    r"duration.*unclear",
    r"unclear.*duration",
    r"h&y.*unknown",
    r"unknown.*h&y",
    r"hy.*unclear",
    r"unclear.*hy",
    r"updrs.*unknown",
    r"unknown.*updrs",
    r"missing.*severity",
    r"severity.*not.*recorded",
    r"no.*h.*y.*score",
    r"hoehn.*yahr.*unknown",
    r"hoehn.*yahr.*unclear",
    r"hoehn.*yahr.*not.*recorded",
    r"hoehn.*yahr.*missing",
]

_ATYPICAL_PARKINSON_PATTERNS = [
    r"unclear.*parkinsonism",
    r"parkinsonism.*unclear",
    r"suspected.*parkinsonism",
    r"parkinsonism.*suspected",
    r"atypical.*parkinsonism",
    r"parkinsonism.*atypical",
    r"secondary.*parkinsonism",
    r"parkinsonism.*secondary",
    r"multiple system atrophy",
    r"\bmsa\b",
    r"poor.*levodopa.*response",
    r"levodopa.*poor.*response",
    r"levodopa.unresponsive",
    r"vascular.*parkinsonism",
    r"drug.induced.*parkinsonism",
    r"parkinson.*plus",
    r"progressive supranuclear",
    r"\bpsp\b",
    r"corticobasal",
    r"\bcbd\b",
    r"dementia with lewy",
    r"\bdlb\b",
]

_IDIOPATHIC_PD_REQUIRED_PATTERNS = [
    r"idiopathic.*parkinson",
    r"parkinson.*idiopathic",
    r"confirmed.*parkinson",
    r"parkinson.*confirmed",
    r"diagnosis.*parkinson.*disease",
    r"established.*parkinson",
    r"uk.*brain.*bank",
    r"brain.*bank.*criteria",
    r"lewy.*body.*confirmed",
]

_ACTIVE_CANCER_PATTERNS = [
    r"active.*cancer",
    r"cancer.*active",
    r"active.*oncology",
    r"current.*chemotherapy",
    r"chemotherapy.*current",
    r"ongoing.*chemotherapy",
    r"current.*radiotherapy",
    r"active.*malignancy",
    r"malignancy.*active",
    r"active.*tumor",
    r"active.*tumour",
    r"undergoing.*cancer.*treatment",
    r"cancer.*treatment.*ongoing",
]

_TRIAL_SAFETY_SENSITIVE_PATTERNS = [
    r"cardiovascular",
    r"cardiac",
    r"hepatic",
    r"renal",
    r"kidney",
    r"liver",
    r"blood pressure",
    r"adverse.*event",
    r"safety",
    r"tolerability",
    r"comorbidities",
    r"serious.*illness",
    r"life.threatening",
    r"malignancy",
    r"immunosuppress",
    r"contraindication",
]

_RECENT_TRIAL_PATTERNS = [
    r"recent.*interventional.*trial",
    r"recent.*clinical.*trial",
    r"prior.*clinical.*trial",
    r"interventional.*study.*participation",
    r"enrolled.*in.*trial",
    r"enrolled.*in.*study",
    r"participated.*in.*trial",
    r"participated.*in.*study",
    r"recent.*study.*participation",
    r"concurrent.*trial",
    r"concurrent.*study",
    r"investigational.*drug.*recent",
    r"recent.*investigational",
    r"currently.*enrolled",
]

_TRIAL_WASHOUT_PATTERNS = [
    r"washout",
    r"prior.*trial",
    r"concurrent.*trial",
    r"interventional.*study",
    r"investigational.*drug",
    r"study.*participation",
    r"enrolled.*in.*another",
    r"prior.*participation",
]

_UNVERIFIABLE_INCLUSION_PATTERNS = [
    r"ability to.*(?:operate|use).*(?:device|app|application|system|software|technology)",
    r"(?:operate|use).*(?:device|app|application|system|software|technology).*independently",
    r"home.*(?:wifi|wi.fi|wireless|internet|broadband|connectivity)",
    r"(?:wifi|wi.fi|wireless|internet|broadband).*(?:access|connection|available|required)",
    r"no.*concurrent.*(?:trial|study|participation|investigational)",
    r"not.*(?:enrolled|participating).*(?:trial|study|investigational)",
    r"concurrent.*(?:trial|study).*(?:exclusion|prohibited|not permitted)",
    r"(?:medical|physician|doctor|clinician).*clearance",
    r"clearance.*(?:from|by).*(?:physician|doctor|clinician|medical)",
    r"written.*(?:clearance|approval|consent).*(?:physician|doctor)",
    r"caregiver.*(?:available|present|willing|required)",
    r"access to.*(?:transport|transportation|clinic|facility)",
    r"ability to.*(?:attend|travel|commute|visit).*(?:clinic|site|centre|center)",
    r"willing.*(?:to comply|to participate|to attend|to complete)",
    r"able to.*(?:comply|participate|attend|complete).*(?:protocol|study|trial|visits)",
]


def _count_unverifiable_inclusion_criteria(trial: dict) -> int:
    """Return the number of inclusion criteria that are logistical/external and cannot be verified from a patient profile."""
    count = 0
    for criterion in trial.get("inclusion_criteria", []):
        c = criterion.lower()
        if _any_match(_UNVERIFIABLE_INCLUSION_PATTERNS, c):
            count += 1
    return count


_PATIENT_COMPLEX_COMORBIDITY_PATTERNS = [
    r"\bfrail",
    r"frailty",
    r"recurrent.*falls",
    r"frequent.*falls",
    r"fall.*risk",
    r"orthostatic.*hypotension",
    r"postural.*hypotension",
    r"autonomic.*dysfunction",
    r"autonomic.*failure",
    r"\bpacemaker\b",
    r"\bimplanted.*device\b",
    r"\bdevice.*implant\b",
    r"cardiac.*device",
    r"deep brain stimulation",
    r"\bdbs\b",
    r"cognitive.*impairment",
    r"mild.*cognitive",
    r"\bmci\b",
    r"depression",
    r"\bdepressed\b",
    r"rem.*sleep.*behavior",
    r"\brbd\b",
    r"rem.*behavior.*disorder",
]

_TRIAL_COMPLEX_FOCUS_PATTERNS = [
    r"device.*trial",
    r"stimulation.*trial",
    r"stimulation.*study",
    r"\btms\b",
    r"\brtms\b",
    r"\btdcs\b",
    r"\bdbs\b",
    r"implant.*study",
    r"mri.*study",
    r"mri.*compatible",
    r"imaging.*study",
    r"neuroimaging",
    r"rehabilitation",
    r"physiotherapy",
    r"physical.*therapy",
    r"exercise.*trial",
    r"exercise.*study",
    r"gait.*study",
    r"gait.*trial",
    r"freezing.*gait",
    r"balance.*study",
    r"fall.*prevention",
    r"fall.*risk",
    r"neuropsychiatric",
    r"cognitive.*trial",
    r"cognitive.*study",
    r"cognitive.*assessment",
    r"protocol.*compliance",
    r"compliance.*protocol",
    r"adherence",
    r"neuropsychological",
]


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def _to_weeks(amount: int, unit: str) -> int:
    """Convert a duration amount+unit to whole weeks (months = 4 weeks)."""
    return amount * 4 if unit.lower().startswith("month") else amount


def _required_weeks(criterion: str) -> int | None:
    """Return the required stability duration in weeks, or None if not specified."""
    m = _STABILITY_CRITERION_PATTERN.search(criterion)
    if not m:
        return None
    return _to_weeks(int(m.group(1)), m.group(2))


def _patient_stable_weeks(patient_med_text: str) -> int | None:
    """Return how many weeks the patient's medication has been stable, or None."""
    m = _PATIENT_STABLE_DURATION_PATTERN.search(patient_med_text)
    if not m:
        return None
    # Groups 1+2 or 3+4 depending on which branch matched
    if m.group(1) is not None:
        return _to_weeks(int(m.group(1)), m.group(2))
    return _to_weeks(int(m.group(3)), m.group(4))


def _patient_changed_weeks_ago(patient_med_text: str) -> int | None:
    """Return how many weeks ago the medication was changed, or None."""
    m = _PATIENT_CHANGED_PATTERN.search(patient_med_text)
    if not m:
        return None
    if m.group(1) is not None:
        return _to_weeks(int(m.group(1)), m.group(2))
    return _to_weeks(int(m.group(3)), m.group(4))


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
    if _has_negated_dbs(patient_text):
        return None, None
    patient_has_dbs = _any_match(_DBS_PATTERNS, patient_text)
    if patient_has_dbs:
        return "deep brain stimulation (DBS) implant is an exclusion criterion", "DBS implant present"

    return None, None


def _check_maob(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Return (blocking_criterion, matched_fact) if MAO-B inhibitor exclusion applies."""
    exclusion_text = _text(trial.get("exclusion_criteria", []))
    if not _MAOB_CRITERION_PATTERN.search(exclusion_text):
        return None, None
    patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
    if _has_maob_inhibitor(patient_med_text):
        return "MAO-B inhibitor use is an exclusion criterion", "MAO-B inhibitor medication present"
    return None, None



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
            else:
                cog_text = _text(patient.get("exclusions", []) + [patient.get("diagnosis", "")])
                if _any_match(_COGNITIVE_EXCLUSION_PATTERNS, cog_text):
                    return (
                        f"cognitive exclusion: MMSE < {threshold}",
                        "cognitive impairment noted but MMSE score not available",
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
        req = _required_weeks(criterion)
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

    if _any_match(_PATIENT_UNCLEAR_STAGE_PATTERNS, patient_all_text):
        return (
            "trial requires disease stage or severity information but patient data is unclear or missing",
            "disease stage, severity, or duration unclear or missing",
        )

    # Also check if disease_stage field is explicitly "unclear"
    disease_stage = str(patient.get("disease_stage", "")).lower()
    if disease_stage in ("unclear", "unknown", "missing", "not recorded", ""):
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

    if _any_match(_IDIOPATHIC_PD_REQUIRED_PATTERNS, inclusion_text):
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


def _check_recent_trial_participation(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if patient has recent trial participation and trial has washout/prior trial criteria."""
    trial_text = _text(
        trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", [])
    )

    if not _any_match(_TRIAL_WASHOUT_PATTERNS, trial_text):
        return None, None

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )

    if _any_match(_RECENT_TRIAL_PATTERNS, patient_all_text):
        return (
            "trial has washout or prior study requirements; patient has recent or concurrent trial participation",
            "recent or concurrent trial participation noted",
        )

    return None, None


def _check_comorbidity_protocol_risk(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Return uncertain criterion if patient has complex comorbidities relevant to a protocol-sensitive trial."""
    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
        + [patient.get("diagnosis", "")]
    )

    if not _any_match(_PATIENT_COMPLEX_COMORBIDITY_PATTERNS, patient_all_text):
        return None, None

    trial_text = _text(
        trial.get("inclusion_criteria", []) + trial.get("exclusion_criteria", [])
    )

    if not _any_match(_TRIAL_COMPLEX_FOCUS_PATTERNS, trial_text):
        return None, None

    return (
        "patient has comorbidity or condition that may affect protocol compliance or safety in this trial type",
        "complex comorbidity noted in context of device/stimulation/imaging/rehabilitation/cognitive/gait-focused trial",
    )


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

    # --- DBS ---
    dbs_block, dbs_fact = _check_dbs(patient, trial)
    if dbs_block:
        blocking_criteria.append(dbs_block)
        matched_facts.append(dbs_fact)

    # --- MAO-B inhibitor ---
    maob_block, maob_fact = _check_maob(patient, trial)
    if maob_block:
        blocking_criteria.append(maob_block)
        if maob_fact:
            matched_facts.append(maob_fact)

    # --- Cognitive / MMSE / MoCA ---
    cog_block, cog_fact = _check_cognitive(patient, trial)
    if cog_block:
        blocking_criteria.append(cog_block)
        if cog_fact:
            matched_facts.append(cog_fact)

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

    # --- Extended: comorbidity risk in protocol-sensitive trial ---
    comorbid_uncertain, comorbid_fact = _check_comorbidity_protocol_risk(patient, trial)
    if comorbid_uncertain:
        uncertain_criteria.append(comorbid_uncertain)
        if comorbid_fact:
            matched_facts.append(comorbid_fact)

    # --- Extended: unverifiable inclusion criteria burden ---
    unverifiable_count = _count_unverifiable_inclusion_criteria(trial)
    if unverifiable_count >= 3 and not blocking_criteria:
        uncertain_criteria.append(
            f"unverifiable inclusion criteria: {unverifiable_count} inclusion criteria"
            " cannot be verified from the patient profile"
            " (e.g. device operation ability, home internet access,"
            " concurrent trial participation, physician clearance)"
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
        req = _required_weeks(criterion)
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
        req = _required_weeks(c_lower)
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
    if _any_match(_DBS_PATTERNS, c_lower):
        patient_text = _text(
            patient.get("key_features", [])
            + patient.get("medications", [])
            + patient.get("exclusions", [])
        )
        if _has_negated_dbs(patient_text):
            return CriterionDecision.not_met, "no DBS history documented"
        if any("dbs" in b or "deep brain" in b for b in blocking):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        if _any_match(_DBS_PATTERNS, patient_text):
            return CriterionDecision.met, "DBS implant present — patient excluded"
        return CriterionDecision.not_met, "no DBS implant found"

    # MAO-B inhibitor
    if _MAOB_CRITERION_PATTERN.search(c_lower):
        patient_med_text = _text(patient.get("medications", []) + patient.get("key_features", []))
        if _has_negated_maob(patient_med_text):
            return CriterionDecision.not_met, "no MAO-B inhibitor use documented"
        if _has_maob_inhibitor(patient_med_text):
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
