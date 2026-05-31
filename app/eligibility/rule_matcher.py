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

# Pairs: (patient comorbidity patterns, trial inclusion target patterns).
# If a patient comorbidity matches AND the trial's *inclusion criteria* signal
# that comorbidity is the enrolment target, suppress the uncertain flag.
_COMORBIDITY_TARGET_PAIRS: list[tuple[list[str], list[str]]] = [
    (
        # FoG / gait patient in a FoG / gait trial
        [r"freezing.*gait", r"\bfog\b", r"gait.*disturbance", r"gait.*impairment"],
        [r"freezing.*gait", r"\bfog\b", r"gait.*study", r"gait.*trial",
         r"somatosensory.*stimulation", r"auditory.*cue", r"gait.*cueing"],
    ),
    (
        # DBS-implanted patient in a DBS-focused study
        [r"\bdbs\b", r"deep brain stimulation", r"subthalamic", r"\bstn\b"],
        [r"dbs.*effect", r"effect.*dbs", r"dbs.*patient", r"dbs.*implant",
         r"deep brain stimulation.*effect", r"lfp.*sensing", r"directional.*lead",
         r"dbs.*facial", r"dbs.*neuropsychiatric", r"dbs.*programming",
         r"subthalamic.*steering", r"dbs.*optimization",
         r"undergone.*dbs", r"dbs.*surgery", r"subthalamic.*dbs",
         r"stn.*surgery", r"subthalamic nucleus.*dbs"],
    ),
    (
        # Frail patient in a frailty / home physiotherapy trial
        [r"\bfrail", r"frailty"],
        [r"frailty.*trial", r"frailty.*study", r"frail.*patient",
         r"home.*physiotherapy", r"home.*physical.*therapy",
         r"frailty.*intervention", r"frailty.*rehabilitation"],
    ),
]

# Hard safety contraindications: (patient patterns, trial procedure patterns).
# If matched, add a blocking criterion → not_eligible.
_HARD_CONTRAINDICATION_PAIRS: list[tuple[list[str], list[str]]] = [
    (
        # Implanted cardiac device + electrical brain stimulation
        [r"\bpacemaker\b", r"cardiac.*pacemaker", r"implanted.*cardiac",
         r"cardiac.*device", r"implanted.*pacemaker"],
        [r"\brtms\b", r"\btms\b", r"\btdcs\b",
         r"transcranial.*magnetic", r"transcranial.*electrical",
         r"repetitive.*transcranial", r"brain.*stimulation.*trial",
         r"non.invasive.*brain.*stimulation"],
    ),
]

# Patterns indicating patient has cognitive impairment / dementia (general, non-numeric)
_PATIENT_COGNITIVE_IMPAIRMENT_PATTERNS = [
    r"cognitive impairment",
    r"dementia",
    r"\bmci\b",
    r"mild cognitive",
    r"low moca",
    r"low mmse",
    r"impaired cognition",
    r"cognitive decline",
    r"neuropsychological impairment",
]

# Trial exclusion criteria that explicitly exclude dementia or cognitive impairment (no numeric threshold).
# Must be explicit — vague words like "cognitive", "memory", or "neuropsychological" alone do not qualify.
_TRIAL_COGNITIVE_EXCLUSION_GENERAL_PATTERNS = [
    r"\bdementia\b",
    r"\bcognitive impairment\b",
    r"neuropsychological impairment",
    r"inability to (?:give )?(?:informed )?consent",
    r"inability to cooperate",
    r"unable to (?:give )?(?:informed )?consent",
    r"unable to cooperate",
    r"lacks? (?:capacity|ability) to consent",
    r"excluded.*(?:dementia|cognitive impairment)",
    r"(?:dementia|cognitive impairment).*excluded",
]

# Trial inclusion criteria requiring cognitive minimum (non-numeric).
# Must be explicit capacity/intact-cognition language — not generic cognitive assessments.
_TRIAL_COGNITIVE_INCLUSION_MIN_PATTERNS = [
    r"intact cognition",
    r"capacity to (?:give )?(?:informed )?consent",
    r"capacity to cooperate",
    r"ability to (?:give )?(?:informed )?consent",
    r"cognitively intact",
    r"no cognitive impairment",
    r"normal cognition",
    r"able to provide informed consent",
    r"neuropsychological testing.*required",
    r"patient.reported outcome.*(?:reliable|valid|required)",
]

_MMSE_INCLUSION_MIN_PATTERN = re.compile(r"mmse\s*[≥>=]+\s*(\d+)", re.IGNORECASE)
_MOCA_INCLUSION_MIN_PATTERN = re.compile(r"moca\s*[≥>=]+\s*(\d+)", re.IGNORECASE)

# Trial inclusion criteria requiring existing/prior/active DBS hardware.
# Only fire when the text clearly requires the patient to already have DBS.
# Do NOT include ambiguous DBS-candidacy patterns.
_TRIAL_DBS_REQUIRED_PATTERNS = [
    r"prior.*bilateral.*stn.*dbs",
    r"bilateral.*stn.*dbs.*surgery",
    r"prior.*dbs.*surgery",
    r"undergone.*dbs",
    r"active.*dbs.*hardware",
    r"subthalamic.*dbs.*surgery",
    r"dbs.*surgery.*required",
    r"must.*have.*(?:undergone|received).*dbs",
    r"requires.*prior.*dbs",
    r"prior.*deep brain stimulation.*surgery",
    r"undergone.*deep brain stimulation",
    r"deep brain stimulation.*surgery.*required",
    r"previously.*implanted.*dbs",
]

# Broader transcranial/electrical stimulation patterns for device contraindication.
# Intentionally excludes generic "brain stimulation", "electrical stimulation", or rehab wording.
_TRIAL_STIMULATION_PATTERNS = [
    r"\brtms\b",
    r"\btms\b",
    r"\btdcs\b",
    r"transcranial.*magnetic",
    r"transcranial.*electric",
    r"transcranial.*direct.*current",
    r"repetitive.*transcranial",
    r"non.invasive.*brain.*stimulation",
]

# Parent / open-label extension study requirement patterns
_TRIAL_PARENT_STUDY_REQUIRED_PATTERNS = [
    r"(?:completed?|completion of).*(?:parent|double.blind|preceding|prior|previous|core).*(?:study|trial|phase)",
    r"(?:parent|double.blind|preceding|prior|previous|core).*(?:study|trial|phase).*(?:completed?|completion)",
    r"participated? in.*(?:parent|core|preceding|double.blind).*(?:study|trial|phase)",
    r"enrolled? in.*(?:parent|core|preceding).*(?:study|trial)",
    r"prior.*participation.*(?:parent|core|double.blind).*(?:study|trial)",
    r"open.label.*extension",
    r"eligible.*open.label.*extension",
    r"eligible.*for.*extension.*(?:study|trial|phase)",
    r"(?:rollover|roll.over).*(?:from|study|trial|phase)",
    r"(?:from|of).*(?:rollover|roll.over)",
    r"continuation.*(?:from|of).*(?:parent|prior|previous).*(?:study|trial)",
    r"\bsp\d{3,}\b",
    r"parent.*(?:study|trial|protocol)",
    r"double.blind.*(?:study|trial|phase).*(?:eligible|qualif|complet|enroll|participat)",
    r"(?:eligible|qualif|complet|enroll|participat).*double.blind.*(?:study|trial|phase)",
    r"prior.*(?:study|trial).*(?:complet|participat|enroll)",
    r"must have completed.*(?:study|trial|phase|treatment period)",
    r"maintenance.*(?:phase|period).*(?:prior|previous|complet)",
]

_PATIENT_PRIOR_STUDY_PATTERNS = [
    r"completed.*(?:parent|prior|previous|double.blind|core).*(?:study|trial|phase)",
    r"participated.*(?:parent|prior|previous|core).*(?:study|trial)",
    r"enrolled.*(?:parent|core|prior).*(?:study|trial)",
    r"rollover.*(?:from|patient)",
    r"prior.*study.*complet",
    r"\bsp\d{3,}\b",
    r"open.label.*extension.*(?:eligible|enrolled|complet)",
]

# Oncology-specific diagnosis required (advanced/metastatic solid tumor etc.)
# NOTE: histolog/cytolog confirmed alone must NOT trigger — only when paired with oncology terms.
_TRIAL_ONCOLOGY_REQUIRED_PATTERNS = [
    r"(?:advanced|metastatic).*(?:solid\s*tumou?r|malignancy|cancer|carcinoma)",
    r"(?:solid\s*tumou?r|malignancy|cancer|carcinoma).*(?:advanced|metastatic)",
    r"histolog(?:ically)?.*(?:confirmed|proven).*(?:solid\s*tumou?r|malignancy|cancer|carcinoma|melanoma|nsclc|sclc|hnscc)",
    r"cytolog(?:ically)?.*(?:confirmed|proven).*(?:solid\s*tumou?r|malignancy|cancer|carcinoma)",
    r"(?:confirmed|proven).*(?:tumou?r|cancer|malignancy|carcinoma)",
    r"\bnsclc\b",
    r"\bsclc\b",
    r"\bhnscc\b",
    r"measurable disease.*recist",
    r"recist.*measurable",
    r"non.small.cell lung",
    r"small.cell lung",
    r"(?:colorectal|colon|rectal).*cancer",
    r"(?:advanced|metastatic).*(?:melanoma|glioblastoma|glioma|hepatocellular|cholangiocarcinoma)",
    r"unresectable.*(?:tumou?r|carcinoma|cancer|malignancy)",
    r"solid\s*tumou?r",
    r"(?:cancer|oncology|malignancy|carcinoma).*(?:diagnosis|confirmed|required|must have)",
]

_PATIENT_CANCER_PATTERNS = [
    r"cancer",
    r"\btumou?r\b",
    r"malignancy",
    r"carcinoma",
    r"solid tumor",
    r"solid tumour",
    r"chemotherapy",
    r"oncology",
    r"radiotherapy",
    r"immunotherapy",
    r"\bnsclc\b",
    r"\bsclc\b",
    r"\bmelanoma\b",
    r"\bglioblastoma\b",
    r"\bglioma\b",
    r"metastatic",
    r"lymphoma",
    r"leukemia",
    r"leukaemia",
]

# High-demand physical exercise trial patterns
_TRIAL_HIGH_DEMAND_EXERCISE_PATTERNS = [
    r"treadmill",
    r"agility.*training",
    r"training.*agility",
    r"high.intensity.*(?:exercise|training|physical)",
    r"(?:exercise|training|physical).*high.intensity",
    r"vigorous.*(?:exercise|physical.*activity)",
    r"aerobic.*(?:exercise|training).*(?:protocol|program|intervention)",
    r"physical.*performance.*(?:test|protocol|requirement)",
    r"exercise.*capacity.*(?:test|required|minimum)",
    r"6.minute.*walk.*(?:test|distance|required)",
    r"physically.*(?:capable|able).*(?:to perform|to complete).*(?:exercise|training)",
    r"able to.*(?:perform|complete|participate).*(?:exercise|training|physical.*activity)",
    r"exercise.*(?:protocol|program|intervention).*(?:required|must)",
]

_PATIENT_FRAILTY_FALL_PATTERNS = [
    r"\bfrail\b",
    r"\bfrailty\b",
    r"recurrent.*falls",
    r"frequent.*falls",
    r"high.*fall.*risk",
    r"wheelchair.*(?:bound|restricted|dependent)",
    r"unable.*to.*walk",
    r"cannot.*walk",
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
        if _any_match(_UNVERIFIABLE_INCLUSION_PATTERNS, criterion.lower()):
            count += 1
    return count



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
    if not patient_has_dbs:
        return None, None

    # If trial is a DBS-candidacy / DBS-indication / DBS-effects study, generic DBS wording in
    # exclusion criteria may refer to surgical contraindications, not existing implants.
    # Only hard-block if exclusion explicitly mentions prior/previous/existing DBS implant/surgery.
    inclusion_text = _text(trial.get("inclusion_criteria", []))
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
        r"lfp\s+sensing",
        r"directional\s+lead",
    ]
    if _any_match(_DBS_CANDIDACY_TRIAL_PATTERNS, inclusion_text):
        # Only block if exclusion explicitly says prior/existing DBS implant excluded
        _EXPLICIT_PRIOR_DBS_EXCLUSION = [
            r"prior.*dbs.*(?:implant|surgery|procedure)",
            r"previous.*dbs.*(?:implant|surgery|procedure)",
            r"existing.*dbs.*(?:implant|hardware)",
            r"dbs.*(?:implant|surgery|procedure).*(?:prior|previous|existing|already)",
            r"already.*(?:implanted|undergone).*dbs",
        ]
        if not _any_match(_EXPLICIT_PRIOR_DBS_EXCLUSION, exclusion_text):
            return None, None

    return "deep brain stimulation (DBS) implant is an exclusion criterion", "DBS implant present"


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
    trial_text = inclusion_text + " " + _text(trial.get("exclusion_criteria", []))

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
        if _any_match(patient_patterns, patient_all_text) and _any_match(trial_inclusion_patterns, inclusion_text):
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


# ---------------------------------------------------------------------------
# New safety blocker checks
# ---------------------------------------------------------------------------

def _check_cognitive_exclusion_general(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when exclusion criteria explicitly exclude dementia/cognitive impairment (no numeric threshold)
    and patient clearly documents dementia or significant cognitive impairment.
    MCI/mild cognitive alone is not sufficient — requires explicit dementia or cognitive impairment."""
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
            ):
                return (
                    "cognitive inclusion requirement: intact cognition or consent capacity required; patient has documented cognitive impairment",
                    "cognitive impairment documented",
                )

    return None, None


def _check_dbs_required(patient: dict, trial: dict) -> tuple[str | None, str | None]:
    """Block when inclusion criteria require DBS but patient has no documented DBS.
    Returns (blocking_criterion, matched_fact) for hard block, or ('unclear', reason) for ambiguous."""
    inclusion_list = trial.get("inclusion_criteria", [])

    # Ambiguous DBS wording: candidacy, effects, neuropsychiatric, scheduled — return unclear not not_eligible
    _AMBIGUOUS_DBS_PATTERNS = [
        r"dbs\s+candidacy",
        r"candidacy.*dbs",
        r"deep brain stimulation\s+candidacy",
        r"dbs\s+(?:neuropsychiatric|effects|programming|optimization|facial|parameters)",
        r"(?:neuropsychiatric|effects|programming|optimization|facial|parameters).*dbs",
        r"lfp\s+sensing",
        r"directional\s+lead",
        r"scheduled\s+to\s+undergo\s+dbs",
        r"dbs.*scheduled",
        r"meets\s+criteria\s+for.*dbs",
        r"criteria\s+for\s+(?:treatment\s+with\s+)?(?:stn.)?dbs",
    ]

    has_hard_requirement = any(
        _any_match(_TRIAL_DBS_REQUIRED_PATTERNS, c.lower()) for c in inclusion_list
    )
    has_ambiguous = any(
        _any_match(_AMBIGUOUS_DBS_PATTERNS, c.lower()) for c in inclusion_list
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

    # Ambiguous only — return unclear signal via uncertain (use special sentinel)
    return (
        "__unclear__:DBS eligibility unclear: trial involves DBS effects/candidacy but patient has no confirmed DBS",
        "no confirmed DBS; ambiguous DBS-related study",
    )


def _check_device_contraindication_stimulation(
    patient: dict, trial: dict
) -> tuple[str | None, str | None]:
    """Block when patient has implanted cardiac device and trial involves transcranial/electrical stimulation."""
    _PACEMAKER_PATTERNS = [
        r"\bpacemaker\b",
        r"cardiac.*pacemaker",
        r"implanted.*cardiac",
        r"cardiac.*device",
        r"implanted.*pacemaker",
        r"implantable.*cardioverter",
        r"\bicd\b",
    ]
    patient_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    if not _any_match(_PACEMAKER_PATTERNS, patient_text):
        return None, None

    # Check all available trial text fields for stimulation keywords
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
    _EXPLICIT_PACEMAKER_EXCLUSION_PATTERNS = [
        r"(?:metal.*implants?.*and.*)?cardiac\s+pacemaker",
        r"pacemaker.*(?:exclusion|excluded|contraindicated|not permitted)",
        r"(?:exclusion|excluded|contraindicated).*pacemaker",
        r"metal.*implants?.*pacemaker",
        r"pacemaker.*metal.*implants?",
    ]
    excl_text = _text(trial.get("exclusion_criteria", []))
    if _any_match(_EXPLICIT_PACEMAKER_EXCLUSION_PATTERNS, excl_text):
        return (
            "hard safety contraindication: patient has implanted cardiac pacemaker which is explicitly excluded",
            "implanted cardiac device present; pacemaker explicitly excluded",
        )

    return None, None


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
    FoG/gait impairment/motor dysfunction alone does NOT count as frailty."""
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
    comorbid_block, comorbid_uncertain, comorbid_fact = _check_comorbidity_protocol_risk(patient, trial)
    if comorbid_block:
        blocking_criteria.append(comorbid_block)
        if comorbid_fact:
            matched_facts.append(comorbid_fact)
    elif comorbid_uncertain:
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
