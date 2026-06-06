"""Clinical term constants and synonym helpers for rule_matcher.py.

Contains all module-level pattern lists, compiled regex constants,
and simple stateless helper functions used by rule_matcher.py.
"""

import re


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


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
_MAOB_DRUGS = [
    r"\brasagiline\b", r"\bazilect\b",
    r"\bselegiline\b", r"\bdeprenyl\b", r"\beldepryl\b", r"\bzelapar\b",
    r"\bsafinamide\b", r"\bxadago\b",
]
_MAOB_NEGATION_PATTERN = re.compile(
    r"\bno\b.{0,40}(?:mao.?b|rasagiline|azilect|selegiline|deprenyl|eldepryl|safinamide|xadago)",
    re.IGNORECASE,
)


def _has_negated_dbs(text: str) -> bool:
    return bool(_DBS_NEGATION_PATTERN.search(text))


_PACEMAKER_NEGATION_PATTERN = re.compile(
    r"(?:\bno\b|\bdenies?\b|\bwithout\b|\bno\s+history\s+of|\bno\s+prior|\bno\s+evidence\s+of|\bnot\b|\babsent\b)"
    r".{0,40}(?:pacemaker|cardiac\s+(?:device|implant|pacemaker|defibrillator)"
    r"|\bicd\b|implanted\s+cardiac|implantable\s+cardioverter"
    r"|cardioverter.defibrillator|cardiac.*implant|implanted.*defibrillator)",
    re.IGNORECASE,
)


def _has_negated_pacemaker(text: str) -> bool:
    """Return True when text explicitly negates the presence of a pacemaker or implanted cardiac device."""
    return bool(_PACEMAKER_NEGATION_PATTERN.search(text))


def _has_maob_inhibitor(text: str) -> bool:
    if _has_negated_maob(text):
        return False
    return _any_match(_MAOB_DRUGS, text)


def _has_negated_maob(text: str) -> bool:
    # Use the broader negation pattern from _NEGATION_PATTERNS which handles
    # "no", "denies", "not", "without", "no history of", "no prior", etc.
    broad_pat = _NEGATION_PATTERNS.get("maob_inhibitor")
    if broad_pat is not None and broad_pat.search(text):
        return True
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
    r"medication.*history.*unclear",
    r"unclear.*medication.*history",
    r"medication.*stability.*unclear",
    r"unclear.*medication.*stability",
    r"medication.*stability.*not.*documented",
    r"stability.*not.*documented",
    r"duration.*not.*documented",
    r"medication.*duration.*unclear",
    r"unclear.*medication.*duration",
    r"history.*unclear",
    r"unclear.*history",
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
    r"medication.*history.*unclear",
    r"unclear.*medication.*history",
    r"medication.*stability.*unclear",
    r"unclear.*medication.*stability",
    r"medication.*stability.*not.*documented",
    r"stability.*not.*documented",
    r"duration.*not.*documented",
    r"medication.*duration.*unclear",
    r"unclear.*medication.*duration",
    r"history.*unclear",
    r"unclear.*history",
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
        # FoG / gait patient in a FoG / gait / rehab / balance / cueing / exercise trial
        [r"freezing.*gait", r"\bfog\b", r"gait.*disturbance", r"gait.*impairment",
         r"gait.*difficulty", r"shuffling.*gait", r"festination"],
        [r"freezing.*gait", r"\bfog\b", r"gait.*study", r"gait.*trial",
         r"somatosensory.*stimulation", r"auditory.*cue", r"gait.*cueing",
         r"gait", r"balance", r"rehabilitation", r"physiotherapy", r"physical.*therapy",
         r"exercise.*trial", r"exercise.*study", r"motor.*function",
         r"lower.*limb", r"lower.*extremity", r"virtual.*reality", r"\bvr\b.*trial",
         r"cueing", r"stepping", r"treadmill", r"fall.*prevention", r"fall.*risk",
         r"mobility", r"locomotion", r"walking"],
    ),
    (
        # DBS-implanted patient in a DBS-specific outcomes/effects/implanted-patient study
        [r"\bdbs\b", r"deep brain stimulation", r"subthalamic", r"\bstn\b"],
        [r"dbs.*effect", r"effect.*dbs", r"dbs.*outcome", r"dbs.*patient", r"dbs.*implant",
         r"deep brain stimulation.*effect", r"lfp.*sensing", r"directional.*lead",
         r"dbs.*facial", r"dbs.*neuropsychiatric", r"dbs.*programming",
         r"subthalamic.*steering", r"dbs.*optimization",
         r"undergone.*dbs", r"dbs.*surgery", r"subthalamic.*dbs",
         r"stn.*surgery", r"subthalamic nucleus.*dbs",
         r"stimulation.*parameter", r"dbs.*follow.up"],
    ),
    (
        # Frail patient in a frailty / home physiotherapy trial
        [r"\bfrail", r"frailty"],
        [r"frailty.*trial", r"frailty.*study", r"frail.*patient",
         r"home.*physiotherapy", r"home.*physical.*therapy",
         r"frailty.*intervention", r"frailty.*rehabilitation"],
    ),
    (
        # Depression / RBD / autonomic / non-motor features in explicitly non-motor/neuropsychiatric/QoL/phenotype trials
        [r"\bdepression\b", r"\bdepressed\b", r"rem.*sleep.*behavior", r"\brbd\b",
         r"rem.*behavior.*disorder", r"autonomic.*dysfunction", r"autonomic.*failure",
         r"orthostatic.*hypotension", r"postural.*hypotension"],
        [r"non.motor", r"neuropsychiatric", r"neuropsychological", r"quality.*of.*life",
         r"\bqol\b", r"pd.*phenotype", r"parkinson.*phenotype", r"phenotype.*study",
         r"depression.*trial", r"depression.*study", r"anxiety.*trial", r"sleep.*study",
         r"sleep.*trial", r"autonomic.*study", r"autonomic.*trial",
         r"rem.*behavior", r"\brbd\b.*trial", r"\brbd\b.*study",
         r"dementia.*evaluation", r"prodromal", r"non.motor.*symptom",
         r"neuropsychiatric.*symptom", r"quality.*life.*parkinson"],
    ),
]

# Hard safety contraindications: (patient patterns, trial procedure patterns).
# If matched, add a blocking criterion → not_eligible.
_HARD_CONTRAINDICATION_PAIRS: list[tuple[list[str], list[str]]] = [
    (
        # Implanted cardiac device + electrical brain stimulation
        [r"\bpacemaker\b", r"cardiac.*pacemaker", r"implanted.*cardiac",
         r"cardiac.*device", r"implanted.*pacemaker"],
        [r"\brtms\b", r"\btms\b", r"\btdcs\b", r"\btacs\b",
         r"transcranial.*magnetic", r"transcranial.*electrical",
         r"transcranial.*alternating.*current",
         r"alternating.*current.*stimulation",
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
    r"lfp.*sensing.*(?:from|via|using).*(?:directional|lead|electrode)",
    r"directional.*lead.*(?:hardware|implant|dbs)",
    r"existing.*dbs.*hardware",
    r"implanted.*dbs.*patient",
]

# Ambiguous DBS protocol patterns: downgrade to unclear, not not_eligible.
_AMBIGUOUS_DBS_INCLUSION_PATTERNS = [
    r"dbs\s+candidacy",
    r"candidacy.*dbs",
    r"deep brain stimulation\s+candidacy",
    r"dbs\s+(?:effects?|outcomes?|programming|optimization|facial|parameters?|neuropsychiatric)",
    r"(?:effects?|outcomes?|programming|optimization|facial|parameters?|neuropsychiatric).*dbs",
    r"scheduled\s+to\s+undergo\s+dbs",
    r"dbs.*scheduled",
    r"meets\s+criteria\s+for.*dbs",
    r"criteria\s+for\s+(?:treatment\s+with\s+)?(?:stn.)?dbs",
    r"lfp\s+sensing",
    r"directional\s+lead",
    r"electrophysiology.*(?:stn|subthalamic)",
    r"(?:stn|subthalamic).*(?:recording|electrophysiology)",
    r"compatible.*(?:dbs|hardware)",
    r"(?:dbs|hardware).*compatible",
    r"mri.compatible.*dbs",
    r"dbs.*mri.compatible",
]

# Broader transcranial/electrical stimulation patterns for device contraindication.
# Intentionally excludes generic "brain stimulation", "electrical stimulation", or rehab wording.
_TRIAL_STIMULATION_PATTERNS = [
    r"\brtms\b",
    r"\btms\b",
    r"\btdcs\b",
    r"\btacs\b",
    r"transcranial.*magnetic",
    r"transcranial.*electric",
    r"transcranial.*direct.*current",
    r"transcranial.*alternating.*current",
    r"alternating.*current.*stimulation",
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

_HEALTHY_CONTROL_TRIAL_PATTERNS = [
    r"healthy.*control", r"control.*group", r"healthy.*volunteer",
    r"comparator.*group", r"reference.*group", r"matched.*control",
    r"age.matched.*control", r"control.*subject", r"healthy.*subject",
    r"non.pd.*control", r"control.*arm",
]

_HEALTHY_CONTROL_AMBIGUITY_SIGNALS = [
    r"control.*group", r"healthy.*control", r"comparator", r"age.matched.*control",
    r"imaging.*cohort", r"biomarker.*cohort", r"observational.*cohort",
]

_PATIENT_HEALTHY_CONTROL_PATTERNS = [
    r"healthy.*(?:control|volunteer)",
    r"(?:control|volunteer).*healthy",
    r"no.*neurological.*diagnosis",
    r"neurologically.*healthy",
    r"healthy.*subject",
    r"no.*parkinson",
    r"no.*neurological.*disease",
    r"no.*neurological.*condition",
]

_INTERVENTIONAL_PD_ONLY_PATTERNS = [
    r"\bstimulation\b", r"\brehabilitation\b", r"\bexercise\b", r"\btreadmill\b",
    r"\btraining\b", r"\bintervention\b", r"\btreatment\b", r"\btherapy\b",
    r"\bdbs\b", r"deep brain stimulation", r"\brtms\b", r"\btms\b", r"\btdcs\b",
    r"\btacs\b", r"transcranial", r"neuroprotection", r"drug.*trial",
    r"medication.*trial", r"randomized", r"randomised", r"placebo",
    r"double.blind", r"single.blind", r"open.label.*treatment",
]

_FOG_GAIT_TRIAL_PATTERNS = [
    r"freezing.*of.*gait", r"\bfog\b", r"gait.*disturbance", r"gait.*impairment",
    r"gait.*disorder", r"auditory.*cue", r"cueing.*gait", r"walking.*impairment",
    r"balance.*impairment", r"gait.*specific", r"gait.*intervention",
    r"gait.*rehabilitation", r"gait.*training",
]
_FOG_GAIT_PATIENT_PATTERNS = [
    r"freezing.*gait", r"\bfog\b", r"gait.*disturbance", r"gait.*impairment",
    r"gait.*difficulty", r"shuffling.*gait", r"festination", r"balance.*impairment",
    r"walking.*impairment", r"gait.*disorder",
]

_COG_MCI_TRIAL_PATTERNS = [
    r"\bpd.mci\b", r"mild.*cognitive.*impairment", r"cognitive.*impairment",
    r"moca.*threshold", r"mmse.*threshold", r"moca\s*[<≤>=]+",
    r"mmse\s*[<≤>=]+", r"cognitive.*training", r"cognitive.*telerehabilitation",
    r"cognitive.*motor.*training", r"cognitive.*rehabilitation",
]
_COG_MCI_PATIENT_PATTERNS = [
    r"\bmci\b", r"mild.*cognitive", r"cognitive.*impairment", r"mmse.*\d",
    r"moca.*\d", r"\d+.*mmse", r"\d+.*moca", r"cognitive.*score",
    r"dementia",
]

_SEVERITY_TRIAL_PATTERNS = [
    r"hoehn.*(?:and|&).*yahr", r"\bh&y\b", r"\bhy\b\s*stage",
    r"\bupdrs\b", r"\bmds.updrs\b", r"disease.*duration",
    r"disease.*stage", r"pd.*stage", r"advanced.*pd", r"advanced.*parkinson",
    r"early.*pd", r"early.*parkinson", r"motor.*fluctuation", r"off.*time",
    r"\bdyskinesia\b", r"\blcig\b", r"intestinal.*gel", r"wearing.off",
]
_SEVERITY_PATIENT_PATTERNS = [
    r"hoehn.*yahr", r"\bh&y\b", r"\bhy\b.*stage", r"\bupdrs\b",
    r"disease.*duration", r"disease.*stage", r"advanced.*pd",
    r"advanced.*parkinson", r"early.*pd", r"motor.*fluctuation",
    r"off.*time", r"\bdyskinesia\b", r"\blcig\b", r"intestinal.*gel",
    r"wearing.off", r"years.*parkinson", r"parkinson.*years",
    r"diagnosed.*\d+.*year", r"\d+.*year.*pd", r"stage\s+[1-5]",
]

_MED_SPECIFIC_TRIAL_PATTERNS = [
    r"levodopa.*response", r"stable.*levodopa", r"levodopa.*stable",
    r"botulinum.*toxin", r"\bbtx\b", r"rotigotine", r"dopamine.*agonist",
    r"comt.*inhibitor", r"mao.b.*inhibitor", r"medication.*free",
    r"drug.*naive", r"medication.*washout", r"washout.*period",
    r"prior.*medication.*exposure", r"medication.*exposure",
]
_MED_DOCUMENTED_PATIENT_PATTERNS = [
    r"levodopa", r"carbidopa", r"rotigotine", r"pramipexole", r"ropinirole",
    r"rasagiline", r"selegiline", r"safinamide", r"entacapone", r"opicapone",
    r"amantadine", r"botulinum", r"medication.*stable", r"stable.*medication",
    r"drug.*naive", r"medication.*free", r"no.*medication",
]

_LANG_SCALE_TRIAL_PATTERNS = [
    r"\burdu\b", r"\bspanish\b", r"\bitalian\b", r"\bfrench\b", r"\bgerman\b",
    r"\bportugues", r"\bchinese\b", r"\bjapanese\b", r"\bkorean\b",
    r"questionnaire.*validation", r"scale.*validation", r"translated.*version",
    r"translation.*stud", r"language.*specific", r"linguistic.*validation",
]
_LANG_PATIENT_PATTERNS = [
    r"speak.*\w+", r"language.*\w+", r"\w+.*speaking", r"native.*language",
    r"fluent", r"literate",
]

_FRAILTY_TARGET_SUPPRESSION_PATTERNS = [
    r"frailty.*trial", r"frailty.*study", r"frail.*patient",
    r"home.*physiotherapy", r"home.*physical.*therapy",
    r"frailty.*rehabilitation", r"frailty.*intervention",
    r"elderly.*frail", r"frail.*elderly",
    r"fall.prevention.*rehabilitation", r"fall.risk.*rehabilitation",
    r"rehabilitation.*fall.prevention", r"rehabilitation.*fall.risk",
]

_RBD_TARGET_SUPPRESSION_PATTERNS = [
    r"non.motor.*symptom", r"non.motor.*pd", r"pd.*non.motor",
    r"non.motor.*parkinson", r"parkinson.*non.motor",
    r"sleep.*phenotype", r"pd.*phenotype", r"parkinson.*phenotype",
    r"dementia.*evaluation", r"neuropsychological.*evaluation",
    r"prodromal.*pd", r"prodromal.*parkinson",
    r"\brbd\b.*study", r"\brbd\b.*trial",
    r"rem.*behavior.*study", r"rem.*behavior.*trial",
]

_RBD_AMBIGUITY_TRIGGER_PATTERNS = [
    r"neuropsychiatric.*protocol", r"protocol.*neuropsychiatric",
    r"psychiatric.*exclusion", r"exclusion.*psychiatric",
    r"sleep.*exclusion", r"exclusion.*sleep",
    r"rem.*sleep.*behavior.*disorder.*exclusion",
    r"rbd.*exclusion", r"exclusion.*rbd",
    r"protocol.*safety.*neuropsychiatric", r"neuropsychiatric.*safety.*protocol",
    r"patient.reported.*neuropsychiatric.*outcome",
]

_DEPRESSION_IMAGING_BIOMARKER_PATTERNS = [
    r"\bpet\b", r"\b18f\b", r"florbetapir", r"\bdtbz\b",
    r"imaging.*biomarker", r"biomarker.*imaging",
    r"molecular.*imaging", r"neuroimaging.*biomarker",
    r"biomarker.*neuroimaging",
]

_ACTIVE_CANCER_PATIENT_PATTERNS = [
    r"active.*cancer", r"cancer.*active", r"current.*chemotherapy",
    r"ongoing.*chemotherapy", r"active.*malignancy", r"malignancy.*active",
    r"active.*tumor", r"active.*tumour", r"cancer.*treatment.*ongoing",
    r"undergoing.*cancer.*treatment",
]

_ACTIVE_CANCER_TRIAL_PATTERNS = [
    r"gait", r"stability", r"balance", r"rehabilitation",
    r"exercise", r"neuroprotection", r"neuroprotective",
    r"surgery", r"stimulation", r"safety.*sensitive",
    r"cardiovascular", r"hepatic", r"tolerability",
]


# ---------------------------------------------------------------------------
# Medication synonym groups
# ---------------------------------------------------------------------------

# Each group: (canonical_name, [regex patterns for all synonyms])
_MED_SYNONYM_GROUPS: list[tuple[str, list[str]]] = [
    (
        "levodopa",
        [
            r"\blevodopa\b",
            r"\bl.dopa\b",
            r"\bldopa\b",
            r"carbidopa.levodopa",
            r"levodopa.carbidopa",
            r"co.careldopa",
            r"\bsinemet\b",
            r"\bduodopa\b",
            r"\bkynmobi\b",
            r"\binbrija\b",
            r"\brytary\b",
        ],
    ),
    (
        "dopamine_agonist",
        [
            r"\bpramipexole\b",
            r"\bmirapex\b",
            r"\bropinirole\b",
            r"\brequip\b",
            r"\brotigotine\b",
            r"\bneupro\b",
            r"\bapomorphine\b",
            r"\bkynmobi\b",
            r"dopamine.*agonist",
            r"agonist.*dopamine",
        ],
    ),
    (
        "maob_inhibitor",
        [
            r"\brasagiline\b",
            r"\bazilect\b",
            r"\bselegiline\b",
            r"\bdeprenyl\b",
            r"\beldepryl\b",
            r"\bzelapar\b",
            r"\bsafinamide\b",
            r"\bxadago\b",
            r"mao.b inhibitor",
            r"monoamine oxidase.*b.*inhibitor",
        ],
    ),
    (
        "comt_inhibitor",
        [
            r"\bentacapone\b",
            r"\bcomtan\b",
            r"\bopicapone\b",
            r"\bongentys\b",
            r"\btolcapone\b",
            r"\btasmar\b",
            r"comt inhibitor",
            r"comt.inhibitor",
        ],
    ),
    (
        "amantadine",
        [
            r"\bamantadine\b",
            r"\bgocovri\b",
            r"\bosmolex\b",
            r"\bsymmetrel\b",
        ],
    ),
]

# Flat lookup: canonical → list of patterns
_MED_SYNONYMS: dict[str, list[str]] = {
    canonical: patterns for canonical, patterns in _MED_SYNONYM_GROUPS
}


def _patient_has_med_class(patient_text: str, canonical: str) -> bool:
    """Return True if patient text contains any synonym for the given medication class."""
    if canonical == "maob_inhibitor" and _has_negated_maob(patient_text):
        return False
    patterns = _MED_SYNONYMS.get(canonical, [])
    return _any_match(patterns, patient_text)


def _trial_requires_med_class(trial_text: str, canonical: str) -> bool:
    """Return True if trial text references any synonym for the given medication class."""
    patterns = _MED_SYNONYMS.get(canonical, [])
    return _any_match(patterns, trial_text)


def _normalize_med_text(text: str) -> str:
    """Return text with medication synonyms replaced by their canonical form.

    Useful for downstream pattern matching that uses canonical drug names only.
    """
    result = text
    for canonical, patterns in _MED_SYNONYM_GROUPS:
        for p in patterns:
            result = re.sub(p, canonical, result, flags=re.IGNORECASE)
    return result


# ---------------------------------------------------------------------------
# Procedure / device synonym groups
# ---------------------------------------------------------------------------

_PROCEDURE_SYNONYM_GROUPS: list[tuple[str, list[str]]] = [
    (
        "dbs",
        [
            r"\bdbs\b",
            r"deep brain stimulation",
            r"subthalamic.*stimulation",
            r"stn.*dbs",
            r"dbs.*stn",
            r"subthalamic nucleus.*stimulation",
        ],
    ),
    (
        "mri",
        [
            r"\bmri\b",
            r"\bfmri\b",
            r"functional\s+mri",
            r"magnetic resonance imaging",
            r"functional magnetic resonance",
            r"neuroimaging",
        ],
    ),
    (
        "tms",
        [
            r"\btms\b",
            r"\brtms\b",
            r"transcranial magnetic stimulation",
            r"repetitive transcranial magnetic",
        ],
    ),
    (
        "tdcs",
        [
            r"\btdcs\b",
            r"\btacs\b",
            r"transcranial direct current stimulation",
            r"transcranial alternating current stimulation",
            r"transcranial.*direct.*current",
            r"transcranial.*alternating.*current",
        ],
    ),
    (
        "pacemaker",
        [
            r"\bpacemaker\b",
            r"implanted cardiac device",
            r"cardiac.*pacemaker",
            r"implanted.*pacemaker",
            r"\bicd\b",
            r"implantable cardioverter",
            r"implantable.*cardioverter.defibrillator",
            r"cardioverter.defibrillator",
            r"cardiac.*implant",
            r"implanted.*cardiac",
            r"cardiac.*defibrillator",
            r"implanted.*defibrillator",
        ],
    ),
    (
        "lcig",
        [
            r"\blcig\b",
            r"levodopa.carbidopa intestinal gel",
            r"intestinal gel infusion",
            r"intestinal gel",
            r"\bduodopa\b",
            r"\bduopa\b",
        ],
    ),
]

_PROCEDURE_SYNONYMS: dict[str, list[str]] = {
    canonical: patterns for canonical, patterns in _PROCEDURE_SYNONYM_GROUPS
}


def _patient_has_procedure(patient_text: str, canonical: str) -> bool:
    """Return True if patient text contains any synonym for the given procedure/device.

    Respects existing negation helpers where available (e.g. DBS, pacemaker).
    """
    if canonical == "dbs" and _has_negated_dbs(patient_text):
        return False
    if canonical == "pacemaker" and _has_negated_pacemaker(patient_text):
        return False
    patterns = _PROCEDURE_SYNONYMS.get(canonical, [])
    return any(re.search(p, patient_text, re.IGNORECASE) for p in patterns)


def _trial_involves_procedure(trial_text: str, canonical: str) -> bool:
    """Return True if trial text references any synonym for the given procedure/device."""
    patterns = _PROCEDURE_SYNONYMS.get(canonical, [])
    return any(re.search(p, trial_text, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Task 8: General negation and contradiction helpers
# ---------------------------------------------------------------------------

# Negation prefixes: "no", "denies", "not", "without", "no history of", etc.
_NEGATION_PREFIX = r"(?:no\b|denies?|not\b|without\b|no\s+history\s+of|no\s+prior|no\s+evidence\s+of|ruled?\s+out|free\s+of|absence\s+of)"

# Clause-boundary chars: comma, semicolon, period, "but", "however", "although", "yet"
_NO_CLAUSE_BREAK = r"(?:(?!,|;|\.|but\b|however\b|although\b|yet\b).)"

# Per-topic negation patterns — do NOT cross clause boundaries
_NEGATION_PATTERNS: dict[str, re.Pattern] = {
    "dbs": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:dbs|deep\s+brain\s+stimulation|subthalamic\s+stimulation)",
        re.IGNORECASE,
    ),
    "maob_inhibitor": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:mao.?b|rasagiline|azilect|selegiline|deprenyl|eldepryl|zelapar|safinamide|xadago)",
        re.IGNORECASE,
    ),
    "cognitive_impairment": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:cognitive\s+impairment|dementia|memory\s+impairment|mci|cognitive\s+decline)",
        re.IGNORECASE,
    ),
    "active_cancer": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:active\s+cancer|cancer|malignancy|tumor|tumour|chemotherapy|oncology)"
        rf"|(?:cancer|malignancy)\s+(?:ruled?\s+out|excluded|resolved|in\s+remission|not\s+active)",
        re.IGNORECASE,
    ),
    "investigational_drug": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:investigational\s+drug|experimental\s+drug|investigational\s+agent)",
        re.IGNORECASE,
    ),
    "trial_participation": re.compile(
        rf"{_NEGATION_PREFIX}{_NO_CLAUSE_BREAK}{{0,40}}(?:clinical\s+trial|investigational\s+study|trial\s+participation)",
        re.IGNORECASE,
    ),
}

# Per-topic positive (affirmative) patterns — presence of topic
_POSITIVE_PATTERNS: dict[str, list[str]] = {
    "dbs": [r"\bdbs\b", r"deep\s+brain\s+stimulation", r"subthalamic\s+(?:nucleus\s+)?stimulation", r"\bstn\b"],
    "maob_inhibitor": [r"\brasagiline\b", r"\bazilect\b", r"\bselegiline\b", r"\bdeprenyl\b", r"\beldepryl\b", r"\bzelapar\b", r"\bsafinamide\b", r"\bxadago\b", r"mao.?b\s+inhibitor"],
    "cognitive_impairment": [r"cognitive\s+impairment", r"\bdementia\b", r"\bmci\b", r"memory\s+impairment", r"cognitive\s+decline"],
    "active_cancer": [r"active\s+cancer", r"active\s+malignancy", r"current\s+chemotherapy", r"currently\s+receiving\s+chemotherapy", r"ongoing\s+chemotherapy", r"active\s+tumor", r"receiving\s+chemotherapy"],
    "investigational_drug": [r"investigational\s+drug", r"experimental\s+drug", r"investigational\s+agent"],
    "trial_participation": [r"clinical\s+trial", r"investigational\s+study", r"enrolled\s+in\s+(?:a\s+)?(?:trial|study)"],
}


def is_negated(text: str, topic: str) -> bool:
    """Return True if the topic is clearly negated in text."""
    pat = _NEGATION_PATTERNS.get(topic)
    if pat is None:
        return False
    return bool(pat.search(text))


def is_affirmed(text: str, topic: str) -> bool:
    """Return True if the topic is positively mentioned in text (ignoring negation)."""
    pats = _POSITIVE_PATTERNS.get(topic, [])
    return _any_match(pats, text)


def _negation_spans(text: str, topic: str) -> list[tuple[int, int]]:
    """Return (start, end) spans covering all keyword matches within each negation match."""
    pat = _NEGATION_PATTERNS.get(topic)
    if pat is None:
        return []
    keyword_pats = _POSITIVE_PATTERNS.get(topic, [])
    spans: list[tuple[int, int]] = []
    for m in pat.finditer(text):
        seg = m.group()
        seg_start = m.start()
        latest_end: int | None = None
        for kp in keyword_pats:
            for km in re.finditer(kp, seg, re.IGNORECASE):
                kend = seg_start + km.end()
                if latest_end is None or kend > latest_end:
                    latest_end = kend
        spans.append((seg_start, latest_end if latest_end is not None else m.end()))
    return spans


def has_non_negated_affirmation(text: str, topic: str) -> bool:
    """Return True if there is a positive mention of topic NOT covered by any negation span.

    A positive match is considered covered (negated) if its start position falls
    within any negation span.  Pure negations like "no dementia" produce only
    negation spans; they are NOT treated as affirmed.
    """
    pats = _POSITIVE_PATTERNS.get(topic, [])
    if not pats:
        return False
    spans = _negation_spans(text, topic)
    for p in pats:
        for m in re.finditer(p, text, re.IGNORECASE):
            pos = m.start()
            if not any(s <= pos < e for s, e in spans):
                return True
    return False


def has_contradiction(text: str, topic: str) -> bool:
    """Return True if text contains both a negation AND a separate non-negated affirmation for topic.

    A pure negation ("no dementia", "cancer ruled out") returns False because
    the only positive keyword match is inside the negation span itself.
    True contradictions ("denies dementia, dementia documented by neurologist") return True
    because there is a positive mention outside the negation span.
    """
    return is_negated(text, topic) and has_non_negated_affirmation(text, topic)
