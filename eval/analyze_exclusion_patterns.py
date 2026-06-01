"""
Task 40: exclusion criterion pattern analysis.

Usage:
    PYTHONPATH=. python eval/analyze_exclusion_patterns.py
"""

import csv
import os
import re
import sys
from collections import defaultdict


INPUT_PATH = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/exclusion_pattern_analysis.md"

DECISION_CANDIDATES = ["decision", "status", "result", "criterion_result", "match_status"]
REASON_CANDIDATES = ["reason", "explanation", "matcher_explanation", "rationale"]

_BLOCKING_DECISIONS = {"not_met", "blocked", "exclusion_met", "failed", "false"}
_BLOCKING_REASON_PHRASES = [
    "blocks eligibility", "exclusion applies", "exclusion criterion met",
    "violates exclusion", "patient excluded",
]
_EXCLUSION_CRITERION_TYPE_WORDS = {"exclusion", "exclude", "excluded"}
_EXCLUSION_TEXT_KEYWORDS = [
    "exclusion", "exclude", "excluded", "not eligible", "contraindication",
    "history of", "prior", "current use", "severe", "unstable", "implanted",
    "pregnancy", "dementia", "pacemaker", "dbs", "deep brain stimulation",
]

_CATEGORIES: list[tuple[str, list[str]]] = [
    ("medication",          ["medication", "drug", "inhibitor", "levodopa", "rasagiline",
                              "dopamine", "carbidopa", "amantadine", "anticholinergic",
                              "mao-b", "comt", "current use", "concomitant"]),
    ("procedure",           ["surgery", "procedure", "transplant", "infusion pump",
                              "implant", "electrode", "stimulation"]),
    ("device",              ["device", "pacemaker", "dbs", "deep brain stimulation",
                              "implanted"]),
    ("cognitive",           ["cognitive", "dementia", "moca", "mmse", "memory",
                              "neuropsychological"]),
    ("diagnosis",           ["diagnosis of", "diagnosed with", "parkinson", "atypical",
                              "multiple system", "secondary", "malignancy", "cancer"]),
    ("age",                 ["age", "years old", "year-old", "aged", "older than",
                              "younger than"]),
    ("safety/comorbidity",  ["severe", "unstable", "cardiac", "renal", "hepatic",
                              "orthostatic", "hypotension", "seizure", "comorbidity",
                              "contraindication"]),
    ("lab/value",           ["creatinine", "hemoglobin", "hematocrit", "bilirubin",
                              "platelet", "albumin", "glucose", "lab", "blood", "serum",
                              "liver", "renal function"]),
    ("reproductive",        ["pregnancy", "pregnant", "breastfeeding", "lactating",
                              "contraception", "childbearing"]),
    ("temporal/prior history", ["history of", "prior", "previous", "within",
                                 "days", "weeks", "months", "years", "washout",
                                 "recent", "past", "last"]),
    ("administrative",      ["consent", "participate", "participation", "protocol",
                              "investigator", "compliance", "enroll", "enrollment"]),
]

MAX_EXAMPLES = 3
TOP_PATIENTS = 10
TOP_TRIALS = 10
TOP_PATTERNS = 20

_NUM_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows in '{path}'.")
    return rows


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def find_first_existing_column(rows: list[dict], candidates: list[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def get_criterion_text(row: dict) -> str:
    for f in ("criterion", "criterion_text", "text"):
        v = row.get(f, "")
        if v:
            return v.strip()
    return ""


def get_decision_value(row: dict) -> str:
    for f in DECISION_CANDIDATES:
        v = row.get(f, "")
        if v:
            return v.strip().lower()
    return ""


def get_reason_text(row: dict) -> str:
    for f in REASON_CANDIDATES:
        v = row.get(f, "")
        if v:
            return v.strip()
    return ""


def normalize_text_pattern(text: str) -> str:
    t = text.strip().lower()
    t = _NUM_RE.sub("<num>", t)
    t = _SPACE_RE.sub(" ", t)
    return t


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def is_exclusion_like(row: dict) -> bool:
    ctype = row.get("criterion_type", "").strip().lower()
    if any(w in ctype for w in _EXCLUSION_CRITERION_TYPE_WORDS):
        return True
    text = get_criterion_text(row).lower()
    return any(kw in text for kw in _EXCLUSION_TEXT_KEYWORDS)


def is_blocking_exclusion(row: dict) -> bool:
    decision = get_decision_value(row)
    if decision in _BLOCKING_DECISIONS:
        return True
    reason = get_reason_text(row).lower()
    return any(phrase in reason for phrase in _BLOCKING_REASON_PHRASES)


def classify_exclusion_category(text: str) -> str:
    t = text.lower()
    for category, keywords in _CATEGORIES:
        if any(kw in t for kw in keywords):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_exclusion_patterns(rows: list[dict]) -> dict:
    total = len(rows)
    excl_rows: list[dict] = []
    blocking_rows: list[dict] = []

    category_counts: dict[str, int] = defaultdict(int)
    blocking_category_counts: dict[str, int] = defaultdict(int)
    patient_blocking: dict[str, int] = defaultdict(int)
    trial_blocking: dict[str, int] = defaultdict(int)
    pattern_counts: dict[str, int] = defaultdict(int)
    category_examples: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        if not is_exclusion_like(row):
            continue
        excl_rows.append(row)

        text = get_criterion_text(row)
        cat = classify_exclusion_category(text)
        category_counts[cat] += 1

        blocking = is_blocking_exclusion(row)
        if blocking:
            blocking_rows.append(row)
            blocking_category_counts[cat] += 1
            pid = row.get("patient_id", "")
            tid = row.get("trial_id", "")
            if pid:
                patient_blocking[pid] += 1
            if tid:
                trial_blocking[tid] += 1

        norm = normalize_text_pattern(text)
        if norm:
            pattern_counts[norm] += 1

        if len(category_examples[cat]) < MAX_EXAMPLES:
            category_examples[cat].append({
                "patient_id": row.get("patient_id", ""),
                "trial_id": row.get("trial_id", ""),
                "decision": get_decision_value(row),
                "criterion_preview": text[:160] + ("…" if len(text) > 160 else ""),
                "reason_preview": get_reason_text(row)[:160],
            })

    top_patients = sorted(patient_blocking.items(), key=lambda x: -x[1])[:TOP_PATIENTS]
    top_trials = sorted(trial_blocking.items(), key=lambda x: -x[1])[:TOP_TRIALS]
    top_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:TOP_PATTERNS]

    return {
        "total": total,
        "excl_count": len(excl_rows),
        "blocking_count": len(blocking_rows),
        "category_counts": dict(category_counts),
        "blocking_category_counts": dict(blocking_category_counts),
        "top_patients": top_patients,
        "top_trials": top_trials,
        "top_patterns": top_patterns,
        "category_examples": dict(category_examples),
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total, 1)}%" if total else "0%"


def format_markdown_report(summary: dict) -> str:
    total = summary["total"]
    excl = summary["excl_count"]
    blocking = summary["blocking_count"]

    lines = [
        "# Exclusion Criterion Pattern Analysis",
        "",
        f"**Total criterion rows:** {total}  ",
        f"**Exclusion-like rows:** {excl} ({_pct(excl, total)})  ",
        f"**Blocking exclusion rows:** {blocking} ({_pct(blocking, excl)}) of exclusion-like rows",
        "",
        "---",
        "",
    ]

    if excl == 0:
        lines += ["No exclusion-like rows were found in the input file.", ""]
        return "\n".join(lines)

    # Category table
    all_cats = sorted(
        set(list(summary["category_counts"].keys()) +
            list(summary["blocking_category_counts"].keys()))
    )
    lines += [
        "### Counts by Exclusion Category",
        "",
        "| Category | Exclusion-like | Blocking |",
        "| --- | --- | --- |",
    ]
    for cat in all_cats:
        ec = summary["category_counts"].get(cat, 0)
        bc = summary["blocking_category_counts"].get(cat, 0)
        lines.append(f"| {cat} | {ec} | {bc} |")
    lines.append("")

    # Top patients
    lines += ["---", "", "### Top Patients by Blocking Exclusions", "",
              "| patient_id | Blocking Exclusions |", "| --- | --- |"]
    for pid, cnt in summary["top_patients"]:
        lines.append(f"| {pid} | {cnt} |")
    lines.append("")

    # Top trials
    lines += ["---", "", "### Top Trials by Blocking Exclusions", "",
              "| trial_id | Blocking Exclusions |", "| --- | --- |"]
    for tid, cnt in summary["top_trials"]:
        lines.append(f"| {tid} | {cnt} |")
    lines.append("")

    # Top patterns
    lines += ["---", "", f"### Top {TOP_PATTERNS} Exclusion Text Patterns (normalized)", "",
              "| # | Pattern | Count |", "| --- | --- | --- |"]
    for i, (pat, cnt) in enumerate(summary["top_patterns"], 1):
        escaped = pat.replace("|", "\\|")
        lines.append(f"| {i} | {escaped} | {cnt} |")
    lines.append("")

    # Examples per category
    lines += ["---", "", "### Example Rows per Category", ""]
    for cat in all_cats:
        examples = summary["category_examples"].get(cat, [])
        if not examples:
            continue
        lines.append(f"#### {cat}")
        lines.append("")
        for e in examples:
            parts = []
            if e["patient_id"]:
                parts.append(f"patient={e['patient_id']}")
            if e["trial_id"]:
                parts.append(f"trial={e['trial_id']}")
            if e["decision"]:
                parts.append(f"decision={e['decision']}")
            meta = " · ".join(parts) or "—"
            lines.append(f"- **{meta}**")
            lines.append(f"  - criterion: {e['criterion_preview']}")
            if e["reason_preview"]:
                lines.append(f"  - reason: {e['reason_preview']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        rows = load_csv_rows(INPUT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_exclusion_patterns(rows)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Rows read          : {summary['total']}")
    print(f"Exclusion-like rows: {summary['excl_count']}")
    print(f"Blocking exclusions: {summary['blocking_count']}")
    print(f"Report             : {REPORT_PATH}")


if __name__ == "__main__":
    main()
