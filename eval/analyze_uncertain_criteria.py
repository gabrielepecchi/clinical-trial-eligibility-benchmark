"""
Task 41: uncertain criterion pattern analysis.

Usage:
    PYTHONPATH=. python eval/analyze_uncertain_criteria.py
"""

import csv
import os
import re
import sys
from collections import defaultdict


INPUT_PATH = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/uncertain_criterion_analysis.md"

DECISION_CANDIDATES = ["decision", "status", "result", "criterion_result", "match_status"]
REASON_CANDIDATES = ["reason", "explanation", "matcher_explanation", "rationale"]

_UNCERTAIN_DECISIONS = {
    "unknown", "uncertain", "unclear", "undetermined",
    "missing_info", "insufficient_information",
}
_UNCERTAIN_REASON_PHRASES = [
    "unclear", "uncertain", "unknown", "missing", "not documented",
    "insufficient", "cannot determine", "not enough information",
    "unavailable", "ambiguous", "cannot evaluate",
]

TOP_PATIENTS = 10
TOP_TRIALS = 10
TOP_PATTERNS = 20
MAX_EXAMPLES = 3

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

def is_uncertain_like(row: dict) -> bool:
    decision = get_decision_value(row)
    if decision in _UNCERTAIN_DECISIONS:
        return True
    reason = get_reason_text(row).lower()
    return any(phrase in reason for phrase in _UNCERTAIN_REASON_PHRASES)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_uncertain_criteria(rows: list[dict]) -> dict:
    total = len(rows)
    uncertain_rows: list[dict] = []

    by_ctype: dict[str, int] = defaultdict(int)
    by_classified: dict[str, int] = defaultdict(int)
    by_patient: dict[str, int] = defaultdict(int)
    by_trial: dict[str, int] = defaultdict(int)
    pattern_counts: dict[str, int] = defaultdict(int)
    ctype_examples: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        if not is_uncertain_like(row):
            continue
        uncertain_rows.append(row)

        ctype = row.get("criterion_type", "").strip() or "unknown"
        classified = row.get("classified_criterion_type", "").strip()
        pid = row.get("patient_id", "")
        tid = row.get("trial_id", "")

        by_ctype[ctype] += 1
        if classified:
            by_classified[classified] += 1
        if pid:
            by_patient[pid] += 1
        if tid:
            by_trial[tid] += 1

        reason = get_reason_text(row)
        norm = normalize_text_pattern(reason) if reason else ""
        if norm:
            pattern_counts[norm] += 1

        if len(ctype_examples[ctype]) < MAX_EXAMPLES:
            text = get_criterion_text(row)
            ctype_examples[ctype].append({
                "patient_id": pid,
                "trial_id": tid,
                "decision": get_decision_value(row),
                "criterion_preview": text[:160] + ("…" if len(text) > 160 else ""),
                "reason_preview": reason[:160],
            })

    return {
        "total": total,
        "uncertain_count": len(uncertain_rows),
        "by_ctype": dict(sorted(by_ctype.items(), key=lambda x: -x[1])),
        "by_classified": dict(sorted(by_classified.items(), key=lambda x: -x[1])),
        "by_patient": dict(sorted(by_patient.items(), key=lambda x: -x[1])),
        "by_trial": dict(sorted(by_trial.items(), key=lambda x: -x[1])),
        "top_patients": sorted(by_patient.items(), key=lambda x: -x[1])[:TOP_PATIENTS],
        "top_trials": sorted(by_trial.items(), key=lambda x: -x[1])[:TOP_TRIALS],
        "top_patterns": sorted(pattern_counts.items(), key=lambda x: -x[1])[:TOP_PATTERNS],
        "ctype_examples": dict(ctype_examples),
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total, 1)}%" if total else "0%"


def format_markdown_report(summary: dict) -> str:
    total = summary["total"]
    unc = summary["uncertain_count"]

    lines = [
        "# Uncertain Criterion Pattern Analysis",
        "",
        f"**Total criterion rows:** {total}  ",
        f"**Uncertain-like rows:** {unc} ({_pct(unc, total)})",
        "",
        "---",
        "",
    ]

    if unc == 0:
        lines += ["No uncertain-like rows were found in the input file.", ""]
        return "\n".join(lines)

    # By criterion_type
    if summary["by_ctype"]:
        lines += ["### Counts by criterion_type", "",
                  "| criterion_type | Count |", "| --- | --- |"]
        for ctype, cnt in summary["by_ctype"].items():
            lines.append(f"| {ctype} | {cnt} |")
        lines.append("")

    # By classified_criterion_type
    if summary["by_classified"]:
        lines += ["### Counts by classified_criterion_type", "",
                  "| classified_criterion_type | Count |", "| --- | --- |"]
        for ctype, cnt in summary["by_classified"].items():
            lines.append(f"| {ctype} | {cnt} |")
        lines.append("")

    # Top patients
    lines += ["---", "", "### Top Patients with Most Uncertain Criteria", "",
              "| patient_id | Uncertain Rows |", "| --- | --- |"]
    for pid, cnt in summary["top_patients"]:
        lines.append(f"| {pid} | {cnt} |")
    lines.append("")

    # Top trials
    lines += ["---", "", "### Top Trials with Most Uncertain Criteria", "",
              "| trial_id | Uncertain Rows |", "| --- | --- |"]
    for tid, cnt in summary["top_trials"]:
        lines.append(f"| {tid} | {cnt} |")
    lines.append("")

    # Top reason patterns
    lines += ["---", "", f"### Top {TOP_PATTERNS} Uncertainty Reason Patterns (normalized)", "",
              "| # | Pattern | Count |", "| --- | --- | --- |"]
    for i, (pat, cnt) in enumerate(summary["top_patterns"], 1):
        escaped = pat.replace("|", "\\|")
        lines.append(f"| {i} | {escaped} | {cnt} |")
    lines.append("")

    # Examples per criterion type
    lines += ["---", "", "### Example Rows per criterion_type", ""]
    for ctype, examples in summary["ctype_examples"].items():
        lines.append(f"#### {ctype}")
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

    summary = analyze_uncertain_criteria(rows)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Rows read       : {summary['total']}")
    print(f"Uncertain rows  : {summary['uncertain_count']}")
    print(f"Report          : {REPORT_PATH}")


if __name__ == "__main__":
    main()
