"""
Task 39: eligibility reason pattern analysis.

Usage:
    PYTHONPATH=. python eval/analyze_eligibility_reasons.py
"""

import csv
import os
import re
import sys
from collections import defaultdict


INPUT_PATH = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/eligibility_reason_patterns.md"

REASON_CANDIDATES = ["reason", "explanation", "matcher_explanation", "rationale"]
DECISION_CANDIDATES = ["decision", "status", "result", "criterion_result", "match_status"]

TOP_OVERALL = 20
TOP_PER_GROUP = 10
MAX_EXAMPLES = 5


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
# Field detection
# ---------------------------------------------------------------------------

def find_first_existing_column(rows: list[dict], candidates: list[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


def normalize_reason(text: str) -> str:
    t = text.strip().lower()
    t = _NUM_RE.sub("<num>", t)
    t = _SPACE_RE.sub(" ", t)
    return t


def preview_text(text: str, max_chars: int = 180) -> str:
    t = text.strip()
    return t[:max_chars] + ("…" if len(t) > max_chars else "")


# ---------------------------------------------------------------------------
# Grouping and counting
# ---------------------------------------------------------------------------

def group_reason_patterns(
    rows: list[dict],
    reason_field: str | None,
    decision_field: str | None,
) -> dict:
    """
    Returns:
      overall_counts: {normalized_reason: count}
      by_decision:    {decision: {normalized_reason: count}}
      examples:       {normalized_reason: [row_dict, ...]}  (up to MAX_EXAMPLES each)
      missing_count:  int
    """
    overall_counts: dict[str, int] = defaultdict(int)
    by_decision: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: dict[str, list[dict]] = defaultdict(list)
    missing_count = 0

    for row in rows:
        raw_reason = row.get(reason_field, "").strip() if reason_field else ""
        decision = (
            row.get(decision_field, "").strip() if decision_field else "unknown_decision"
        ) or "unknown_decision"

        if not raw_reason:
            missing_count += 1
            continue

        norm = normalize_reason(raw_reason)
        overall_counts[norm] += 1
        by_decision[decision][norm] += 1

        if len(examples[norm]) < MAX_EXAMPLES:
            examples[norm].append({
                "patient_id": row.get("patient_id", ""),
                "trial_id": row.get("trial_id", ""),
                "criterion_type": row.get("criterion_type", ""),
                "decision": decision,
                "reason_preview": preview_text(raw_reason),
            })

    return {
        "overall_counts": dict(overall_counts),
        "by_decision": {d: dict(c) for d, c in by_decision.items()},
        "examples": dict(examples),
        "missing_count": missing_count,
    }


def analyze_reason_patterns(rows: list[dict]) -> dict:
    reason_field = find_first_existing_column(rows, REASON_CANDIDATES)
    decision_field = find_first_existing_column(rows, DECISION_CANDIDATES)

    grouped = group_reason_patterns(rows, reason_field, decision_field)

    top_overall = sorted(
        grouped["overall_counts"].items(), key=lambda x: -x[1]
    )[:TOP_OVERALL]

    top_by_decision = {
        decision: sorted(counts.items(), key=lambda x: -x[1])[:TOP_PER_GROUP]
        for decision, counts in grouped["by_decision"].items()
    }

    return {
        "total_rows": len(rows),
        "reason_field": reason_field,
        "decision_field": decision_field,
        "missing_count": grouped["missing_count"],
        "top_overall": top_overall,
        "top_by_decision": top_by_decision,
        "examples": grouped["examples"],
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _examples_block(pattern: str, examples: dict[str, list[dict]]) -> str:
    ex_list = examples.get(pattern, [])
    if not ex_list:
        return ""
    lines = ["", "  *Examples:*", ""]
    for e in ex_list:
        parts = []
        if e["patient_id"]:
            parts.append(f"patient={e['patient_id']}")
        if e["trial_id"]:
            parts.append(f"trial={e['trial_id']}")
        if e["criterion_type"]:
            parts.append(f"type={e['criterion_type']}")
        parts.append(f"decision={e['decision']}")
        meta = " · ".join(parts)
        lines.append(f"  - `{meta}` — {e['reason_preview']}")
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(summary: dict) -> str:
    parts = [
        "# Eligibility Reason Pattern Analysis",
        "",
        f"**Total rows:** {summary['total_rows']}  ",
        f"**Reason field used:** `{summary['reason_field'] or 'none found'}`  ",
        f"**Decision field used:** `{summary['decision_field'] or 'none — grouped as unknown_decision'}`  ",
        f"**Rows with missing reason:** {summary['missing_count']}",
        "",
        "---",
        "",
    ]

    if summary["reason_field"] is None:
        parts += [
            "No reason field was found in the input file.",
            "Expected one of: " + ", ".join(f"`{c}`" for c in REASON_CANDIDATES),
            "",
        ]
        return "\n".join(parts)

    # Top overall
    parts += [f"### Top {TOP_OVERALL} Reason Patterns Overall", ""]
    if summary["top_overall"]:
        parts += ["| # | Pattern | Count |", "| --- | --- | --- |"]
        for i, (pat, cnt) in enumerate(summary["top_overall"], 1):
            escaped = pat.replace("|", "\\|")
            parts.append(f"| {i} | {escaped} | {cnt} |")
        parts.append("")
        # examples for top 5
        parts.append("#### Example rows for top 5 patterns")
        for pat, _ in summary["top_overall"][:5]:
            parts.append(f"\n**`{pat}`**")
            parts.append(_examples_block(pat, summary["examples"]))
    else:
        parts.append("_No patterns found._")
        parts.append("")

    parts += ["---", ""]

    # Per decision
    parts.append(f"### Top {TOP_PER_GROUP} Patterns by Decision / Status")
    parts.append("")
    for decision in sorted(summary["top_by_decision"].keys()):
        patterns = summary["top_by_decision"][decision]
        parts.append(f"#### Decision: `{decision}`")
        parts.append("")
        if patterns:
            parts += ["| # | Pattern | Count |", "| --- | --- | --- |"]
            for i, (pat, cnt) in enumerate(patterns, 1):
                escaped = pat.replace("|", "\\|")
                parts.append(f"| {i} | {escaped} | {cnt} |")
            parts.append("")
        else:
            parts.append("_No patterns._")
            parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        rows = load_csv_rows(INPUT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_reason_patterns(rows)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Rows read      : {summary['total_rows']}")
    print(f"Reason field   : {summary['reason_field'] or 'none found'}")
    print(f"Decision field : {summary['decision_field'] or 'none found'}")
    print(f"Missing reasons: {summary['missing_count']}")
    print(f"Report         : {REPORT_PATH}")


if __name__ == "__main__":
    main()
