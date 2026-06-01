"""
Task 29: criterion length and complexity analysis.

Usage:
    PYTHONPATH=. python eval/analyze_criterion_lengths.py
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Any


INPUT_PATH = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/criterion_length_analysis.md"

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
# Text helpers
# ---------------------------------------------------------------------------

def get_criterion_text(row: dict) -> str:
    for field in ("criterion", "criterion_text", "text"):
        val = row.get(field, "")
        if val:
            return val.strip()
    return ""


def count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def compute_basic_stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    return {
        "min": int(s[0]),
        "max": int(s[-1]),
        "mean": round(sum(s) / n, 1),
        "median": round(median, 1),
    }


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_length(char_count: int) -> str:
    if char_count <= 50:
        return "0–50"
    if char_count <= 150:
        return "51–150"
    if char_count <= 300:
        return "151–300"
    if char_count <= 600:
        return "301–600"
    return ">600"


def bucket_word_count(word_count: int) -> str:
    if word_count <= 10:
        return "0–10"
    if word_count <= 25:
        return "11–25"
    if word_count <= 50:
        return "26–50"
    if word_count <= 100:
        return "51–100"
    return ">100"


# ---------------------------------------------------------------------------
# Complexity signals
# ---------------------------------------------------------------------------

_NUMERIC_RE_CHARS = set("0123456789")
_AGE_KEYWORDS = ["age", "years old", "year-old", "aged"]
_MED_KEYWORDS = ["medication", "drug", "inhibitor", "therapy", "levodopa",
                  "rasagiline", "dopamine", "carbidopa", "amantadine",
                  "anticholinergic", "mao-b", "comt"]
_PROC_KEYWORDS = ["surgery", "stimulation", "dbs", "device", "implant",
                   "procedure", "electrode", "transplant", "infusion pump"]
_COG_KEYWORDS = ["cognitive", "dementia", "moca", "mmse", "memory",
                  "neuropsychological", "confusion"]
_LAB_KEYWORDS = ["creatinine", "hemoglobin", "hematocrit", "liver",
                  "renal", "hepatic", "bilirubin", "platelet", "wbc",
                  "albumin", "glucose", "lab", "blood", "serum"]
_TEMPORAL_KEYWORDS = ["within", "prior to", "at least", "days", "weeks",
                       "months", "years", "duration", "washout", "history of",
                       "recent", "past", "last", "before", "after"]


def _contains_any(text_lower: str, keywords: list[str]) -> bool:
    return any(kw in text_lower for kw in keywords)


def detect_complexity_signals(text: str) -> dict[str, bool]:
    t = text.lower()
    has_numeric = any(c in _NUMERIC_RE_CHARS for c in t)
    clause_count = t.count(";") + t.count(",") + t.count(" and ") + t.count(" or ")
    return {
        "numeric_threshold": has_numeric,
        "age_criterion": _contains_any(t, _AGE_KEYWORDS),
        "medication_keyword": _contains_any(t, _MED_KEYWORDS),
        "procedure_device_keyword": _contains_any(t, _PROC_KEYWORDS),
        "cognitive_keyword": _contains_any(t, _COG_KEYWORDS),
        "lab_keyword": _contains_any(t, _LAB_KEYWORDS),
        "temporal_keyword": _contains_any(t, _TEMPORAL_KEYWORDS),
        "multiple_clauses": clause_count >= 3,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_rows(rows: list[dict]) -> dict:
    char_lengths: list[int] = []
    word_counts: list[int] = []
    length_buckets: dict[str, int] = defaultdict(int)
    word_buckets: dict[str, int] = defaultdict(int)
    signal_totals: dict[str, int] = defaultdict(int)

    by_ctype: dict[str, list[int]] = defaultdict(list)
    by_classified: dict[str, list[int]] = defaultdict(list)

    top_rows: list[dict] = []

    for row in rows:
        text = get_criterion_text(row)
        cl = len(text)
        wc = count_words(text)
        char_lengths.append(cl)
        word_counts.append(wc)
        length_buckets[bucket_length(cl)] += 1
        word_buckets[bucket_word_count(wc)] += 1

        signals = detect_complexity_signals(text)
        for k, v in signals.items():
            if v:
                signal_totals[k] += 1

        ctype = row.get("criterion_type", "").strip()
        if ctype:
            by_ctype[ctype].append(cl)

        classified = row.get("classified_criterion_type", "").strip()
        if classified:
            by_classified[classified].append(cl)

        top_rows.append({
            "patient_id": row.get("patient_id", ""),
            "trial_id": row.get("trial_id", ""),
            "criterion_type": ctype,
            "length": cl,
            "words": wc,
            "preview": text[:120] + ("…" if len(text) > 120 else ""),
        })

    top_rows.sort(key=lambda r: -r["length"])
    top10 = top_rows[:10]

    # grouping stats
    ctype_stats = {k: compute_basic_stats(v) for k, v in sorted(by_ctype.items())}
    classified_stats = {k: compute_basic_stats(v) for k, v in sorted(by_classified.items())}

    return {
        "total": len(rows),
        "char_stats": compute_basic_stats(char_lengths),
        "word_stats": compute_basic_stats(word_counts),
        "length_buckets": dict(length_buckets),
        "word_buckets": dict(word_buckets),
        "signal_totals": dict(signal_totals),
        "ctype_stats": ctype_stats,
        "classified_stats": classified_stats,
        "top10": top10,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _stats_table(stats: dict) -> str:
    return (
        "| Stat | Value |\n| --- | --- |\n"
        f"| Min | {stats['min']} |\n"
        f"| Max | {stats['max']} |\n"
        f"| Mean | {stats['mean']} |\n"
        f"| Median | {stats['median']} |\n"
    )


def _count_table(title: str, counts: dict[str, int], ordered_keys: list[str]) -> str:
    lines = [f"### {title}", "", "| Bucket | Count |", "| --- | --- |"]
    for k in ordered_keys:
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    lines.append("")
    return "\n".join(lines)


def _group_stats_table(title: str, group_stats: dict[str, dict]) -> str:
    if not group_stats:
        return ""
    lines = [f"### {title}", "",
             "| Type | Min | Max | Mean | Median |",
             "| --- | --- | --- | --- | --- |"]
    for k, s in group_stats.items():
        lines.append(f"| {k} | {s['min']} | {s['max']} | {s['mean']} | {s['median']} |")
    lines.append("")
    return "\n".join(lines)


def _top10_table(top10: list[dict]) -> str:
    lines = [
        "### Top 10 Longest Criteria", "",
        "| patient_id | trial_id | type | length | words | preview |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in top10:
        preview = r["preview"].replace("|", "\\|")
        lines.append(
            f"| {r['patient_id']} | {r['trial_id']} | {r['criterion_type']} "
            f"| {r['length']} | {r['words']} | {preview} |"
        )
    lines.append("")
    return "\n".join(lines)


def _signals_table(signal_totals: dict[str, int], total: int) -> str:
    lines = [
        "### Complexity Signals", "",
        "| Signal | Count | % |",
        "| --- | --- | --- |",
    ]
    for k, v in sorted(signal_totals.items(), key=lambda x: -x[1]):
        pct = round(100 * v / total, 1) if total else 0
        lines.append(f"| {k} | {v} | {pct}% |")
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(summary: dict) -> str:
    length_order = ["0–50", "51–150", "151–300", "301–600", ">600"]
    word_order = ["0–10", "11–25", "26–50", "51–100", ">100"]

    parts = [
        "# Criterion Length & Complexity Analysis",
        "",
        f"**Total criterion rows:** {summary['total']}",
        "",
        "---",
        "",
        "### Character Length Statistics",
        "",
        _stats_table(summary["char_stats"]),
        "",
        "### Word Count Statistics",
        "",
        _stats_table(summary["word_stats"]),
        "",
        _count_table("Counts by Character Length Bucket", summary["length_buckets"], length_order),
        _count_table("Counts by Word Count Bucket", summary["word_buckets"], word_order),
        _signals_table(summary["signal_totals"], summary["total"]),
    ]

    if summary["ctype_stats"]:
        parts.append(_group_stats_table("Character Length by criterion_type", summary["ctype_stats"]))

    if summary["classified_stats"]:
        parts.append(_group_stats_table("Character Length by classified_criterion_type", summary["classified_stats"]))

    parts.append(_top10_table(summary["top10"]))

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

    summary = analyze_rows(rows)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Rows read    : {summary['total']}")
    print(f"Longest      : {summary['char_stats']['max']} chars")
    print(f"Report       : {REPORT_PATH}")


if __name__ == "__main__":
    main()
