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
CRITERION_COMPLEXITY_CSV = "data/processed/criterion_complexity_scores.csv"
TRIAL_COMPLEXITY_CSV = "data/processed/trial_complexity_scores.csv"

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


def write_csv(rows: list[dict], fieldnames: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def bucket_complexity(score: int) -> str:
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


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


def compute_criterion_complexity_score(char_length: int, signals: dict[str, bool]) -> int:
    """Deterministic complexity score for a single criterion."""
    score = 0
    # length contribution
    lb = bucket_length(char_length)
    if lb in ("151–300", "301–600"):
        score += 1
    elif lb == ">600":
        score += 2
    # signal contributions
    if signals.get("numeric_threshold"):
        score += 1
    if signals.get("temporal_keyword"):
        score += 1
    if signals.get("medication_keyword"):
        score += 1
    if signals.get("procedure_device_keyword"):
        score += 1
    if signals.get("cognitive_keyword"):
        score += 1
    if signals.get("lab_keyword"):
        score += 1
    if signals.get("multiple_clauses"):
        score += 1
    return score


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def normalize_label(val: str) -> str:
    return val.strip().lower() if val else ""


def is_error_row(row: dict) -> bool | None:
    """Return True if gold != predicted, False if equal, None if labels missing."""
    gold = normalize_label(row.get("gold_label", ""))
    pred = normalize_label(row.get("predicted_label", ""))
    if not gold or not pred:
        return None
    return gold != pred


# ---------------------------------------------------------------------------
# Per-criterion records
# ---------------------------------------------------------------------------

def build_criterion_records(rows: list[dict]) -> list[dict]:
    """Return one enriched dict per row with complexity fields."""
    records = []
    for row in rows:
        text = get_criterion_text(row)
        cl = len(text)
        wc = count_words(text)
        lb = bucket_length(cl)
        wb = bucket_word_count(wc)
        signals = detect_complexity_signals(text)
        score = compute_criterion_complexity_score(cl, signals)
        cb = bucket_complexity(score)
        records.append({
            "patient_id": row.get("patient_id", ""),
            "trial_id": row.get("trial_id", ""),
            "criterion_type": row.get("criterion_type", "").strip(),
            "classified_criterion_type": row.get("classified_criterion_type", "").strip(),
            "gold_label": row.get("gold_label", ""),
            "predicted_label": row.get("predicted_label", ""),
            "char_length": cl,
            "word_count": wc,
            "length_bucket": lb,
            "word_count_bucket": wb,
            "numeric_threshold": int(signals["numeric_threshold"]),
            "age_criterion": int(signals["age_criterion"]),
            "temporal_keyword": int(signals["temporal_keyword"]),
            "medication_keyword": int(signals["medication_keyword"]),
            "procedure_device_keyword": int(signals["procedure_device_keyword"]),
            "cognitive_keyword": int(signals["cognitive_keyword"]),
            "lab_keyword": int(signals["lab_keyword"]),
            "multiple_clauses": int(signals["multiple_clauses"]),
            "complexity_score": score,
            "complexity_bucket": cb,
            "preview": text[:120] + ("…" if len(text) > 120 else ""),
        })
    return records


# ---------------------------------------------------------------------------
# Trial complexity aggregation
# ---------------------------------------------------------------------------

def compute_trial_complexity_score(
    criteria_count: int,
    mean_length: float,
    numeric_count: int,
    temporal_count: int,
    medication_count: int,
    procedure_count: int,
    cognitive_count: int,
    lab_count: int,
) -> int:
    """Deterministic trial-level complexity score."""
    score = 0
    if criteria_count >= 10:
        score += 1
    if criteria_count >= 20:
        score += 1
    if mean_length > 150:
        score += 1
    if mean_length > 300:
        score += 1
    for cnt in (numeric_count, temporal_count, medication_count,
                procedure_count, cognitive_count, lab_count):
        if cnt > 0:
            score += 1
    return score


def aggregate_trial_complexity(criterion_records: list[dict]) -> list[dict]:
    """Group criterion records by trial_id and compute trial-level complexity."""
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for rec in criterion_records:
        tid = rec["trial_id"]
        if tid:
            by_trial[tid].append(rec)

    trial_records = []
    for tid in sorted(by_trial.keys()):
        recs = by_trial[tid]
        lengths = [r["char_length"] for r in recs]
        n = len(recs)
        mean_len = round(sum(lengths) / n, 1) if n else 0.0
        max_len = max(lengths) if lengths else 0
        numeric_count = sum(r["numeric_threshold"] for r in recs)
        temporal_count = sum(r["temporal_keyword"] for r in recs)
        medication_count = sum(r["medication_keyword"] for r in recs)
        procedure_count = sum(r["procedure_device_keyword"] for r in recs)
        cognitive_count = sum(r["cognitive_keyword"] for r in recs)
        lab_count = sum(r["lab_keyword"] for r in recs)
        score = compute_trial_complexity_score(
            n, mean_len, numeric_count, temporal_count,
            medication_count, procedure_count, cognitive_count, lab_count
        )
        trial_records.append({
            "trial_id": tid,
            "criteria_count": n,
            "mean_criterion_length": mean_len,
            "max_criterion_length": max_len,
            "numeric_criteria_count": numeric_count,
            "temporal_criteria_count": temporal_count,
            "medication_criteria_count": medication_count,
            "procedure_device_criteria_count": procedure_count,
            "cognitive_criteria_count": cognitive_count,
            "lab_criteria_count": lab_count,
            "complexity_score": score,
            "complexity_bucket": bucket_complexity(score),
        })
    return trial_records


# ---------------------------------------------------------------------------
# Error-rate analysis
# ---------------------------------------------------------------------------

def _error_rate_by_bucket(
    criterion_records: list[dict], bucket_field: str, ordered_keys: list[str]
) -> dict[str, dict]:
    """Return {bucket: {total, errors, error_rate}} for rows with labels."""
    totals: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    for rec in criterion_records:
        err = is_error_row(rec)
        if err is None:
            continue
        b = rec[bucket_field]
        totals[b] += 1
        if err:
            errors[b] += 1
    result = {}
    for k in ordered_keys:
        t = totals.get(k, 0)
        e = errors.get(k, 0)
        result[k] = {
            "total": t,
            "errors": e,
            "error_rate": round(e / t * 100, 1) if t else None,
        }
    return result


def _error_rate_by_signal(criterion_records: list[dict]) -> dict[str, dict]:
    """Return error rates for rows with vs without each complexity signal."""
    signal_fields = [
        "numeric_threshold", "temporal_keyword", "medication_keyword",
        "procedure_device_keyword", "cognitive_keyword", "lab_keyword",
        "multiple_clauses",
    ]
    result = {}
    for sig in signal_fields:
        for val, label in ((1, "yes"), (0, "no")):
            key = f"{sig}={label}"
            recs = [r for r in criterion_records if r[sig] == val]
            labeled = [r for r in recs if is_error_row(r) is not None]
            errs = sum(1 for r in labeled if is_error_row(r))
            t = len(labeled)
            result[key] = {
                "total": t,
                "errors": errs,
                "error_rate": round(errs / t * 100, 1) if t else None,
            }
    return result


def _error_rate_by_trial_complexity(
    criterion_records: list[dict], trial_records: list[dict]
) -> dict[str, dict]:
    """Return error rates bucketed by trial complexity bucket."""
    trial_bucket_map = {r["trial_id"]: r["complexity_bucket"] for r in trial_records}
    buckets = ["low", "medium", "high"]
    totals: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    for rec in criterion_records:
        err = is_error_row(rec)
        if err is None:
            continue
        tb = trial_bucket_map.get(rec["trial_id"])
        if tb is None:
            continue
        totals[tb] += 1
        if err:
            errors[tb] += 1
    result = {}
    for b in buckets:
        t = totals.get(b, 0)
        e = errors.get(b, 0)
        result[b] = {
            "total": t,
            "errors": e,
            "error_rate": round(e / t * 100, 1) if t else None,
        }
    return result


def _top_error_heavy_trials(
    criterion_records: list[dict], trial_records: list[dict], top_n: int = 10
) -> list[dict]:
    """Return top trials by error count among labeled rows."""
    trial_bucket_map = {r["trial_id"]: r["complexity_bucket"] for r in trial_records}
    trial_score_map = {r["trial_id"]: r["complexity_score"] for r in trial_records}
    totals: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    for rec in criterion_records:
        err = is_error_row(rec)
        if err is None:
            continue
        tid = rec["trial_id"]
        if not tid:
            continue
        totals[tid] += 1
        if err:
            errors[tid] += 1
    rows = []
    for tid, t in totals.items():
        e = errors.get(tid, 0)
        rows.append({
            "trial_id": tid,
            "labeled_criteria": t,
            "errors": e,
            "error_rate": round(e / t * 100, 1) if t else None,
            "complexity_score": trial_score_map.get(tid, ""),
            "complexity_bucket": trial_bucket_map.get(tid, ""),
        })
    rows.sort(key=lambda r: (-r["errors"], r["trial_id"]))
    return rows[:top_n]


def compute_error_analysis(
    criterion_records: list[dict], trial_records: list[dict]
) -> dict:
    length_order = ["0–50", "51–150", "151–300", "301–600", ">600"]
    word_order = ["0–10", "11–25", "26–50", "51–100", ">100"]
    complexity_order = ["low", "medium", "high"]
    return {
        "by_length_bucket": _error_rate_by_bucket(criterion_records, "length_bucket", length_order),
        "by_word_bucket": _error_rate_by_bucket(criterion_records, "word_count_bucket", word_order),
        "by_complexity_bucket": _error_rate_by_bucket(criterion_records, "complexity_bucket", complexity_order),
        "by_signal": _error_rate_by_signal(criterion_records),
        "by_trial_complexity_bucket": _error_rate_by_trial_complexity(criterion_records, trial_records),
        "top_error_trials": _top_error_heavy_trials(criterion_records, trial_records),
    }


# ---------------------------------------------------------------------------
# Legacy analyze_rows (preserves existing interface)
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
# Complexity score summaries
# ---------------------------------------------------------------------------

def summarize_criterion_complexity(criterion_records: list[dict]) -> dict:
    scores = [r["complexity_score"] for r in criterion_records]
    bucket_counts: dict[str, int] = defaultdict(int)
    for r in criterion_records:
        bucket_counts[r["complexity_bucket"]] += 1
    return {
        "stats": compute_basic_stats(scores),
        "bucket_counts": dict(bucket_counts),
    }


def summarize_trial_complexity(trial_records: list[dict]) -> dict:
    scores = [r["complexity_score"] for r in trial_records]
    bucket_counts: dict[str, int] = defaultdict(int)
    for r in trial_records:
        bucket_counts[r["complexity_bucket"]] += 1
    return {
        "total_trials": len(trial_records),
        "stats": compute_basic_stats(scores),
        "bucket_counts": dict(bucket_counts),
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


def _criterion_complexity_summary_section(csum: dict) -> str:
    complexity_order = ["low", "medium", "high"]
    lines = [
        "## Criterion Complexity Score Summary",
        "",
        "### Score Statistics",
        "",
        _stats_table(csum["stats"]),
        "",
        _count_table("Counts by Complexity Bucket", csum["bucket_counts"], complexity_order),
    ]
    return "\n".join(lines)


def _trial_complexity_summary_section(tsum: dict) -> str:
    complexity_order = ["low", "medium", "high"]
    lines = [
        "## Trial Complexity Score Summary",
        "",
        f"**Total trials:** {tsum['total_trials']}",
        "",
        "### Trial Score Statistics",
        "",
        _stats_table(tsum["stats"]),
        "",
        _count_table("Trials by Complexity Bucket", tsum["bucket_counts"], complexity_order),
    ]
    return "\n".join(lines)


def _error_rate_table(title: str, data: dict, ordered_keys: list[str]) -> str:
    lines = [
        f"### {title}", "",
        "| Bucket | Total | Errors | Error Rate |",
        "| --- | --- | --- | --- |",
    ]
    for k in ordered_keys:
        d = data.get(k, {"total": 0, "errors": 0, "error_rate": None})
        er = f"{d['error_rate']}%" if d["error_rate"] is not None else "n/a"
        lines.append(f"| {k} | {d['total']} | {d['errors']} | {er} |")
    lines.append("")
    return "\n".join(lines)


def _error_rate_signal_table(signal_data: dict) -> str:
    signals = [
        "numeric_threshold", "temporal_keyword", "medication_keyword",
        "procedure_device_keyword", "cognitive_keyword", "lab_keyword",
        "multiple_clauses",
    ]
    lines = [
        "### Error Rate by Complexity Signal",
        "",
        "| Signal | Present | Total | Errors | Error Rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sig in signals:
        for label in ("yes", "no"):
            key = f"{sig}={label}"
            d = signal_data.get(key, {"total": 0, "errors": 0, "error_rate": None})
            er = f"{d['error_rate']}%" if d["error_rate"] is not None else "n/a"
            lines.append(f"| {sig} | {label} | {d['total']} | {d['errors']} | {er} |")
    lines.append("")
    return "\n".join(lines)


def _top_error_trials_table(rows: list[dict]) -> str:
    lines = [
        "### Top Error-Heavy Complex Trials",
        "",
        "| trial_id | labeled_criteria | errors | error_rate | complexity_score | complexity_bucket |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        er = f"{r['error_rate']}%" if r["error_rate"] is not None else "n/a"
        lines.append(
            f"| {r['trial_id']} | {r['labeled_criteria']} | {r['errors']} "
            f"| {er} | {r['complexity_score']} | {r['complexity_bucket']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _error_analysis_section(ea: dict) -> str:
    length_order = ["0–50", "51–150", "151–300", "301–600", ">600"]
    word_order = ["0–10", "11–25", "26–50", "51–100", ">100"]
    complexity_order = ["low", "medium", "high"]
    parts = [
        "## Length vs Error Analysis",
        "",
        _error_rate_table("Error Rate by Character Length Bucket", ea["by_length_bucket"], length_order),
        _error_rate_table("Error Rate by Word Count Bucket", ea["by_word_bucket"], word_order),
        _error_rate_table("Error Rate by Criterion Complexity Bucket", ea["by_complexity_bucket"], complexity_order),
        _error_rate_signal_table(ea["by_signal"]),
        _error_rate_table("Error Rate by Trial Complexity Bucket", ea["by_trial_complexity_bucket"], complexity_order),
    ]
    return "\n".join(parts)


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

    # New sections
    if "criterion_complexity_summary" in summary:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(_criterion_complexity_summary_section(summary["criterion_complexity_summary"]))

    if "trial_complexity_summary" in summary:
        parts.append("")
        parts.append(_trial_complexity_summary_section(summary["trial_complexity_summary"]))

    if "error_analysis" in summary:
        ea = summary["error_analysis"]
        parts.append("")
        parts.append(_error_analysis_section(ea))
        parts.append("")
        parts.append("## Top Error-Heavy Complex Trials")
        parts.append("")
        parts.append(_top_error_trials_table(ea["top_error_trials"]))

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

    # Legacy summary
    summary = analyze_rows(rows)

    # Criterion-level complexity
    criterion_records = build_criterion_records(rows)
    summary["criterion_complexity_summary"] = summarize_criterion_complexity(criterion_records)

    # Trial-level complexity
    trial_records = aggregate_trial_complexity(criterion_records)
    summary["trial_complexity_summary"] = summarize_trial_complexity(trial_records)

    # Error analysis
    summary["error_analysis"] = compute_error_analysis(criterion_records, trial_records)

    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    # Optional CSV outputs
    crit_fields = [
        "patient_id", "trial_id", "criterion_type", "classified_criterion_type",
        "gold_label", "predicted_label", "char_length", "word_count",
        "length_bucket", "word_count_bucket", "numeric_threshold", "age_criterion",
        "temporal_keyword", "medication_keyword", "procedure_device_keyword",
        "cognitive_keyword", "lab_keyword", "multiple_clauses",
        "complexity_score", "complexity_bucket",
    ]
    try:
        write_csv(criterion_records, crit_fields, CRITERION_COMPLEXITY_CSV)
    except OSError as exc:
        print(f"WARNING: could not write criterion CSV: {exc}", file=sys.stderr)

    trial_fields = [
        "trial_id", "criteria_count", "mean_criterion_length", "max_criterion_length",
        "numeric_criteria_count", "temporal_criteria_count", "medication_criteria_count",
        "procedure_device_criteria_count", "cognitive_criteria_count", "lab_criteria_count",
        "complexity_score", "complexity_bucket",
    ]
    try:
        write_csv(trial_records, trial_fields, TRIAL_COMPLEXITY_CSV)
    except OSError as exc:
        print(f"WARNING: could not write trial CSV: {exc}", file=sys.stderr)

    print(f"Rows read    : {summary['total']}")
    print(f"Longest      : {summary['char_stats']['max']} chars")
    print(f"Report       : {REPORT_PATH}")
    print(f"Crit CSV     : {CRITERION_COMPLEXITY_CSV}")
    print(f"Trial CSV    : {TRIAL_COMPLEXITY_CSV}")


if __name__ == "__main__":
    main()
