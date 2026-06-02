"""
Task 93: error clustering analysis.

Usage:
    PYTHONPATH=. python eval/analyze_error_clustering.py
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict


INPUT_RESULTS = "data/processed/results_llm_reviewed.json"
INPUT_ERRORS = "data/processed/error_analysis_llm_reviewed.json"
INPUT_CRITERION = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/error_clustering_analysis.md"

TOP_REASON_PATTERNS = 25
TOP_COMBINED = 25
MAX_EXAMPLES = 3

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "with", "this",
    "that", "it", "as", "by", "from", "not", "no", "but", "if", "has",
    "have", "had", "does", "do", "did", "will", "would", "could", "should",
    "may", "might", "can", "patient", "criterion", "criteria",
}
_PUNCT_RE = re.compile(r"[^\w\s]")
_NUM_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str, required: bool = False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: '{path}'")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str, required: bool = False) -> list[dict] | None:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: '{path}'")
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def extract_predictions(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("predictions", "results", "records", "cases"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Cannot locate a records list in the JSON.")


def extract_error_records(data) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("errors", "error_records", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def pair_key(record: dict) -> tuple[str, str] | None:
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    if not pid or not tid:
        return None
    return (pid, tid)


def get_gold_label(record: dict) -> str:
    for f in ("gold_label", "gold", "label", "expected"):
        v = record.get(f, "")
        if v:
            return str(v).strip().lower()
    return ""


def get_predicted_label(record: dict) -> str:
    for f in ("predicted_label", "predicted", "prediction", "output"):
        v = record.get(f, "")
        if v:
            return str(v).strip().lower()
    return ""


def is_incorrect(record: dict) -> bool:
    g = get_gold_label(record)
    p = get_predicted_label(record)
    return bool(g and p and g != p)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    if not value:
        return ""
    t = str(value).strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _NUM_RE.sub("<num>", t)
    t = _SPACE_RE.sub(" ", t)
    return t


def preview_text(value: str, max_chars: int = 180) -> str:
    t = str(value).strip()
    return t[:max_chars] + ("…" if len(t) > max_chars else "")


def normalize_reason_pattern(text: str) -> str:
    if not text or not text.strip():
        return "no_reason_text"
    t = normalize_text(text)
    tokens = [tok for tok in t.split() if tok not in _STOPWORDS and len(tok) > 1]
    return " ".join(tokens[:8]) or "no_reason_text"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_error_analysis(error_data) -> dict[tuple, dict]:
    records = extract_error_records(error_data)
    index: dict[tuple, dict] = {}
    for r in records:
        k = pair_key(r)
        if k:
            index[k] = r
    return index


def index_criterion_rows(rows: list[dict] | None) -> dict[tuple, list[dict]]:
    if not rows:
        return {}
    index: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        pid = str(r.get("patient_id", "")).strip()
        tid = str(r.get("trial_id", "")).strip()
        if pid and tid:
            index[(pid, tid)].append(r)
    return dict(index)


# ---------------------------------------------------------------------------
# Collect enrichment
# ---------------------------------------------------------------------------

def collect_reason_text(
    record: dict,
    error_record: dict | None = None,
    criterion_rows: list[dict] | None = None,
) -> str:
    for f in ("explanation", "matcher_explanation", "reason", "rationale",
              "reasoning", "blocking_criteria", "uncertain_criteria"):
        v = record.get(f, "")
        if v:
            return str(v)
    if error_record:
        for f in ("explanation", "reason", "rationale"):
            v = error_record.get(f, "")
            if v:
                return str(v)
    if criterion_rows:
        texts = []
        for cr in criterion_rows:
            for f in ("reason", "explanation", "rationale"):
                v = cr.get(f, "")
                if v:
                    texts.append(str(v))
                    break
        if texts:
            return " | ".join(texts[:3])
    return ""


def collect_criterion_types(criterion_rows: list[dict] | None) -> list[str]:
    if not criterion_rows:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for cr in criterion_rows:
        for f in ("classified_criterion_type", "criterion_type", "type", "category"):
            v = cr.get(f, "").strip()
            if v and v not in seen:
                seen.add(v)
                result.append(v)
            if v:
                break
    return result


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_errors(
    predictions: list[dict],
    error_index: dict[tuple, dict],
    criterion_index: dict[tuple, list[dict]],
) -> dict:
    label_pair_counts: dict[str, int] = defaultdict(int)
    error_type_counts: dict[str, int] = defaultdict(int)
    severity_counts: dict[str, int] = defaultdict(int)
    ctype_counts: dict[str, int] = defaultdict(int)
    reason_pattern_counts: dict[str, int] = defaultdict(int)
    combined_counts: dict[str, int] = defaultdict(int)

    label_pair_examples: dict[str, list] = defaultdict(list)
    reason_pattern_examples: dict[str, list] = defaultdict(list)
    combined_examples: dict[str, list] = defaultdict(list)

    skipped = 0
    incorrect_rows: list[dict] = []

    for rec in predictions:
        gold = get_gold_label(rec)
        pred = get_predicted_label(rec)
        if not gold or not pred:
            skipped += 1
            continue
        if gold == pred:
            continue

        k = pair_key(rec)
        err_rec = error_index.get(k) if k else None
        crit_rows = criterion_index.get(k) if k else None

        error_type = ""
        severity = ""
        if err_rec:
            error_type = str(err_rec.get("error_type", "")).strip()
            severity = str(err_rec.get("severity", "")).strip()
        if not error_type:
            error_type = str(rec.get("error_type", "")).strip()
        if not severity:
            severity = str(rec.get("severity", "")).strip()

        ctypes = collect_criterion_types(crit_rows)
        reason_raw = collect_reason_text(rec, err_rec, crit_rows)
        pattern = normalize_reason_pattern(reason_raw)
        lp = f"{gold}→{pred}"
        ctype_str = ctypes[0] if ctypes else "unknown"
        combined = f"{lp} | {ctype_str} | {pattern}"

        label_pair_counts[lp] += 1
        if error_type:
            error_type_counts[error_type] += 1
        if severity:
            severity_counts[severity] += 1
        for ct in ctypes:
            ctype_counts[ct] += 1
        reason_pattern_counts[pattern] += 1
        combined_counts[combined] += 1

        entry = {
            "patient_id": str(rec.get("patient_id", "")),
            "trial_id": str(rec.get("trial_id", "")),
            "gold_label": gold,
            "predicted_label": pred,
            "error_type": error_type,
            "severity": severity,
            "criterion_types": ctypes,
            "reason_pattern": pattern,
            "explanation_preview": preview_text(reason_raw),
        }
        if len(label_pair_examples[lp]) < MAX_EXAMPLES:
            label_pair_examples[lp].append(entry)
        if len(reason_pattern_examples[pattern]) < MAX_EXAMPLES:
            reason_pattern_examples[pattern].append(entry)
        if len(combined_examples[combined]) < MAX_EXAMPLES:
            combined_examples[combined].append(entry)

        incorrect_rows.append(entry)

    top_reason = sorted(reason_pattern_counts.items(), key=lambda x: -x[1])[:TOP_REASON_PATTERNS]
    top_combined = sorted(combined_counts.items(), key=lambda x: -x[1])[:TOP_COMBINED]

    return {
        "incorrect_count": len(incorrect_rows),
        "skipped": skipped,
        "label_pair_counts": dict(sorted(label_pair_counts.items(), key=lambda x: -x[1])),
        "error_type_counts": dict(sorted(error_type_counts.items(), key=lambda x: -x[1])),
        "severity_counts": dict(sorted(severity_counts.items(), key=lambda x: -x[1])),
        "ctype_counts": dict(sorted(ctype_counts.items(), key=lambda x: -x[1])),
        "top_reason_patterns": top_reason,
        "top_combined": top_combined,
        "label_pair_examples": dict(label_pair_examples),
        "reason_pattern_examples": dict(reason_pattern_examples),
        "combined_examples": dict(combined_examples),
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _count_table(title: str, counts: dict[str, int]) -> list[str]:
    if not counts:
        return [f"### {title}", "", "_No data._", ""]
    lines = [f"### {title}", "", "| Value | Count |", "| --- | --- |"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def _example_block(entries: list[dict]) -> list[str]:
    lines = []
    for e in entries:
        parts = []
        if e["patient_id"]:
            parts.append(f"patient={e['patient_id']}")
        if e["trial_id"]:
            parts.append(f"trial={e['trial_id']}")
        parts.append(f"{e['gold_label']}→{e['predicted_label']}")
        if e["error_type"]:
            parts.append(f"type={e['error_type']}")
        if e["severity"]:
            parts.append(f"sev={e['severity']}")
        if e["criterion_types"]:
            parts.append(f"ctypes={','.join(e['criterion_types'][:3])}")
        meta = " · ".join(parts)
        lines.append(f"  - **{meta}**")
        if e["explanation_preview"]:
            lines.append(f"    - {e['explanation_preview']}")
    return lines


def format_markdown_report(summary: dict) -> str:
    lines = [
        "# Error Clustering Analysis",
        "",
        f"**Total records read:** {summary['total_records']}  ",
        f"**Incorrect records:** {summary['clusters']['incorrect_count']}  ",
        f"**Skipped (missing labels):** {summary['clusters']['skipped']}",
        "",
        "---",
        "",
    ]

    cl = summary["clusters"]

    if cl["incorrect_count"] == 0:
        lines += ["No incorrect prediction records found.", ""]
        return "\n".join(lines)

    lines += _count_table("Clusters by Label Pair (gold→predicted)", cl["label_pair_counts"])
    lines += ["---", ""]

    if cl["error_type_counts"]:
        lines += _count_table("Clusters by Error Type", cl["error_type_counts"])
        lines += ["---", ""]

    if cl["severity_counts"]:
        lines += _count_table("Clusters by Severity", cl["severity_counts"])
        lines += ["---", ""]

    if cl["ctype_counts"]:
        lines += _count_table("Clusters by Criterion Type", cl["ctype_counts"])
        lines += ["---", ""]

    # Top reason patterns table
    lines += [f"### Top {TOP_REASON_PATTERNS} Normalized Reason-Pattern Clusters", "",
              "| # | Pattern | Count |", "| --- | --- | --- |"]
    for i, (pat, cnt) in enumerate(cl["top_reason_patterns"], 1):
        lines.append(f"| {i} | {pat.replace('|', chr(8739))} | {cnt} |")
    lines.append("")

    # Examples for top reason patterns
    lines += ["#### Examples for Top Reason Patterns", ""]
    for pat, _ in cl["top_reason_patterns"][:5]:
        ex = cl["reason_pattern_examples"].get(pat, [])
        if ex:
            lines.append(f"**Pattern:** `{pat}`")
            lines += _example_block(ex)
            lines.append("")

    lines += ["---", ""]

    # Top combined clusters
    lines += [f"### Top {TOP_COMBINED} Combined Clusters (label_pair | ctype | reason_pattern)", "",
              "| # | Cluster | Count |", "| --- | --- | --- |"]
    for i, (combo, cnt) in enumerate(cl["top_combined"], 1):
        lines.append(f"| {i} | {combo.replace('|', chr(8739))} | {cnt} |")
    lines.append("")

    # Examples for top combined clusters
    lines += ["#### Examples for Top Combined Clusters", ""]
    for combo, _ in cl["top_combined"][:5]:
        ex = cl["combined_examples"].get(combo, [])
        if ex:
            lines.append(f"**Cluster:** `{combo}`")
            lines += _example_block(ex)
            lines.append("")

    # Label pair examples
    lines += ["---", "", "### Examples by Label Pair", ""]
    for lp, ex in cl["label_pair_examples"].items():
        lines.append(f"#### `{lp}`")
        lines += _example_block(ex)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        results_data = load_json(INPUT_RESULTS, required=True)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        predictions = extract_predictions(results_data)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    error_data = load_json(INPUT_ERRORS, required=False)
    criterion_rows = load_csv_rows(INPUT_CRITERION, required=False)

    error_index = index_error_analysis(error_data)
    criterion_index = index_criterion_rows(criterion_rows)

    clusters = cluster_errors(predictions, error_index, criterion_index)

    summary = {
        "total_records": len(predictions),
        "clusters": clusters,
    }

    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Records read    : {len(predictions)}")
    print(f"Incorrect       : {clusters['incorrect_count']}")
    print(f"Top clusters    : {len(clusters['top_combined'])}")
    print(f"Report          : {REPORT_PATH}")


if __name__ == "__main__":
    main()
