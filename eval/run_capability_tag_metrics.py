"""
eval/run_capability_tag_metrics.py

Task 52 — Capability-tag metrics.

Reads data/processed/results_llm_reviewed.json and optionally
reports/minimal_pairs_report.json (if present).
Writes a Markdown report to reports/capability_tag_metrics.md.

Usage:
    PYTHONPATH=. python eval/run_capability_tag_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Tag definitions and keyword maps
# ---------------------------------------------------------------------------

SUPPORTED_TAGS = [
    "age_threshold",
    "medication",
    "procedure",
    "device",
    "cognitive",
    "diagnosis",
    "temporal",
    "lab_value",
    "missing_info",
    "negation",
    "uncertainty",
    "safety",
    "other",
]

_TAG_KEYWORDS: dict[str, list[str]] = {
    "age_threshold": ["age", "years old", "year-old", "older than", "younger than", "≥", "≤", "age range"],
    "medication": ["medication", "drug", "inhibitor", "levodopa", "rasagiline", "carbidopa", "dose", "pharmacolog", "treatment", "mao-b", "maob", "washout"],
    "procedure": ["surgery", "procedure", "implant", "operation", "dbs implant", "ablation", "transplant"],
    "device": ["device", "dbs", "deep brain stimulation", "pacemaker", "implantable", "stimulator"],
    "cognitive": ["cognitive", "moca", "mmse", "dementia", "memory", "cognition", "mild cognitive"],
    "diagnosis": ["diagnosis", "diagnosed", "parkinson", "idiopathic", "condition", "disease", "disorder", "subtype"],
    "temporal": ["within", "days", "weeks", "months", "years", "duration", "since", "stable for", "history of at least", "washout", "recent", "prior to", "last visit"],
    "lab_value": ["creatinine", "hemoglobin", "bmi", "weight", "lab", "score", "updrs", "hoehn", "yahr", "mmse", "moca", "level", "mg/dl", "mL/min"],
    "missing_info": ["missing", "not reported", "not documented", "unknown", "not mentioned", "unclear", "not available", "no information"],
    "negation": ["no ", "not ", "without", "denies", "absence of", "never", "none", "negative for", "no history"],
    "uncertainty": ["uncertain", "unclear", "ambiguous", "possible", "possible diagnosis", "may have", "might", "possibly"],
    "safety": ["unsafe", "critical", "not_eligible", "contraindication", "serious", "adverse", "risk"],
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    """Load and return parsed JSON from path. Raises SystemExit on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(results: Any) -> list[dict]:
    """
    Return a flat list of prediction records from results_llm_reviewed.json.
    Handles both list-of-records and dict-with-predictions-key shapes.
    """
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in ("predictions", "results", "cases"):
            if key in results and isinstance(results[key], list):
                return results[key]
        # dict keyed by patient_id or similar
        flat = []
        for v in results.values():
            if isinstance(v, list):
                flat.extend(v)
            elif isinstance(v, dict):
                flat.append(v)
        return flat
    return []


def collect_text_for_tagging(record: dict) -> str:
    """
    Gather all text fields from a record useful for keyword-based tag inference.
    Returns a single lowercased string.
    """
    parts: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                _add(item)
        elif isinstance(value, dict):
            for v in value.values():
                _add(v)

    for field in (
        "criterion",
        "criteria_text",
        "explanation",
        "matched_facts",
        "blocking_criteria",
        "uncertain_criteria",
        "reasoning_trace",
        "error_explanation",
        "notes",
    ):
        if field in record:
            _add(record[field])

    return " ".join(parts).lower()


def infer_capability_tags(text: str) -> list[str]:
    """
    Return a sorted list of inferred capability tags from lowercased text.
    Each tag is prefixed with no marker here; callers track inferred vs explicit.
    Falls back to ["other"] if nothing matches.
    """
    found = []
    for tag, keywords in _TAG_KEYWORDS.items():
        if tag == "other":
            continue
        for kw in keywords:
            if kw in text:
                found.append(tag)
                break
    return sorted(found) if found else ["other"]


def get_record_tags(record: dict) -> tuple[list[str], str]:
    """
    Return (tags, source) where source is 'explicit' or 'inferred'.
    Checks known explicit tag fields first; falls back to keyword inference.
    """
    for field in ("capability_tags", "tags", "error_tags"):
        value = record.get(field)
        if isinstance(value, list) and value:
            return sorted(str(t) for t in value), "explicit"
        if isinstance(value, str) and value.strip():
            return [value.strip()], "explicit"

    # Check nested reasoning_trace
    rt = record.get("reasoning_trace")
    if isinstance(rt, dict):
        nested = rt.get("capability_tags")
        if isinstance(nested, list) and nested:
            return sorted(str(t) for t in nested), "explicit"
    elif isinstance(rt, list):
        for step in rt:
            if isinstance(step, dict):
                nested = step.get("capability_tags")
                if isinstance(nested, list) and nested:
                    return sorted(str(t) for t in nested), "explicit"

    text = collect_text_for_tagging(record)
    return infer_capability_tags(text), "inferred"


def compute_metrics_by_tag(
    predictions: list[dict],
) -> dict[str, Any]:
    """
    For each capability tag compute accuracy, label distributions, and
    example errors. Also return summary counts and co-occurrence.
    """
    tag_records: dict[str, list[dict]] = defaultdict(list)
    explicit_count = 0
    inferred_count = 0
    untagged_count = 0
    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)

    enriched: list[dict] = []
    for rec in predictions:
        tags, source = get_record_tags(rec)
        enriched.append({"record": rec, "tags": tags, "source": source})

        if source == "explicit":
            explicit_count += 1
        elif tags == ["other"]:
            untagged_count += 1
        else:
            inferred_count += 1

        for tag in tags:
            tag_records[tag].append(rec)

        # Co-occurrence
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1 :]:
                pair = (t1, t2) if t1 <= t2 else (t2, t1)
                cooccurrence[pair] += 1

    tag_metrics: dict[str, dict] = {}
    for tag in SUPPORTED_TAGS:
        recs = tag_records.get(tag, [])
        total = len(recs)
        if total == 0:
            continue

        correct = 0
        errors: list[dict] = []
        gold_dist: dict[str, int] = defaultdict(int)
        pred_dist: dict[str, int] = defaultdict(int)

        for rec in recs:
            gold = rec.get("gold_label") or rec.get("label") or rec.get("expected_label") or "unknown"
            pred = rec.get("predicted_label") or rec.get("prediction") or rec.get("predicted") or "unknown"
            gold_dist[gold] += 1
            pred_dist[pred] += 1
            if gold == pred:
                correct += 1
            else:
                errors.append({
                    "patient_id": rec.get("patient_id", ""),
                    "trial_id": rec.get("trial_id", ""),
                    "gold_label": gold,
                    "predicted_label": pred,
                })

        accuracy = correct / total if total > 0 else 0.0
        tag_metrics[tag] = {
            "total": total,
            "correct": correct,
            "errors": len(errors),
            "accuracy": round(accuracy, 4),
            "gold_distribution": dict(gold_dist),
            "predicted_distribution": dict(pred_dist),
            "top_errors": errors[:5],
        }

    return {
        "total_records": len(predictions),
        "explicit_count": explicit_count,
        "inferred_count": inferred_count,
        "untagged_count": untagged_count,
        "tag_metrics": tag_metrics,
        "cooccurrence": {f"{a}+{b}": v for (a, b), v in sorted(cooccurrence.items(), key=lambda x: -x[1])},
    }


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Render the metrics summary as a Markdown string."""
    lines: list[str] = []
    lines.append("# Capability Tag Metrics Report\n")

    lines.append("## Summary\n")
    lines.append(f"- **Total records read:** {summary['total_records']}")
    lines.append(f"- **Records with explicit tags:** {summary['explicit_count']}")
    lines.append(f"- **Records with inferred tags:** {summary['inferred_count']}")
    lines.append(f"- **Records with no usable tag (other):** {summary['untagged_count']}")
    lines.append("")

    tag_metrics = summary["tag_metrics"]
    if not tag_metrics:
        lines.append("_No tag metrics computed._\n")
        return "\n".join(lines)

    lines.append("## Metrics by Capability Tag\n")
    lines.append("| Tag | Total | Correct | Errors | Accuracy |")
    lines.append("|-----|------:|--------:|-------:|---------:|")
    for tag in SUPPORTED_TAGS:
        if tag not in tag_metrics:
            continue
        m = tag_metrics[tag]
        lines.append(
            f"| {tag} | {m['total']} | {m['correct']} | {m['errors']} | {m['accuracy']:.2%} |"
        )
    lines.append("")

    lines.append("## Per-Tag Detail\n")
    for tag in SUPPORTED_TAGS:
        if tag not in tag_metrics:
            continue
        m = tag_metrics[tag]
        lines.append(f"### `{tag}`\n")
        lines.append(f"- Total: {m['total']} | Correct: {m['correct']} | Errors: {m['errors']} | Accuracy: {m['accuracy']:.2%}")

        lines.append("\n**Gold label distribution:**")
        for lbl, cnt in sorted(m["gold_distribution"].items()):
            lines.append(f"  - {lbl}: {cnt}")

        lines.append("\n**Predicted label distribution:**")
        for lbl, cnt in sorted(m["predicted_distribution"].items()):
            lines.append(f"  - {lbl}: {cnt}")

        if m["top_errors"]:
            lines.append("\n**Top errors (up to 5):**")
            lines.append("| patient_id | trial_id | gold | predicted |")
            lines.append("|------------|----------|------|-----------|")
            for e in m["top_errors"]:
                lines.append(
                    f"| {e['patient_id']} | {e['trial_id']} | {e['gold_label']} | {e['predicted_label']} |"
                )
        lines.append("")

    cooc = summary.get("cooccurrence", {})
    if cooc:
        lines.append("## Tag Co-occurrence\n")
        lines.append("| Tag pair | Count |")
        lines.append("|----------|------:|")
        for pair, cnt in list(cooc.items())[:20]:
            lines.append(f"| {pair} | {cnt} |")
        lines.append("")

    lines.append("_Tags marked as inferred are derived from keyword matching on criterion text, explanations, and trace fields. Explicit tags come from capability_tags / tags / error_tags fields in the source data._\n")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write text to path, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    results_path = "data/processed/results_llm_reviewed.json"
    minimal_pairs_path = "reports/minimal_pairs_report.json"
    output_path = "reports/capability_tag_metrics.md"

    raw = load_json(results_path)
    predictions = extract_predictions(raw)

    if not predictions:
        print("WARNING: No prediction records found in results file.", file=sys.stderr)

    # Optionally merge minimal pairs records if they have capability_tags
    if os.path.exists(minimal_pairs_path):
        try:
            mp_raw = json.loads(open(minimal_pairs_path, encoding="utf-8").read())
            mp_preds = extract_predictions(mp_raw)
            tagged = [r for r in mp_preds if r.get("capability_tags") or r.get("tags")]
            if tagged:
                predictions = predictions + tagged
                print(f"INFO: Merged {len(tagged)} tagged records from {minimal_pairs_path}.")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Could not read {minimal_pairs_path}: {exc}", file=sys.stderr)

    summary = compute_metrics_by_tag(predictions)
    report = format_markdown_report(summary)
    write_text(report, output_path)

    print(f"Capability tag metrics report written to: {output_path}")
    print(f"Total records: {summary['total_records']}")
    print(f"Explicit tags: {summary['explicit_count']}")
    print(f"Inferred tags: {summary['inferred_count']}")
    print(f"Untagged (other): {summary['untagged_count']}")
    print(f"Tags with data: {list(summary['tag_metrics'].keys())}")


if __name__ == "__main__":
    main()
