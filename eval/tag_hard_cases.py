"""Tag benchmark records with hard-case difficulty labels.

Tags assigned:
  hard_negative               — gold label is not_eligible and text signals hard exclusion
  hard_positive               — gold label is eligible and text signals non-trivial complexity
  ambiguous_clinical_severity — gold label is unclear and text signals severity/missing info

Usage:
    PYTHONPATH=. python eval/tag_hard_cases.py
"""

import argparse
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_LABELS = Path("data/processed/labels_llm_reviewed.json")
DEFAULT_PATIENTS = Path("data/processed/patient_cases.json")
DEFAULT_TRIALS = Path("data/processed/trial_cases.json")
DEFAULT_RESULTS = Path("data/processed/results_llm_reviewed.json")
DEFAULT_OUTPUT_JSON = Path("data/processed/hard_case_subsets.json")
DEFAULT_OUTPUT_CSV = Path("data/processed/hard_case_subsets.csv")
DEFAULT_OUTPUT_METRICS_JSON = Path("data/processed/hard_case_metrics.json")
DEFAULT_OUTPUT_METRICS_CSV = Path("data/processed/hard_case_metrics.csv")

ALL_TAGS = ["hard_negative", "hard_positive", "ambiguous_clinical_severity"]

# ---------------------------------------------------------------------------
# Signal word lists
# ---------------------------------------------------------------------------

_HARD_NEGATIVE_SIGNALS: list[str] = [
    r"\bhard exclusion\b",
    r"\bcontraindication\b",
    r"\bfailed inclusion\b",
    r"\bthreshold\b",
    r"\bdevice\b",
    r"\bmedication exclusion\b",
    r"\bdiagnosis mismatch\b",
    r"\bdbs\b",
    r"\bpacemaker\b",
    r"\bmao-?b\b",
    r"\bmonoamine oxidase\b",
    r"\bcognitive impairment\b",
    r"\bcognitive decline\b",
    r"\bdementia\b",
    r"\bage boundary\b",
    r"\bactive cancer\b",
    r"\bmalignancy\b",
    r"\bexclud\w*\b",
    r"\bdoes not meet\b",
    r"\bfails? criterion\b",
    r"\bineligible\b",
    r"\bnot eligible\b",
    r"\bviolat\w*\b",
    r"\bprior (dbs|surgery|stimulation)\b",
    r"\bcurrent use of\b",
    r"\bconcurrent\b",
    r"\bwashout\b",
]

_HARD_POSITIVE_SIGNALS: list[str] = [
    r"\bmultiple criteria\b",
    r"\bexclusion\b",
    r"\bthreshold\b",
    r"\bmedication\b",
    r"\bdevice\b",
    r"\bprocedure\b",
    r"\bstage\b",
    r"\bcognitive\b",
    r"\bscore\b",
    r"\bmoca\b",
    r"\bmmse\b",
    r"\bupdrs\b",
    r"\bhoehn\b",
    r"\byahr\b",
    r"\bcomorbidity\b",
    r"\bcomorbid\b",
    r"\bhistory of\b",
    r"\bprior\b",
    r"\bcurrent\b",
    r"\bage \d",
    r"\b\d+\s*(mg|years?|months?)\b",
    r"\binclusion\b",
    r"\ball criteria\b",
    r"\bsatisf\w+\b",
    r"\bno (prior|history|dbs|pacemaker)\b",
]

_AMBIGUOUS_SIGNALS: list[str] = [
    r"\bdisease stage\b",
    r"\bduration\b",
    r"\bseverity\b",
    r"\bhoehn\b",
    r"\byahr\b",
    r"\bupdrs\b",
    r"\bcognitive score\b",
    r"\bmoca\b",
    r"\bmmse\b",
    r"\bmedication history\b",
    r"\bcomorbidity\b",
    r"\bcomorbid\b",
    r"\bfrailty\b",
    r"\bgait\b",
    r"\bfreezing\b",
    r"\bmissing\b",
    r"\bunknown\b",
    r"\bambiguous\b",
    r"\binsufficient\b",
    r"\bnot (documented|recorded|specified|available|provided|stated)\b",
    r"\bunspecified\b",
    r"\bcannot (be )?determined\b",
    r"\bunclear\b",
    r"\binformation (not|is missing|unavailable)\b",
    r"\bno (data|information|record)\b",
    r"\btype unspecified\b",
    r"\bdose (not|unknown|unspecified)\b",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_patient_index(patients: list[dict]) -> dict[str, dict]:
    return {p["patient_id"]: p for p in patients}


def build_trial_index(trials: list[dict]) -> dict[str, dict]:
    return {t["trial_id"]: t for t in trials}


def build_result_index(results_payload: dict | list | None) -> dict[tuple[str, str], dict]:
    """Index result records by (patient_id, trial_id).

    Accepts either a raw list of prediction records or the full benchmark
    output dict (with a ``predictions`` key).
    """
    if results_payload is None:
        return {}
    records: list[dict] = []
    if isinstance(results_payload, list):
        records = results_payload
    elif isinstance(results_payload, dict):
        records = results_payload.get("predictions", [])
    return {(r["patient_id"], r["trial_id"]): r for r in records if "patient_id" in r and "trial_id" in r}


def _collect_text(*sources: object) -> str:
    """Flatten arbitrarily nested strings/lists/dicts into one lowercase string."""
    parts: list[str] = []

    def _walk(obj: object) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)

    for s in sources:
        _walk(s)
    return " ".join(parts).lower()


def _any_signal(text: str, patterns: list[str]) -> list[str]:
    """Return a list of matched pattern strings (for tag_reasons)."""
    matched: list[str] = []
    for pat in patterns:
        if re.search(pat, text):
            matched.append(pat)
    return matched


def assign_hard_case_tags(
    label_record: dict,
    patient: dict,
    trial: dict,
    result_record: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Return (tags, reasons) for a single benchmark record.

    Tags are a sorted subset of ALL_TAGS.
    Reasons are human-readable strings explaining each tag.
    """
    gold = label_record.get("label", "")

    # Gather all text available for the record
    combined = _collect_text(
        label_record.get("rationale", ""),
        label_record.get("evidence", {}),
        patient,
        trial,
        result_record or {},
    )

    tags: list[str] = []
    reasons: list[str] = []

    if gold == "not_eligible":
        matched = _any_signal(combined, _HARD_NEGATIVE_SIGNALS)
        if matched:
            tags.append("hard_negative")
            reasons.append(f"hard_negative: exclusion/threshold signals detected ({len(matched)} match(es))")

    if gold == "eligible":
        matched = _any_signal(combined, _HARD_POSITIVE_SIGNALS)
        if matched:
            tags.append("hard_positive")
            reasons.append(f"hard_positive: eligibility complexity signals detected ({len(matched)} match(es))")

    if gold == "unclear":
        matched = _any_signal(combined, _AMBIGUOUS_SIGNALS)
        if matched:
            tags.append("ambiguous_clinical_severity")
            reasons.append(f"ambiguous_clinical_severity: severity/missing-info signals detected ({len(matched)} match(es))")

    return sorted(tags), reasons


def build_hard_case_records(
    labels: list[dict],
    patients: list[dict],
    trials: list[dict],
    results_payload: dict | list | None = None,
) -> list[dict]:
    patient_index = build_patient_index(patients)
    trial_index = build_trial_index(trials)
    result_index = build_result_index(results_payload)

    records: list[dict] = []
    for lr in labels:
        pid = lr.get("patient_id", "")
        tid = lr.get("trial_id", "")
        patient = patient_index.get(pid, {})
        trial = trial_index.get(tid, {})
        result = result_index.get((pid, tid))

        tags, reasons = assign_hard_case_tags(lr, patient, trial, result)

        predicted = ""
        if result is not None:
            predicted = result.get("predicted_label", result.get("prediction", ""))

        records.append({
            "patient_id": pid,
            "trial_id": tid,
            "gold_label": lr.get("label", ""),
            "predicted_label": predicted,
            "hard_case_tags": tags,
            "tag_reasons": reasons,
        })

    return records


def build_summary(records: list[dict]) -> dict:
    tag_counts: dict[str, int] = {t: 0 for t in ALL_TAGS}
    label_distribution_by_tag: dict[str, dict[str, int]] = {t: {} for t in ALL_TAGS}

    for rec in records:
        gold = rec["gold_label"]
        for tag in rec["hard_case_tags"]:
            tag_counts[tag] += 1
            dist = label_distribution_by_tag[tag]
            dist[gold] = dist.get(gold, 0) + 1

    return {
        "total_records": len(records),
        "tag_counts": tag_counts,
        "label_distribution_by_tag": label_distribution_by_tag,
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_CSV_FIELDS = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "hard_case_tags", "tag_reasons",
]


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({
                **rec,
                "hard_case_tags": "; ".join(rec["hard_case_tags"]),
                "tag_reasons": " | ".join(rec["tag_reasons"]),
            })


# ---------------------------------------------------------------------------
# Per-tag metrics
# ---------------------------------------------------------------------------

_LABEL_CLASSES = ["eligible", "not_eligible", "unclear"]


def compute_classification_metrics(
    gold_labels: list[str], predicted_labels: list[str]
) -> dict:
    """Compute accuracy, macro F1, and per-class precision/recall/F1/support."""
    classes = _LABEL_CLASSES
    n = len(gold_labels)

    if n == 0:
        empty_per_class = {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in classes}
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": empty_per_class}

    correct = sum(g == p for g, p in zip(gold_labels, predicted_labels))
    accuracy = correct / n

    per_class: dict[str, dict] = {}
    f1_scores: list[float] = []
    for cls in classes:
        tp = sum(g == cls and p == cls for g, p in zip(gold_labels, predicted_labels))
        fp = sum(g != cls and p == cls for g, p in zip(gold_labels, predicted_labels))
        fn = sum(g == cls and p != cls for g, p in zip(gold_labels, predicted_labels))
        support = sum(g == cls for g in gold_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def build_metrics_by_tag(records: list[dict]) -> dict[str, dict]:
    """Compute classification metrics separately for each hard-case tag."""
    tag_records: dict[str, list[dict]] = {t: [] for t in ALL_TAGS}
    for rec in records:
        for tag in rec["hard_case_tags"]:
            if tag in tag_records:
                tag_records[tag].append(rec)

    metrics_by_tag: dict[str, dict] = {}
    for tag, recs in tag_records.items():
        total = len(recs)
        evaluated = [r for r in recs if r.get("predicted_label", "").strip()]
        gold = [r["gold_label"] for r in evaluated]
        pred = [r["predicted_label"] for r in evaluated]
        m = compute_classification_metrics(gold, pred)
        metrics_by_tag[tag] = {
            "total_records": total,
            "evaluated_records": len(evaluated),
            **m,
        }
    return metrics_by_tag


_METRICS_CSV_FIELDS = [
    "tag", "total_records", "evaluated_records", "accuracy", "macro_f1",
    "eligible_precision", "eligible_recall", "eligible_f1", "eligible_support",
    "not_eligible_precision", "not_eligible_recall", "not_eligible_f1", "not_eligible_support",
    "unclear_precision", "unclear_recall", "unclear_f1", "unclear_support",
]


def write_metrics_csv(metrics_by_tag: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_CSV_FIELDS)
        writer.writeheader()
        for tag in ALL_TAGS:
            m = metrics_by_tag.get(tag, {})
            pc = m.get("per_class", {})
            row: dict = {"tag": tag, "total_records": m.get("total_records", 0),
                         "evaluated_records": m.get("evaluated_records", 0),
                         "accuracy": m.get("accuracy", 0.0),
                         "macro_f1": m.get("macro_f1", 0.0)}
            for cls in _LABEL_CLASSES:
                cls_m = pc.get(cls, {})
                row[f"{cls}_precision"] = cls_m.get("precision", 0.0)
                row[f"{cls}_recall"] = cls_m.get("recall", 0.0)
                row[f"{cls}_f1"] = cls_m.get("f1", 0.0)
                row[f"{cls}_support"] = cls_m.get("support", 0)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag benchmark records with hard-case difficulty labels."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--patients", type=Path, default=DEFAULT_PATIENTS)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-metrics-json", type=Path, default=DEFAULT_OUTPUT_METRICS_JSON)
    parser.add_argument("--output-metrics-csv", type=Path, default=DEFAULT_OUTPUT_METRICS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading labels  : {args.labels}")
    labels: list[dict] = load_json(args.labels)  # type: ignore[assignment]

    print(f"Loading patients: {args.patients}")
    patients: list[dict] = load_json(args.patients)  # type: ignore[assignment]

    print(f"Loading trials  : {args.trials}")
    trials: list[dict] = load_json(args.trials)  # type: ignore[assignment]

    results_payload = None
    if args.results.exists():
        print(f"Loading results : {args.results}")
        results_payload = load_json(args.results)
    else:
        print(f"Results file not found, skipping: {args.results}")

    print("Tagging records …")
    records = build_hard_case_records(labels, patients, trials, results_payload)
    summary = build_summary(records)
    metrics_by_tag = build_metrics_by_tag(records)

    payload = {"summary": summary, "metrics_by_tag": metrics_by_tag, "records": records}

    write_json(payload, args.output_json)
    print(f"JSON written to : {args.output_json}")

    write_csv(records, args.output_csv)
    print(f"CSV  written to : {args.output_csv}")

    write_json(metrics_by_tag, args.output_metrics_json)
    print(f"Metrics JSON    : {args.output_metrics_json}")

    write_metrics_csv(metrics_by_tag, args.output_metrics_csv)
    print(f"Metrics CSV     : {args.output_metrics_csv}")

    print("\n=== Hard-case tag summary ===")
    print(f"Total records   : {summary['total_records']}")
    for tag in ALL_TAGS:
        n = summary["tag_counts"][tag]
        pct = 0.0 if summary["total_records"] == 0 else n / summary["total_records"] * 100
        print(f"  {tag:<35} {n:>4}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
