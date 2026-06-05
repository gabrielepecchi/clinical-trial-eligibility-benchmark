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

    payload = {"summary": summary, "records": records}

    write_json(payload, args.output_json)
    print(f"JSON written to : {args.output_json}")

    write_csv(records, args.output_csv)
    print(f"CSV  written to : {args.output_csv}")

    print("\n=== Hard-case tag summary ===")
    print(f"Total records   : {summary['total_records']}")
    for tag in ALL_TAGS:
        n = summary["tag_counts"][tag]
        pct = 0.0 if summary["total_records"] == 0 else n / summary["total_records"] * 100
        print(f"  {tag:<35} {n:>4}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
