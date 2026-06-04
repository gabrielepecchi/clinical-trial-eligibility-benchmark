"""Run the real draft benchmark using LLM-reviewed labels.

This evaluates the rule-based matcher against labels_llm_reviewed.json.
The labels are benchmark draft labels and still need spot-checking.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from app.eligibility.rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from app.eligibility.evidence_span import extract_criterion_evidence
from eval.evaluate import compute_metrics

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")
RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")
RESULTS_CSV_FILE = Path("data/processed/results_llm_reviewed.csv")
CRITERION_CSV_FILE = Path("data/processed/criterion_level_results.csv")
CRITERION_TYPE_JSON_FILE = Path("data/processed/criterion_type_summary.json")
CRITERION_TYPE_CSV_FILE = Path("data/processed/criterion_type_summary.csv")


_CSV_FIELDNAMES = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "correct", "label_status", "confidence",
    "matched_facts", "blocking_criteria", "uncertain_criteria",
    "matcher_explanation", "gold_rationale", "reasoning_trace",
]


def build_reasoning_trace(
    predicted_label: str,
    matched_facts: list[str] | None,
    blocking_criteria: list[str] | None,
    uncertain_criteria: list[str] | None,
    explanation: str,
) -> list[str]:
    """Build a structured reasoning trace from existing prediction fields."""
    trace: list[str] = [f"predicted: {predicted_label}"]
    for f in matched_facts or []:
        trace.append(f"matched_fact: {f}")
    for c in blocking_criteria or []:
        trace.append(f"blocking_criterion: {c}")
    for c in uncertain_criteria or []:
        trace.append(f"uncertain_criterion: {c}")
    if explanation:
        trace.append(f"explanation: {explanation}")
    return trace


def build_llm_reviewed_csv_rows(prediction_records: list[dict]) -> list[dict]:
    rows = []
    for r in prediction_records:
        gold = r.get("gold_label", "")
        predicted = r.get("predicted_label", "")
        rows.append({
            "patient_id": r.get("patient_id", ""),
            "trial_id": r.get("trial_id", ""),
            "gold_label": gold,
            "predicted_label": predicted,
            "correct": gold == predicted,
            "label_status": r.get("label_status", ""),
            "confidence": r.get("confidence", ""),
            "matched_facts": "; ".join(r.get("matched_facts") or []),
            "blocking_criteria": "; ".join(r.get("blocking_criteria") or []),
            "uncertain_criteria": "; ".join(r.get("uncertain_criteria") or []),
            "matcher_explanation": r.get("matcher_explanation", ""),
            "gold_rationale": r.get("gold_rationale", ""),
            "reasoning_trace": " | ".join(r.get("reasoning_trace") or []),
        })
    return rows


def write_llm_reviewed_csv_rows(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_CRITERION_CSV_FIELDNAMES = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "criterion", "criterion_type", "decision", "reason",
    "patient_evidence", "trial_evidence",
    "patient_span_start", "patient_span_end",
    "trial_span_start", "trial_span_end",
]


_PATIENT_TEXT_FIELDS: list[str] = [
    "text", "clinical_notes", "notes", "description", "patient_text",
    "case_text", "history", "clinical_history", "presentation", "summary",
]

_TRIAL_TEXT_FIELDS: list[str] = [
    "criteria_text", "eligibility_criteria", "inclusion_criteria",
    "exclusion_criteria", "criteria", "description", "detailed_description",
    "brief_summary", "summary",
]


def _get_patient_text(patient: dict) -> str:
    """Collect patient narrative text from known fields."""
    parts: list[str] = []
    for field in _PATIENT_TEXT_FIELDS:
        val = patient.get(field, "")
        if val:
            parts.append(str(val).strip())
    return " ".join(parts)


def _get_trial_text(trial: dict) -> str:
    """Collect trial eligibility/criteria text from known fields."""
    parts: list[str] = []
    for field in _TRIAL_TEXT_FIELDS:
        val = trial.get(field, "")
        if isinstance(val, list):
            parts.extend(str(v).strip() for v in val if v)
        elif val:
            parts.append(str(val).strip())
    return " ".join(parts)


def build_criterion_level_csv_rows(prediction_records: list[dict]) -> list[dict]:
    rows = []
    for r in prediction_records:
        for cr in r.get("criterion_results") or []:
            rows.append({
                "patient_id": r.get("patient_id", ""),
                "trial_id": r.get("trial_id", ""),
                "gold_label": r.get("gold_label", ""),
                "predicted_label": r.get("predicted_label", ""),
                "criterion": cr.get("criterion_text", ""),
                "criterion_type": cr.get("criterion_type", ""),
                "decision": cr.get("decision", ""),
                "reason": cr.get("reason", ""),
                "patient_evidence": cr.get("patient_evidence", ""),
                "trial_evidence": cr.get("trial_evidence", ""),
                "patient_span_start": cr.get("patient_span_start", ""),
                "patient_span_end": cr.get("patient_span_end", ""),
                "trial_span_start": cr.get("trial_span_start", ""),
                "trial_span_end": cr.get("trial_span_end", ""),
            })
    return rows


def write_criterion_level_csv_rows(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CRITERION_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_criterion_type_summary(prediction_records: list[dict]) -> list[dict]:
    """Aggregate criterion-level decisions by criterion_type."""
    # decision counts per type
    decision_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # pairs that have at least one criterion of this type
    pair_correct: dict[str, list[bool]] = defaultdict(list)

    for r in prediction_records:
        correct = r.get("gold_label", "") == r.get("predicted_label", "")
        seen_types: set[str] = set()
        for cr in r.get("criterion_results") or []:
            ctype = cr.get("criterion_type") or "unknown"
            decision = cr.get("decision") or "unknown"
            decision_counts[ctype][decision] += 1
            decision_counts[ctype]["total_criteria"] += 1
            if ctype not in seen_types:
                pair_correct[ctype].append(correct)
                seen_types.add(ctype)

    rows = []
    for ctype, counts in sorted(decision_counts.items()):
        pairs = pair_correct.get(ctype, [])
        total_pairs = len(pairs)
        correct_pairs = sum(pairs)
        row: dict = {
            "criterion_type": ctype,
            "total_criteria": counts.get("total_criteria", 0),
            "decision_met": counts.get("met", 0),
            "decision_not_met": counts.get("not_met", 0),
            "decision_unclear": counts.get("unclear", 0),
        }
        if "not_applicable" in counts:
            row["decision_not_applicable"] = counts["not_applicable"]
        known = {"total_criteria", "met", "not_met", "unclear", "not_applicable"}
        for k, v in counts.items():
            if k not in known:
                row[f"decision_{k}"] = v
        row["correct_pairs"] = correct_pairs
        row["total_pairs"] = total_pairs
        row["pair_accuracy"] = correct_pairs / total_pairs if total_pairs else 0.0
        rows.append(row)
    return rows


_CRITERION_TYPE_CSV_BASE_FIELDNAMES = [
    "criterion_type", "total_criteria",
    "decision_met", "decision_not_met", "decision_unclear",
    "correct_pairs", "total_pairs", "pair_accuracy",
]


def write_criterion_type_summary_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        fieldnames = _CRITERION_TYPE_CSV_BASE_FIELDNAMES
    else:
        extra = sorted({k for r in rows for k in r if k not in _CRITERION_TYPE_CSV_BASE_FIELDNAMES})
        fieldnames = _CRITERION_TYPE_CSV_BASE_FIELDNAMES[:5] + extra + _CRITERION_TYPE_CSV_BASE_FIELDNAMES[5:]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_criterion_type_summary(rows: list[dict]) -> str:
    lines = ["\n=== Criterion Type Summary ==="]
    if not rows:
        lines.append("  (no criterion data)")
        return "\n".join(lines)
    header = f"  {'criterion_type':<30} {'total':>6} {'met':>6} {'not_met':>8} {'unclear':>8} {'pair_acc':>9}"
    lines.append(header)
    for r in rows:
        lines.append(
            f"  {r['criterion_type']:<30} "
            f"{r['total_criteria']:>6} "
            f"{r['decision_met']:>6} "
            f"{r['decision_not_met']:>8} "
            f"{r['decision_unclear']:>8} "
            f"{r['pair_accuracy']:>9.3f}"
        )
    return "\n".join(lines)


def build_safety_uncertainty_summary(prediction_records: list[dict]) -> dict:
    """Compute safety and uncertainty error counts and rates."""
    total = len(prediction_records)
    unsafe = 0
    uncertainty = 0
    conservative = 0
    gold_unclear = 0
    true_unclear = 0
    predicted_unclear = 0
    overcommitted = 0

    for r in prediction_records:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold == "not_eligible" and pred == "eligible":
            unsafe += 1
        if gold == "unclear" and pred in {"eligible", "not_eligible"}:
            uncertainty += 1
        if gold == "eligible" and pred == "not_eligible":
            conservative += 1
        if gold == "unclear":
            gold_unclear += 1
            if pred == "unclear":
                true_unclear += 1
            if pred in {"eligible", "not_eligible"}:
                overcommitted += 1
        if pred == "unclear":
            predicted_unclear += 1

    return {
        "total_predictions": total,
        "unsafe_eligible_errors": unsafe,
        "uncertainty_errors": uncertainty,
        "overly_conservative_errors": conservative,
        "unclear_recall": true_unclear / gold_unclear if gold_unclear else 0,
        "unclear_precision": true_unclear / predicted_unclear if predicted_unclear else 0,
        "overcommitment_rate": overcommitted / gold_unclear if gold_unclear else 0,
    }


def build_benchmark_output(
    metadata: dict,
    metrics: dict,
    safety_uncertainty_summary: dict,
    error_severity_summary: dict,
    prediction_records: list[dict],
    criterion_type_summary: list[dict] | None = None,
) -> dict:
    """Assemble the final benchmark output dict.

    Keep prediction_records as the fifth argument for backward compatibility
    with existing tests and callers.
    """
    output = {
        "metadata": metadata,
        "metrics": metrics,
        "safety_uncertainty_summary": safety_uncertainty_summary,
        "error_severity_summary": error_severity_summary,
        "predictions": prediction_records,
    }
    if criterion_type_summary is not None:
        output["criterion_type_summary"] = criterion_type_summary
    return output


def format_safety_uncertainty_summary(s: dict) -> str:
    """Format safety and uncertainty summary as a printable string."""
    lines = [
        "\n=== Safety & Uncertainty Summary ===",
        f"Total predictions    : {s['total_predictions']}",
        f"Unsafe eligible errors     : {s['unsafe_eligible_errors']}",
        f"Overly conservative errors : {s['overly_conservative_errors']}",
        f"Uncertainty errors         : {s['uncertainty_errors']}",
        f"Unclear recall             : {s['unclear_recall']:.3f}",
        f"Unclear precision          : {s['unclear_precision']:.3f}",
        f"Overcommitment rate        : {s['overcommitment_rate']:.3f}",
    ]
    return "\n".join(lines)


def format_error_severity_summary(s: dict) -> str:
    """Format error severity summary as a printable string."""
    lines = [
        "\n=== Error Severity Summary ===",
        f"Total errors         : {s['total_errors']}",
        f"Critical errors      : {s['critical_errors']}",
        f"Major errors         : {s['major_errors']}",
        f"Minor errors         : {s['minor_errors']}",
        f"Critical error rate  : {s['critical_error_rate']:.3f}",
        f"Major error rate     : {s['major_error_rate']:.3f}",
        f"Minor error rate     : {s['minor_error_rate']:.3f}",
    ]
    return "\n".join(lines)


_LABEL_ORDER = ["eligible", "not_eligible", "unclear"]


def build_confusion_matrix(gold_labels: list[str], predictions: list[str]) -> dict[str, dict[str, int]]:
    """Return a nested dict: matrix[true_label][predicted_label] = count."""
    matrix: dict[str, dict[str, int]] = {
        label: {other: 0 for other in _LABEL_ORDER} for label in _LABEL_ORDER
    }
    for gold, pred in zip(gold_labels, predictions):
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def format_confusion_matrix(matrix: dict[str, dict[str, int]]) -> str:
    """Format confusion matrix as a readable terminal table."""
    col_w = 15
    header = " " * (col_w + 2) + "".join(f"{'pred_' + l:>{col_w}}" for l in _LABEL_ORDER)
    lines = ["\n=== Confusion Matrix ===", header]
    for true_label in _LABEL_ORDER:
        row_label = f"true_{true_label}"
        counts = "".join(f"{matrix[true_label][pred]:>{col_w}}" for pred in _LABEL_ORDER)
        lines.append(f"{row_label:<{col_w + 2}}{counts}")
    return "\n".join(lines)


def build_error_severity_summary(prediction_records: list[dict]) -> dict:
    """Compute error severity counts and rates."""
    total = len(prediction_records)
    total_errors = 0
    critical = 0
    major = 0
    minor = 0

    for r in prediction_records:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold != pred:
            total_errors += 1
        if gold == "not_eligible" and pred == "eligible":
            critical += 1
        if (gold == "unclear" and pred in {"eligible", "not_eligible"}) or \
           (gold in {"eligible", "not_eligible"} and pred == "unclear"):
            major += 1
        if (gold == "eligible" and pred == "not_eligible") or \
           (gold == "not_eligible" and pred == "unclear"):
            minor += 1

    return {
        "total_predictions": total,
        "total_errors": total_errors,
        "critical_errors": critical,
        "major_errors": major,
        "minor_errors": minor,
        "critical_error_rate": critical / total if total else 0,
        "major_error_rate": major / total if total else 0,
        "minor_error_rate": minor / total if total else 0,
    }


def load_json(path: Path) -> list[dict]:
    """Load a JSON list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    patients = load_json(PATIENTS_FILE)
    trials = load_json(TRIALS_FILE)
    labels = load_json(LABELS_FILE)

    patient_index = {patient["patient_id"]: patient for patient in patients}
    trial_index = {trial["trial_id"]: trial for trial in trials}

    gold_labels: list[str] = []
    predictions: list[str] = []
    prediction_records: list[dict] = []
    skipped = 0

    for record in labels:
        patient_id = record["patient_id"]
        trial_id = record["trial_id"]

        patient = patient_index.get(patient_id)
        trial = trial_index.get(trial_id)

        if patient is None or trial is None:
            skipped += 1
            continue

        _TRIAL_META_FIELDS = [
            "title", "brief_title", "official_title", "summary", "brief_summary",
            "description", "detailed_description", "intervention", "intervention_name",
            "intervention_type", "interventions", "keywords", "conditions",
        ]
        enriched_trial = {**trial, **{f: trial[f] for f in _TRIAL_META_FIELDS if f in trial}}
        result = match_patient_to_trial(patient, enriched_trial)
        predicted_label = result["prediction"]
        gold_label = record["label"]

        patient_text = _get_patient_text(patient)
        trial_text = _get_trial_text(enriched_trial)

        criterion_results = []
        for cr in match_patient_to_trial_criteria(patient, enriched_trial):
            evidence = extract_criterion_evidence(
                patient_text,
                trial_text,
                cr.criterion_text,
                cr.reason or "",
            )
            criterion_results.append({
                "criterion_text": cr.criterion_text,
                "criterion_type": cr.criterion_type.value,
                "decision": cr.decision.value,
                "reason": cr.reason,
                **evidence,
            })

        reasoning_trace = build_reasoning_trace(
            predicted_label,
            result["matched_facts"],
            result["blocking_criteria"],
            result["uncertain_criteria"],
            result["explanation"],
        )

        gold_labels.append(gold_label)
        predictions.append(predicted_label)

        prediction_records.append(
            {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "label_status": record.get("label_status", ""),
                "confidence": result["confidence"],
                "matched_facts": result["matched_facts"],
                "blocking_criteria": result["blocking_criteria"],
                "uncertain_criteria": result["uncertain_criteria"],
                "matcher_explanation": result["explanation"],
                "gold_rationale": record.get("rationale", ""),
                "gold_evidence": record.get("evidence", {}),
                "criterion_results": criterion_results,
                "reasoning_trace": reasoning_trace,
            }
        )

    metrics = compute_metrics(gold_labels, predictions)

    safety_summary = build_safety_uncertainty_summary(prediction_records)
    error_summary = build_error_severity_summary(prediction_records)
    criterion_type_summary = build_criterion_type_summary(prediction_records)

    metadata = {
        "label_source": str(LABELS_FILE),
        "label_status": "llm_reviewed_needs_spotcheck",
        "evaluated_pairs": len(gold_labels),
        "skipped_pairs": skipped,
    }

    output = build_benchmark_output(
        metadata, metrics, safety_summary, error_summary, prediction_records, criterion_type_summary
    )

    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_rows = build_llm_reviewed_csv_rows(prediction_records)
    write_llm_reviewed_csv_rows(csv_rows, RESULTS_CSV_FILE)

    criterion_rows = build_criterion_level_csv_rows(prediction_records)
    write_criterion_level_csv_rows(criterion_rows, CRITERION_CSV_FILE)

    CRITERION_TYPE_JSON_FILE.write_text(json.dumps(criterion_type_summary, indent=2), encoding="utf-8")
    write_criterion_type_summary_csv(criterion_type_summary, CRITERION_TYPE_CSV_FILE)

    print("\n=== LLM-Reviewed Draft Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Skipped pairs   : {skipped}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")

    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")

    cm = build_confusion_matrix(gold_labels, predictions)
    print(format_confusion_matrix(cm))

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Predictions CSV saved to {RESULTS_CSV_FILE}")
    print(f"Criterion-level CSV saved to {CRITERION_CSV_FILE}")
    print(f"Criterion type summary saved to {CRITERION_TYPE_JSON_FILE}")
    print(f"Criterion type CSV saved to {CRITERION_TYPE_CSV_FILE}")

    print(format_safety_uncertainty_summary(safety_summary))
    print(format_error_severity_summary(error_summary))
    print(format_criterion_type_summary(criterion_type_summary))


if __name__ == "__main__":
    main()
