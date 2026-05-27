"""Run the real draft benchmark using LLM-reviewed labels.

This evaluates the rule-based matcher against labels_llm_reviewed.json.
The labels are benchmark draft labels and still need spot-checking.
"""

import json
from pathlib import Path

from app.eligibility.rule_matcher import match_patient_to_trial
from eval.evaluate import compute_metrics

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")
RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")


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

        result = match_patient_to_trial(patient, trial)
        predicted_label = result["prediction"]
        gold_label = record["label"]

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
            }
        )

    metrics = compute_metrics(gold_labels, predictions)

    output = {
        "metadata": {
            "label_source": str(LABELS_FILE),
            "label_status": "llm_reviewed_needs_spotcheck",
            "evaluated_pairs": len(gold_labels),
            "skipped_pairs": skipped,
        },
        "metrics": metrics,
        "predictions": prediction_records,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== LLM-Reviewed Draft Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Skipped pairs   : {skipped}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")

    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
