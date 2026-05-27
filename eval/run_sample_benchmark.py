"""Run the sample benchmark: match patients to trials and evaluate predictions."""

import json
from pathlib import Path

from app.eligibility.rule_matcher import match_patient_to_trial
from eval.evaluate import compute_metrics

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PATIENTS_FILE = Path("data/processed/patient_cases_sample.json")
TRIALS_FILE = Path("data/processed/trial_cases_sample.json")
LABELS_FILE = Path("data/processed/labels_sample.json")
RESULTS_FILE = Path("data/processed/results_sample.json")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # Load data
    patients = load_json(PATIENTS_FILE)
    trials = load_json(TRIALS_FILE)
    labels = load_json(LABELS_FILE)

    # Index patients and trials by ID for quick lookup
    patient_index = {p["patient_id"]: p for p in patients}
    trial_index = {t["trial_id"]: t for t in trials}

    # Run predictions for each label record
    gold_labels: list[str] = []
    predictions: list[str] = []
    prediction_records: list[dict] = []

    for record in labels:
        patient_id = record["patient_id"]
        trial_id = record["trial_id"]
        gold_label = record["label"]

        patient = patient_index.get(patient_id)
        trial = trial_index.get(trial_id)

        if patient is None:
            print(f"WARNING: patient {patient_id!r} not found — skipping")
            continue
        if trial is None:
            print(f"WARNING: trial {trial_id!r} not found — skipping")
            continue

        result = match_patient_to_trial(patient, trial)
        predicted_label = result["prediction"]

        gold_labels.append(gold_label)
        predictions.append(predicted_label)

        prediction_records.append({
            "patient_id": patient_id,
            "trial_id": trial_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "confidence": result["confidence"],
            "matched_facts": result["matched_facts"],
            "blocking_criteria": result["blocking_criteria"],
            "uncertain_criteria": result["uncertain_criteria"],
            "explanation": result["explanation"],
        })

    # Compute metrics
    metrics = compute_metrics(gold_labels, predictions)

    # Print summary
    print("\n=== Sample Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")
    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")

    # Save results
    output = {
        "metrics": metrics,
        "predictions": prediction_records,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
