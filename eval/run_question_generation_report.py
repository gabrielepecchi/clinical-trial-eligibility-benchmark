"""
run_question_generation_report.py — Task 70: Clarification question generation.

Generates clarification questions for uncertain eligibility criteria using
existing matcher uncertainty signals. Questions are derived from
uncertain_criteria fields only — no clinical facts are invented.

Usage:
    PYTHONPATH=. python eval/run_question_generation_report.py
    PYTHONPATH=. python eval/run_question_generation_report.py --input PATH --output PATH
"""

import json
import os
import sys
import argparse
import re

DEFAULT_INPUT = "data/processed/results_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/question_generation_report.json"

NOTE = (
    "Questions are generated from matcher uncertainty signals (uncertain_criteria fields) "
    "and are not clinical recommendations. They are intended to identify information gaps "
    "in the benchmark patient profiles, not to guide real clinical decisions."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_predictions(data: dict) -> list:
    if not isinstance(data, dict) or "predictions" not in data:
        print("ERROR: results JSON missing 'predictions' key.", file=sys.stderr)
        sys.exit(1)
    preds = data["predictions"]
    if not isinstance(preds, list):
        print("ERROR: 'predictions' is not a list.", file=sys.stderr)
        sys.exit(1)
    return preds


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def _clean_criterion(text: str) -> str:
    """Return a cleaned, lowercase summary of a criterion string."""
    text = text.strip()
    # Remove leading numbering like "1." or "a)"
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = re.sub(r"^[a-z][\.\)]\s*", "", text)
    return text


def generate_questions_for_criterion(criterion: str) -> list:
    """
    Generate one or more clarification questions for a single uncertain criterion.
    Derived only from the criterion text — no clinical facts invented.
    """
    cleaned = _clean_criterion(criterion)
    if not cleaned:
        return []

    questions = []

    # Primary question — always generated
    questions.append(
        f"Can the patient's eligibility for the following criterion be confirmed "
        f"from the clinical record? Criterion: \"{cleaned}\""
    )

    # Supplementary questions based on criterion content signals
    lower = cleaned.lower()

    if re.search(r"\b(age|years?\s+of\s+age|years?\s+old)\b", lower):
        questions.append(
            f"Is the patient's age documented and within the required range for: \"{cleaned}\"?"
        )

    if re.search(r"\b(diagnosis|diagnosed|idiopathic|parkinson)\b", lower):
        questions.append(
            f"Is there a confirmed clinical diagnosis that satisfies: \"{cleaned}\"?"
        )

    if re.search(r"\b(medication|drug|levodopa|rasagiline|selegiline|washout|treatment)\b", lower):
        questions.append(
            f"Is the patient's current or recent medication history documented for: \"{cleaned}\"?"
        )

    if re.search(r"\b(dbs|deep\s+brain\s+stimulat|implant|device|pacemaker)\b", lower):
        questions.append(
            f"Is the patient's device or surgical procedure history documented for: \"{cleaned}\"?"
        )

    if re.search(r"\b(moca|mmse|cognitive|dementia|impairment)\b", lower):
        questions.append(
            f"Is a cognitive assessment score available to evaluate: \"{cleaned}\"?"
        )

    if re.search(r"\b(within\s+\d+|days?|weeks?|months?|years?\s+prior|recent|washout)\b", lower):
        questions.append(
            f"Can the relevant dates or timing be confirmed for: \"{cleaned}\"?"
        )

    if re.search(r"\b(weight|bmi|body\s+mass|hemoglobin|creatinine|lab|blood)\b", lower):
        questions.append(
            f"Is the relevant laboratory or measurement value available for: \"{cleaned}\"?"
        )

    return questions


def generate_record_questions(pred: dict) -> dict | None:
    """
    Return a question record for a prediction with uncertain_criteria,
    or None if there are no uncertain criteria.
    """
    uncertain = pred.get("uncertain_criteria") or []
    if isinstance(uncertain, str):
        uncertain = [uncertain]
    uncertain = [str(c).strip() for c in uncertain if str(c).strip()]

    if not uncertain:
        return None

    all_questions = []
    for criterion in uncertain:
        qs = generate_questions_for_criterion(criterion)
        all_questions.extend(qs)

    return {
        "patient_id": pred.get("patient_id", ""),
        "trial_id": pred.get("trial_id", ""),
        "gold_label": pred.get("gold_label", ""),
        "predicted_label": pred.get("predicted_label", "") or pred.get("prediction", ""),
        "uncertain_criteria": uncertain,
        "clarification_questions": all_questions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clarification question generation report.")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Results JSON path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data = load_json(args.input)
    predictions = extract_predictions(data)

    records = []
    for pred in predictions:
        rec = generate_record_questions(pred)
        if rec is not None:
            records.append(rec)

    total_questions = sum(len(r["clarification_questions"]) for r in records)

    report = {
        "total_predictions": len(predictions),
        "predictions_with_uncertainty": len(records),
        "total_generated_questions": total_questions,
        "note": NOTE,
        "records": records,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Total predictions        : {len(predictions)}")
    print(f"Predictions with uncertainty: {len(records)}")
    print(f"Total generated questions: {total_questions}")
    print(f"Report written           : {args.output}")


if __name__ == "__main__":
    main()
