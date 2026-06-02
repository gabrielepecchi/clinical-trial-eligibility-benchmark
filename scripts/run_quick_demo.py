"""
run_quick_demo.py — Task 55: Lightweight quick demo.

Evaluates a small subset (~10 patient-trial pairs) from labels_llm_reviewed.json
using the existing matcher, prints a compact terminal summary, and writes a
simple HTML report to reports/quick_demo.html.

Usage:
    PYTHONPATH=. python scripts/run_quick_demo.py
    PYTHONPATH=. python scripts/run_quick_demo.py --n 10 --seed 42
"""

import json
import os
import sys
import argparse
import html
from collections import Counter

DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_PATIENTS = "data/processed/patient_cases.json"
DEFAULT_TRIALS = "data/processed/trial_cases.json"
DEFAULT_OUTPUT = "reports/quick_demo.html"
DEFAULT_N = 10
DEFAULT_SEED = 42

DISCLAIMER = (
    "DISCLAIMER: This demo uses fully synthetic patient profiles and draft "
    "LLM-reviewed benchmark labels. It is a research/portfolio benchmark tool only. "
    "It is NOT validated for clinical use and must NOT be used for real patient care "
    "or clinical trial eligibility decisions."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str, required: bool = True) -> object:
    """Load JSON from path. If required=False and file missing, return None."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_list(data, list_keys=("patients", "trials", "records")) -> list:
    """Return list from data if it's a list or wrapped in a known key."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in list_keys:
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


# ---------------------------------------------------------------------------
# Subset selection — deterministic, no hardcoding
# ---------------------------------------------------------------------------

def select_subset(labels: list, n: int, seed: int) -> list:
    """
    Select up to n labels deterministically using a simple modular stride.
    No patient_id or trial_id is hardcoded.
    """
    if len(labels) <= n:
        return list(labels)
    # Deterministic stride-based selection seeded by the seed value
    step = max(1, len(labels) // n)
    offset = seed % step
    selected = []
    i = offset
    while len(selected) < n and i < len(labels):
        selected.append(labels[i])
        i += step
    return selected


# ---------------------------------------------------------------------------
# Matcher invocation
# ---------------------------------------------------------------------------

def run_matcher(patient: dict, trial: dict) -> dict:
    """Call match_patient_to_trial and return the result dict."""
    from app.eligibility.rule_matcher import match_patient_to_trial
    return match_patient_to_trial(patient, trial)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(evaluated: list) -> dict:
    """Compute accuracy and macro F1 from evaluated pairs."""
    labels_seen = set()
    for e in evaluated:
        labels_seen.add(e["gold_label"])
        labels_seen.add(e["predicted"])

    tp: dict = Counter()
    fp: dict = Counter()
    fn: dict = Counter()

    correct = 0
    for e in evaluated:
        g = e["gold_label"]
        p = e["predicted"]
        if g == p:
            tp[g] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[g] += 1

    total = len(evaluated)
    accuracy = correct / total if total > 0 else None

    f1s = []
    for lbl in labels_seen:
        t = tp[lbl]
        precision = t / (t + fp[lbl]) if (t + fp[lbl]) > 0 else 0.0
        recall = t / (t + fn[lbl]) if (t + fn[lbl]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else None

    return {"accuracy": accuracy, "macro_f1": macro_f1, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _h(text) -> str:
    """HTML-escape a value."""
    return html.escape(str(text) if text is not None else "")


def build_html_report(evaluated: list, metrics: dict, skipped: int) -> str:
    """Build a simple self-contained HTML report."""
    acc_str = f"{metrics['accuracy']:.4f}" if metrics["accuracy"] is not None else "n/a"
    f1_str = f"{metrics['macro_f1']:.4f}" if metrics["macro_f1"] is not None else "n/a"

    rows = []
    for e in evaluated:
        correct_class = "correct" if e["gold_label"] == e["predicted"] else "error"
        conf_str = f"{e['confidence']:.2f}" if e["confidence"] is not None else "—"
        rows.append(
            f"<tr class='{correct_class}'>"
            f"<td>{_h(e['patient_id'])}</td>"
            f"<td>{_h(e['trial_id'])}</td>"
            f"<td>{_h(e['gold_label'])}</td>"
            f"<td>{_h(e['predicted'])}</td>"
            f"<td>{conf_str}</td>"
            f"<td>{_h(e['explanation'][:120])}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Quick Demo Report</title>
<style>
  body {{ font-family: sans-serif; max-width: 1000px; margin: 2em auto; color: #222; }}
  .disclaimer {{ background: #fff3cd; border: 1px solid #ffc107; padding: 1em; margin-bottom: 1.5em; border-radius: 4px; }}
  .metrics {{ background: #f8f9fa; border: 1px solid #dee2e6; padding: 1em; margin-bottom: 1.5em; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.4em 0.6em; text-align: left; }}
  th {{ background: #e9ecef; }}
  tr.correct td {{ background: #d4edda; }}
  tr.error td {{ background: #f8d7da; }}
</style>
</head>
<body>
<h1>Quick Demo Report</h1>
<div class="disclaimer"><strong>Disclaimer:</strong> {_h(DISCLAIMER)}</div>
<div class="metrics">
  <strong>Pairs evaluated:</strong> {metrics['total']} &nbsp;|&nbsp;
  <strong>Skipped:</strong> {skipped} &nbsp;|&nbsp;
  <strong>Correct:</strong> {metrics['correct']} &nbsp;|&nbsp;
  <strong>Accuracy:</strong> {acc_str} &nbsp;|&nbsp;
  <strong>Macro F1:</strong> {f1_str}
</div>
<table>
  <thead>
    <tr>
      <th>Patient</th><th>Trial</th><th>Gold</th><th>Predicted</th>
      <th>Confidence</th><th>Explanation</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
</body>
</html>
"""


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick demo benchmark run.")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--patients", default=DEFAULT_PATIENTS)
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of pairs to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Selection seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    labels_data = load_json(args.labels, required=True)
    labels = labels_data if isinstance(labels_data, list) else extract_list(labels_data)

    patients_data = load_json(args.patients, required=False)
    trials_data = load_json(args.trials, required=False)

    patients_list = extract_list(patients_data, ("patients",)) if patients_data else []
    trials_list = extract_list(trials_data, ("trials",)) if trials_data else []

    patient_index = {p["patient_id"]: p for p in patients_list if "patient_id" in p}
    trial_index = {t["trial_id"]: t for t in trials_list if "trial_id" in t}

    print(f"Loaded {len(trials_list)} trials")
    print(f"Loaded {len(patients_list)} patients")

    subset = select_subset(labels, args.n, args.seed)

    evaluated = []
    skipped = 0

    for label_rec in subset:
        pid = label_rec.get("patient_id", "")
        tid = label_rec.get("trial_id", "")
        gold = label_rec.get("label", "")

        patient = patient_index.get(pid)
        trial = trial_index.get(tid)

        if patient is None or trial is None:
            skipped += 1
            continue

        try:
            result = run_matcher(patient, trial)
        except Exception as e:
            print(f"  WARN: matcher error for {pid}/{tid}: {e}", file=sys.stderr)
            skipped += 1
            continue

        predicted = result.get("prediction", "") or result.get("predicted_label", "")
        confidence = result.get("confidence")
        explanation = result.get("explanation", "") or result.get("matcher_explanation", "")

        evaluated.append({
            "patient_id": pid,
            "trial_id": tid,
            "gold_label": gold,
            "predicted": predicted,
            "confidence": confidence,
            "explanation": explanation,
        })

    print(f"Evaluated {len(evaluated)} pairs")
    print(f"Skipped {skipped} pairs")

    if not evaluated:
        print("No pairs evaluated — check that patient_cases.json and trial_cases.json exist.")
        sys.exit(0)

    metrics = compute_metrics(evaluated)

    acc_str = f"{metrics['accuracy']:.4f}" if metrics["accuracy"] is not None else "n/a"
    f1_str = f"{metrics['macro_f1']:.4f}" if metrics["macro_f1"] is not None else "n/a"
    print(f"Accuracy: {acc_str}")
    print(f"Macro F1: {f1_str}")

    report_html = build_html_report(evaluated, metrics, skipped)
    write_text(report_html, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
