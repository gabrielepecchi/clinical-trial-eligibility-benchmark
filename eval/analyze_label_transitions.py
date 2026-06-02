"""
Task 72: label/prediction transition analysis across benchmark snapshots.

Usage:
    PYTHONPATH=. python eval/analyze_label_transitions.py
    PYTHONPATH=. python eval/analyze_label_transitions.py --snapshots path1.json path2.json
"""

import argparse
import json
import os
import sys


DEFAULT_SNAPSHOT = "data/processed/results_llm_reviewed.json"
REPORT_PATH = "reports/label_transitions.md"
TOP_EXAMPLES = 20


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot not found: '{path}'")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def parse_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_correct(record: dict) -> bool | None:
    gold = get_gold_label(record)
    pred = get_predicted_label(record)
    if not gold or not pred:
        return None
    return gold == pred


# ---------------------------------------------------------------------------
# Snapshot indexing
# ---------------------------------------------------------------------------

def index_snapshot(records: list[dict]) -> tuple[dict[tuple, dict], int]:
    """Return (key→record dict, skipped_count)."""
    index: dict[tuple, dict] = {}
    skipped = 0
    for r in records:
        k = pair_key(r)
        if k is None:
            skipped += 1
            continue
        index[k] = r
    return index, skipped


def label_distribution(records: list[dict], label_getter) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in records:
        lbl = label_getter(r) or "missing"
        dist[lbl] = dist.get(lbl, 0) + 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Single snapshot summary
# ---------------------------------------------------------------------------

def summarize_single_snapshot(snapshot_name: str, records: list[dict]) -> dict:
    index, skipped = index_snapshot(records)
    valid = list(index.values())
    correct = sum(1 for r in valid if is_correct(r) is True)
    incorrect = sum(1 for r in valid if is_correct(r) is False)
    return {
        "snapshot_name": snapshot_name,
        "total": len(records),
        "valid": len(valid),
        "skipped": skipped,
        "correct": correct,
        "incorrect": incorrect,
        "gold_dist": label_distribution(valid, get_gold_label),
        "pred_dist": label_distribution(valid, get_predicted_label),
    }


# ---------------------------------------------------------------------------
# Two-snapshot comparison
# ---------------------------------------------------------------------------

def compare_two_snapshots(
    prev_name: str,
    prev_records: list[dict],
    curr_name: str,
    curr_records: list[dict],
) -> dict:
    prev_index, prev_skipped = index_snapshot(prev_records)
    curr_index, curr_skipped = index_snapshot(curr_records)

    prev_keys = set(prev_index.keys())
    curr_keys = set(curr_index.keys())
    shared = prev_keys & curr_keys
    added = curr_keys - prev_keys
    removed = prev_keys - curr_keys

    pred_flips = []
    gold_flips = []
    correct_to_incorrect = []
    incorrect_to_correct = []
    confidence_changes = []

    for k in sorted(shared):
        pr = prev_index[k]
        cr = curr_index[k]

        pg = get_gold_label(pr)
        cg = get_gold_label(cr)
        pp = get_predicted_label(pr)
        cp = get_predicted_label(cr)
        pc = parse_confidence(pr.get("confidence"))
        cc = parse_confidence(cr.get("confidence"))

        entry = {"patient_id": k[0], "trial_id": k[1],
                 "prev_gold": pg, "curr_gold": cg,
                 "prev_pred": pp, "curr_pred": cp,
                 "prev_conf": pc, "curr_conf": cc}

        if pp != cp:
            pred_flips.append(entry)

        if pg != cg:
            gold_flips.append(entry)

        prev_ok = is_correct(pr)
        curr_ok = is_correct(cr)
        if prev_ok is True and curr_ok is False:
            correct_to_incorrect.append(entry)
        elif prev_ok is False and curr_ok is True:
            incorrect_to_correct.append(entry)

        if pc is not None and cc is not None and pc != cc:
            confidence_changes.append({**entry, "delta": round(cc - pc, 4)})

    return {
        "prev_name": prev_name,
        "curr_name": curr_name,
        "prev_total": len(prev_records),
        "curr_total": len(curr_records),
        "prev_skipped": prev_skipped,
        "curr_skipped": curr_skipped,
        "shared": len(shared),
        "added": len(added),
        "removed": len(removed),
        "pred_flips": pred_flips,
        "gold_flips": gold_flips,
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "confidence_changes": confidence_changes,
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_label_transitions(snapshot_paths: list[str]) -> dict:
    loaded = []
    for path in snapshot_paths:
        data = load_json(path)
        records = extract_predictions(data)
        loaded.append((path, records))

    if len(loaded) == 1:
        path, records = loaded[0]
        return {
            "mode": "single",
            "single": summarize_single_snapshot(path, records),
            "comparisons": [],
        }

    comparisons = []
    for i in range(len(loaded) - 1):
        prev_path, prev_rec = loaded[i]
        curr_path, curr_rec = loaded[i + 1]
        comparisons.append(
            compare_two_snapshots(prev_path, prev_rec, curr_path, curr_rec)
        )

    return {
        "mode": "multi",
        "single": None,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _dist_table(dist: dict[str, int]) -> list[str]:
    lines = ["| Label | Count |", "| --- | --- |"]
    for lbl, cnt in dist.items():
        lines.append(f"| {lbl} | {cnt} |")
    return lines


def _example_rows(entries: list[dict], limit: int = TOP_EXAMPLES) -> list[str]:
    lines = ["| patient_id | trial_id | prev_gold | curr_gold | prev_pred | curr_pred |",
             "| --- | --- | --- | --- | --- | --- |"]
    for e in entries[:limit]:
        lines.append(
            f"| {e['patient_id']} | {e['trial_id']} "
            f"| {e['prev_gold']} | {e['curr_gold']} "
            f"| {e['prev_pred']} | {e['curr_pred']} |"
        )
    return lines


def format_markdown_report(summary: dict) -> str:
    lines = ["# Label Transition Analysis", "", "---", ""]

    if summary["mode"] == "single":
        s = summary["single"]
        lines += [
            "**Note:** Transition analysis requires at least two snapshots.  ",
            "Only one snapshot was provided. Showing single-snapshot summary.",
            "",
            f"**Snapshot:** `{s['snapshot_name']}`  ",
            f"**Total records:** {s['total']}  ",
            f"**Valid records (patient_id + trial_id present):** {s['valid']}  ",
            f"**Skipped (missing patient_id or trial_id):** {s['skipped']}  ",
            f"**Correct predictions:** {s['correct']}  ",
            f"**Incorrect predictions:** {s['incorrect']}",
            "",
            "### Gold Label Distribution", "",
        ]
        lines += _dist_table(s["gold_dist"])
        lines += ["", "### Predicted Label Distribution", ""]
        lines += _dist_table(s["pred_dist"])
        lines.append("")
        return "\n".join(lines)

    # Multi-snapshot
    for comp in summary["comparisons"]:
        lines += [
            f"## Comparison: `{os.path.basename(comp['prev_name'])}` → "
            f"`{os.path.basename(comp['curr_name'])}`",
            "",
            f"| | {os.path.basename(comp['prev_name'])} | "
            f"{os.path.basename(comp['curr_name'])} |",
            "| --- | --- | --- |",
            f"| Total records | {comp['prev_total']} | {comp['curr_total']} |",
            f"| Skipped | {comp['prev_skipped']} | {comp['curr_skipped']} |",
            "",
            f"- **Shared pairs:** {comp['shared']}",
            f"- **Added pairs:** {comp['added']}",
            f"- **Removed pairs:** {comp['removed']}",
            f"- **Prediction flips:** {len(comp['pred_flips'])}",
            f"- **Gold label flips:** {len(comp['gold_flips'])}",
            f"- **Correct → Incorrect:** {len(comp['correct_to_incorrect'])}",
            f"- **Incorrect → Correct:** {len(comp['incorrect_to_correct'])}",
            f"- **Confidence changed:** {len(comp['confidence_changes'])}",
            "",
        ]

        if comp["pred_flips"]:
            lines += [f"### Prediction Flips (top {TOP_EXAMPLES})", ""]
            lines += _example_rows(comp["pred_flips"])
            lines.append("")

        if comp["correct_to_incorrect"]:
            lines += [f"### Correct → Incorrect (top {TOP_EXAMPLES})", ""]
            lines += _example_rows(comp["correct_to_incorrect"])
            lines.append("")

        if comp["incorrect_to_correct"]:
            lines += [f"### Incorrect → Correct (top {TOP_EXAMPLES})", ""]
            lines += _example_rows(comp["incorrect_to_correct"])
            lines.append("")

        if comp["gold_flips"]:
            lines += [f"### Gold Label Flips (top {TOP_EXAMPLES})", ""]
            lines += _example_rows(comp["gold_flips"])
            lines.append("")

        lines += ["---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Label transition analysis.")
    parser.add_argument(
        "--snapshots", nargs="+", default=None,
        help="Paths to results JSON snapshots (ordered oldest to newest).",
    )
    parser.add_argument(
        "--output", default=REPORT_PATH,
        help=f"Output Markdown report path (default: {REPORT_PATH}).",
    )
    args = parser.parse_args()

    snapshot_paths = args.snapshots or [DEFAULT_SNAPSHOT]

    try:
        summary = analyze_label_transitions(snapshot_paths)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    report = format_markdown_report(summary)

    try:
        write_text(report, args.output)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Snapshots loaded: {len(snapshot_paths)}")
    print(f"Comparisons made: {len(summary['comparisons'])}")
    print(f"Report          : {args.output}")


if __name__ == "__main__":
    main()
