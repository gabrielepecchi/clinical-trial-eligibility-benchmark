"""
eval/run_inter_rater_analysis.py

Inter-rater / label consistency analysis for the clinical trial eligibility benchmark.

Usage:
    PYTHONPATH=. python eval/run_inter_rater_analysis.py

Reads available label sources from data/processed/ and computes pairwise agreement
metrics. Writes a Markdown report to reports/inter_rater_analysis.md.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

DEFAULT_LABEL_SOURCES = [
    "data/processed/labels_llm_reviewed.json",
    "data/processed/labels_seed.json",
    "data/processed/labels_sample.json",
    "data/processed/labels_reviewed.json",
]

DEFAULT_REPORT_PATH = "reports/inter_rater_analysis.md"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def load_json(path: str) -> object:
    """Load and return JSON content from *path*. Raises on malformed JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_label_source(path: str) -> List[dict]:
    """
    Load a label source file and return a list of record dicts.

    Each record must have at minimum: patient_id, trial_id, label.
    Raises ValueError if the file is structurally invalid.
    """
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(raw).__name__}")
    for i, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {i} is not an object")
        for field in ("patient_id", "trial_id", "label"):
            if field not in record:
                raise ValueError(f"{path}: record {i} missing field '{field}'")
    return raw


def index_labels(records: List[dict]) -> Dict[Tuple[str, str], str]:
    """
    Build a dict mapping (patient_id, trial_id) -> label for the given records.

    Only records whose label is in VALID_LABELS are included; others are skipped.
    """
    index: Dict[Tuple[str, str], str] = {}
    for record in records:
        key = (str(record["patient_id"]), str(record["trial_id"]))
        label = record.get("label", "")
        if label in VALID_LABELS:
            index[key] = label
    return index


def find_available_label_sources(paths: List[str]) -> List[Tuple[str, List[dict]]]:
    """
    Return a list of (path, records) for each path in *paths* that exists on disk.

    Files that are present but malformed raise ValueError (propagated to caller).
    """
    available = []
    for path in paths:
        if os.path.isfile(path):
            records = load_label_source(path)
            available.append((path, records))
    return available


def percent_agreement(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Compute simple percent agreement between two parallel label lists.

    Returns a float in [0.0, 1.0]. Returns 0.0 for empty lists.
    """
    if not labels_a:
        return 0.0
    matches = sum(a == b for a, b in zip(labels_a, labels_b))
    return matches / len(labels_a)


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Compute Cohen's kappa for two parallel label lists.

    Returns a float in [-1.0, 1.0]. Returns 0.0 for empty lists or when
    expected agreement equals 1.0 (degenerate case).
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    all_labels = sorted(VALID_LABELS)
    count_a: Dict[str, int] = {lbl: 0 for lbl in all_labels}
    count_b: Dict[str, int] = {lbl: 0 for lbl in all_labels}
    observed_agree = 0

    for a, b in zip(labels_a, labels_b):
        if a in count_a:
            count_a[a] += 1
        if b in count_b:
            count_b[b] += 1
        if a == b:
            observed_agree += 1

    p_o = observed_agree / n
    p_e = sum((count_a[lbl] / n) * (count_b[lbl] / n) for lbl in all_labels)

    if p_e >= 1.0:
        return 0.0

    return (p_o - p_e) / (1.0 - p_e)


def compare_label_sources(
    source_a: Tuple[str, List[dict]],
    source_b: Tuple[str, List[dict]],
) -> dict:
    """
    Compare two label sources and return a comparison summary dict.

    Keys:
        source_a_path, source_b_path,
        shared_pairs, percent_agreement, cohen_kappa,
        disagreement_counts (dict of "label_a->label_b" -> count),
        top_disagreements (list of dicts with patient_id, trial_id, source_a_label, source_b_label)
    """
    path_a, records_a = source_a
    path_b, records_b = source_b

    index_a = index_labels(records_a)
    index_b = index_labels(records_b)

    shared_keys = sorted(set(index_a.keys()) & set(index_b.keys()))

    labels_a = [index_a[k] for k in shared_keys]
    labels_b = [index_b[k] for k in shared_keys]

    disagreement_counts: Dict[str, int] = {}
    top_disagreements = []

    for key, la, lb in zip(shared_keys, labels_a, labels_b):
        if la != lb:
            pair_key = f"{la}->{lb}"
            disagreement_counts[pair_key] = disagreement_counts.get(pair_key, 0) + 1
            top_disagreements.append(
                {
                    "patient_id": key[0],
                    "trial_id": key[1],
                    "source_a_label": la,
                    "source_b_label": lb,
                }
            )

    # Keep at most 10 top disagreement examples
    top_disagreements = top_disagreements[:10]

    return {
        "source_a_path": path_a,
        "source_b_path": path_b,
        "shared_pairs": len(shared_keys),
        "percent_agreement": percent_agreement(labels_a, labels_b),
        "cohen_kappa": cohen_kappa(labels_a, labels_b),
        "disagreement_counts": disagreement_counts,
        "top_disagreements": top_disagreements,
    }


def build_inter_rater_summary(
    sources: List[Tuple[str, List[dict]]],
) -> dict:
    """
    Build a full inter-rater summary for all pairwise combinations of sources.

    Returns a dict with keys:
        source_paths, comparisons
    """
    comparisons = []
    for src_a, src_b in combinations(sources, 2):
        comparisons.append(compare_label_sources(src_a, src_b))

    return {
        "source_paths": [s[0] for s in sources],
        "comparisons": comparisons,
    }


def format_markdown_report(summary: dict) -> str:
    """Render the inter-rater summary as a Markdown string."""
    lines = [
        "# Inter-Rater Label Consistency Analysis",
        "",
        "## Label Sources Found",
        "",
    ]
    for path in summary["source_paths"]:
        lines.append(f"- `{path}`")
    lines.append("")

    if not summary["comparisons"]:
        lines.append("_No pairwise comparisons available (fewer than 2 sources)._")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "## Pairwise Comparisons",
        "",
    ]

    for comp in summary["comparisons"]:
        name_a = os.path.basename(comp["source_a_path"])
        name_b = os.path.basename(comp["source_b_path"])
        lines += [
            f"### {name_a} vs {name_b}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Shared pairs | {comp['shared_pairs']} |",
            f"| Percent agreement | {comp['percent_agreement']:.3f} |",
            f"| Cohen's kappa | {comp['cohen_kappa']:.3f} |",
            "",
        ]

        if comp["disagreement_counts"]:
            lines += [
                "**Disagreement breakdown:**",
                "",
                "| Pair (A→B) | Count |",
                "|------------|-------|",
            ]
            for pair, count in sorted(
                comp["disagreement_counts"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"| `{pair}` | {count} |")
            lines.append("")

        if comp["top_disagreements"]:
            lines += [
                "**Top disagreement examples (up to 10):**",
                "",
                "| patient_id | trial_id | source_a_label | source_b_label |",
                "|------------|----------|----------------|----------------|",
            ]
            for ex in comp["top_disagreements"]:
                lines.append(
                    f"| {ex['patient_id']} | {ex['trial_id']} "
                    f"| {ex['source_a_label']} | {ex['source_b_label']} |"
                )
            lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        sources = find_available_label_sources(DEFAULT_LABEL_SOURCES)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Label sources found: {len(sources)}")
    for path, records in sources:
        print(f"  {path}  ({len(records)} records)")

    if len(sources) < 2:
        print(
            "\nFewer than 2 label sources available. "
            "No pairwise comparison can be run.\n"
            "Exiting successfully."
        )
        sys.exit(0)

    summary = build_inter_rater_summary(sources)
    report_text = format_markdown_report(summary)
    write_text(report_text, DEFAULT_REPORT_PATH)

    print(f"\nComparisons run: {len(summary['comparisons'])}")
    for comp in summary["comparisons"]:
        name_a = os.path.basename(comp["source_a_path"])
        name_b = os.path.basename(comp["source_b_path"])
        print(
            f"  {name_a} vs {name_b}: "
            f"shared={comp['shared_pairs']}  "
            f"agreement={comp['percent_agreement']:.3f}  "
            f"kappa={comp['cohen_kappa']:.3f}"
        )

    print(f"\nReport written to: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
