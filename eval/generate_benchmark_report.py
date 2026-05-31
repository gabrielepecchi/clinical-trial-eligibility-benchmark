"""Generate a local HTML report for the LLM-reviewed benchmark."""

import json
from pathlib import Path

RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")
ERROR_ANALYSIS_FILE = Path("data/processed/error_analysis_llm_reviewed.json")
CRITERION_TYPE_FILE = Path("data/processed/criterion_type_summary.json")
REPORT_FILE = Path("reports/benchmark_report.html")

LABELS = ["eligible", "not_eligible", "unclear"]


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Section builders ────────────────────────────────────────────────────────

def section(title: str, body: str) -> str:
    return f"<section>\n<h2>{esc(title)}</h2>\n{body}\n</section>\n"


def kv_table(rows: list[tuple[str, str]]) -> str:
    inner = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{inner}</table>"


def generic_table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# ── HTML template ────────────────────────────────────────────────────────────

CSS = """
body { font-family: sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h1 { color: #c0392b; }
h2 { border-bottom: 2px solid #ddd; padding-bottom: .3rem; margin-top: 2rem; }
.disclaimer { background: #fff3cd; border: 1px solid #ffc107; padding: .8rem 1rem; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .9rem; }
th, td { border: 1px solid #ccc; padding: .4rem .7rem; text-align: left; }
thead { background: #f0f0f0; }
table.kv th { width: 40%; background: #f8f8f8; }
.good { color: #27ae60; font-weight: bold; }
.bad  { color: #c0392b; font-weight: bold; }
.warn { color: #e67e22; font-weight: bold; }
pre { background: #f4f4f4; padding: .6rem; border-radius: 4px; white-space: pre-wrap; font-size: .82rem; }
"""


def html_page(body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>Benchmark Report</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )


# ── Section renderers ────────────────────────────────────────────────────────

def render_title_and_disclaimer(metadata: dict) -> str:
    label_source = esc(metadata.get("label_source", "unknown"))
    html = (
        "<h1>LLM-Reviewed Benchmark Report</h1>\n"
        "<div class='disclaimer'>"
        "<strong>⚠ Disclaimer:</strong> This report is based on <strong>synthetic patient cases</strong> "
        "and <strong>draft LLM-reviewed labels</strong> that have not been fully validated. "
        "Results are for research and development purposes only. "
        "<strong>Not for clinical use.</strong> "
        f"Label source: <code>{label_source}</code>"
        "</div>\n"
    )
    return html


def render_metadata(metadata: dict) -> str:
    rows = [(k, str(v)) for k, v in metadata.items()]
    return section("Benchmark Metadata", kv_table(rows))


def render_global_metrics(metrics: dict) -> str:
    top_rows = [
        ("Accuracy", pct(metrics.get("accuracy", 0))),
        ("Macro Precision", fmt(metrics.get("macro_precision", 0))),
        ("Macro Recall", fmt(metrics.get("macro_recall", 0))),
        ("Macro F1", fmt(metrics.get("macro_f1", 0))),
    ]
    body = kv_table(top_rows)

    per_class = metrics.get("per_class", {})
    if per_class:
        headers = ["Label", "Precision", "Recall", "F1"]
        pc_rows = [
            [label, fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1"])]
            for label, v in per_class.items()
        ]
        body += "<h3>Per-class metrics</h3>" + generic_table(headers, pc_rows)

    return section("Global Metrics", body)


def render_confusion_matrix(metrics: dict) -> str:
    cm = metrics.get("confusion_matrix")
    if not cm:
        return ""
    gold_labels = list(cm.keys())
    pred_labels = sorted({p for row in cm.values() for p in row})
    headers = ["Gold \\ Predicted"] + pred_labels
    rows = [
        [gold] + [str(cm.get(gold, {}).get(pred, 0)) for pred in pred_labels]
        for gold in gold_labels
    ]
    return section("Confusion Matrix", generic_table(headers, rows))


def render_safety_uncertainty(s: dict) -> str:
    rows = [
        ("Total predictions", str(s.get("total_predictions", 0))),
        ("Unsafe eligible errors (not_eligible → eligible)", str(s.get("unsafe_eligible_errors", 0))),
        ("Overly conservative errors (eligible → not_eligible)", str(s.get("overly_conservative_errors", 0))),
        ("Uncertainty errors (unclear → eligible/not_eligible)", str(s.get("uncertainty_errors", 0))),
        ("Unclear recall", fmt(s.get("unclear_recall", 0))),
        ("Unclear precision", fmt(s.get("unclear_precision", 0))),
        ("Overcommitment rate", fmt(s.get("overcommitment_rate", 0))),
    ]
    return section("Safety & Uncertainty Summary", kv_table(rows))


def render_error_severity(s: dict) -> str:
    total = s.get("total_predictions", 1) or 1
    rows = [
        ("Total predictions", str(s.get("total_predictions", 0))),
        ("Total errors", str(s.get("total_errors", 0))),
        ("Critical errors (not_eligible → eligible)", str(s.get("critical_errors", 0))),
        ("Major errors", str(s.get("major_errors", 0))),
        ("Minor errors", str(s.get("minor_errors", 0))),
        ("Critical error rate", pct(s.get("critical_error_rate", 0))),
        ("Major error rate", pct(s.get("major_error_rate", 0))),
        ("Minor error rate", pct(s.get("minor_error_rate", 0))),
    ]
    return section("Error Severity Summary", kv_table(rows))


def _fmt_cell(v: object) -> str:
    """Format a table cell: round floats to 3 decimals, else str."""
    if isinstance(v, float):
        return fmt(v)
    return str(v) if v is not None else ""


def render_criterion_type_summary(rows: list[dict]) -> str:
    if not rows:
        return section("Criterion Type Summary", "<p>No data.</p>")
    all_keys = list(rows[0].keys())
    table_rows = [[_fmt_cell(r.get(k, "")) for k in all_keys] for r in rows]
    return section("Criterion Type Summary", generic_table(all_keys, table_rows))


def render_errors_by_type(error_analysis: list[dict] | None, predictions: list[dict]) -> str:
    body = ""

    # Table 1: error_type counts from error_analysis records
    if error_analysis and isinstance(error_analysis, list):
        type_counts: dict[str, int] = {}
        for r in error_analysis:
            etype = r.get("error_type") or "unknown"
            type_counts[etype] = type_counts.get(etype, 0) + 1
        if type_counts:
            headers = ["error_type", "count"]
            rows = [[esc(t), str(c)] for t, c in sorted(type_counts.items(), key=lambda x: -x[1])]
            body += "<h3>Errors by error_type</h3>" + generic_table(headers, rows)
        else:
            body += "<p>No error_type data found in error_analysis records.</p>"
    else:
        body += "<p>error_analysis_llm_reviewed.json not available or empty.</p>"

    # Table 2: gold/predicted pair counts from predictions
    pair_counts: dict[tuple[str, str], int] = {}
    for r in predictions:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold != pred:
            key = (gold, pred)
            pair_counts[key] = pair_counts.get(key, 0) + 1
    if pair_counts:
        headers2 = ["Gold label", "Predicted label", "Count"]
        rows2 = [[g, p, str(c)] for (g, p), c in sorted(pair_counts.items(), key=lambda x: -x[1])]
        body += "<h3>Errors by Gold/Predicted Pair</h3>" + generic_table(headers2, rows2)

    return section("Errors by Type", body) if body else section("Errors by Type", "<p>No errors.</p>")


def render_error_examples(predictions: list[dict], n: int = 15) -> str:
    errors = [r for r in predictions if r.get("gold_label") != r.get("predicted_label")][:n]
    if not errors:
        return section("Error Examples (first 15)", "<p>No errors.</p>")
    cards = []
    for i, r in enumerate(errors, 1):
        gold = esc(r.get("gold_label", ""))
        pred = esc(r.get("predicted_label", ""))
        pid = esc(r.get("patient_id", ""))
        tid = esc(r.get("trial_id", ""))
        explanation = esc(r.get("matcher_explanation", ""))
        rationale = esc(r.get("gold_rationale", ""))
        blocking = esc("; ".join(r.get("blocking_criteria") or []))
        uncertain = esc("; ".join(r.get("uncertain_criteria") or []))
        cards.append(
            f"<details><summary><strong>#{i}</strong> patient={pid} trial={tid} "
            f"— gold=<span class='good'>{gold}</span> predicted=<span class='bad'>{pred}</span></summary>"
            f"<pre>"
            f"Blocking criteria : {blocking}\n"
            f"Uncertain criteria: {uncertain}\n"
            f"Matcher explanation:\n{explanation}\n\n"
            f"Gold rationale:\n{rationale}"
            f"</pre></details>"
        )
    return section(f"Error Examples (first {n})", "\n".join(cards))


_GENERATED_FILES = [
    "data/processed/results_llm_reviewed.json",
    "data/processed/results_llm_reviewed.csv",
    "data/processed/criterion_level_results.csv",
    "data/processed/criterion_type_summary.json",
    "data/processed/criterion_type_summary.csv",
    "data/processed/error_analysis_llm_reviewed.json",
    "data/processed/error_analysis_llm_reviewed.csv",
    "reports/benchmark_report.html",
]


def render_generated_files() -> str:
    items = "".join(f"<li><code>{esc(f)}</code></li>" for f in _GENERATED_FILES)
    return section("Generated Files", f"<ul>{items}</ul>")


def render_error_analysis(error_analysis: dict | list | None) -> str:
    if not error_analysis:
        return ""
    if isinstance(error_analysis, list):
        body = f"<p>{len(error_analysis)} error analysis records loaded.</p>"
    else:
        rows = [(esc(k), esc(str(v))) for k, v in error_analysis.items()]
        body = kv_table(rows)
    return section("Error Analysis (from error_analysis_llm_reviewed.json)", body)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results = load_json(RESULTS_FILE)
    if results is None:
        print(f"ERROR: {RESULTS_FILE} not found.")
        return

    error_analysis = load_json(ERROR_ANALYSIS_FILE)
    criterion_type_summary = load_json(CRITERION_TYPE_FILE)

    metadata = results.get("metadata", {})
    metrics = results.get("metrics", {})
    safety = results.get("safety_uncertainty_summary", {})
    error_severity = results.get("error_severity_summary", {})
    predictions = results.get("predictions", [])

    # Prefer standalone file; fall back to embedded in results
    if criterion_type_summary is None:
        criterion_type_summary = results.get("criterion_type_summary", [])

    body = render_title_and_disclaimer(metadata)
    body += render_metadata(metadata)
    body += render_global_metrics(metrics)
    body += render_confusion_matrix(metrics)
    body += render_safety_uncertainty(safety)
    body += render_error_severity(error_severity)
    body += render_criterion_type_summary(criterion_type_summary or [])
    body += render_generated_files()
    body += render_error_analysis(error_analysis)
    body += render_errors_by_type(error_analysis if isinstance(error_analysis, list) else None, predictions)
    body += render_error_examples(predictions)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(html_page(body), encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
