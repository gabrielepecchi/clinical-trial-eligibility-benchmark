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

def title_to_id(title: str) -> str:
    """Convert a section title to a stable lowercase anchor id."""
    import re
    s = title.lower()
    s = s.replace(" ", "-").replace("/", "-")
    s = re.sub(r"[^a-z0-9\-_]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def section(title: str, body: str) -> str:
    anchor = title_to_id(title)
    return f"<section id='{anchor}'>\n<h2>{esc(title)}</h2>\n{body}\n</section>\n"


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
.cards { display: flex; flex-wrap: wrap; gap: .8rem; margin: .8rem 0; }
.card { background: #f8f8f8; border: 1px solid #ddd; border-radius: 6px; padding: .7rem 1.1rem; min-width: 160px; flex: 1 1 160px; }
.card .card-label { font-size: .8rem; color: #555; margin-bottom: .25rem; }
.card .card-value { font-size: 1.4rem; font-weight: bold; color: #222; }
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


def render_safety_critical_summary(safety: dict, error_severity: dict) -> str:
    """Render a compact safety-critical summary table."""
    if not safety and not error_severity:
        return section("Safety-Critical Summary", "<p>No safety-critical data available.</p>")
    rows = [
        ("Unsafe eligible errors (not_eligible → eligible)",
         str(safety.get("unsafe_eligible_errors", 0))),
        ("Critical errors", str(error_severity.get("critical_errors", 0))),
        ("Critical error rate", pct(error_severity.get("critical_error_rate", 0))),
        ("Major errors", str(error_severity.get("major_errors", 0))),
        ("Major error rate", pct(error_severity.get("major_error_rate", 0))),
        ("Overcommitment rate", pct(safety.get("overcommitment_rate", 0))),
    ]
    return section("Safety-Critical Summary", kv_table(rows))


def render_uncertainty_signals(predictions: list[dict]) -> str:
    """Render counts of uncertainty signals across prediction records."""
    if not predictions:
        return section("Missing Information / Uncertainty Signals", "<p>No prediction data available.</p>")
    total_uncertain_criteria = 0
    predictions_with_uncertain = 0
    unclear_gold = 0
    unclear_predicted = 0
    overcommitted_unclear = 0
    for r in predictions:
        uc = r.get("uncertain_criteria") or []
        count = len(uc)
        total_uncertain_criteria += count
        if count > 0:
            predictions_with_uncertain += 1
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold == "unclear":
            unclear_gold += 1
            if pred in {"eligible", "not_eligible"}:
                overcommitted_unclear += 1
        if pred == "unclear":
            unclear_predicted += 1
    rows = [
        ("Total uncertain criteria items", str(total_uncertain_criteria)),
        ("Predictions with uncertain criteria", str(predictions_with_uncertain)),
        ("Unclear gold label predictions", str(unclear_gold)),
        ("Unclear predicted label predictions", str(unclear_predicted)),
        ("Overcommitted on unclear gold (predicted eligible/not_eligible)", str(overcommitted_unclear)),
    ]
    return section("Missing Information / Uncertainty Signals", kv_table(rows))


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


def render_label_distribution(predictions: list[dict]) -> str:
    """Render gold vs predicted label counts for all three labels."""
    if not predictions:
        return section("Label Distribution", "<p>No prediction data available.</p>")
    gold_counts: dict[str, int] = {l: 0 for l in LABELS}
    pred_counts: dict[str, int] = {l: 0 for l in LABELS}
    for r in predictions:
        g = r.get("gold_label", "")
        p = r.get("predicted_label", "")
        if g in gold_counts:
            gold_counts[g] += 1
        if p in pred_counts:
            pred_counts[p] += 1
    rows = [[label, str(gold_counts[label]), str(pred_counts[label])] for label in LABELS]
    return section("Label Distribution", generic_table(["label", "gold_count", "predicted_count"], rows))


def render_quick_metrics_cards(metrics: dict, safety: dict, error_severity: dict) -> str:
    """Render compact metric cards for key benchmark figures."""
    if not metrics and not safety and not error_severity:
        return section("Quick Metrics", "<p>No quick metrics available.</p>")

    def card(label: str, value: str) -> str:
        return (
            f"<div class='card'>"
            f"<div class='card-label'>{esc(label)}</div>"
            f"<div class='card-value'>{esc(value)}</div>"
            f"</div>"
        )

    cards = []
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        cards.append(card("Accuracy", pct(accuracy)))
    macro_f1 = metrics.get("macro_f1")
    if macro_f1 is not None:
        cards.append(card("Macro F1", fmt(macro_f1)))
    unsafe = safety.get("unsafe_eligible_errors")
    if unsafe is not None:
        cards.append(card("Unsafe Eligible Errors", str(unsafe)))
    critical = error_severity.get("critical_errors")
    if critical is not None:
        cards.append(card("Critical Errors", str(critical)))
    major = error_severity.get("major_errors")
    if major is not None:
        cards.append(card("Major Errors", str(major)))
    overcommitment = safety.get("overcommitment_rate")
    if overcommitment is not None:
        cards.append(card("Overcommitment Rate", pct(overcommitment)))

    if not cards:
        return section("Quick Metrics", "<p>No quick metrics available.</p>")
    return section("Quick Metrics", f"<div class='cards'>{''.join(cards)}</div>")


def render_model_behavior_summary(predictions: list[dict]) -> str:
    """Render counts and rates of predicted labels across all predictions."""
    if not predictions:
        return section("Model Behavior Summary", "<p>No prediction data available.</p>")
    total = len(predictions)
    eligible_count = sum(1 for r in predictions if r.get("predicted_label") == "eligible")
    not_eligible_count = sum(1 for r in predictions if r.get("predicted_label") == "not_eligible")
    unclear_count = sum(1 for r in predictions if r.get("predicted_label") == "unclear")
    eligible_rate = eligible_count / total if total else 0.0
    unclear_rate = unclear_count / total if total else 0.0
    rows = [
        ("Total predictions", str(total)),
        ("Predicted eligible", str(eligible_count)),
        ("Predicted not_eligible", str(not_eligible_count)),
        ("Predicted unclear", str(unclear_count)),
        ("Eligible prediction rate", pct(eligible_rate)),
        ("Unclear prediction rate", pct(unclear_rate)),
    ]
    return section("Model Behavior Summary", kv_table(rows))


def render_report_navigation() -> str:
    """Render a list of major report sections with anchor links."""
    sections = [
        "Benchmark Metadata",
        "Report Navigation",
        "Quick Metrics",
        "Global Metrics",
        "Label Distribution",
        "Model Behavior Summary",
        "Top Error Types",
        "Key Takeaways",
        "Safety-Critical Summary",
        "Missing Information / Uncertainty Signals",
        "Confusion Matrix",
        "Safety & Uncertainty Summary",
        "Error Severity Summary",
        "Criterion Type Summary",
        "Criterion-Level Examples",
        "Generated Files",
        "Error Analysis (from error_analysis_llm_reviewed.json)",
        "Errors by Type",
        "Worst Error Examples",
        "Error Examples",
    ]
    items = "".join(
        f"<li><a href='#{title_to_id(s)}'>{esc(s)}</a></li>" for s in sections
    )
    return section("Report Navigation", f"<ul>{items}</ul>")


def render_report_footer() -> str:
    """Render a report footer with provenance and disclaimer notes."""
    return (
        "<footer style='margin-top:3rem;padding-top:1rem;border-top:1px solid #ddd;"
        "font-size:.85rem;color:#666;'>"
        "<p>This report was <strong>generated locally</strong> from benchmark results. "
        "All patient data consists of <strong>synthetic patients</strong> only. "
        "Labels are <strong>draft LLM-reviewed labels</strong> that have not been fully validated. "
        "<strong>Not for clinical use.</strong></p>"
        "</footer>\n"
    )


def render_top_error_types(error_analysis: list[dict] | None, n: int = 5) -> str:
    """Render a table of the top n error types from error_analysis records."""
    if not error_analysis:
        return section("Top Error Types", "<p>No error analysis data available.</p>")
    type_counts: dict[str, int] = {}
    for r in error_analysis:
        etype = r.get("error_type") or "unknown"
        type_counts[etype] = type_counts.get(etype, 0) + 1
    if not type_counts:
        return section("Top Error Types", "<p>No error analysis data available.</p>")
    top = sorted(type_counts.items(), key=lambda x: -x[1])[:n]
    rows = [[esc(t), str(c)] for t, c in top]
    return section("Top Error Types", generic_table(["error_type", "count"], rows))


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


def _error_severity(gold: str, pred: str) -> tuple[int, str]:
    """Return (sort_key, severity_label) for a gold/predicted pair."""
    if gold == "not_eligible" and pred == "eligible":
        return 0, "critical"
    if (gold == "unclear" and pred in {"eligible", "not_eligible"}) or \
       (gold in {"eligible", "not_eligible"} and pred == "unclear"):
        return 1, "major"
    if (gold == "eligible" and pred == "not_eligible") or \
       (gold == "not_eligible" and pred == "unclear"):
        return 2, "minor"
    return 3, "other"


def render_worst_error_examples(predictions: list[dict], n: int = 10) -> str:
    errors = [r for r in predictions if r.get("gold_label") != r.get("predicted_label")]
    if not errors:
        return section("Worst Error Examples", "<p>No errors.</p>")
    errors.sort(key=lambda r: _error_severity(r.get("gold_label", ""), r.get("predicted_label", ""))[0])
    errors = errors[:n]
    cards = []
    for i, r in enumerate(errors, 1):
        gold = esc(r.get("gold_label", ""))
        pred = esc(r.get("predicted_label", ""))
        _, severity = _error_severity(r.get("gold_label", ""), r.get("predicted_label", ""))
        pid = esc(r.get("patient_id", ""))
        tid = esc(r.get("trial_id", ""))
        explanation = esc(r.get("matcher_explanation", ""))
        rationale = esc(r.get("gold_rationale", ""))
        cards.append(
            f"<details><summary><strong>#{i}</strong> patient={pid} trial={tid} "
            f"— gold=<span class='good'>{gold}</span> predicted=<span class='bad'>{pred}</span> "
            f"[<strong>{esc(severity)}</strong>]</summary>"
            f"<pre>"
            f"Severity          : {esc(severity)}\n"
            f"Matcher explanation:\n{explanation}\n\n"
            f"Gold rationale:\n{rationale}"
            f"</pre></details>"
        )
    return section(f"Worst Error Examples (top {n})", "\n".join(cards))


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


def render_key_takeaways(
    metrics: dict,
    safety: dict,
    error_analysis: list[dict] | None,
) -> str:
    """Render 3-5 neutral research-focused bullet points from available data."""
    bullets = []

    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    if accuracy is not None and macro_f1 is not None:
        bullets.append(
            f"Overall accuracy is <strong>{pct(accuracy)}</strong> with a macro F1 of "
            f"<strong>{fmt(macro_f1)}</strong>."
        )

    unsafe = safety.get("unsafe_eligible_errors")
    total = safety.get("total_predictions")
    if unsafe is not None and total:
        bullets.append(
            f"Unsafe eligible errors (predicted eligible when gold is not_eligible): "
            f"<strong>{unsafe}</strong> of {total} pairs "
            f"({pct(unsafe / total)})."
        )

    overcommitment = safety.get("overcommitment_rate")
    if overcommitment is not None:
        bullets.append(
            f"Overcommitment rate on unclear gold labels: "
            f"<strong>{fmt(overcommitment)}</strong>."
        )

    if error_analysis:
        type_counts: dict[str, int] = {}
        for r in error_analysis:
            etype = r.get("error_type") or "unknown"
            type_counts[etype] = type_counts.get(etype, 0) + 1
        if type_counts:
            top_type, top_count = max(type_counts.items(), key=lambda x: x[1])
            bullets.append(
                f"The most frequent error type in the analysis set is "
                f"<strong>{esc(top_type)}</strong> ({top_count} records)."
            )

    if not bullets:
        return ""
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return section("Key Takeaways", f"<ul>{items}</ul>")


def render_criterion_level_examples(predictions: list[dict], n: int = 10) -> str:
    """Render up to n prediction records that contain criterion_results."""
    records = [r for r in predictions if r.get("criterion_results")][:n]
    if not records:
        return section("Criterion-Level Examples", "<p>No criterion-level results available.</p>")
    cards = []
    for i, r in enumerate(records, 1):
        pid = esc(r.get("patient_id", ""))
        tid = esc(r.get("trial_id", ""))
        gold = esc(r.get("gold_label", ""))
        pred = esc(r.get("predicted_label", ""))
        criteria = (r.get("criterion_results") or [])[:5]
        rows = []
        for c in criteria:
            ctype = esc(c.get("criterion_type", ""))
            decision = esc(c.get("decision", ""))
            text = esc(c.get("criterion_text") or c.get("criterion", ""))
            evidence = esc(c.get("evidence", ""))
            confidence = esc(str(c.get("confidence", ""))) if c.get("confidence") is not None else ""
            row = (
                f"  criterion_type : {ctype}\n"
                f"  decision       : {decision}\n"
                f"  criterion      : {text}\n"
            )
            if evidence:
                row += f"  evidence       : {evidence}\n"
            if confidence:
                row += f"  confidence     : {confidence}\n"
            rows.append(row)
        cblock = "\n".join(rows)
        cards.append(
            f"<details><summary><strong>#{i}</strong> patient={pid} trial={tid} "
            f"— gold=<span class='good'>{gold}</span> predicted=<span class='bad'>{pred}</span>"
            f"</summary><pre>{cblock}</pre></details>"
        )
    return section("Criterion-Level Examples", "\n".join(cards))


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
    body += render_report_navigation()
    body += render_quick_metrics_cards(metrics, safety, error_severity)
    body += render_global_metrics(metrics)
    body += render_label_distribution(predictions)
    body += render_model_behavior_summary(predictions)
    body += render_top_error_types(error_analysis if isinstance(error_analysis, list) else None)
    body += render_key_takeaways(metrics, safety, error_analysis if isinstance(error_analysis, list) else None)
    body += render_safety_critical_summary(safety, error_severity)
    body += render_uncertainty_signals(predictions)
    body += render_confusion_matrix(metrics)
    body += render_safety_uncertainty(safety)
    body += render_error_severity(error_severity)
    body += render_criterion_type_summary(criterion_type_summary or [])
    body += render_criterion_level_examples(predictions)
    body += render_generated_files()
    body += render_error_analysis(error_analysis)
    body += render_errors_by_type(error_analysis if isinstance(error_analysis, list) else None, predictions)
    body += render_worst_error_examples(predictions)
    body += render_error_examples(predictions)
    body += render_report_footer()

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(html_page(body), encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
