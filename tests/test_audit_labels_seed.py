"""Unit tests for audit_labels_seed.py."""

import json

from audit_labels_seed import main


def test_audit_labels_seed_prints_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    labels = [
        {
            "patient_id": "P001",
            "trial_id": "T001",
            "label": "unclear",
            "rationale": "Needs review.",
            "evidence": {"patient_facts": [], "trial_criteria": []},
            "label_status": "seed_needs_review",
        },
        {
            "patient_id": "P002",
            "trial_id": "T002",
            "label": "not_eligible",
            "rationale": "Age outside range.",
            "evidence": {"patient_facts": [], "trial_criteria": []},
            "label_status": "seed_needs_review",
        },
    ]
    trials = [
        {"trial_id": "T001", "category": "rehabilitation"},
        {"trial_id": "T002", "category": "drug_treatment"},
    ]

    (processed / "labels_seed.json").write_text(json.dumps(labels), encoding="utf-8")
    (processed / "trial_cases.json").write_text(json.dumps(trials), encoding="utf-8")

    main()

    output = capsys.readouterr().out
    assert "Total seed labels: 2" in output
    assert "Labels:" in output
    assert "unclear" in output
    assert "not_eligible" in output
    assert "Labels by trial category:" in output
    assert "rehabilitation" in output
    assert "drug_treatment" in output
