"""Unit tests for summarize_error_analysis.py."""

import pytest

from summarize_error_analysis import main


@pytest.fixture
def output(capsys, tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "error_analysis_sample.json").write_text(
        '[{"case_id":"ERR001","patient_id":"P003","trial_id":"T002",'
        '"gold_label":"not_eligible","predicted_label":"eligible",'
        '"error_type":"missed_exclusion","explanation":"Test.","possible_fix":"Fix."}]',
        encoding="utf-8",
    )
    main()
    return capsys.readouterr().out


def test_prints_summary_header(output):
    assert "Error Analysis Summary" in output


def test_prints_errors_by_type(output):
    assert "Errors by type" in output


def test_prints_errors_by_gold_label(output):
    assert "Errors by gold label" in output


def test_prints_errors_by_predicted_label(output):
    assert "Errors by predicted label" in output


def test_known_error_type_appears(output):
    assert "missed_exclusion" in output


def test_known_label_appears(output):
    assert "not_eligible" in output
