"""Unit tests for run_sample_benchmark.py."""

import json

from run_sample_benchmark import (
    load_json,
    parse_args,
    QUICK_DEMO_DEFAULT_LIMIT,
)


def test_load_json_returns_list(tmp_path):
    data = [{"patient_id": "P001", "label": "eligible"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert isinstance(result, list)


def test_load_json_returns_correct_length(tmp_path):
    data = [
        {"patient_id": "P001", "label": "eligible"},
        {"patient_id": "P002", "label": "not_eligible"},
    ]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert len(result) == 2


def test_load_json_preserves_string_value(tmp_path):
    data = [{"patient_id": "P001", "label": "eligible"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["patient_id"] == "P001"


def test_load_json_preserves_label_value(tmp_path):
    data = [{"patient_id": "P001", "label": "unclear"}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["label"] == "unclear"


def test_load_json_preserves_integer_value(tmp_path):
    data = [{"patient_id": "P001", "age": 62}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["age"] == 62


def test_load_json_preserves_list_value(tmp_path):
    data = [{"patient_id": "P001", "diagnosis": ["Parkinson disease"]}]
    file = tmp_path / "sample.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    result = load_json(file)

    assert result[0]["diagnosis"] == ["Parkinson disease"]


def test_load_json_empty_list(tmp_path):
    file = tmp_path / "empty.json"
    file.write_text(json.dumps([]), encoding="utf-8")

    result = load_json(file)

    assert result == []


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py"])
    args = parse_args()
    assert args.quick_demo is False
    assert args.limit is None


def test_parse_args_quick_demo_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py", "--quick-demo"])
    args = parse_args()
    assert args.quick_demo is True


def test_parse_args_limit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py", "--limit", "5"])
    args = parse_args()
    assert args.limit == 5


def test_parse_args_quick_demo_with_limit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py", "--quick-demo", "--limit", "2"])
    args = parse_args()
    assert args.quick_demo is True
    assert args.limit == 2


def test_quick_demo_default_limit_is_small():
    assert QUICK_DEMO_DEFAULT_LIMIT <= 5


def test_quick_demo_limit_overrides_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py", "--quick-demo", "--limit", "1"])
    args = parse_args()
    effective_limit = args.limit if args.limit is not None else QUICK_DEMO_DEFAULT_LIMIT
    assert effective_limit == 1


def test_limit_without_quick_demo(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sample_benchmark.py", "--limit", "4"])
    args = parse_args()
    assert args.quick_demo is False
    assert args.limit == 4
