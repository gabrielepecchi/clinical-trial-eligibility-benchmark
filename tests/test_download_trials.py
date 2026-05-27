"""Unit tests for download_trials.py — no real HTTP calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from download_trials import build_params, fetch_trials, save_trials


def make_response(studies: list[dict], next_page_token: str | None = None) -> MagicMock:
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    payload = {"studies": studies}
    if next_page_token:
        payload["nextPageToken"] = next_page_token
    mock.json.return_value = payload
    return mock


STUDY_A = {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}
STUDY_B = {"protocolSection": {"identificationModule": {"nctId": "NCT00000002"}}}


def test_build_params_without_token():
    params = build_params(None, 10)
    assert params["query.cond"] == "Parkinson disease"
    assert params["pageSize"] == 10
    assert params["format"] == "json"
    assert "pageToken" not in params


def test_build_params_with_token():
    params = build_params("abc123", 10)
    assert params["pageToken"] == "abc123"


def test_fetch_trials_returns_studies():
    with patch("download_trials.requests.get", return_value=make_response([STUDY_A])):
        result = fetch_trials(max_trials=10)
    assert len(result) == 1
    assert result[0] == STUDY_A


def test_fetch_trials_pagination():
    responses = [
        make_response([STUDY_A], next_page_token="token123"),
        make_response([STUDY_B]),
    ]
    with patch("download_trials.requests.get", side_effect=responses):
        result = fetch_trials(max_trials=10)
    assert len(result) == 2
    assert result[0] == STUDY_A
    assert result[1] == STUDY_B


def test_fetch_trials_stops_on_empty():
    responses = [
        make_response([STUDY_A], next_page_token="token123"),
        make_response([]),
    ]
    with patch("download_trials.requests.get", side_effect=responses):
        result = fetch_trials(max_trials=10)
    assert len(result) == 1


def test_fetch_trials_respects_max(tmp_path):
    # Each page returns one study; max_trials=1 should stop after first page
    with patch("download_trials.requests.get", return_value=make_response([STUDY_A], next_page_token="tok")):
        result = fetch_trials(max_trials=1)
    assert len(result) == 1


def test_save_trials_writes_json(tmp_path):
    output_file = tmp_path / "raw" / "trials.json"
    save_trials([STUDY_A, STUDY_B], output_file)
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0] == STUDY_A
