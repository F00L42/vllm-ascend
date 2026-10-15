# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check paired benchmark isolation and failure accounting without an accelerator."""

import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "benchmarks/scripts/benchmark_systemone_shared_prefix.py"
_SPEC = importlib.util.spec_from_file_location("systemone_shared_prefix_benchmark", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def test_cold_pairs_use_distinct_salts_and_only_warm_reuses_them():
    payload = {"state": "evidence", "questions": {str(i): {"instructions": str(i)} for i in range(20)}}
    calls = []

    def capture(url, request, timeout):
        calls.append(request)
        return {"ok": True, "elapsed_ms": 100 if len(request["questions"]) == 1 else 140}

    pairs = _BENCHMARK.run_pairs(payload, "7", 8, 4, "http://unused", 10, True, capture)
    assert len(pairs) == 16
    assert all(pair["passed"] and pair["ratio"] == 1.4 for pair in pairs)
    cold_salts = [call["cache_salt"] for call in calls[::2]]
    assert len(set(cold_salts)) == 16
    assert all(calls[i] is calls[i + 1] for i in range(0, len(calls), 2))
    assert {tuple(pair["order"]) for pair in pairs} == {(1, 20), (20, 1)}
    assert all(call["state"] == payload["state"] for call in calls)
    assert all(list(call["questions"]) == ["7"] for call in calls if len(call["questions"]) == 1)
    assert "cache_salt" not in payload


def test_failure_and_excess_latency_both_count_as_failed_pairs():
    pairs = [
        {
            "cache": "logical_cold",
            "repeat": 0,
            "ratio": None,
            "passed": False,
            "1": {"ok": False, "elapsed_ms": 100},
            "20": {"ok": True, "elapsed_ms": 140},
        },
        {
            "cache": "logical_cold",
            "repeat": 1,
            "ratio": 2,
            "passed": False,
            "1": {"ok": True, "elapsed_ms": 100},
            "20": {"ok": True, "elapsed_ms": 200},
        },
    ]
    summary = _BENCHMARK.summarize(pairs)["logical_cold"]
    assert summary["pairs"] == summary["failed_or_over_limit_pairs"] == 2
    assert summary["http_failed_pairs"] == 1
    assert summary["failed_repeats"] == [0, 1]
    assert summary["all_pairs_passed"] is False
    assert summary["ratio_percentiles_successful_pairs"]["p50"] == 2


def test_failed_cold_request_cannot_establish_warm_baseline():
    payload = {"questions": {str(i): {} for i in range(20)}}
    calls = []

    def capture(*args):
        calls.append(None)
        return {"ok": len(calls) != 1, "elapsed_ms": 10}

    pairs = _BENCHMARK.run_pairs(payload, "0", 1, 0, "unused", 10, True, capture)
    assert [pair["ratio"] for pair in pairs] == [None, None]
    assert all(not pair["passed"] for pair in pairs)


@pytest.mark.parametrize("body", [[], {"answers": None}, {"answers": {}}, "invalid JSON"])
def test_capture_rejects_invalid_or_incomplete_http_response(monkeypatch, body):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return body.encode() if isinstance(body, str) else json.dumps(body).encode()

    monkeypatch.setattr(_BENCHMARK, "urlopen", lambda *args, **kwargs: Response())
    record = _BENCHMARK.capture("http://unused", {"questions": {"q": {}}}, 10)
    assert record["http_status"] == 200
    assert not record["ok"]
    assert "error" in record and record["elapsed_ms"] >= 0


def test_capture_records_transport_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("injected timeout")

    monkeypatch.setattr(_BENCHMARK, "urlopen", fail)
    record = _BENCHMARK.capture("http://unused", {"questions": {"q": {}}}, 10)
    assert not record["ok"] and "injected timeout" in record["error"]
