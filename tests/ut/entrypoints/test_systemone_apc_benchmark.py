# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression coverage for APC experiment evidence and acceptance gates without an accelerator."""

import copy
import hashlib
import importlib.util
import io
import json
import threading
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "benchmarks/scripts/benchmark_systemone_apc.py"
_SPEC = importlib.util.spec_from_file_location("systemone_apc_benchmark", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def payload(count=20):
    ids = ["department", "refund", "urgency"] + [f"question_{index}" for index in range(17)]
    return {
        "state": "unchanged evidence",
        "questions": {
            question_id: {"type": "choice", "instructions": question_id, "criteria": ["A", "B"]}
            for question_id in ids[:count]
        },
    }


def arguments(**changes):
    return SimpleNamespace(
        **{
            **dict(
                repeats=1,
                single_question="department",
                bridge_questions=["department", "refund", "urgency"],
                question_counts=[3, 20],
                concurrency=[1, 3],
                atol=1e-5,
                rtol=1e-4,
                ratio_limit=1.5,
                seed=0,
            ),
            **changes,
        }
    )


class FakeCapture:
    def __init__(self, ratio=1.4):
        self.calls = []
        self.seen = set()
        self.ratio = ratio
        self.lock = threading.Lock()

    def __call__(self, request):
        with self.lock:
            request = copy.deepcopy(request)
            self.calls.append(request)
            details = []
            for question_id in request["questions"]:
                key = request["cache_salt"], question_id
                details.append(
                    {
                        "question_id": question_id,
                        "prompt_tokens": 64,
                        "cached_tokens": 32 if key in self.seen else 0,
                        "probabilities": [0.7, 0.3],
                        "status": "completed",
                        "engine_request_id": f"engine-{len(self.calls)}-{question_id}",
                        "output_request_id": None,
                    }
                )
                self.seen.add(key)
            return {
                "request_id": f"request-{len(self.calls)}",
                "question_ids": list(request["questions"]),
                "cache_salt": request["cache_salt"],
                "ok": True,
                "elapsed_ms": 100 if len(request["questions"]) == 1 else self.ratio * 100,
                "response": {
                    "answers": {
                        question_id: {"type": "choice", "choice": "A", "confidence": 0.7}
                        for question_id in request["questions"]
                    },
                    "diagnostics": {
                        "schema_version": 1,
                        "parent_request_id": f"request-{len(self.calls)}",
                        "configured_strategy": "first_question",
                        "strategy": "direct" if len(request["questions"]) == 1 else "first_question",
                        "cache_salt_sha256": hashlib.sha256(request["cache_salt"].encode()).hexdigest(),
                        "question_details": details,
                    },
                },
            }


def test_salt_requires_real_zero_positive_zero_and_identical_lengths():
    fake = FakeCapture()
    report = _BENCHMARK.run_salt(payload(), arguments(repeats=2), fake)
    assert report["passed"]
    assert [trial["cached_tokens_A_A_B"] for trial in report["trials"]] == [[0, 32, 0], [0, 32, 0]]
    salts = [call["cache_salt"] for call in fake.calls]
    assert salts[0] == salts[1] and salts[3] == salts[4]
    assert len(set(salts)) == 4 and len({len(salt) for salt in salts}) == 1
    assert all(call["questions"] == fake.calls[0]["questions"] for call in fake.calls)


@pytest.mark.parametrize("warm_cached", [None, 0])
def test_salt_does_not_accept_unknown_or_absent_warm_hits(warm_cached):
    fake = FakeCapture()

    def capture(request):
        record = fake(request)
        if len(fake.calls) == 2:
            record["response"]["diagnostics"]["question_details"][0]["cached_tokens"] = warm_cached
        return record

    report = _BENCHMARK.run_salt(payload(), arguments(), capture)
    assert not report["passed"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parent_request_id", "wrong", "X-Request-Id"),
        ("strategy", "direct", "fallback"),
        ("configured_strategy", "producer", "configured_strategy"),
        ("cache_salt_sha256", "wrong", "cache salt hash"),
    ],
)
def test_response_rejects_wrong_server_identity_or_strategy(field, value, message):
    request = _BENCHMARK._payload(payload(3), payload(3)["questions"])
    record = FakeCapture()(request)
    record["response"]["diagnostics"][field] = value
    with pytest.raises(ValueError, match=message):
        _BENCHMARK.validate_response(record, request, "first_question")


@pytest.mark.parametrize("cached_tokens", [None, -1, 65, True])
def test_response_rejects_unknown_or_invalid_cache_counters(cached_tokens):
    request = _BENCHMARK._payload(payload(1), ["department"])
    record = FakeCapture()(request)
    record["response"]["diagnostics"]["question_details"][0]["cached_tokens"] = cached_tokens
    with pytest.raises(ValueError, match="cached_tokens"):
        _BENCHMARK.validate_response(record, request, "first_question")


def test_response_accepts_direct_single_question_and_missing_output_request_id():
    request = _BENCHMARK._payload(payload(1), ["department"])
    record = FakeCapture()(request)
    _BENCHMARK.validate_response(record, request, "first_question")


def test_capture_keeps_full_http_failure_body_and_client_timestamps(monkeypatch):
    def fail(request, timeout):
        raise HTTPError(request.full_url, 503, "injected", {}, io.BytesIO(b'{"error":"engine unavailable"}'))

    monkeypatch.setattr(_BENCHMARK, "urlopen", fail)
    record = _BENCHMARK.capture("http://unused", _BENCHMARK._payload(payload(1), ["department"]), 1, "first_question")
    assert not record["ok"] and record["http_status"] == 503
    assert record["response"] == {"error": "engine unavailable"}
    assert "engine unavailable" in record["body_text"]
    assert record["complete_monotonic_ns"] >= record["submit_monotonic_ns"]
    assert record["complete_wall_ns"] >= record["submit_wall_ns"]
    assert record["elapsed_ms"] >= 0


@pytest.mark.parametrize("body", [b'{"answers":NaN}', b'{"answers":'])
def test_capture_preserves_invalid_json_without_unserializable_numeric_values(monkeypatch, body):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return body

    monkeypatch.setattr(_BENCHMARK, "urlopen", lambda *args, **kwargs: Response())
    record = _BENCHMARK.capture("http://unused", _BENCHMARK._payload(payload(1), ["department"]), 1, "first_question")
    assert not record["ok"] and record["body_text"] == body.decode()
    json.dumps(record, allow_nan=False)


def test_capture_preserves_truncated_http_response(monkeypatch):
    def fail(*args, **kwargs):
        raise IncompleteRead(b'{"answers":', 50)

    monkeypatch.setattr(_BENCHMARK, "urlopen", fail)
    record = _BENCHMARK.capture("http://unused", _BENCHMARK._payload(payload(1), ["department"]), 1, "first_question")
    assert not record["ok"]
    assert record["body_text"] == '{"answers":'
    assert "IncompleteRead" in record["error"]


def test_bridge_reuses_one_salt_across_concurrent_cold_and_warm_http_requests():
    fake = FakeCapture()
    report = _BENCHMARK.run_bridge(payload(), arguments(repeats=2), fake)
    assert report["passed"] and len(report["trials"]) == 8
    salts = set()
    for trial in report["trials"]:
        assert trial["cache_salt"] not in salts
        salts.add(trial["cache_salt"])
        records = trial["cold"] + trial["warm"]
        assert len(records) == 2 * trial["concurrency"]
        assert {record["cache_salt"] for record in records} == {trial["cache_salt"]}
        assert len({record["request_id"] for record in records}) == len(records)
        if trial["question_count"] == 3:
            assert trial["question_ids"] == ["department", "refund", "urgency"]


def test_bridge_failure_cannot_be_hidden_by_successful_warm_repeat():
    fake = FakeCapture()

    def capture(request):
        record = fake(request)
        if record["request_id"] == "request-1":
            record.update(ok=False, error="injected cold failure")
        return record

    report = _BENCHMARK.run_bridge(payload(), arguments(question_counts=[3], concurrency=[1]), capture)
    assert not report["passed"]
    assert report["trials"][0]["warm"][0]["ok"]
    assert not report["trials"][0]["comparisons"][0]["passed"]


def test_raw_probability_change_fails_even_with_equal_rounded_answer():
    fake = FakeCapture()
    request = _BENCHMARK._payload(payload(1), ["department"])
    reference, observed = fake(request), fake(request)
    observed["response"]["diagnostics"]["question_details"][0]["probabilities"] = [0.700001, 0.299999]
    result = _BENCHMARK.compare_records(reference, observed, atol=1e-8, rtol=0)
    assert not result["passed"]
    assert result["questions"]["department"]["max_abs"] > 1e-8
    assert not result["questions"]["department"]["choice_changed"]


def test_correctness_covers_independent_noise_reverse_and_duplicate_question():
    fake = FakeCapture()
    report = _BENCHMARK.run_correctness(payload(3), arguments(), fake)
    assert report["passed"] and len(fake.calls) == 9
    trial = report["trials"][0]
    assert len(trial["independent"]) == 3
    assert trial["reversed"]["question_ids"] == ["urgency", "refund", "department"]
    duplicate = trial["comparisons"]["duplicated"]["questions"]["__apc_duplicate_department"]
    assert duplicate["reference_question_id"] == "department" and duplicate["passed"]
    assert len({call["cache_salt"] for call in fake.calls}) == len(fake.calls)


def test_correctness_never_expands_tolerance_to_absorb_observed_noise():
    fake = FakeCapture()

    def capture(request):
        record = fake(request)
        if len(fake.calls) == 2:
            record["response"]["diagnostics"]["question_details"][0]["probabilities"] = [0.71, 0.29]
        return record

    args = arguments(atol=1e-8, rtol=0)
    report = _BENCHMARK.run_correctness(payload(3), args, capture)
    assert not report["passed"]
    assert args.atol == 1e-8 and args.rtol == 0
    assert not report["trials"][0]["independent"][0]["comparison"]["passed"]


def test_performance_preserves_failed_pairs_and_distinguishes_target_failure():
    fake = FakeCapture(ratio=2)
    report = _BENCHMARK.run_performance(payload(), arguments(repeats=3), fake)
    assert not report["passed"] and report["data_passed"] and not report["target_passed"]
    assert len(report["pairs"]) == 6 and len(fake.calls) == 32
    assert all(pair["ratio"] == 2 and not pair["passed"] for pair in report["pairs"])
    assert report["summary"]["logical_cold"]["target_failed_valid_pairs"] == 3
    assert report["summary"]["logical_cold"]["data_failed_pairs"] == 0
    measured = fake.calls[20:]
    assert all(measured[index]["cache_salt"] == measured[index + 1]["cache_salt"] for index in range(0, 12, 2))
    assert len({call["cache_salt"] for call in measured}) == 6


def test_performance_checks_every_question_and_invalidates_failed_cold_predecessor():
    fake = FakeCapture()

    def capture(request):
        record = fake(request)
        if len(fake.calls) == 23:
            assert len(request["questions"]) == 20
            record["response"]["diagnostics"]["question_details"][-1]["probabilities"] = [0.6, 0.4]
        return record

    report = _BENCHMARK.run_performance(payload(), arguments(), capture)
    assert not report["data_passed"]
    assert all(pair["ratio"] is None and not pair["passed"] for pair in report["pairs"])
    assert not report["pairs"][0]["comparisons"]["many"]["questions"]["question_16"]["passed"]
    assert not report["pairs"][1]["cold_predecessor_passed"]


def test_cli_refuses_to_fabricate_twenty_questions(tmp_path):
    source = tmp_path / "three.json"
    source.write_text(json.dumps(payload(3)), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _BENCHMARK.parse_args(
            ["--input", str(source), "--output", str(tmp_path / "out.json"), "--expected-strategy", "first_question"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(("ratio", "expected_exit"), [(1.4, 0), (2, 1)])
def test_main_saves_raw_journal_and_parameter_evidence(tmp_path, monkeypatch, ratio, expected_exit):
    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(json.dumps(payload()), encoding="utf-8")
    fake = FakeCapture(ratio=ratio)
    monkeypatch.setattr(_BENCHMARK, "capture", lambda url, measured, timeout, expected_strategy: fake(measured))
    result = _BENCHMARK.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--expected-strategy",
            "first_question",
            "--phase",
            "performance",
            "--repeats",
            "1",
        ]
    )
    assert result == expected_exit
    report = json.loads(output.read_text(encoding="utf-8"))
    journal = [json.loads(line) for line in Path(report["journal_path"]).read_text(encoding="utf-8").splitlines()]
    assert report["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["parameters"]["atol"] == 1e-5
    assert report["data_passed"] and report["status"] == "completed"
    assert len(journal) == len(fake.calls) == 24


def test_main_stops_dependent_phases_after_salt_failure_and_returns_data_failure(tmp_path, monkeypatch):
    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(json.dumps(payload()), encoding="utf-8")
    fake = FakeCapture()

    def capture(url, measured, timeout, expected_strategy):
        record = fake(measured)
        record.update(ok=False, error="diagnostics unavailable")
        return record

    monkeypatch.setattr(_BENCHMARK, "capture", capture)
    result = _BENCHMARK.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--expected-strategy",
            "first_question",
            "--repeats",
            "1",
        ]
    )
    assert result == 2 and len(fake.calls) == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["phases"]["performance"]["skipped"]
    assert not report["data_passed"]
