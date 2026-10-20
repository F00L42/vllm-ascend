# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def analysis(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "benchmarks/scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("systemone_apc_analysis", directory / "analyze_systemone_apc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_fixture():
    external = "systemone-client-uuid-0"
    steps = [
        {
            "step_id": index + 1,
            "monotonic_ns": index,
            "requests": [
                {
                    "request_id": external + "-aabbccdd",
                    "num_scheduled_tokens": 4,
                    "computed_before": index * 4 + 4,
                    "num_cached_tokens": 4 if index == 0 else None,
                    "num_preemptions": 0,
                }
            ],
        }
        for index in range(2)
    ]
    report = {
        "trials": [
            {
                "request_id": "client",
                "ok": True,
                "response": {
                    "diagnostics": {
                        "parent_request_id": "client",
                        "strategy": "direct",
                        "question_details": [
                            {
                                "engine_request_id": external,
                                "question_id": "q",
                                "prompt_tokens": 12,
                                "cached_tokens": 4,
                            }
                        ],
                    }
                },
            }
        ]
    }
    return steps, report


def test_trace_sums_work_without_summing_cached_progress(analysis):
    steps, report = trace_fixture()
    result = analysis.join_trace(steps, [report])
    assert result["passed"]
    row = result["requests"][0]
    assert row["first_cached_tokens"] == 4
    assert row["scheduled_new_tokens"] == 8
    assert row["scheduled_minus_uncached_prompt"] == 0


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_first_or_last_chunk_fails(analysis, missing):
    steps, report = trace_fixture()
    del steps[missing]
    assert not analysis.join_trace(steps, [report])["passed"]


def test_missing_middle_step_fails(analysis):
    steps, report = trace_fixture()
    steps[1]["step_id"] = 3
    assert not analysis.join_trace(steps, [report])["passed"]


def test_preemption_is_preserved_but_needs_review(analysis):
    steps, report = trace_fixture()
    steps[1]["requests"][0]["num_preemptions"] = 1
    result = analysis.join_trace(steps, [report])
    assert not result["passed"]
    assert result["requests"][0]["preemptions"] == 1
    assert any("recomputation review" in error for error in result["errors"])


def test_request_zero_does_not_match_question_ten(analysis):
    assert analysis.internal_matches("systemone-parent-0", "systemone-parent-0-aabbccdd")
    assert not analysis.internal_matches("systemone-parent-0", "systemone-parent-01-aabbccdd")
    assert not analysis.internal_matches("systemone-parent-0", "systemone-parent-0-arbitrary")


def test_log_keeps_line_numbers_and_rejects_truncation(analysis, tmp_path):
    path = tmp_path / "server.log"
    path.write_text('prefix SystemOne scheduler {"step_id":1}\nSystemOne scheduler {\n', encoding="utf-8")
    steps, _, errors = analysis.load_log(path)
    assert steps[0]["log_line"] == 1
    assert len(errors) == 1


def test_manifest_mismatch_and_missing_sha_fail(analysis):
    first = {"trace_scheduler": True, "vllm": {"sha": "a", "diff": ""}, "vllm_ascend": {"sha": "b", "diff": ""}}
    assert not analysis.compare_manifests([first, first])
    second = {**first, "vllm": {"sha": "different", "diff": ""}}
    assert analysis.compare_manifests([first, second])
    assert analysis.compare_manifests([first, {**first, "vllm": {"sha": None}}])


def test_trace_rejects_contradictory_endpoint_cache_statistics(analysis):
    steps, report = trace_fixture()
    report["trials"][0]["response"]["diagnostics"]["question_details"][0]["cached_tokens"] = 8
    result = analysis.join_trace(steps, [report])
    assert not result["passed"]
    assert any("Scheduler and endpoint cache counts differ" in error for error in result["errors"])


def test_network_failure_without_response_is_not_dropped(analysis):
    steps, report = trace_fixture()
    report["trials"].append({"request_id": "failed", "ok": False, "error": "connection refused"})
    result = analysis.join_trace(steps, [report])
    assert not result["passed"]
    assert any("connection refused" in error for error in result["errors"])


def capture_fixture(question_ids, strategy, elapsed_ms=100):
    return {
        "ok": True,
        "request_id": "request-" + "-".join(question_ids),
        "question_ids": question_ids,
        "elapsed_ms": elapsed_ms,
        "response": {
            "answers": {qid: {"type": "choice", "choice": "a", "confidence": 0.7} for qid in question_ids},
            "diagnostics": {
                "runtime": {
                    "model": "same-checkpoint",
                    "scheduler": {"max_num_batched_tokens": 2048},
                    "shared_prefix": {
                        "enabled": strategy != "direct",
                        "strategy": strategy,
                        "verified_boundary_tokens": 1536,
                        "min_shared_tokens": 4096,
                        "min_shared_tokens_applies": strategy == "producer",
                        "capability_reason": "disabled" if strategy == "direct" else "ready",
                    },
                },
                "question_details": [
                    {"question_id": qid, "prompt_tokens": 16000, "probabilities": [0.7, 0.3]} for qid in question_ids
                ],
            },
        },
    }


def report_fixture(strategy="direct", phase="correctness", elapsed_ms=140, target_passed=True):
    questions = ["department", *[f"q{index}" for index in range(19)]]
    report = {
        "input_sha256": "a" * 64,
        "question_ids": questions,
        "parameters": {
            "expected_strategy": strategy,
            "atol": 1e-5,
            "rtol": 1e-4,
            "single_question": "department",
            "seed": 42,
            "repeats": 2,
            "ratio_limit": 1.5,
        },
        "data_passed": True,
        "target_passed": target_passed,
    }
    if phase == "correctness":
        report["phases"] = {
            "correctness": {
                "passed": True,
                "trials": [
                    {
                        "repeat": repeat,
                        "batch": capture_fixture(questions, strategy),
                        "reversed": capture_fixture(list(reversed(questions)), strategy),
                        "duplicated": capture_fixture([*questions, "__apc_duplicate_department"], strategy),
                    }
                    for repeat in range(2)
                ],
            }
        }
    else:
        report["phases"] = {
            "performance": {
                "passed": target_passed,
                "data_passed": True,
                "target_passed": target_passed,
                "pairs": [
                    {
                        "repeat": repeat,
                        "cache": cache,
                        "data_passed": True,
                        "target_passed": target_passed,
                        "passed": target_passed,
                        "one": capture_fixture(["department"], strategy),
                        "many": capture_fixture(questions, strategy, elapsed_ms=elapsed_ms),
                    }
                    for repeat in range(2)
                    for cache in ("logical_cold", "warm_repeat")
                ],
            }
        }
    return report


def manifest_fixture(strategy="direct"):
    shared = {"enabled": strategy != "direct", "strategy": strategy, "verified_boundary_tokens": 1536}
    config = {"systemone_config": {"diagnostics": True, "shared_prefix": shared}}
    return {
        "trace_scheduler": False,
        "vllm": {"sha": "vllm-sha", "diff": ""},
        "vllm_ascend": {"sha": "ascend-sha", "diff": ""},
        "additional_config": config,
        "effective_command": ["vllm", "serve", "model", "--additional-config", json.dumps(config)],
    }


def test_correctness_comparison_keeps_question_id_alignment(analysis):
    direct, first = report_fixture(), report_fixture("first_question")
    result = analysis.compare_runs([direct, first])
    assert result["passed"] and result["phase"] == "correctness"
    assert len(result["comparisons"]) == 6
    first["phases"]["correctness"]["trials"][0]["reversed"]["response"]["diagnostics"]["question_details"][0][
        "probabilities"
    ] = [0.6, 0.4]
    assert not analysis.compare_runs([direct, first])["passed"]


@pytest.mark.parametrize("phases", [{}, {"correctness": {"passed": False, "skipped": "salt failed"}}])
def test_missing_or_skipped_correctness_returns_failure_instead_of_raising(analysis, phases):
    reference = report_fixture()
    reference["phases"] = phases
    result = analysis.compare_runs([reference, report_fixture("producer")])
    assert not result["passed"] and result["errors"]


def test_runtime_only_ignores_allowed_strategy_fields(analysis):
    direct, first = report_fixture(), report_fixture("first_question")
    runtime = first["phases"]["correctness"]["trials"][0]["batch"]["response"]["diagnostics"]["runtime"]
    runtime["shared_prefix"]["verified_boundary_tokens"] = 256
    result = analysis.compare_runs([direct, first])
    assert not result["passed"]
    assert any("Runtime configuration differs" in error for error in result["errors"])


def test_performance_targets_do_not_determine_comparability(analysis):
    direct = report_fixture(phase="performance", elapsed_ms=200, target_passed=False)
    first = report_fixture("first_question", phase="performance", elapsed_ms=140, target_passed=True)
    first["phases"]["performance"]["pairs"].reverse()
    result = analysis.compare_runs([direct, first])
    assert result["passed"] and result["phase"] == "performance"
    assert len(result["comparisons"]) == 8
    assert result["summaries"][0]["target_passed"] is False
    assert result["summaries"][1]["target_passed"] is True
    group = result["summaries"][1]["performance"]["logical_cold"]
    assert group["expected_pairs"] == group["observed_pairs"] == group["valid_pairs"] == 2
    assert group["many_latency_ms_valid_pairs"] == {"p50": 140, "p95": 140, "p99": 140}
    assert group["many_change_vs_reference"]["p95"]["absolute_ms"] == -60
    assert group["many_change_vs_reference"]["p95"]["relative_percent"] == pytest.approx(-30)


def test_performance_checks_raw_probabilities_in_every_paired_response(analysis):
    direct = report_fixture(phase="performance")
    first = report_fixture("first_question", phase="performance")
    first["phases"]["performance"]["pairs"][3]["many"]["response"]["diagnostics"]["question_details"][-1][
        "probabilities"
    ] = [0.65, 0.35]
    result = analysis.compare_runs([direct, first])
    assert not result["passed"]
    failed = [comparison for comparison in result["comparisons"] if not comparison["passed"]]
    assert len(failed) == 1
    assert failed[0]["repeat"] == 1 and failed[0]["cache"] == "warm_repeat" and failed[0]["variant"] == "many"


def test_performance_rejects_data_failure_even_if_target_was_passed(analysis):
    direct, first = report_fixture(phase="performance"), report_fixture("first_question", phase="performance")
    first["phases"]["performance"]["data_passed"] = False
    result = analysis.compare_runs([direct, first])
    assert not result["passed"]
    assert any("data_passed=false" in error for error in result["errors"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "incomplete"])
def test_performance_requires_complete_pair_and_question_coverage(analysis, change):
    direct, first = report_fixture(phase="performance"), report_fixture("first_question", phase="performance")
    pairs = first["phases"]["performance"]["pairs"]
    if change == "missing":
        pairs.pop()
    elif change == "duplicate":
        pairs[-1] = copy.deepcopy(pairs[0])
    else:
        pairs[-1]["many"]["question_ids"].pop()
    result = analysis.compare_runs([direct, first])
    assert not result["passed"]


def test_skipped_performance_produces_error_report(analysis):
    reference = report_fixture(phase="performance")
    reference["phases"]["performance"] = {"passed": False, "skipped": "salt failed"}
    result = analysis.compare_runs([reference, report_fixture("first_question", phase="performance")])
    assert not result["passed"]
    assert result["summaries"][0]["performance"]["logical_cold"]["valid_pairs"] == 0


@pytest.mark.parametrize("field", ["input_sha256", "atol", "seed", "ratio_limit"])
def test_performance_rejects_input_or_protocol_drift(analysis, field):
    direct, first = report_fixture(phase="performance"), report_fixture("first_question", phase="performance")
    if field == "input_sha256":
        first[field] = "b" * 64
    else:
        first["parameters"][field] += 1
    assert not analysis.compare_runs([direct, first])["passed"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_compare_cli_validates_manifests_for_performance_reports(analysis, tmp_path, mismatch):
    reports, manifests = [], []
    for strategy in ("direct", "first_question"):
        report_path = tmp_path / f"{strategy}.json"
        report_path.write_text(json.dumps(report_fixture(strategy, phase="performance")), encoding="utf-8")
        manifest_path = tmp_path / f"{strategy}-manifest.json"
        manifest = manifest_fixture(strategy)
        if strategy == "first_question" and mismatch:
            manifest["vllm_ascend"]["sha"] = "different-code"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reports.append(str(report_path))
        manifests.append(str(manifest_path))
    output = tmp_path / "comparison.json"
    exit_code = analysis.main(["--compare", "--results", *reports, "--manifests", *manifests, "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == (2 if mismatch else 0)
    assert result["passed"] is not mismatch
