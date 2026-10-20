# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Trace-contract checks using a minimal upstream scheduler stub (CPU only)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/core/systemone_trace_scheduler.py"


@pytest.fixture
def trace_module(monkeypatch):
    class SchedulerStub:
        def __init__(self, vllm_config, kv_cache_config):
            self.vllm_config = vllm_config
            self.cache_config = vllm_config.cache_config
            self.kv_cache_config = kv_cache_config
            self.block_size = 1536
            self.hash_block_size = 64
            self.max_num_scheduled_tokens = 2048
            self.max_num_running_reqs = 8
            self.requests = {}
            self.waiting = []
            self.skipped_waiting = []
            self.running = []
            self.upstream_updates = 0

        def _update_after_schedule(self, output):
            self.upstream_updates += 1
            for request_id, count in output.num_scheduled_tokens.items():
                self.requests[request_id].num_computed_tokens += count

    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = SchedulerStub
    logger_module = ModuleType("vllm.logger")
    logger_module.logger = Mock()
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)
    spec = importlib.util.spec_from_file_location("systemone_trace_scheduler_cpu_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align"),
    )


def make_scheduler(module, config=None):
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=1536), layer_names=["attention"]),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=1536), layer_names=["mamba"]),
        ]
    )
    return module.SystemOneTraceScheduler(config or make_config(), kv_cache_config=kv_cache_config)


def make_request(request_id="systemone-client-parent-0-123456ab", computed=0, cached=0, status="RUNNING"):
    return SimpleNamespace(
        request_id=request_id,
        num_computed_tokens=computed,
        num_prompt_tokens=5000,
        num_preemptions=0,
        status=SimpleNamespace(name=status),
        prefill_stats=SimpleNamespace(num_prompt_tokens=5000, num_cached_tokens=cached),
    )


def make_output(scheduled=None, preempted=(), finished=()):
    scheduled = scheduled or {}
    return SimpleNamespace(
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        preempted_req_ids=set(preempted),
        finished_req_ids=set(finished),
    )


def log_records(module, prefix="SystemOne scheduler %s"):
    return [json.loads(call.args[1]) for call in module.logger.info.call_args_list if call.args[0] == prefix]


def test_runtime_reports_actual_group_and_hash_block_sizes(trace_module):
    make_scheduler(trace_module)
    runtime = log_records(trace_module, "SystemOne scheduler runtime %s")[0]
    assert runtime["scheduler_block_size"] == 1536
    assert runtime["hash_block_size"] == 64
    assert [group["block_size"] for group in runtime["cache_groups"]] == [1536, 1536]
    assert runtime["max_num_scheduled_tokens"] == 2048


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scheduler_config", "async_scheduling", True, "no-async-scheduling"),
        ("scheduler_config", "async_scheduling", None, "no-async-scheduling"),
        ("model_config", "enforce_eager", False, "enforce-eager"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "prefill_context_parallel_size=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "decode_context_parallel_size=1"),
    ],
)
def test_rejects_unsupported_observation_modes(trace_module, section, field, value, message):
    config = make_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        make_scheduler(trace_module, config)


def test_apc_admission_offset_is_observed_before_upstream_increment(trace_module):
    scheduler = make_scheduler(trace_module)
    request = make_request(computed=1536, cached=1536)
    scheduler.requests[request.request_id] = request
    scheduler.running = [request]
    scheduler._update_after_schedule(make_output({request.request_id: 1536}))
    scheduler._update_after_schedule(make_output({request.request_id: 1928}))
    first, second = log_records(trace_module)
    assert first["requests"][0]["computed_before"] == 1536
    assert second["requests"][0]["computed_before"] == 3072
    assert [step["requests"][0]["num_cached_tokens"] for step in (first, second)] == [1536, 1536]
    assert request.num_computed_tokens == 5000
    assert scheduler.upstream_updates == 2


def test_own_cold_chunks_are_not_cache_hits(trace_module):
    scheduler = make_scheduler(trace_module)
    request = make_request()
    scheduler.requests[request.request_id] = request
    scheduler.running = [request]
    scheduler._update_after_schedule(make_output({request.request_id: 1536}))
    scheduler._update_after_schedule(make_output({request.request_id: 1536}))
    row = log_records(trace_module)[1]["requests"][0]
    assert row["computed_before"] == 1536
    assert row["num_cached_tokens"] == 0
    assert row["cached_tokens_source"] == "request.prefill_stats.num_cached_tokens"


def test_preemption_is_separate_from_unknown_wait_reason(trace_module):
    scheduler = make_scheduler(trace_module)
    request = make_request(status="PREEMPTED")
    request.num_preemptions = 1
    scheduler.requests[request.request_id] = request
    scheduler.waiting = [request]
    scheduler._update_after_schedule(make_output(preempted=[request.request_id]))
    step = log_records(trace_module)[0]
    row = step["requests"][0]
    assert step["preempted_req_ids"] == [request.request_id]
    assert row["computed_before"] == 0
    assert row["preempted"] is True
    assert row["num_preemptions"] == 1
    assert row["queue"] == "waiting"
    assert row["waiting_reason"] == "unknown"
    assert row["num_scheduled_tokens"] == 0


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("WAITING", "unknown"),
        ("RUNNING", "unknown"),
        ("WAITING_FOR_REMOTE_KVS", "remote_kvs"),
        ("WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR", "structured_output_grammar"),
        ("WAITING_FOR_STREAMING_REQ", "streaming_input"),
    ],
)
def test_only_status_proven_wait_reasons_are_reported(trace_module, status, reason):
    scheduler = make_scheduler(trace_module)
    request = make_request(status=status)
    request.prefill_stats.num_prompt_tokens = 0
    scheduler.requests[request.request_id] = request
    scheduler.skipped_waiting = [request]
    scheduler._update_after_schedule(make_output())
    row = log_records(trace_module)[0]["requests"][0]
    assert row["waiting_reason"] == reason
    assert row["num_cached_tokens"] is None
    assert row["cached_tokens_source"] is None
    assert row["queue"] == "skipped_waiting"


def test_old_cache_counter_and_missing_external_id(trace_module):
    request = make_request()
    del request.prefill_stats
    request.num_cached_tokens = 1536
    scheduler = make_scheduler(trace_module)
    scheduler.requests[request.request_id] = request
    scheduler._update_after_schedule(make_output())
    row = log_records(trace_module)[0]["requests"][0]
    assert row["num_cached_tokens"] == 1536
    assert row["external_req_id"] is None
    request.num_cached_tokens = -1
    assert trace_module._cached_tokens(request) == (None, None)


def test_non_systemone_requests_are_filtered_but_count_in_engine_totals(trace_module):
    scheduler = make_scheduler(trace_module)
    ordinary = make_request("chatcmpl-ordinary")
    scheduler.requests[ordinary.request_id] = ordinary
    scheduler.running = [ordinary]
    scheduler._update_after_schedule(make_output({ordinary.request_id: 32}))
    assert log_records(trace_module) == []
    systemone = make_request(status="WAITING")
    scheduler.requests[systemone.request_id] = systemone
    scheduler.waiting = [systemone]
    scheduler._update_after_schedule(make_output({ordinary.request_id: 2048}))
    step = log_records(trace_module)[0]
    assert step["step_id"] == 2
    assert step["total_num_scheduled_tokens"] == 2048
    assert step["num_scheduled_requests"] == 1
    assert step["token_budget_exhausted"] is True
    assert [row["request_id"] for row in step["requests"]] == [systemone.request_id]
    assert step["requests"][0]["waiting_reason"] == "unknown"


def test_finished_ids_survive_without_retaining_request_history(trace_module):
    scheduler = make_scheduler(trace_module)
    initial_fields = set(vars(scheduler))
    for index in range(200):
        request = make_request(f"systemone-{index}")
        scheduler.requests = {request.request_id: request}
        scheduler.running = [request]
        scheduler._update_after_schedule(make_output({request.request_id: 16}))
        scheduler.requests.clear()
        scheduler.running.clear()
        scheduler._update_after_schedule(make_output(finished=[request.request_id]))
    assert set(vars(scheduler)) == initial_fields
    assert scheduler._systemone_step_id == 400
    assert log_records(trace_module)[-1]["requests"] == []
    assert log_records(trace_module)[-1]["finished_req_ids"] == ["systemone-199"]
    assert scheduler.requests == {}
