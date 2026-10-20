# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in, synchronous scheduler observations for SystemOne experiments.

Select with ``--scheduler-cls
vllm_ascend.core.systemone_trace_scheduler.SystemOneTraceScheduler``.
This class delegates scheduling to vLLM and only observes its final decision.
The hook matches vLLM v0.27.1's ``Scheduler._update_after_schedule`` contract.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


def _cached_tokens(request: Request) -> tuple[int | None, str | None]:
    """Read cache statistics, never infer cache hits from computed progress.

    v0.27.1 uses PrefillStats and clears it after the first engine output.
    Its zero-initialized values do not describe a lookup until set() fills in
    num_prompt_tokens. Older serving builds expose num_cached_tokens directly.
    These counters describe the first prefill, not a post-preemption lookup.
    """
    stats = getattr(request, "prefill_stats", None)
    if stats is not None and getattr(stats, "num_prompt_tokens", 0) > 0:
        value = getattr(stats, "num_cached_tokens", None)
        if type(value) is int and value >= 0:
            return value, "request.prefill_stats.num_cached_tokens"
    value = getattr(request, "num_cached_tokens", None)
    if type(value) is int and value >= 0:
        return value, "request.num_cached_tokens"
    return None, None


def _status_name(request: Request) -> str:
    return getattr(request.status, "name", str(request.status))


def _waiting_reason(status: str, scheduled_tokens: int) -> str | None:
    if scheduled_tokens:
        return None
    # A queue snapshot cannot establish whether budget, KV capacity, request
    # priority, or another constraint actually prevented this admission.
    if status == "WAITING_FOR_REMOTE_KVS":
        return "remote_kvs"
    if status == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR":
        return "structured_output_grammar"
    if status == "WAITING_FOR_STREAMING_REQ":
        return "streaming_input"
    return "unknown"


class SystemOneTraceScheduler(Scheduler):
    """Log live SystemOne requests before vLLM advances computed counters.

    No per-request history is retained. Each step contains a snapshot of live
    requests, plus finished/preempted IDs supplied by the scheduler output.
    Logging and queue inspection add overhead; use this for diagnosis only.
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        if vllm_config.scheduler_config.async_scheduling is not False:
            raise ValueError("SystemOneTraceScheduler requires --no-async-scheduling")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("SystemOneTraceScheduler requires --enforce-eager")
        parallel = vllm_config.parallel_config
        for field in (
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, field, 1) != 1:
                raise ValueError(f"SystemOneTraceScheduler requires {field}=1")
        super().__init__(vllm_config, *args, **kwargs)
        self._systemone_step_id = 0
        groups = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            groups.append(
                {
                    "group_id": group_id,
                    "cache_spec_type": type(spec).__name__,
                    "block_size": spec.block_size,
                    "num_layers": len(group.layer_names),
                }
            )
        logger.info(
            "SystemOne scheduler runtime %s",
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_block_size": self.block_size,
                    "hash_block_size": self.hash_block_size,
                    "cache_groups": groups,
                    "enable_prefix_caching": self.cache_config.enable_prefix_caching,
                    "mamba_cache_mode": getattr(self.cache_config, "mamba_cache_mode", None),
                    "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
                    "max_num_seqs": self.max_num_running_reqs,
                    "async_scheduling": False,
                    "enforce_eager": True,
                    "data_parallel_size": parallel.data_parallel_size,
                },
                separators=(",", ":"),
            ),
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # At this point the first APC lookup has already assigned the admission
        # offset. super() subsequently advances num_computed_tokens by q_i.
        self._systemone_step_id += 1
        self._log_systemone_step(scheduler_output)
        super()._update_after_schedule(scheduler_output)

    def _log_systemone_step(self, output: SchedulerOutput) -> None:
        requests = [request for request in self.requests.values() if request.request_id.startswith("systemone-")]
        finished = sorted(request_id for request_id in output.finished_req_ids if request_id.startswith("systemone-"))
        preempted = output.preempted_req_ids or set()
        systemone_preempted = sorted(request_id for request_id in preempted if request_id.startswith("systemone-"))
        if not requests and not finished and not systemone_preempted:
            return
        waiting = {request.request_id for request in self.waiting}
        skipped = {request.request_id for request in getattr(self, "skipped_waiting", ())}
        running = {request.request_id for request in self.running}
        rows = []
        for request in sorted(requests, key=lambda request: request.request_id):
            request_id = request.request_id
            scheduled_tokens = output.num_scheduled_tokens.get(request_id, 0)
            cached_tokens, cached_tokens_source = _cached_tokens(request)
            status = _status_name(request)
            queue = (
                "waiting"
                if request_id in waiting
                else "skipped_waiting"
                if request_id in skipped
                else "running"
                if request_id in running
                else "other"
            )
            rows.append(
                {
                    "request_id": request_id,
                    # EngineCoreRequest has an external ID, but v0.27.1's
                    # scheduler Request does not retain it. Do not guess it.
                    "external_req_id": getattr(request, "external_req_id", None),
                    "num_scheduled_tokens": scheduled_tokens,
                    "computed_before": request.num_computed_tokens,
                    "num_cached_tokens": cached_tokens,
                    "cached_tokens_source": cached_tokens_source,
                    "prompt_tokens": request.num_prompt_tokens,
                    "status": status,
                    "queue": queue,
                    "waiting_reason": _waiting_reason(status, scheduled_tokens),
                    "preempted": request_id in preempted,
                    "num_preemptions": request.num_preemptions,
                }
            )
        logger.info(
            "SystemOne scheduler %s",
            json.dumps(
                {
                    "schema_version": 1,
                    "step_id": self._systemone_step_id,
                    "wall_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
                    "num_scheduled_requests": len(output.num_scheduled_tokens),
                    "num_running_requests": len(running),
                    "num_waiting_requests": len(waiting | skipped),
                    "token_budget_exhausted": output.total_num_scheduled_tokens >= self.max_num_scheduled_tokens,
                    "running_capacity_full": len(running) >= self.max_num_running_reqs,
                    "preempted_req_ids": systemone_preempted,
                    "finished_req_ids": finished,
                    "requests": rows,
                },
                separators=(",", ":"),
            ),
        )
