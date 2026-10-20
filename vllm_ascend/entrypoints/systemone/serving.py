# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-native answer-token readout through vLLM generation and prefix caching."""

import asyncio
import hashlib
import json
import math
import time
import uuid
from contextlib import aclosing, asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm import envs
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, validate_json_request, with_cancellation
from vllm.exceptions import VLLMNotFoundError
from vllm.logger import logger
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tracing import contains_trace_headers, extract_trace_headers, log_tracing_disabled_warning
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.async_utils import make_async

from .config import load_readout_config
from .protocol import MAX_OPTIONS, SystemOneRequest, get_answer_token_ids, plan_request, probabilities, to_answers
from .shared_prefix import SharedPrefixPlan, inspect_shared_prefix_capability, plan_shared_prefix


def load_temperature(model_config):
    # Resolve the calibration file from the same snapshot as the loaded model.
    revision = getattr(model_config.hf_config, "_commit_hash", None) or model_config.revision
    config = get_hf_file_to_dict("jevk5_config.json", model_config.model, revision)
    if not isinstance(config, dict) or "temperature" not in config:
        raise ValueError("Plumb requires jevk5_config.json with its calibrated temperature next to the model weights")
    temperature = config["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Plumb temperature must be a finite positive number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Plumb temperature must be a finite positive number")
    return float(temperature)


def validate_runtime(config):
    from vllm_ascend.ascend_config import get_ascend_config

    model = config.model_config
    if model.runner_type != "generate" or model.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("SystemOne requires a Qwen3.5/3.6 language model with --runner generate")
    if config.use_v2_model_runner:
        raise ValueError("SystemOne candidate logprobs require VLLM_USE_V2_MODEL_RUNNER=0")
    if model.logprobs_mode != "raw_logprobs":
        raise ValueError("SystemOne requires --logprobs-mode raw_logprobs")
    if model.max_logprobs != -1 and model.max_logprobs < MAX_OPTIONS:
        raise ValueError(f"SystemOne requires --max-logprobs at least {MAX_OPTIONS}")
    if config.speculative_config is not None or config.quant_config is not None:
        raise ValueError("SystemOne initially requires unquantized weights without speculative decoding")
    if get_ascend_config().enable_reduce_sample:
        raise ValueError("SystemOne candidate logprobs require enable_reduce_sample=false")
    if config.cache_config.enable_prefix_caching and (
        config.cache_config.mamba_cache_mode != "align" or not config.scheduler_config.enable_chunked_prefill
    ):
        raise ValueError(
            "SystemOne hybrid prefix caching requires --mamba-cache-mode align and --enable-chunked-prefill"
        )


def extract_probabilities(output, plan, temperature):
    if len(output.outputs) != 1:
        raise RuntimeError("SystemOne expected exactly one completion per question")
    completion = output.outputs[0]
    if completion.finish_reason in ("error", "abort"):
        raise RuntimeError(f"SystemOne question ended with {completion.finish_reason}")
    if len(completion.token_ids) != 1 or not completion.logprobs or len(completion.logprobs) != 1:
        raise RuntimeError("SystemOne did not receive one complete step of candidate logprobs")
    scores = completion.logprobs[0]
    if any(token_id not in scores for token_id in plan.candidate_token_ids):
        raise RuntimeError("SystemOne response is missing candidate token logprobs; use the V1 runner")
    # logprob_i = logit_i - logsumexp(all logits). The common constant cancels
    # in the candidate-only softmax, preserving JevK5's calibrated readout.
    try:
        return probabilities([scores[token_id].logprob for token_id in plan.candidate_token_ids], temperature)
    except ValueError as exc:
        raise RuntimeError("SystemOne engine returned invalid candidate scores") from exc


class SystemOneService:
    def __init__(self, engine):
        validate_runtime(engine.vllm_config)
        self.engine = engine
        self.readout_config = load_readout_config(engine.vllm_config)
        self.shared_prefix_capability = inspect_shared_prefix_capability(
            engine.vllm_config,
            self.readout_config.shared_prefix,
            prefix_cache_retention_interval=getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None),
        )
        if self.readout_config.mode == "plumb":
            self.temperature = load_temperature(engine.model_config)
        else:
            self.temperature = self.readout_config.temperature if self.readout_config.temperature is not None else 1.0
        self.tokenizer = engine.renderer.get_tokenizer()
        get_answer_token_ids(self.tokenizer)
        self.max_model_len = engine.model_config.max_model_len
        names = engine.model_config.served_model_name
        names = [names] if isinstance(names, str) else list(names or [engine.model_config.model])
        self.default_model = names[0]
        self.names = set(names)
        self.plan_async = make_async(self._plan_request, executor=engine.renderer._executor)
        self.response_async = make_async(self._build_response, executor=engine.renderer._executor)

    async def evaluate(self, request, raw_request):
        model_name = self._model_name(request)
        if not envs.VLLM_SKIP_MODEL_NAME_VALIDATION and model_name not in self.names:
            raise VLLMNotFoundError(f"Unknown served model: {model_name}")
        started = time.perf_counter()
        plans, input_tokens, shared_plan, planning = await self.plan_async(request, time.perf_counter_ns())
        planning_ms = (time.perf_counter() - started) * 1000
        planning["planning_ms"] = round(planning_ms, 3)
        distributions, observation = await self._execute(plans, request, raw_request, shared_plan, planning)
        response = await self.response_async(request, plans, distributions, input_tokens, started)
        if self.readout_config.diagnostics:
            response["diagnostics"] = observation
        return response

    def _model_name(self, request):
        return self.default_model if request.model is None else request.model

    def _plan_request(self, request, queued_ns):
        started_ns = time.perf_counter_ns()
        plans, input_tokens = plan_request(request, self.tokenizer, self.max_model_len)
        planned_ns = time.perf_counter_ns()
        shared = self.readout_config.shared_prefix
        if shared.strategy == "first_question":
            # This strategy uses a complete original question. It does not scan
            # an LCP or apply producer-only min_shared_tokens.
            reason = "single_question" if len(plans) == 1 else self.shared_prefix_capability.reason
            if len(plans) > 1 and reason == "ready":
                reason = "first_question_ready"
            shared_plan = SharedPrefixPlan(0, 0, reason)
            lcp_ms = 0.0
        else:
            shared_plan = plan_shared_prefix(plans, self.shared_prefix_capability, shared.min_shared_tokens)
            lcp_ms = (time.perf_counter_ns() - planned_ns) / 1_000_000
        planning = {
            "executor_queue_ms": round((started_ns - queued_ns) / 1_000_000, 3),
            "plan_request_ms": round((planned_ns - started_ns) / 1_000_000, 3),
            "lcp_ms": round(lcp_ms, 3),
        }
        return plans, input_tokens, shared_plan, planning

    def _build_response(self, request, plans, distributions, input_tokens, started):
        return {
            "model": self._model_name(request),
            "answers": to_answers(distributions, plans),
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def _execute(self, plans, request, raw_request, shared_plan, planning):
        parent_request_id = BaseServing._base_request_id(raw_request)
        # Public abort uses external IDs. Do not let repeated client headers
        # make separate parent groups share an abort identity.
        parent = f"systemone-{parent_request_id}-{uuid.uuid4().hex}"
        trace_headers = None
        if contains_trace_headers(raw_request.headers):
            if await self.engine.is_tracing_enabled():
                trace_headers = extract_trace_headers(raw_request.headers)
            else:
                log_tracing_disabled_warning()

        first_question = shared_plan.reason == "first_question_ready"
        strategy = "first_question" if first_question else "producer" if shared_plan.prefill_tokens else "direct"
        observation = {
            "request_id": parent,
            "parent_request_id": parent_request_id,
            "strategy": strategy,
            "configured_strategy": self.readout_config.shared_prefix.strategy
            if self.readout_config.shared_prefix.enabled
            else "direct",
            "reason": shared_plan.reason,
            "questions": len(plans),
            "common_tokens": shared_plan.common_tokens,
            "prefill_tokens": shared_plan.prefill_tokens,
            "prompt_tokens": [len(plan.token_ids) for plan in plans],
            "unshared_common_tokens": shared_plan.common_tokens - shared_plan.prefill_tokens,
            **planning,
            "prefill_ms": None,
            "first_question_ms": None,
            "branch_ms": None,
            "prefill_cached_tokens": None,
            "branch_cached_tokens": [None] * len(plans),
            "reuse": "not_attempted",
            "status": "failed",
        }
        details = None
        producer = None
        if self.readout_config.diagnostics:
            details = [
                {
                    **self._request_observation(f"{parent}-{index}", len(plan.token_ids)),
                    "index": index,
                    "question_id": plan.question_id,
                    "probabilities": None,
                }
                for index, plan in enumerate(plans)
            ]
            if shared_plan.prefill_tokens:
                producer = self._request_observation(f"{parent}-prefill", shared_plan.prefill_tokens)
            observation.update(
                schema_version=1,
                runtime=self.runtime_observation(),
                cache_salt_sha256=hashlib.sha256(request.cache_salt.encode("utf-8")).hexdigest()
                if request.cache_salt is not None
                else None,
                question_details=details,
                producer=producer,
            )
        try:
            if shared_plan.prefill_tokens:
                prefill_started = time.perf_counter()
                try:
                    observation["prefill_cached_tokens"] = await self._prefill(
                        plans[0].token_ids[: shared_plan.prefill_tokens], request, parent, trace_headers, producer
                    )
                finally:
                    observation["prefill_ms"] = round((time.perf_counter() - prefill_started) * 1000, 3)
            first_result = None
            if first_question:
                first_started = time.perf_counter()
                try:
                    first_result = await self._consume_question(
                        0, plans[0], request, parent, trace_headers, observation["branch_cached_tokens"], details
                    )
                finally:
                    observation["first_question_ms"] = round((time.perf_counter() - first_started) * 1000, 3)
            branch_started = time.perf_counter()
            try:
                results = await self._execute_questions(
                    plans,
                    request,
                    parent,
                    trace_headers,
                    observation["branch_cached_tokens"],
                    details,
                    start_index=1 if first_question else 0,
                )
                if first_question:
                    results.insert(0, first_result)
            finally:
                observation["branch_ms"] = round((time.perf_counter() - branch_started) * 1000, 3)
            if shared_plan.prefill_tokens:
                cached = observation["branch_cached_tokens"]
                if any(count is not None and count < shared_plan.prefill_tokens for count in cached):
                    observation["reuse"] = "insufficient"
                elif any(count is None for count in cached):
                    observation["reuse"] = "unavailable"
                else:
                    observation["reuse"] = "reported_sufficient"
            elif first_question:
                # With no LCP scan there is no expected hit count to compare.
                # These are observations, not proof of complete hybrid reuse.
                observation["reuse"] = (
                    "unavailable"
                    if any(count is None for count in observation["branch_cached_tokens"][1:])
                    else "reported"
                )
            observation["status"] = "completed"
            return results, observation
        except asyncio.CancelledError:
            observation["status"] = "cancelled"
            raise
        finally:
            # Counts are the engine's reported cache hits, not a measurement of
            # device work, snapshot correctness, eviction or recomputation.
            logger.info("SystemOne shared_prefix %s", json.dumps(observation))

    def _generate(self, token_ids, params, request, request_id, trace_headers):
        prompt = {"prompt_token_ids": token_ids}
        if request.cache_salt is not None:
            prompt["cache_salt"] = request.cache_salt
        return self.engine.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=request_id,
            priority=request.priority,
            trace_headers=trace_headers,
        )

    @staticmethod
    def _request_observation(request_id, prompt_tokens):
        return {
            # Public generate/abort ID, also used for correlation in engine logs.
            "engine_request_id": request_id,
            "output_request_id": None,
            "prompt_tokens": prompt_tokens,
            "submit_monotonic_ns": None,
            "submit_wall_ns": None,
            "complete_monotonic_ns": None,
            "complete_wall_ns": None,
            "cached_tokens": None,
            "status": "not_submitted",
        }

    @asynccontextmanager
    async def _request_stream(self, token_ids, params, request, request_id, trace_headers, observation=None):
        if observation is not None:
            # Endpoint boundaries, not engine admission or scheduler timestamps.
            observation["submit_monotonic_ns"] = time.monotonic_ns()
            observation["submit_wall_ns"] = time.time_ns()
            observation["status"] = "submitted"
        try:
            async with aclosing(self._generate(token_ids, params, request, request_id, trace_headers)) as generator:
                yield generator
        except BaseException as exc:
            if observation is not None:
                observation["status"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            # Native generate may be cancelled after local registration but
            # before add_request returns its collector. Close alone cannot
            # abort that window in vLLM 0.27.1; public abort resolves the ID.
            try:
                await self.engine.abort(request_id)
            except Exception:
                logger.exception("SystemOne could not abort unfinished request %s", request_id)
            raise
        else:
            if observation is not None:
                observation["status"] = "completed"
        finally:
            if observation is not None:
                observation["complete_monotonic_ns"] = time.monotonic_ns()
                observation["complete_wall_ns"] = time.time_ns()

    async def _prefill(self, token_ids, request, parent, trace_headers, observation=None):
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        finished = False
        cached_tokens = None
        async with self._request_stream(
            token_ids, params, request, f"{parent}-prefill", trace_headers, observation
        ) as generator:
            async for output in generator:
                if output.finished:
                    if (
                        len(output.outputs) != 1
                        or output.outputs[0].finish_reason not in ("length", "stop")
                        or len(output.outputs[0].token_ids) != 1
                    ):
                        raise RuntimeError("SystemOne shared-prefix prefill did not complete normally")
                    finished = True
                    cached_tokens = getattr(output, "num_cached_tokens", None)
                    if observation is not None:
                        observation["output_request_id"] = getattr(output, "request_id", None)
                        observation["cached_tokens"] = cached_tokens
            if not finished:
                raise RuntimeError("SystemOne shared-prefix prefill ended without a finished result")
        return cached_tokens

    async def _consume_question(self, index, plan, request, parent, trace_headers, cached_tokens, details=None):
        observation = details[index] if details is not None else None
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            logprobs=len(plan.candidate_token_ids),
            logprob_token_ids=plan.candidate_token_ids,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        result = None
        async with self._request_stream(
            plan.token_ids, params, request, f"{parent}-{index}", trace_headers, observation
        ) as generator:
            async for output in generator:
                if output.finished:
                    cached_tokens[index] = getattr(output, "num_cached_tokens", None)
                    result = extract_probabilities(output, plan, self.temperature)
                    if observation is not None:
                        observation["output_request_id"] = getattr(output, "request_id", None)
                        observation["cached_tokens"] = cached_tokens[index]
                        observation["probabilities"] = result
            if result is None:
                raise RuntimeError("SystemOne did not receive results for all questions")
        return result

    async def _execute_questions(
        self, plans, request, parent, trace_headers, cached_tokens, details=None, start_index=0
    ):
        # Each task owns and closes its native generator. Cancel and await every
        # sibling before returning so native abort completes on failure/cancel.
        tasks = [
            asyncio.create_task(
                self._consume_question(index, plan, request, parent, trace_headers, cached_tokens, details)
            )
            for index, plan in enumerate(plans)
            if index >= start_index
        ]
        try:
            # as_completed does not cancel children when its consumer is
            # cancelled. This owner cancels once, preserving slow aborts.
            for completed in asyncio.as_completed(tasks):
                await completed
            return [task.result() for task in tasks]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def runtime_observation(self):
        config = self.engine.vllm_config
        cache = config.cache_config
        scheduler = config.scheduler_config
        model = config.model_config
        parallel = getattr(config, "parallel_config", None)
        shared = self.readout_config.shared_prefix
        return {
            "schema_version": 1,
            "model": self.default_model,
            "model_path": model.model,
            "model_revision": getattr(model.hf_config, "_commit_hash", None) or model.revision,
            "dtype": str(model.dtype) if getattr(model, "dtype", None) is not None else None,
            "enforce_eager": getattr(model, "enforce_eager", None),
            "use_v2_model_runner": config.use_v2_model_runner,
            "mode": self.readout_config.mode,
            "temperature": self.temperature,
            "max_prompt_tokens": self.max_model_len - 1,
            "diagnostics": self.readout_config.diagnostics,
            "shared_prefix": {
                **shared.model_dump(),
                "capability_reason": self.shared_prefix_capability.reason,
                "min_shared_tokens_applies": shared.strategy == "producer",
            },
            "cache": {
                field: getattr(cache, field, None)
                for field in (
                    "enable_prefix_caching",
                    "mamba_cache_mode",
                    "block_size",
                    "mamba_block_size",
                    "prefix_match_unit",
                    "num_gpu_blocks",
                    "cache_dtype",
                )
            },
            "scheduler": {
                field: getattr(scheduler, field, None)
                for field in (
                    "enable_chunked_prefill",
                    "async_scheduling",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "max_num_partial_prefills",
                    "max_long_partial_prefills",
                    "long_prefill_token_threshold",
                )
            },
            "parallel": {
                field: getattr(parallel, field, None)
                for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
            },
            "prefix_cache_retention_interval": getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None),
        }


class SystemOnePlugin:
    name = "ascend_systemone"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI):
        @app.post("/v1/systemone", dependencies=[Depends(validate_json_request)])
        @with_cancellation
        @load_aware_call
        async def systemone(request: SystemOneRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "systemone_service", None)
            if service is None:
                raise HTTPException(503, "SystemOne engine is not available")
            return JSONResponse(content=await service.evaluate(request, raw_request))

    async def init_state(self, engine_client, state, args):
        state.systemone_service = None
        if engine_client is not None:
            state.systemone_service = SystemOneService(engine_client)
            logger.info("SystemOne runtime %s", json.dumps(state.systemone_service.runtime_observation()))
