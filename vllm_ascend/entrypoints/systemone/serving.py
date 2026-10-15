# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-native answer-token readout through vLLM generation and prefix caching."""

import asyncio
import json
import logging
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
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tracing import contains_trace_headers, extract_trace_headers, log_tracing_disabled_warning
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.async_utils import make_async

from .config import load_readout_config
from .protocol import MAX_OPTIONS, SystemOneRequest, get_answer_token_ids, plan_request, probabilities, to_answers
from .shared_prefix import inspect_shared_prefix_capability, plan_shared_prefix

logger = logging.getLogger(__name__)


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
        plans, input_tokens, shared_plan = await self.plan_async(request)
        planning_ms = (time.perf_counter() - started) * 1000
        distributions = await self._execute(plans, request, raw_request, shared_plan, planning_ms)
        return await self.response_async(request, plans, distributions, input_tokens, started)

    def _model_name(self, request):
        return self.default_model if request.model is None else request.model

    def _plan_request(self, request):
        plans, input_tokens = plan_request(request, self.tokenizer, self.max_model_len)
        shared_plan = plan_shared_prefix(
            plans, self.shared_prefix_capability, self.readout_config.shared_prefix.min_shared_tokens
        )
        return plans, input_tokens, shared_plan

    def _build_response(self, request, plans, distributions, input_tokens, started):
        return {
            "model": self._model_name(request),
            "answers": to_answers(distributions, plans),
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def _execute(self, plans, request, raw_request, shared_plan, planning_ms):
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

        observation = {
            "request_id": parent,
            "parent_request_id": parent_request_id,
            "reason": shared_plan.reason,
            "questions": len(plans),
            "common_tokens": shared_plan.common_tokens,
            "prefill_tokens": shared_plan.prefill_tokens,
            "prompt_tokens": [len(plan.token_ids) for plan in plans],
            "unshared_common_tokens": shared_plan.common_tokens - shared_plan.prefill_tokens,
            "planning_ms": round(planning_ms, 3),
            "prefill_ms": None,
            "branch_ms": None,
            "prefill_cached_tokens": None,
            "branch_cached_tokens": [None] * len(plans),
            "reuse": "not_attempted",
            "status": "failed",
        }
        try:
            if shared_plan.prefill_tokens:
                prefill_started = time.perf_counter()
                try:
                    observation["prefill_cached_tokens"] = await self._prefill(
                        plans[0].token_ids[: shared_plan.prefill_tokens], request, parent, trace_headers
                    )
                finally:
                    observation["prefill_ms"] = round((time.perf_counter() - prefill_started) * 1000, 3)
            branch_started = time.perf_counter()
            try:
                results = await self._execute_questions(
                    plans, request, parent, trace_headers, observation["branch_cached_tokens"]
                )
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
            observation["status"] = "completed"
            return results
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

    @asynccontextmanager
    async def _request_stream(self, token_ids, params, request, request_id, trace_headers):
        try:
            async with aclosing(self._generate(token_ids, params, request, request_id, trace_headers)) as generator:
                yield generator
        except BaseException:
            # Native generate may be cancelled after local registration but
            # before add_request returns its collector. Close alone cannot
            # abort that window in vLLM 0.27.1; public abort resolves the ID.
            try:
                await self.engine.abort(request_id)
            except Exception:
                logger.exception("SystemOne could not abort unfinished request %s", request_id)
            raise

    async def _prefill(self, token_ids, request, parent, trace_headers):
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        finished = False
        cached_tokens = None
        async with self._request_stream(token_ids, params, request, f"{parent}-prefill", trace_headers) as generator:
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
            if not finished:
                raise RuntimeError("SystemOne shared-prefix prefill ended without a finished result")
        return cached_tokens

    async def _consume_question(self, index, plan, request, parent, trace_headers, cached_tokens):
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
            plan.token_ids, params, request, f"{parent}-{index}", trace_headers
        ) as generator:
            async for output in generator:
                if output.finished:
                    cached_tokens[index] = getattr(output, "num_cached_tokens", None)
                    result = extract_probabilities(output, plan, self.temperature)
            if result is None:
                raise RuntimeError("SystemOne did not receive results for all questions")
        return result

    async def _execute_questions(self, plans, request, parent, trace_headers, cached_tokens):
        # Each task owns and closes its native generator. Cancel and await every
        # sibling before returning so native abort completes on failure/cancel.
        tasks = [
            asyncio.create_task(self._consume_question(index, plan, request, parent, trace_headers, cached_tokens))
            for index, plan in enumerate(plans)
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
            logger.info(
                "SystemOne endpoint /v1/systemone initialized: model=%s, mode=%s, temperature=%s, "
                "max_prompt_tokens=%d, prefix_caching=%s, shared_prefix=%s",
                state.systemone_service.default_model,
                state.systemone_service.readout_config.mode,
                state.systemone_service.temperature,
                state.systemone_service.max_model_len - 1,
                engine_client.vllm_config.cache_config.enable_prefix_caching,
                state.systemone_service.shared_prefix_capability,
            )
            cache = engine_client.vllm_config.cache_config
            logger.info(
                "SystemOne shared_prefix runtime: block_size=%s, mamba_block_size=%s, prefix_match_unit=%s, "
                "retention_interval=%s",
                getattr(cache, "block_size", None),
                getattr(cache, "mamba_block_size", None),
                getattr(cache, "prefix_match_unit", None),
                getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None),
            )
