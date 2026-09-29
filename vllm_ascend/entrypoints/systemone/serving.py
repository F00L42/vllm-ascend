# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Plumb decisions through the native vLLM generation and prefix-cache path."""

import logging
import math
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm import envs
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, validate_json_request, with_cancellation
from vllm.exceptions import VLLMNotFoundError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tracing import contains_trace_headers, extract_trace_headers, log_tracing_disabled_warning
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.async_utils import make_async, merge_async_iterators

from .protocol import MAX_OPTIONS, SystemOneRequest, get_answer_token_ids, plan_request, probabilities, to_answers

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
        raise ValueError("Plumb SystemOne requires the Qwen3.5 text model with --runner generate")
    if config.use_v2_model_runner:
        raise ValueError("Plumb candidate logprobs require VLLM_USE_V2_MODEL_RUNNER=0")
    if model.logprobs_mode != "raw_logprobs":
        raise ValueError("Plumb requires --logprobs-mode raw_logprobs")
    if model.max_logprobs != -1 and model.max_logprobs < MAX_OPTIONS:
        raise ValueError(f"Plumb requires --max-logprobs at least {MAX_OPTIONS}")
    if config.speculative_config is not None or config.quant_config is not None:
        raise ValueError("Plumb initially requires unquantized weights without speculative decoding")
    if get_ascend_config().enable_reduce_sample:
        raise ValueError("Plumb candidate logprobs require enable_reduce_sample=false")
    if config.cache_config.enable_prefix_caching and (
        config.cache_config.mamba_cache_mode != "align" or not config.scheduler_config.enable_chunked_prefill
    ):
        raise ValueError("Plumb hybrid prefix caching requires --mamba-cache-mode align and --enable-chunked-prefill")


def extract_probabilities(output, plan, temperature):
    if len(output.outputs) != 1:
        raise RuntimeError("Plumb expected exactly one completion per question")
    completion = output.outputs[0]
    if completion.finish_reason in ("error", "abort"):
        raise RuntimeError(f"Plumb question ended with {completion.finish_reason}")
    if len(completion.token_ids) != 1 or not completion.logprobs or len(completion.logprobs) != 1:
        raise RuntimeError("Plumb did not receive one complete step of candidate logprobs")
    scores = completion.logprobs[0]
    if any(token_id not in scores for token_id in plan.candidate_token_ids):
        raise RuntimeError("Plumb response is missing candidate token logprobs; use the V1 runner")
    # logprob_i = logit_i - logsumexp(all logits). The common constant cancels
    # in the candidate-only softmax, preserving JevK5's calibrated readout.
    try:
        return probabilities([scores[token_id].logprob for token_id in plan.candidate_token_ids], temperature)
    except ValueError as exc:
        raise RuntimeError("Plumb engine returned invalid candidate scores") from exc


class PlumbService:
    def __init__(self, engine):
        validate_runtime(engine.vllm_config)
        self.engine = engine
        self.temperature = load_temperature(engine.model_config)
        self.tokenizer = engine.renderer.get_tokenizer()
        get_answer_token_ids(self.tokenizer)
        self.max_model_len = engine.model_config.max_model_len
        names = engine.model_config.served_model_name
        self.names = {names} if isinstance(names, str) else set(names or [])
        self.plan_async = make_async(plan_request, executor=engine.renderer._executor)
        self.response_async = make_async(self._build_response, executor=engine.renderer._executor)

    async def evaluate(self, request, raw_request):
        if not envs.VLLM_SKIP_MODEL_NAME_VALIDATION and request.model not in self.names:
            raise VLLMNotFoundError(f"Unknown served model: {request.model}")
        started = time.perf_counter()
        plans, input_tokens = await self.plan_async(request, self.tokenizer, self.max_model_len)
        distributions = await self._execute(plans, request, raw_request)
        return await self.response_async(request, plans, distributions, input_tokens, started)

    def _build_response(self, request, plans, distributions, input_tokens, started):
        return {
            "model": request.model,
            "answers": to_answers(distributions, plans),
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def _execute(self, plans, request, raw_request):
        parent = f"plumb-{BaseServing._base_request_id(raw_request)}"
        trace_headers = None
        if contains_trace_headers(raw_request.headers):
            if await self.engine.is_tracing_enabled():
                trace_headers = extract_trace_headers(raw_request.headers)
            else:
                log_tracing_disabled_warning()

        generators = []
        for index, plan in enumerate(plans):
            params = SamplingParams(
                temperature=0,
                max_tokens=1,
                ignore_eos=True,
                detokenize=False,
                logprobs=len(plan.candidate_token_ids),
                logprob_token_ids=plan.candidate_token_ids,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            prompt = {"prompt_token_ids": plan.token_ids}
            if request.cache_salt is not None:
                prompt["cache_salt"] = request.cache_salt
            generators.append(
                self.engine.generate(
                    prompt=prompt,
                    sampling_params=params,
                    request_id=f"{parent}-{index}",
                    priority=request.priority,
                    trace_headers=trace_headers,
                )
            )

        # Each question uses the public generate lifecycle, including native
        # batching, DP routing, prefix caching and abort on cancellation.
        results = [None] * len(plans)
        merged = merge_async_iterators(*generators)
        try:
            async for index, output in merged:
                if output.finished:
                    results[index] = extract_probabilities(output, plans[index], self.temperature)
        finally:
            await merged.aclose()
        if any(result is None for result in results):
            raise RuntimeError("Plumb did not receive results for all questions")
        return results


class SystemOnePlugin:
    name = "ascend_systemone"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI):
        @app.post("/v1/systemone", dependencies=[Depends(validate_json_request)])
        @with_cancellation
        @load_aware_call
        async def systemone(request: SystemOneRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "plumb_service", None)
            if service is None:
                raise HTTPException(503, "Plumb engine is not available")
            return JSONResponse(content=await service.evaluate(request, raw_request))

    async def init_state(self, engine_client, state, args):
        state.plumb_service = None
        if engine_client is not None:
            state.plumb_service = PlumbService(engine_client)
            logger.info(
                "Plumb endpoint /v1/systemone initialized: temperature=%s, max_prompt_tokens=%d, prefix_caching=%s",
                state.plumb_service.temperature,
                state.plumb_service.max_model_len - 1,
                engine_client.vllm_config.cache_config.enable_prefix_caching,
            )
