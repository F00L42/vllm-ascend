# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU service contract tests; no vLLM installation or accelerator is required."""

import ast
import asyncio
import contextlib
import functools
import importlib.util
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SOURCE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/entrypoints/systemone"


async def _merge_async_iterators(*iterators):
    # The native vLLM merge implementation from v0.27.1, commit
    # 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac, async_utils.py (Apache-2.0).
    # Kept here so cancellation tests exercise its actual cleanup behavior
    # without requiring a separate local vLLM checkout or GPU dependencies.
    if len(iterators) == 1:
        iterator = iterators[0]
        try:
            async for item in iterator:
                yield 0, item
            iterator = None
        finally:
            if iterator is not None:
                with contextlib.suppress(BaseException):
                    await iterator.aclose()
        return

    loop = asyncio.get_running_loop()
    awaits = {loop.create_task(anext(it)): (i, it) for i, it in enumerate(iterators)}
    try:
        while awaits:
            done, _ = await asyncio.wait(awaits.keys(), return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[loop.create_task(anext(it))] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


def _make_async(function, executor):
    async def call(*args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(executor, functools.partial(function, *args, **kwargs))

    return call


class _DependencyImports(ast.NodeTransformer):
    """Inject framework dependencies while executing the actual service source."""

    def visit_ImportFrom(self, node):
        if (node.module or "").startswith("vllm") or (node.level and node.module == "protocol"):
            return None
        return node


@pytest.fixture
def serving(monkeypatch):
    spec = importlib.util.spec_from_file_location("plumb_serving_protocol_cpu", _SOURCE_DIR / "protocol.py")
    assert spec is not None and spec.loader is not None
    protocol = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, protocol)
    spec.loader.exec_module(protocol)

    module = ModuleType("plumb_serving_cpu")
    module.__dict__.update(
        envs=SimpleNamespace(VLLM_SKIP_MODEL_NAME_VALIDATION=False),
        BaseServing=SimpleNamespace(_base_request_id=lambda raw: raw.headers["X-Request-Id"]),
        load_aware_call=lambda function: function,
        with_cancellation=lambda function: function,
        validate_json_request=lambda: None,
        VLLMNotFoundError=ValueError,
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final_only"),
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        contains_trace_headers=lambda headers: "traceparent" in headers,
        extract_trace_headers=lambda headers: {"traceparent": headers["traceparent"]},
        log_tracing_disabled_warning=lambda: None,
        get_hf_file_to_dict=lambda *args: {"temperature": 2.07},
        make_async=_make_async,
        merge_async_iterators=_merge_async_iterators,
        get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=False),
    )
    source = _SOURCE_DIR / "serving.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module == "protocol":
            module.__dict__.update({name.asname or name.name: getattr(protocol, name.name) for name in node.names})
    tree = _DependencyImports().visit(tree)
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.protocol = protocol
    return module


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if len(text) == 1 and text in "ABCDEFGHIJKLMNOP":
            return [32 + ord(text) - ord("A")]
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"


def _output(scores):
    return SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(
                finish_reason="length",
                token_ids=[999],  # The sampled token is irrelevant to the decision.
                logprobs=[{token: SimpleNamespace(logprob=score) for token, score in scores.items()}],
            )
        ],
    )


class _Engine:
    def __init__(self, executor, *, fail=False, block=False):
        self.calls = []
        self.completed = []
        self.cleaned = set()
        self.all_started = asyncio.Event()
        self.second_finished = asyncio.Event()
        self.release = asyncio.Event()
        self.clean_events = [asyncio.Event(), asyncio.Event()]
        self.fail = fail
        self.block = block
        self.model_config = SimpleNamespace(
            runner_type="generate",
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            hf_config=SimpleNamespace(_commit_hash="resolved-model-commit"),
            revision="branch-name",
            model="model-repository",
            served_model_name=["plumb-4b"],
            max_model_len=65537,
            max_logprobs=16,
            logprobs_mode="raw_logprobs",
        )
        self.vllm_config = SimpleNamespace(
            model_config=self.model_config,
            use_v2_model_runner=False,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align"),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        self.renderer = SimpleNamespace(get_tokenizer=_Tokenizer, _executor=executor)

    async def is_tracing_enabled(self):
        return True

    async def generate(self, **kwargs):
        index = int(kwargs["request_id"].rsplit("-", 1)[1])
        self.calls.append(kwargs)
        if len(self.calls) == 2:
            self.all_started.set()
        try:
            await self.all_started.wait()
            if self.fail and index == 0:
                raise RuntimeError("injected child failure")
            if self.block or self.fail:
                await self.release.wait()
            if index == 0:
                await self.second_finished.wait()
            else:
                self.second_finished.set()
            self.completed.append(index)
            scores = {32: -10.0, 33: -12.07} if index == 0 else {32: -22.07, 33: -20.0, 34: -24.14}
            yield _output({**scores, 999: -0.001})
        finally:
            self.cleaned.add(index)
            self.clean_events[index].set()

    async def wait_for_cleanup(self):
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in self.clean_events)), timeout=2)


def _request(serving):
    return serving.SystemOneRequest(
        state={"shared": "evidence " * 64},
        questions={
            "first": {"type": "noul", "instructions": "Is the claim true?"},
            "second": {"type": "choice", "instructions": "Pick.", "criteria": ["red", "blue", "green"]},
        },
        cache_salt="tenant-prefix",
        priority=3,
    )


def _raw_request():
    return SimpleNamespace(headers={"X-Request-Id": "request-id", "traceparent": "trace-id"})


def test_concurrent_native_requests_preserve_prefix_and_restore_answer_order(serving):
    async def run(executor):
        engine = _Engine(executor)
        service = serving.PlumbService(engine)
        request = _request(serving)
        plans, tokens = serving.plan_request(request, service.tokenizer, service.max_model_len)
        response = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert engine.completed == [1, 0]
        assert list(response["answers"]) == ["first", "second"]
        assert response["answers"]["first"]["noul"] == pytest.approx(1 / (1 + math.exp(-1)))
        assert response["answers"]["second"]["choice"] == "blue"
        assert response["usage"] == {"input_tokens": tokens, "output_tokens": 0}
        assert engine.cleaned == {0, 1}
        for index, call in enumerate(engine.calls):
            assert call["prompt"] == {"prompt_token_ids": plans[index].token_ids, "cache_salt": request.cache_salt}
            assert call["priority"] == 3
            assert call["trace_headers"] == {"traceparent": "trace-id"}
            assert call["request_id"] == f"plumb-request-id-{index}"
            params = call["sampling_params"]
            assert params.max_tokens == 1
            assert params.logprob_token_ids == plans[index].candidate_token_ids
            assert params.logprobs == len(plans[index].candidate_token_ids)
            assert params.temperature == 0
            assert params.ignore_eos and not params.detokenize
            assert params.output_kind == serving.RequestOutputKind.FINAL_ONLY
            assert not getattr(params, "skip_reading_prefix_cache", False)
        assert plans[0].token_ids[:256] == plans[1].token_ids[:256]

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_missing_candidate_is_rejected_instead_of_renormalized(serving):
    plan = SimpleNamespace(candidate_token_ids=[32, 33])
    with pytest.raises(RuntimeError, match="missing candidate"):
        serving.extract_probabilities(_output({32: -10.0, 999: -0.001}), plan, 2.07)


def test_nonfinite_engine_scores_are_server_errors(serving):
    plan = SimpleNamespace(candidate_token_ids=[32, 33])
    with pytest.raises(RuntimeError, match="invalid candidate scores"):
        serving.extract_probabilities(_output({32: float("nan"), 33: -1.0}), plan, 2.07)


@pytest.mark.parametrize("cancel", [False, True], ids=["child-failure", "parent-cancellation"])
def test_failure_or_cancellation_closes_sibling_generators(serving, cancel):
    async def run(executor):
        engine = _Engine(executor, fail=not cancel, block=cancel)
        service = serving.PlumbService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await asyncio.wait_for(engine.all_started.wait(), timeout=2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="injected child failure"):
                await asyncio.wait_for(task, timeout=2)
        await engine.wait_for_cleanup()
        assert engine.cleaned == {0, 1}

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_temperature_file_uses_loaded_model_revision(serving, monkeypatch):
    calls = []

    def get_config(*args):
        calls.append(args)
        return {"temperature": 2.07}

    monkeypatch.setattr(serving, "get_hf_file_to_dict", get_config)
    model = SimpleNamespace(
        hf_config=SimpleNamespace(_commit_hash="resolved-sha"), revision="moving-branch", model="repo"
    )
    assert serving.load_temperature(model) == 2.07
    assert calls == [("jevk5_config.json", "repo", "resolved-sha")]


@pytest.mark.parametrize("config", [None, {}, {"temperature": True}, {"temperature": 0}, {"temperature": float("nan")}])
def test_missing_or_invalid_calibration_fails_closed(serving, monkeypatch, config):
    monkeypatch.setattr(serving, "get_hf_file_to_dict", lambda *args: config)
    model = SimpleNamespace(hf_config=SimpleNamespace(_commit_hash=None), revision="pinned-sha", model="repo")
    with pytest.raises(ValueError, match="temperature"):
        serving.load_temperature(model)


@pytest.mark.parametrize(
    "criteria",
    [
        ["可以稍后处理", "本周处理", "今天处理"],
        ["可以稍后处理", {"deadline": "本周", "days": 7}, ["今天处理", {"refund": True}]],
    ],
    ids=["issue-1-text-rubric", "structured-rubric"],
)
def test_typesafe_sdk_decodes_all_answer_types(serving, criteria):
    # Exercise the real SDK request and response path without an NPU or network.
    # Run with typesafe-sdk==0.7.2; it is a client test dependency, not a server dependency.
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = serving.PlumbService(_Engine(executor))

        def handle(request):
            assert request.method == "POST" and request.url.path == "/v1/systemone"
            typed_request = serving.SystemOneRequest.model_validate_json(request.content)
            plans, tokens = serving.plan_request(typed_request, service.tokenizer, service.max_model_len)
            response = service._build_response(
                typed_request,
                plans,
                [[0.1, 0.8, 0.1], [0.812345, 0.187655], [0.2, 0.3, 0.5]],
                tokens,
                time.perf_counter(),
            )
            return httpx.Response(200, json=response, request=request)

        with sdk.TypeSafeClient(
            base_url="http://plumb.test",
            model="plumb-4b",
            api_key="local",
            retry=sdk.RetryPolicy(max_retries=0),
            transport=httpx.MockTransport(handle),
        ) as client:
            result = client.system_one(
                state="订单被重复扣款，客户要求今天处理退款。",
                questions={
                    "department": sdk.Choice(
                        instructions="这个问题应由哪个部门处理？",
                        criteria={"network": "网络运维", "billing": "支付与账务", "delivery": "物流服务"},
                    ),
                    "refund_requested": sdk.Noul(instructions="客户是否明确提出退款要求？"),
                    "urgency": sdk.Score(instructions="判断处理的紧急程度。", criteria=criteria),
                },
            )

    assert result.choices["department"].choice == "billing"
    assert result.nouls["refund_requested"].noul == 0.812345
    assert result.scores["urgency"].score == pytest.approx(1.3)
    assert result.scores["urgency"].confidence == 0.5
    assert result.scores["urgency"].legend == dict(enumerate(criteria))
    assert result.scores["urgency"].probabilities == {0: 0.2, 1: 0.3, 2: 0.5}
    raw = result.raw_http_response.json()
    assert raw["answers"]["urgency"]["legend"] == {str(index): level for index, level in enumerate(criteria)}
    assert result.usage.input_tokens == raw["usage"]["input_tokens"] > 0
    assert result.usage.output_tokens == 0
