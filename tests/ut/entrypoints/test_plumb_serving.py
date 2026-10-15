# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU service contract tests; no vLLM installation or accelerator is required."""

import ast
import asyncio
import contextlib
import functools
import importlib.util
import json
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
        if (node.module or "").startswith("vllm") or (
            node.level and node.module in ("protocol", "config", "shared_prefix")
        ):
            return None
        return node


@pytest.fixture
def serving(monkeypatch):
    package = ModuleType("systemone_serving_cpu")
    package.__path__ = [str(_SOURCE_DIR)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    dependencies = {}
    for name in ("protocol", "config", "shared_prefix"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", _SOURCE_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        dependency = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, dependency)
        spec.loader.exec_module(dependency)
        dependencies[name] = dependency

    module = ModuleType("plumb_serving_cpu")
    module.__dict__.update(
        envs=SimpleNamespace(VLLM_SKIP_MODEL_NAME_VALIDATION=False, VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None),
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
        if isinstance(node, ast.ImportFrom) and node.level and node.module in dependencies:
            module.__dict__.update(
                {name.asname or name.name: getattr(dependencies[node.module], name.name) for name in node.names}
            )
    tree = _DependencyImports().visit(tree)
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.protocol = dependencies["protocol"]
    return module


class _Tokenizer:
    answer_token_start = 32

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if len(text) == 1 and text in "ABCDEFGHIJKLMNOP":
            return [self.answer_token_start + ord(text) - ord("A")]
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"


class _DifferentTokenizer(_Tokenizer):
    answer_token_start = 4000

    def apply_chat_template(self, messages, **kwargs):
        return "<new-model-template>" + super().apply_chat_template(messages, **kwargs)


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
    def __init__(self, executor, *, fail=False, block=False, generic=False):
        self.calls = []
        self.completed = []
        self.cleaned = set()
        self.all_started = asyncio.Event()
        self.second_finished = asyncio.Event()
        self.release = asyncio.Event()
        self.clean_events = [asyncio.Event(), asyncio.Event()]
        self.fail = fail
        self.block = block
        self.aborted = []
        self.remaining = set()
        self.abort_fails = False
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
            additional_config={"systemone_config": {"mode": "answer_tokens"}} if generic else {},
        )
        self.renderer = SimpleNamespace(
            get_tokenizer=_DifferentTokenizer if generic else _Tokenizer, _executor=executor
        )
        if generic:
            self.model_config.hf_config = SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                tie_word_embeddings=False,
                text_config=self.model_config.hf_text_config,
            )
            self.model_config.served_model_name = ["qwen3.6-27b-custom", "qwen-custom"]

    async def is_tracing_enabled(self):
        return True

    async def abort(self, request_id):
        self.aborted.append(request_id)
        if self.abort_fails:
            raise RuntimeError("injected public abort failure")
        self.remaining.discard(request_id)

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
            candidates = kwargs["sampling_params"].logprob_token_ids
            values = [-10.0, -12.07] if index == 0 else [-22.07, -20.0, -24.14]
            scores = dict(zip(candidates, values, strict=True))
            yield _output({**scores, 999: -0.001})
        finally:
            self.cleaned.add(index)
            self.clean_events[index].set()

    async def wait_for_cleanup(self):
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in self.clean_events)), timeout=2)


class _SharedEngine(_Engine):
    """Record generator creation as well as iteration and native cleanup."""

    def __init__(
        self,
        executor,
        *,
        generic=False,
        prefill_behavior="normal",
        prefill_block=False,
        prefill_cleanup_block=False,
        fail=False,
        block=False,
        cache_report="sufficient",
    ):
        super().__init__(executor, generic=generic, fail=fail, block=block)
        self.vllm_config.additional_config.setdefault("systemone_config", {})["shared_prefix"] = {
            "enabled": True,
            "verified_boundary_tokens": 64,
        }
        self.model_config.enforce_eager = True
        self.vllm_config.cache_config.block_size = 64
        self.vllm_config.cache_config.mamba_block_size = 64
        self.vllm_config.parallel_config = SimpleNamespace(data_parallel_size=1)
        self.vllm_config.scheduler_config.async_scheduling = False
        self.vllm_config.kv_transfer_config = None
        self.prefill_behavior = prefill_behavior
        self.prefill_block = prefill_block
        self.prefill_cleanup_block = prefill_cleanup_block
        self.cache_report = cache_report
        self.branch_count = 2
        self.branch_behavior = "normal"
        self.events = []
        self.prefill_started = asyncio.Event()
        self.prefill_cleanup_started = asyncio.Event()
        self.prefill_closed = asyncio.Event()
        self.prefill_release = asyncio.Event()
        self.prefill_cleanup_release = asyncio.Event()

    def generate(self, **kwargs):
        # A regular function deliberately detects eager branch construction,
        # even if its async generator has not yet been iterated.
        label = kwargs["request_id"].rsplit("-", 1)[1]
        self.calls.append(kwargs)
        self.events.append(("created", label))
        return self._produce(label, kwargs)

    async def _produce(self, label, call):
        self.events.append(("started", label))
        if label == "prefill":
            try:
                self.prefill_started.set()
                if self.prefill_block:
                    await self.prefill_release.wait()
                if self.prefill_behavior == "raise":
                    raise RuntimeError("injected prefill failure")
                if self.prefill_behavior == "empty":
                    return
                output = SimpleNamespace(
                    finished=self.prefill_behavior != "unfinished",
                    outputs=[
                        SimpleNamespace(
                            finish_reason=self.prefill_behavior
                            if self.prefill_behavior in ("abort", "error")
                            else "length",
                            token_ids=[] if self.prefill_behavior == "empty-token" else [999],
                        )
                    ],
                    num_cached_tokens=0,
                )
                yield output
                if self.prefill_behavior == "raise-after-finished":
                    raise RuntimeError("injected prefill failure after finished")
            finally:
                self.prefill_cleanup_started.set()
                if self.prefill_cleanup_block:
                    await self.prefill_cleanup_release.wait()
                self.events.append(("closed", label))
                self.prefill_closed.set()
            return

        index = int(label)
        branch_calls = [call for call in self.calls if not call["request_id"].endswith("-prefill")]
        if len(branch_calls) == self.branch_count:
            self.all_started.set()
        try:
            await self.all_started.wait()
            if self.fail and index == 0:
                raise RuntimeError("injected child failure")
            if self.block or self.fail:
                await self.release.wait()
            if index == 0 and self.branch_count > 1:
                await self.second_finished.wait()
            else:
                self.second_finished.set()
            self.completed.append(index)
            candidates = call["sampling_params"].logprob_token_ids
            values = [-10.0, -12.07] if index == 0 else [-22.07, -20.0, -24.14]
            output = _output(dict(zip(candidates, values[: len(candidates)], strict=True)))
            if index == 0 and self.branch_behavior == "unfinished":
                output.finished = False
            elif index == 0 and self.branch_behavior == "missing-candidate":
                output.outputs[0].logprobs[0].pop(candidates[0])
            if self.cache_report != "unavailable":
                output.num_cached_tokens = (
                    len(self.calls[0]["prompt"]["prompt_token_ids"]) if self.cache_report == "sufficient" else 0
                )
            yield output
        finally:
            self.events.append(("closed", label))
            self.cleaned.add(index)
            self.clean_events[index].set()


class _RegistrationEngine(_SharedEngine):
    """Reproduce native registration before add_request returns its collector."""

    def __init__(self, executor, *, phase, slow_abort=False):
        super().__init__(executor)
        self.phase = phase
        self.slow_abort = slow_abort
        self.registered_ids = []
        self.registration_changed = asyncio.Event()
        self.registration_release = asyncio.Event()
        self.native_leaked = set()
        self.abort_started = set()
        self.abort_interrupted = set()
        self.slow_abort_started = asyncio.Event()
        self.fast_abort_finished = asyncio.Event()
        self.abort_release = asyncio.Event()

    async def _produce(self, label, call):
        registering = label == "prefill" if self.phase == "prefill" else label != "prefill"
        if not registering:
            async with contextlib.aclosing(super()._produce(label, call)) as generator:
                async for output in generator:
                    yield output
            return

        request_id = call["request_id"]
        self.remaining.add(request_id)
        self.registered_ids.append(request_id)
        self.registration_changed.set()
        collector_ready = False
        try:
            await self.registration_release.wait()
            collector_ready = True
            async with contextlib.aclosing(super()._produce(label, call)) as generator:
                async for output in generator:
                    yield output
        finally:
            if collector_ready:
                self.remaining.discard(request_id)
            else:
                # Native close cannot abort while its local collector is None.
                # Only the endpoint's public abort can release this request.
                self.native_leaked.add(request_id)

    async def wait_for_registrations(self, count):
        async def wait():
            while len(self.registered_ids) < count:
                self.registration_changed.clear()
                if len(self.registered_ids) < count:
                    await self.registration_changed.wait()

        await asyncio.wait_for(wait(), timeout=2)

    async def abort(self, request_id):
        self.abort_started.add(request_id)
        try:
            if self.slow_abort and request_id.endswith("-0"):
                self.slow_abort_started.set()
                await self.abort_release.wait()
            await super().abort(request_id)
            if request_id.endswith("-1"):
                self.fast_abort_finished.set()
        except asyncio.CancelledError:
            self.abort_interrupted.add(request_id)
            raise


def _assert_group_request_ids(calls, suffixes):
    parents = {call["request_id"].rsplit("-", 1)[0] for call in calls}
    assert len(parents) == 1
    parent = parents.pop()
    assert parent.startswith("systemone-request-id-")
    unique = parent.removeprefix("systemone-request-id-")
    assert len(unique) == 32
    assert int(unique, 16) >= 0
    assert [call["request_id"] for call in calls] == [f"{parent}-{suffix}" for suffix in suffixes]
    return parent


def _shared_observations(caplog):
    prefix = "SystemOne shared_prefix "
    return [json.loads(record.message[len(prefix) :]) for record in caplog.records if record.message.startswith(prefix)]


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


@pytest.mark.parametrize("generic", [False, True], ids=["plumb", "answer-tokens"])
def test_concurrent_native_requests_preserve_prefix_and_restore_answer_order(serving, generic, monkeypatch):
    async def run(executor):
        engine = _Engine(executor, generic=generic)
        if generic:
            # A normal checkpoint does not need or read Plumb calibration metadata.
            def unexpected_calibration_read(*args):
                raise AssertionError("answer_tokens mode tried to load jevk5_config.json")

            monkeypatch.setattr(serving, "get_hf_file_to_dict", unexpected_calibration_read)
        service = serving.SystemOneService(engine)
        request = _request(serving)
        plans, tokens = serving.plan_request(request, service.tokenizer, service.max_model_len)
        response = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert engine.completed == [1, 0]
        assert list(response["answers"]) == ["first", "second"]
        temperature = 1.0 if generic else 2.07
        assert service.temperature == temperature
        assert response["answers"]["first"]["noul"] == pytest.approx(1 / (1 + math.exp(-2.07 / temperature)))
        assert response["answers"]["second"]["choice"] == "blue"
        assert response["usage"] == {"input_tokens": tokens, "output_tokens": 0}
        assert request.model is None
        assert response["model"] == ("qwen3.6-27b-custom" if generic else "plumb-4b")
        assert engine.cleaned == {0, 1}
        _assert_group_request_ids(engine.calls, ["0", "1"])
        assert not engine.aborted
        for index, call in enumerate(engine.calls):
            assert call["prompt"] == {"prompt_token_ids": plans[index].token_ids, "cache_salt": request.cache_salt}
            assert call["priority"] == 3
            assert call["trace_headers"] == {"traceparent": "trace-id"}
            params = call["sampling_params"]
            assert params.max_tokens == 1
            assert params.logprob_token_ids == plans[index].candidate_token_ids
            assert params.logprobs == len(plans[index].candidate_token_ids)
            assert params.temperature == 0
            assert params.ignore_eos and not params.detokenize
            assert params.output_kind == serving.RequestOutputKind.FINAL_ONLY
            assert not getattr(params, "skip_reading_prefix_cache", False)
        assert plans[0].token_ids[:256] == plans[1].token_ids[:256]
        assert plans[0].candidate_token_ids == ([4000, 4001] if generic else [32, 33])
        if generic:
            assert "".join(map(chr, plans[0].token_ids)).startswith("<new-model-template>")

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize("generic", [False, True], ids=["plumb", "answer-tokens"])
@pytest.mark.parametrize("cache_salt", [None, "tenant-prefix"])
def test_shared_prefill_preserves_native_prompts_readout_and_usage(serving, generic, cache_salt, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor, generic=generic)
        service = serving.SystemOneService(engine)
        request = _request(serving)
        request.cache_salt = cache_salt
        plans, input_tokens = serving.plan_request(request, service.tokenizer, service.max_model_len)
        shared = serving.plan_shared_prefix(plans, service.shared_prefix_capability, 0)
        assert shared.reason == "ready"
        assert 0 < shared.prefill_tokens <= shared.common_tokens
        assert shared.prefill_tokens % 64 == 0
        assert shared.prefill_tokens < min(len(plan.token_ids) for plan in plans)

        result = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert len(engine.calls) == 3
        assert engine.events.index(("closed", "prefill")) < engine.events.index(("created", "0"))
        assert engine.events.index(("closed", "prefill")) < engine.events.index(("created", "1"))
        prefill, *branches = engine.calls
        expected_prompt = {"prompt_token_ids": plans[0].token_ids[: shared.prefill_tokens]}
        if cache_salt is not None:
            expected_prompt["cache_salt"] = cache_salt
        assert prefill["prompt"] == expected_prompt
        parent = _assert_group_request_ids(engine.calls, ["prefill", "0", "1"])
        assert not engine.aborted
        params = prefill["sampling_params"]
        assert params.temperature == 0 and params.max_tokens == 1
        assert params.ignore_eos and not params.detokenize
        assert params.output_kind == serving.RequestOutputKind.FINAL_ONLY
        assert getattr(params, "logprobs", None) is None
        assert getattr(params, "logprob_token_ids", None) is None
        for index, branch in enumerate(branches):
            expected_prompt["prompt_token_ids"] = plans[index].token_ids
            assert branch["prompt"] == expected_prompt
            assert branch["sampling_params"].logprob_token_ids == plans[index].candidate_token_ids
        for call in engine.calls:
            assert call["priority"] == request.priority
            assert call["trace_headers"] == {"traceparent": "trace-id"}

        assert engine.completed == [1, 0]
        assert engine.cleaned == {0, 1}
        assert list(result["answers"]) == ["first", "second"]
        assert result["answers"]["second"]["choice"] == "blue"
        assert result["answers"]["first"]["noul"] == pytest.approx(1 / (1 + math.exp(-2.07 / service.temperature)))
        assert result["usage"] == {"input_tokens": input_tokens, "output_tokens": 0}
        observation = _shared_observations(caplog)[-1]
        assert observation["request_id"] == parent
        assert observation["parent_request_id"] == "request-id"
        assert observation["reason"] == "ready"
        assert observation["prefill_tokens"] == shared.prefill_tokens
        assert observation["common_tokens"] == shared.common_tokens
        assert observation["prefill_cached_tokens"] == 0
        assert observation["branch_cached_tokens"] == [shared.prefill_tokens] * 2
        assert observation["reuse"] == "reported_sufficient"
        assert observation["status"] == "completed"
        assert all(observation[key] >= 0 for key in ("planning_ms", "prefill_ms", "branch_ms"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_shared_prefill_must_finish_generator_cleanup_before_creating_branches(serving):
    async def run(executor):
        engine = _SharedEngine(executor, prefill_cleanup_block=True)
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await asyncio.wait_for(engine.prefill_cleanup_started.wait(), timeout=2)
        assert len(engine.calls) == 1
        assert not engine.prefill_closed.is_set()
        engine.prefill_cleanup_release.set()
        await asyncio.wait_for(task, timeout=2)
        assert engine.prefill_closed.is_set()
        assert len(engine.calls) == 3

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize(
    "behavior", ["raise", "empty", "unfinished", "error", "abort", "empty-token", "raise-after-finished"]
)
def test_shared_prefill_failure_or_missing_completion_never_starts_branches(serving, behavior, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor, prefill_behavior=behavior)
        service = serving.SystemOneService(engine)
        with pytest.raises(RuntimeError, match="prefill"):
            await asyncio.wait_for(service.evaluate(_request(serving), _raw_request()), timeout=2)
        assert len(engine.calls) == 1
        assert engine.prefill_closed.is_set()
        assert not engine.cleaned
        assert engine.aborted == [engine.calls[0]["request_id"]]
        observation = _shared_observations(caplog)[-1]
        assert observation["status"] == "failed"
        assert observation["branch_ms"] is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_parent_cancellation_closes_active_prefill_without_starting_branches(serving, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor, prefill_block=True)
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await asyncio.wait_for(engine.prefill_started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(engine.calls) == 1
        assert engine.prefill_closed.is_set()
        assert not engine.cleaned
        assert engine.aborted == [engine.calls[0]["request_id"]]
        assert _shared_observations(caplog)[-1]["status"] == "cancelled"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize("phase", ["prefill", "branch"])
def test_registration_window_cancellation_uses_public_abort_to_release_requests(serving, phase):
    async def run(executor):
        engine = _RegistrationEngine(executor, phase=phase)
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await engine.wait_for_registrations(1 if phase == "prefill" else 2)
        registered = set(engine.registered_ids)
        assert engine.remaining == registered
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert engine.native_leaked == registered
        assert set(engine.aborted) == registered
        assert not engine.remaining
        if phase == "prefill":
            assert len(engine.calls) == 1
        else:
            assert len(engine.calls) == 3
            assert engine.prefill_closed.is_set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_repeated_client_request_ids_keep_concurrent_parent_abort_isolated(serving, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _RegistrationEngine(executor, phase="prefill")
        service = serving.SystemOneService(engine)
        first = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await engine.wait_for_registrations(1)
        second = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await engine.wait_for_registrations(2)
        first_id, second_id = engine.registered_ids
        assert first_id != second_id
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert engine.aborted == [first_id]
        assert engine.remaining == {second_id}
        assert not second.done()

        engine.registration_release.set()
        result = await asyncio.wait_for(second, timeout=2)
        assert result["answers"]["second"]["choice"] == "blue"
        assert not engine.remaining
        assert engine.aborted == [first_id]
        observations = _shared_observations(caplog)
        assert len(observations) == 2
        assert {item["parent_request_id"] for item in observations} == {"request-id"}
        assert len({item["request_id"] for item in observations}) == 2
        assert {item["status"] for item in observations} == {"cancelled", "completed"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_parent_cancellation_waits_for_slow_abort_without_interrupting_cleanup(serving):
    async def run(executor):
        unexpected_loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unexpected_loop_errors.append(context))
        engine = _RegistrationEngine(executor, phase="branch", slow_abort=True)
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await engine.wait_for_registrations(2)
        task.cancel()
        await asyncio.wait_for(engine.slow_abort_started.wait(), timeout=2)
        await asyncio.wait_for(engine.fast_abort_finished.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.02)
        assert not task.done()
        assert not engine.abort_interrupted
        assert len(engine.remaining) == 1
        engine.abort_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert not engine.remaining
        assert not engine.abort_interrupted
        assert set(engine.aborted) == set(engine.registered_ids)
        await asyncio.sleep(0)
        assert not unexpected_loop_errors

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize("cancel", [False, True], ids=["stream-failure", "cancellation"])
def test_public_abort_failure_is_logged_without_replacing_original_stream_failure(serving, caplog, cancel):
    async def run(executor):
        engine = _SharedEngine(executor, prefill_behavior="normal" if cancel else "raise", prefill_block=cancel)
        engine.abort_fails = True
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        if cancel:
            await asyncio.wait_for(engine.prefill_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            with pytest.raises(RuntimeError, match="injected prefill failure"):
                await asyncio.wait_for(task, timeout=2)
        assert engine.aborted == [engine.calls[0]["request_id"]]
        assert engine.prefill_closed.is_set()
        errors = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(errors) == 1
        assert "could not abort unfinished request" in errors[0].message
        assert errors[0].exc_info is not None
        assert str(errors[0].exc_info[1]) == "injected public abort failure"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize(
    ("behavior", "expected_error"),
    [("unfinished", "results for all questions"), ("missing-candidate", "missing candidate")],
)
def test_missing_branch_completion_or_candidate_scores_aborts_its_native_request(serving, behavior, expected_error):
    async def run(executor):
        engine = _SharedEngine(executor)
        engine.branch_behavior = behavior
        service = serving.SystemOneService(engine)
        with pytest.raises(RuntimeError, match=expected_error):
            await asyncio.wait_for(service.evaluate(_request(serving), _raw_request()), timeout=2)
        assert engine.aborted == [engine.calls[1]["request_id"]]
        assert engine.cleaned == {0, 1}
        assert engine.prefill_closed.is_set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize("cancel", [False, True], ids=["child-failure", "parent-cancellation"])
def test_shared_branches_are_all_closed_before_failure_or_cancellation_returns(serving, cancel):
    async def run(executor):
        engine = _SharedEngine(executor, fail=not cancel, block=cancel)
        service = serving.SystemOneService(engine)
        task = asyncio.create_task(service.evaluate(_request(serving), _raw_request()))
        await asyncio.wait_for(engine.all_started.wait(), timeout=2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="injected child failure"):
                await asyncio.wait_for(task, timeout=2)
        assert engine.prefill_closed.is_set()
        assert engine.cleaned == {0, 1}
        assert len(engine.calls) == 3

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize(
    ("cache_report", "expected_reuse", "expected_counts"),
    [("insufficient", "insufficient", [0, 0]), ("unavailable", "unavailable", [None, None])],
)
def test_shared_cache_miss_or_missing_metrics_returns_answers_without_retry(
    serving, cache_report, expected_reuse, expected_counts, caplog
):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor, cache_report=cache_report)
        service = serving.SystemOneService(engine)
        result = await asyncio.wait_for(service.evaluate(_request(serving), _raw_request()), timeout=2)
        assert result["answers"]["second"]["choice"] == "blue"
        assert len(engine.calls) == 3
        observation = _shared_observations(caplog)[-1]
        assert observation["reuse"] == expected_reuse
        assert observation["branch_cached_tokens"] == expected_counts
        assert observation["status"] == "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize(
    ("fallback", "expected_reason"),
    [
        ("disabled", "disabled"),
        ("dp", "data_parallel_unsupported_or_unknown"),
        ("short", "below_min_shared_tokens"),
        ("retention", "sparse_cache_retention_unsupported"),
    ],
)
def test_shared_prefill_fallback_executes_original_branches(serving, fallback, expected_reason, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor)
        settings = engine.vllm_config.additional_config["systemone_config"]["shared_prefix"]
        if fallback == "disabled":
            settings["enabled"] = False
        elif fallback == "dp":
            engine.vllm_config.parallel_config.data_parallel_size = 2
        elif fallback == "retention":
            serving.envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 2
        else:
            settings["min_shared_tokens"] = 65536
        service = serving.SystemOneService(engine)
        request = _request(serving)
        plans, _ = serving.plan_request(request, service.tokenizer, service.max_model_len)
        result = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert result["answers"]["second"]["choice"] == "blue"
        assert len(engine.calls) == 2
        assert not engine.prefill_started.is_set()
        assert [call["prompt"]["prompt_token_ids"] for call in engine.calls] == [plan.token_ids for plan in plans]
        observation = _shared_observations(caplog)[-1]
        assert observation["reason"] == expected_reason
        assert observation["prefill_tokens"] == 0
        assert observation["prefill_ms"] is None
        assert observation["reuse"] == "not_attempted"
        assert observation["status"] == "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_shared_prefill_single_question_uses_only_its_original_prompt(serving, caplog):
    caplog.set_level("INFO", logger="plumb_serving_cpu")

    async def run(executor):
        engine = _SharedEngine(executor)
        engine.branch_count = 1
        service = serving.SystemOneService(engine)
        request = _request(serving)
        request.questions.pop("second")
        plans, input_tokens = serving.plan_request(request, service.tokenizer, service.max_model_len)
        result = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert list(result["answers"]) == ["first"]
        assert result["usage"] == {"input_tokens": input_tokens, "output_tokens": 0}
        assert len(engine.calls) == 1
        assert engine.calls[0]["prompt"]["prompt_token_ids"] == plans[0].token_ids
        assert not engine.prefill_started.is_set()
        assert engine.cleaned == {0}
        assert _shared_observations(caplog)[-1]["reason"] == "single_question"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_identical_questions_leave_one_uncached_input_token_and_keep_answer_order(serving):
    async def run(executor):
        engine = _SharedEngine(executor)
        service = serving.SystemOneService(engine)
        question = {"type": "noul", "instructions": "Is the claim true?"}
        request = serving.SystemOneRequest(state="evidence " * 64, questions={"first": question, "second": question})
        plans, _ = serving.plan_request(request, service.tokenizer, service.max_model_len)
        # Force a block-aligned full prompt to expose off-by-one boundary reuse.
        request.state += "x" * (-len(plans[0].token_ids) % 64)
        plans, _ = serving.plan_request(request, service.tokenizer, service.max_model_len)
        assert plans[0].token_ids == plans[1].token_ids
        assert len(plans[0].token_ids) % 64 == 0
        result = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert len(engine.calls[0]["prompt"]["prompt_token_ids"]) == len(plans[0].token_ids) - 64
        assert engine.completed == [1, 0]
        assert list(result["answers"]) == ["first", "second"]
        assert all(call["prompt"]["prompt_token_ids"] == plans[0].token_ids for call in engine.calls[1:])

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_generic_readout_honors_explicit_temperature(serving):
    async def run(executor):
        engine = _Engine(executor, generic=True)
        engine.vllm_config.additional_config["systemone_config"]["temperature"] = 2.0
        service = serving.SystemOneService(engine)
        result = await asyncio.wait_for(service.evaluate(_request(serving), _raw_request()), timeout=2)
        assert result["answers"]["first"]["noul"] == pytest.approx(1 / (1 + math.exp(-2.07 / 2.0)))

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_explicit_alias_is_accepted_and_unknown_model_never_reaches_engine(serving):
    async def run(executor):
        engine = _Engine(executor, generic=True)
        service = serving.SystemOneService(engine)
        request = _request(serving)
        request.model = "other-model"
        with pytest.raises(serving.VLLMNotFoundError, match="Unknown served model"):
            await service.evaluate(request, _raw_request())
        assert not engine.calls
        request.model = "qwen-custom"
        result = await asyncio.wait_for(service.evaluate(request, _raw_request()), timeout=2)
        assert result["model"] == "qwen-custom"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize(
    ("names", "expected"),
    [(["first", "second"], "first"), ("single", "single"), ([], "model-repository"), (None, "model-repository")],
)
def test_omitted_request_model_uses_instance_default_alias(serving, names, expected):
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine = _Engine(executor, generic=True)
        engine.model_config.served_model_name = names
        service = serving.SystemOneService(engine)
        request = _request(serving)
        plans, tokens = serving.plan_request(request, service.tokenizer, service.max_model_len)
        result = service._build_response(request, plans, [[0.7, 0.3], [0.2, 0.6, 0.2]], tokens, time.perf_counter())
        assert result["model"] == expected


def test_default_plumb_mode_still_requires_checkpoint_calibration(serving, monkeypatch):
    monkeypatch.setattr(serving, "get_hf_file_to_dict", lambda *args: None)
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(ValueError, match="jevk5_config.json"):
        serving.SystemOneService(_Engine(executor))


@pytest.mark.parametrize("generic", [False, True], ids=["plumb", "answer-tokens"])
def test_entrypoint_initializes_instance_service_and_mounts_shared_endpoint(serving, generic):
    async def run(executor):
        state = SimpleNamespace()
        plugin = serving.SystemOnePlugin()
        engine = _Engine(executor, generic=generic)
        await plugin.init_state(engine, state, None)
        app = serving.FastAPI()
        app.state.systemone_service = state.systemone_service
        plugin.attach_router(app)
        route = next(route for route in app.routes if route.path == "/v1/systemone")
        assert route.methods == {"POST"}
        raw = _raw_request()
        raw.app = app
        response = await asyncio.wait_for(route.endpoint(_request(serving), raw), timeout=2)
        assert response.status_code == 200
        result = json.loads(response.body)
        assert result["model"] == ("qwen3.6-27b-custom" if generic else "plumb-4b")
        assert result["answers"]["second"]["choice"] == "blue"

    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(run(executor))


def test_endpoint_without_an_engine_returns_service_unavailable(serving):
    async def run():
        plugin = serving.SystemOnePlugin()
        app = serving.FastAPI()
        await plugin.init_state(None, app.state, None)
        plugin.attach_router(app)
        route = next(route for route in app.routes if route.path == "/v1/systemone")
        raw = _raw_request()
        raw.app = app
        with pytest.raises(serving.HTTPException) as raised:
            await route.endpoint(_request(serving), raw)
        assert raised.value.status_code == 503

    asyncio.run(run())


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
        service = serving.SystemOneService(engine)
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
@pytest.mark.parametrize("generic", [False, True], ids=["plumb", "answer-tokens"])
def test_typesafe_sdk_decodes_all_answer_types(serving, criteria, generic):
    # Exercise the real SDK request and response path without an NPU or network.
    # Run with typesafe-sdk==0.7.2; it is a client test dependency, not a server dependency.
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = serving.SystemOneService(_Engine(executor, generic=generic))
        model = "qwen3.6-27b-custom" if generic else "plumb-4b"

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
            model=model,
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
    assert raw["model"] == model
    assert raw["answers"]["urgency"]["legend"] == {str(index): level for index, level in enumerate(criteria)}
    assert result.usage.input_tokens == raw["usage"]["input_tokens"] > 0
    assert result.usage.output_tokens == 0
