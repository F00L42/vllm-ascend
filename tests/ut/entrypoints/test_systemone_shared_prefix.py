# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for conservative shared-prefix capability and token planning."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/entrypoints/systemone/shared_prefix.py"
_SPEC = importlib.util.spec_from_file_location("systemone_shared_prefix_cpu_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_SHARED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SHARED
_SPEC.loader.exec_module(_SHARED)


def runtime_config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            block_size=64,
            mamba_block_size=256,
            prefix_match_unit=32,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, async_scheduling=False),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        kv_transfer_config=None,
    )


def shared_config(**overrides):
    values = {"enabled": True, "verified_boundary_tokens": 256, "min_shared_tokens": 0}
    return SimpleNamespace(**(values | overrides))


def plans(*token_ids):
    return [SimpleNamespace(token_ids=list(tokens)) for tokens in token_ids]


def test_explicit_verified_boundary_must_cover_both_physical_sizes():
    config = runtime_config()
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config())
    assert result == _SHARED.SharedPrefixCapability(256, "ready")
    assert (
        _SHARED.inspect_shared_prefix_capability(config, shared_config(verified_boundary_tokens=512)).boundary_tokens
        == 512
    )
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config(verified_boundary_tokens=128))
    assert result == _SHARED.SharedPrefixCapability(None, "verified_boundary_misaligned")


def test_disabled_is_the_default_and_does_not_require_runtime_attributes():
    result = _SHARED.inspect_shared_prefix_capability(SimpleNamespace(), shared_config(enabled=False))
    assert result == _SHARED.SharedPrefixCapability(None, "disabled")


def test_matching_granularity_does_not_prove_a_saved_state():
    config = runtime_config()
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config(verified_boundary_tokens=None))
    assert result == _SHARED.SharedPrefixCapability(None, "verified_boundary_required")
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config(verified_boundary_tokens=32))
    assert result == _SHARED.SharedPrefixCapability(None, "verified_boundary_misaligned")


@pytest.mark.parametrize(
    ("section", "field", "value", "reason"),
    [
        ("cache_config", "enable_prefix_caching", False, "prefix_caching_disabled_or_unknown"),
        ("cache_config", "enable_prefix_caching", None, "prefix_caching_disabled_or_unknown"),
        ("cache_config", "mamba_cache_mode", "all", "mamba_cache_mode_unsupported_or_unknown"),
        ("cache_config", "mamba_cache_mode", None, "mamba_cache_mode_unsupported_or_unknown"),
        ("cache_config", "block_size", None, "physical_block_sizes_unknown"),
        ("cache_config", "block_size", True, "physical_block_sizes_unknown"),
        ("cache_config", "mamba_block_size", None, "physical_block_sizes_unknown"),
        ("cache_config", "mamba_block_size", 0, "physical_block_sizes_unknown"),
        ("cache_config", "mamba_block_size", 1024, "verified_boundary_misaligned"),
        ("scheduler_config", "enable_chunked_prefill", False, "chunked_prefill_disabled_or_unknown"),
        ("scheduler_config", "async_scheduling", True, "synchronous_scheduling_required"),
        ("scheduler_config", "async_scheduling", None, "synchronous_scheduling_required"),
        ("model_config", "enforce_eager", False, "eager_required"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_unsupported_or_unknown"),
        ("parallel_config", "data_parallel_size", None, "data_parallel_unsupported_or_unknown"),
        ("parallel_config", "data_parallel_size", True, "data_parallel_unsupported_or_unknown"),
        ("parallel_config", "data_parallel_external_lb", True, "external_routing_unsupported"),
        ("parallel_config", "data_parallel_hybrid_lb", True, "external_routing_unsupported"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context_parallel_unsupported"),
        ("parallel_config", "decode_context_parallel_size", 2, "context_parallel_unsupported"),
    ],
)
def test_unsupported_or_unknown_runtime_falls_back(section, field, value, reason):
    config = runtime_config()
    setattr(getattr(config, section), field, value)
    assert _SHARED.inspect_shared_prefix_capability(config, shared_config()) == _SHARED.SharedPrefixCapability(
        None, reason
    )


@pytest.mark.parametrize("missing", ["cache_config", "scheduler_config", "parallel_config", "model_config"])
def test_missing_runtime_sections_fail_closed(missing):
    config = runtime_config()
    delattr(config, missing)
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config())
    assert result.boundary_tokens is None


def test_kv_transfer_cannot_share_a_local_completion_barrier():
    config = runtime_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector="test")
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config())
    assert result == _SHARED.SharedPrefixCapability(None, "kv_transfer_unsupported")


@pytest.mark.parametrize("retention_interval", [0, 64, 1024])
def test_sparse_retention_is_rejected_from_both_configuration_sources(retention_interval):
    config = runtime_config()
    result = _SHARED.inspect_shared_prefix_capability(
        config, shared_config(), prefix_cache_retention_interval=retention_interval
    )
    assert result == _SHARED.SharedPrefixCapability(None, "sparse_cache_retention_unsupported")
    config.cache_config.prefix_cache_retention_interval = retention_interval
    result = _SHARED.inspect_shared_prefix_capability(config, shared_config())
    assert result == _SHARED.SharedPrefixCapability(None, "sparse_cache_retention_unsupported")


def test_explicit_none_retention_is_dense():
    config = runtime_config()
    config.cache_config.prefix_cache_retention_interval = None
    assert _SHARED.inspect_shared_prefix_capability(config, shared_config()).boundary_tokens == 256


@pytest.mark.parametrize(
    ("inputs", "boundary", "minimum", "expected"),
    [
        (([1, 2, 3, 4, 9], [1, 2, 3, 4, 8]), 2, 0, (4, 4, "ready")),
        (([1, 2, 3, 4, 9], [1, 2, 7, 4, 8]), 2, 0, (2, 2, "ready")),
        (([1, 2, 3, 4], [1, 2, 3, 4, 8]), 2, 0, (4, 2, "ready")),
        (([1, 2, 3, 4], [1, 2, 3, 4]), 2, 0, (4, 2, "ready")),
        (([1], [1, 2]), 1, 0, (1, 0, "below_verified_boundary")),
        (([1, 2, 9], [1, 2, 8]), 4, 0, (2, 0, "below_verified_boundary")),
        (([1, 2, 3, 4, 9], [1, 2, 3, 4, 8]), 2, 4, (4, 4, "ready")),
        (([1, 2, 3, 4, 9], [1, 2, 3, 4, 8]), 2, 5, (4, 0, "below_min_shared_tokens")),
        (([1, 2, 3, 9], [1, 2, 3, 8]), 2, 3, (3, 0, "below_min_shared_tokens")),
        (([1, 2], [3, 4]), 1, 0, (0, 0, "no_common_prefix")),
        (([], [1]), 1, 0, (0, 0, "no_common_prefix")),
    ],
)
def test_token_lcp_alignment_and_consumer_lookup_limit(inputs, boundary, minimum, expected):
    inputs_plans = plans(*inputs)
    result = _SHARED.plan_shared_prefix(inputs_plans, _SHARED.SharedPrefixCapability(boundary, "ready"), minimum)
    assert result == _SHARED.SharedPrefixPlan(*expected)
    assert [plan.token_ids for plan in inputs_plans] == list(inputs)
    if result.prefill_tokens:
        assert result.prefill_tokens % boundary == 0
        assert result.prefill_tokens <= min(len(tokens) - 1 for tokens in inputs)


def test_lcp_is_shared_by_every_question_including_the_last():
    result = _SHARED.plan_shared_prefix(
        plans([1, 2, 3, 4], [1, 2, 3, 9], [1, 7, 3, 9]), _SHARED.SharedPrefixCapability(1, "ready")
    )
    assert result == _SHARED.SharedPrefixPlan(1, 1, "ready")


@pytest.mark.parametrize("reason", ["disabled", "data_parallel_unsupported_or_unknown"])
def test_unavailable_capability_does_not_inspect_prompt_tokens(reason):
    class UnreadablePlan:
        @property
        def token_ids(self):
            raise AssertionError("An unavailable capability must bypass token inspection")

    result = _SHARED.plan_shared_prefix(
        [UnreadablePlan(), UnreadablePlan()], _SHARED.SharedPrefixCapability(None, reason)
    )
    assert result == _SHARED.SharedPrefixPlan(0, 0, reason)


def test_single_question_has_no_extra_producer():
    result = _SHARED.plan_shared_prefix(plans(range(1024)), _SHARED.SharedPrefixCapability(256, "ready"))
    assert result == _SHARED.SharedPrefixPlan(0, 0, "single_question")
    assert _SHARED.plan_shared_prefix([], _SHARED.SharedPrefixCapability(256, "ready")).reason == "no_questions"


@pytest.mark.parametrize("minimum", [-1, True, 1.0])
def test_planner_rejects_invalid_minimum(minimum):
    with pytest.raises(ValueError):
        _SHARED.plan_shared_prefix(plans([1, 2], [1, 3]), _SHARED.SharedPrefixCapability(1, "ready"), minimum)


def test_capability_and_plan_are_immutable():
    capability = _SHARED.SharedPrefixCapability(256, "ready")
    result = _SHARED.SharedPrefixPlan(512, 512, "ready")
    with pytest.raises(FrozenInstanceError):
        capability.boundary_tokens = 128
    with pytest.raises(FrozenInstanceError):
        result.prefill_tokens = 128
