# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Conservative endpoint planning; the engine owns all cache lookup and state."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class TokenPlan(Protocol):
    token_ids: list[int]


@dataclass(frozen=True)
class SharedPrefixCapability:
    boundary_tokens: int | None
    reason: str


@dataclass(frozen=True)
class SharedPrefixPlan:
    # Zero also means unmeasured when capability is unavailable or N <= 1.
    common_tokens: int
    prefill_tokens: int
    reason: str


def _positive_integer(value) -> bool:
    return type(value) is int and value > 0


def inspect_shared_prefix_capability(
    vllm_config,
    shared_config,
    *,
    prefix_cache_retention_interval: int | None = None,
) -> SharedPrefixCapability:
    """Check visible runtime limits without inferring saved states from hashes.

    ``verified_boundary_tokens`` is the operator's assertion that producer
    completion saves complete hybrid state at *every* such boundary on this
    deployment. Physical sizes must already have been resolved by the engine.
    This check cannot verify that assertion or reserve cache capacity.

    vLLM 0.27.1 exposes retention through its central ``envs`` module; the
    caller supplies that resolved value. Newer versions may also expose the
    setting on ``cache_config``. Phase one accepts only dense retention (None).
    """
    if not shared_config.enabled:
        return SharedPrefixCapability(None, "disabled")

    cache = getattr(vllm_config, "cache_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    model = getattr(vllm_config, "model_config", None)
    if getattr(cache, "enable_prefix_caching", None) is not True:
        return SharedPrefixCapability(None, "prefix_caching_disabled_or_unknown")
    if type(getattr(parallel, "data_parallel_size", None)) is not int or parallel.data_parallel_size != 1:
        return SharedPrefixCapability(None, "data_parallel_unsupported_or_unknown")
    if any(
        getattr(parallel, field, False) is not False
        for field in ("data_parallel_external_lb", "data_parallel_hybrid_lb")
    ):
        return SharedPrefixCapability(None, "external_routing_unsupported")
    if any(
        type(getattr(parallel, field, 1)) is not int or getattr(parallel, field, 1) != 1
        for field in ("prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        return SharedPrefixCapability(None, "context_parallel_unsupported")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return SharedPrefixCapability(None, "kv_transfer_unsupported")
    if getattr(scheduler, "enable_chunked_prefill", None) is not True:
        return SharedPrefixCapability(None, "chunked_prefill_disabled_or_unknown")
    if getattr(cache, "mamba_cache_mode", None) != "align":
        return SharedPrefixCapability(None, "mamba_cache_mode_unsupported_or_unknown")
    if (
        prefix_cache_retention_interval is not None
        or getattr(cache, "prefix_cache_retention_interval", None) is not None
    ):
        return SharedPrefixCapability(None, "sparse_cache_retention_unsupported")
    if getattr(model, "enforce_eager", None) is not True:
        return SharedPrefixCapability(None, "eager_required")
    if getattr(scheduler, "async_scheduling", None) is not False:
        return SharedPrefixCapability(None, "synchronous_scheduling_required")

    boundary = shared_config.verified_boundary_tokens
    if not _positive_integer(boundary):
        return SharedPrefixCapability(None, "verified_boundary_required")
    physical_sizes = (getattr(cache, "block_size", None), getattr(cache, "mamba_block_size", None))
    if any(not _positive_integer(size) for size in physical_sizes):
        return SharedPrefixCapability(None, "physical_block_sizes_unknown")
    if any(boundary % size != 0 for size in physical_sizes):
        return SharedPrefixCapability(None, "verified_boundary_misaligned")
    return SharedPrefixCapability(boundary, "ready")


def plan_shared_prefix(
    plans: Sequence[TokenPlan],
    capability: SharedPrefixCapability,
    min_shared_tokens: int = 0,
) -> SharedPrefixPlan:
    """Find a verified producer endpoint while leaving each consumer a token.

    All comparisons use complete original prompt token IDs. Equal prompt
    strings or separately encoded state strings are not sufficient evidence.
    """
    if not plans:
        return SharedPrefixPlan(0, 0, "no_questions")
    if len(plans) == 1:
        return SharedPrefixPlan(0, 0, "single_question")
    if type(min_shared_tokens) is not int or min_shared_tokens < 0:
        raise ValueError("min_shared_tokens must be a nonnegative integer")
    boundary = capability.boundary_tokens
    if not _positive_integer(boundary):
        return SharedPrefixPlan(0, 0, capability.reason)

    shortest_length = min(len(plan.token_ids) for plan in plans)
    common_tokens = 0
    first_ids = plans[0].token_ids
    other_plans = plans[1:]
    for index in range(shortest_length):
        if any(plan.token_ids[index] != first_ids[index] for plan in other_plans):
            break
        common_tokens += 1

    if common_tokens == 0:
        return SharedPrefixPlan(0, 0, "no_common_prefix")
    prefill_tokens = min(common_tokens, shortest_length - 1) // boundary * boundary
    if prefill_tokens == 0:
        return SharedPrefixPlan(common_tokens, 0, "below_verified_boundary")
    if prefill_tokens < min_shared_tokens:
        return SharedPrefixPlan(common_tokens, 0, "below_min_shared_tokens")
    return SharedPrefixPlan(common_tokens, prefill_tokens, "ready")
