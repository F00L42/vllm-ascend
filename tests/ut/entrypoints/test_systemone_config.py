# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for explicit per-instance SystemOne readout modes."""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/entrypoints/systemone/config.py"
_SPEC = importlib.util.spec_from_file_location("systemone_config_cpu_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_CONFIG = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONFIG
_SPEC.loader.exec_module(_CONFIG)


@pytest.mark.parametrize("additional", [None, {}, {"enable_reduce_sample": False}])
def test_missing_namespace_preserves_strict_plumb_mode(additional):
    config = _CONFIG.load_readout_config(SimpleNamespace(additional_config=additional))
    assert config.mode == "plumb"
    assert config.temperature is None
    assert config.shared_prefix.enabled is False
    assert config.shared_prefix.verified_boundary_tokens is None
    assert config.shared_prefix.min_shared_tokens == 0


def test_readout_is_nested_and_does_not_consume_ascend_options():
    config = _CONFIG.load_readout_config(
        SimpleNamespace(
            additional_config={"enable_reduce_sample": False, "systemone_config": {"mode": "answer_tokens"}}
        )
    )
    assert config.mode == "answer_tokens"
    assert config.temperature is None


@pytest.mark.parametrize("temperature", [True, "1.0", 0, -1, math.nan, math.inf])
def test_generic_mode_rejects_invalid_temperature(temperature):
    with pytest.raises(ValidationError):
        _CONFIG.ReadoutConfig(mode="answer_tokens", temperature=temperature)


def test_generic_temperature_is_finite_positive_and_configuration_is_immutable():
    config = _CONFIG.ReadoutConfig(mode="answer_tokens", temperature=0.5)
    assert config.temperature == 0.5
    with pytest.raises(ValidationError):
        config.temperature = 1.0


def test_plumb_calibration_cannot_be_overridden():
    with pytest.raises(ValidationError, match="cannot be overridden"):
        _CONFIG.ReadoutConfig(mode="plumb", temperature=1.0)


@pytest.mark.parametrize("raw", [None, {"mode": "answer_token"}, {"mode": "answer_tokens", "unknown_field": 2}])
def test_explicit_invalid_namespace_fails_instead_of_falling_back_to_plumb(raw):
    with pytest.raises(ValidationError):
        _CONFIG.load_readout_config(SimpleNamespace(additional_config={"systemone_config": raw}))


def test_shared_prefix_configuration_is_nested_explicit_and_immutable():
    config = _CONFIG.load_readout_config(
        SimpleNamespace(
            additional_config={
                "systemone_config": {
                    "mode": "answer_tokens",
                    "shared_prefix": {"enabled": True, "verified_boundary_tokens": 1024, "min_shared_tokens": 2048},
                }
            }
        )
    )
    assert config.shared_prefix.enabled is True
    assert config.shared_prefix.verified_boundary_tokens == 1024
    assert config.shared_prefix.min_shared_tokens == 2048
    with pytest.raises(ValidationError):
        config.shared_prefix.enabled = False
    with pytest.raises(ValidationError):
        config.shared_prefix = _CONFIG.SharedPrefixConfig()


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_shared_prefix_enabled_requires_a_boolean(enabled):
    with pytest.raises(ValidationError):
        _CONFIG.SharedPrefixConfig(enabled=enabled)


@pytest.mark.parametrize("boundary", [False, True, 0, -1, 16.0, "16"])
def test_verified_boundary_requires_a_positive_integer(boundary):
    with pytest.raises(ValidationError):
        _CONFIG.SharedPrefixConfig(verified_boundary_tokens=boundary)


@pytest.mark.parametrize("minimum", [False, True, -1, 0.0, "0", None])
def test_minimum_shared_tokens_requires_a_nonnegative_integer(minimum):
    with pytest.raises(ValidationError):
        _CONFIG.SharedPrefixConfig(min_shared_tokens=minimum)


@pytest.mark.parametrize("raw", [None, {"enable": True}, {"boundary_tokens": 64}])
def test_shared_prefix_unknown_or_invalid_fields_fail_closed(raw):
    with pytest.raises(ValidationError):
        _CONFIG.ReadoutConfig(shared_prefix=raw)
