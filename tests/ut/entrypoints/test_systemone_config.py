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


@pytest.mark.parametrize("raw", [None, {"mode": "answer_token"}, {"mode": "answer_tokens", "temperatur": 2}])
def test_explicit_invalid_namespace_fails_instead_of_falling_back_to_plumb(raw):
    with pytest.raises(ValidationError):
        _CONFIG.load_readout_config(SimpleNamespace(additional_config={"systemone_config": raw}))
