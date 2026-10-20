# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-instance configuration for native answer-token readout."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SharedPrefixConfig(BaseModel):
    """Operator opt-in to an independently verified hybrid-cache boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False, strict=True)
    strategy: Literal["producer", "first_question"] = "producer"
    # This is an assertion about the deployed runtime, not its hash granularity.
    # Every producer ending at this interval must save all hybrid cache states.
    verified_boundary_tokens: int | None = Field(default=None, gt=0, strict=True)
    # Only producer scans a common prefix and applies this minimum.
    min_shared_tokens: int = Field(default=0, ge=0, strict=True)


class ReadoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Keep existing Plumb deployments strict unless the operator opts in.
    mode: Literal["plumb", "answer_tokens"] = "plumb"
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    shared_prefix: SharedPrefixConfig = Field(default_factory=SharedPrefixConfig)
    diagnostics: bool = Field(default=False, strict=True)

    @field_validator("temperature", mode="before")
    @classmethod
    def check_numeric_temperature(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("SystemOne readout temperature must be a finite positive number")
        return value

    @model_validator(mode="after")
    def check_plumb_calibration(self):
        if self.mode == "plumb" and self.temperature is not None:
            raise ValueError("Plumb temperature comes from jevk5_config.json and cannot be overridden")
        return self


def load_readout_config(vllm_config) -> ReadoutConfig:
    additional_config = vllm_config.additional_config
    raw = additional_config.get("systemone_config", {}) if additional_config is not None else {}
    return ReadoutConfig.model_validate(raw)
