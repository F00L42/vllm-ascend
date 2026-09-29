# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

# Adapted from allebee/jevk5 (Apache-2.0), runtime.py and server.py at
# 85238d7be5527370c43206fe54cd752eb3134c1b. The prompt follows SemIf
# (TheoLeeCJ/SemIf, MIT), as attributed by JevK5. Changes add request validation,
# token-only vLLM request planning, and CPU readout from candidate logprobs.
"""Plumb's prompt and decision semantics for the TypeSafe /v1/systemone API."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, JsonValue, model_validator

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LETTERS)
INTERNAL_OUTPUT_TOKENS = 1
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] | list[str]

    @model_validator(mode="after")
    def check_options(self):
        # dict.fromkeys matches the reference's first-occurrence ordering.
        if isinstance(self.criteria, list):
            self.criteria = dict.fromkeys(self.criteria)
        if not 2 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"choice criteria must name 2..{MAX_OPTIONS} distinct options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JsonValue
    criteria: list[JsonValue] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1)
    model: str = "plumb-4b"
    priority: int = 0
    cache_salt: str | None = None


class PlumbTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


@dataclass(frozen=True)
class PlumbQuestionPlan:
    question_id: str
    kind: Literal["noul", "choice", "score"]
    option_ids: tuple[str, ...]
    token_ids: list[int]
    candidate_token_ids: list[int]
    legend: dict[str, JsonValue] | None = None


def decision_options(question: Noul | Choice | Score) -> list[tuple[str, str]]:
    """Return reference-compatible option IDs and descriptions, in model order."""
    if question.type == "noul":
        criteria = question.criteria or {}
        pairs = [(key, criteria.get(key) or f"The proposition is {key}.") for key in ("true", "false")]
    elif question.type == "choice":
        # Choice validation has already normalized list criteria to a dict.
        assert isinstance(question.criteria, dict)
        pairs = [(key, description or key) for key, description in question.criteria.items()]
    else:
        pairs = [(str(index), level) for index, level in enumerate(question.criteria)]
    return [(key, f"{key}: {description}") for key, description in pairs]


def messages(state: JsonValue, criterion: JsonValue, options: list[str]) -> list[dict[str, str]]:
    """Keep evidence first so native prefix caching can share the common state."""
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[index], "description": text} for index, text in enumerate(options)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]


def get_answer_token_ids(tokenizer: PlumbTokenizer) -> list[int]:
    slots = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(token_ids) != 1 for token_ids in slots):
        raise ValueError("Every Plumb answer letter A..P must encode as one token")
    token_ids = [slot[0] for slot in slots]
    if len(set(token_ids)) != MAX_OPTIONS:
        raise ValueError("Plumb answer letters A..P must have distinct token IDs")
    return token_ids


def plan_request(
    request: SystemOneRequest, tokenizer: PlumbTokenizer, max_model_len: int
) -> tuple[list[PlumbQuestionPlan], int]:
    """Encode every question without truncation; reserve one native output token."""
    answer_token_ids = get_answer_token_ids(tokenizer)
    plans = []
    input_tokens = 0
    for question_id, question in request.questions.items():
        options = decision_options(question)
        prompt = tokenizer.apply_chat_template(
            messages(request.state, question.instructions, [text for _, text in options]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Question {question_id!r} produced an empty prompt")
        if len(token_ids) + INTERNAL_OUTPUT_TOKENS > max_model_len:
            raise ValueError(
                f"Question {question_id!r} has {len(token_ids)} input tokens; "
                f"max_model_len={max_model_len} must also reserve {INTERNAL_OUTPUT_TOKENS} output token"
            )
        plans.append(
            PlumbQuestionPlan(
                question_id=question_id,
                kind=question.type,
                option_ids=tuple(key for key, _ in options),
                token_ids=token_ids,
                candidate_token_ids=answer_token_ids[: len(options)],
                # TypeSafe requires score rubrics separately from the model's
                # numbered option text. Keep SDK-supported structured criteria.
                legend={
                    str(index): level if isinstance(level, (str, dict, list)) else str(level)
                    for index, level in enumerate(question.criteria)
                }
                if question.type == "score"
                else None,
            )
        )
        input_tokens += len(token_ids)
    return plans, input_tokens


def probabilities(logprobs: Sequence[float], temperature: float) -> list[float]:
    """Calibrate only the candidate scores; the full-vocabulary offset cancels."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Plumb calibration temperature must be positive and finite")
    if not 2 <= len(logprobs) <= MAX_OPTIONS or any(not math.isfinite(value) for value in logprobs):
        raise ValueError(f"Expected 2..{MAX_OPTIONS} finite candidate logprobs")
    # Subtract before dividing, so a small valid temperature cannot overflow.
    maximum = max(logprobs)
    weights = [math.exp((value - maximum) / temperature) for value in logprobs]
    total = sum(weights)
    return [weight / total for weight in weights]


def build_answer(plan: PlumbQuestionPlan, values: Sequence[float]) -> dict[str, Any]:
    distribution = dict(zip(plan.option_ids, values, strict=True))
    answer: dict[str, Any] = {"type": plan.kind, "confidence": max(values)}
    if plan.kind == "noul":
        answer["noul"] = distribution["true"]
    elif plan.kind == "choice":
        # Ties retain the first option, matching Python's max in the reference.
        answer.update(choice=max(distribution, key=distribution.__getitem__), probabilities=distribution)
    else:
        answer.update(
            score=sum(int(key) * value for key, value in distribution.items()),
            probabilities=distribution,
            legend=plan.legend,
        )
    return answer


def to_answers(probs: list[list[float]], plans: list[PlumbQuestionPlan]) -> dict[str, Any]:
    return {plan.question_id: build_answer(plan, values) for values, plan in zip(probs, plans, strict=True)}
