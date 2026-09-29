# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

# The package initializer requires vLLM. Load this CPU-only module directly so
# these protocol tests can run before installing an NPU serving environment.
_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/entrypoints/systemone/protocol.py"
_SPEC = importlib.util.spec_from_file_location("plumb_protocol_cpu_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_PROTOCOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROTOCOL
_SPEC.loader.exec_module(_PROTOCOL)

LETTERS = _PROTOCOL.LETTERS
SYSTEM = _PROTOCOL.SYSTEM
SystemOneRequest = _PROTOCOL.SystemOneRequest
decision_options = _PROTOCOL.decision_options
get_answer_token_ids = _PROTOCOL.get_answer_token_ids
plan_request = _PROTOCOL.plan_request
probabilities = _PROTOCOL.probabilities
to_answers = _PROTOCOL.to_answers


class FakeTokenizer:
    def __init__(self):
        self.conversations = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in LETTERS and len(text) == 1:
            return [32 + LETTERS.index(text)]
        return [ord(char) for char in text]

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.conversations.append(conversation)
        return "<system>" + conversation[0]["content"] + "<user>" + conversation[1]["content"] + "<assistant>"


class TestPlumbProtocol(unittest.TestCase):
    def request(self, questions, state="evidence"):
        return SystemOneRequest(state=state, questions=questions)

    def test_reference_prompt_keeps_json_and_option_order(self):
        tokenizer = FakeTokenizer()
        request = self.request({"yes": {"type": "noul", "instructions": "Refund?"}}, {"text": "退款", "flag": True})
        plans, tokens = plan_request(request, tokenizer, 4096)
        self.assertEqual(tokenizer.conversations[0][0], {"role": "system", "content": SYSTEM})
        self.assertEqual(
            tokenizer.conversations[0][1]["content"],
            '{"evidence": {"text": "退款", "flag": true}, "criterion": "Refund?", '
            '"options": [{"letter": "A", "description": "true: The proposition is true."}, '
            '{"letter": "B", "description": "false: The proposition is false."}]}',
        )
        self.assertEqual(plans[0].option_ids, ("true", "false"))
        self.assertEqual(plans[0].candidate_token_ids, [32, 33])
        self.assertEqual(tokens, len(plans[0].token_ids))
        self.assertEqual(request.model, "plumb-4b")

    def test_choice_list_deduplicates_without_sorting(self):
        request = self.request({"q": {"type": "choice", "instructions": "pick", "criteria": ["z", "a", "z"]}})
        self.assertEqual(decision_options(request.questions["q"]), [("z", "z: z"), ("a", "a: a")])

    def test_custom_noul_choice_fallback_and_score_prefixes(self):
        request = self.request(
            {
                "n": {"type": "noul", "instructions": "n", "criteria": {"true": "correct", "false": ""}},
                "c": {"type": "choice", "instructions": "c", "criteria": {"z": "zero", "a": None}},
                "s": {"type": "score", "instructions": "s", "criteria": ["low", "high"]},
            }
        )
        self.assertEqual(
            decision_options(request.questions["n"]),
            [("true", "true: correct"), ("false", "false: The proposition is false.")],
        )
        self.assertEqual(decision_options(request.questions["c"]), [("z", "z: zero"), ("a", "a: a")])
        self.assertEqual(decision_options(request.questions["s"]), [("0", "0: low"), ("1", "1: high")])

    def test_prefix_shared_state_precedes_question(self):
        tokenizer = FakeTokenizer()
        request = self.request(
            {key: {"type": "noul", "instructions": question} for key, question in [("a", "Refund?"), ("b", "Cancel?")]},
            {"shared": "long evidence"},
        )
        plans, total = plan_request(request, tokenizer, 4096)
        first, second = [conversation[1]["content"] for conversation in tokenizer.conversations]
        prefix = '{"evidence": {"shared": "long evidence"}, "criterion": "'
        self.assertTrue(first.startswith(prefix) and second.startswith(prefix))
        self.assertEqual(total, sum(len(plan.token_ids) for plan in plans))

    def test_model_length_reserves_native_output_without_truncation(self):
        request = self.request({"q": {"type": "noul", "instructions": "n"}})
        plans, _ = plan_request(request, FakeTokenizer(), 4096)
        token_count = len(plans[0].token_ids)
        accepted, _ = plan_request(request, FakeTokenizer(), token_count + 1)
        self.assertEqual(accepted[0].token_ids, plans[0].token_ids)
        with self.assertRaisesRegex(ValueError, "must also reserve 1 output token"):
            plan_request(request, FakeTokenizer(), token_count)

    def test_invalid_question_counts_and_discriminators(self):
        invalid = [
            {},
            {"q": {"type": "other", "instructions": "x"}},
            {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},
            {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(17)}}},
            {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}},
        ]
        for questions in invalid:
            with self.subTest(questions=questions), self.assertRaises(ValidationError):
                self.request(questions)

    def test_all_letters_must_be_single_distinct_tokens(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(get_answer_token_ids(tokenizer), list(range(32, 48)))
        tokenizer.encode = lambda *args, **kwargs: [1, 2]
        with self.assertRaisesRegex(ValueError, "one token"):
            get_answer_token_ids(tokenizer)
        tokenizer.encode = lambda *args, **kwargs: [1]
        with self.assertRaisesRegex(ValueError, "distinct"):
            get_answer_token_ids(tokenizer)

    def test_calibration_uses_only_candidates_and_cancels_vocab_offset(self):
        reference = probabilities([1.0, 2.0, -1.0], 2.07)
        shifted = probabilities([-100.0, -99.0, -102.0], 2.07)
        for actual, expected in zip(shifted, reference, strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(reference), 1.0)
        self.assertAlmostEqual(reference[1] / reference[0], math.exp(1.0 / 2.07))
        for temperature in [0.0, -1.0, math.nan, math.inf]:
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                probabilities([1.0, 2.0], temperature)
        with self.assertRaises(ValueError):
            probabilities([0.0, math.nan], 2.07)

    def test_reference_answer_shapes_do_not_round_or_add_kev_fields(self):
        request = self.request(
            {
                "n": {"type": "noul", "instructions": "n"},
                "c": {"type": "choice", "instructions": "c", "criteria": ["z", "a"]},
                "s": {"type": "score", "instructions": "s", "criteria": ["low", "mid", "high"]},
            }
        )
        plans, _ = plan_request(request, FakeTokenizer(), 4096)
        answers = to_answers([[0.123456, 0.876544], [0.5, 0.5], [0.2, 0.3, 0.5]], plans)
        self.assertEqual(answers["n"], {"type": "noul", "noul": 0.123456, "confidence": 0.876544})
        self.assertEqual(answers["c"]["choice"], "z")
        self.assertEqual(answers["c"]["confidence"], 0.5)
        self.assertEqual(answers["s"]["score"], 1.3)
        self.assertEqual(answers["s"]["probabilities"], {"0": 0.2, "1": 0.3, "2": 0.5})
        self.assertNotIn("legend", answers["s"])
        self.assertNotIn("input_tokens", answers["n"])
        json.dumps(answers, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
