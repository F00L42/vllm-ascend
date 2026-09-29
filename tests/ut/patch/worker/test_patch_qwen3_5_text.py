# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.model_executor.models.interfaces import supports_mrope
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
)

from vllm_ascend.patch.worker import patch_qwen3_5


@pytest.mark.parametrize("checkpoint_prefix", ["model.", "model.language_model."])
def test_text_checkpoint_loads_with_tied_embeddings(checkpoint_prefix):
    # Exercise the real AutoWeightsLoader with a tiny tied model, without
    # initializing an NPU, a distributed group, or the full Qwen3.5 backbone.
    model = torch.nn.Module()
    model.hf_to_vllm_mapper = Qwen3_5ForCausalLMBase.hf_to_vllm_mapper
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.model.norm = torch.nn.LayerNorm(2, bias=False)
    model.lm_head = model.model.embed_tokens
    embedding = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    norm = torch.tensor([2.0, 3.0])
    weights = iter(
        [
            (f"{checkpoint_prefix}embed_tokens.weight", embedding),
            (f"{checkpoint_prefix}norm.weight", norm),
            ("mtp.unused.weight", torch.ones(1)),
        ]
    )

    loaded = Qwen3_5ForCausalLMBase.load_weights(model, weights)

    assert loaded == {"model.embed_tokens.weight", "model.norm.weight"}
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert torch.equal(model.model.norm.weight, norm)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(model.lm_head.weight, embedding)


@pytest.mark.parametrize("model_cls", [Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM])
@pytest.mark.parametrize("token_ids", [[], [101, 102, 103]])
def test_text_models_expose_three_identical_mrope_axes(model_cls, token_ids):
    assert supports_mrope(model_cls)
    positions, delta = model_cls.get_mrope_input_positions(None, token_ids, [])

    assert positions.shape == (3, len(token_ids))
    assert positions.dtype == torch.long
    assert positions.device.type == "cpu"
    assert torch.equal(positions, torch.arange(len(token_ids)).expand(3, -1))
    assert delta == 0


def test_text_mrope_rejects_multimodal_input():
    with pytest.raises(ValueError, match="do not accept multimodal"):
        patch_qwen3_5._qwen3_5_text_mrope_input_positions(None, [101], [object()])


def test_multimodal_model_keeps_native_weight_loader_and_mrope():
    assert Qwen3_5ForConditionalGeneration.load_weights is not patch_qwen3_5._qwen3_5_text_load_weights
    assert (
        Qwen3_5ForConditionalGeneration.get_mrope_input_positions
        is not patch_qwen3_5._qwen3_5_text_mrope_input_positions
    )
