# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from xLLM-AI/xllm-ops fee381637f072a4b3a3d3e320aad327b17e27738,
# test/python_test/test_mega_gdn_mtp_decode.py. NPU imports removed so this oracle
# can run on CPU. This preserves xLLM rounding and is not native-vLLM equivalence.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

HEAD_DIM = 128
SUPPORTED_SPECULATIVE_TOKENS = tuple(range(1, 17))


@dataclass(frozen=True)
class MegaGdnMtpResult:
    conv_out: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    out: torch.Tensor


def make_inputs(
    speculative_tokens: int,
    *,
    batch_size: int = 1,
    num_k_heads: int = 1,
    num_v_heads: int = 1,
    same_slot: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        20260804 + speculative_tokens + 10 * batch_size + 100 * num_k_heads + 1000 * num_v_heads
    )
    sequence_length = speculative_tokens + 1
    num_state_slots = batch_size if same_slot else batch_size + 1
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM

    def rand_bfloat16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32).mul_(0.1).to(torch.bfloat16)

    if same_slot:
        state_indices = torch.arange(batch_size, dtype=torch.int32)
        read_state_indices = state_indices
        write_state_indices = state_indices.clone()
    else:
        read_state_indices = torch.zeros(batch_size, dtype=torch.int32)
        write_state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32)

    return {
        "qkv": rand_bfloat16(batch_size, sequence_length, conv_dim),
        "z": rand_bfloat16(batch_size, sequence_length, num_v_heads, HEAD_DIM),
        "b": rand_bfloat16(batch_size, sequence_length, num_v_heads),
        "a": rand_bfloat16(batch_size, sequence_length, num_v_heads),
        "conv_weight": rand_bfloat16(4, conv_dim),
        "conv_state": rand_bfloat16(num_state_slots, sequence_length + 2, conv_dim),
        "a_log": torch.full((num_v_heads,), -1.0, dtype=torch.float32),
        "dt_bias": torch.zeros(num_v_heads, dtype=torch.float32),
        "ssm_state": torch.randn(
            num_state_slots * sequence_length,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.1),
        "read_state_indices": read_state_indices,
        "write_state_indices": write_state_indices,
        "num_accepted_tokens": torch.full(
            (batch_size,),
            (sequence_length + 1) // 2,
            dtype=torch.int32,
        ),
        "norm_weight": rand_bfloat16(HEAD_DIM),
    }


def _l2_normalize(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)


def reference(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_output: torch.Tensor | None = None,
    fla_ssm_state_layout: bool = True,
) -> MegaGdnMtpResult:
    batch_size, sequence_length, conv_dim = qkv.shape
    num_v_heads = z.size(2)
    num_k_heads = (conv_dim - num_v_heads * HEAD_DIM) // (2 * HEAD_DIM)
    state_stride = sequence_length

    conv_state_out = conv_state.clone()
    ssm_state_out = ssm_state.clone()
    conv_outputs = []
    outputs = []

    read_conv_snapshots = conv_state.index_select(0, read_state_indices.to(torch.int64)).clone()
    read_checkpoints = read_state_indices.to(torch.int64) * state_stride + num_accepted_tokens.to(torch.int64) - 1
    read_ssm_snapshots = ssm_state.index_select(0, read_checkpoints).clone()

    for batch_idx in range(batch_size):
        write_slot = int(write_state_indices[batch_idx])
        accepted = int(num_accepted_tokens[batch_idx])
        read_conv = read_conv_snapshots[batch_idx]
        if conv_output is None:
            history = read_conv[accepted - 1 : accepted + 2].float()
            token_conv_outputs = []
            for token_idx in range(sequence_length):
                token = qkv[batch_idx, token_idx].float()
                conv_acc = (history * conv_weight[:3].float()).sum(dim=0) + token * conv_weight[3].float()
                conv_fp32 = conv_acc * torch.reciprocal(torch.exp(-conv_acc) + 1.0)
                token_conv_outputs.append(conv_fp32.to(torch.bfloat16))
                history = torch.cat((history[1:], token.unsqueeze(0)), dim=0)
            batch_conv = torch.stack(token_conv_outputs)
        else:
            batch_conv = conv_output[batch_idx]
        conv_outputs.append(batch_conv)
        conv_state_out[write_slot, :2] = read_conv[accepted : accepted + 2]
        conv_state_out[write_slot, 2 : sequence_length + 2] = qkv[batch_idx]

        q = batch_conv[:, : num_k_heads * HEAD_DIM].reshape(sequence_length, num_k_heads, HEAD_DIM)
        k = batch_conv[:, num_k_heads * HEAD_DIM : 2 * num_k_heads * HEAD_DIM].reshape(
            sequence_length, num_k_heads, HEAD_DIM
        )
        v = batch_conv[:, 2 * num_k_heads * HEAD_DIM :].reshape(sequence_length, num_v_heads, HEAD_DIM)
        q = _l2_normalize(q.float()) / HEAD_DIM**0.5
        k = _l2_normalize(k.float())
        repeats = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)

        stored_state = read_ssm_snapshots[batch_idx].float()
        state = stored_state if fla_ssm_state_layout else stored_state.transpose(-1, -2)
        token_outputs = []
        for token_idx in range(sequence_length):
            g = (-torch.exp(a_log) * F.softplus(a[batch_idx, token_idx].float() + dt_bias)).to(torch.bfloat16).float()
            decay = torch.exp(g)
            beta = torch.sigmoid(b[batch_idx, token_idx].float()).to(torch.bfloat16).float()
            state = state * decay[:, None, None]
            prediction = torch.einsum("hkv,hk->hv", state, k[token_idx])
            delta = (v[token_idx].float() - prediction) * beta[:, None]
            state = state + torch.einsum("hk,hv->hkv", k[token_idx], delta)
            readout = torch.einsum("hkv,hk->hv", state, q[token_idx])

            checkpoint = write_slot * state_stride + token_idx
            ssm_state_out[checkpoint] = state if fla_ssm_state_layout else state.transpose(-1, -2)

            norm_input = readout.to(torch.bfloat16).float()
            rms_inv = torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + 1e-6)
            norm_output = norm_input * rms_inv * norm_weight.float()
            norm_output = norm_output * F.silu(z[batch_idx, token_idx].float())
            token_outputs.append(norm_output.to(torch.bfloat16))
        outputs.append(torch.stack(token_outputs))

    return MegaGdnMtpResult(
        conv_out=torch.stack(conv_outputs),
        conv_state=conv_state_out,
        ssm_state=ssm_state_out,
        out=torch.stack(outputs),
    )
