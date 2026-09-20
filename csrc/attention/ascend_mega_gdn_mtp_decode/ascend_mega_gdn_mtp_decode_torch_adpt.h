// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The vLLM Ascend contributors.
#ifndef ASCEND_MEGA_GDN_MTP_DECODE_TORCH_ADPT_H
#define ASCEND_MEGA_GDN_MTP_DECODE_TORCH_ADPT_H

#include <ATen/MemoryOverlap.h>

namespace vllm_ascend {

// Experimental P1 slot ABI. This is NOT a vLLM physical-cache adapter.
// The caller owns index validity and cross-request read/write exclusivity;
// inspecting device index values here would introduce a host synchronization.
std::tuple<at::Tensor, at::Tensor> npu_mega_gdn_mtp_decode(
    const at::Tensor& qkv, const at::Tensor& z,
    const at::Tensor& b, const at::Tensor& a,
    const at::Tensor& conv_weight, at::Tensor& conv_state,
    const at::Tensor& a_log, const at::Tensor& dt_bias,
    at::Tensor& ssm_state, const at::Tensor& read_state_indices,
    const at::Tensor& write_state_indices,
    const at::Tensor& num_accepted_tokens, const at::Tensor& norm_weight,
    bool fla_ssm_state_layout)
{
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kConvWidth = 4;
    constexpr int64_t kMaxBatch = 32;
    constexpr int64_t kMaxSequence = 17;
    constexpr int64_t kMaxSlots = 1024;
    constexpr int64_t kMaxKeyHeads = 16;
    constexpr int64_t kMaxHeadRatio = 4;
    TORCH_CHECK(qkv.device().type() == c10::DeviceType::PrivateUse1,
                "MEGA requires NPU tensors");
    const auto check = [&](const at::Tensor& t, at::ScalarType dtype,
                           int64_t rank, const char* name) {
        TORCH_CHECK(t.device() == qkv.device(), name, " must be on the same NPU as qkv");
        TORCH_CHECK(t.scalar_type() == dtype, name, " has unsupported dtype");
        TORCH_CHECK(t.dim() == rank, name, " has unsupported rank");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (no implicit state copies)");
        TORCH_CHECK(IsOpInputBaseFormat(t), name, " must use a base NPU storage format");
    };
    check(qkv, at::kBFloat16, 3, "qkv");
    check(z, at::kBFloat16, 4, "z");
    check(b, at::kBFloat16, 3, "b");
    check(a, at::kBFloat16, 3, "a");
    check(conv_weight, at::kBFloat16, 2, "conv_weight");
    check(conv_state, at::kBFloat16, 3, "conv_state");
    check(a_log, at::kFloat, 1, "a_log");
    check(dt_bias, at::kFloat, 1, "dt_bias");
    check(ssm_state, at::kFloat, 4, "ssm_state");
    check(read_state_indices, at::kInt, 1, "read_state_indices");
    check(write_state_indices, at::kInt, 1, "write_state_indices");
    check(num_accepted_tokens, at::kInt, 1, "num_accepted_tokens");
    check(norm_weight, at::kBFloat16, 1, "norm_weight");
    const auto batch = qkv.size(0);
    const auto sequence = qkv.size(1);
    const auto channels = qkv.size(2);
    const auto value_heads = z.size(2);
    const auto slots = conv_state.size(0);
    const auto qk_channels = channels - value_heads * kHeadDim;
    TORCH_CHECK(batch >= 1 && batch <= kMaxBatch && sequence >= 2 && sequence <= kMaxSequence,
                "MEGA requires 1<=B<=32 and 2<=S<=17");
    TORCH_CHECK(slots >= 1 && slots <= kMaxSlots, "MEGA requires 1<=N<=1024 slots");
    TORCH_CHECK(qk_channels > 0 && qk_channels % (2 * kHeadDim) == 0,
                "MEGA qkv channels must equal (2*NK+NV)*128");
    const auto key_heads = qk_channels / (2 * kHeadDim);
    TORCH_CHECK(key_heads <= kMaxKeyHeads && (key_heads & (key_heads - 1)) == 0 &&
                value_heads >= key_heads && value_heads % key_heads == 0 &&
                value_heads / key_heads <= kMaxHeadRatio, "MEGA unsupported head geometry");
    const auto shape = [](const at::Tensor& t, std::initializer_list<int64_t> dims,
                          const char* name) {
        TORCH_CHECK(t.sizes() == at::IntArrayRef(dims), name, " has unsupported shape");
    };
    shape(z, {batch, sequence, value_heads, kHeadDim}, "z");
    shape(a, {batch, sequence, value_heads}, "a");
    shape(b, {batch, sequence, value_heads}, "b");
    shape(conv_weight, {kConvWidth, channels}, "conv_weight");
    shape(conv_state, {slots, sequence + kConvWidth - 2, channels}, "conv_state");
    shape(ssm_state, {slots * sequence, value_heads, kHeadDim, kHeadDim}, "ssm_state");
    shape(a_log, {value_heads}, "a_log");
    shape(dt_bias, {value_heads}, "dt_bias");
    shape(read_state_indices, {batch}, "read_state_indices");
    shape(write_state_indices, {batch}, "write_state_indices");
    shape(num_accepted_tokens, {batch}, "num_accepted_tokens");
    shape(norm_weight, {kHeadDim}, "norm_weight");
    // In-place pools may not overlap read-only input storage or each other.
    at::assert_no_overlap(conv_state, ssm_state);
    for (const auto* input : {&qkv, &z, &b, &a, &conv_weight, &a_log, &dt_bias,
                              &read_state_indices, &write_state_indices,
                              &num_accepted_tokens, &norm_weight}) {
        at::assert_no_overlap(conv_state, *input);
        at::assert_no_overlap(ssm_state, *input);
    }
    auto conv_out = at::empty_like(qkv);
    auto out = at::empty_like(z);
    EXEC_NPU_CMD(aclnnAscendMegaGdnMtpDecode,
                 qkv, z, b, a, conv_weight, conv_state, a_log, dt_bias, ssm_state,
                 read_state_indices, write_state_indices, num_accepted_tokens,
                 norm_weight, fla_ssm_state_layout,
                 conv_out, conv_state, ssm_state, out);
    return {conv_out, out};
}

} // namespace vllm_ascend
#endif
