# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.mega_gdn.capture_native import cache_snapshot


def test_capture_retains_real_strides_and_gathers_only_referenced_blocks():
    conv = torch.arange(8 * 5 * 12).reshape(8, 5, 12)[..., :8]
    ssm = torch.arange(8 * 2 * 4 * 4).reshape(8, 2, 4, 4).transpose(-1, -2)
    table = torch.tensor([[5, 1, 7], [-1, -1, -1]], dtype=torch.int32)
    snapshot = cache_snapshot(SimpleNamespace(kv_cache=(conv, ssm)), SimpleNamespace(spec_state_indices_tensor=table))
    assert snapshot["conv_ids"].tolist() == [5]
    assert snapshot["ssm_ids"].tolist() == [1, 5, 7]
    assert snapshot["conv_layout"]["stride"] == list(conv.stride())
    assert snapshot["ssm_layout"]["stride"] == list(ssm.stride())
    torch.testing.assert_close(snapshot["ssm_blocks"], ssm[[1, 5, 7]])
    torch.testing.assert_close(snapshot["physical_table"], table)


def test_capture_does_not_clamp_invalid_positive_block_ids():
    layer = SimpleNamespace(kv_cache=(torch.zeros(2, 5, 8), torch.zeros(2, 1, 4, 4)))
    metadata = SimpleNamespace(spec_state_indices_tensor=torch.tensor([[0, 9]], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="physical index outside"):
        cache_snapshot(layer, metadata)
