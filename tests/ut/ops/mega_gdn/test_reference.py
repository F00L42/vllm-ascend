# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.mega_gdn.reference import make_inputs, reference


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_conv_history_and_untouched_state_are_exact(accepted):
    inputs = make_inputs(2)
    inputs["num_accepted_tokens"].fill_(accepted)
    before = {name: value.clone() for name, value in inputs.items()}
    result = reference(**inputs, fla_ssm_state_layout=False)
    torch.testing.assert_close(
        result.conv_state[1, :2], before["conv_state"][0, accepted : accepted + 2], rtol=0, atol=0
    )
    torch.testing.assert_close(result.conv_state[1, 2:], inputs["qkv"][0], rtol=0, atol=0)
    torch.testing.assert_close(result.conv_state[0], before["conv_state"][0], rtol=0, atol=0)
    torch.testing.assert_close(result.ssm_state[:3], before["ssm_state"][:3], rtol=0, atol=0)
    for name, value in inputs.items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_matrix_layout_transpose_preserves_math():
    inputs = make_inputs(2, num_k_heads=2, num_v_heads=4)
    fla = reference(**inputs, fla_ssm_state_layout=True)
    inputs["ssm_state"] = inputs["ssm_state"].transpose(-1, -2).contiguous()
    non_fla = reference(**inputs, fla_ssm_state_layout=False)
    torch.testing.assert_close(fla.out, non_fla.out, rtol=0, atol=0)
    torch.testing.assert_close(fla.ssm_state.transpose(-1, -2), non_fla.ssm_state, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [1, 3])
def test_second_round_uses_only_accepted_checkpoint_and_history(accepted):
    inputs = make_inputs(2, same_slot=True)
    first = reference(**inputs, fla_ssm_state_layout=False)
    inputs["conv_state"] = first.conv_state
    inputs["ssm_state"] = first.ssm_state
    inputs["num_accepted_tokens"].fill_(accepted)
    expected = reference(**inputs, fla_ssm_state_layout=False)
    # Destroy every rejected/non-selected SSM checkpoint and unused Conv row.
    perturbed = {name: value.clone() for name, value in inputs.items()}
    for index in range(3):
        if index != accepted - 1:
            perturbed["ssm_state"][index].fill_(1000)
    for index in range(5):
        if index not in range(accepted - 1, accepted + 2):
            perturbed["conv_state"][0, index].fill_(1000)
    actual = reference(**perturbed, fla_ssm_state_layout=False)
    torch.testing.assert_close(expected.out, actual.out, rtol=0, atol=0)
    torch.testing.assert_close(expected.ssm_state, actual.ssm_state, rtol=0, atol=0)
    torch.testing.assert_close(expected.conv_state, actual.conv_state, rtol=0, atol=0)
