# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.mega_gdn.contract import Access, physical_accesses, slot_accesses, validate_accesses


def test_noncontiguous_physical_blocks_are_not_slot_arithmetic():
    plan = physical_accesses([[7, 2, 11], [4, 9, 1]], [2, 3], 12, 12)
    assert plan == (Access(7, 7, 1, 2, (7, 2, 11)), Access(4, 4, 2, 1, (4, 9, 1)))
    assert plan[0].ssm_read != 7 * 3 + 2 - 1


@pytest.mark.parametrize("accepted", [1, 2, 3])
def test_slot_boundary_and_conv_window(accepted):
    plan = slot_accesses([2], [4], [accepted], 3, 5)[0]
    assert plan.ssm_read == 6 + accepted - 1
    assert plan.ssm_writes == (12, 13, 14)
    assert plan.conv_start == accepted - 1
    assert plan.conv_start + 3 <= 5


def test_batch_reorder_follows_request_table():
    first = physical_accesses([[7, 2, 11], [4, 9, 1]], [2, 3], 12, 12)
    second = physical_accesses([[4, 9, 1], [7, 2, 11]], [3, 2], 12, 12)
    assert second == first[::-1]


def test_postprocess_compaction_resets_cursor_to_one():
    # Emulate a postprocess copy of accepted checkpoint 11 to new running block 8.
    pool = {2: "rejected", 7: "old", 11: "accepted"}
    pool[8] = pool[11]
    next_plan = physical_accesses([[8, 3, 6]], [1], 12, 12)[0]
    assert pool[next_plan.ssm_read] == "accepted"
    assert next_plan.conv_start == 0


def test_shared_read_only_slots_and_self_alias_are_valid():
    slot_accesses([0, 0], [1, 2], [1, 3], 3, 3)
    slot_accesses([0, 1], [0, 1], [1, 3], 3, 2)


@pytest.mark.parametrize(
    "reads,writes,accepted",
    [
        ([0], [1], [0]),
        ([0], [1], [4]),
        ([-1], [1], [1]),
        ([0], [3], [1]),
        ([0, 0], [1, 1], [1, 1]),
        ([0, 1], [1, 2], [1, 1]),
        ([0, 1], [1, 0], [1, 1]),
    ],
)
def test_slot_rejects_invalid_cursor_padding_and_races(reads, writes, accepted):
    with pytest.raises(ValueError):
        slot_accesses(reads, writes, accepted, 3, 3)


@pytest.mark.parametrize("table", [[[1, -1, 3]], [[1, 1, 2]], [[1, 2, 9]], [[1, 2, 3], [4, 2, 5]]])
def test_physical_rejects_padding_duplicates_and_shared_writes(table):
    with pytest.raises(ValueError):
        physical_accesses(table, [1] * len(table), 9, 9)


def test_independent_conv_and_ssm_conflicts():
    with pytest.raises(ValueError, match="SSM"):
        validate_accesses([Access(0, 1, 0, 0, (2, 3)), Access(4, 5, 0, 2, (6, 7))], 8, 8)
