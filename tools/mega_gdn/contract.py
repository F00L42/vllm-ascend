# SPDX-License-Identifier: Apache-2.0
"""CPU address oracle for migration tests; never called in a model hot path."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Access:
    conv_read: int
    conv_write: int
    conv_start: int
    ssm_read: int
    ssm_writes: tuple[int, ...]


def validate_accesses(accesses: Sequence[Access], conv_blocks: int, ssm_blocks: int) -> None:
    """Reject invalid addresses and races before any test transfers data to NPU."""
    for request in accesses:
        if not 0 <= request.conv_read < conv_blocks or not 0 <= request.conv_write < conv_blocks:
            raise ValueError("Conv block out of range; padding is not an active row")
        if request.conv_start < 0:
            raise ValueError("negative Conv cursor")
        if not 0 <= request.ssm_read < ssm_blocks:
            raise ValueError("SSM read block out of range")
        if not request.ssm_writes or any(not 0 <= block < ssm_blocks for block in request.ssm_writes):
            raise ValueError("SSM write block out of range")
        if len(set(request.ssm_writes)) != len(request.ssm_writes):
            raise ValueError("duplicate checkpoints within one request")
    for index, request in enumerate(accesses):
        for other in accesses[index + 1 :]:
            if request.conv_write in (other.conv_read, other.conv_write) or other.conv_write == request.conv_read:
                raise ValueError("cross-request Conv conflict; copy-on-write is required")
            if set(request.ssm_writes).intersection((other.ssm_read, *other.ssm_writes)):
                raise ValueError("cross-request SSM conflict; copy-on-write is required")
            if other.ssm_writes.count(request.ssm_read):
                raise ValueError("cross-request SSM read/write conflict")


def slot_accesses(
    reads: Sequence[int], writes: Sequence[int], accepted: Sequence[int], sequence: int, slots: int
) -> tuple[Access, ...]:
    """Original Mega: read r*S+m-1, write w*S+t; Conv reads m-1:m+2."""
    if not 2 <= sequence <= 17 or not 1 <= slots <= 1024:
        raise ValueError("unsupported slot geometry")
    if not 1 <= len(reads) <= 32 or len(reads) != len(writes) or len(reads) != len(accepted):
        raise ValueError("inconsistent batch vectors")
    if any(not 1 <= count <= sequence for count in accepted):
        raise ValueError("accepted must be in [1, S]")
    accesses = tuple(
        Access(
            read, write, count - 1, read * sequence + count - 1, tuple(range(write * sequence, (write + 1) * sequence))
        )
        for read, write, count in zip(reads, writes, accepted)
    )
    validate_accesses(accesses, slots, slots * sequence)
    return accesses


def physical_accesses(
    table: Sequence[Sequence[int]], accepted: Sequence[int], conv_blocks: int, ssm_blocks: int
) -> tuple[Access, ...]:
    """Current Ascend native fixed-S path: Conv row[0], SSM row[m-1]/row[t].

    The table is already selected by the cache group and mamba cache mode.
    Actual tensor strides remain separate from these physical block IDs.
    """
    if not table or len(table) != len(accepted):
        raise ValueError("inconsistent batch vectors")
    sequence = len(table[0])
    if not 2 <= sequence <= 17 or any(len(row) != sequence for row in table):
        raise ValueError("expected fixed S")
    if any(not 1 <= count <= sequence for count in accepted):
        raise ValueError("accepted must be in [1, S]")
    accesses = tuple(
        Access(row[0], row[0], count - 1, row[count - 1], tuple(row)) for row, count in zip(table, accepted)
    )
    validate_accesses(accesses, conv_blocks, ssm_blocks)
    return accesses
