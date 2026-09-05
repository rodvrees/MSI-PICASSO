"""Thread-pool sizing for steps whose per-call temporaries scale with the dataset.

The mobility-colocalization M0 rebuild allocates two length-N boolean masks per
call, N being every relevant peak in the acquisition. Threading that multiplies
peak memory by the worker count, and the pool was sized from the CPU count
alone: on her2 (3.16e9 relevant peaks) 64 threads asked for 405 GB of transient
masks and the process died silently at that exact line three runs in a row.
"""

import os

import pytest

from msi_picasso.maldi_features import _available_memory_bytes, _memory_capped_workers


def test_cap_binds_when_the_per_worker_cost_is_large():
    """her2's real numbers: 6.3 GB per worker against 680 GB free."""
    workers = _memory_capped_workers(2 * 3_161_426_135, available=680_000_000_000)

    assert workers < 64, "64 threads x 6.3 GB is what ran the machine out of memory"
    # and the resulting footprint fits inside the headroom fraction
    assert workers * 2 * 3_161_426_135 <= 0.5 * 680_000_000_000


def test_cap_does_not_bind_for_cheap_work():
    workers = _memory_capped_workers(1_000_000, available=680_000_000_000)

    assert workers == min(os.cpu_count() or 1, 64)


def test_never_returns_zero_workers():
    """Better slow than not running at all."""
    assert _memory_capped_workers(10**15, available=1_000) == 1


def test_defaults_to_real_free_memory_not_to_no_cap():
    """The default must not silently restore the unbounded behaviour.

    An `available=None` default that meant "no limit" would reintroduce exactly
    the bug this guards, for any caller that forgot the argument.
    """
    huge = 10**12  # 1 TB per worker: no real machine can run two of these
    assert _memory_capped_workers(huge) == 1


def test_scales_with_available_memory():
    per_worker = 2 * 3_161_426_135
    small = _memory_capped_workers(per_worker, available=100_000_000_000)
    large = _memory_capped_workers(per_worker, available=680_000_000_000)

    assert small < large


def test_available_memory_is_readable_or_none():
    avail = _available_memory_bytes()

    assert avail is None or avail > 0
