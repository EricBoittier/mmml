"""Verlet-skin reuse for host-built neighbor lists.

The safety property that matters: a list built at ``cutoff + skin`` may be
reused only while every atom has moved less than ``skin / 2``, because two
atoms can close on each other at twice the per-atom rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.neighbor_cache import with_verlet_skin


def _counting_neighbor_fn():
    calls = {"n": 0, "last_pos": None}

    def neighbor_fn(pos, box):
        calls["n"] += 1
        calls["last_pos"] = np.asarray(pos).copy()
        return {"pair_i": np.zeros(4, dtype=np.int32), "pair_mask": np.ones(4, dtype=np.int8)}

    return neighbor_fn, calls


def test_first_call_always_builds():
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    cached(np.zeros((3, 3)), None)
    assert calls["n"] == 1


def test_reuses_while_displacement_is_under_half_the_skin():
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    pos = np.zeros((3, 3))
    cached(pos, None)

    moved = pos.copy()
    moved[0, 0] = 0.49  # < skin/2 = 0.5
    cached(moved, None)

    assert calls["n"] == 1, "should have reused the cached list"


def test_rebuilds_once_displacement_exceeds_half_the_skin():
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    pos = np.zeros((3, 3))
    cached(pos, None)

    moved = pos.copy()
    moved[0, 0] = 0.51  # > skin/2
    cached(moved, None)

    assert calls["n"] == 2


def test_displacement_is_measured_from_the_last_rebuild_not_the_last_call():
    """Drift must accumulate, or many sub-threshold steps slip past the skin."""
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    pos = np.zeros((3, 3))
    cached(pos, None)

    # Ten steps of 0.1 Å: each step is small, the total is 1.0 Å > skin/2.
    for step in range(1, 11):
        moved = pos.copy()
        moved[0, 0] = 0.1 * step
        cached(moved, None)

    assert calls["n"] == 2, "cache must compare against the last rebuild frame"


def test_box_change_forces_a_rebuild_even_without_motion():
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    pos = np.zeros((3, 3))
    box = np.eye(3) * 20.0

    cached(pos, box)
    cached(pos, box * 1.05)

    assert calls["n"] == 2


def test_reuse_returns_the_identical_mapping_object():
    fn, _ = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    first = cached(np.zeros((3, 3)), None)
    second = cached(np.zeros((3, 3)), None)
    assert first is second


def test_nonpositive_skin_disables_caching_entirely():
    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=0.0, to_device=False)
    assert cached is fn
    cached(np.zeros((3, 3)), None)
    cached(np.zeros((3, 3)), None)
    assert calls["n"] == 2


def test_wrapper_advertises_device_native_to_the_driver():
    fn, _ = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=0.5, to_device=False)
    assert getattr(cached, "device_native", False) is True


def test_stats_track_reuse_and_rebuilds():
    fn, _ = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=False)
    pos = np.zeros((3, 3))
    cached(pos, None)
    cached(pos, None)
    cached(pos + 10.0, None)

    stats = cached.stats.as_dict()
    assert stats["calls"] == 3
    assert stats["rebuilds"] == 2
    assert stats["reused"] == 1


def test_rejects_non_mapping_result():
    def bad_fn(pos, box):
        return (np.zeros(4), np.ones(4))

    cached = with_verlet_skin(bad_fn, skin_A=1.0, to_device=False)
    with pytest.raises(TypeError, match="must return a mapping"):
        cached(np.zeros((3, 3)), None)


def test_device_array_input_skips_the_host_download_on_reuse():
    jnp = pytest.importorskip("jax.numpy")

    fn, calls = _counting_neighbor_fn()
    cached = with_verlet_skin(fn, skin_A=1.0, to_device=True)
    pos = jnp.zeros((3, 3))

    cached(pos, None)
    cached(pos, None)

    stats = cached.stats.as_dict()
    assert stats["reused"] == 1
    # One download for the single rebuild, none for the reused call.
    assert stats["host_syncs"] == 1
    # The host build must still receive numpy, not a tracer/device array.
    assert isinstance(calls["last_pos"], np.ndarray)
