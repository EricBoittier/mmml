"""Unit tests for neighbor_pair_cache_should_reuse (fast, no CHARMM)."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_energy_forces import neighbor_pair_cache_should_reuse


def test_skin_zero_interval_reuse_stable_box() -> None:
    R = np.zeros((4, 3), dtype=np.float64)
    box = np.array([40.0, 40.0, 40.0])
    assert neighbor_pair_cache_should_reuse(
        calls=2,
        interval=3,
        skin=0.0,
        R=R,
        last_R=R.copy(),
        box=box,
        last_box=box.copy(),
        have_cache=True,
    )


def test_skin_zero_box_change_no_reuse() -> None:
    R = np.zeros((4, 3), dtype=np.float64)
    box0 = np.array([40.0, 40.0, 40.0])
    box1 = np.array([40.8, 40.8, 40.8])
    assert not neighbor_pair_cache_should_reuse(
        calls=2,
        interval=3,
        skin=0.0,
        R=R,
        last_R=R.copy(),
        box=box1,
        last_box=box0,
        have_cache=True,
    )


def test_skin_positive_small_disp_reuse() -> None:
    R0 = np.zeros((4, 3), dtype=np.float64)
    R1 = R0 + 0.01
    box = np.array([40.0, 40.0, 40.0])
    assert neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=0.5,
        R=R1,
        last_R=R0,
        box=box,
        last_box=box.copy(),
        have_cache=True,
    )
