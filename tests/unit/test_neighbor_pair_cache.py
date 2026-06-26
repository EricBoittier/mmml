"""Unit tests for neighbor_pair_cache_should_reuse (fast, no CHARMM)."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_CAPACITY_MULTIPLIER,
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    _fractional_positions_for_jax_md_neighbor_list,
    neighbor_pair_cache_should_reuse,
)


def test_default_jax_md_neighbor_tuning_constants() -> None:
    assert DEFAULT_JAX_MD_SKIN_DISTANCE_A == 0.25
    assert DEFAULT_JAX_MD_CAPACITY_MULTIPLIER == 1.75


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


def test_skin_positive_interval_reuse_skips_displacement_check() -> None:
    R0 = np.zeros((4, 3), dtype=np.float64)
    R1 = R0 + 10.0
    box = np.array([40.0, 40.0, 40.0])
    assert neighbor_pair_cache_should_reuse(
        calls=2,
        interval=3,
        skin=0.5,
        R=R1,
        last_R=R0,
        box=box,
        last_box=box.copy(),
        have_cache=True,
    )


def test_interval_reuse_requires_matching_box_state() -> None:
    R = np.zeros((4, 3), dtype=np.float64)
    assert not neighbor_pair_cache_should_reuse(
        calls=2,
        interval=3,
        skin=0.5,
        R=R,
        last_R=R.copy(),
        box=np.array([40.0, 40.0, 40.0]),
        last_box=None,
        have_cache=True,
    )


def test_fractional_wrap_helper_matches_legacy_shape() -> None:
    R = np.array([[0.1, 0.2, 0.3], [1.1, 0.0, 0.0]], dtype=np.float64)
    R_frac, box_diag = _fractional_positions_for_jax_md_neighbor_list(R, np.array(30.0))
    assert R_frac.shape == R.shape
    assert box_diag.shape == (3,)
    assert np.all(R_frac >= 0.0) and np.all(R_frac < 1.0)
