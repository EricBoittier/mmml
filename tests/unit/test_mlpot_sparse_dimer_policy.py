"""Tests for sparse ML dimer cap policy."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    max_dimer_pairs,
    resolve_max_active_dimers,
    validate_sparse_dimer_cap,
)


def test_resolve_max_active_dimers_default_90():
    assert resolve_max_active_dimers(90, 4005) == 1000


def test_resolve_max_active_dimers_free_space_uses_all_pairs():
    assert max_dimer_pairs(90) == 4005
    assert resolve_max_active_dimers(90, 4005, free_space=True) == 4005


def test_resolve_max_active_dimers_free_space_promotes_lower_explicit():
    assert resolve_max_active_dimers(90, 4005, explicit=1000, free_space=True) == 4005


def test_resolve_max_active_dimers_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_MAX_ACTIVE_DIMERS", "1500")
    assert resolve_max_active_dimers(90, 4005) == 1500


def test_resolve_max_active_dimers_free_space_promotes_lower_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_MAX_ACTIVE_DIMERS", "1500")
    assert resolve_max_active_dimers(90, 4005, free_space=True) == 4005


def test_resolve_max_active_dimers_small_cluster():
    assert resolve_max_active_dimers(5, 10) == 10


def test_validate_sparse_dimer_cap_random_sparse():
    rng = np.random.default_rng(0)
    n = 20
    apm = 10
    pos = rng.standard_normal((n * apm, 3)) * 5.0
    stats = validate_sparse_dimer_cap(pos, n, apm, mm_switch_on=7.0, box_side_A=None)
    assert stats["n_dimers_total"] == n * (n - 1) // 2
    assert "verdict" in stats
    assert isinstance(stats["ok"], bool)


def test_count_near_dimer_pairs_free_space_cap_is_all_pairs():
    n = 10
    apm = 5
    pos = np.zeros((n * apm, 3), dtype=np.float64)
    stats = validate_sparse_dimer_cap(pos, n, apm, mm_switch_on=7.0, free_space=True)
    assert stats["max_active_dimers_cap"] == n * (n - 1) // 2
    assert stats["free_space"] is True
