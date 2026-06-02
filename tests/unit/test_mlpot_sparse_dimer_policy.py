"""Tests for sparse ML dimer cap policy."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    resolve_max_active_dimers,
    validate_sparse_dimer_cap,
)


def test_resolve_max_active_dimers_default_90():
    assert resolve_max_active_dimers(90, 4005) == 1000


def test_resolve_max_active_dimers_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_MAX_ACTIVE_DIMERS", "1500")
    assert resolve_max_active_dimers(90, 4005) == 1500


def test_resolve_max_active_dimers_small_cluster():
    assert resolve_max_active_dimers(5, 10) == 10


def test_validate_sparse_dimer_cap_random_sparse():
    rng = np.random.default_rng(0)
    n = 20
    apm = 10
    pos = rng.standard_normal((n * apm, 3)) * 5.0
    stats = validate_sparse_dimer_cap(pos, n, apm, mm_switch_on=5.0, box_side_A=None)
    assert stats["n_dimers_total"] == n * (n - 1) // 2
    assert "verdict" in stats
    assert isinstance(stats["ok"], bool)
