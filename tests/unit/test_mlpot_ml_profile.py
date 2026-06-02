"""Tests for MLpot callback profiling."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
    get_mlpot_profile_stats,
    mlpot_profiling_enabled,
    reset_mlpot_profile_stats,
)


def test_mlpot_profile_accumulates(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_PROFILE", "1")
    assert mlpot_profiling_enabled()
    reset_mlpot_profile_stats()
    stats = get_mlpot_profile_stats()
    stats.record_ml(0.5)
    stats.record_charmm_gap()
    stats.record_ml(0.3)
    line = stats.summary_line()
    assert "2 ML callbacks" in line
    assert "ML=" in line
