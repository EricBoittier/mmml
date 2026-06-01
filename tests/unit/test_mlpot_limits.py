"""Tests for CHARMM MLpot compile-time limits."""

from __future__ import annotations

from unittest import mock

from mmml.interfaces.pycharmmInterface.mlpot import mlpot_limits


def test_validate_rejects_too_many_ml_atoms(monkeypatch):
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_ML", "100")
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "100000")
    try:
        mlpot_limits.validate_mlpot_system_size(450)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "max_Nml" in str(exc) or "450" in str(exc)


def test_validate_accepts_within_limits(monkeypatch):
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_ML", "512")
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "300000")
    mlpot_limits.validate_mlpot_system_size(450)


def test_max_mlpot_ml_pairs():
    assert mlpot_limits.max_mlpot_ml_pairs(450) == 450 * 449
