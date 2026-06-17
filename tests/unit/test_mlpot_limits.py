"""Tests for CHARMM MLpot compile-time limits."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from mmml.interfaces.pycharmmInterface.mlpot import mlpot_limits


def _clear_limits_cache() -> None:
    mlpot_limits.charmm_mlpot_limits_from_source.cache_clear()


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


def test_limits_status_reads_charmmsetup_and_repo_api_func(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    f90 = repo / "setup" / "api" / "api_func.F90"
    f90.parent.mkdir(parents=True)
    f90.write_text(
        "integer, parameter :: max_Nml = 50000\n"
        "integer, parameter :: max_Npr = 3998000\n",
        encoding="utf-8",
    )
    charmm_home = repo / "setup" / "charmm"
    lib_dir = charmm_home / "lib"
    lib_dir.mkdir(parents=True)
    lib = lib_dir / "libcharmm.so"
    lib.write_bytes(b"so")
    (repo / "CHARMMSETUP").write_text(
        f"CHARMM_HOME={charmm_home}\nCHARMM_LIB_DIR={lib_dir}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("CHARMM_HOME", raising=False)
    monkeypatch.delenv("CHARMM_LIB_DIR", raising=False)
    monkeypatch.setattr(mlpot_limits, "_repo_root", lambda: repo)
    _clear_limits_cache()

    status = mlpot_limits.mlpot_limits_status()
    assert status.max_nml == 50000
    assert status.max_npr == 3998000
    assert status.api_func_f90 == f90.resolve()
    assert status.libcharmm == lib.resolve()
    assert "up to date" in status.source


def test_limits_status_stale_lib_uses_conservative_fallback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    f90 = repo / "setup" / "charmm" / "source" / "api" / "api_func.F90"
    f90.parent.mkdir(parents=True)
    f90.write_text(
        "integer, parameter :: max_Nml = 50000\n"
        "integer, parameter :: max_Npr = 3998000\n",
        encoding="utf-8",
    )
    lib = repo / "setup" / "charmm" / "libcharmm.so"
    lib.write_bytes(b"old")
    old = f90.stat().st_mtime - 10
    import os

    os.utime(lib, (old, old))

    monkeypatch.setenv("CHARMM_HOME", str(repo / "setup" / "charmm"))
    monkeypatch.setenv("CHARMM_LIB_DIR", str(repo / "setup" / "charmm"))
    monkeypatch.setattr(mlpot_limits, "_repo_root", lambda: repo)
    _clear_limits_cache()

    status = mlpot_limits.mlpot_limits_status()
    assert status.max_nml == 100
    assert status.max_npr == 100_000
    assert "older than api_func.F90" in status.source
