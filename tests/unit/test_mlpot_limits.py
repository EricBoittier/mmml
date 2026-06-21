"""Tests for CHARMM MLpot compile-time limits."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

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


def test_max_mlpot_ml_pairs_pbc_aco_165():
    # 165 monomers × 5 atoms; observed fort.104 at ~4.1M pairs with default max_Npr.
    n = 825
    pairs = mlpot_limits.max_mlpot_ml_pairs_pbc(n)
    assert pairs > mlpot_limits.max_mlpot_ml_pairs(n)
    assert pairs > 4_000_000


def test_pbc_image_copies_dense_aco_200_l32():
    copies = mlpot_limits.pbc_image_copies_per_atom(2000, 32.0)
    assert copies > 5.0
    pairs = mlpot_limits.max_mlpot_ml_pairs_pbc(2000, box_side_A=32.0)
    assert pairs > mlpot_limits.max_mlpot_ml_pairs_pbc(2000)
    assert mlpot_limits.required_max_npr(2000, pbc=True, box_side_A=32.0) > 36_000_000


def test_select_npr_tier_aco_200_pbc_l32():
    assert mlpot_limits.select_npr_tier(2000, pbc=True, box_side_A=32.0) == "xxxlarge"


def test_select_npr_tier():
    assert mlpot_limits.select_npr_tier(89) == "default"
    assert mlpot_limits.select_npr_tier(2195) == "large"
    assert mlpot_limits.select_npr_tier(2200) == "large"
    assert mlpot_limits.select_npr_tier(3000) == "xlarge"
    assert mlpot_limits.select_npr_tier(4390) == "xxlarge"
    assert mlpot_limits.select_npr_tier(6000) == "xxxlarge"
    with pytest.raises(ValueError, match="largest tier"):
        mlpot_limits.select_npr_tier(7000)


def test_select_npr_tier_pbc():
    assert mlpot_limits.select_npr_tier(825, pbc=True) == "large"
    assert mlpot_limits.select_npr_tier(2200, pbc=True) == "xxlarge"


def test_required_max_npr_margin():
    assert mlpot_limits.required_max_npr(2195) > mlpot_limits.max_mlpot_ml_pairs(2195)
    assert mlpot_limits.required_max_npr(825, pbc=True) > 4_000_000


def test_ensure_mlpot_limits_for_system_raises_with_tier(monkeypatch):
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_ML", "50000")
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "3998000")
    with pytest.raises(ValueError, match="ensure_charmm_mlpot_limits"):
        mlpot_limits.ensure_mlpot_limits_for_system(2195)


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


def test_estimate_ml_atoms_aco_solvent():
    assert mlpot_limits.estimate_ml_atoms(110, solvent="ACO") == 1100
    assert mlpot_limits.estimate_ml_atoms(110, solvent="DCM") == 550


def test_select_npr_tier_aco_110_pbc():
    assert mlpot_limits.select_npr_tier(1100, pbc=True) == "xlarge"


def test_select_npr_tier_aco_165_pbc():
    assert mlpot_limits.select_npr_tier(1650, pbc=True) == "xxlarge"


def test_select_npr_tier_for_build_aco_266_dense_l32():
    assert (
        mlpot_limits.select_npr_tier_for_build(2660, pbc=True, box_side_A=32.0)
        == "xxxlarge"
    )
    assert mlpot_limits.pbc_pair_budget_box_side_A(2660, 32.0) is None
    assert mlpot_limits.pbc_pair_budget_box_side_A(2000, 32.0) == 32.0


def test_validate_pbc_needs_larger_npr_than_vacuum(monkeypatch):
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_ML", "50000")
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "12000000")
    _clear_limits_cache()
    # Vacuum estimate (~2.7M pairs) fits xlarge; PBC estimate (~16M) does not.
    mlpot_limits.validate_mlpot_system_size(1650, pbc=False)
    with pytest.raises(ValueError, match="max_Npr"):
        mlpot_limits.validate_mlpot_system_size(1650, pbc=True)


def test_validate_dense_aco_200_l32_needs_xxxlarge(monkeypatch):
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_ML", "50000")
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "36000000")
    _clear_limits_cache()
    with pytest.raises(ValueError, match="max_Npr"):
        mlpot_limits.validate_mlpot_system_size(2000, pbc=True, box_side_A=32.0)
    monkeypatch.setenv("MMML_CHARMM_MLPOT_MAX_PAIRS", "56000000")
    _clear_limits_cache()
    mlpot_limits.validate_mlpot_system_size(2000, pbc=True, box_side_A=32.0)


def test_register_mlpot_validates_pbc_pair_budget(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.setup import register_mlpot

    calls: list[tuple[int, bool, float | None]] = []

    def _capture(n_ml: int, *, pbc: bool = False, box_side_A: float | None = None) -> None:
        calls.append((n_ml, pbc, box_side_A))

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
        _capture,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._import_pycharmm",
        lambda: mock.MagicMock(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.physnet_ml_atomic_numbers",
        lambda z: z,
    )
    sel = mock.MagicMock()
    sel.get_atom_indexes.return_value = list(range(1650))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=mock.MagicMock(
            __enter__=mock.Mock(return_value=None),
            __exit__=mock.Mock(return_value=False),
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        return_value="all",
    ), mock.patch("mmml.interfaces.pycharmmInterface.mlpot.setup._require_mlpot_skip_iblo_support"), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._install_ml_exclusions"
    ):
        register_mlpot(mock.MagicMock(), list(range(1650)), sel, use_pbc=True)
    assert calls == [(1650, True, None)]


def test_limits_status_reads_tier_api_func(tmp_path, monkeypatch):
    import os

    tier_dir = tmp_path / "tier_8000000_nodomdec"
    lib_dir = tier_dir / "lib"
    lib_dir.mkdir(parents=True)
    lib = lib_dir / "libcharmm.so"
    lib.write_bytes(b"so")
    tier_f90 = tier_dir / "api_func.F90"
    tier_f90.write_text(
        "integer, parameter :: max_Nml = 50000\n"
        "integer, parameter :: max_Npr = 8000000\n",
        encoding="utf-8",
    )
    charmm_home = tmp_path / "setup" / "charmm"
    repo_f90 = charmm_home / "source" / "api" / "api_func.F90"
    repo_f90.parent.mkdir(parents=True)
    repo_f90.write_text(
        "integer, parameter :: max_Nml = 50000\n"
        "integer, parameter :: max_Npr = 3998000\n",
        encoding="utf-8",
    )
    stamp = old = lib.stat().st_mtime - 20
    os.utime(tier_f90, (stamp, stamp))
    os.utime(repo_f90, (stamp + 5, stamp + 5))
    os.utime(lib, (stamp + 10, stamp + 10))

    monkeypatch.setenv("CHARMM_LIB_DIR", str(lib_dir))
    monkeypatch.setenv("CHARMM_HOME", str(charmm_home))
    _clear_limits_cache()

    status = mlpot_limits.mlpot_limits_status()
    assert status.max_nml == 50000
    assert status.max_npr == 8_000_000
    assert status.api_func_f90 == tier_f90.resolve()
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
