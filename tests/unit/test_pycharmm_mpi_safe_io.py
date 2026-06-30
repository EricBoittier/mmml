"""Source-level checks for MPI-safe PyCHARMM file I/O (no WriteScript on hot paths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_read_rtf_uses_c_api_not_command_script():
    block = _read("pycharmm/read.py").split("def rtf(")[1].split("\ndef ")[0]
    assert "read_rtf_file" in block
    assert "CommandScript" not in block
    assert "c_api_path_buffer" in block
    assert "byref(append)" in block


def test_read_prm_uses_c_api_not_command_script():
    block = _read("pycharmm/read.py").split("def prm(")[1].split("\ndef ")[0]
    assert "read_param_file" in block
    assert "CommandScript" not in block


def test_write_coor_pdb_uses_c_api_not_write_script():
    block = _read("pycharmm/write.py").split("def coor_pdb(")[1].split("\ndef ")[0]
    assert "write_coor_pdb" in block
    assert "WriteScript" not in block
    assert "c_api_path_buffer" in block


def test_write_coor_card_avoids_write_script_when_mmml_available():
    block = _read("pycharmm/write.py").split("def coor_card(")[1].split("\ndef ")[0]
    assert "write_charmm_crd_from_charmm" in block
    assert "WriteScript" in block  # fallback when mmml not importable


def test_write_charmm_crd_from_charmm_avoids_charmm_write():
    block = (
        _read("mmml/interfaces/pycharmmInterface/mlpot/setup.py")
        .split("def write_charmm_crd_from_charmm(")[1]
        .split("\ndef ")[0]
    )
    assert "coor_card" not in block
    assert "WriteScript" not in block
    assert " EXT" in block


def test_write_charmm_restart_from_memory_roundtrip(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
        read_restart_natom,
        write_charmm_restart_from_memory,
    )

    pos = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=float,
    )
    res = tmp_path / "mini_box_equil.res"

    write_charmm_restart_from_memory(
        res, positions=pos, title="seed", include_crystal=False
    )

    assert read_restart_natom(res) == 3
    np.testing.assert_allclose(read_restart_coordinates(res), pos, rtol=0, atol=1e-12)


def test_rewrite_dynamics_restart_avoids_charmm_script():
    block = (
        _read("mmml/interfaces/pycharmmInterface/mlpot/bonded_mm_recovery.py")
        .split("def rewrite_dynamics_restart_from_current_state(")[1]
        .split("\ndef ")[0]
    )
    assert "write_charmm_restart_from_memory" in block
    assert "charmm_script" not in block
    assert "write restart" not in block
