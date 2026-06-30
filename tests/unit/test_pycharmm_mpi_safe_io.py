"""Source-level checks for MPI-safe PyCHARMM file I/O (no WriteScript on hot paths)."""

from __future__ import annotations

from pathlib import Path


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
