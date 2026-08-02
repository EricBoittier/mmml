"""Unit tests for composition PDB + CGenFF parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import parse_composition_dict
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition
from mmml.interfaces.pycharmmInterface.mlpot.composition_spec import (
    CompositionEntry,
    composition_mode,
    is_composition_pdb_token,
    parse_composition_entries,
    reject_pdb_composition_for_builder,
    resolve_composition_plan,
)
from mmml.paths import bundled_file


DCM_MONOMER = bundled_file("data", "molecules", "dcm_monomer.pdb")


def test_is_composition_pdb_token() -> None:
    assert is_composition_pdb_token("solute.pdb")
    assert is_composition_pdb_token("./foo.pdb")
    assert is_composition_pdb_token("path/to/x.PDB")
    assert not is_composition_pdb_token("DCM")
    assert not is_composition_pdb_token("water")


def test_parse_composition_cgenff_and_aliases() -> None:
    assert parse_composition("DCM:2,ACO:1") == [("DCM", 2), ("ACO", 1)]
    assert parse_composition("water:3") == [("TIP3", 3)]
    assert parse_composition("ACO") == [("ACO", 1)]
    with pytest.raises(ValueError, match="Unknown CGenFF"):
        parse_composition("ZZZZZ:1")


def test_composition_mode_cgenff() -> None:
    entries = parse_composition_entries("DCM:2,MEOH:1")
    assert composition_mode(entries) == "cgenff"


def test_composition_mode_full_system_pdb() -> None:
    entries = parse_composition_entries(str(DCM_MONOMER), resolve_pdb_files=True)
    assert len(entries) == 1
    assert entries[0].pdb_path is not None
    assert composition_mode(entries) == "full_system_pdb"


def test_composition_mode_packmol_pdb_mix(tmp_path: Path) -> None:
    monomer = tmp_path / "solute.pdb"
    monomer.write_text(DCM_MONOMER.read_text(encoding="utf-8"), encoding="utf-8")
    spec = f"{monomer}:1,ACO:2"
    entries = parse_composition_entries(spec)
    assert composition_mode(entries) == "packmol_pdb"
    _entries, mode, pairs, templates = resolve_composition_plan(spec)
    assert mode == "packmol_pdb"
    assert ("DCM", 1) in pairs
    assert ("ACO", 2) in pairs
    assert templates is not None
    assert "DCM" in templates


def test_composition_mode_packmol_pdb_count_gt_one(tmp_path: Path) -> None:
    monomer = tmp_path / "dcm.pdb"
    monomer.write_text(DCM_MONOMER.read_text(encoding="utf-8"), encoding="utf-8")
    entries = parse_composition_entries(f"{monomer}:5")
    assert composition_mode(entries) == "packmol_pdb"


def test_parse_composition_dict_keys_by_resn(tmp_path: Path) -> None:
    monomer = tmp_path / "solute.pdb"
    monomer.write_text(DCM_MONOMER.read_text(encoding="utf-8"), encoding="utf-8")
    out = parse_composition_dict(f"{monomer}:1,ACO:3")
    assert out == {"DCM": 1, "ACO": 3}


def test_parse_composition_dict_full_system_returns_none() -> None:
    assert parse_composition_dict(str(DCM_MONOMER)) is None


def test_reject_pdb_with_pyxtal(tmp_path: Path) -> None:
    monomer = tmp_path / "solute.pdb"
    monomer.write_text(DCM_MONOMER.read_text(encoding="utf-8"), encoding="utf-8")
    entries = [
        CompositionEntry("DCM", 1, monomer),
        CompositionEntry("ACO", 2, None),
    ]
    with pytest.raises(ValueError, match="Packmol"):
        reject_pdb_composition_for_builder(entries, pyxtal=True)
    with pytest.raises(ValueError, match="Packmol"):
        reject_pdb_composition_for_builder(entries, packmol=False)
