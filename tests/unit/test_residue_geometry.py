"""Unit tests for CGenFF monomer / solvent geometry helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.analysis.residue_geometry import (
    known_solvent_density_kg_m3,
    load_residue_monomer_atoms,
    resolve_solvent_density_kg_m3,
)
from mmml.interfaces.pycharmmInterface.cgenff_residues import (
    is_cgenff_residue_name,
    normalize_cgenff_residue_name,
    require_cgenff_residue_name,
)


def test_normalize_solvent_aliases() -> None:
    assert normalize_cgenff_residue_name("water") == "TIP3"
    assert normalize_cgenff_residue_name("octanol") == "OCOH"
    assert normalize_cgenff_residue_name("meoh") == "MEOH"


def test_require_cgenff_residue_accepts_common_solvents() -> None:
    assert require_cgenff_residue_name("TIP3") == "TIP3"
    assert require_cgenff_residue_name("ACO") == "ACO"
    assert require_cgenff_residue_name("water") == "TIP3"
    with pytest.raises(ValueError, match="Unknown CGenFF"):
        require_cgenff_residue_name("NOTAREALRESIDUE")


def test_known_solvent_densities() -> None:
    assert known_solvent_density_kg_m3("TIP3") == 1000.0
    assert known_solvent_density_kg_m3("water") == 1000.0
    assert known_solvent_density_kg_m3("OCOH") == 824.0
    assert known_solvent_density_kg_m3("MEOH") is None
    assert resolve_solvent_density_kg_m3("MEOH", 792.0) == 792.0
    with pytest.raises(ValueError, match="No built-in density"):
        resolve_solvent_density_kg_m3("MEOH", None)


def test_load_bundled_and_campaign_monomers() -> None:
    tip3 = load_residue_monomer_atoms("TIP3", generate=False)
    assert len(tip3) == 3
    aco = load_residue_monomer_atoms("ACO", generate=False)
    assert len(aco) >= 9  # acetone
    # Campaign ACE geometry still loads
    ace = load_residue_monomer_atoms("ACE", generate=False)
    assert len(ace) == len(aco)


def test_load_from_cwd_pdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    # Minimal H2 "residue" PDB for a real CGenFF name that has no bundle.
    # Use a fake local override for TIP3 to prove cwd wins over bundle.
    (pdb_dir / "tip3.pdb").write_text(
        "ATOM      1  OH2 TIP3    1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  H1  TIP3    1       0.957   0.000   0.000  1.00  0.00           H\n"
        "ATOM      3  H2  TIP3    1      -0.240   0.927   0.000  1.00  0.00           H\n"
        "END\n",
        encoding="utf-8",
    )
    atoms = load_residue_monomer_atoms("TIP3", generate=False)
    assert len(atoms) == 3
    assert is_cgenff_residue_name("CYBZ")
