"""Mechanical-embedding ML region helpers for jaxmd-unified md-system."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.ml_region import (
    apply_ml_resnames_mechanical_embedding,
    merge_ml_region_mol_id,
    parse_ml_resnames,
    per_atom_residue_names,
    resolve_ml_region_indices,
)
from mmml.md.system import MolecularSystem


def _toy_solute_solvent() -> MolecularSystem:
    # AMM1 (2) + CH3CL (2) + TIP3 (3)
    R = np.zeros((7, 3), dtype=np.float64)
    Z = np.array([7, 1, 6, 17, 8, 1, 1], dtype=np.int32)
    mol_id = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    monomers = [
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
    ]
    return MolecularSystem(
        R=R,
        Z=Z,
        box=np.eye(3) * 20.0,
        mol_id=mol_id,
        monomer_indices=monomers,
        water_indices=[monomers[2]],
        metadata={"residue_names": ("AMM1", "CH3CL", "TIP3")},
    )


def test_parse_ml_resnames_cli_and_yaml():
    assert parse_ml_resnames(None) is None
    assert parse_ml_resnames("AMM1,CH3CL") == ("AMM1", "CH3CL")
    assert parse_ml_resnames(["AMM1", "CH3CL"]) == ("AMM1", "CH3CL")
    assert parse_ml_resnames("  ") is None


def test_per_atom_residue_names_expands_per_molecule():
    system = _toy_solute_solvent()
    names = per_atom_residue_names(system)
    assert names == ["AMM1", "AMM1", "CH3CL", "CH3CL", "TIP3", "TIP3", "TIP3"]


def test_apply_ml_resnames_restricts_ml_and_merges_mol_id():
    system = _toy_solute_solvent()
    new_system, term_kwargs, ml_idx = apply_ml_resnames_mechanical_embedding(
        system, ("AMM1", "CH3CL")
    )
    assert ml_idx.tolist() == [0, 1, 2, 3]
    assert new_system.mol_id[0] == new_system.mol_id[3]
    assert new_system.mol_id[0] != new_system.mol_id[4]
    assert len(new_system.monomer_indices) == 2  # merged solute + TIP3
    assert term_kwargs["ml_intra"]["monomer_indices"][0].tolist() == [0, 1, 2, 3]
    assert term_kwargs["mm_bonded"]["ml_atom_indices"].tolist() == [0, 1, 2, 3]
    # Solute–solute would share mol_id → intermolecular MM filter drops them
    assert merge_ml_region_mol_id(system.mol_id, ml_idx)[0] == merge_ml_region_mol_id(
        system.mol_id, ml_idx
    )[2]


def test_resolve_rejects_unknown_resname():
    with pytest.raises(ValueError, match="no atoms match"):
        resolve_ml_region_indices(["TIP3", "TIP3", "TIP3"], ("AMM1",))


def test_sol_tip3_yaml_points_at_model_ext_and_ml_resnames():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "examples/m/yaml/sol_tip3_30A_md.yaml").read_text())
    assert raw["checkpoint"].endswith("model_ext.json")
    assert raw["ml_resnames"] == ["AMM1", "CH3CL"]
    assert raw.get("jaxmd_unified") is True
