"""Unit tests for the mm_bonded mechanical-embedding term."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TIP3_PDB = Path("tests/functionality/pycharmmETC/pdb/initial.pdb")


@pytest.mark.skipif(not TIP3_PDB.is_file(), reason="TIP3 fixture PDB missing")
def test_mm_bonded_term_tip3_energy_positive_when_strained():
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        load_cgenff_bonded_from_charmm_files,
    )
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms.mm_bonded import MMBondedTerm
    from mmml.md.system import MolecularSystem

    cgenff = load_cgenff_bonded_from_charmm_files(TIP3_PDB, residue_name="TIP3")
    R = np.asarray(cgenff.positions, dtype=np.float64).copy()
    R[1] += np.array([0.05, 0.0, 0.0])  # stretch one O–H
    system = MolecularSystem(
        R=R,
        Z=np.array([8, 1, 1], dtype=np.int32),
        box=None,
        mol_id=np.zeros(3, dtype=np.int32),
        monomer_indices=[np.arange(3, dtype=np.int32)],
    )
    term = MMBondedTerm(
        ml_atom_indices=(),  # all MM
        topology=cgenff.topology,
        bonded=cgenff.bonded,
        urey_k=cgenff.urey_k,
        urey_r0=cgenff.urey_r0,
    )
    fns = term.make(system, EnergyContext())
    e = float(fns.jax_energy_fn(R))
    assert e > 0.0


def test_mm_bonded_registered():
    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy.registry import get_term

    assert get_term("mm_bonded") is not None
