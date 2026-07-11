"""Tests for the PSF system builder and FFParams ← NonbondedSystemData bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.md.builders import (
    PsfSystemBuilder,
    molecule_ids_from_bonds,
    monomer_indices_from_mol_id,
)
from mmml.md.system import FFParams, SystemSpec

REPO = Path(__file__).resolve().parents[2]


# --- FFParams field-for-field bridge (pure) ---------------------------------


def test_ffparams_from_nonbonded_system_data():
    pytest.importorskip("jax")  # NonbondedSystemData lives in a jax module
    from mmml.interfaces.pycharmmInterface.mm_system_energy import NonbondedSystemData

    n = 5
    nb = NonbondedSystemData(
        charges=np.linspace(-0.5, 0.5, n),
        at_codes=np.arange(n, dtype=np.int32),
        epsilon=np.full(n, 0.1),
        rmin=np.full(n, 1.7),  # CHARMM Rmin/2
        excluded_pairs=frozenset({(0, 1), (1, 2)}),
        e14_pairs=frozenset({(0, 3)}),
        psf_path=Path("fake.psf"),
        psf_bonds=np.array([[0, 1], [1, 2]], dtype=np.int32),
    )
    ff = FFParams.from_nonbonded_system_data(nb)

    assert np.array_equal(ff.charges, nb.charges)
    assert np.array_equal(ff.epsilon, nb.epsilon)
    assert np.array_equal(ff.rmin_half, nb.rmin)  # CHARMM rmin IS Rmin/2
    assert np.array_equal(ff.at_codes, nb.at_codes)
    assert ff.exclusions.shape == (2, 2)
    assert ff.e14_pairs.shape == (1, 2)
    # frozenset -> sorted deterministic array
    assert ff.exclusions.tolist() == [[0, 1], [1, 2]]
    assert ff.psf_path == Path("fake.psf")


def test_ffparams_empty_pairs():
    pytest.importorskip("jax")
    from mmml.interfaces.pycharmmInterface.mm_system_energy import NonbondedSystemData

    nb = NonbondedSystemData(
        charges=np.zeros(2),
        at_codes=np.zeros(2, np.int32),
        epsilon=np.zeros(2),
        rmin=np.zeros(2),
        excluded_pairs=frozenset(),
        e14_pairs=frozenset(),
    )
    ff = FFParams.from_nonbonded_system_data(nb)
    assert ff.exclusions.shape == (0, 2)
    assert ff.e14_pairs.shape == (0, 2)


# --- molecule partitioning from bonds ---------------------------------------


def test_molecule_ids_from_bonds():
    # two waters (0-1-2, 3-4-5) + one isolated ion (6)
    bonds = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    mol_id = molecule_ids_from_bonds(7, bonds)
    assert list(mol_id[:3]) == [0, 0, 0]
    assert list(mol_id[3:6]) == [1, 1, 1]
    assert mol_id[6] == 2

    mons = monomer_indices_from_mol_id(mol_id)
    assert len(mons) == 3
    assert sorted(mons[0].tolist()) == [0, 1, 2]
    assert mons[2].tolist() == [6]
    # partition is complete and disjoint
    assert sum(len(m) for m in mons) == 7


def test_molecule_ids_no_bonds():
    mol_id = molecule_ids_from_bonds(3, np.zeros((0, 2), dtype=np.int64))
    assert list(mol_id) == [0, 1, 2]  # every atom its own molecule


# --- integration: real PSF build (gated on libcharmm + fixtures) ------------


def test_psf_builder_integration():
    pytest.importorskip("ase")
    try:
        import pycharmm  # noqa: F401  (triggers libcharmm load)
    except OSError:
        pytest.skip("libcharmm not available")

    psf_path = REPO / "pept.psf"
    pdb_path = REPO / "pept.pdb"
    if not (psf_path.exists() and pdb_path.exists()):
        pytest.skip("pept.psf / pept.pdb fixtures missing")

    from ase.io import read

    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_cgenff_toppar_paths

    atoms = read(str(pdb_path))
    n = len(atoms)
    prm = resolve_cgenff_toppar_paths().prm

    spec = SystemSpec(
        builder="psf",
        params={
            "psf_path": psf_path,
            "prm_paths": [prm],
            "positions": atoms.get_positions(),
            "atomic_numbers": atoms.get_atomic_numbers(),
            "box": None,
        },
    )
    system = PsfSystemBuilder().build(spec)

    # FFParams populated field-for-field, shapes aligned to atom count
    assert system.ff_params is not None
    assert system.ff_params.charges.shape == (n,)
    assert system.ff_params.epsilon.shape == (n,)
    assert system.ff_params.rmin_half.shape == (n,)
    assert np.any(system.ff_params.charges != 0.0)  # real PSF charges
    assert system.ff_params.exclusions.shape[0] > 0  # bonded exclusions exist

    # molecule partition covers every atom exactly once
    assert system.mol_id.shape == (n,)
    assert sum(len(m) for m in system.monomer_indices) == n
    assert system.metadata["n_molecules"] == len(system.monomer_indices)
