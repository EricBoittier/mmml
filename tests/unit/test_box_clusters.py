"""Minimum-image cluster cutting from periodic frames + labelled frame writer."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from mmml.distill.box_clusters import (
    SOURCE_BOX_DIMER,
    SOURCE_BOX_MONOMER,
    SOURCE_REFERENCE,
    BoxClusterConfig,
    box_cluster_pool,
    whole_molecules,
)
from mmml.md.metatomic_pbc import append_training_frame

L = 10.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _box() -> Atoms:
    # molecule 0 straddles the x boundary; molecule 1 sits 3 Å away through it.
    m0 = WATER + [L - 0.3, 5.0, 5.0]
    m1 = WATER + [2.7, 5.0, 5.0]
    m2 = WATER + [6.0, 0.5, 5.0]  # ~5.6-5.8 Å from m0/m1, > cutoff below
    atoms = Atoms("OH2OH2OH2", positions=np.vstack([m0, m1, m2]), cell=[L] * 3, pbc=True)
    atoms.wrap()
    return atoms


def test_whole_molecules_undo_wrap() -> None:
    mols = whole_molecules(_box(), 3)
    for mol in mols:
        assert np.allclose(mol - mol[0], WATER, atol=1e-9)


def test_dimers_use_nearest_image() -> None:
    geos = box_cluster_pool(
        [_box()],
        BoxClusterConfig(atoms_per_monomer=3, dimer_com_cutoff_A=4.0, max_dimers_per_frame=10),
        reference_monomer=Atoms("OH2", positions=WATER),
    )
    assert geos[0].source == SOURCE_REFERENCE
    dimers = [g for g in geos if g.source == SOURCE_BOX_DIMER]
    assert len(dimers) == 1  # only m0-m1 through the boundary
    d = dimers[0]
    assert d.r_com_A == pytest.approx(3.0, abs=1e-9)
    com_a, com_b = d.positions[:3].mean(0), d.positions[3:].mean(0)
    assert np.linalg.norm(com_b - com_a) < 4.0
    assert len([g for g in geos if g.source == SOURCE_BOX_MONOMER]) == 3


def test_bad_atoms_per_monomer() -> None:
    with pytest.raises(ValueError):
        whole_molecules(_box(), 4)


class _Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        r = atoms.get_positions()
        self.results = {"energy": float(0.5 * np.sum(r**2)), "forces": -r}


def test_append_training_frame_round_trip(tmp_path) -> None:
    atoms = _box()
    atoms.calc = _Harmonic()
    path = tmp_path / "traj.extxyz"
    for step in (0, 20):
        append_training_frame(path, atoms, step=step, dt_fs=0.5)
    frames = read(str(path), index=":")
    assert len(frames) == 2
    assert frames[1].info["time_fs"] == pytest.approx(10.0)
    assert np.allclose(frames[0].cell.lengths(), [L] * 3)
    assert frames[0].pbc.all()
    assert frames[0].get_potential_energy() == pytest.approx(atoms.get_potential_energy())
    assert np.allclose(frames[0].get_forces(), atoms.get_forces())


def test_topology_filter_drops_reacted_molecule() -> None:
    atoms = _box()
    pos = atoms.get_positions()
    pos[4] += [0.0, 0.0, 3.0]  # molecule 1 loses an H (index 4 = its 2nd atom's H)
    atoms.set_positions(pos)
    stats: dict = {}
    geos = box_cluster_pool(
        [atoms],
        BoxClusterConfig(atoms_per_monomer=3, dimer_com_cutoff_A=8.0, max_dimers_per_frame=10),
        reference_monomer=Atoms("OH2", positions=WATER),
        stats=stats,
    )
    assert stats["n_broken_molecules"] == 1
    assert len([g for g in geos if g.source == SOURCE_BOX_MONOMER]) == 2
    # only the m0-m2 pair survives; every pair with molecule 1 is gone
    assert len([g for g in geos if g.source == SOURCE_BOX_DIMER]) == 1
