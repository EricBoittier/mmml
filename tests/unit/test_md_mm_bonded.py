"""Unit tests for the mm_bonded energy term's ML-region filtering.

The CHARMM-parity check for the underlying bonded kernels lives in the
``cgenff_bonded`` tests; what is new here is the ``ml_atoms`` filtering, which is
what keeps a reactive ML solute from being pinned by leftover CGenFF bonds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


@dataclasses.dataclass
class _Topo:
    """Minimal stand-in with the fields ``_drop_ml_rows`` touches."""

    bonds: np.ndarray
    angles: np.ndarray
    torsions: np.ndarray
    impropers: np.ndarray
    cmap_atoms: np.ndarray
    cmap_map_idx: np.ndarray


@dataclasses.dataclass
class _Params:
    bond_k: np.ndarray
    bond_r0: np.ndarray
    angle_k: np.ndarray
    angle_theta0: np.ndarray
    torsion_k: np.ndarray
    torsion_n: np.ndarray
    torsion_gamma: np.ndarray
    improper_k: np.ndarray
    improper_n: np.ndarray
    improper_gamma: np.ndarray


def _system():
    """Atoms 0-2 are the 'ML solute'; atoms 3-8 are 'MM solvent'."""
    topo = _Topo(
        # 0-1 and 1-2 inside ML; 2-3 spans the boundary; 3-4, 4-5 pure MM
        bonds=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
        angles=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        torsions=np.array([[0, 1, 2, 3], [3, 4, 5, 6]]),
        impropers=np.array([[1, 0, 2, 3], [4, 3, 5, 6]]),
        cmap_atoms=np.empty((0, 8), dtype=int),
        cmap_map_idx=np.empty(0, dtype=int),
    )
    params = _Params(
        bond_k=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        bond_r0=np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
        angle_k=np.array([10.0, 20.0, 30.0]),
        angle_theta0=np.array([1.9, 1.8, 1.7]),
        torsion_k=np.array([0.5, 0.6]),
        torsion_n=np.array([3.0, 3.0]),
        torsion_gamma=np.array([0.0, 0.0]),
        improper_k=np.array([7.0, 8.0]),
        improper_n=np.array([0.0, 0.0]),
        improper_gamma=np.array([0.0, 0.0]),
    )
    return topo, params


def test_rows_touching_the_ml_region_are_dropped():
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    topo, params = _system()
    new_topo, new_params, report = _drop_ml_rows(topo, params, frozenset({0, 1, 2}))

    # Only the two pure-MM bonds survive; 2-3 crosses the boundary and goes too.
    np.testing.assert_array_equal(new_topo.bonds, [[3, 4], [4, 5]])
    np.testing.assert_allclose(new_params.bond_k, [4.0, 5.0])
    np.testing.assert_allclose(new_params.bond_r0, [1.4, 1.5])
    assert report["dropped"]["bonds"] == 3


def test_parameters_stay_aligned_with_their_topology_rows():
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    topo, params = _system()
    new_topo, new_params, _ = _drop_ml_rows(topo, params, frozenset({0, 1, 2}))

    assert new_params.bond_k.shape[0] == new_topo.bonds.shape[0]
    assert new_params.angle_k.shape[0] == new_topo.angles.shape[0]
    assert new_params.torsion_k.shape[0] == new_topo.torsions.shape[0]
    assert new_params.improper_k.shape[0] == new_topo.impropers.shape[0]


def test_angles_torsions_and_impropers_are_filtered_too():
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    topo, params = _system()
    new_topo, new_params, _ = _drop_ml_rows(topo, params, frozenset({0, 1, 2}))

    np.testing.assert_array_equal(new_topo.angles, [[3, 4, 5], [6, 7, 8]])
    np.testing.assert_allclose(new_params.angle_k, [20.0, 30.0])
    np.testing.assert_array_equal(new_topo.torsions, [[3, 4, 5, 6]])
    np.testing.assert_array_equal(new_topo.impropers, [[4, 3, 5, 6]])


def test_empty_ml_region_keeps_everything():
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    topo, params = _system()
    new_topo, new_params, report = _drop_ml_rows(topo, params, frozenset())

    np.testing.assert_array_equal(new_topo.bonds, topo.bonds)
    np.testing.assert_array_equal(new_topo.angles, topo.angles)
    np.testing.assert_allclose(new_params.bond_k, params.bond_k)
    assert sum(report["dropped"].values()) == 0


def test_the_reactive_c_cl_bond_is_removed():
    """The specific failure this filtering exists to prevent.

    CGenFF gives CH3CL a harmonic C1-CL1 bond (k ~ 220 kcal/mol/A^2). If it
    survives alongside PhysNet, the chloride cannot leave and there is no SN2.
    """
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    # Solute atoms 0..8 (Cl, N, C, H*6); solvent water at 9, 10, 11.
    topo = _Topo(
        bonds=np.array([[2, 0], [2, 6], [9, 10], [9, 11]]),  # C-Cl, C-H, 2x O-H
        angles=np.array([[0, 2, 6], [10, 9, 11]]),
        torsions=np.empty((0, 4), dtype=int),
        impropers=np.empty((0, 4), dtype=int),
        cmap_atoms=np.empty((0, 8), dtype=int),
        cmap_map_idx=np.empty(0, dtype=int),
    )
    params = _Params(
        bond_k=np.array([220.0, 322.0, 450.0, 450.0]),
        bond_r0=np.array([1.78, 1.09, 0.96, 0.96]),
        angle_k=np.array([50.0, 55.0]),
        angle_theta0=np.array([1.9, 1.82]),
        torsion_k=np.empty(0),
        torsion_n=np.empty(0),
        torsion_gamma=np.empty(0),
        improper_k=np.empty(0),
        improper_n=np.empty(0),
        improper_gamma=np.empty(0),
    )
    new_topo, new_params, _ = _drop_ml_rows(topo, params, frozenset(range(9)))

    assert not any(set(row) >= {2, 0} for row in new_topo.bonds.tolist())
    assert 220.0 not in new_params.bond_k.tolist()
    # Water survives untouched.
    np.testing.assert_array_equal(new_topo.bonds, [[9, 10], [9, 11]])
    np.testing.assert_allclose(new_params.bond_k, [450.0, 450.0])


def test_urey_bradley_arrays_follow_the_angle_mask():
    """UB terms ride on angle rows, so a stale UB array would misalign."""
    from mmml.md.energy.terms.mm_bonded import _drop_ml_rows

    topo, params = _system()
    urey_k = np.array([100.0, 200.0, 300.0])
    _, _, report = _drop_ml_rows(topo, params, frozenset({0, 1, 2}))
    mask = report["angle_mask"]
    assert mask.shape == (3,)
    np.testing.assert_allclose(urey_k[mask], [200.0, 300.0])


def test_term_is_registered():
    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy.registry import available_terms, get_term
    from mmml.md.energy.terms.mm_bonded import MMBondedTerm

    assert "mm_bonded" in available_terms()
    assert get_term("mm_bonded") is MMBondedTerm


def test_missing_psf_is_reported_clearly():
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms.mm_bonded import MMBondedTerm
    from mmml.md.system import MolecularSystem

    system = MolecularSystem(
        R=np.zeros((3, 3)),
        Z=np.array([8, 1, 1]),
        box=None,
        mol_id=np.zeros(3, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="needs a PSF"):
        MMBondedTerm().make(system, EnergyContext())
