"""Regression tests for CHARMM PSF atom ordering in the dimer scan campaign.

``evaluate_charmm_scan`` builds a PSF whose atoms are permuted relative to the ASE/scan
order (DCM and BENZ have non-identity permutations). It previously called
``sync_positions(geom.atoms.positions)`` with the *unpermuted* coordinates, so a
chlorine's position was written onto a hydrogen's nucleus. That collapses atoms onto one
another and yields a constant, enormous VDW term (~+4e5 kcal/mol for DCM-BENZ), silently
corrupting the CGenFF baseline for every pair whose permutation is not the identity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_dimer_scan_campaign.py"


@pytest.fixture(scope="module")
def scan():
    spec = importlib.util.spec_from_file_location("run_dimer_scan_campaign", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairwise(positions: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return np.sort(d[np.triu_indices(len(positions), k=1)])


def test_permutation_matches_psf_elements(scan):
    """Each permutation must map the ASE element order onto the PSF's declared elements."""
    geometries = scan._charmm_residue_geometries()
    for resname, perm in scan.CHARMM_PSF_PERMUTATION.items():
        source = scan._CHARMM_RESIDUE_SOURCE[resname]
        ase_z = np.asarray(scan.MOLECULES[source].get_atomic_numbers())
        _, _, psf_z = geometries[resname]
        assert np.array_equal(ase_z[perm], np.asarray(psf_z)), resname


def test_reorder_preserves_internal_geometry(scan):
    """Reordering is a permutation: the internal distance spectrum must be unchanged."""
    for resname in scan.CHARMM_PSF_PERMUTATION:
        source = scan._CHARMM_RESIDUE_SOURCE[resname]
        pos = np.asarray(scan.MOLECULES[source].positions)
        reordered = scan.charmm_reorder_fragment(pos, resname)
        assert np.allclose(_pairwise(pos), _pairwise(reordered))


def test_charmm_ordered_positions_does_not_collapse_atoms(scan):
    """The regression itself: PSF-ordered coordinates must not place atoms on top of
    each other, and must reproduce each monomer's own internal geometry."""
    for (label_a, label_b), cfg in scan.PAIR_SCAN_CONFIG.items():
        distances, _ = scan.build_pair_distance_grid(label_a, label_b)
        geom = next(iter(scan.make_oriented_scan_geometries(
            label_a, label_b, distances[-1:], [0.0]  # widest separation
        )))
        ordered = scan.charmm_ordered_positions(geom, label_a, label_b)

        assert ordered.shape == geom.atoms.positions.shape
        d = np.linalg.norm(ordered[:, None, :] - ordered[None, :, :], axis=-1)
        d = d[np.triu_indices(len(ordered), k=1)]
        assert d.min() > 0.5, f"{label_a}-{label_b}: atoms collapsed (min dist {d.min():.3f} A)"

        # Fragment-internal geometry must survive the reorder untouched.
        n_a = len(geom.fragments[0])
        src = np.asarray(geom.atoms.positions)
        assert np.allclose(_pairwise(ordered[:n_a]), _pairwise(src[list(geom.fragments[0])]))
        assert np.allclose(_pairwise(ordered[n_a:]), _pairwise(src[list(geom.fragments[1])]))


def test_rtf_bonds_are_bonded_distances(scan):
    """The load-bearing invariant: every CGenFF RTF bond must map to an actual bonded
    distance in the PSF-ordered coordinates.

    Element-only matching satisfies the element check above while still scrambling atoms
    *within* an element (which methyl H belongs to which carbon; benzene's cyclic order;
    methanol's hydroxyl vs methyl H). That leaves RTF bonds spanning 2.4-3.5 A, so
    CHARMM's 1-2/1-3 exclusions cover the wrong pairs and the intramolecular VDW
    explodes (+14,365 kcal/mol for acetone, +41,142 for benzene) -- or, for methanol,
    silently puts the +0.42 hydroxyl charge on a methyl hydrogen.
    """
    for resname, src in scan._CHARMM_RESIDUE_SOURCE.items():
        pos = scan.charmm_reorder_fragment(scan.MOLECULES[src].positions, resname)
        slot = {n: i for i, n in enumerate(scan.CHARMM_RESIDUE_ATOMS[resname])}
        for a, b in scan.CHARMM_RESIDUE_BONDS[resname]:
            d = float(np.linalg.norm(pos[slot[a]] - pos[slot[b]]))
            assert d < 1.8, f"{resname}: RTF bond {a}-{b} spans {d:.2f} A -- atoms mis-ordered"


def test_permutation_is_a_permutation(scan):
    for resname, perm in scan.CHARMM_PSF_PERMUTATION.items():
        assert sorted(perm) == list(range(len(perm))), resname
