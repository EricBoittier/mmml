"""A monoatomic residue must count as having resolved geometry.

``_has_resolved_geometry`` rejects a residue whose atoms all sit on one point,
which is the signature of an unresolved CHARMM internal-coordinate table. A
single-atom residue has exactly that signature for a legitimate reason -- there
is no internal geometry -- so the span test used to make every monoatomic
residue unbuildable: ``mmml liquid-box --composition AR1:500`` died with
"PyCHARMM make-res coordinate generation failed for residue 'AR1'". That blocks
the noble gases and the monoatomic ions (CLA/POT/SOD/LIT).
"""

import numpy as np
import pytest

from mmml.cli.run.md_pbc_suite.ase import _has_resolved_geometry


def test_single_atom_is_resolved():
    assert _has_resolved_geometry(np.zeros((1, 3)))
    assert _has_resolved_geometry(np.array([[1.5, -2.0, 3.25]]))


def test_single_atom_must_still_be_finite():
    assert not _has_resolved_geometry(np.array([[np.nan, 0.0, 0.0]]))
    assert not _has_resolved_geometry(np.array([[np.inf, 0.0, 0.0]]))


def test_polyatomic_collapsed_onto_one_point_is_still_rejected():
    # The case the guard exists for: an unresolved IC table.
    assert not _has_resolved_geometry(np.zeros((6, 3)))
    assert not _has_resolved_geometry(np.full((3, 3), 2.0))


def test_polyatomic_with_real_extent_is_resolved():
    water = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    assert _has_resolved_geometry(water)


def test_empty_is_rejected():
    assert not _has_resolved_geometry(np.zeros((0, 3)))


@pytest.mark.parametrize("n", [2, 3, 10])
def test_span_just_below_threshold_is_rejected(n):
    coords = np.zeros((n, 3))
    coords[-1, 0] = 1.0e-6
    assert not _has_resolved_geometry(coords)


# --- the same assumption, one layer up -------------------------------------
# `_monomer_geometry_is_3d` requires y and z spans >= 0.3 A, which a single
# atom can never satisfy. It sits behind the second failure:
#   RuntimeError: Monomer AR1 not 3D after minimization (spans x=0.00 ...)
from mmml.interfaces.pycharmmInterface.cluster_geometry import (  # noqa: E402
    _monomer_geometry_is_3d,
    ensure_monomer_3d_coords,
)


def test_monoatomic_monomer_counts_as_3d():
    assert _monomer_geometry_is_3d(np.zeros((1, 3)))
    assert _monomer_geometry_is_3d(np.array([[3.0, -1.0, 0.5]]))


def test_flat_polyatomic_is_still_rejected():
    # A collinear/planar IC build must still fail -- that is the guard's job.
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert not _monomer_geometry_is_3d(collinear)
    planar = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.5, 0.0]])
    assert not _monomer_geometry_is_3d(planar)


def test_genuinely_3d_polyatomic_passes():
    tetra = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    assert _monomer_geometry_is_3d(tetra)


def test_agrees_with_ensure_monomer_3d_coords_on_monoatomic():
    # ensure_monomer_3d_coords already short-circuits n < 2 by returning the
    # coords untouched; the predicate must agree that those coords are fine.
    one = np.array([[0.25, -0.5, 2.0]])
    assert np.allclose(ensure_monomer_3d_coords(one), one)
    assert _monomer_geometry_is_3d(ensure_monomer_3d_coords(one))
