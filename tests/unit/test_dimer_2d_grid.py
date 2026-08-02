"""The 2D dimer grid must be geometrically what it claims to be.

The grid gates condensed-phase work: it is scanned to check the hybrid dimer
surface for the spurious attractive well implied by the bulk-water result
(99.3 kcal/mol per molecule released on relaxation). A grid with distorted
monomers or a wrong separation would make any conclusion drawn from the surface
meaningless, so the geometry is pinned here.
"""

import numpy as np
import pytest

from scripts.make_dimer_2d_grid import (
    TIP3_ANGLE_DEG,
    TIP3_R_OH,
    build_grid,
    com,
    tip3_monomer,
)


@pytest.fixture(scope="module")
def grid():
    r = np.linspace(2.5, 7.0, 9)
    th = np.linspace(0.0, 360.0, 8, endpoint=False)
    frames, z = build_grid(r, th)
    return frames, z, r, th


def test_monomer_matches_tip3_geometry():
    xyz, z = tip3_monomer()
    assert list(z) == [8, 1, 1]
    for h in (1, 2):
        assert np.linalg.norm(xyz[h] - xyz[0]) == pytest.approx(TIP3_R_OH, abs=1e-12)
    v1, v2 = xyz[1] - xyz[0], xyz[2] - xyz[0]
    ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
    assert ang == pytest.approx(TIP3_ANGLE_DEG, abs=1e-9)


def test_monomers_stay_rigid_across_the_whole_grid(grid):
    frames, _, _, _ = grid
    for f in frames:
        for o, h1, h2 in ((0, 1, 2), (3, 4, 5)):
            assert np.linalg.norm(f[h1] - f[o]) == pytest.approx(TIP3_R_OH, abs=1e-9)
            assert np.linalg.norm(f[h2] - f[o]) == pytest.approx(TIP3_R_OH, abs=1e-9)
            v1, v2 = f[h1] - f[o], f[h2] - f[o]
            ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
            assert ang == pytest.approx(TIP3_ANGLE_DEG, abs=1e-7)


def test_com_separation_equals_the_requested_R(grid):
    frames, z, r_values, th = grid
    z1 = z[:3]
    k = 0
    for r in r_values:
        for _ in th:
            f = frames[k]
            d = np.linalg.norm(com(f[3:], z1) - com(f[:3], z1))
            assert d == pytest.approx(float(r), abs=1e-9)
            k += 1


def test_rotation_actually_changes_the_geometry(grid):
    """theta must move atoms -- a broken rotation would give a 1D scan."""
    frames, _, r_values, th = grid
    n_th = len(th)
    first_shell = frames[:n_th]          # same R, different theta
    spread = max(
        float(np.abs(first_shell[i] - first_shell[0]).max()) for i in range(1, n_th)
    )
    assert spread > 0.5, "theta does not move the atoms"


def test_no_two_atoms_coincide(grid):
    frames, _, _, _ = grid
    for f in frames:
        d = np.linalg.norm(f[:3, None, :] - f[None, 3:, :], axis=-1)
        assert d.min() > 0.3
