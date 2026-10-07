"""Nosé–Klein cell vs the crystallographic embedding."""

from __future__ import annotations

import numpy as np
import pytest

from karml.interfaces.pycharmmInterface.crystal_cell import (
    CHARMM_ANGLE_RSMALL_DEG,
    cell_metric,
    charmm_symmetric_cell,
    crystallographic_cell,
)


def test_orthogonal_cell_is_diagonal() -> None:
    h = charmm_symmetric_cell(10.0, 11.0, 12.0, 90.0, 90.0, 90.0)
    np.testing.assert_allclose(h, np.diag([10.0, 11.0, 12.0]), atol=1e-10)


def test_sheared_cells_share_metric_and_differ_as_matrices() -> None:
    lengths = (20.0, 21.0, 22.0)
    angles = (80.0, 85.0, 75.0)
    h_sym = charmm_symmetric_cell(*lengths, *angles)
    h_cryst = crystallographic_cell(*lengths, *angles)
    np.testing.assert_allclose(cell_metric(h_sym), cell_metric(h_cryst), atol=1e-8)
    assert not np.allclose(h_sym, h_cryst, atol=1e-4)
    assert np.allclose(h_sym, h_sym.T, atol=1e-12)


def test_genten_snaps_only_inside_rsmall() -> None:
    exact = charmm_symmetric_cell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    inside = charmm_symmetric_cell(
        10.0, 10.0, 10.0, 90.0, 90.0, 90.0 + 0.1 * CHARMM_ANGLE_RSMALL_DEG
    )
    np.testing.assert_allclose(inside, exact, atol=1e-12)
    # Finite-difference shear is ~0.003°, above both 1e-4 and 1e-10.
    sheared = charmm_symmetric_cell(10.0, 10.0, 10.0, 90.0, 90.0, 90.003)
    assert abs(float(sheared[0, 1])) > 1e-4


def test_fd_sized_shear_keeps_the_box_metric() -> None:
    length = 28.868571
    eps = 3.0e-5
    strain = np.eye(3)
    strain[0, 1] = strain[1, 0] = eps
    box = np.diag([length, length, length]) @ strain
    a_v, b_v, c_v = box[:, 0], box[:, 1], box[:, 2]

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    lengths = tuple(float(np.linalg.norm(v)) for v in (a_v, b_v, c_v))
    angles = (ang(b_v, c_v), ang(a_v, c_v), ang(a_v, b_v))
    assert abs(angles[2] - 90.0) > 1.0e-4
    h = charmm_symmetric_cell(*lengths, *angles)
    np.testing.assert_allclose(cell_metric(h), cell_metric(box), atol=1e-8)
    assert abs(float(h[0, 1])) > 1e-6


def test_inconsistent_angles_are_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent cell"):
        charmm_symmetric_cell(10.0, 10.0, 10.0, 10.0, 10.0, 170.0)
