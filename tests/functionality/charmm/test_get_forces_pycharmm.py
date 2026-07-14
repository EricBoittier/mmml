"""Regression: ``get_forces_pycharmm()`` returns forces, not coordinates.

The original implementation ran ``coor force sele all end`` and then read the main
coordinate set back with ``coor.get_positions()``. That lingo script does not write
forces into the main coordinate set, so the function silently returned the positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    charmm_positions_xyz_array,
    set_charmm_positions,
)
from tests.conftest import can_import_pycharmm

pytestmark = pytest.mark.skipif(
    not can_import_pycharmm(),
    reason="pycharmm / libcharmm not available",
)

# Central-difference step (Å): above CHARMM's energy resolution, small enough that
# O(h^2) truncation stays inside the tolerance.
FD_STEP = 1e-4


def _build_acetone() -> np.ndarray:
    """Build ACO from the CGENFF topology and return its minimized geometry.

    Uses ``setupRes`` rather than the committed ACO PDB: that fixture carries CHARMM's
    9999 placeholder coordinates, which give a ~1e65 kcal/mol energy and make finite
    differences meaningless.
    """
    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )

    setupRes.main("ACO")
    setupRes.generate_coordinates()
    reset_block()
    reset_block_no_internal()
    return charmm_positions_xyz_array()


def _energy_at(positions: np.ndarray) -> float:
    import pycharmm
    import pycharmm.energy as energy

    set_charmm_positions(positions)
    pycharmm.lingo.charmm_script("ENER")
    return float(energy.get_total())


def _fd_force(positions: np.ndarray, atom: int, comp: int) -> float:
    """Central-difference force ``-dE/dx`` for one (atom, component)."""
    plus = positions.copy()
    plus[atom, comp] += FD_STEP
    minus = positions.copy()
    minus[atom, comp] -= FD_STEP
    return -(_energy_at(plus) - _energy_at(minus)) / (2.0 * FD_STEP)


@pytest.fixture()
def perturbed_acetone(pycharmm_workdir) -> np.ndarray:
    """Acetone displaced off its minimum so the forces are unambiguously non-zero."""
    positions = _build_acetone()
    rng = np.random.default_rng(3)
    positions = positions + rng.normal(scale=0.05, size=positions.shape)
    set_charmm_positions(positions)
    return positions


def test_get_forces_pycharmm_returns_forces_not_positions(perturbed_acetone) -> None:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import get_forces_pycharmm

    positions = perturbed_acetone
    forces = np.asarray(get_forces_pycharmm(), dtype=float)

    assert forces.shape == positions.shape
    # The bug: the returned array was element-wise identical to the coordinates.
    assert not np.allclose(forces, positions), "get_forces_pycharmm() returned the positions"
    assert np.abs(forces).max() > 1e-6, "forces are all zero"
    # get_forces_pycharmm() must also leave the coordinates untouched.
    assert np.allclose(charmm_positions_xyz_array(), positions, atol=1e-8)


def test_get_forces_pycharmm_matches_finite_difference(perturbed_acetone) -> None:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import get_forces_pycharmm

    positions = perturbed_acetone
    forces = np.asarray(get_forces_pycharmm(), dtype=float)

    # Energy is well-conditioned here (a few kcal/mol), so the central difference is
    # trustworthy for every component.
    probes = [(a, c) for a in range(min(positions.shape[0], 3)) for c in range(3)]
    analytic = np.array([forces[a, c] for a, c in probes])
    expected = np.array([_fd_force(positions, a, c) for a, c in probes])

    assert np.allclose(analytic, expected, rtol=2e-2, atol=1e-2), (
        f"forces do not match -dE/dx\nanalytic:    {analytic}\nfinite-diff: {expected}"
    )
