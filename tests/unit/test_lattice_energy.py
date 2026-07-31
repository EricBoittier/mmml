"""Unit tests for the periodic lattice-energy machinery.

The Ewald and lattice-sum code here is the reason the acetone workflow does not
go through the cubic-only MD paths, so it needs its own correctness anchor
rather than agreement with the rest of the codebase. The anchor is the Madelung
constant of rock salt, which is known analytically to as many digits as anyone
needs, and is checked on both a cubic cell and a deliberately non-cubic one.
"""

from __future__ import annotations

import numpy as np
import pytest

# Rock salt, from the standard lattice-sum literature.
NACL_MADELUNG = 1.7475645946331821
COULOMB_KCAL = 332.063711


def _nacl_cell(r0: float = 2.82, reps: tuple[int, int, int] = (1, 1, 1)):
    """Conventional cubic NaCl cell (4 formula units), optionally tiled."""
    a = 2.0 * r0
    fcc = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    base = np.vstack([fcc, fcc + [0.5, 0.0, 0.0]]) * a
    charges = np.array([1.0] * 4 + [-1.0] * 4)

    positions = []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                positions.append(base + np.array([i, j, k]) * a)
    cell = np.diag([a * reps[0], a * reps[1], a * reps[2]])
    return np.vstack(positions), np.tile(charges, int(np.prod(reps))), cell


def test_ewald_reproduces_the_nacl_madelung_constant():
    from mmml.analysis.lattice_energy import periodic_coulomb_energy

    r0 = 2.82
    positions, charges, cell = _nacl_cell(r0)
    expected = 4 * (-NACL_MADELUNG * COULOMB_KCAL / r0)

    energy, _, _ = periodic_coulomb_energy(positions, charges, cell, cutoff_A=10.0)
    assert energy == pytest.approx(expected, rel=1e-5)


def test_ewald_answer_does_not_depend_on_the_real_reciprocal_split():
    """The cutoff moves work between real and reciprocal space, nothing else.

    This is the property that makes a truncated Coulomb sum unnecessary, and the
    one that fails first if the self or exclusion term is wrong.
    """
    from mmml.analysis.lattice_energy import periodic_coulomb_energy

    positions, charges, cell = _nacl_cell()
    energies = [
        periodic_coulomb_energy(positions, charges, cell, cutoff_A=rc)[0]
        for rc in (8.0, 10.0, 12.0, 14.0)
    ]
    assert max(energies) - min(energies) == pytest.approx(0.0, abs=1e-2)


def test_ewald_is_correct_for_a_non_cubic_cell():
    """A 1x2x3 supercell holds six times the content, so six times the energy.

    The whole point of this module is that it carries a full cell rather than
    one side length; a cubic-only reduction would fail here by construction.
    """
    from mmml.analysis.lattice_energy import periodic_coulomb_energy

    r0 = 2.82
    unit_pos, unit_q, unit_cell = _nacl_cell(r0)
    super_pos, super_q, super_cell = _nacl_cell(r0, reps=(1, 2, 3))

    lengths = np.linalg.norm(super_cell, axis=1)
    assert len({round(float(x), 6) for x in lengths}) == 3, "supercell must be non-cubic"

    unit_energy, _, _ = periodic_coulomb_energy(unit_pos, unit_q, unit_cell, cutoff_A=10.0)
    super_energy, _, _ = periodic_coulomb_energy(super_pos, super_q, super_cell, cutoff_A=10.0)
    assert super_energy == pytest.approx(6.0 * unit_energy, rel=1e-5)

    expected = 24 * (-NACL_MADELUNG * COULOMB_KCAL / r0)
    assert super_energy == pytest.approx(expected, rel=1e-5)


def test_ewald_refuses_a_charged_cell():
    """Without a neutralising background the reciprocal sum diverges at k -> 0."""
    from mmml.analysis.lattice_energy import periodic_coulomb_energy

    positions, charges, cell = _nacl_cell()
    charges = charges.copy()
    charges[0] = 2.0
    with pytest.raises(ValueError, match="neutral cell"):
        periodic_coulomb_energy(positions, charges, cell, cutoff_A=10.0)


def test_unwrap_rebuilds_a_molecule_split_across_a_cell_face():
    """Molecules straddling a boundary must come back whole, not as fragments."""
    from mmml.analysis.lattice_energy import unwrap_molecules

    cell = np.diag([10.0, 12.0, 14.0])
    # A water-like triatomic sitting on the x face: O just inside, H just outside
    # and therefore wrapped round to the far side.
    positions = np.array(
        [[9.8, 5.0, 5.0], [0.15, 5.0, 5.0], [9.3, 5.9, 5.0]]
    )
    z = np.array([8, 1, 1])

    mol_id, unwrapped = unwrap_molecules(positions, z, cell)
    assert set(mol_id) == {0}, "the wrapped hydrogen should not become its own molecule"

    bonds = np.linalg.norm(unwrapped[1:] - unwrapped[0], axis=1)
    assert bonds.max() < 1.5, f"molecule was not made contiguous: bonds {bonds}"


def test_unwrap_puts_every_centroid_inside_the_cell():
    """The lattice-shift bound assumes centroids are wrapped; check they are."""
    from mmml.analysis.lattice_energy import unwrap_molecules

    cell = np.diag([10.0, 12.0, 14.0])
    positions = np.array([[9.8, 5.0, 5.0], [0.15, 5.0, 5.0], [9.3, 5.9, 5.0]])
    mol_id, unwrapped = unwrap_molecules(positions, np.array([8, 1, 1]), cell)

    centroid = unwrapped[mol_id == 0].mean(axis=0)
    frac = centroid @ np.linalg.inv(cell)
    assert np.all(frac >= 0.0) and np.all(frac < 1.0), f"centroid escaped the cell: {frac}"


def test_lattice_shift_vectors_grow_with_cutoff_and_include_the_home_cell():
    from mmml.analysis.lattice_energy import lattice_shift_vectors

    cell = np.diag([9.17, 7.53, 21.25])
    near = lattice_shift_vectors(cell, 6.0)
    far = lattice_shift_vectors(cell, 14.0)

    assert len(far) > len(near)
    for shifts in (near, far):
        assert np.any(np.all(shifts == 0.0, axis=1)), "home cell must be summed too"
    # The short c axis needs fewer images than the long one at fixed cutoff.
    assert len(np.unique(far[:, 2])) < len(np.unique(far[:, 1]))


def test_sublimation_enthalpy_is_minus_lattice_energy_less_2rt():
    from mmml.analysis.lattice_energy import (
        GAS_CONSTANT_KCAL_MOL_K,
        sublimation_enthalpy_kcal_mol,
    )

    assert sublimation_enthalpy_kcal_mol(-10.0, 0.0) == pytest.approx(10.0)
    assert sublimation_enthalpy_kcal_mol(-10.0, 150.0) == pytest.approx(
        10.0 - 2.0 * GAS_CONSTANT_KCAL_MOL_K * 150.0
    )
    # Warmer crystals sublime with a smaller enthalpy at fixed lattice energy.
    assert sublimation_enthalpy_kcal_mol(-10.0, 300.0) < sublimation_enthalpy_kcal_mol(-10.0, 0.0)


def test_non_cubic_cell_warns_when_collapsed_to_a_cubic_side():
    """Averaging 9.17/7.53/21.25 into one number must not happen in silence."""
    from mmml.cli.run.md_stage_summary import cubic_box_side_from_cell

    cell = np.diag([9.17, 7.53, 21.25])
    with pytest.warns(RuntimeWarning, match="Non-cubic cell"):
        side = cubic_box_side_from_cell(cell)
    assert side == pytest.approx((9.17 + 7.53 + 21.25) / 3.0)

    # Suppressible for callers that only want a length scale.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cubic_box_side_from_cell(cell, warn_non_cubic=False)


def test_cubic_cell_does_not_warn():
    """Existing cubic workflows must stay quiet, float noise included."""
    import warnings

    from mmml.cli.run.md_stage_summary import cubic_box_side_from_cell

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cubic_box_side_from_cell(np.eye(3) * 28.0) == pytest.approx(28.0)
        assert cubic_box_side_from_cell(
            np.diag([28.0, 28.000001, 27.999999])
        ) == pytest.approx(28.0, rel=1e-6)
