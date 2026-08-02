"""Component-level tests for :mod:`mmml.analysis.lattice_energy`.

``test_lattice_energy.py`` anchors the assembled Ewald sum on the rock-salt
Madelung constant. It does not touch the pieces that surround that sum -- the
minimum-image helper, the LJ combining rules, the analytic tail integral, the
rigid-body cell rescaling, the exclusion bookkeeping -- and those are where a
wrong factor produces a plausible number instead of a crash.

Each assertion below is derivable without running the code under test: an
analytic integral, a combining rule stated in the CHARMM parameter format, a
scaling law, or a hand-enumerated index set.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.analysis.lattice_energy import (
    GAS_CONSTANT_KCAL_MOL_K,
    GPA_A3_TO_KCAL_MOL,
    KCAL_MOL_TO_KJ_MOL,
    LatticeEnergyResult,
    MolecularCell,
    SublimationReference,
    _intramolecular_pairs,
    _lj_real_space,
    _lj_tail_correction,
    _mic_displacements,
    _pair_tables,
    _rigid_scaled_positions,
    lattice_shift_vectors,
    molecular_reach_A,
    periodic_coulomb_energy,
    sublimation_enthalpy_kcal_mol,
)
from mmml.models.cgenff_mm import COULOMB_CONSTANT, sigma_to_rmin_half


def _cell_of(positions, mol_id, *, cell, charges=None, type_idx=None, z=None):
    n = len(positions)
    n_mol = int(np.max(mol_id)) + 1
    return MolecularCell(
        positions=np.asarray(positions, dtype=float),
        atomic_numbers=np.asarray(z if z is not None else np.full(n, 6), dtype=int),
        cell=np.asarray(cell, dtype=float).reshape(3, 3),
        mol_id=np.asarray(mol_id, dtype=np.int64),
        type_idx=np.asarray(
            type_idx if type_idx is not None else np.zeros(n), dtype=np.int32
        ),
        charges=np.asarray(charges if charges is not None else np.zeros(n), dtype=float),
        residues=tuple(f"M{i}" for i in range(n_mol)),
    )


# --- minimum image ----------------------------------------------------------


def test_mic_maps_every_displacement_into_the_central_box():
    cell = np.diag([10.0, 12.0, 8.0])
    delta = np.random.default_rng(0).uniform(-40.0, 40.0, size=(200, 3))

    out = _mic_displacements(delta, cell)

    assert np.all(np.abs(out) <= np.array([5.0, 6.0, 4.0]) + 1e-9)
    # ...and differs from the input only by a whole lattice translation.
    n = (delta - out) @ np.linalg.inv(cell)
    assert np.allclose(n, np.round(n), atol=1e-9)


def test_mic_is_the_identity_inside_the_box():
    cell = np.diag([10.0, 10.0, 10.0])
    delta = np.array([[1.0, -2.0, 3.0]])
    assert np.allclose(_mic_displacements(delta, cell), delta)


def test_mic_handles_a_triclinic_cell():
    """A skewed cell is exactly where a naive per-axis wrap goes wrong."""
    cell = np.array([[10.0, 0.0, 0.0], [4.0, 9.0, 0.0], [1.0, 2.0, 8.0]])
    delta = np.random.default_rng(1).uniform(-30.0, 30.0, size=(100, 3))

    frac = _mic_displacements(delta, cell) @ np.linalg.inv(cell)

    assert np.all(np.abs(frac) <= 0.5 + 1e-9)


# --- lattice shift grid -----------------------------------------------------


def test_shift_grid_is_symmetric_under_negation():
    shifts = lattice_shift_vectors(np.diag([10.0] * 3), cutoff_A=10.0)
    assert shifts.shape == (27, 3)  # (2*ceil(10/10)+1)^3
    as_set = {tuple(np.round(s, 9)) for s in shifts}
    assert all(tuple(np.round(-np.array(s), 9)) in as_set for s in as_set)


def test_shift_count_follows_the_ceiling_rule():
    cell = np.diag([10.0] * 3)
    assert [len(lattice_shift_vectors(cell, c)) for c in (5.0, 15.0, 25.0)] == [
        27,
        125,
        343,
    ]


def test_molecular_reach_widens_the_grid():
    """Only centroids are guaranteed inside the cell, so atoms overhang by
    ``reach`` on each side and the span is ``cutoff + 2 * reach``."""
    cell = np.diag([10.0] * 3)
    assert len(lattice_shift_vectors(cell, 9.0, reach_A=0.0)) == 27
    assert len(lattice_shift_vectors(cell, 9.0, reach_A=3.0)) == 125  # ceil(15/10)=2


def test_shift_grid_uses_perpendicular_spacing_not_edge_length():
    """A sheared cell has a perpendicular spacing far below its edge length; an
    edge-length bound would silently generate too few images."""
    cell = np.array([[10.0, 0.0, 0.0], [9.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
    spacing_1 = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[0], cell[2]))
    assert spacing_1 == pytest.approx(4.0)
    assert np.linalg.norm(cell[1]) == pytest.approx(9.849, abs=1e-3)

    n_per_axis = np.round(
        lattice_shift_vectors(cell, cutoff_A=8.0) @ np.linalg.inv(cell)
    ).astype(int)

    assert n_per_axis[:, 1].max() == 2  # ceil(8/4), not ceil(8/9.85) == 1


def test_degenerate_cell_is_rejected():
    flat = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-positive cell volume"):
        lattice_shift_vectors(flat, 10.0)


# --- LJ combining rules and tail --------------------------------------------


def test_pair_tables_use_arithmetic_rmin_and_geometric_epsilon():
    """CHARMM combines Rmin/2 arithmetically and epsilon geometrically."""
    sigmas = np.array([3.0, 4.0])
    epsilons = np.array([0.04, 0.09])
    mcell = _cell_of(
        np.zeros((2, 3)), [0, 1], cell=np.diag([10.0] * 3), type_idx=[0, 1]
    )

    pair_rmin, pair_eps = _pair_tables(mcell, sigmas, epsilons)

    half = np.asarray(sigma_to_rmin_half(sigmas))
    assert pair_rmin[0, 1] == pytest.approx(half[0] + half[1])
    assert pair_rmin[0, 0] == pytest.approx(2.0 * half[0])
    assert pair_eps[0, 1] == pytest.approx(np.sqrt(0.04 * 0.09))
    assert pair_eps[0, 0] == pytest.approx(0.04)


def test_lj_tail_matches_the_analytic_integral():
    """``E = (2 pi / V) sum_ij eps (Rmin^12 / 9 rc^9 - 2 Rmin^6 / 3 rc^3)``."""
    mcell = _cell_of(np.zeros((2, 3)), [0, 1], cell=np.diag([10.0] * 3))
    rmin = np.full((2, 2), 3.5)
    eps = np.full((2, 2), 0.07)
    rc = 12.0

    expected = (
        2.0
        * np.pi
        / 1000.0
        * np.sum(eps * (rmin**12 / (9.0 * rc**9) - 2.0 * rmin**6 / (3.0 * rc**3)))
    )
    assert _lj_tail_correction(mcell, rmin, eps, rc) == pytest.approx(expected, rel=1e-12)


def test_lj_tail_is_attractive_and_decays_as_rc_cubed():
    mcell = _cell_of(np.zeros((4, 3)), [0, 1, 2, 3], cell=np.diag([20.0] * 3))
    rmin = np.full((4, 4), 3.5)
    eps = np.full((4, 4), 0.07)

    near = _lj_tail_correction(mcell, rmin, eps, 10.0)
    far = _lj_tail_correction(mcell, rmin, eps, 20.0)

    assert near < 0.0 and far < 0.0
    # The r^-6 term dominates at these cutoffs, so doubling rc cuts |E| ~8x.
    assert abs(near / far) == pytest.approx(8.0, rel=0.02)


def test_lj_tail_scales_inversely_with_volume():
    rmin = np.full((2, 2), 3.5)
    eps = np.full((2, 2), 0.07)
    small = _lj_tail_correction(
        _cell_of(np.zeros((2, 3)), [0, 1], cell=np.diag([10.0] * 3)), rmin, eps, 12.0
    )
    big = _lj_tail_correction(
        _cell_of(np.zeros((2, 3)), [0, 1], cell=np.diag([20.0] * 3)), rmin, eps, 12.0
    )
    assert small == pytest.approx(8.0 * big, rel=1e-12)


def test_lj_tail_vanishes_for_zero_epsilon():
    mcell = _cell_of(np.zeros((2, 3)), [0, 1], cell=np.diag([10.0] * 3))
    assert _lj_tail_correction(
        mcell, np.full((2, 2), 3.5), np.zeros((2, 2)), 12.0
    ) == pytest.approx(0.0)


# --- explicit LJ lattice sum ------------------------------------------------


def test_lj_real_space_matches_a_hand_written_pair_sum():
    """Two atoms in a box big enough that only the home cell contributes."""
    cell = np.diag([40.0, 40.0, 40.0])
    r = 4.0
    mcell = _cell_of([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], [0, 1], cell=cell)
    rmin = np.full((2, 2), 3.5)
    eps = np.full((2, 2), 0.07)
    shifts = np.zeros((1, 3))

    got = _lj_real_space(mcell, rmin, eps, shifts, cutoff_A=12.0)

    x = (3.5 / r) ** 6
    assert got == pytest.approx(0.07 * (x * x - 2.0 * x), rel=1e-12)


def test_lj_real_space_excludes_intramolecular_pairs_in_the_home_cell():
    """The same two atoms, now declared to be one molecule, must not interact."""
    cell = np.diag([40.0] * 3)
    coords = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    rmin = np.full((2, 2), 3.5)
    eps = np.full((2, 2), 0.07)
    shifts = np.zeros((1, 3))

    same = _lj_real_space(_cell_of(coords, [0, 0], cell=cell), rmin, eps, shifts, 12.0)
    assert same == pytest.approx(0.0)


def test_lj_real_space_respects_the_cutoff():
    cell = np.diag([40.0] * 3)
    mcell = _cell_of([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], [0, 1], cell=cell)
    rmin = np.full((2, 2), 3.5)
    eps = np.full((2, 2), 0.07)
    shifts = np.zeros((1, 3))

    assert _lj_real_space(mcell, rmin, eps, shifts, 8.0) == pytest.approx(0.0)
    assert _lj_real_space(mcell, rmin, eps, shifts, 10.0) != 0.0


def test_lj_self_image_interaction_is_counted_outside_the_home_cell():
    """A lone atom still interacts with its own periodic images -- that term is
    lattice energy, not an intramolecular one."""
    cell = np.diag([6.0, 40.0, 40.0])
    mcell = _cell_of([[0.0, 0.0, 0.0]], [0], cell=cell)
    rmin = np.full((1, 1), 3.5)
    eps = np.full((1, 1), 0.07)
    shifts = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [-6.0, 0.0, 0.0]])

    got = _lj_real_space(mcell, rmin, eps, shifts, cutoff_A=12.0)

    x = (3.5 / 6.0) ** 6
    single = 0.07 * (x * x - 2.0 * x)
    # Two images, each counted with the 1/2 double-counting factor.
    assert got == pytest.approx(single, rel=1e-12)


# --- Ewald exclusions -------------------------------------------------------


def test_excluded_pairs_contribute_exactly_zero_not_merely_screened():
    """Dropping a pair from the real-space sum is only half the job; without the
    matching reciprocal correction the pair keeps an ``erf(alpha r)/r`` residue."""
    cell = np.eye(3) * 6.0
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 3.0, 3.0], [4.0, 3.0, 3.0]])
    q = np.array([1.0, -1.0, 1.0, -1.0])

    with_excl, _, _ = periodic_coulomb_energy(
        pos, q, cell, cutoff_A=12.0, excluded_pairs=np.array([[0, 1], [2, 3]])
    )
    without, _, _ = periodic_coulomb_energy(pos, q, cell, cutoff_A=12.0)

    # Removing them must remove exactly the bare 1/r energy of each pair.
    bare_pair = COULOMB_CONSTANT * (1.0 * -1.0) / 1.0
    assert with_excl - without == pytest.approx(-2.0 * bare_pair, rel=1e-4)


def test_ewald_energy_is_quadratic_in_the_charges():
    cell = np.eye(3) * 6.0
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    q = np.array([1.0, -1.0, -1.0, 1.0])

    e_1, _, _ = periodic_coulomb_energy(pos, q, cell, cutoff_A=12.0)
    e_2, _, _ = periodic_coulomb_energy(pos, 2.0 * q, cell, cutoff_A=12.0)

    assert e_2 == pytest.approx(4.0 * e_1, rel=1e-9)


def test_intramolecular_pairs_enumerates_each_within_molecule_pair_once():
    pairs = _intramolecular_pairs(np.array([0, 0, 0, 1, 1]))
    assert {tuple(p) for p in pairs} == {(0, 1), (0, 2), (1, 2), (3, 4)}


def test_intramolecular_pairs_is_empty_when_every_atom_is_its_own_molecule():
    assert len(_intramolecular_pairs(np.arange(5))) == 0


# --- geometry bookkeeping ---------------------------------------------------


def test_molecular_reach_is_the_largest_atom_to_centroid_distance():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # Molecule 0 centroid sits at x=1 (reach 1.0); molecule 1 is a lone atom.
    assert molecular_reach_A(pos, np.array([0, 0, 1])) == pytest.approx(1.0)


def test_molecular_reach_of_single_atoms_is_zero():
    assert molecular_reach_A(np.random.default_rng(2).normal(size=(4, 3)), np.arange(4)) == 0.0


def test_rigid_scaling_preserves_internal_geometry_and_fractional_centroids():
    cell = np.diag([10.0] * 3)
    new_cell = np.diag([12.0, 10.0, 10.0])
    pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [7.0, 5.0, 5.0]])
    mol_id = np.array([0, 0, 1])

    out = _rigid_scaled_positions(pos, mol_id, cell, new_cell)

    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(1.0)
    old_frac = pos[:2].mean(axis=0) @ np.linalg.inv(cell)
    new_frac = out[:2].mean(axis=0) @ np.linalg.inv(new_cell)
    assert np.allclose(old_frac, new_frac)


def test_rigid_scaling_is_the_identity_for_an_unchanged_cell():
    cell = np.diag([10.0] * 3)
    pos = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    assert np.allclose(_rigid_scaled_positions(pos, np.zeros(2, int), cell, cell), pos)


# --- cell properties --------------------------------------------------------


def test_volume_is_the_determinant_for_a_triclinic_cell():
    cell = np.array([[10.0, 0.0, 0.0], [4.0, 9.0, 0.0], [1.0, 2.0, 8.0]])
    mcell = _cell_of(np.zeros((1, 3)), [0], cell=cell)
    assert mcell.volume_A3 == pytest.approx(10.0 * 9.0 * 8.0)


def test_density_uses_ase_masses_by_default():
    from ase.data import atomic_masses

    z = np.array([8, 1, 1])
    mcell = _cell_of(np.zeros((3, 3)), [0, 0, 0], cell=np.diag([10.0] * 3), z=z)
    assert mcell.n_molecules == 1
    assert mcell.density_g_cm3() == pytest.approx(
        float(np.sum(atomic_masses[z])) / 1000.0 / 0.6022140857
    )


def test_density_accepts_explicit_masses():
    mcell = _cell_of(np.zeros((2, 3)), [0, 1], cell=np.diag([10.0] * 3))
    assert mcell.density_g_cm3(np.array([10.0, 10.0])) == pytest.approx(
        20.0 / 1000.0 / 0.6022140857
    )


# --- thermodynamic reporting ------------------------------------------------


def test_result_dataclass_delegates_to_the_module_function():
    result = LatticeEnergyResult(
        e_lattice=-11.0,
        e_lj=-8.0,
        e_lj_tail=-1.0,
        e_coulomb=-2.0,
        n_molecules=4,
        cutoff_A=12.0,
        n_lattice_shifts=27,
        n_kvectors=100,
        ewald_alpha=0.3,
        density_g_cm3=1.2,
        cell_lengths_A=(9.17, 7.53, 21.25),
        residues=("ACO",) * 4,
    )
    assert result.sublimation_enthalpy(300.0) == pytest.approx(
        sublimation_enthalpy_kcal_mol(-11.0, 300.0)
    )


def test_unit_conversion_constants_match_their_definitions():
    n_a = 6.02214076e23
    assert KCAL_MOL_TO_KJ_MOL == pytest.approx(4.184)
    # 1 GPa acting through 1 A^3 per mole, in kcal/mol.
    assert GPA_A3_TO_KCAL_MOL == pytest.approx(1e9 * 1e-30 * n_a / 4184.0, rel=1e-6)
    # R = 8.314462618 J/(mol K) expressed in kcal/(mol K).
    assert GAS_CONSTANT_KCAL_MOL_K == pytest.approx(8.314462618 / 4184.0, rel=1e-6)


def test_sublimation_reference_sums_the_thermodynamic_cycle():
    ref = SublimationReference(
        dvap_h_kj_mol=31.3,
        dvap_h_temperature_K=178.5,
        dvap_h_source="test",
        dfus_h_kj_mol=5.77,
        dfus_h_temperature_K=178.5,
        dfus_h_source="test",
    )
    assert ref.dsub_h_kj_mol == pytest.approx(37.07)
    assert ref.dsub_h_kcal_mol == pytest.approx(37.07 / 4.184)
