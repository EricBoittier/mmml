"""Unit tests for the bundled dichloromethane crystal structures and their energetics.

Source of the structural reference numbers: M. Podsiadło, K. F. Dziubek and
A. Katrusiak, *Acta Crystallogr.* B **61**, 595 (2005),
doi:10.1107/S0108768105017374, coordinates via the Crystallography Open
Database. The ambient-pressure cell is Kawaguchi, Tanaka, Takeuchi & Watanabé,
*Bull. Chem. Soc. Jpn.* **46**, 62 (1973).

Two of these tests are the load-bearing ones. The first is that the hydrogens
have to be rebuilt before anything quantitative happens -- the deposited ones
are noisier than the effect under study and reverse the sign of the contact
trend. The second is the paper's own conclusion, that H...Cl and not Cl...Cl
dominates cohesion, which is a claim about energy and so is directly checkable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=".*crystal system.*")

CUTOFF_A = 12.0


@pytest.fixture(scope="module")
def phases():
    from mmml.analysis.dcm_crystal import DCM_CRYSTAL_PHASES

    return DCM_CRYSTAL_PHASES


@pytest.fixture(scope="module")
def cells():
    """Both structures with rebuilt hydrogens, read once."""
    from mmml.analysis.dcm_crystal import DCM_CRYSTAL_PHASES, read_dcm_phase

    return {key: read_dcm_phase(key, rebuild_hydrogens=True) for key in DCM_CRYSTAL_PHASES}


# --- the deposited structures -------------------------------------------------


def test_both_deposited_pressure_points_are_bundled(phases):
    assert set(phases) == {"pbcn_133gpa", "pbcn_163gpa"}
    for phase in phases.values():
        assert phase.cif_path().is_file(), f"missing CIF for {phase.key}"


@pytest.mark.parametrize(
    "key,lengths,volume,pressure",
    [
        ("pbcn_133gpa", (3.984, 7.863, 9.357), 293.12, 1.33),
        ("pbcn_163gpa", (3.924, 7.793, 9.335), 285.46, 1.63),
    ],
)
def test_deposited_cell_matches_the_paper(key, lengths, volume, pressure):
    from mmml.analysis.dcm_crystal import dcm_phase, read_dcm_phase

    phase = dcm_phase(key)
    assert phase.space_group_number == 60  # Pbcn
    assert phase.z == 4
    assert phase.pressure_GPa == pytest.approx(pressure)

    atoms = read_dcm_phase(key)
    assert atoms.cell.lengths() == pytest.approx(lengths, abs=1e-3)
    assert atoms.get_volume() == pytest.approx(volume, rel=2e-3)


def test_symmetry_expands_to_four_whole_molecules(cells):
    from mmml.analysis.lattice_energy import unwrap_molecules

    for key, atoms in cells.items():
        z = atoms.get_atomic_numbers()
        assert len(atoms) == 20, key
        mol_id, _ = unwrap_molecules(atoms.get_positions(), z, atoms.cell.array)
        assert mol_id.max() + 1 == 4, key
        for m in range(4):
            block = sorted(z[mol_id == m])
            assert block == [1, 1, 6, 17, 17], f"{key} molecule {m} is not CH2Cl2"


def test_unknown_phase_names_the_alternatives():
    from mmml.analysis.dcm_crystal import dcm_phase

    with pytest.raises(KeyError, match="pbcn_133gpa"):
        dcm_phase("ambient")


def test_the_ambient_reference_carries_a_cell_but_no_structure():
    """The 1973 structure is a comparison target, not something to build from."""
    from mmml.analysis.dcm_crystal import KAWAGUCHI_AMBIENT_CELL

    ref = KAWAGUCHI_AMBIENT_CELL
    assert ref.cell_lengths_A == pytest.approx((4.249, 8.138, 9.492))
    assert ref.cell_volume_A3 == pytest.approx(328.2, rel=1e-3)
    assert not hasattr(ref, "cif_path")
    # Both deposited structures are compressed relative to it, which is the
    # reason the relaxation step exists at all.
    from mmml.analysis.dcm_crystal import DCM_CRYSTAL_PHASES

    for phase in DCM_CRYSTAL_PHASES.values():
        assert phase.cell_volume_A3 < ref.cell_volume_A3


# --- the hydrogen problem ------------------------------------------------------


def test_deposited_hydrogens_reverse_the_compression_trend():
    """Why the rebuild is not optional.

    Volume falls 2.6% between the two pressure points, so contacts must shorten.
    With the deposited hydrogens the shortest H...Cl contact appears to grow.
    """
    from mmml.analysis.dcm_crystal import h_cl_contacts, read_dcm_phase

    lo = h_cl_contacts(read_dcm_phase("pbcn_133gpa"), rebuild_hydrogens=False)
    hi = h_cl_contacts(read_dcm_phase("pbcn_163gpa"), rebuild_hydrogens=False)
    assert hi[0].distance_A > lo[0].distance_A


def test_rebuilt_hydrogens_restore_the_compression_trend(cells):
    from mmml.analysis.dcm_crystal import h_cl_contacts

    lo = h_cl_contacts(cells["pbcn_133gpa"], rebuild_hydrogens=False)
    hi = h_cl_contacts(cells["pbcn_163gpa"], rebuild_hydrogens=False)
    assert hi[0].distance_A < lo[0].distance_A


def test_rebuild_leaves_the_heavy_atoms_alone(cells):
    from mmml.analysis.dcm_crystal import read_dcm_phase

    deposited = read_dcm_phase("pbcn_133gpa")
    rebuilt = cells["pbcn_133gpa"]
    heavy = deposited.get_atomic_numbers() != 1
    assert rebuilt.get_positions()[heavy] == pytest.approx(
        deposited.get_positions()[heavy], abs=1e-9
    )
    assert rebuilt.cell.array == pytest.approx(deposited.cell.array)


def test_rebuild_produces_the_intended_ch2cl2_geometry(cells):
    from mmml.analysis.dcm_crystal import GAS_PHASE_CH_A, GAS_PHASE_HCH_DEG
    from mmml.analysis.lattice_energy import unwrap_molecules

    atoms = cells["pbcn_163gpa"]
    z = atoms.get_atomic_numbers()
    mol_id, positions = unwrap_molecules(atoms.get_positions(), z, atoms.cell.array)
    for m in range(int(mol_id.max()) + 1):
        sel = np.flatnonzero(mol_id == m)
        c = positions[sel[z[sel] == 6]][0]
        h = positions[sel[z[sel] == 1]]
        assert np.linalg.norm(h - c, axis=1) == pytest.approx(GAS_PHASE_CH_A, abs=1e-9)
        v1, v2 = h[0] - c, h[1] - c
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert np.degrees(np.arccos(cos)) == pytest.approx(GAS_PHASE_HCH_DEG, abs=1e-6)


def test_rebuild_refuses_a_molecule_that_is_not_ch2cl2():
    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.dcm_crystal import rebuild_methylene_hydrogens

    with pytest.raises(ValueError, match="not CH2Cl2"):
        rebuild_methylene_hydrogens(read_acetone_phase("pbca_150k"))


def test_normalizing_ch_length_alone_is_not_enough():
    """Distance normalisation does not fix a wrong hydrogen *direction*."""
    from mmml.analysis.crystal_contacts import normalize_hydrogen_positions
    from mmml.analysis.dcm_crystal import h_cl_contacts, read_dcm_phase

    lo = normalize_hydrogen_positions(read_dcm_phase("pbcn_133gpa"))
    hi = normalize_hydrogen_positions(read_dcm_phase("pbcn_163gpa"))
    shortest_lo = h_cl_contacts(lo, rebuild_hydrogens=False)[0].distance_A
    shortest_hi = h_cl_contacts(hi, rebuild_hydrogens=False)[0].distance_A
    assert shortest_hi > shortest_lo


def test_normalize_hydrogen_positions_sets_the_requested_bond_length():
    from mmml.analysis.crystal_contacts import normalize_hydrogen_positions
    from mmml.analysis.dcm_crystal import read_dcm_phase
    from mmml.analysis.lattice_energy import unwrap_molecules

    atoms = normalize_hydrogen_positions(read_dcm_phase("pbcn_133gpa"), target_A=1.1)
    z = atoms.get_atomic_numbers()
    mol_id, positions = unwrap_molecules(atoms.get_positions(), z, atoms.cell.array)
    for h in np.flatnonzero(z == 1):
        same = np.flatnonzero((mol_id == mol_id[h]) & (z != 1))
        d = np.linalg.norm(positions[same] - positions[h], axis=1)
        assert d.min() == pytest.approx(1.1, abs=1e-9)


# --- contacts ------------------------------------------------------------------


def test_the_shortest_halogen_contact_is_a_type_ii_sigma_hole_geometry(cells):
    from mmml.analysis.dcm_crystal import halogen_contacts

    for key, atoms in cells.items():
        closest = halogen_contacts(atoms)[0]
        assert closest.distance_A < 3.5, key  # inside the Bondi vdW sum
        assert closest.motif.startswith("II"), key
        assert closest.angle_deg > 150.0, key


def test_halogen_contacts_shorten_under_compression(cells):
    from mmml.analysis.dcm_crystal import halogen_contacts

    lo = halogen_contacts(cells["pbcn_133gpa"])[0].distance_A
    hi = halogen_contacts(cells["pbcn_163gpa"])[0].distance_A
    assert hi < lo


@pytest.mark.parametrize(
    "theta1,theta2,expected",
    [
        (170.0, 95.0, "II"),
        (95.0, 170.0, "II"),  # the classification is symmetric in its arguments
        (150.0, 150.0, "I"),
        (95.0, 100.0, "I"),
        (140.0, 100.0, "intermediate"),
    ],
)
def test_halogen_motif_classification(theta1, theta2, expected):
    from mmml.analysis.dcm_crystal import classify_halogen_motif

    assert classify_halogen_motif(theta1, theta2).startswith(expected)


# --- the paper's conclusion ----------------------------------------------------


@pytest.fixture(scope="module")
def decompositions(cells):
    from mmml.analysis.lattice_energy import decompose_lattice_energy_by_element_pair

    return {
        key: decompose_lattice_energy_by_element_pair(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.cell.array,
            cutoff_A=CUTOFF_A,
        )
        for key, atoms in cells.items()
    }


def test_h_cl_contacts_dominate_cohesion_as_published(decompositions):
    """Podsiadło et al.: cohesion is "dominated by H...Cl rather than Cl...Cl"."""
    for key, dec in decompositions.items():
        assert set(dec.dominant_contact()) == {"H", "Cl"}, key
        by = {frozenset(k): v for k, v in dec.by_contact.items()}
        h_cl = sum(by[frozenset({"H", "Cl"})][:2])
        cl_cl = sum(by[frozenset({"Cl"})][:2])
        assert h_cl < cl_cl < 0.0, key
        assert h_cl / dec.e_total > 0.5, key
        assert cl_cl / dec.e_total < 0.25, key


def test_the_halogen_contact_binds_by_dispersion_not_electrostatics(decompositions):
    """CGenFF has no sigma hole, so its Cl...Cl attraction is entirely dispersive."""
    for key, dec in decompositions.items():
        lj, coulomb, _ = dec.by_contact[("Cl", "Cl")]
        assert lj < 0.0, key
        assert coulomb > 0.0, key


def test_the_decomposition_reproduces_the_lattice_sum(cells, decompositions):
    """Buckets add up, and the dimer Coulomb sum agrees with Ewald."""
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    for key, atoms in cells.items():
        dec = decompositions[key]
        total = crystal_lattice_energy(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.cell.array,
            cutoff_A=CUTOFF_A,
        )
        # The decomposition uses a group-based cutoff and the reference an
        # atom-based one, so they differ by the pairs that straddle it.
        assert dec.e_lj == pytest.approx(total.e_lj, abs=0.05)
        assert dec.e_coulomb_direct == pytest.approx(total.e_coulomb, abs=0.05)
        assert abs(dec.coulomb_truncation_error) < 0.05


def test_an_atom_pair_split_would_have_been_meaningless(cells):
    """Why the decomposition is over molecule pairs.

    Splitting the same Coulomb sum by atom pair produces buckets an order of
    magnitude larger than the total, of both signs: the monopole terms cancel
    between element pairs and mean nothing individually.
    """
    from mmml.models.cgenff_mm import COULOMB_CONSTANT
    from mmml.analysis.lattice_energy import build_molecular_cell

    atoms = cells["pbcn_133gpa"]
    mcell, _, _ = build_molecular_cell(
        atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array
    )
    pos, q, z = mcell.positions, mcell.charges, mcell.atomic_numbers
    delta = pos[None, :, :] - pos[:, None, :]
    r = np.linalg.norm(delta, axis=-1)
    inter = mcell.mol_id[:, None] != mcell.mol_id[None, :]
    e = np.where(inter, COULOMB_CONSTANT * q[:, None] * q[None, :] / np.where(r > 0, r, 1), 0.0)
    h_h = 0.5 * e[(z[:, None] == 1) & (z[None, :] == 1)].sum() / mcell.n_molecules
    assert abs(h_h) > 10.0 * abs(e.sum() * 0.5 / mcell.n_molecules)


# --- energetics ----------------------------------------------------------------


def test_lattice_energy_is_bound_and_dispersion_dominated(cells):
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    for key, atoms in cells.items():
        r = crystal_lattice_energy(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.cell.array,
            cutoff_A=CUTOFF_A,
        )
        assert r.residues == ("DCM",) * 4, key
        assert r.e_lattice < 0.0, key
        assert r.e_lj + r.e_lj_tail < r.e_coulomb, key


def test_the_more_compressed_cell_is_less_bound(cells):
    """Past the minimum, squeezing costs energy. A basic sanity check."""
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    energies = {
        key: crystal_lattice_energy(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.cell.array,
            cutoff_A=CUTOFF_A,
        ).e_lattice
        for key, atoms in cells.items()
    }
    assert energies["pbcn_163gpa"] > energies["pbcn_133gpa"]


def test_lattice_energy_regression(cells):
    """Pins the absolute value so a silent change in typing or charges is caught."""
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    r = crystal_lattice_energy(
        cells["pbcn_133gpa"].get_positions(),
        cells["pbcn_133gpa"].get_atomic_numbers(),
        cells["pbcn_133gpa"].cell.array,
        cutoff_A=CUTOFF_A,
    )
    assert r.e_lattice == pytest.approx(-9.00, abs=0.05)


# --- relaxation ----------------------------------------------------------------


@pytest.fixture(scope="module")
def relaxations(cells):
    from mmml.analysis.lattice_energy import relax_cell_lengths

    atoms = cells["pbcn_133gpa"]
    return {
        p: relax_cell_lengths(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.cell.array,
            pressure_GPa=p,
            cutoff_A=CUTOFF_A,
        )
        for p in (0.0, 1.33, 1.63)
    }


def test_relaxing_at_the_measured_pressures_reproduces_the_measured_volumes(
    phases, relaxations
):
    """The real test of the relaxation: the answers are already known."""
    for phase in phases.values():
        relaxed = relaxations[phase.pressure_GPa]
        assert relaxed.converged
        error = abs(relaxed.volume_A3 - phase.cell_volume_A3) / phase.cell_volume_A3
        assert error < 0.03, f"{phase.key}: {100 * error:.1f}% volume error"


def test_relaxing_to_ambient_pressure_approaches_the_1973_cell(relaxations):
    from mmml.analysis.dcm_crystal import KAWAGUCHI_AMBIENT_CELL

    relaxed = relaxations[0.0]
    ref = KAWAGUCHI_AMBIENT_CELL
    # Static relaxation versus a 153 K measurement, with molecular orientation
    # frozen: within 10% of the volume, and smaller, is what to expect.
    assert relaxed.volume_A3 < ref.cell_volume_A3
    assert relaxed.volume_A3 == pytest.approx(ref.cell_volume_A3, rel=0.10)


def test_higher_pressure_gives_a_smaller_cell_and_a_weaker_crystal(relaxations):
    volumes = [relaxations[p].volume_A3 for p in (0.0, 1.33, 1.63)]
    energies = [relaxations[p].e_lattice for p in (0.0, 1.33, 1.63)]
    assert volumes == sorted(volumes, reverse=True)
    assert energies == sorted(energies)


def test_relaxation_keeps_molecules_rigid_and_the_cell_orthorhombic(cells, relaxations):
    from mmml.analysis.lattice_energy import unwrap_molecules

    atoms = cells["pbcn_133gpa"]
    relaxed = relaxations[0.0]
    angles = np.degrees(
        [
            np.arccos(
                np.dot(relaxed.cell[i], relaxed.cell[j])
                / (np.linalg.norm(relaxed.cell[i]) * np.linalg.norm(relaxed.cell[j]))
            )
            for i, j in ((1, 2), (0, 2), (0, 1))
        ]
    )
    assert angles == pytest.approx([90.0, 90.0, 90.0], abs=1e-8)

    z = atoms.get_atomic_numbers()
    mol_id, before = unwrap_molecules(atoms.get_positions(), z, atoms.cell.array)
    after = relaxed.positions
    for m in range(int(mol_id.max()) + 1):
        sel = mol_id == m
        d_before = np.linalg.norm(before[sel] - before[sel].mean(axis=0), axis=1)
        d_after = np.linalg.norm(after[sel] - after[sel].mean(axis=0), axis=1)
        assert d_after == pytest.approx(d_before, abs=1e-9)


def test_relaxation_lowers_the_energy_it_started_from(relaxations):
    relaxed = relaxations[0.0]
    assert relaxed.e_lattice < relaxed.e_lattice_initial


# --- sublimation ----------------------------------------------------------------


def test_relaxed_sublimation_enthalpy_matches_the_thermodynamic_cycle(relaxations):
    from mmml.analysis.dcm_crystal import DCM_SUBLIMATION_REFERENCE
    from mmml.analysis.lattice_energy import KCAL_MOL_TO_KJ_MOL

    reference = DCM_SUBLIMATION_REFERENCE
    predicted = (
        relaxations[0.0].sublimation_enthalpy(reference.dfus_h_temperature_K)
        * KCAL_MOL_TO_KJ_MOL
    )
    assert predicted == pytest.approx(reference.dsub_h_kj_mol, rel=0.15)


def test_relaxing_improves_agreement_with_experiment(cells, relaxations):
    """The reason the relaxation step exists.

    A structure measured at 1.33 GPa is compressed onto its repulsive wall, so
    its static energy underestimates cohesion.
    """
    from mmml.analysis.dcm_crystal import DCM_SUBLIMATION_REFERENCE
    from mmml.analysis.lattice_energy import (
        KCAL_MOL_TO_KJ_MOL,
        crystal_lattice_energy,
        sublimation_enthalpy_kcal_mol,
    )

    reference = DCM_SUBLIMATION_REFERENCE
    t = reference.dfus_h_temperature_K
    atoms = cells["pbcn_133gpa"]
    deposited = crystal_lattice_energy(
        atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=CUTOFF_A
    )
    as_is = sublimation_enthalpy_kcal_mol(deposited.e_lattice, t) * KCAL_MOL_TO_KJ_MOL
    relaxed = relaxations[0.0].sublimation_enthalpy(t) * KCAL_MOL_TO_KJ_MOL
    assert abs(relaxed - reference.dsub_h_kj_mol) < abs(as_is - reference.dsub_h_kj_mol)


def test_the_experimental_reference_is_a_cycle_not_a_measurement():
    from mmml.analysis.dcm_crystal import DCM_SUBLIMATION_REFERENCE

    ref = DCM_SUBLIMATION_REFERENCE
    assert ref.dsub_h_kj_mol == pytest.approx(ref.dvap_h_kj_mol + ref.dfus_h_kj_mol)
    assert ref.dsub_h_kcal_mol == pytest.approx(ref.dsub_h_kj_mol / 4.184)
    for source in (ref.dvap_h_source, ref.dfus_h_source):
        assert "NIST" in source


# --- learned LJ scales ----------------------------------------------------------


def test_learned_lj_scales_change_the_lattice_energy(cells):
    """Sublimation enthalpy is an observable hybrid training never sees."""
    from mmml.analysis.lattice_energy import crystal_lattice_energy
    from mmml.data.cgenff_dataset import load_reference

    atoms = cells["pbcn_133gpa"]
    n_types = len(load_reference().sigmas)
    weakened = crystal_lattice_energy(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        atoms.cell.array,
        cutoff_A=CUTOFF_A,
        epsilon_scale=np.full(n_types, 0.5),
    )
    baseline = crystal_lattice_energy(
        atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=CUTOFF_A
    )
    assert weakened.e_lattice > baseline.e_lattice
