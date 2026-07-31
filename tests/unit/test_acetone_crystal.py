"""Unit tests for the bundled acetone crystal structures and their energetics.

Source of every reference number below: D. R. Allan, S. J. Clark, R. M.
Ibberson, S. Parsons, C. R. Pulham and L. Sawyer, *Chem. Commun.* 1999, 751
(doi:10.1039/a900558g). Coordinates via the Crystallography Open Database.

The contact tests are the load-bearing ones. Lattice parameters and molecule
counts can all be right while the structure is still wrong -- a mis-applied
symmetry operator, molecules broken across a cell face -- and the published
intermolecular distances are what catch that, because they are what the authors
measured.
"""

from __future__ import annotations

import numpy as np
import pytest

# The paper quotes distances to three decimals; anything inside 0.01 A means we
# are looking at the same structure it describes.
CONTACT_TOLERANCE_A = 0.01


@pytest.fixture(scope="module")
def phases():
    from mmml.analysis.acetone_crystal import ACETONE_CRYSTAL_PHASES

    return ACETONE_CRYSTAL_PHASES


def test_all_five_published_phases_are_bundled(phases):
    assert set(phases) == {
        "pbca_5k",
        "pbca_110k",
        "pbca_150k",
        "cmcm_160k",
        "cmcm_15kbar",
    }
    for phase in phases.values():
        assert phase.cif_path().is_file(), f"missing CIF for {phase.key}"


@pytest.mark.parametrize(
    "key,lengths,volume,z,space_group",
    [
        ("pbca_5k", (9.16686, 7.53231, 21.24861), 1467.17, 16, 61),
        ("pbca_110k", (9.172, 7.761, 21.66), 1542.0, 16, 61),
        ("pbca_150k", (8.873, 8.000, 22.027), 1563.5, 16, 61),
        ("cmcm_160k", (6.514, 5.4159, 10.756), 379.5, 4, 63),
        ("cmcm_15kbar", (6.1219, 5.2029, 10.244), 326.29, 4, 63),
    ],
)
def test_deposited_cell_matches_the_paper(key, lengths, volume, z, space_group):
    from mmml.analysis.acetone_crystal import acetone_phase, read_acetone_phase

    phase = acetone_phase(key)
    assert phase.space_group_number == space_group
    assert phase.z == z

    atoms = read_acetone_phase(key)
    assert np.allclose(atoms.cell.lengths(), lengths, atol=0.01)
    assert atoms.get_volume() == pytest.approx(volume, rel=0.005)
    assert np.allclose(atoms.cell.angles(), [90.0, 90.0, 90.0], atol=1e-6)


def test_symmetry_expansion_gives_whole_molecules_not_the_asymmetric_unit(phases):
    """Pbca has 8 general positions and Z=16, so the CIF holds only 2 molecules.

    If ASE ever stopped applying the operators this would silently return an
    eighth of the crystal, and every energy downstream would be wrong.
    """
    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.lattice_energy import unwrap_molecules

    for key, phase in phases.items():
        atoms = read_acetone_phase(key)
        mol_id, _ = unwrap_molecules(
            atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array
        )
        sizes = np.bincount(mol_id)
        assert len(sizes) == phase.z, f"{key}: expected Z={phase.z}, got {len(sizes)}"
        # Ordered phases are C3H6O; the disordered one carries 12 half-occupancy H.
        expected_atoms = 10 if phase.ordered_hydrogens else 16
        assert set(sizes) == {expected_atoms}, f"{key}: molecule sizes {set(sizes)}"


def test_deuterated_neutron_structure_is_protiated_on_request():
    """The 5 K refinement is on acetone-d6 and ASE keeps the D masses.

    Left alone they inflate the density by 10%, which would quietly corrupt any
    comparison against the X-ray phases.
    """
    from mmml.analysis.acetone_crystal import read_acetone_phase

    heavy = read_acetone_phase("pbca_5k", protiate=False)
    light = read_acetone_phase("pbca_5k", protiate=True)

    assert sum(heavy.get_masses()) > sum(light.get_masses())
    density = lambda a: sum(a.get_masses()) / a.get_volume() / 0.6022140857  # noqa: E731
    assert density(light) == pytest.approx(1.052, rel=0.01)
    assert density(heavy) == pytest.approx(1.161, rel=0.01)


def test_disordered_high_pressure_phase_is_flagged(phases):
    """Rotationally disordered methyls cannot carry a force field."""
    assert phases["cmcm_15kbar"].ordered_hydrogens is False
    assert phases["cmcm_15kbar"].usable_for_mm is False
    for key in ("pbca_5k", "pbca_110k", "pbca_150k", "cmcm_160k"):
        assert phases[key].usable_for_mm is True


@pytest.mark.parametrize(
    "key,published",
    [
        # Fig. 2 caption and text, 5 K neutron refinement.
        ("pbca_5k", 3.231),  # Type II antiparallel
        ("pbca_5k", 3.368),  # Type I perpendicular, layer (a)
        ("pbca_5k", 3.391),  # Type I perpendicular, chains
        # Text, 150 K stable phase.
        ("pbca_150k", 3.300),
        ("pbca_150k", 3.458),
        ("pbca_150k", 3.491),
        # Text, 110 K "locked-in" Type I contact.
        ("pbca_110k", 3.417),
        # Footnote and text, the two Cmcm phases: Type III sheared-parallel.
        ("cmcm_160k", 3.587),
        ("cmcm_15kbar", 3.365),
    ],
)
def test_published_carbonyl_contacts_are_reproduced(key, published):
    from mmml.analysis.acetone_crystal import carbonyl_contacts, read_acetone_phase

    contacts = carbonyl_contacts(read_acetone_phase(key), max_distance_A=3.8)
    distances = [c.distance_A for c in contacts]
    nearest = min(distances, key=lambda d: abs(d - published))
    assert nearest == pytest.approx(published, abs=CONTACT_TOLERANCE_A), (
        f"{key}: published {published} A, computed {sorted(distances)}"
    )


@pytest.mark.parametrize(
    "key,published",
    [
        ("pbca_5k", 2.336),  # between chains
        ("pbca_5k", 2.479),  # within chains
        ("pbca_110k", 2.511),
        ("pbca_110k", 2.604),
        ("pbca_150k", 2.618),
        ("pbca_150k", 2.710),
    ],
)
def test_published_ch_o_contacts_are_reproduced(key, published):
    from mmml.analysis.acetone_crystal import ch_o_contacts, read_acetone_phase

    contacts = ch_o_contacts(read_acetone_phase(key), max_distance_A=3.0)
    distances = [c.distance_A for c in contacts]
    nearest = min(distances, key=lambda d: abs(d - published))
    assert nearest == pytest.approx(published, abs=CONTACT_TOLERANCE_A), (
        f"{key}: published {published} A, computed {sorted(distances)[:6]}"
    )


def test_carbonyl_motifs_are_classified_as_the_paper_describes():
    """Pbca shows antiparallel and perpendicular; Cmcm shows sheared-parallel."""
    from mmml.analysis.acetone_crystal import carbonyl_contacts, read_acetone_phase

    pbca = carbonyl_contacts(read_acetone_phase("pbca_150k"), max_distance_A=3.6)
    motifs = {c.motif for c in pbca}
    assert any("antiparallel" in m for m in motifs)
    assert any("perpendicular" in m for m in motifs)

    cmcm = carbonyl_contacts(read_acetone_phase("cmcm_160k"), max_distance_A=3.8)
    assert cmcm and all("sheared-parallel" in c.motif for c in cmcm)
    # The paper's Type III motif is parallel by definition.
    assert cmcm[0].angle_deg == pytest.approx(0.0, abs=1.0)


def test_contacts_shorten_on_cooling_as_the_paper_reports():
    """The structural claim behind the 127 K heat-capacity anomaly."""
    from mmml.analysis.acetone_crystal import ch_o_contacts, read_acetone_phase

    shortest = [
        ch_o_contacts(read_acetone_phase(key), max_distance_A=3.0)[0].distance_A
        for key in ("pbca_150k", "pbca_110k", "pbca_5k")
    ]
    assert shortest == sorted(shortest, reverse=True), f"not monotonic: {shortest}"


@pytest.fixture(scope="module")
def lattice_energy_150k():
    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    atoms = read_acetone_phase("pbca_150k")
    return crystal_lattice_energy(
        atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=12.0
    )


def test_cgenff_types_every_molecule_as_acetone(lattice_energy_150k):
    """Typing is geometry-based, so a broken cell shows up as a wrong residue."""
    assert lattice_energy_150k.residues == ("ACO",) * 16


def test_lattice_energy_is_bound_and_dominated_by_dispersion(lattice_energy_150k):
    result = lattice_energy_150k
    assert result.n_molecules == 16
    assert result.e_lattice < 0.0, "crystal must be bound"
    # Acetone is strongly dipolar, but CGenFF still puts most of the cohesion in
    # the LJ term. Both contributions must be attractive.
    assert result.e_lj < 0.0 and result.e_coulomb < 0.0
    assert abs(result.e_lj) > abs(result.e_coulomb)
    assert result.density_g_cm3 == pytest.approx(0.987, rel=0.01)


def test_lattice_energy_regression(lattice_energy_150k):
    """Pin the number so a change in CGenFF handling or the sums is visible."""
    assert lattice_energy_150k.e_lattice == pytest.approx(-10.99, abs=0.05)


def test_sublimation_enthalpy_is_within_reach_of_experiment(lattice_energy_150k):
    """CGenFF overbinds the crystal, but by a force-field amount, not an order."""
    from mmml.analysis.acetone_crystal import ACETONE_SUBLIMATION_REFERENCE
    from mmml.analysis.lattice_energy import KCAL_MOL_TO_KJ_MOL

    dh_kj = lattice_energy_150k.sublimation_enthalpy(150.0) * KCAL_MOL_TO_KJ_MOL
    reference = ACETONE_SUBLIMATION_REFERENCE.dsub_h_kj_mol

    assert reference == pytest.approx(38.6, abs=0.1)
    assert dh_kj == pytest.approx(43.5, abs=1.0)
    assert 0.9 < dh_kj / reference < 1.35


def test_lattice_energy_is_converged_at_the_default_cutoff():
    """Dispersion needs the tail correction; electrostatics needs Ewald.

    Uses the 40-atom Cmcm cell rather than a 160-atom Pbca one so the test costs
    seconds, not a minute -- the convergence behaviour is a property of the sums,
    not of which crystal they are applied to.
    """
    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.lattice_energy import crystal_lattice_energy

    atoms = read_acetone_phase("cmcm_160k")
    results = [
        crystal_lattice_energy(
            atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=rc
        )
        for rc in (10.0, 14.0)
    ]

    assert results[1].e_lattice == pytest.approx(results[0].e_lattice, abs=0.02)
    # The bare LJ term is what the tail correction has to absorb.
    assert abs(results[1].e_lj - results[0].e_lj) > 0.1
    assert results[1].e_coulomb == pytest.approx(results[0].e_coulomb, abs=1e-3)


def test_learned_lj_scales_change_the_lattice_energy():
    """The hook that makes sublimation enthalpy a test of a trained fit.

    Shrinking every well depth must weaken the crystal; the assertion is on the
    direction, since the magnitude depends on which types acetone uses.
    """
    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.lattice_energy import crystal_lattice_energy
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm

    atoms = read_acetone_phase("cmcm_160k")
    n_types = len(cgenff_type_names_from_prm())
    args = (atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array)

    stock = crystal_lattice_energy(*args, cutoff_A=10.0)
    weakened = crystal_lattice_energy(
        *args, cutoff_A=10.0, epsilon_scale=np.full(n_types, 0.5)
    )

    assert weakened.e_lj > stock.e_lj, "halving epsilon must weaken the LJ cohesion"
    assert weakened.e_coulomb == pytest.approx(stock.e_coulomb, abs=1e-6), (
        "LJ scales must not touch electrostatics"
    )
