"""Unit tests for CGENFF bonded JAX terms vs jax-md reference."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    KCAL_MOL_TO_EV,
    bonded_energy_and_forces,
    bonded_energy_components,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    extract_residue_rtf,
    load_cgenff_bonded_from_charmm_files,
    load_cgenff_bonded_from_psf,
    parse_psf_ext,
    mm_atom_mask_complement,
    filter_bonded_topology_for_mm,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF
from mmml.interfaces.pycharmmInterface.mixed_ml_mm import (
    MixedMlMmConfig,
    build_mixed_ml_mm_energy_fn,
    prepare_mm_bonded_system,
)

TIP3_PDB = Path("tests/functionality/pycharmmETC/pdb/initial.pdb")
ACO_PSF = Path("tests/functionality/pycharmmETC/psf/aco-1.psf")
ACO_PDB = Path("tests/functionality/pycharmmETC/pdb/aco.pdb")


def _jaxmd_bonded_components(positions, topology, bonded):
    from jax import vmap
    from jax_md import space
    from jax_md.util import normalize, safe_arccos, safe_norm
    disp_fn, _ = space.free()

    def bond_energy(pos):
        if topology.bonds.shape[0] == 0:
            return jnp.array(0.0)
        i, j = topology.bonds[:, 0], topology.bonds[:, 1]
        disp = vmap(disp_fn)(pos[i], pos[j])
        r = safe_norm(disp)
        return jnp.sum(bonded.bond_k * (r - bonded.bond_r0) ** 2)

    def angle_energy(pos):
        if topology.angles.shape[0] == 0:
            return jnp.array(0.0)
        i, j, k = topology.angles[:, 0], topology.angles[:, 1], topology.angles[:, 2]
        rij = vmap(disp_fn)(pos[i], pos[j])
        rkj = vmap(disp_fn)(pos[k], pos[j])
        cos_theta = jnp.sum(normalize(rij) * normalize(rkj), axis=-1)
        theta = safe_arccos(cos_theta)
        return jnp.sum(bonded.angle_k * (theta - bonded.angle_theta0) ** 2)

    return {
        "bond": bond_energy(positions),
        "angle": angle_energy(positions),
    }


def test_extract_residue_rtf_tip3() -> None:
    text = extract_residue_rtf(CGENFF_RTF, "TIP3")
    assert "RESI TIP3" in text
    assert "ATOM OH2" in text
    assert "RESI ACO" not in text


def test_load_cgenff_bonded_tip3() -> None:
    system = load_cgenff_bonded_from_charmm_files(
        TIP3_PDB,
        residue_name="TIP3",
    )
    assert system.n_atoms == 3
    assert system.topology.bonds.shape[0] == 3
    # Bond-graph inference yields 3 angles for TIP3 (O–H–H triangle).
    assert system.topology.angles.shape[0] == 3


def test_bonded_energy_matches_jaxmd_tip3() -> None:
    system = load_cgenff_bonded_from_charmm_files(
        TIP3_PDB,
        residue_name="TIP3",
    )
    positions = system.positions + jnp.array(
        [[0.01, -0.02, 0.03], [0.0, 0.01, -0.01], [-0.02, 0.0, 0.02]]
    )

    ours = bonded_energy_components(positions, system.topology, system.bonded)
    ref = _jaxmd_bonded_components(positions, system.topology, system.bonded)

    for key in ("bond", "angle"):
        assert float(ours[key]) == pytest.approx(float(ref[key]), rel=1e-6, abs=1e-8)

    _, forces = bonded_energy_and_forces(
        positions,
        system.topology,
        system.bonded,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
        energy_unit="kcal/mol",
    )
    assert forces.shape == (3, 3)
    assert jnp.all(jnp.isfinite(forces))


def test_bonded_forces_finite_difference() -> None:
    system = load_cgenff_bonded_from_charmm_files(
        TIP3_PDB,
        residue_name="TIP3",
    )
    positions = system.positions
    _, forces = bonded_energy_and_forces(
        positions,
        system.topology,
        system.bonded,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
        energy_unit="kcal/mol",
    )

    eps = 1e-4
    numeric = np.zeros((3, 3), dtype=np.float64)

    def energy_at(pos):
        return float(
            bonded_energy_components(
                pos,
                system.topology,
                system.bonded,
                urey_k=system.urey_k,
                urey_r0=system.urey_r0,
            )["total"]
        )

    pos_np = np.asarray(positions)
    for atom in range(3):
        for dim in range(3):
            forward = pos_np.copy()
            backward = pos_np.copy()
            forward[atom, dim] += eps
            backward[atom, dim] -= eps
            numeric[atom, dim] = -(energy_at(forward) - energy_at(backward)) / (2 * eps)

    assert np.allclose(np.asarray(forces), numeric, rtol=5e-4, atol=5e-4)


def test_mm_mask_filters_ml_bonded_terms() -> None:
    system = load_cgenff_bonded_from_charmm_files(
        TIP3_PDB,
        residue_name="TIP3",
    )
    # Pretend atom 0 is ML; MM mask keeps atoms 1,2 only.
    mm_system, mm_mask = prepare_mm_bonded_system(system, ml_atom_indices=(0,))
    assert int(jnp.sum(mm_mask)) == 2
    assert mm_system.topology.bonds.shape[0] == 1  # H1-H2 only


def test_mixed_ml_mm_splits_energy() -> None:
    system = load_cgenff_bonded_from_charmm_files(
        TIP3_PDB,
        residue_name="TIP3",
    )
    ml_indices = (0,)  # oxygen as ML "molecule"

    def ml_energy_fn(pos_ml):
        # Harmonic trap on ML atoms only
        e = jnp.sum(pos_ml**2)
        f = 2.0 * pos_ml
        return e * KCAL_MOL_TO_EV, f * KCAL_MOL_TO_EV

    config = MixedMlMmConfig(ml_atom_indices=ml_indices, energy_unit="eV")
    evaluate = build_mixed_ml_mm_energy_fn(system, config, ml_energy_fn)
    breakdown = evaluate(system.positions)

    assert breakdown.ml_energy > 0
    assert breakdown.mm_bonded_energy >= 0
    assert breakdown.total_energy == pytest.approx(
        breakdown.ml_energy + breakdown.mm_bonded_energy
    )
    assert breakdown.total_forces.shape == (3, 3)


def test_mixed_two_molecule_concat_placeholder() -> None:
    """Two TIP3 copies: ML on first, MM bonded on second."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import concat_cgenff_systems

    one = load_cgenff_bonded_from_charmm_files(TIP3_PDB, residue_name="TIP3")
    two = concat_cgenff_systems([one, one])
    assert two.n_atoms == 6

    mm_mask = mm_atom_mask_complement((0, 1, 2), two.n_atoms)
    topo, bonded, urey_k, urey_r0 = filter_bonded_topology_for_mm(
        two.topology, two.bonded, mm_mask, urey_k=two.urey_k, urey_r0=two.urey_r0
    )
    # Second water keeps 3 bonds + 3 inferred angles.
    assert topo.bonds.shape[0] == 3
    assert topo.angles.shape[0] == 3


def test_parse_psf_ext_aco_fixture() -> None:
    psf = parse_psf_ext(ACO_PSF)
    assert psf.n_atoms == 10
    assert psf.bonds.shape == (9, 2)
    assert psf.angles.shape == (15, 3)
    assert psf.torsions.shape == (12, 4)
    assert psf.impropers.shape == (1, 4)
    assert psf.atom_types[0] == "OG2D3"


def test_load_cgenff_bonded_from_psf_aco_smoke() -> None:
    from jax_md.mm_forcefields.io.charmm import parse_pdb_simple

    _, positions = parse_pdb_simple(str(ACO_PDB))
    system = load_cgenff_bonded_from_psf(ACO_PSF, positions)
    components, forces = bonded_energy_and_forces(
        jnp.asarray(positions),
        system.topology,
        system.bonded,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
        energy_unit="kcal/mol",
    )
    assert float(components["total"]) > 0.0
    assert forces.shape == (10, 3)
    assert jnp.all(jnp.isfinite(forces))


def test_improper_energy_matches_charmm_n0_formula_aco() -> None:
    """ACO carbonyl improper: CHARMM n=0 uses 2k*(1+cos(psi)), not constant 2k."""
    from jax_md.mm_forcefields.io.charmm import parse_pdb_simple

    _, positions = parse_pdb_simple(str(ACO_PDB))
    positions = np.asarray(positions, dtype=float)
    assert not np.allclose(positions[:, 1], positions[:, 2]), "ACO fixture y/z columns duplicated"
    assert np.all(np.ptp(positions, axis=0) > 0.2), "ACO fixture must span x,y,z"
    system = load_cgenff_bonded_from_psf(ACO_PSF, positions)
    components = bonded_energy_components(
        jnp.asarray(positions),
        system.topology,
        system.bonded,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
    )
    # Wrong n*psi formula gives 140 kcal/mol regardless of geometry.
    assert float(components["improper"]) < 60.0


def test_parse_charmm_prm_urey_bradley() -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        parse_charmm_prm_urey_bradley,
    )

    line = "CG311  CG321  HGA2     33.43    110.10   22.53   2.17900 ! PROT alkanes"
    prm = Path("tests/unit/fixtures/urey_sample.prm")
    prm.parent.mkdir(parents=True, exist_ok=True)
    prm.write_text("ANGLES\n" + line + "\nEND\n", encoding="utf-8")
    params = parse_charmm_prm_urey_bradley(prm)
    assert params[("CG311", "CG321", "HGA2")] == pytest.approx((22.53, 2.179))
    assert params[("HGA2", "CG321", "CG311")] == pytest.approx((22.53, 2.179))


def test_urey_bradley_energy_and_forces() -> None:
    from jax_md.mm_forcefields.base import BondedParameters
    from jax_md.mm_forcefields.oplsaa.topology import create_topology

    from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
        bonded_energy_components,
        free_space_displacement,
        urey_bradley_energy,
    )

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.5, 0.0],
        ],
        dtype=jnp.float64,
    )
    topology = create_topology(
        n_atoms=3,
        bonds=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
        angles=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        torsions=jnp.zeros((0, 4), dtype=jnp.int32),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        molecule_id=jnp.zeros(3, dtype=jnp.int32),
    )
    bonded = BondedParameters(
        bond_k=jnp.zeros(2),
        bond_r0=jnp.ones(2),
        angle_k=jnp.zeros(1),
        angle_theta0=jnp.zeros(1),
        torsion_k=jnp.zeros(0),
        torsion_n=jnp.zeros(0, dtype=jnp.int32),
        torsion_gamma=jnp.zeros(0),
        improper_k=jnp.zeros(0),
        improper_n=jnp.zeros(0, dtype=jnp.int32),
        improper_gamma=jnp.zeros(0),
        cmap_maps=None,
    )
    kub = jnp.array([10.0])
    r0 = jnp.array([2.0])
    r_02 = float(np.linalg.norm([1.0, 1.5, 0.0]))
    expected = 10.0 * (r_02 - 2.0) ** 2
    e = urey_bradley_energy(
        positions,
        topology,
        kub,
        r0,
        free_space_displacement(),
    )
    assert float(e) == pytest.approx(expected, rel=1e-6)
    comp = bonded_energy_components(
        positions, topology, bonded, urey_k=kub, urey_r0=r0
    )
    assert float(comp["urey"]) == pytest.approx(expected, rel=1e-6)
    assert float(comp["total"]) == pytest.approx(expected, rel=1e-6)


def test_urey_arrays_for_topology_angles() -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        urey_arrays_for_topology_angles,
    )

    prm = Path("tests/unit/fixtures/urey_sample.prm")
    angles = np.array([[0, 1, 2]], dtype=np.int32)
    atom_types = ("CG311", "CG321", "HGA2")
    urey_k, urey_r0 = urey_arrays_for_topology_angles(atom_types, angles, prm)
    assert float(urey_k[0]) == pytest.approx(22.53)
    assert float(urey_r0[0]) == pytest.approx(2.179)


def test_load_protein_urey_from_psf(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
        bonded_energy_and_forces_from_system,
    )
    from mmml.interfaces.pycharmmInterface.cgenff_topology import load_cgenff_bonded_from_psf

    prm = Path("tests/unit/fixtures/urey_sample.prm")
    psf = tmp_path / "prot_ub.psf"
    psf.write_text(
        "\n".join(
            [
                "PSF EXT",
                "",
                "         3 !NATOM",
                "         1 ALA      1        ALA      C1       CG311    0.000000       12.0110           0",
                "         2 ALA      1        ALA      C2       CG321    0.000000       12.0110           0",
                "         3 ALA      1        ALA      H1       HGA2     0.000000        1.0080           0",
                "",
                "         2 !NBOND: bonds",
                "         1         2         2         3",
                "",
                "         1 !NTHETA: angles",
                "         1         2         3",
                "",
                "         0 !NPHI: dihedrals",
                "",
                "         0 !NIMPHI: impropers",
            ]
        ),
        encoding="utf-8",
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.2, 0.0],
        ],
        dtype=np.float64,
    )
    system = load_cgenff_bonded_from_psf(psf, positions, prm_file=prm)
    assert float(system.urey_k[0]) == pytest.approx(22.53)
    assert float(system.urey_r0[0]) == pytest.approx(2.179)
    components, forces = bonded_energy_and_forces_from_system(system)
    assert float(components["urey"]) > 0.0
    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
