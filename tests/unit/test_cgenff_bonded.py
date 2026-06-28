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
        positions, system.topology, system.bonded, energy_unit="kcal/mol"
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
        positions, system.topology, system.bonded, energy_unit="kcal/mol"
    )

    eps = 1e-4
    numeric = np.zeros((3, 3), dtype=np.float64)

    def energy_at(pos):
        return float(
            bonded_energy_components(pos, system.topology, system.bonded)["total"]
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
    topo, bonded = filter_bonded_topology_for_mm(two.topology, two.bonded, mm_mask)
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
        energy_unit="kcal/mol",
    )
    assert float(components["total"]) > 0.0
    assert forces.shape == (10, 3)
    assert jnp.all(jnp.isfinite(forces))


def test_improper_energy_matches_charmm_central_atom_order_aco() -> None:
    """ACO carbonyl improper must use PSF central atom (I), not raw I-J-K-L dihedral."""
    from jax_md.mm_forcefields.io.charmm import parse_pdb_simple

    _, positions = parse_pdb_simple(str(ACO_PDB))
    system = load_cgenff_bonded_from_psf(ACO_PSF, positions)
    components = bonded_energy_components(
        jnp.asarray(positions),
        system.topology,
        system.bonded,
    )
    # Minimized ACO: CHARMM reports ~0.025 kcal/mol; wrong ordering gives ~140.
    assert float(components["improper"]) == pytest.approx(0.025, abs=0.05)
