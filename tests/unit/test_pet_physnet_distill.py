"""PET-MAD → PhysNet acetone distillation (dummy teacher, no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from mmml.cli.misc.pet_physnet_distill import build_parser, main as distill_main
from mmml.distill.acetone_pool import (
    ATOMS_PER_ACETONE,
    DIMER_ATOMS,
    AcetonePoolConfig,
    assemble_dimer,
    build_acetone_pool,
    load_acetone_monomer,
    load_dataset_dimers,
    min_pair_distance,
    pool_config_for_preset,
    random_rotation,
)
from mmml.distill.npz_export import write_distill_npz
from mmml.distill.teacher_label import ENERGY_MODE_INTERACTION, ENERGY_MODE_TOTAL, label_geometries


class PairwiseDistanceCalculator(Calculator):
    """E = sum_{i<j} |r_j-r_i| eV; analytic forces. Isolated atom E=0."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms object is required")
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        n = pos.shape[0]
        energy = 0.0
        forces = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[j] - pos[i]
                r = float(np.linalg.norm(delta))
                energy += r
                if r > 1.0e-12:
                    rhat = delta / r
                    forces[i] += rhat
                    forces[j] -= rhat
        self.results = {"energy": float(energy), "forces": forces}


def _tiny_pool() -> AcetonePoolConfig:
    return AcetonePoolConfig(
        seed=0,
        n_monomer_noise=2,
        n_monomer_large_noise=1,
        n_bond_stretch=1,
        n_dimer_orientations=1,
        n_com_per_region=1,
        n_far_field=1,
        n_global_rotations_extra=0,
        include_dataset=True,
    )


def test_acetone_monomer_is_c3h6o() -> None:
    atoms = load_acetone_monomer()
    assert len(atoms) == ATOMS_PER_ACETONE
    z = atoms.get_atomic_numbers()
    assert int(np.sum(z == 6)) == 3
    assert int(np.sum(z == 8)) == 1
    assert int(np.sum(z == 1)) == 6


def test_dataset_dimers_are_20_atoms() -> None:
    frames = load_dataset_dimers()
    assert len(frames) >= 1
    assert all(len(f) == DIMER_ATOMS for f in frames)


def test_random_rotation_is_so3() -> None:
    rng = np.random.default_rng(1)
    rot = random_rotation(rng)
    assert rot.shape == (3, 3)
    assert abs(np.linalg.det(rot) - 1.0) < 1.0e-10
    ident = rot.T @ rot
    assert np.allclose(ident, np.eye(3), atol=1.0e-10)


def test_assemble_dimer_com_distance() -> None:
    mono = load_acetone_monomer().get_positions()
    rot = np.eye(3)
    dimer = assemble_dimer(mono, mono, com_distance_A=5.0, rotation_b=rot)
    com_a = dimer[:10].mean(0)
    com_b = dimer[10:].mean(0)
    assert abs(float(np.linalg.norm(com_b - com_a)) - 5.0) < 1.0e-8


def test_min_pair_filter_drops_clash() -> None:
    cfg = AcetonePoolConfig(
        seed=0,
        n_monomer_noise=0,
        n_monomer_large_noise=0,
        n_bond_stretch=0,
        n_dimer_orientations=1,
        n_com_per_region=1,
        n_far_field=0,
        include_dataset=False,
        min_pair_distance_A=0.75,
        com_repulsive_A=(0.2, 0.3),
        com_well_A=(0.2, 0.3),
        com_shoulder_A=(0.2, 0.3),
        com_long_A=(0.2, 0.3),
    )
    geos = build_acetone_pool(cfg)
    kinds = {g.kind for g in geos}
    assert "monomer" in kinds
    assert "dimer" not in kinds


def test_pool_covers_sources_and_is_reproducible() -> None:
    a = build_acetone_pool(_tiny_pool())
    b = build_acetone_pool(_tiny_pool())
    assert len(a) == len(b)
    sources = {g.source for g in a}
    assert "pdb_eq" in sources
    assert "noise" in sources
    assert "dmc_extxyz" in sources
    assert any(g.kind == "dimer" and g.source.startswith("com_") for g in a)
    assert any(g.source == "far_field" for g in a)
    for ga, gb in zip(a, b):
        assert np.allclose(ga.positions, gb.positions)


def test_interaction_labels_match_cross_pairs() -> None:
    mono = load_acetone_monomer()
    z = mono.get_atomic_numbers()
    r = mono.get_positions()
    dimer_pos = assemble_dimer(r, r, com_distance_A=6.0, rotation_b=np.eye(3))
    from mmml.distill.acetone_pool import Geometry

    geos = [
        Geometry(z, r, "monomer", "pdb_eq", None, (10,)),
        Geometry(
            np.concatenate([z, z]),
            dimer_pos,
            "dimer",
            "com_well",
            6.0,
            (10, 10),
        ),
    ]
    calc = PairwiseDistanceCalculator()
    labeled = label_geometries(calc, geos, energy_mode=ENERGY_MODE_INTERACTION)
    assert labeled[0].energy_eV == pytest.approx(0.0)
    e_int = labeled[1].energy_int_eV
    assert e_int is not None
    # E_int = sum of A–B distances for this dummy.
    pos = dimer_pos
    cross = 0.0
    for i in range(10):
        for j in range(10, 20):
            cross += float(np.linalg.norm(pos[j] - pos[i]))
    assert e_int == pytest.approx(cross, rel=1.0e-12)
    assert labeled[1].energy_eV == pytest.approx(e_int)


def test_total_mode_keeps_raw_teacher_energy() -> None:
    from mmml.distill.acetone_pool import Geometry

    mono = load_acetone_monomer()
    geo = Geometry(mono.get_atomic_numbers(), mono.get_positions(), "monomer", "pdb_eq", None, (10,))
    labeled = label_geometries(PairwiseDistanceCalculator(), [geo], energy_mode=ENERGY_MODE_TOTAL)
    assert labeled[0].energy_eV == pytest.approx(labeled[0].energy_total_eV)
    assert labeled[0].energy_eV != pytest.approx(0.0)


def test_write_npz_units_and_padding(tmp_path: Path) -> None:
    geos = build_acetone_pool(_tiny_pool())
    labeled = label_geometries(PairwiseDistanceCalculator(), geos)
    paths = write_distill_npz(labeled, tmp_path, seed=0, valid_fraction=0.2)
    train = np.load(paths["train"], allow_pickle=True)
    units = json.loads(str(train["_mmml_units"].item()))
    assert units["E"] == "ev"
    assert units["F"] == "ev_angstrom"
    assert train["R"].shape[1] == DIMER_ATOMS
    monomers = train["kind"] == 0
    if np.any(monomers):
        assert np.all(train["N"][monomers] == ATOMS_PER_ACETONE)
        assert np.all(train["Z"][monomers, ATOMS_PER_ACETONE:] == 0)
    assert paths["valid"].is_file()
    report = json.loads(paths["report"].read_text())
    assert report["n_samples"] == len(labeled)
    assert report["n_train"] + report["n_valid"] == report["n_samples"]


def test_cli_geometries_only(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--out-dir", str(tmp_path), "--geometries-only", "--preset", "smoke", "--seed", "1"]
    )
    assert args.geometries_only
    rc = distill_main(
        ["--out-dir", str(tmp_path / "run"), "--geometries-only", "--preset", "smoke", "--seed", "1"]
    )
    assert rc == 0
    assert (tmp_path / "run" / "train.npz").is_file()


def test_pool_preset_md_is_larger_than_smoke() -> None:
    smoke = pool_config_for_preset("smoke")
    md = pool_config_for_preset("md")
    assert md.n_dimer_orientations > smoke.n_dimer_orientations
    assert md.n_monomer_noise > smoke.n_monomer_noise


def test_clash_helper() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert min_pair_distance(pos) == pytest.approx(0.1)
