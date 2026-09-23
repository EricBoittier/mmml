"""Collapse guards in ``mmml pet-box-dataset``: intact molecules and FIRE-end energy outliers."""

from __future__ import annotations

import json

import numpy as np
from ase import Atoms

from mmml.distill.box_dataset import (
    REJECTED_TRAJ_NAME,
    TRAJ_NAME,
    flag_energy_outliers,
    molecule_damage_check,
    random_packed_box,
)

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]])


def _water() -> Atoms:
    return Atoms("OH2", positions=WATER)


def test_damage_check_on_a_wrapped_intact_box() -> None:
    box = random_packed_box(_water(), 20, 12.0, seed=3, jitter_frac=0.25)
    n_damaged, d_min = molecule_damage_check(_water(), 0.4)(box)
    assert n_damaged == 0  # wrapped molecules are made whole first
    assert d_min > 1.0
    assert molecule_damage_check(_water(), None) is None


def _pair(shift_b) -> Atoms:
    pos = np.concatenate([WATER, WATER + np.asarray(shift_b, float)])
    return Atoms("OH2OH2", positions=pos, cell=[20.0] * 3, pbc=True)


def test_damage_check_flags_stretched_and_intermolecularly_bonded_molecules() -> None:
    check = molecule_damage_check(_water(), 0.4)
    assert check(_pair([5.0, 0.0, 0.0]))[0] == 0
    stretched = _pair([5.0, 0.0, 0.0])
    pos = stretched.get_positions()
    pos[2] += [0.0, 0.0, -3.0]  # water A loses an H into vacuum
    stretched.set_positions(pos)
    assert check(stretched)[0] == 1
    # B's O 1.0 A from A's first H: every intramolecular bond intact, the pair is bonded
    fused = _pair(WATER[1] + [1.0, 0.0, 0.0])
    n_damaged, d_min = check(fused)
    assert n_damaged == 2
    assert np.isclose(d_min, 1.0, atol=1e-6)


def _seed(tmp_path, seed: int, e_per_mol: float, n_mol: int = 100) -> None:
    d = tmp_path / f"seed_{seed:04d}"
    d.mkdir()
    (d / TRAJ_NAME).write_text("frames\n")
    summary = {"seed": seed, "n_molecules": n_mol, "rejected": None, "E_fire_end_eV": e_per_mol * n_mol}
    (d / "summary.json").write_text(json.dumps(summary))


def test_energy_outlier_rejects_collapsed_seed_only(tmp_path) -> None:
    # accepted seeds within 0.03 eV/molecule; seed 9 collapsed 2.2 eV/molecule lower (DCM run)
    for seed, e in ((0, -22.05), (1, -22.04), (2, -22.07), (3, -22.06), (9, -24.20)):
        _seed(tmp_path, seed, e)
    rejected = flag_energy_outliers(tmp_path, 0.25)
    assert [s["seed"] for s in rejected] == [9]
    d = tmp_path / "seed_0009"
    assert not (d / TRAJ_NAME).exists() and (d / REJECTED_TRAJ_NAME).exists()
    assert "below the other seeds' median" in json.loads((d / "summary.json").read_text())["rejected"]
    assert (tmp_path / "seed_0000" / TRAJ_NAME).exists()
    assert flag_energy_outliers(tmp_path, 0.25) == []  # idempotent


def test_energy_outlier_needs_three_seeds_and_can_be_disabled(tmp_path) -> None:
    _seed(tmp_path, 0, -22.0)
    _seed(tmp_path, 1, -30.0)
    assert flag_energy_outliers(tmp_path, 0.25) == []
    _seed(tmp_path, 2, -22.0)
    assert flag_energy_outliers(tmp_path, None) == []
