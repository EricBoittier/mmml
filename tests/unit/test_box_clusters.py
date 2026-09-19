"""Minimum-image cluster cutting from periodic frames + labelled frame writer."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from mmml.distill.box_clusters import (
    SOURCE_BOX_DIMER,
    SOURCE_BOX_MONOMER,
    SOURCE_REFERENCE,
    BoxClusterConfig,
    box_cluster_pool,
    whole_molecules,
)
from mmml.md.metatomic_pbc import append_training_frame

L = 10.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _box() -> Atoms:
    # molecule 0 straddles the x boundary; molecule 1 sits 3 Å away through it.
    m0 = WATER + [L - 0.3, 5.0, 5.0]
    m1 = WATER + [2.7, 5.0, 5.0]
    m2 = WATER + [6.0, 0.5, 5.0]  # ~5.6-5.8 Å from m0/m1, > cutoff below
    atoms = Atoms("OH2OH2OH2", positions=np.vstack([m0, m1, m2]), cell=[L] * 3, pbc=True)
    atoms.wrap()
    return atoms


def test_whole_molecules_undo_wrap() -> None:
    mols = whole_molecules(_box(), 3)
    for mol in mols:
        assert np.allclose(mol - mol[0], WATER, atol=1e-9)


def test_dimers_use_nearest_image() -> None:
    geos = box_cluster_pool(
        [_box()],
        BoxClusterConfig(atoms_per_monomer=3, dimer_com_cutoff_A=4.0, max_dimers_per_frame=10),
        reference_monomer=Atoms("OH2", positions=WATER),
    )
    assert geos[0].source == SOURCE_REFERENCE
    dimers = [g for g in geos if g.source == SOURCE_BOX_DIMER]
    assert len(dimers) == 1  # only m0-m1 through the boundary
    d = dimers[0]
    assert d.r_com_A == pytest.approx(3.0, abs=1e-9)
    com_a, com_b = d.positions[:3].mean(0), d.positions[3:].mean(0)
    assert np.linalg.norm(com_b - com_a) < 4.0
    assert len([g for g in geos if g.source == SOURCE_BOX_MONOMER]) == 3


def test_bad_atoms_per_monomer() -> None:
    with pytest.raises(ValueError):
        whole_molecules(_box(), 4)


class _Harmonic(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        r = atoms.get_positions()
        self.results = {"energy": float(0.5 * np.sum(r**2)), "forces": -r}


def test_append_training_frame_round_trip(tmp_path) -> None:
    atoms = _box()
    atoms.calc = _Harmonic()
    path = tmp_path / "traj.extxyz"
    for step in (0, 20):
        append_training_frame(path, atoms, step=step, dt_fs=0.5)
    frames = read(str(path), index=":")
    assert len(frames) == 2
    assert frames[1].info["time_fs"] == pytest.approx(10.0)
    assert np.allclose(frames[0].cell.lengths(), [L] * 3)
    assert frames[0].pbc.all()
    assert frames[0].get_potential_energy() == pytest.approx(atoms.get_potential_energy())
    assert np.allclose(frames[0].get_forces(), atoms.get_forces())


def test_topology_filter_drops_reacted_molecule() -> None:
    atoms = _box()
    pos = atoms.get_positions()
    pos[4] += [0.0, 0.0, 3.0]  # molecule 1 loses an H (index 4 = its 2nd atom's H)
    atoms.set_positions(pos)
    stats: dict = {}
    geos = box_cluster_pool(
        [atoms],
        BoxClusterConfig(atoms_per_monomer=3, dimer_com_cutoff_A=8.0, max_dimers_per_frame=10),
        reference_monomer=Atoms("OH2", positions=WATER),
        stats=stats,
    )
    assert stats["n_broken_molecules"] == 1
    assert len([g for g in geos if g.source == SOURCE_BOX_MONOMER]) == 2
    # only the m0-m2 pair survives; every pair with molecule 1 is gone
    assert len([g for g in geos if g.source == SOURCE_BOX_DIMER]) == 1


# --- provenance + grouped (per-trajectory) split ---------------------------


def _seeded_frames(seeds=(0, 1, 2, 3), n_frames=2) -> list[Atoms]:
    rng = np.random.default_rng(11)
    frames = []
    for seed in seeds:
        for k in range(n_frames):
            atoms = _box()
            atoms.set_positions(atoms.get_positions() + rng.normal(scale=0.02, size=(9, 3)))
            atoms.wrap()
            atoms.info.update({"seed": int(seed), "phase": "md", "step": 10 * k})
            frames.append(atoms)
    return frames


def _cfg() -> BoxClusterConfig:
    return BoxClusterConfig(atoms_per_monomer=3, dimer_com_cutoff_A=8.0, max_dimers_per_frame=10)


def _pairwise_teacher():
    from mmml.distill.teacher_label import AseTeacher

    class _Pair(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            r = atoms.get_positions()
            d = r[:, None] - r[None]
            dist = np.linalg.norm(d, axis=-1) + np.eye(len(r))
            e = float(np.sum(np.triu(1.0 / dist, 1)))
            f = np.sum(d / dist[..., None] ** 3, axis=1)
            self.results = {"energy": e, "forces": f}

    return AseTeacher(_Pair())


def test_box_clusters_carry_seed_frame_step_phase() -> None:
    frames = _seeded_frames(seeds=(5, 9), n_frames=2)
    geos = box_cluster_pool(frames, _cfg(), reference_monomer=Atoms("OH2", positions=WATER))
    ref, rest = geos[0], geos[1:]
    assert ref.group_seed is None and ref.group_frame is None
    assert {g.group_seed for g in rest} == {5, 9}
    assert {g.group_frame for g in rest} == {0, 1, 2, 3}
    for g in rest:
        assert g.group_seed == frames[g.group_frame].info["seed"]
        assert g.group_step == frames[g.group_frame].info["step"]
        assert g.group_phase == "md"
        assert g.group_file == 0


def test_box_clusters_file_fallback_without_seed() -> None:
    frames = [_box(), _box()]  # no info["seed"] (e.g. one metatomic-pbc-md run each)
    geos = box_cluster_pool(frames, _cfg(), frame_files=[0, 1])
    assert all(g.group_seed is None for g in geos)
    assert {g.group_file for g in geos} == {0, 1}
    with pytest.raises(ValueError):
        box_cluster_pool(frames, _cfg(), frame_files=[0])


def test_dimer_fragments_inherit_dimer_group() -> None:
    from mmml.distill.teacher_label import ENERGY_MODE_MLMM, label_geometries

    frames = _seeded_frames(seeds=(0, 1), n_frames=1)
    geos = box_cluster_pool(frames, _cfg(), reference_monomer=Atoms("OH2", positions=WATER))
    got = label_geometries(
        _pairwise_teacher(), geos, energy_mode=ENERGY_MODE_MLMM, include_dimer_fragments=True
    )
    n_frag = 0
    for k, s in enumerate(got):
        if s.geometry.kind != "dimer":
            continue
        for frag in (got[k + 1].geometry, got[k + 2].geometry):
            assert frag.source.endswith(":frag")
            for attr in ("group_seed", "group_file", "group_frame", "group_step", "group_phase"):
                assert getattr(frag, attr) == getattr(s.geometry, attr)
            n_frag += 1
    assert n_frag > 0


def _labelled_box_samples(seeds=(0, 1, 2, 3, 4, 5)):
    from mmml.distill.teacher_label import ENERGY_MODE_MLMM, label_geometries

    geos = box_cluster_pool(
        _seeded_frames(seeds=seeds), _cfg(), reference_monomer=Atoms("OH2", positions=WATER)
    )
    return label_geometries(
        _pairwise_teacher(), geos, energy_mode=ENERGY_MODE_MLMM, include_dimer_fragments=True
    )


def test_seed_split_keeps_trajectories_and_triples_together(tmp_path) -> None:
    import json

    from mmml.distill.npz_export import write_distill_npz

    samples = _labelled_box_samples()
    paths = write_distill_npz(samples, tmp_path, pad_atoms=6, valid_fraction=0.34, seed=3, split="seed")
    train = np.load(paths["train"], allow_pickle=True)
    valid = np.load(paths["valid"], allow_pickle=True)
    s_train = set(train["group_seed"][train["group_seed"] >= 0].tolist())
    s_valid = set(valid["group_seed"].tolist())
    assert s_train and s_valid and not (s_train & s_valid)
    assert -1 not in s_valid  # the pdb_eq reference (no group) stays in train
    assert s_train | s_valid == {0, 1, 2, 3, 4, 5}
    report = json.loads(paths["report"].read_text())
    assert report["split"] == "seed"
    assert sorted(report["valid_seeds"]) == sorted(s_valid)
    assert report["n_valid_groups"] == 2  # round(0.34 * 6)
    assert report["n_train"] + report["n_valid"] == len(samples)
    # AB/A/B triples: every dimer's fragments sit on the dimer's side.
    for part in (train, valid):
        kinds, srcs = part["kind"], part["source"]
        for k in np.nonzero(kinds == 1)[0]:
            assert str(srcs[k + 1]).endswith(":frag") and str(srcs[k + 2]).endswith(":frag")
            for key in ("group_seed", "group_frame"):
                assert part[key][k] == part[key][k + 1] == part[key][k + 2]
    assert set(valid["group_phase"].tolist()) == {"md"}


def test_seed_split_needs_two_groups_and_sample_split_is_default(tmp_path) -> None:
    from mmml.distill.npz_export import (
        samples_to_arrays,
        split_train_valid,
        split_train_valid_grouped,
    )

    with pytest.raises(ValueError, match="2 trajectory groups"):
        split_train_valid_grouped(["seed:0", "seed:0", None], valid_fraction=0.5, seed=0)
    is_train, valid = split_train_valid_grouped(
        ["seed:0", "seed:1", None, "file:0"], valid_fraction=0.9, seed=0
    )
    assert len(valid) == 2 and is_train[2]  # >=1 train group left; None stays train
    samples = _labelled_box_samples(seeds=(0, 1))
    arr = samples_to_arrays(samples, pad_atoms=6, valid_fraction=0.2, seed=4)
    assert arr["_split"] == "sample" and arr["_valid_groups"] is None
    expected = split_train_valid(len(samples), valid_fraction=0.2, seed=4)
    assert np.array_equal(arr["is_train"].astype(bool), expected)
    with pytest.raises(ValueError):
        samples_to_arrays(samples, pad_atoms=6, split="frame")


def test_cli_box_pool_defaults_to_seed_split(tmp_path) -> None:
    import json

    from ase.io import write

    from mmml.cli.misc.pet_physnet_distill import main as distill_main

    paths = []
    for seed in range(4):
        p = tmp_path / f"seed_{seed}.extxyz"
        write(str(p), _seeded_frames(seeds=(seed,), n_frames=2), format="extxyz")
        paths.append(str(p))
    nos = tmp_path / "noseed.extxyz"  # frames without seed: grouped by file
    write(str(nos), [_box(), _box()], format="extxyz")
    out = tmp_path / "out"
    rc = distill_main(
        ["--out-dir", str(out), "--geometries-only", "--from-box-extxyz", *paths, str(nos),
         "--atoms-per-monomer", "3", "--energy-mode", "total", "--valid-fraction", "0.4"]
    )
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["split"] == "seed"
    assert report["n_groups"] == 5 and report["n_valid_groups"] == 2
    train = np.load(out / "train.npz", allow_pickle=True)
    valid = np.load(out / "valid.npz", allow_pickle=True)
    key = lambda p: {(int(s), int(f)) for s, f in zip(p["group_seed"], p["group_file"])}  # noqa: E731
    assert not (key(train) & key(valid))
    assert set(train["group_file"].tolist()) | set(valid["group_file"].tolist()) == set(range(5))
