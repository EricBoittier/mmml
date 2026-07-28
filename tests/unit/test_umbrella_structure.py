"""Tests for umbrella structure loading and CV stretch seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.umbrella.structure import (
    load_structure,
    load_structure_frames,
    pack_window_seeds,
    stretch_distance_seed,
    stretch_two_distances,
)


def test_stretch_distance_seed_sets_target():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = stretch_distance_seed(r, 0, 1, 3.0)
    assert float(np.linalg.norm(out[1] - out[0])) == pytest.approx(3.0)
    np.testing.assert_allclose(out[0], r[0])  # atom_i fixed
    np.testing.assert_allclose(out[2], r[2])  # untouched


def test_stretch_distance_seed_move_with():
    # Fix C=2, move N=1 and H=3 together along N–C
    r = np.zeros((4, 3))
    r[1] = [2.0, 0.0, 0.0]  # N
    r[2] = [0.0, 0.0, 0.0]  # C
    r[3] = [2.5, 0.5, 0.0]  # H bonded to N
    out = stretch_distance_seed(r, 2, 1, 3.0, move_with=(1, 3))
    assert float(np.linalg.norm(out[1] - out[2])) == pytest.approx(3.0)
    np.testing.assert_allclose(out[2], r[2])
    # H shifts by the same vector as N
    np.testing.assert_allclose(out[3] - r[3], out[1] - r[1])


def test_stretch_two_distances_shared_hub():
    # hub=2 (C), Cl=0, N=1
    r = np.zeros((3, 3))
    r[0] = [-2.0, 0.0, 0.0]
    r[1] = [2.0, 0.0, 0.0]
    r[2] = [0.0, 0.0, 0.0]
    out = stretch_two_distances(r, (0, 2), 1.5, (1, 2), 2.5)
    assert float(np.linalg.norm(out[0] - out[2])) == pytest.approx(1.5)
    assert float(np.linalg.norm(out[1] - out[2])) == pytest.approx(2.5)
    np.testing.assert_allclose(out[2], r[2])


def test_pack_window_seeds_stretch():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    packed = pack_window_seeds(
        positions=r,
        atom_pairs=((0, 1),),
        targets_per_cv=((1.5, 2.5, 3.5),),
        seed_mode="stretch",
    )
    assert packed.shape == (6, 3)
    for k, t in enumerate((1.5, 2.5, 3.5)):
        rk = packed.reshape(3, 2, 3)[k]
        assert float(np.linalg.norm(rk[1] - rk[0])) == pytest.approx(t)


def test_pack_window_seeds_2d():
    r = np.zeros((3, 3))
    r[0, 0] = -2.0
    r[1, 0] = 2.0
    packed = pack_window_seeds(
        positions=r,
        atom_pairs=((0, 2), (1, 2)),
        targets_per_cv=((1.5, 2.0), (2.5, 3.0)),
        seed_mode="stretch",
    )
    assert packed.shape == (6, 3)
    batch = packed.reshape(2, 3, 3)
    assert float(np.linalg.norm(batch[0, 0] - batch[0, 2])) == pytest.approx(1.5)
    assert float(np.linalg.norm(batch[0, 1] - batch[0, 2])) == pytest.approx(2.5)
    assert float(np.linalg.norm(batch[1, 0] - batch[1, 2])) == pytest.approx(2.0)
    assert float(np.linalg.norm(batch[1, 1] - batch[1, 2])) == pytest.approx(3.0)


def test_load_structure_npz(tmp_path: Path):
    r = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    z = np.array([1, 1], dtype=np.int32)
    path = tmp_path / "mol.npz"
    np.savez(path, R=r, Z=z)

    r0, z0 = load_structure(path, index=1)
    assert z0.tolist() == [1, 1]
    np.testing.assert_allclose(r0[1, 0], 1.5)

    r_multi, z1 = load_structure_frames(path, n_frames=2, start_index=0)
    assert r_multi.shape == (2, 2, 3)
    assert z1.tolist() == [1, 1]

    packed = pack_window_seeds(
        positions=r_multi[0],
        atom_pairs=((0, 1),),
        targets_per_cv=((1.0, 1.5),),
        seed_mode="frames",
        frames=r_multi,
    )
    assert packed.shape == (4, 3)


def test_load_structure_xyz(tmp_path: Path):
    xyz = tmp_path / "m.xyz"
    xyz.write_text(
        "2\n\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    r, z = load_structure(xyz)
    assert z.tolist() == [1, 1]
    assert r.shape == (2, 3)
