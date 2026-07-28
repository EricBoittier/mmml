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
)


def test_stretch_distance_seed_sets_target():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = stretch_distance_seed(r, 0, 1, 3.0)
    assert float(np.linalg.norm(out[1] - out[0])) == pytest.approx(3.0)
    # third atom unchanged
    np.testing.assert_allclose(out[2], r[2])
    # pair COM preserved
    np.testing.assert_allclose(0.5 * (out[0] + out[1]), 0.5 * (r[0] + r[1]))


def test_pack_window_seeds_stretch():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    packed = pack_window_seeds(
        positions=r,
        atom_i=0,
        atom_j=1,
        targets_A=(1.5, 2.5, 3.5),
        seed_mode="stretch",
    )
    assert packed.shape == (6, 3)
    for k, t in enumerate((1.5, 2.5, 3.5)):
        rk = packed.reshape(3, 2, 3)[k]
        assert float(np.linalg.norm(rk[1] - rk[0])) == pytest.approx(t)


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
        atom_i=0,
        atom_j=1,
        targets_A=(1.0, 1.5),
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
