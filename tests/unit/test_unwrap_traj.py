from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
h5py = pytest.importorskip("h5py")

from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

from mmml.cli.misc import unwrap_traj


def test_unwrap_positions_crossing_boundary() -> None:
    positions = np.array(
        [
            [[9.5, 1.0, 1.0]],
            [[0.2, 1.0, 1.0]],
            [[0.9, 1.0, 1.0]],
        ]
    )

    out = unwrap_traj.unwrap_positions(positions, cell=np.diag([10.0, 10.0, 10.0]))

    assert np.allclose(out[:, 0, 0], [9.5, 10.2, 10.9])


def test_cli_unwraps_ase_traj_to_fast_extxyz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = tmp_path / "wrapped.traj"
    with Trajectory(str(wrapped), "w") as traj:
        for x in (9.5, 0.2):
            atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
            traj.write(atoms)

    output = tmp_path / "unwrapped.extxyz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmml unwrap-traj", str(wrapped), "-o", str(output), "--format", "extxyz", "--fast", "--quiet"],
    )

    assert unwrap_traj.main() == 0
    frames = read(str(output), index=":")
    assert len(frames) == 2
    assert np.allclose([frame.positions[0, 0] for frame in frames], [9.5, 10.2])


def test_cli_unwraps_h5_coordinates_to_fast_xyz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = tmp_path / "coords.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("R", data=np.array([[[9.5, 0.0, 0.0]], [[0.2, 0.0, 0.0]]]))
        handle.create_dataset("Z", data=np.array([1], dtype=np.int32))
        handle.create_dataset("cell", data=np.diag([10.0, 10.0, 10.0]))

    output = tmp_path / "unwrapped.xyz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmml unwrap-traj", str(h5_path), "-o", str(output), "--format", "xyz", "--fast", "--quiet"],
    )

    assert unwrap_traj.main() == 0
    frames = read(str(output), index=":")
    assert len(frames) == 2
    assert np.allclose([frame.positions[0, 0] for frame in frames], [9.5, 10.2])
