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


def test_unwrap_positions_keeps_contiguous_molecule_whole() -> None:
    positions = np.array(
        [
            [[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]],
        ]
    )

    out = unwrap_traj.unwrap_positions(
        positions,
        cell=np.diag([10.0, 10.0, 10.0]),
        group_size=2,
    )

    assert np.allclose(out[:, :, 0], [[9.8, 10.2], [10.1, 10.5]])


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


def test_cli_writes_cell_metadata_to_xyz_comment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = tmp_path / "wrapped.traj"
    with Trajectory(str(wrapped), "w") as traj:
        traj.write(Atoms("H", positions=[[9.5, 0.0, 0.0]], cell=[10.0, 11.0, 12.0], pbc=True))

    output = tmp_path / "unwrapped.xyz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmml unwrap-traj", str(wrapped), "-o", str(output), "--format", "xyz", "--quiet"],
    )

    assert unwrap_traj.main() == 0
    comment = output.read_text(encoding="utf-8").splitlines()[1]
    assert 'Lattice="10 0 0 0 11 0 0 0 12"' in comment
    assert 'pbc="T T T"' in comment


def test_cli_infers_variable_size_molecules_for_ase_traj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = tmp_path / "mixed.traj"
    with Trajectory(str(wrapped), "w") as traj:
        traj.write(
            Atoms(
                numbers=[8, 1, 1, 1, 1],
                positions=[
                    [9.8, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [9.8, 0.7, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.7, 0.0, 0.0],
                ],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
        )
        traj.write(
            Atoms(
                numbers=[8, 1, 1, 1, 1],
                positions=[
                    [0.1, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.1, 0.7, 0.0],
                    [4.2, 0.0, 0.0],
                    [4.9, 0.0, 0.0],
                ],
                cell=[10.0, 10.0, 10.0],
                pbc=True,
            )
        )

    output = tmp_path / "unwrapped.xyz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmml unwrap-traj", str(wrapped), "-o", str(output), "--format", "xyz", "--fast", "--quiet"],
    )

    assert unwrap_traj.main() == 0
    frames = read(str(output), index=":")
    assert len(frames) == 2
    assert np.allclose(frames[0].positions[:3, 0], [9.8, 10.2, 9.8])
    assert np.allclose(frames[1].positions[:3, 0], [10.1, 10.5, 10.1])


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


def test_cli_unwraps_coordinate_only_h5_with_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = tmp_path / "coords_only.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("positions", data=np.array([[[9.5, 0.0, 0.0]], [[0.2, 0.0, 0.0]]]))

    reference = tmp_path / "reference.traj"
    with Trajectory(str(reference), "w") as traj:
        traj.write(Atoms("H", positions=[[9.5, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True))

    output = tmp_path / "unwrapped.extxyz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml unwrap-traj",
            str(h5_path),
            "-o",
            str(output),
            "--reference",
            str(reference),
            "--format",
            "extxyz",
            "--fast",
            "--quiet",
        ],
    )

    assert unwrap_traj.main() == 0
    frames = read(str(output), index=":")
    assert len(frames) == 2
    assert np.allclose([frame.positions[0, 0] for frame in frames], [9.5, 10.2])


def test_cli_dcd_and_psf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Create a dummy PSF file
    psf_path = tmp_path / "reference.psf"
    psf_content = """* Remarks...
       2 !NATOM
       1 ACO  1    ACO  C1   CH3    -0.270000       12.0110           0
       2 ACO  1    ACO  H11  H       0.090000        1.0080           0
"""
    psf_path.write_text(psf_content, encoding="utf-8")

    # Verify PSF parsing
    parsed_numbers = unwrap_traj._read_psf_atomic_numbers(psf_path)
    assert np.array_equal(parsed_numbers, [6, 1])

    # 2. Create input h5 coords
    h5_path = tmp_path / "coords.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("positions", data=np.array([
            [[9.5, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]],
        ]))
        handle.create_dataset("cell", data=np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))

    # 3. Write to DCD using unwrap-traj
    dcd_output = tmp_path / "unwrapped.dcd"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml unwrap-traj",
            str(h5_path),
            "-o",
            str(dcd_output),
            "--reference",
            str(psf_path),
            "--format",
            "dcd",
            "--quiet",
        ],
    )
    assert unwrap_traj.main() == 0
    assert dcd_output.exists()

    # 4. Now read the DCD back using unwrap-traj and write to xyz
    xyz_output = tmp_path / "unwrapped.xyz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml unwrap-traj",
            str(dcd_output),
            "-o",
            str(xyz_output),
            "--reference",
            str(psf_path),
            "--cell",
            "10.0,10.0,10.0",
            "--format",
            "xyz",
            "--fast",
            "--quiet",
        ],
    )
    assert unwrap_traj.main() == 0
    frames = read(str(xyz_output), index=":")
    assert len(frames) == 2
    assert np.allclose([frame.positions[0, 0] for frame in frames], [9.5, 10.2])
    assert np.allclose([frame.positions[1, 0] for frame in frames], [10.2, 9.8])
