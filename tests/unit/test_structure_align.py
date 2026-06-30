from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.utils.structure_align import (
    align_positions,
    kabsch_rotation,
    load_npz_structure,
    plot_aligned_structures,
    select_structure_indices,
    structure_rmsd,
)


def test_kabsch_recovers_known_rotation() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    theta = np.deg2rad(35.0)
    rotation_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = points @ rotation_true
    aligned = align_positions(rotated, points)
    target_center = points - points.mean(axis=0)
    assert structure_rmsd(rotated, points) == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(aligned, target_center, atol=1e-10)


def _write_mini_npz(path: Path, frames: list[np.ndarray], z: np.ndarray, energies: list[float]) -> None:
    n_atoms = len(z)
    padded = np.zeros((len(frames), n_atoms, 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        padded[i, : len(frame)] = frame
    np.savez(
        path,
        R=padded,
        Z=np.tile(z, (len(frames), 1)),
        N=np.full(len(frames), n_atoms, dtype=np.int32),
        E=np.asarray(energies, dtype=np.float64),
        F=np.zeros((len(frames), n_atoms, 3), dtype=np.float64),
    )


def test_plot_aligned_structures_writes_png(tmp_path: Path) -> None:
    z = np.array([1, 1], dtype=np.int32)
    base = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    shifted = base + np.array([0.2, -0.1, 0.05])
    npz_path = tmp_path / "mini.npz"
    _write_mini_npz(npz_path, [base, shifted], z, energies=[-1.0, -2.0])

    out = tmp_path / "overlay.png"
    plot_aligned_structures(
        npz_path=npz_path,
        indices=[0, 1],
        reference_index=0,
        out_path=out,
        title="test",
    )
    assert out.is_file()
    assert out.stat().st_size > 100


def test_load_npz_structure_trims_padding(tmp_path: Path) -> None:
    z = np.array([6, 1, 1], dtype=np.int32)
    frame = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    npz_path = tmp_path / "trim.npz"
    _write_mini_npz(npz_path, [frame], z, energies=[0.0])
    atoms = load_npz_structure(npz_path, 0)
    assert len(atoms) == 3
    assert atoms.get_atomic_numbers().tolist() == z.tolist()


def test_select_structure_indices_priority() -> None:
    suspects = [{"index": 5}, {"index": 2}]
    global_outliers = [{"index": 2}, {"index": 9}]
    assert select_structure_indices(
        suspect_samples=suspects,
        global_outliers=global_outliers,
        max_structures=3,
    ) == [5, 2, 9]
