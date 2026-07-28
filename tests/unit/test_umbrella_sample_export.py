"""Unit tests for umbrella trajectory export helpers (no MD)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.sample import center_com_positions, select_lowest_energy_frames


def test_center_com_positions_moves_mass_weighted_com_to_origin():
    pos = np.array(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    masses = np.array([1.0, 3.0], dtype=np.float64)
    centered = center_com_positions(pos, masses)
    com = np.sum(centered * masses[:, None], axis=0) / np.sum(masses)
    assert com == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    # Relative geometry preserved
    assert centered[1] - centered[0] == pytest.approx(pos[1] - pos[0])


def test_center_com_positions_batches_over_leading_axes():
    pos = np.zeros((2, 3, 2, 3), dtype=np.float64)
    pos[..., 0, :] = [10.0, 0.0, 0.0]
    pos[..., 1, :] = [12.0, 0.0, 0.0]
    masses = np.array([1.0, 1.0], dtype=np.float64)
    centered = center_com_positions(pos, masses)
    assert centered.shape == pos.shape
    com = centered.mean(axis=-2)
    assert com == pytest.approx(0.0, abs=1e-12)


def test_select_lowest_energy_frames_per_window():
    # K=2 windows, T=3 frames, N=1 atom
    positions = np.zeros((2, 3, 1, 3), dtype=np.float64)
    positions[0, 0, 0] = [0.0, 0.0, 0.0]
    positions[0, 1, 0] = [1.0, 0.0, 0.0]
    positions[0, 2, 0] = [2.0, 0.0, 0.0]
    positions[1, 0, 0] = [0.0, 1.0, 0.0]
    positions[1, 1, 0] = [0.0, 2.0, 0.0]
    positions[1, 2, 0] = [0.0, 3.0, 0.0]
    energies = np.array(
        [
            [5.0, 1.0, 3.0],  # window 0 → frame 1
            [4.0, 4.5, 0.5],  # window 1 → frame 2
        ],
        dtype=np.float64,
    )
    chosen, idx, ene = select_lowest_energy_frames(positions, energies)
    assert idx.tolist() == [1, 2]
    assert ene.tolist() == [1.0, 0.5]
    assert chosen[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert chosen[1, 0].tolist() == [0.0, 3.0, 0.0]
