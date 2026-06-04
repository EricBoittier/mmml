"""Unit tests for monomer COM analysis (no DCD file)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.analyze_monomer_com_dcd import analyze_com, monomer_offsets  # noqa: E402


def test_monomer_offsets_uniform() -> None:
    off = monomer_offsets(10, 2, 5)
    assert off.tolist() == [0, 5, 10]


def test_analyze_com_detects_outlier() -> None:
    n_frames = 4
    n_mol = 3
    per = 2
    n_atoms = n_mol * per
    pos = np.zeros((n_frames, n_atoms, 3), dtype=float)
    for t in range(n_frames):
        for mi in range(n_mol):
            s, e = mi * per, (mi + 1) * per
            pos[t, s:e, 0] = mi * 2.0 + t * (5.0 if mi == 1 else 0.1)
    off = monomer_offsets(n_atoms, n_mol, per)
    stats = analyze_com(pos, off, outlier_factor=2.0, max_cluster_disp_A=1e6)
    assert stats["worst_monomer_1based"] == 2
    assert stats["n_frames"] == n_frames
    assert not stats["ok"]
    assert "outlier_ratio" in stats["fail_reasons"]


def test_analyze_com_fails_uniform_cluster_drift() -> None:
    """Rigid translation of all monomers should fail cluster MSD, not outlier ratio."""
    n_frames = 8
    n_mol = 5
    per = 2
    n_atoms = n_mol * per
    pos = np.zeros((n_frames, n_atoms, 3), dtype=float)
    for t in range(n_frames):
        pos[t, :, 0] = 100.0 * t
        for mi in range(n_mol):
            pos[t, mi * per : (mi + 1) * per, 1] = float(mi)
    off = monomer_offsets(n_atoms, n_mol, per)
    stats = analyze_com(pos, off, max_cluster_disp_A=50.0)
    assert stats["outlier_ratio"] < 2.0
    assert not stats["ok"]
    assert "cluster_com_disp" in stats["fail_reasons"]
    assert stats["mean_msd_cluster_A2"] > 1e4


def test_analyze_com_stable_cluster_passes() -> None:
    n_frames = 10
    n_mol = 3
    per = 2
    n_atoms = n_mol * per
    pos = np.zeros((n_frames, n_atoms, 3), dtype=float)
    for mi in range(n_mol):
        pos[:, mi * per : (mi + 1) * per, 0] = float(mi) * 3.0
    off = monomer_offsets(n_atoms, n_mol, per)
    stats = analyze_com(pos, off)
    assert stats["ok"]
    assert stats["max_cluster_com_disp_A"] == 0.0
