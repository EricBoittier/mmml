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
    stats = analyze_com(pos, off, outlier_factor=2.0)
    assert stats["worst_monomer_1based"] == 2
    assert stats["n_frames"] == n_frames
