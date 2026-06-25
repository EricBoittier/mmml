"""Shared helpers for neighbor-list functionality scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_pass(msg: str) -> None:
    print(f"PASS: {msg}")


def print_fail(msg: str) -> None:
    print(f"FAIL: {msg}")


def two_dimer_cluster(
    *,
    box_side: float = 40.0,
    com_separation: float = 8.0,
    atoms_per_monomer: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, cell_3x3, monomer_offsets, monomer_id) for two dimers."""
    n_monomers = 2
    n_atoms = n_monomers * atoms_per_monomer
    offsets = np.array([0, atoms_per_monomer, n_atoms], dtype=np.int32)
    monomer_id = np.empty(n_atoms, dtype=np.int32)
    for mi in range(n_monomers):
        monomer_id[offsets[mi] : offsets[mi + 1]] = mi

    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for mi in range(n_monomers):
        start = int(offsets[mi])
        end = int(offsets[mi + 1])
        com = np.array([com_separation * mi, 0.0, 0.0], dtype=np.float64)
        for k in range(start, end):
            positions[k] = com + np.array(
                [0.3 * (k - start), 0.2 * ((k - start) % 2), 0.1 * (k - start)],
                dtype=np.float64,
            )

    cell = float(box_side) * np.eye(3, dtype=np.float64)
    return positions, cell, offsets, monomer_id


def npt_box_sequence(base_side: float = 40.0) -> list[np.ndarray]:
    """Box side vectors for NPT invalidation tests."""
    return [
        np.array([base_side, base_side, base_side], dtype=np.float64),
        np.array([base_side * 1.02, base_side * 1.02, base_side * 1.02], dtype=np.float64),
    ]
