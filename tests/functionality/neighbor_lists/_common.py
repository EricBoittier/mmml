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
    from mmml.utils.rich_report import emit_status

    emit_status(True, msg)


def print_fail(msg: str) -> None:
    from mmml.utils.rich_report import emit_status

    emit_status(False, msg)


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


def setup_charmm_composition_cluster(
    composition: str,
    *,
    box_side: float = 40.0,
    spacing: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a CGENFF cluster from ``RES:COUNT`` composition (md-system style).

    Returns ``(positions, cell_3x3, monomer_offsets, monomer_id, atomic_numbers)``.
    Monomers are placed on a spacing grid, then centered in a cubic PBC box.
    Requires PyCHARMM + CGENFF.
    """
    import pandas as pd

    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_from_composition,
        _parse_composition,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        coor,
    )
    from mmml.interfaces.pycharmmInterface.nl_reference import monomer_id_from_offsets

    if CGENFF_PRM is None:
        raise RuntimeError("PyCHARMM/CGENFF not available")

    comp = _parse_composition(composition)
    z, positions, atoms_per_list, _residue_labels = _build_cluster_from_composition(
        composition=comp,
        spacing=float(spacing),
    )
    positions = np.asarray(positions, dtype=np.float64)
    positions = positions - positions.mean(axis=0) + np.array(
        [box_side / 2, box_side / 2, box_side / 2], dtype=np.float64
    )
    coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))

    n_monomers = len(atoms_per_list)
    offsets = np.zeros(n_monomers + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=np.int32))
    monomer_id = monomer_id_from_offsets(offsets, positions.shape[0])
    cell = float(box_side) * np.eye(3, dtype=np.float64)
    return positions, cell, offsets, monomer_id, np.asarray(z, dtype=int)


def synthetic_monomer_cluster(
    *,
    com_positions: np.ndarray,
    cell: np.ndarray,
    atoms_per_monomer: int = 5,
    monomer_atom_offsets: np.ndarray | list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic PBC cluster from monomer COM coordinates.

    Returns ``(positions, cell_3x3, monomer_offsets, monomer_id)``.

    ``monomer_atom_offsets`` may be one (N,3) array for all monomers or a per-monomer list.
    """
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        cell_matrix_3x3 as _cell_matrix_3x3,
        monomer_id_from_offsets as _monomer_id_from_offsets,
    )

    default_offsets = np.array(
        [[0.3 * k, 0.2 * (k % 2), 0.1 * k] for k in range(atoms_per_monomer)],
        dtype=np.float64,
    )

    coms = np.asarray(com_positions, dtype=np.float64).reshape(-1, 3)
    n_monomers = int(coms.shape[0])
    n_atoms = n_monomers * int(atoms_per_monomer)
    offsets = np.arange(0, n_atoms + 1, atoms_per_monomer, dtype=np.int32)
    monomer_id = _monomer_id_from_offsets(offsets, n_atoms)

    if monomer_atom_offsets is None:
        per_monomer_offsets = [default_offsets] * n_monomers
    elif isinstance(monomer_atom_offsets, list):
        per_monomer_offsets = [
            default_offsets if off is None else np.asarray(off, dtype=np.float64)
            for off in monomer_atom_offsets
        ]
    else:
        per_monomer_offsets = [np.asarray(monomer_atom_offsets, dtype=np.float64)] * n_monomers

    cell_mat = _cell_matrix_3x3(cell)
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for mi in range(n_monomers):
        start = int(offsets[mi])
        atom_off = per_monomer_offsets[mi]
        for k in range(atoms_per_monomer):
            positions[start + k] = coms[mi] + atom_off[k]
    return positions, cell_mat, offsets, monomer_id


def cell_matrix_3x3(cell: np.ndarray) -> np.ndarray:
    """Normalize scalar, (3,), or (3,3) cell spec to a 3×3 matrix (Å)."""
    from mmml.interfaces.pycharmmInterface.nl_reference import cell_matrix_3x3 as _cm

    return _cm(cell)


def monomer_id_from_offsets(monomer_offsets: np.ndarray, n_atoms: int) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.nl_reference import monomer_id_from_offsets as _mid

    return _mid(monomer_offsets, n_atoms)


def extreme_pbc_cases() -> list[dict[str, object]]:
    """Named extreme PBC geometries for NL backend stress tests."""
  # fmt: off
    return [
        {
            "name": "opposite_corners",
            "description": "Dimers at opposite cube corners (max MIC wrap path)",
            "box_side": 20.0,
            "cutoff": 13.0,
            "com_positions": np.array([[1.0, 1.0, 1.0], [19.0, 19.0, 19.0]]),
        },
        {
            "name": "tight_box_many_images",
            "description": "Small box (L≈cutoff+ε) forces many periodic images",
            "box_side": 15.0,
            "cutoff": 13.0,
            "com_positions": np.array([[2.0, 7.5, 7.5], [13.0, 7.5, 7.5]]),
        },
        {
            "name": "cutoff_near_half_box",
            "description": "Cutoff ≈ L/2 (minimum-image boundary regime)",
            "box_side": 26.0,
            "cutoff": 13.0,
            "com_positions": np.array([[6.5, 13.0, 13.0], [19.5, 13.0, 13.0]]),
        },
        {
            "name": "wrap_straddle_x",
            "description": "Monomer 1 atoms straddle the x periodic face",
            "box_side": 20.0,
            "cutoff": 13.0,
            "com_positions": np.array([[10.0, 10.0, 10.0], [2.0, 10.0, 10.0]]),
            "monomer_atom_offsets": [
                None,
                np.array(
                    [
                        [0.3, 0.0, 0.0],
                        [0.6, 0.2, 0.0],
                        [-1.7, 0.0, 0.0],
                        [17.8, 0.0, 0.0],
                        [18.1, 0.1, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ],
        },
        {
            "name": "four_dimers_dense",
            "description": "Four dimers on a 2×2 grid in a modest box",
            "box_side": 22.0,
            "cutoff": 13.0,
            "com_positions": np.array(
                [
                    [5.0, 5.0, 11.0],
                    [17.0, 5.0, 11.0],
                    [5.0, 17.0, 11.0],
                    [17.0, 17.0, 11.0],
                ],
                dtype=np.float64,
            ),
        },
        {
            "name": "orthorhombic_cell",
            "description": "Non-cubic orthorhombic cell (diagonal 28×20×16 Å)",
            "cell": np.diag([28.0, 20.0, 16.0]),
            "cutoff": 13.0,
            "com_positions": np.array([[4.0, 4.0, 4.0], [24.0, 16.0, 12.0]]),
        },
        {
            "name": "minimal_com_spacing",
            "description": "Very close monomer COMs (still inter-monomer pairs)",
            "box_side": 30.0,
            "cutoff": 13.0,
            "com_positions": np.array([[14.0, 15.0, 15.0], [16.5, 15.0, 15.0]]),
        },
        {
            "name": "edge_face_contact",
            "description": "One monomer on −x face, partner just inside +x face",
            "box_side": 24.0,
            "cutoff": 13.0,
            "com_positions": np.array([[1.2, 12.0, 12.0], [22.8, 12.0, 12.0]]),
        },
        {
            "name": "high_cutoff_fraction",
            "description": "Cutoff > L/3 in a compact box (dense pair shell)",
            "box_side": 18.0,
            "cutoff": 15.0,
            "com_positions": np.array([[6.0, 9.0, 9.0], [12.0, 9.0, 9.0], [9.0, 6.0, 9.0]]),
        },
    ]
  # fmt: on


def build_extreme_pbc_case(
    case: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Materialize one extreme case. Returns geometry + cutoff + description."""
    if "cell" in case:
        cell_spec = case["cell"]
    else:
        cell_spec = float(case["box_side"])  # type: ignore[arg-type]
    cutoff = float(case["cutoff"])
    com_positions = np.asarray(case["com_positions"], dtype=np.float64)
    offsets_arg = case.get("monomer_atom_offsets")
    positions, cell, offsets, monomer_id = synthetic_monomer_cluster(
        com_positions=com_positions,
        cell=np.asarray(cell_spec, dtype=np.float64),
        monomer_atom_offsets=offsets_arg,  # type: ignore[arg-type]
    )
    desc = str(case.get("description", case.get("name", "extreme")))
    return positions, cell, offsets, monomer_id, cutoff, desc


def setup_charmm_aco_dimer_cluster(
    *,
    box_side: float = 40.0,
    com_separation: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2×ACO dimer PSF in PyCHARMM for NL integration tests.

    Returns the same tuple as :func:`two_dimer_cluster` but with a loaded PSF
    (20 atoms, 10 per monomer) and geometry centered in a cubic box.
    """
    positions, cell, offsets, monomer_id, _z = setup_charmm_composition_cluster(
        "ACO:2",
        box_side=box_side,
        spacing=com_separation,
    )
    return positions, cell, offsets, monomer_id
