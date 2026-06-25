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


def setup_charmm_aco_dimer_cluster(
    *,
    box_side: float = 40.0,
    com_separation: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2×ACO dimer PSF in PyCHARMM for NL integration tests.

    Returns the same tuple as :func:`two_dimer_cluster` but with a loaded PSF
    (20 atoms, 10 per monomer) and geometry centered in a cubic box.
    """
    import pandas as pd
    import pycharmm
    import pycharmm.generate as gen
    import pycharmm.ic as ic

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM,
        CGENFF_RTF,
        coor,
        pycharmm_quiet,
        read,
        reset_block,
        settings,
    )

    if CGENFF_PRM is None or CGENFF_RTF is None:
        raise RuntimeError("PyCHARMM/CGENFF not available")

    atoms_per_monomer = 10
    n_monomers = 2
    n_atoms = atoms_per_monomer * n_monomers

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    pycharmm_quiet()
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")
    read.sequence_string("ACO ACO")
    gen.new_segment(seg_name="DIMR", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    import pycharmm.psf as psf

    if int(psf.get_natom()) != n_atoms:
        raise RuntimeError(
            f"expected {n_atoms} PSF atoms for ACO dimer, got {psf.get_natom()}"
        )

    r = coor.get_positions().to_numpy(dtype=float)
    r0 = r[:atoms_per_monomer].copy()
    r1 = r[atoms_per_monomer:].copy()
    r0 -= r0.mean(axis=0)
    r1 -= r1.mean(axis=0)
    r1 += np.array([com_separation, 0.0, 0.0], dtype=np.float64)
    positions = np.vstack([r0, r1])
    positions = positions - positions.mean(axis=0) + np.array(
        [box_side / 2, box_side / 2, box_side / 2], dtype=np.float64
    )
    coor.set_positions(pd.DataFrame(positions, columns=["x", "y", "z"]))

    offsets = np.array([0, atoms_per_monomer, n_atoms], dtype=np.int32)
    monomer_id = np.empty(n_atoms, dtype=np.int32)
    for mi in range(n_monomers):
        monomer_id[offsets[mi] : offsets[mi + 1]] = mi
    cell = float(box_side) * np.eye(3, dtype=np.float64)
    return positions, cell, offsets, monomer_id
