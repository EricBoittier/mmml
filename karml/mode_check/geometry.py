"""Build vacuum monomer / small-cluster geometries for mode checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from mmml.analysis.dimer_molecules import MOLECULES


def parse_composition_spec(spec: str) -> list[tuple[str, int]]:
    """Parse ``TIP3:2`` or ``MEOH:1,TIP3:1`` into ``[(RES, n), ...]``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition

    return parse_composition(spec)


def composition_n_monomers(composition: list[tuple[str, int]] | tuple[tuple[str, int], ...]) -> int:
    return int(sum(int(n) for _, n in composition))


def build_vacuum_cluster_from_molecules(
    composition: list[tuple[str, int]] | tuple[tuple[str, int], ...],
    *,
    separation_A: float = 2.8,
) -> tuple[Atoms, list[int], list[str]]:
    """Stack named monomers from ``MOLECULES`` along +x with fixed separation.

    Returns ``(atoms, atoms_per_monomer, residue_labels)``.
    """
    positions: list[np.ndarray] = []
    numbers: list[int] = []
    atoms_per: list[int] = []
    labels: list[str] = []
    cursor = 0.0
    for res, count in composition:
        key = str(res).upper()
        if key not in MOLECULES:
            raise KeyError(
                f"residue {key!r} not in mmml.analysis.dimer_molecules.MOLECULES; "
                f"available={sorted(MOLECULES)}"
            )
        mono = MOLECULES[key]
        mono_pos = np.asarray(mono.get_positions(), dtype=float)
        mono_pos = mono_pos - mono_pos.mean(axis=0)
        z = np.asarray(mono.get_atomic_numbers(), dtype=int)
        for _ in range(int(count)):
            positions.append(mono_pos + np.array([cursor, 0.0, 0.0]))
            numbers.extend(int(x) for x in z)
            atoms_per.append(int(len(z)))
            labels.append(key)
            cursor += float(separation_A)
    atoms = Atoms(numbers=numbers, positions=np.vstack(positions), pbc=False)
    return atoms, atoms_per, labels


def load_atoms_xyz(
    path: Path,
    *,
    atoms_per_monomer: list[int] | tuple[int, ...] | None = None,
) -> tuple[Atoms, list[int] | None]:
    """Load a vacuum geometry from XYZ/PDB; optional monomer layout."""
    atoms = ase_read(str(Path(path).expanduser()))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"expected a single ASE Atoms from {path}")
    atoms.set_pbc(False)
    apm = None if atoms_per_monomer is None else [int(n) for n in atoms_per_monomer]
    if apm is not None and int(sum(apm)) != len(atoms):
        raise ValueError(
            f"atoms_per_monomer sum ({sum(apm)}) != natoms ({len(atoms)}) in {path}"
        )
    return atoms, apm
