"""Cut whole monomers and neighbour dimers out of periodic liquid frames.

PhysNet here trains on gas-phase clusters (no cell in the training loop), so
liquid MD frames become teacher-labelled monomers and COM-close dimers. Each
molecule is made whole with the minimum image before cutting; the partner in a
dimer is shifted to its nearest image. Positions Å.

Frames are ASE ``Atoms`` with a cell, ``atoms_per_monomer`` atoms per molecule
in order (liquid-box / ``metatomic-pbc-md`` layout).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mmml.distill.acetone_pool import Geometry

SOURCE_REFERENCE = "pdb_eq"  # label_geometries takes E_ref from this monomer
SOURCE_BOX_MONOMER = "box_monomer"
SOURCE_BOX_DIMER = "box_dimer"


@dataclass(frozen=True)
class BoxClusterConfig:
    atoms_per_monomer: int
    dimer_com_cutoff_A: float = 6.0
    max_monomers_per_frame: int = 8
    max_dimers_per_frame: int = 24
    seed: int = 0


def _mic(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def whole_molecules(atoms: Atoms, atoms_per_monomer: int) -> np.ndarray:
    """``(n_mol, apm, 3)`` positions, each molecule unwrapped around its first atom."""
    apm = int(atoms_per_monomer)
    n = len(atoms)
    if apm < 1 or n % apm:
        raise ValueError(f"{n} atoms is not a multiple of atoms_per_monomer={apm}")
    pos = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1, apm, 3)
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    if not np.any(atoms.get_pbc()) or abs(np.linalg.det(cell)) < 1e-9:
        return pos
    anchor = pos[:, :1, :]
    delta = _mic((pos - anchor).reshape(-1, 3), cell).reshape(pos.shape)
    return anchor + delta


def box_cluster_pool(
    frames: list[Atoms],
    cfg: BoxClusterConfig,
    *,
    reference_monomer: Atoms | None = None,
) -> list[Geometry]:
    """Monomers + dimers (COM within ``dimer_com_cutoff_A``) from each frame.

    ``reference_monomer`` (e.g. the gas-phase equilibrium xyz) is emitted first
    as ``pdb_eq`` so interaction-mode labels have an ``E_ref``.
    """
    apm = int(cfg.atoms_per_monomer)
    rng = np.random.default_rng(int(cfg.seed))
    geos: list[Geometry] = []
    if reference_monomer is not None:
        if len(reference_monomer) != apm:
            raise ValueError(
                f"reference monomer has {len(reference_monomer)} atoms, expected {apm}"
            )
        geos.append(
            Geometry(
                numbers=np.asarray(reference_monomer.get_atomic_numbers(), dtype=int),
                positions=np.asarray(reference_monomer.get_positions(), dtype=np.float64),
                kind="monomer",
                source=SOURCE_REFERENCE,
                r_com_A=None,
                atoms_per_monomer=(apm,),
            )
        )
    for frame in frames:
        mols = whole_molecules(frame, apm)
        z_mol = np.asarray(frame.get_atomic_numbers(), dtype=int)[:apm]
        masses = np.asarray(frame.get_masses(), dtype=np.float64)[:apm]
        com = (mols * masses[None, :, None]).sum(1) / masses.sum()
        cell = np.asarray(frame.get_cell(), dtype=np.float64)
        periodic = bool(np.any(frame.get_pbc())) and abs(np.linalg.det(cell)) > 1e-9
        n_mol = mols.shape[0]

        for i in rng.permutation(n_mol)[: int(cfg.max_monomers_per_frame)]:
            geos.append(
                Geometry(
                    numbers=z_mol.copy(),
                    positions=mols[i] - com[i],
                    kind="monomer",
                    source=SOURCE_BOX_MONOMER,
                    r_com_A=None,
                    atoms_per_monomer=(apm,),
                )
            )

        pairs: list[tuple[int, int, np.ndarray, float]] = []
        for i in range(n_mol):
            d = com[i + 1 :] - com[i]
            if periodic:
                d = _mic(d, cell)
            r = np.linalg.norm(d, axis=1)
            for k in np.nonzero(r < float(cfg.dimer_com_cutoff_A))[0]:
                j = i + 1 + int(k)
                pairs.append((i, j, d[k], float(r[k])))
        if not pairs:
            continue
        for idx in rng.permutation(len(pairs))[: int(cfg.max_dimers_per_frame)]:
            i, j, d_ij, r_ij = pairs[int(idx)]
            pos_a = mols[i] - com[i]
            pos_b = mols[j] - com[j] + d_ij
            geos.append(
                Geometry(
                    numbers=np.concatenate([z_mol, z_mol]),
                    positions=np.concatenate([pos_a, pos_b], axis=0),
                    kind="dimer",
                    source=SOURCE_BOX_DIMER,
                    r_com_A=r_ij,
                    atoms_per_monomer=(apm, apm),
                )
            )
    return geos
