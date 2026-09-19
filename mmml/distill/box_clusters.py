"""Cut whole monomers and neighbour dimers out of periodic liquid frames.

PhysNet here trains on gas-phase clusters (no cell in the training loop), so
liquid MD frames become teacher-labelled monomers and COM-close dimers. Each
molecule is made whole with the minimum image before cutting; the partner in a
dimer is shifted to its nearest image. Positions Å.

Frames are ASE ``Atoms`` with a cell, ``atoms_per_monomer`` atoms per molecule
in order (liquid-box / ``metatomic-pbc-md`` layout).

ML/MM coverage: ``r_com`` is the unweighted centroid distance, the same
quantity ``ml_switch_scale`` uses. At the MLpot defaults (mm_switch_on 6.0,
ml_switch_width 1.5) the ML dimer term is full below 4.5 Å, tapers to 0 at
6.0 Å, and sparse ML evaluates dimers up to 7.5 Å. Dimers are drawn evenly
from ``dimer_r_bins_A`` so contact and taper pairs are not swamped by the
first-shell peak. The frame's ``info["phase"]`` (``fire``/``md``) is kept in
``Geometry.source``.
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
    dimer_com_cutoff_A: float = 7.5
    # Stratified draw: equal quota per bin (bins past the cutoff are ignored).
    dimer_r_bins_A: tuple[float, ...] = (0.0, 3.5, 4.5, 5.25, 6.0, 7.5)
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


def _stratified_pick(
    r: np.ndarray, edges: np.ndarray, n_max: int, rng: np.random.Generator
) -> list[int]:
    """Up to ``n_max`` indices, equal quota per ``edges`` bin; unused quota spills over."""
    bins = np.digitize(r, edges) - 1
    n_bins = len(edges) - 1
    pools = [list(rng.permutation(np.nonzero(bins == b)[0])) for b in range(n_bins)]
    picked: list[int] = []
    quota = max(n_max // max(n_bins, 1), 1)
    for pool in pools:
        picked.extend(int(i) for i in pool[:quota])
        del pool[:quota]
    rest = [int(i) for pool in pools for i in pool]
    rng.shuffle(rest)
    picked.extend(rest[: max(n_max - len(picked), 0)])
    return picked[:n_max]


def box_cluster_pool(
    frames: list[Atoms],
    cfg: BoxClusterConfig,
    *,
    reference_monomer: Atoms | None = None,
) -> list[Geometry]:
    """Monomers + dimers (centroid distance < ``dimer_com_cutoff_A``) from each frame.

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
    edges = np.asarray(
        [e for e in cfg.dimer_r_bins_A if e < float(cfg.dimer_com_cutoff_A)]
        + [float(cfg.dimer_com_cutoff_A)],
        dtype=np.float64,
    )
    for frame in frames:
        phase = str(frame.info.get("phase", "")).strip()
        tag = f":{phase}" if phase else ""
        mols = whole_molecules(frame, apm)
        z_mol = np.asarray(frame.get_atomic_numbers(), dtype=int)[:apm]
        com = mols.mean(axis=1)  # unweighted centroid, as in ml_switch_scale
        cell = np.asarray(frame.get_cell(), dtype=np.float64)
        periodic = bool(np.any(frame.get_pbc())) and abs(np.linalg.det(cell)) > 1e-9
        n_mol = mols.shape[0]

        for i in rng.permutation(n_mol)[: int(cfg.max_monomers_per_frame)]:
            geos.append(
                Geometry(
                    numbers=z_mol.copy(),
                    positions=mols[i] - com[i],
                    kind="monomer",
                    source=SOURCE_BOX_MONOMER + tag,
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
        for idx in _stratified_pick(
            np.array([p[3] for p in pairs]), edges, int(cfg.max_dimers_per_frame), rng
        ):
            i, j, d_ij, r_ij = pairs[int(idx)]
            pos_a = mols[i] - com[i]
            pos_b = mols[j] - com[j] + d_ij
            geos.append(
                Geometry(
                    numbers=np.concatenate([z_mol, z_mol]),
                    positions=np.concatenate([pos_a, pos_b], axis=0),
                    kind="dimer",
                    source=SOURCE_BOX_DIMER + tag,
                    r_com_A=r_ij,
                    atoms_per_monomer=(apm, apm),
                )
            )
    return geos
