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

Provenance: every cut cluster carries its frame's ``info["seed"]``
(``group_seed``; ``mmml pet-box-dataset`` writes one per trajectory), the
input-file index (``group_file``, the fallback group for frames without a
seed), the frame's index in ``frames`` (``group_frame``), ``info["step"]`` and
the phase. ``write_distill_npz(split="seed")`` keeps each trajectory on one
side of the train/valid split.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    # Drop molecules whose covalent bonds (graph from the reference monomer)
    # stretch past this, or that form a covalent contact with another molecule:
    # reactive frames (e.g. an H hopped to a neighbour, a C-O bond between two
    # carbonyls) keep atom indices but no longer hold intact molecules. None disables.
    max_bond_stretch_A: float | None = 0.4


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


_COVALENT_R_A = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05, 17: 1.02}


def bond_graph(numbers: np.ndarray, positions: np.ndarray, scale: float = 1.2) -> np.ndarray:
    """``(n_bonds, 2)`` atom pairs closer than ``scale`` × covalent radius sum."""
    z = np.asarray(numbers, dtype=int)
    r = np.asarray(positions, dtype=np.float64)
    rad = np.array([_COVALENT_R_A.get(int(a), 0.8) for a in z])
    d = np.linalg.norm(r[:, None] - r[None], axis=-1)
    i, j = np.triu_indices(len(z), 1)
    keep = d[i, j] < scale * (rad[i] + rad[j])
    return np.stack([i[keep], j[keep]], axis=1)


def intact_molecules(mols: np.ndarray, bonds: np.ndarray, ref_lengths: np.ndarray, tol_A: float) -> np.ndarray:
    """Bool per molecule: every reference bond within ``tol_A`` of its reference length."""
    if len(bonds) == 0:
        return np.ones(mols.shape[0], dtype=bool)
    d = np.linalg.norm(mols[:, bonds[:, 0]] - mols[:, bonds[:, 1]], axis=-1)
    return np.all(np.abs(d - ref_lengths[None, :]) <= float(tol_A), axis=1)


def fused_molecules(atoms: Atoms, atoms_per_monomer: int, scale: float = 1.2) -> tuple[np.ndarray, float]:
    """Bool per molecule: covalently bonded to another molecule; plus the closest intermolecular distance.

    A contact is an atom pair from different molecules closer than ``scale`` ×
    covalent radius sum (same rule as :func:`bond_graph`). Minimum image when
    the frame is periodic. The distance is ``inf`` when no pair lies within 2.5 Å.
    """
    from ase.neighborlist import neighbor_list

    apm = int(atoms_per_monomer)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    n_mol = len(z) // apm
    fused = np.zeros(n_mol, dtype=bool)
    i, j, d = neighbor_list("ijd", atoms, 2.5)
    inter = (i // apm) != (j // apm)
    if not inter.any():
        return fused, float("inf")
    i, j, d = i[inter], j[inter], d[inter]
    rad = np.array([_COVALENT_R_A.get(int(a), 0.8) for a in z])
    bonded = d < scale * (rad[i] + rad[j])
    fused[i[bonded] // apm] = True
    return fused, float(d.min())


def damaged_molecules(
    atoms: Atoms, atoms_per_monomer: int, bonds: np.ndarray, ref_lengths: np.ndarray, tol_A: float
) -> tuple[np.ndarray, float]:
    """Bool per molecule: a reference bond stretched past ``tol_A`` or a covalent contact
    with another molecule (a reaction that keeps every intramolecular bond). Also returns
    the closest intermolecular distance (Å)."""
    ok = intact_molecules(whole_molecules(atoms, atoms_per_monomer), bonds, ref_lengths, tol_A)
    fused, d_min = fused_molecules(atoms, atoms_per_monomer)
    return ~ok | fused, d_min


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
    stats: dict | None = None,
    frame_files: Sequence[int] | None = None,
) -> list[Geometry]:
    """Monomers + dimers (centroid distance < ``dimer_com_cutoff_A``) from each frame.

    ``reference_monomer`` (e.g. the gas-phase equilibrium xyz) is emitted first
    as ``pdb_eq`` so mlmm/interaction labels have an ``E_ref``; its bond graph
    also drives the intact-molecule filter. ``stats`` (if given) receives
    ``n_broken_molecules``. ``frame_files`` gives each frame's input-file
    index (``group_file``); default 0 for every frame.
    """
    if frame_files is not None and len(frame_files) != len(frames):
        raise ValueError(f"frame_files has {len(frame_files)} entries for {len(frames)} frames")
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
    bonds = ref_lengths = None
    n_broken = 0
    if cfg.max_bond_stretch_A is not None and reference_monomer is not None:
        ref_r = np.asarray(reference_monomer.get_positions(), dtype=np.float64)
        bonds = bond_graph(reference_monomer.get_atomic_numbers(), ref_r)
        ref_lengths = np.linalg.norm(ref_r[bonds[:, 0]] - ref_r[bonds[:, 1]], axis=-1)
    edges = np.asarray(
        [e for e in cfg.dimer_r_bins_A if e < float(cfg.dimer_com_cutoff_A)]
        + [float(cfg.dimer_com_cutoff_A)],
        dtype=np.float64,
    )
    for i_frame, frame in enumerate(frames):
        phase = str(frame.info.get("phase", "")).strip()
        tag = f":{phase}" if phase else ""
        seed = frame.info.get("seed")
        step = frame.info.get("step")
        group = dict(
            group_seed=None if seed is None else int(seed),
            group_file=0 if frame_files is None else int(frame_files[i_frame]),
            group_frame=int(i_frame),
            group_step=None if step is None else int(step),
            group_phase=phase or None,
        )
        mols = whole_molecules(frame, apm)
        z_mol = np.asarray(frame.get_atomic_numbers(), dtype=int)[:apm]
        com = mols.mean(axis=1)  # unweighted centroid, as in ml_switch_scale
        cell = np.asarray(frame.get_cell(), dtype=np.float64)
        periodic = bool(np.any(frame.get_pbc())) and abs(np.linalg.det(cell)) > 1e-9
        n_mol = mols.shape[0]
        if bonds is not None:
            ok = ~damaged_molecules(frame, apm, bonds, ref_lengths, float(cfg.max_bond_stretch_A))[0]
        else:
            ok = np.ones(n_mol, dtype=bool)
        n_broken += int((~ok).sum())

        for i in rng.permutation(np.nonzero(ok)[0])[: int(cfg.max_monomers_per_frame)]:
            geos.append(
                Geometry(
                    numbers=z_mol.copy(),
                    positions=mols[i] - com[i],
                    kind="monomer",
                    source=SOURCE_BOX_MONOMER + tag,
                    r_com_A=None,
                    atoms_per_monomer=(apm,),
                    **group,
                )
            )

        pairs: list[tuple[int, int, np.ndarray, float]] = []
        for i in range(n_mol):
            if not ok[i]:
                continue
            d = com[i + 1 :] - com[i]
            if periodic:
                d = _mic(d, cell)
            r = np.linalg.norm(d, axis=1)
            for k in np.nonzero(r < float(cfg.dimer_com_cutoff_A))[0]:
                j = i + 1 + int(k)
                if not ok[j]:
                    continue
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
                    **group,
                )
            )
    if stats is not None:
        stats["n_broken_molecules"] = n_broken
        stats["topology_filter"] = bonds is not None
    return geos
