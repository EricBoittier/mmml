"""Rigid dimer/trimer placement and 2D COM-distance scan grids."""

from __future__ import annotations

from typing import Callable

import numpy as np

ScanEvalFn = Callable[[np.ndarray], dict[str, float]]


def monomer_offsets(atoms_per: list[int]) -> np.ndarray:
    off = np.zeros(len(atoms_per) + 1, dtype=int)
    for i, n in enumerate(atoms_per):
        off[i + 1] = off[i] + int(n)
    return off


def monomer_com(pos: np.ndarray, off: int, n: int) -> np.ndarray:
    return np.mean(pos[off : off + n], axis=0)


def min_intra_dist(pos: np.ndarray, off: int, n: int) -> float:
    sub = pos[off : off + n]
    if n < 2:
        return float("inf")
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(d[iu].min())


def min_inter_dist(pos: np.ndarray, off_a: int, na: int, off_b: int, nb: int) -> float:
    a = pos[off_a : off_a + na]
    b = pos[off_b : off_b + nb]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min())


def rigid_shift_monomer(
    pos: np.ndarray,
    ref: np.ndarray,
    off: int,
    n: int,
    target_com: np.ndarray,
) -> None:
    ref_com = monomer_com(ref, off, n)
    pos[off : off + n] = ref[off : off + n] + (target_com - ref_com)


def place_trimer(
    ref: np.ndarray,
    atoms_per: list[int],
    d01: float,
    d02: float,
    angle_02_rad: float,
) -> np.ndarray:
    """Rigid-body move monomers 1 and, when present, 2 relative to monomer 0 COM."""
    if len(atoms_per) < 2:
        raise ValueError(f"scan requires at least 2 monomers, got {len(atoms_per)}")
    pos = np.array(ref, dtype=np.float64, copy=True)
    off = monomer_offsets(atoms_per)
    com0 = monomer_com(ref, int(off[0]), int(atoms_per[0]))
    target1 = com0 + np.array([d01, 0.0, 0.0], dtype=float)
    rigid_shift_monomer(pos, ref, int(off[1]), int(atoms_per[1]), target1)
    if len(atoms_per) >= 3:
        target2 = com0 + d02 * np.array(
            [np.cos(angle_02_rad), np.sin(angle_02_rad), 0.0], dtype=float
        )
        rigid_shift_monomer(pos, ref, int(off[2]), int(atoms_per[2]), target2)
    return pos


def com_distances(pos: np.ndarray, atoms_per: list[int]) -> np.ndarray:
    off = monomer_offsets(atoms_per)
    coms = [monomer_com(pos, int(off[i]), int(atoms_per[i])) for i in range(len(atoms_per))]
    pairs = [
        float(np.linalg.norm(coms[b] - coms[a]))
        for a in range(len(coms))
        for b in range(a + 1, len(coms))
    ]
    return np.array(pairs, dtype=np.float64)


def distance_report(pos: np.ndarray, atoms_per: list[int]) -> dict[str, float]:
    off = monomer_offsets(atoms_per)
    out: dict[str, float] = {}
    for i, n in enumerate(atoms_per):
        out[f"min_intra_m{i}"] = min_intra_dist(pos, int(off[i]), int(n))
    for a in range(len(atoms_per)):
        for b in range(a + 1, len(atoms_per)):
            out[f"min_inter_{a}{b}"] = min_inter_dist(
                pos, int(off[a]), int(atoms_per[a]), int(off[b]), int(atoms_per[b])
            )
    com_d = com_distances(pos, atoms_per)
    pair_index = 0
    for a in range(len(atoms_per)):
        for b in range(a + 1, len(atoms_per)):
            out[f"com_d{a}{b}"] = float(com_d[pair_index])
            pair_index += 1
    return out


def atoms_per_monomer_from_psf() -> list[int]:
    """Return atom counts per CHARMM resid (one monomer per resid)."""
    import pycharmm.psf as psf

    natom = int(psf.get_natom()) if hasattr(psf, "get_natom") else 0
    ibase = np.asarray(psf.get_ibase(), dtype=int) if hasattr(psf, "get_ibase") else np.asarray([], dtype=int)
    if ibase.size:
        counts = np.diff(np.concatenate(([0], ibase))).astype(int).tolist()
        if natom > 0 and sum(counts) != natom:
            raise ValueError(f"Invalid PSF residue boundaries from ibase={ibase.tolist()}, natom={natom}")
        if any(c <= 0 for c in counts):
            raise ValueError(f"Invalid resid atom counts from PSF ibase={ibase.tolist()}: {counts}")
        return counts

    resids = np.asarray(psf.get_resid(), dtype=int) if hasattr(psf, "get_resid") else np.asarray([], dtype=int)
    if resids.size == 0:
        raise ValueError("PSF has no atoms")
    if natom > 0 and resids.size != natom:
        raise ValueError(
            f"PSF get_resid returned {resids.size} entries for {natom} atoms; "
            "get_ibase is required to recover residue atom counts on this PyCHARMM build"
        )
    n_res = int(resids.max())
    counts = [int(np.sum(resids == resid)) for resid in range(1, n_res + 1)]
    if any(c <= 0 for c in counts):
        raise ValueError(f"Invalid resid atom counts from PSF: {counts}")
    return counts


def run_scan_2d(
    eval_fn: ScanEvalFn,
    ref_pos: np.ndarray,
    atoms_per: list[int],
    d1_grid: np.ndarray,
    d2_grid: np.ndarray,
    *,
    angle_02_deg: float,
    metric_keys: tuple[str, ...],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate ``eval_fn(pos)`` on a d01 × d02 grid; store selected metric keys."""
    if len(atoms_per) < 2:
        raise ValueError(f"scan requires at least 2 monomers, got {len(atoms_per)}")
    n1, n2 = len(d1_grid), len(d2_grid)
    store = {k: np.full((n1, n2), np.nan, dtype=np.float64) for k in metric_keys}
    angle = np.deg2rad(float(angle_02_deg))
    for i, d01 in enumerate(d1_grid):
        for j, d02 in enumerate(d2_grid):
            pos = place_trimer(ref_pos, atoms_per, float(d01), float(d02), angle)
            dist = distance_report(pos, atoms_per)
            rec = eval_fn(pos)
            for k in metric_keys:
                if k.startswith("min_") and k in dist:
                    store[k][i, j] = dist[k]
                elif k in rec:
                    store[k][i, j] = rec[k]
        if progress is not None:
            progress(i + 1, n1)
    return store


def default_scan_2d_metric_keys(*, include_mm: bool = True) -> tuple[str, ...]:
    keys = (
        "energy_kcal",
        "internal_E_kcal",
        "ml_2b_E_kcal",
        "dH_kcal",
        "charmm_ENER_kcal",
        "charmm_USER_kcal",
        "callback_energy_kcal",
        "min_inter_01",
        "min_inter_02",
        "min_inter_12",
    )
    if include_mm:
        return keys + ("mm_E_kcal", "charmm_VDW_kcal", "charmm_ELEC_kcal")
    return keys
