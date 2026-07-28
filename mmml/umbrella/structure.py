"""Load starting geometries for umbrella sampling (XYZ / PDB / NPZ)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

SeedMode = str  # "stretch" | "tile" | "frames"


def load_structure(
    path: Path | str,
    *,
    index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single geometry as ``(R, Z)`` with shapes ``(N, 3)`` and ``(N,)``.

    Supports:
    - ASE formats (``.xyz``, ``.pdb``, …)
    - MMML NPZ with ``R`` / ``Z`` (optionally multi-frame; ``index`` selects frame)
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"structure not found: {path}")

    if path.suffix.lower() == ".npz":
        return _load_from_npz(path, index=index)

    from ase.io import read

    atoms = read(str(path), index=index)
    if isinstance(atoms, list):
        atoms = atoms[0]
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    r = np.asarray(atoms.get_positions(), dtype=np.float64)
    return r, z


def load_structure_frames(
    path: Path | str,
    *,
    n_frames: int,
    start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``n_frames`` geometries as ``(R_multi, Z)`` with ``R_multi`` ``(K, N, 3)``.

    NPZ: consecutive frames from ``R``.
    ASE multi-frame files (XYZ traj / PDB): consecutive models.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"structure not found: {path}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1 (got {n_frames})")

    if path.suffix.lower() == ".npz":
        r_all, z = _load_npz_all(path)
        if r_all.shape[0] < start_index + n_frames:
            raise ValueError(
                f"NPZ {path} has {r_all.shape[0]} frames; need "
                f"{start_index + n_frames} for start_index={start_index}, "
                f"n_frames={n_frames}"
            )
        return r_all[start_index : start_index + n_frames].copy(), z

    from ase.io import read

    atoms_list = read(str(path), index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if len(atoms_list) < start_index + n_frames:
        raise ValueError(
            f"{path} has {len(atoms_list)} frames; need "
            f"{start_index + n_frames} for start_index={start_index}, "
            f"n_frames={n_frames}"
        )
    selected = atoms_list[start_index : start_index + n_frames]
    z = np.asarray(selected[0].get_atomic_numbers(), dtype=np.int32)
    n_atoms = int(len(z))
    r_multi = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    for k, atoms in enumerate(selected):
        zk = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        if zk.shape != z.shape or not np.array_equal(zk, z):
            raise ValueError(f"frame {start_index + k} atomic numbers differ from frame 0")
        r_multi[k] = np.asarray(atoms.get_positions(), dtype=np.float64)[:n_atoms]
    return r_multi, z


def stretch_distance_seed(
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    target_A: float,
    move_with: Sequence[int] | None = None,
) -> np.ndarray:
    """Set distance ``i–j`` to ``target_A`` by translating ``j`` (and friends).

    Atom ``i`` is held fixed. Atom ``j`` and every index in ``move_with`` are
    shifted by the same vector so a monomer/group can move rigidly.
    """
    r = np.asarray(positions, dtype=np.float64).copy()
    if target_A <= 0:
        raise ValueError(f"target_A must be > 0 (got {target_A})")
    disp = r[atom_j] - r[atom_i]
    dist = float(np.linalg.norm(disp))
    if dist < 1e-8:
        raise ValueError(
            f"cannot stretch atoms ({atom_i}, {atom_j}): current distance ~0"
        )
    u = disp / dist
    shift = (float(target_A) - dist) * u
    group = {int(atom_j)}
    if move_with:
        group.update(int(a) for a in move_with)
    for a in sorted(group):
        r[a] = r[a] + shift
    return r


def reflect_atoms_through_plane(
    positions: np.ndarray,
    hub: int,
    normal: np.ndarray,
    atom_indices: Sequence[int],
) -> np.ndarray:
    """Reflect ``atom_indices`` through the plane at ``hub`` with given normal."""
    r = np.asarray(positions, dtype=np.float64).copy()
    n = np.asarray(normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return r
    n = n / n_norm
    origin = r[int(hub)]
    for a in atom_indices:
        v = r[int(a)] - origin
        r[int(a)] = origin + v - 2.0 * float(np.dot(v, n)) * n
    return r


def sn2_progress_weight(
    d_leaving: float,
    d_nucleophile: float,
    d_leaving_ref: float,
    d_nucleophile_ref: float,
    *,
    span_A: float = 3.0,
) -> float:
    """Map SN2-like ξ = r_LG−r_Nu from the reference toward product into [0, 1]."""
    xi = float(d_leaving) - float(d_nucleophile)
    xi_ref = float(d_leaving_ref) - float(d_nucleophile_ref)
    if span_A <= 0:
        raise ValueError(f"span_A must be > 0 (got {span_A})")
    return float(np.clip((xi - xi_ref) / float(span_A), 0.0, 1.0))


def stretch_two_distances(
    positions: np.ndarray,
    pair_x: tuple[int, int],
    target_x: float,
    pair_y: tuple[int, int],
    target_y: float,
    move_with_x: Sequence[int] | None = None,
    move_with_y: Sequence[int] | None = None,
    invert_with: Sequence[int] | None = None,
) -> np.ndarray:
    """Stretch two distance CVs (shared hub fixes the common atom).

    Optional ``invert_with`` blends a Walden-like reflection of those atoms
    through the hub plane normal to (nucleophile − leaving), weighted by SN2
    progress. Use for CH₃ hydrogens on a shared-carbon 2D grid.
    """
    a, b = int(pair_x[0]), int(pair_x[1])
    c, d = int(pair_y[0]), int(pair_y[1])
    shared = set(pair_x) & set(pair_y)
    if len(shared) == 1:
        hub = int(next(iter(shared)))
        other_x = b if a == hub else a
        other_y = d if c == hub else c
        r_ref = np.asarray(positions, dtype=np.float64)
        d_x_ref = float(np.linalg.norm(r_ref[other_x] - r_ref[hub]))
        d_y_ref = float(np.linalg.norm(r_ref[other_y] - r_ref[hub]))
        r = r_ref.copy()
        r = stretch_distance_seed(r, hub, other_x, float(target_x), move_with=move_with_x)
        r = stretch_distance_seed(r, hub, other_y, float(target_y), move_with=move_with_y)
        if invert_with:
            # Closer reference contact = leaving group; farther = nucleophile
            # (SN2 reactant: LG bonded, Nu approaching from the back side).
            if d_x_ref <= d_y_ref:
                leaving, nuc = other_x, other_y
                d_lg, d_nu = float(target_x), float(target_y)
                d_lg_ref, d_nu_ref = d_x_ref, d_y_ref
            else:
                leaving, nuc = other_y, other_x
                d_lg, d_nu = float(target_y), float(target_x)
                d_lg_ref, d_nu_ref = d_y_ref, d_x_ref
            w = sn2_progress_weight(d_lg, d_nu, d_lg_ref, d_nu_ref)
            if w > 0.0:
                r_inv = reflect_atoms_through_plane(
                    r, hub, r[nuc] - r[leaving], invert_with
                )
                for idx in invert_with:
                    i = int(idx)
                    r[i] = (1.0 - w) * r[i] + w * r_inv[i]
        return r
    r = stretch_distance_seed(positions, a, b, target_x, move_with=move_with_x)
    r = stretch_distance_seed(r, c, d, target_y, move_with=move_with_y)
    return r



def pack_window_seeds(
    *,
    positions: np.ndarray,
    atom_pairs: Sequence[tuple[int, int]],
    targets_per_cv: Sequence[Sequence[float]],
    seed_mode: SeedMode = "stretch",
    frames: np.ndarray | None = None,
    move_groups: Sequence[Sequence[int]] | None = None,
    invert_with: Sequence[int] | None = None,
) -> np.ndarray:
    """Build packed ``(K*N, 3)`` initial coordinates for ``K`` umbrella windows.

    - ``stretch``: fix ``atom_i``, translate ``atom_j`` (+ ``move_groups``) to ξ₀
    - ``tile``: duplicate the reference geometry unchanged (legacy; can explode)
    - ``frames``: use ``frames`` shape ``(K, N, 3)`` as window seeds

    ``invert_with`` (2D stretch only): blend Walden inversion of those atoms.
    """
    from mmml.umbrella.energy import pack_positions

    pairs = tuple((int(i), int(j)) for i, j in atom_pairs)
    targets = [tuple(float(x) for x in row) for row in targets_per_cv]
    if not pairs or not targets:
        raise ValueError("atom_pairs and targets_per_cv are required")
    k = len(targets[0])
    if any(len(row) != k for row in targets):
        raise ValueError("all CV target rows must have the same length K")
    if len(pairs) != len(targets):
        raise ValueError("atom_pairs length must match targets_per_cv length")
    if move_groups is None:
        groups: list[tuple[int, ...]] = [() for _ in pairs]
    else:
        if len(move_groups) != len(pairs):
            raise ValueError("move_groups length must match atom_pairs length")
        groups = [tuple(int(a) for a in g) for g in move_groups]
    invert = tuple(int(a) for a in invert_with) if invert_with else ()

    if seed_mode == "tile":
        return pack_positions(positions, k)
    if seed_mode == "frames":
        if frames is None:
            raise ValueError("seed_mode='frames' requires frames array")
        fr = np.asarray(frames, dtype=np.float64)
        if fr.ndim != 3 or fr.shape[0] != k:
            raise ValueError(
                f"frames must have shape (K, N, 3) with K={k}, got {fr.shape}"
            )
        return fr.reshape(k * fr.shape[1], 3)
    if seed_mode != "stretch":
        raise ValueError(
            f"unknown seed_mode {seed_mode!r}; expected stretch|tile|frames"
        )

    r0 = np.asarray(positions, dtype=np.float64)
    n_atoms = int(r0.shape[0])
    out = np.zeros((k, n_atoms, 3), dtype=np.float64)
    if len(pairs) == 1:
        i, j = pairs[0]
        for wid in range(k):
            out[wid] = stretch_distance_seed(
                r0, i, j, targets[0][wid], move_with=groups[0]
            )
    elif len(pairs) == 2:
        for wid in range(k):
            out[wid] = stretch_two_distances(
                r0,
                pairs[0],
                targets[0][wid],
                pairs[1],
                targets[1][wid],
                move_with_x=groups[0],
                move_with_y=groups[1],
                invert_with=invert,
            )
    else:
        raise ValueError(f"only 1D/2D distance umbrellas supported (got {len(pairs)} CVs)")
    return out.reshape(k * n_atoms, 3)



def _trim_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.int32)
    if z.ndim == 2:
        z = z[0]
    n_atoms = int(np.sum(z > 0)) if np.any(z > 0) else int(len(z))
    return z[:n_atoms]


def _load_from_npz(path: Path, *, index: int) -> tuple[np.ndarray, np.ndarray]:
    r_all, z = _load_npz_all(path)
    if index < 0 or index >= r_all.shape[0]:
        raise ValueError(
            f"structure index {index} out of range for NPZ with {r_all.shape[0]} frames"
        )
    return r_all[index].copy(), z


def _load_npz_all(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "R" not in data or "Z" not in data:
        keys = sorted(data.files)
        raise KeyError(
            f"NPZ {path} must contain 'R' and 'Z' keys (found {keys})"
        )
    z = _trim_z(np.asarray(data["Z"]))
    n_atoms = int(len(z))
    r_all = np.asarray(data["R"], dtype=np.float64)
    if r_all.ndim == 2:
        r_all = r_all[np.newaxis, ...]
    if r_all.ndim != 3:
        raise ValueError(f"NPZ R must be (N,3) or (F,N,3), got shape {r_all.shape}")
    return r_all[:, :n_atoms, :].copy(), z
