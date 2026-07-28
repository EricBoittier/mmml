"""Load starting geometries for umbrella sampling (XYZ / PDB / NPZ)."""

from __future__ import annotations

from pathlib import Path

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

    # index=':' loads all; then slice
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
) -> np.ndarray:
    """Return a copy of ``positions`` with atoms ``i,j`` set to distance ``target_A``.

    The pair COM is held fixed; other atoms are unchanged.
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
    mid = 0.5 * (r[atom_i] + r[atom_j])
    half = 0.5 * float(target_A)
    r[atom_i] = mid - half * u
    r[atom_j] = mid + half * u
    return r


def pack_window_seeds(
    *,
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    targets_A: tuple[float, ...] | list[float],
    seed_mode: SeedMode = "stretch",
    frames: np.ndarray | None = None,
) -> np.ndarray:
    """Build packed ``(K*N, 3)`` initial coordinates for ``K`` umbrella windows.

    - ``stretch``: one reference geometry; stretch the CV pair to each ξ₀
    - ``tile``: duplicate the reference geometry unchanged (legacy; can explode)
    - ``frames``: use ``frames`` shape ``(K, N, 3)`` as window seeds
    """
    from mmml.umbrella.energy import pack_positions

    targets = tuple(float(x) for x in targets_A)
    k = len(targets)
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
    for wid, xi0 in enumerate(targets):
        out[wid] = stretch_distance_seed(r0, atom_i, atom_j, xi0)
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
