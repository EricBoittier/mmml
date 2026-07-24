"""Trajectory / geometry writers for ``examples/m`` MD smokes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory


def atoms_from_zr(z: np.ndarray, r: np.ndarray) -> Atoms:
    return Atoms(numbers=np.asarray(z, dtype=int), positions=np.asarray(r, dtype=float))


def attach_ase_trajectory(
    dyn,
    atoms: Atoms,
    path: Path | str,
    *,
    interval: int = 1,
) -> Trajectory:
    """Attach an ASE ``.traj`` writer to an MD object; returns the open Trajectory."""
    traj_path = Path(path)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(traj_path), "w", atoms)
    dyn.attach(traj.write, interval=max(1, int(interval)))
    return traj


def write_xyz_frames(
    path: Path | str,
    z: np.ndarray,
    frames: Sequence[np.ndarray],
    *,
    comment_prefix: str = "frame",
) -> Path:
    """Write a multi-frame XYZ (one structure per frame)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms_list = [
        atoms_from_zr(z, np.asarray(frame, dtype=float)) for frame in frames
    ]
    for i, atoms in enumerate(atoms_list):
        atoms.info["comment"] = f"{comment_prefix} {i}"
    ase_write(str(out), atoms_list)
    return out


def write_final_geometry(
    out_dir: Path | str,
    z: np.ndarray,
    r: np.ndarray,
    *,
    stem: str = "final",
) -> dict[str, str]:
    """Write ``final.npz`` + ``final.xyz``; return relative path map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.int32)
    npz_path = out / f"{stem}.npz"
    xyz_path = out / f"{stem}.xyz"
    np.savez(npz_path, Z=z, R=r)
    ase_write(str(xyz_path), atoms_from_zr(z, r))
    return {"npz": npz_path.name, "xyz": xyz_path.name}


def write_jaxmd_trajectory(
    out_dir: Path | str,
    z: np.ndarray,
    frames: Sequence[np.ndarray],
    *,
    traj_name: str = "md.traj",
    xyz_name: str = "md.xyz",
) -> dict[str, str]:
    """Write ASE ``.traj`` + multi-frame ``.xyz`` from collected JAX-MD positions."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_path = out / traj_name
    xyz_path = out / xyz_name
    atoms0 = atoms_from_zr(z, frames[0])
    with Trajectory(str(traj_path), "w", atoms0) as traj:
        for frame in frames:
            atoms = atoms_from_zr(z, frame)
            traj.write(atoms)
    write_xyz_frames(xyz_path, z, frames)
    return {"traj": traj_path.name, "xyz": xyz_path.name}
