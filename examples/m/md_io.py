"""Trajectory / geometry writers for ``examples/m`` MD smokes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory


def atoms_from_zr(z: np.ndarray, r: np.ndarray) -> Atoms:
    return Atoms(numbers=np.asarray(z, dtype=int), positions=np.asarray(r, dtype=float))


def atoms_with_efv(
    z: np.ndarray,
    r: np.ndarray,
    *,
    energy: float | None = None,
    forces: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
) -> Atoms:
    """Build an ``Atoms`` with optional energy/forces (SPC) and velocities."""
    atoms = atoms_from_zr(z, r)
    if velocities is not None:
        atoms.set_velocities(np.asarray(velocities, dtype=float))
    if energy is not None or forces is not None:
        kw: dict = {}
        if energy is not None:
            kw["energy"] = float(energy)
        if forces is not None:
            kw["forces"] = np.asarray(forces, dtype=float)
        atoms.calc = SinglePointCalculator(atoms, **kw)
    return atoms


def attach_ase_trajectory(
    dyn,
    atoms: Atoms,
    path: Path | str,
    *,
    interval: int = 1,
) -> Trajectory:
    """Attach an ASE ``.traj`` writer that stores energy, forces, and velocities.

    ASE's ``Trajectory.write`` already persists calculator results and momenta
    when present; we wrap write so each frame re-reads E/F from the live
    calculator (ensuring results are populated) before serializing.
    """
    traj_path = Path(path)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(traj_path), "w", atoms)

    def _write_efv() -> None:
        # Touch energy/forces so calculator.results is filled before write.
        _ = float(atoms.get_potential_energy())
        _ = np.asarray(atoms.get_forces())
        traj.write(atoms)

    dyn.attach(_write_efv, interval=max(1, int(interval)))
    return traj


def write_xyz_frames(
    path: Path | str,
    z: np.ndarray,
    frames: Sequence[np.ndarray],
    *,
    comment_prefix: str = "frame",
    energies: Sequence[float] | None = None,
) -> Path:
    """Write a multi-frame XYZ (one structure per frame)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms_list = []
    for i, frame in enumerate(frames):
        atoms = atoms_from_zr(z, np.asarray(frame, dtype=float))
        comment = f"{comment_prefix} {i}"
        if energies is not None and i < len(energies):
            comment = f"{comment} E={float(energies[i]):.8f}"
        atoms.info["comment"] = comment
        atoms_list.append(atoms)
    ase_write(str(out), atoms_list)
    return out


def write_final_geometry(
    out_dir: Path | str,
    z: np.ndarray,
    r: np.ndarray,
    *,
    stem: str = "final",
    energy: float | None = None,
    forces: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
) -> dict[str, str]:
    """Write ``final.npz`` + ``final.xyz`` (+ ``final.traj`` when E/F/V given)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.int32)
    npz_path = out / f"{stem}.npz"
    xyz_path = out / f"{stem}.xyz"
    payload: dict[str, np.ndarray] = {"Z": z, "R": r}
    if energy is not None:
        payload["E"] = np.asarray([energy], dtype=np.float64)
    if forces is not None:
        payload["F"] = np.asarray(forces, dtype=np.float64)
    if velocities is not None:
        payload["V"] = np.asarray(velocities, dtype=np.float64)
    np.savez(npz_path, **payload)
    ase_write(str(xyz_path), atoms_from_zr(z, r))
    artifacts = {"npz": npz_path.name, "xyz": xyz_path.name}
    if energy is not None or forces is not None or velocities is not None:
        traj_path = out / f"{stem}.traj"
        atoms = atoms_with_efv(
            z, r, energy=energy, forces=forces, velocities=velocities
        )
        with Trajectory(str(traj_path), "w", atoms) as traj:
            traj.write(atoms)
        artifacts["traj_final"] = traj_path.name
    return artifacts


def write_jaxmd_trajectory(
    out_dir: Path | str,
    z: np.ndarray,
    frames: Sequence[np.ndarray],
    *,
    energies: Sequence[float],
    forces: Sequence[np.ndarray],
    velocities: Sequence[np.ndarray],
    traj_name: str = "md.traj",
    xyz_name: str = "md.xyz",
) -> dict[str, str]:
    """Write ASE ``.traj`` (E/F/V) + multi-frame ``.xyz`` from JAX-MD samples."""
    if not (len(frames) == len(energies) == len(forces) == len(velocities)):
        raise ValueError(
            "frames/energies/forces/velocities length mismatch: "
            f"{len(frames)}, {len(energies)}, {len(forces)}, {len(velocities)}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_path = out / traj_name
    xyz_path = out / xyz_name
    atoms0 = atoms_with_efv(
        z,
        frames[0],
        energy=float(energies[0]),
        forces=forces[0],
        velocities=velocities[0],
    )
    with Trajectory(str(traj_path), "w", atoms0) as traj:
        for frame, e, f, v in zip(frames, energies, forces, velocities):
            atoms = atoms_with_efv(
                z, frame, energy=float(e), forces=f, velocities=v
            )
            traj.write(atoms)
    write_xyz_frames(xyz_path, z, frames, energies=energies)
    return {"traj": traj_path.name, "xyz": xyz_path.name}
