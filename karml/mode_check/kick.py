"""Short Verlet kick + FFT peak for a selected bond stretch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.units import fs as ASE_FS

# fs⁻¹ → cm⁻¹
_FS_INV_TO_CM_INV = 1.0e15 / 2.99792458e10


def kick_bond_fft(
    atoms: Atoms,
    atom_i: int,
    atom_j: int,
    *,
    kick_delta_A: float = 0.03,
    timestep_fs: float = 0.1,
    n_steps: int = 500,
    output_r_path: Path | None = None,
) -> dict[str, Any]:
    """Displace a bond, integrate short NVE, return FFT peak of r(t).

    Uses a copy of ``atoms`` so the caller's geometry is unchanged. The copy
    reuses ``atoms.calc`` (ASE calculators are typically attached by reference).
    """
    work = atoms.copy()
    work.calc = atoms.calc
    pos = work.get_positions()
    vec = pos[atom_j] - pos[atom_i]
    r0 = float(np.linalg.norm(vec))
    if r0 <= 0.0:
        raise ValueError("zero bond length")
    u = vec / r0
    pos = pos.copy()
    pos[atom_j] = pos[atom_i] + u * (r0 + float(kick_delta_A))
    work.set_positions(pos)
    work.set_velocities(np.zeros_like(pos))

    dt_fs = float(timestep_fs)
    dyn = VelocityVerlet(work, timestep=dt_fs * ASE_FS)
    rs: list[float] = []
    for _ in range(int(n_steps)):
        dyn.run(1)
        p = work.get_positions()
        rs.append(float(np.linalg.norm(p[atom_j] - p[atom_i])))
    r_arr = np.asarray(rs, dtype=float)
    if output_r_path is not None:
        output_r_path = Path(output_r_path)
        output_r_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_r_path, r_arr)

    x = r_arr - r_arr.mean()
    window = np.hanning(len(x))
    ft = np.abs(np.fft.rfft(x * window))
    freq_cm = np.fft.rfftfreq(len(x), d=dt_fs) * _FS_INV_TO_CM_INV
    if ft.size < 2:
        peak = float("nan")
    else:
        peak = float(freq_cm[1:][int(np.argmax(ft[1:]))])
    return {
        "atom_i": int(atom_i),
        "atom_j": int(atom_j),
        "fft_peak_cm": peak,
        "r_std": float(r_arr.std()),
        "r_min": float(r_arr.min()),
        "r_max": float(r_arr.max()),
        "n_steps": int(n_steps),
        "timestep_fs": dt_fs,
        "kick_delta_A": float(kick_delta_A),
        "ASE_FS": float(ASE_FS),
        "r_path": str(output_r_path) if output_r_path is not None else None,
    }
