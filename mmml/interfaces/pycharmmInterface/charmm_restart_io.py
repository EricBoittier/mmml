"""CHARMM restart (.res) read/write without ``WRITE RESTART`` script commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _format_fortran_restart_float(value: float) -> str:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"non-finite restart value: {v}")
    return f"{v:.15E}".replace("E", "D")


def _restart_section_coord_lines(arr: np.ndarray) -> list[str]:
    flat = np.asarray(arr, dtype=float).reshape(-1)
    lines: list[str] = []
    for i in range(0, len(flat), 3):
        chunk = flat[i : i + 3]
        lines.append(
            " "
            + " ".join(_format_fortran_restart_float(float(v)) for v in chunk)
        )
    return lines


def _restart_natom_counter_line(
    *,
    natom: int,
    nstep: int = 0,
    nsavc: int = 1,
    jhstrt: int = 0,
) -> str:
    """``!NATOM`` data line using Fortran ``I10`` columns (READYN-safe)."""
    fields = (
        int(natom),
        0,
        int(nstep),
        int(nsavc),
        0,
        int(jhstrt),
        0,
        0,
        0,
    )
    return "".join(f"{v:>10d}" for v in fields)


def write_charmm_restart_from_memory(
    path: Path,
    *,
    positions: np.ndarray | None = None,
    title: str = "MMML snapshot",
    global_step: int | None = None,
    nsavc: int = 1,
    include_velocities: bool = True,
    include_crystal: bool = True,
) -> Path:
    """Write a CHARMM ``.res`` from coordinates (no ``WRITE RESTART`` script).

    MPI-linked ``libcharmm.so`` under ``mpirun`` can abort in Fortran ``parse.F90`` on
    ``write restart`` (gfortrantmp EOF on unit 90) even when PSF/PDB C API writes work.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    if positions is not None:
        pos = np.asarray(positions, dtype=float)
    else:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            get_charmm_positions_array,
        )

        pos = np.asarray(get_charmm_positions_array(), dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"restart write: positions must be (N, 3), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("restart write: CHARMM coordinates must be finite")

    natom = int(pos.shape[0])
    step = 0 if global_step is None else max(0, int(global_step))
    if global_step is None and p.is_file():
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
            read_restart_last_step,
        )

        prior = read_restart_last_step(p)
        if prior is not None and prior > 0:
            step = int(prior)

    lines: list[str] = [
        f"REST     1{step:10d}",
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL",
        _restart_natom_counter_line(
            natom=natom,
            nstep=step,
            nsavc=int(nsavc),
            jhstrt=step,
        ),
    ]
    if title:
        lines.extend(
            [
                "       1 !NTITLE followed by title",
                f"* {title}",
            ]
        )
    lines.append(" !X, Y, Z")
    lines.extend(_restart_section_coord_lines(pos))

    if include_velocities:
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
                _charmm_velocities_array,
            )

            vel = _charmm_velocities_array()
        except Exception:
            vel = None
        if vel is not None:
            vel = np.asarray(vel, dtype=float)
            if vel.shape == pos.shape and np.all(np.isfinite(vel)):
                lines.append(" !VELOCITIES")
                lines.extend(_restart_section_coord_lines(vel))

    if include_crystal:
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                _is_cubic_box_sides,
                _read_charmm_box_sides_A,
            )

            lx, ly, lz = _read_charmm_box_sides_A()
            if min(lx, ly, lz) > 0.0 and _is_cubic_box_sides(lx, ly, lz):
                side = float((lx + ly + lz) / 3.0)
                lines.append(" !CRYSTAL PARAMETERS")
                z = _format_fortran_restart_float(0.0)
                s = _format_fortran_restart_float(side)
                lines.append(f" {s} {z} {z}")
                lines.append(f" {z} {s} {z}")
                lines.append(f" {z} {z} {s}")
        except Exception:
            pass

    p.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return p
