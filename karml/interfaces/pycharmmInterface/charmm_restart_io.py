"""CHARMM restart (.res) read/write without ``WRITE RESTART`` script commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _format_fortran_restart_float(value: float) -> str:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"non-finite restart value: {v}")
    return f"{v:24.15E}".replace("E", "D")


def lattice_type_for_cell(cell: np.ndarray) -> str:
    """CHARMM ``XTLTYP`` token for a Cartesian 3×3 cell."""
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    off = matrix - np.diag(np.diag(matrix))
    if float(np.max(np.abs(off))) > 1.0e-6:
        return "TRIC"
    lengths = np.diag(matrix)
    if float(np.max(np.abs(lengths - lengths[0]))) <= 1.0e-6 * max(float(abs(lengths[0])), 1.0):
        return "CUBI"
    return "ORTH"


def format_rest_header(*, ivers: int = 1, ldyna: int = 1, xtltyp: str = "") -> str:
    """``REST`` line ``(A4,2I6,2X,A4)``: version, leap-frog flag, lattice token.

    ``LDYNA`` is the integrator flag, not the step counter. A blank lattice
    token leaves ``READYN`` skipping the crystal block.
    """
    token = str(xtltyp or "").upper()[:4]
    return f"REST{int(ivers):6d}{int(ldyna):6d}  {token:<4s}"


def rest_header_lattice_token(header: str) -> str:
    """Lattice token in columns 19–22 of a Fortran ``REST`` header, or ``''``."""
    if not header.startswith("REST") or len(header) < 16:
        return ""
    try:
        int(header[4:10])
        int(header[10:16])
    except ValueError:
        return ""
    return header[18:22].strip() if len(header) >= 22 else ""


def set_rest_header_lattice_token(header: str, xtltyp: str) -> str:
    """Write ``xtltyp`` into an existing Fortran ``REST`` header.

    Headers that are not ``(A4,2I6,2X,A4)`` (synthetic title lines) are left
    unchanged so a version-48 restart is not rewritten as version 1.
    """
    if rest_header_lattice_token(header) == "" and not _rest_header_has_lattice_slot(header):
        return header
    token = str(xtltyp or "").upper()[:4]
    padded = header.ljust(22)
    return padded[:18] + f"{token:<4s}" + padded[22:]


def _rest_header_has_lattice_slot(header: str) -> bool:
    if not header.startswith("REST") or len(header) < 16:
        return False
    try:
        int(header[4:10])
        int(header[10:16])
    except ValueError:
        return False
    return True


def crystal_parameter_lines(cell: np.ndarray) -> list[str]:
    """``!CRYSTAL PARAMETERS`` plus the full Cartesian 3×3, including shear."""
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"crystal cell must be a finite 3×3, got {matrix.shape}")
    lines = [" !CRYSTAL PARAMETERS"]
    for row in matrix:
        lines.append("".join(_format_fortran_restart_float(float(v)) for v in row))
    return lines


def parse_restart_crystal(text: str) -> tuple[str, np.ndarray]:
    """Return ``(XTLTYP, 3×3)`` from a restart we wrote.

    The matrix is the three lines under ``!CRYSTAL PARAMETERS``. The lattice
    token comes from the ``REST`` header when that header has the Fortran slot.
    """
    lines = text.splitlines()
    token = rest_header_lattice_token(lines[0]) if lines else ""
    idx = next(
        (i for i, ln in enumerate(lines) if "CRYSTAL PARAMETERS" in ln.upper()),
        None,
    )
    if idx is None or idx + 3 >= len(lines):
        raise ValueError("restart has no 3×3 !CRYSTAL PARAMETERS block")
    rows: list[list[float]] = []
    for ln in lines[idx + 1 : idx + 4]:
        vals = []
        for raw in ln.replace("D", "E").replace("d", "E").split():
            vals.append(float(raw))
        if len(vals) != 3:
            raise ValueError(f"crystal row must hold 3 values, got {ln!r}")
        rows.append(vals)
    matrix = np.asarray(rows, dtype=float)
    if not token:
        token = lattice_type_for_cell(matrix)
    return token, matrix


def _resolve_restart_cell(
    cell: np.ndarray | None,
    *,
    include_crystal: bool,
) -> np.ndarray | None:
    """Cartesian cell to write, or ``None`` when the restart stays vacuum."""
    if not include_crystal:
        return None
    if cell is not None:
        return np.asarray(cell, dtype=float).reshape(3, 3)
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            _read_charmm_box_sides_A,
        )

        lx, ly, lz = _read_charmm_box_sides_A()
        if min(lx, ly, lz) > 0.0:
            return np.diag([lx, ly, lz])
    except Exception:
        return None
    return None


def _restart_section_coord_lines(arr: np.ndarray) -> list[str]:
    flat = np.asarray(arr, dtype=float).reshape(-1)
    lines: list[str] = []
    for i in range(0, len(flat), 3):
        chunk = flat[i : i + 3]
        lines.append(
            "".join(_format_fortran_restart_float(float(v)) for v in chunk)
        )
    return lines


def _restart_natom_counter_line(
    *,
    natom: int,
    nstep: int = 0,
    nsavc: int = 1,
    nsavv: int = 0,
    jhstrt: int = 0,
) -> str:
    """``!NATOM`` data line using Fortran ``I10`` columns (READYN-safe)."""
    fields = (
        int(natom),
        0,
        int(nstep),
        int(nsavc),
        int(nsavv),
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
    nsavv: int = 0,
    include_velocities: bool = True,
    include_crystal: bool = True,
    velocities_akma: np.ndarray | None = None,
    cell: np.ndarray | None = None,
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

    crystal_cell = _resolve_restart_cell(cell, include_crystal=include_crystal)
    xtltyp = lattice_type_for_cell(crystal_cell) if crystal_cell is not None else ""

    lines: list[str] = [
        # (A4,2I6,2X,A4): HDR, IVERS, LDYNA, XTLTYP.  LDYNA=1 is leap-frog, not a
        # step counter (a step here trips READYN's Verlet->leap-frog conversion, #219).
        format_rest_header(xtltyp=xtltyp),
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL",
        _restart_natom_counter_line(
            natom=natom,
            nstep=step,
            nsavc=int(nsavc),
            nsavv=int(nsavv),
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
        vel = velocities_akma
        if vel is None:
            try:
                from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
                    charmm_synced_velocities_akma,
                )

                vel = charmm_synced_velocities_akma()
            except Exception:
                vel = None
        if vel is None:
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
                lines.append(" !VX, VY, VZ")
                lines.extend(_restart_section_coord_lines(vel))

    if crystal_cell is not None:
        lines.extend(crystal_parameter_lines(crystal_cell))

    p.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    return p
