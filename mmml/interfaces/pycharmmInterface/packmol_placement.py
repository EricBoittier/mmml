"""Packmol input generation and cluster placement (cube or sphere; no PyCHARMM import)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PackmolPlacement = Literal["cube", "sphere"]

PACKMOL_PATH = Path("~/mmml/mmml/generate/packmol/packmol").expanduser()


def _binary_runs_on_host(path: Path) -> bool:
    """Return False for committed Linux ELFs on macOS (or other foreign binaries)."""
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        proc = subprocess.run(
            ["file", "-b", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if proc.returncode != 0:
        return True
    desc = proc.stdout.lower()
    if sys.platform == "darwin":
        return "mach-o" in desc
    if sys.platform.startswith("linux"):
        return "elf" in desc
    return True


def packmol_executable() -> str:
    from mmml.paths import bundled_file

    candidates = [
        bundled_file("generate", "packmol", "packmol"),
        bundled_file("generate", "packmol", "bin", "packmol"),
        Path(os.path.expanduser(str(PACKMOL_PATH))),
    ]
    for path in candidates:
        if _binary_runs_on_host(path):
            return str(path)
    found = shutil.which("packmol")
    if found:
        return found
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "packmol not found for this platform "
        f"(tried {tried}). Run: bash scripts/rebuild_charmm_mlpot.sh "
        "or bash scripts/rebuild_packmol.sh"
    )


@dataclass(frozen=True)
class PackmolRunResult:
    """Captured Packmol subprocess outcome (stdout/stderr parsed for summary)."""

    exit_code: int
    log_text: str
    inp_path: Path
    success: bool
    objective: float | None = None
    max_distance_violation: float | None = None
    max_constraint_violation: float | None = None
    error_message: str | None = None


PACKMOL_EXIT_LABELS: dict[int, str] = {
    170: "general error",
    171: "input error",
    172: "file open error",
    173: "failed to converge",
    174: "command-line error",
}


def _summarize_packmol_log_tail(log_text: str, *, max_lines: int = 12) -> str:
    lines = [ln.rstrip() for ln in (log_text or "").splitlines() if ln.strip()]
    if not lines:
        return "(no Packmol output captured)"
    interesting = [
        ln
        for ln in lines
        if re.search(
            r"ERROR|STOP:|WARNING|Success!|converge|GENCAN|outside|file-open",
            ln,
            flags=re.IGNORECASE,
        )
    ]
    if interesting:
        return "\n".join(interesting[-max_lines:])
    return "\n".join(lines[-max_lines:])


def packmol_failure_message(result: PackmolRunResult) -> str:
    if result.error_message:
        return result.error_message
    label = PACKMOL_EXIT_LABELS.get(int(result.exit_code))
    if label:
        return f"packmol {label} (exit {result.exit_code})"
    tail = _summarize_packmol_log_tail(result.log_text, max_lines=3)
    if tail != "(no Packmol output captured)":
        one_line = tail.replace("\n", " | ")
        return f"packmol failed (exit {result.exit_code}): {one_line}"
    return f"packmol failed with exit code {result.exit_code}"


def parse_packmol_log(log_text: str) -> dict[str, Any]:
    """Extract success metrics and the first error line from Packmol output."""
    log = log_text or ""
    success = "Success!" in log
    objective: float | None = None
    max_dist: float | None = None
    max_constraint: float | None = None
    error_message: str | None = None

    mobj = re.search(
        r"Final objective function value:\s*([-\d.eE+]+)",
        log,
    )
    if mobj:
        objective = float(mobj.group(1))
    dist_m = re.search(
        r"Maximum violation of target distance:\s*([-\d.eE+]+)",
        log,
    )
    if dist_m:
        max_dist = float(dist_m.group(1))
    constr_m = re.search(
        r"Maximum violation of the constraints:\s*([-\d.eE+]+)",
        log,
    )
    if constr_m:
        max_constraint = float(constr_m.group(1))

    for line in log.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if (
            "ERROR:" in stripped
            or stripped.startswith("STOP:")
            or "FILE-OPEN ERROR" in upper
            or "FORTRAN RUNTIME ERROR" in upper
            or "STOP " in upper
        ):
            error_message = stripped
            break
    if error_message is None:
        for line in log.splitlines():
            if "GENCAN loops achieved" in line or "failed to converge" in line.lower():
                error_message = line.strip()
                break

    return {
        "success": success,
        "objective": objective,
        "max_distance_violation": max_dist,
        "max_constraint_violation": max_constraint,
        "error_message": error_message,
    }


def _format_composition_summary(blocks: list[tuple[Path, int]]) -> str:
    parts = [
        f"{path.stem.upper()}:{int(count)}"
        for path, count in blocks
        if int(count) > 0
    ]
    return ", ".join(parts) if parts else "(none)"


def emit_packmol_build_summary(
    *,
    placement: PackmolPlacement,
    blocks: list[tuple[Path, int]] | None = None,
    composition: list[tuple[str, int]] | None = None,
    center: tuple[float, float, float],
    tolerance: float,
    seed: int | None,
    output_pdb: str | Path,
    inp_path: Path | None = None,
    cube_side: float | None = None,
    radius: float | None = None,
    sim_cell_side: float | None = None,
    box_sizing_source: str | None = None,
    packmol_padding_A: float | None = None,
    result: PackmolRunResult | None = None,
    cache_status: str | None = None,
    cache_key: str | None = None,
    n_atoms: int | None = None,
    span_A: tuple[float, float, float] | None = None,
    quiet: bool = False,
) -> None:
    """Emit one Rich table instead of Packmol's verbose Fortran stdout."""
    if composition is not None:
        comp_txt = ", ".join(f"{str(r).upper()}:{int(n)}" for r, n in composition)
    elif blocks is not None:
        comp_txt = _format_composition_summary(blocks)
    else:
        comp_txt = "(unknown)"

    rows: list[tuple[str, Any]] = [
        ("Placement", placement),
        ("Composition", comp_txt),
        (
            "Center (Å)",
            f"({float(center[0]):.3f}, {float(center[1]):.3f}, {float(center[2]):.3f})",
        ),
        ("Tolerance (Å)", f"{float(tolerance):.3f}"),
    ]
    if seed is not None:
        rows.append(("Seed", int(seed)))
    if placement == "cube" and cube_side is not None:
        rows.append(("Packmol cube (Å)", f"{float(cube_side):.3f}"))
    if placement == "sphere" and radius is not None:
        rows.append(("Packmol radius (Å)", f"{float(radius):.3f}"))
    if sim_cell_side is not None:
        rows.append(("Simulation cell (Å)", f"{float(sim_cell_side):.3f}"))
    if packmol_padding_A is not None:
        rows.append(("Cube padding (Å/side)", f"{float(packmol_padding_A):.3f}"))
    if box_sizing_source:
        rows.append(("Box sizing", box_sizing_source))
    if cache_status:
        cache_txt = cache_status if not cache_key else f"{cache_status} ({cache_key})"
        rows.append(("Cache", cache_txt))
    if inp_path is not None:
        rows.append(("Input", str(inp_path)))
    rows.append(("Output", str(output_pdb)))
    if result is not None:
        rows.append(("Status", "success" if result.success else "failed"))
        if result.objective is not None:
            rows.append(("Objective", f"{result.objective:.5e}"))
        if result.max_distance_violation is not None:
            rows.append(
                ("Max distance violation (Å)", f"{result.max_distance_violation:.6f}")
            )
        if result.max_constraint_violation is not None:
            rows.append(
                ("Max constraint violation", f"{result.max_constraint_violation:.5e}")
            )
        if result.error_message:
            rows.append(("Error", result.error_message))
    elif cache_status:
        rows.append(("Status", "cached"))
    if n_atoms is not None:
        rows.append(("Atoms", int(n_atoms)))
    if span_A is not None:
        rows.append(
            (
                "Span (Å)",
                f"x={span_A[0]:.1f} y={span_A[1]:.1f} z={span_A[2]:.1f}",
            )
        )

    border = "green"
    if result is not None and not result.success:
        border = "red"
    elif cache_status:
        border = "cyan"

    from mmml.utils.rich_report import emit_table

    emit_table("Packmol", rows, border_style=border, quiet=quiet)


def execute_packmol_script(packmol_input: str, inp_path: Path) -> PackmolRunResult:
    """Run Packmol with captured stdout/stderr (no Fortran spam on the terminal)."""
    inp_path = Path(inp_path).expanduser().resolve()
    os.makedirs(inp_path.parent, exist_ok=True)
    inp_path.write_text(packmol_input)
    packmol_bin = packmol_executable()
    proc = _run_packmol_subprocess(packmol_bin, inp_path)
    log_text = (proc.stdout or "") + (
        ("\n" + proc.stderr) if proc.stderr else ""
    )
    parsed = parse_packmol_log(log_text)
    success = int(proc.returncode) == 0 and bool(parsed["success"])
    result = PackmolRunResult(
        exit_code=int(proc.returncode),
        log_text=log_text,
        inp_path=inp_path,
        success=success,
        objective=parsed["objective"],
        max_distance_violation=parsed["max_distance_violation"],
        max_constraint_violation=parsed["max_constraint_violation"],
        error_message=parsed["error_message"],
    )
    if not result.success:
        from mmml.utils.rich_report import emit_panel, is_verbose

        tail = _summarize_packmol_log_tail(result.log_text)
        emit_panel("Packmol failed", tail, border_style="red")
        if is_verbose() and result.log_text.strip():
            emit_panel("Packmol log (full)", result.log_text.strip(), border_style="red")
        raise RuntimeError(packmol_failure_message(result))
    return result


def _packmol_log_suggests_cli_rejection(log_text: str, exit_code: int) -> bool:
    log = (log_text or "").lower()
    if exit_code == 174:
        return True
    return (
        "command-line error" in log
        or "unrecognized command-line argument" in log
        or "packmol must be run with" in log
    )


def _run_packmol_subprocess(packmol_bin: str, inp_path: Path) -> subprocess.CompletedProcess[str]:
    """Invoke Packmol; fall back to stdin redirection for pre-CLI binaries."""
    common = {
        "capture_output": True,
        "text": True,
        "check": False,
        "cwd": str(inp_path.parent),
    }
    proc = subprocess.run([packmol_bin, "-i", str(inp_path)], **common)
    log_text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if int(proc.returncode) == 0 or not _packmol_log_suggests_cli_rejection(
        log_text, int(proc.returncode)
    ):
        return proc
    with inp_path.open(encoding="utf-8") as fh:
        return subprocess.run([packmol_bin], stdin=fh, **common)


def resolve_packmol_use(
    *,
    composition: str | None,
    packmol: bool | None = None,
    pyxtal: bool | None = None,
    builder: str | None = None,
) -> bool:
    """Use Packmol for ``--composition`` unless grid builder or ``--no-packmol``."""
    if composition is None:
        return False
    if pyxtal is True:
        return False
    if builder is not None:
        name = str(builder).strip().lower()
        if name in ("crystal", "gas", "liquid"):
            return False
    if packmol is False:
        return False
    if packmol is True:
        return True
    return True


def resolve_packmol_placement_mode(
    *,
    packmol_placement: str | None = None,
    packmol_sphere: bool | None = None,
) -> PackmolPlacement:
    """Return ``cube`` (default) or ``sphere`` (legacy ``--packmol-sphere``)."""
    if packmol_placement is not None:
        mode = str(packmol_placement).strip().lower()
        if mode in ("cube", "sphere"):
            return mode  # type: ignore[return-value]
        raise ValueError(
            f"Invalid packmol placement {packmol_placement!r}; expected 'cube' or 'sphere'."
        )
    if packmol_sphere is True:
        return "sphere"
    return "cube"


def resolve_packmol_cube_side(
    *,
    box_size: float | None = None,
    packmol_box_size: float | None = None,
    packmol_radius: float | None = None,
    flat_bottom_radius: float | None = None,
) -> float:
    """Cube edge length (Å) for ``inside cube``; prefer explicit box / packmol box sizes."""
    if packmol_box_size is not None and float(packmol_box_size) > 0.0:
        return float(packmol_box_size)
    if box_size is not None and float(box_size) > 0.0:
        return float(box_size)
    if packmol_radius is not None and float(packmol_radius) > 0.0:
        return 2.0 * float(packmol_radius)
    if flat_bottom_radius is not None and float(flat_bottom_radius) > 0.0:
        return 2.0 * float(flat_bottom_radius)
    raise ValueError(
        "Packmol cube placement requires --box-size > 0 (or --packmol-box-size, "
        "or legacy --packmol-radius / --flat-bottom-radius for a diameter estimate)."
    )


def packmol_center_for_cold_start(args) -> tuple[float, float, float]:
    """Packmol restraint center: explicit ``--packmol-center`` or sim-cell center."""
    center = getattr(args, "packmol_center", None)
    if center is not None:
        if len(center) != 3:
            raise ValueError("--packmol-center requires three floats: CX CY CZ")
        return (float(center[0]), float(center[1]), float(center[2]))
    sim = getattr(args, "_cold_start_sim_cell_side_A", None)
    if sim is not None and float(sim) > 0.0:
        half = 0.5 * float(sim)
        return (half, half, half)
    return (0.0, 0.0, 0.0)


def resolve_packmol_cube_side_from_args(args) -> float:
    """Cube edge (Å) for Packmol: inner cube sized below the simulation cell."""
    if getattr(args, "packmol_box_size", None) is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
            parse_composition_dict,
            resolve_initial_pbc_box_side,
            resolve_packmol_cube_side_for_sim_cell,
        )

        comp = parse_composition_dict(getattr(args, "composition", None))
        n_mol = int(getattr(args, "n_molecules", 0) or 0) or None
        if comp is not None and n_mol is None:
            n_mol = int(sum(comp.values()))
        sim_side, _source = resolve_initial_pbc_box_side(
            args,
            np.zeros((1, 3), dtype=float),
            composition=comp,
            n_molecules=n_mol,
        )
        setattr(args, "_cold_start_sim_cell_side_A", float(sim_side))
        return resolve_packmol_cube_side_for_sim_cell(args, sim_side)

    # Legacy radius-only estimate (no --box-size / density sizing).
    if getattr(args, "packmol_radius", None) is not None or getattr(
        args, "flat_bottom_radius", None
    ) is not None:
        if getattr(args, "box_size", None) is None:
            try:
                return resolve_packmol_cube_side(
                    packmol_radius=getattr(args, "packmol_radius", None),
                    flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
                )
            except ValueError:
                pass

    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        parse_composition_dict,
        resolve_initial_pbc_box_side,
        resolve_packmol_box_padding_A,
        resolve_packmol_cube_side_for_sim_cell,
    )

    comp = parse_composition_dict(getattr(args, "composition", None))
    n_mol = int(getattr(args, "n_molecules", 0) or 0) or None
    if comp is not None and n_mol is None:
        n_mol = int(sum(comp.values()))
    sim_side, source = resolve_initial_pbc_box_side(
        args,
        np.zeros((1, 3), dtype=float),
        composition=comp,
        n_molecules=n_mol,
    )
    setattr(args, "_cold_start_sim_cell_side_A", float(sim_side))
    packmol_side = resolve_packmol_cube_side_for_sim_cell(args, sim_side)
    padding = resolve_packmol_box_padding_A(args)
    setattr(args, "_cold_start_packmol_padding_A", float(padding))
    setattr(args, "_cold_start_box_sizing_source", str(source))
    return packmol_side


def packmol_cube_origin(
    center: tuple[float, float, float],
    side: float,
) -> tuple[float, float, float]:
    """Minimum corner for a cube centered at ``center`` with edge length ``side``."""
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    half = float(side) / 2.0
    return (cx - half, cy - half, cz - half)


def resolve_packmol_sphere_use(
    *,
    composition: str | None,
    packmol_radius: float | None = None,
    flat_bottom_radius: float | None = None,
    packmol_sphere: bool | None = None,
    packmol: bool | None = None,
) -> bool:
    """True when spherical (not cube) Packmol placement is selected."""
    if not resolve_packmol_use(composition=composition, packmol=packmol):
        return False
    return resolve_packmol_placement_mode(
        packmol_sphere=packmol_sphere,
    ) == "sphere"


def resolve_packmol_sphere_radius(
    packmol_radius: float | None,
    flat_bottom_radius: float | None = None,
) -> float:
    """Return Packmol sphere radius; --packmol-radius overrides --flat-bottom-radius."""
    if packmol_radius is not None and float(packmol_radius) > 0.0:
        return float(packmol_radius)
    if flat_bottom_radius is not None and float(flat_bottom_radius) > 0.0:
        return float(flat_bottom_radius)
    raise ValueError(
        "Spherical Packmol placement requires --packmol-radius > 0 "
        "(or --flat-bottom-radius > 0 for legacy combined mode)."
    )


def require_packmol_sphere_radius(
    flat_bottom_radius: float | None,
    packmol_radius: float | None = None,
) -> float:
    """Backward-compatible alias for resolve_packmol_sphere_radius."""
    return resolve_packmol_sphere_radius(packmol_radius, flat_bottom_radius)


def _element_symbol(atomic_number: int) -> str:
    """Map atomic number to PDB element column (prefer ASE table when installed)."""
    zi = int(atomic_number)
    try:
        from ase.data import chemical_symbols

        if 1 <= zi < len(chemical_symbols) and chemical_symbols[zi]:
            return str(chemical_symbols[zi])
    except ImportError:
        pass
    fallback = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }
    return fallback.get(zi, "X")


def write_monomer_pdb_for_packmol(
    pdb_path: Path,
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    atom_names: list[str] | None = None,
    resname: str = "UNK",
) -> None:
    """Write a centered monomer PDB for Packmol.

    When ``atom_names`` are supplied (CHARMM PSF ``atype`` labels), they are written to
    the PDB name column so Packmol output can be mapped back to PSF order.
    """
    Z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    coords_arr = np.asarray(coords, dtype=float)
    if int(Z.shape[0]) != int(coords_arr.shape[0]):
        raise ValueError(
            f"atomic_numbers length ({Z.shape[0]}) != coords rows ({coords_arr.shape[0]})"
        )
    coords_arr = coords_arr - coords_arr.mean(axis=0)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    resn = str(resname).upper()[:3] or "UNK"

    if atom_names is not None:
        names = [str(n) for n in atom_names]
        if len(names) != int(coords_arr.shape[0]):
            raise ValueError(
                f"atom_names length ({len(names)}) != coords rows ({coords_arr.shape[0]})"
            )
        lines = [
            "REMARK   mmml packmol monomer (CHARMM atom names for PSF reordering)",
            "CRYST1   200.000   200.000   200.000  90.00  90.00  90.00 P 1           1",
        ]
        for i, (name, (x, y, z_coord)) in enumerate(zip(names, coords_arr), start=1):
            elem = _element_symbol(Z[i - 1])
            lines.append(
                f"ATOM  {i:5d} {name[:4]:>4s} {resn:<3s} A   1    "
                f"{float(x):8.3f}{float(y):8.3f}{float(z_coord):8.3f}  1.00  0.00          "
                f"{elem:>2s}"
            )
        lines.append("END")
        pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    from ase import Atoms
    from ase.io import write

    symbols = [_element_symbol(zi) for zi in Z]
    mol = Atoms(symbols=symbols, positions=coords_arr)
    write(pdb_path, mol)


def _parse_pdb_atom_records(
    pdb_path: Path | str,
) -> tuple[list[str], list[int], np.ndarray]:
    """Read ATOM/HETATM records from a PDB file (no MDAnalysis)."""
    names: list[str] = []
    resids: list[int] = []
    positions: list[list[float]] = []

    with open(pdb_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                raise RuntimeError(f"Truncated PDB ATOM record in {pdb_path}")
            name = line[12:16].strip()
            try:
                resid = int(line[22:26].strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid PDB residue number in {pdb_path}: {line[22:26]!r}"
                ) from exc
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid PDB coordinates in {pdb_path}: {line[30:54]!r}"
                ) from exc
            names.append(name)
            resids.append(resid)
            positions.append([x, y, z])

    if not names:
        raise RuntimeError(f"No ATOM/HETATM records found in {pdb_path}")

    return names, resids, np.asarray(positions, dtype=float)


def assign_packmol_pdb_to_psf_order(
    pdb_path: Path | str,
    psf_atom_names: list[str],
    atoms_per_list: list[int],
) -> np.ndarray:
    """Map Packmol-packed PDB coordinates onto CHARMM PSF atom order.

    Packmol output atom order does not generally match PSF ``atype`` order.  Match by
    ``(residue_index, atom_name)`` using the same recipe as ``mmml_ase.load_pdb_data``.
    """
    atypes = [str(x) for x in psf_atom_names]
    n_atoms = len(atypes)
    if n_atoms != int(np.sum(atoms_per_list)):
        raise ValueError(
            f"PSF atom count ({n_atoms}) != sum(atoms_per_list) ({sum(atoms_per_list)})"
        )

    charmm_resids: list[int] = []
    for i, n_per in enumerate(atoms_per_list):
        charmm_resids.extend([int(i)] * int(n_per))

    pdb_names, pdb_resids, pdb_positions = _parse_pdb_atom_records(pdb_path)
    if int(pdb_positions.shape[0]) != n_atoms:
        raise RuntimeError(
            f"Packmol PDB atom count ({pdb_positions.shape[0]}) != PSF ({n_atoms})"
        )

    mda_names = [str(s) for s in pdb_names]
    mda_resids = [int(s) for s in pdb_resids]

    mda_res_at_dict = {
        (int(a) - 1, b): i for i, (a, b) in enumerate(zip(mda_resids, mda_names))
    }
    charmm_res_at_dict = {
        (int(a), b): i for i, (a, b) in enumerate(zip(charmm_resids, atypes))
    }
    an_mda_res_at_dict = {v: k for k, v in mda_res_at_dict.items()}

    out = np.zeros((n_atoms, 3), dtype=float)
    missing: list[tuple[int, tuple[int, str] | None]] = []
    for pdb_i in range(n_atoms):
        key = an_mda_res_at_dict.get(pdb_i)
        if key is None:
            missing.append((pdb_i, None))
            continue
        psf_i = charmm_res_at_dict.get(key)
        if psf_i is None:
            missing.append((pdb_i, key))
            continue
        out[psf_i] = pdb_positions[pdb_i]

    if missing:
        sample = missing[:5]
        raise RuntimeError(
            f"Packmol PDB does not match PSF atom order (first failures: {sample})"
        )

    span = np.ptp(out, axis=0)
    if float(span[1]) < 0.3 or float(span[2]) < 0.3:
        raise RuntimeError(
            f"Packmol cluster not 3D (spans Å x={span[0]:.2f} y={span[1]:.2f} z={span[2]:.2f})"
        )
    return out


def _packmol_inside_restraint_line(
    placement: PackmolPlacement,
    *,
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
) -> str:
    if placement == "cube":
        if cube_side is None or float(cube_side) <= 0.0:
            raise ValueError(f"cube side must be positive, got {cube_side}")
        x0, y0, z0 = packmol_cube_origin(center, float(cube_side))
        return f"  inside cube {x0} {y0} {z0} {float(cube_side)}"
    if placement == "sphere":
        if radius is None or float(radius) <= 0.0:
            raise ValueError(f"sphere radius must be positive, got {radius}")
        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
        return f"  inside sphere {cx} {cy} {cz} {float(radius)}"
    raise ValueError(f"unsupported Packmol placement {placement!r}")


def run_packmol_mixed(
    blocks: list[tuple[Path, int]],
    *,
    placement: PackmolPlacement = "cube",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cube_side: float | None = None,
    radius: float | None = None,
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    input_path: str | Path | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    quiet: bool = False,
    sim_cell_side: float | None = None,
    box_sizing_source: str | None = None,
    packmol_padding_A: float | None = None,
    emit_summary: bool = True,
) -> tuple[str, PackmolRunResult]:
    """Pack multiple structure types inside one cube or sphere (composition order)."""
    if not blocks:
        raise ValueError("run_packmol_mixed: no structure blocks")

    restraint = _packmol_inside_restraint_line(
        placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
    )
    out = Path(output_pdb).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    structure_lines: list[str] = []
    for pdb_path, count in blocks:
        if count <= 0:
            continue
        structure_lines.append(
            f"structure {Path(pdb_path).expanduser().resolve()}\n"
            f"  chain A\n"
            f"  resnumbers 2\n"
            f"  number {int(count)}\n"
            f"{restraint}\n"
            f"end structure"
        )

    if not structure_lines:
        raise ValueError("run_packmol_mixed: all molecule counts are zero")

    randint = int(seed) if seed is not None else int(np.random.randint(1_000_000))
    packmol_input = (
        f"seed {randint}\n"
        f"output {out}\n"
        f"filetype pdb\n"
        f"tolerance {float(tolerance)}\n\n"
        + "\n\n".join(structure_lines)
        + "\n"
    )
    default_inp = (
        "packmol_cube.inp" if placement == "cube" else "packmol_sphere.inp"
    )
    inp_path = (
        Path(input_path).expanduser().resolve()
        if input_path is not None
        else (Path("packmol") / default_inp).resolve()
    )
    result = execute_packmol_script(packmol_input, inp_path)
    if emit_summary:
        emit_packmol_build_summary(
            placement=placement,
            blocks=blocks,
            center=center,
            tolerance=float(tolerance),
            seed=randint,
            output_pdb=out,
            inp_path=inp_path,
            cube_side=cube_side,
            radius=radius,
            sim_cell_side=sim_cell_side,
            box_sizing_source=box_sizing_source,
            packmol_padding_A=packmol_padding_A,
            result=result,
            quiet=quiet,
        )
    return str(out), result


def run_packmol_sphere(
    n_molecules: int,
    center: tuple[float, float, float],
    radius: float,
    *,
    structure_pdb: str | Path = "pdb/initial.pdb",
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    tolerance: float = 2.0,
    seed: int | None = None,
) -> str:
    """Pack *n_molecules* copies of one structure inside a sphere."""
    out, _result = run_packmol_sphere_mixed(
        [(Path(structure_pdb), int(n_molecules))],
        center=center,
        radius=float(radius),
        output_pdb=output_pdb,
        tolerance=tolerance,
        seed=seed,
    )
    return out


def run_packmol_cube_mixed(
    blocks: list[tuple[Path, int]],
    center: tuple[float, float, float],
    cube_side: float,
    *,
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    input_path: str | Path | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    quiet: bool = False,
    sim_cell_side: float | None = None,
    box_sizing_source: str | None = None,
    packmol_padding_A: float | None = None,
    emit_summary: bool = True,
) -> tuple[str, PackmolRunResult]:
    """Pack multiple structure types inside one cube (composition order)."""
    return run_packmol_mixed(
        blocks,
        placement="cube",
        center=center,
        cube_side=float(cube_side),
        output_pdb=output_pdb,
        input_path=input_path,
        tolerance=tolerance,
        seed=seed,
        quiet=quiet,
        sim_cell_side=sim_cell_side,
        box_sizing_source=box_sizing_source,
        packmol_padding_A=packmol_padding_A,
        emit_summary=emit_summary,
    )


def run_packmol_sphere_mixed(
    blocks: list[tuple[Path, int]],
    center: tuple[float, float, float],
    radius: float,
    *,
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    input_path: str | Path | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    quiet: bool = False,
    sim_cell_side: float | None = None,
    box_sizing_source: str | None = None,
    packmol_padding_A: float | None = None,
    emit_summary: bool = True,
) -> tuple[str, PackmolRunResult]:
    """Pack multiple structure types inside one sphere (composition order)."""
    return run_packmol_mixed(
        blocks,
        placement="sphere",
        center=center,
        radius=float(radius),
        output_pdb=output_pdb,
        input_path=input_path,
        tolerance=tolerance,
        seed=seed,
        quiet=quiet,
        sim_cell_side=sim_cell_side,
        box_sizing_source=box_sizing_source,
        packmol_padding_A=packmol_padding_A,
        emit_summary=emit_summary,
    )
