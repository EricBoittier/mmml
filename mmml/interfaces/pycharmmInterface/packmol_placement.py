"""Packmol input generation and cluster placement (cube or sphere; no PyCHARMM import)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

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


def execute_packmol_script(packmol_input: str, inp_path: Path) -> None:
    os.makedirs(inp_path.parent, exist_ok=True)
    inp_path.write_text(packmol_input)
    packmol_bin = packmol_executable()
    cmd = f"{packmol_bin} < {inp_path}"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"packmol failed with exit code {ret}")


def resolve_packmol_use(
    *,
    composition: str | None,
    packmol: bool | None = None,
    pyxtal: bool | None = None,
    builder: str | None = None,
) -> bool:
    """Use Packmol only when explicitly requested for ``--composition``."""
    if pyxtal is True and composition is not None:
        return False
    if builder is not None and str(builder).strip().lower() == "crystal":
        return False
    if packmol is True:
        return composition is not None
    if packmol is False:
        return False
    return False


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


def resolve_packmol_cube_side_from_args(args) -> float:
    """Cube edge (Å) for Packmol from explicit box flags or ``--box-auto density``."""
    try:
        return resolve_packmol_cube_side(
            box_size=getattr(args, "box_size", None),
            packmol_box_size=getattr(args, "packmol_box_size", None),
            packmol_radius=getattr(args, "packmol_radius", None),
            flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        )
    except ValueError:
        pass
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        resolve_box_auto_mode,
        resolve_density_packmol_cube_side,
    )

    if resolve_box_auto_mode(args) == "density":
        return resolve_density_packmol_cube_side(args)
    raise ValueError(
        "Packmol cube placement requires --box-size > 0 (or --packmol-box-size, "
        "or --box-auto density with --target-density-g-cm3 / --bulk-density-fraction, "
        "or legacy --packmol-radius / --flat-bottom-radius for a diameter estimate)."
    )


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
) -> str:
    """Pack multiple structure types inside one cube or sphere (composition order)."""
    if not blocks:
        raise ValueError("run_packmol_mixed: no structure blocks")

    restraint = _packmol_inside_restraint_line(
        placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
    )
    out = Path(output_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)

    structure_lines: list[str] = []
    for pdb_path, count in blocks:
        if count <= 0:
            continue
        structure_lines.append(
            f"structure {pdb_path}\n"
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
    inp_path = Path(input_path) if input_path is not None else Path("packmol") / default_inp
    execute_packmol_script(packmol_input, inp_path)
    print(f"Generated {out}", flush=True)
    return str(out)


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
    return run_packmol_sphere_mixed(
        [(Path(structure_pdb), int(n_molecules))],
        center=center,
        radius=float(radius),
        output_pdb=output_pdb,
        tolerance=tolerance,
        seed=seed,
    )


def run_packmol_cube_mixed(
    blocks: list[tuple[Path, int]],
    center: tuple[float, float, float],
    cube_side: float,
    *,
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    input_path: str | Path | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
) -> str:
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
) -> str:
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
    )
