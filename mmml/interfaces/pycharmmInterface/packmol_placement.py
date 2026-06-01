"""Packmol input generation and spherical cluster placement (no PyCHARMM import)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

PACKMOL_PATH = Path("~/mmml/mmml/generate/packmol/packmol").expanduser()


def packmol_executable() -> str:
    path = Path(os.path.expanduser(str(PACKMOL_PATH)))
    if path.is_file():
        return str(path)
    found = shutil.which("packmol")
    if found:
        return found
    return str(path)


def execute_packmol_script(packmol_input: str, inp_path: Path) -> None:
    os.makedirs(inp_path.parent, exist_ok=True)
    inp_path.write_text(packmol_input)
    packmol_bin = packmol_executable()
    cmd = f"{packmol_bin} < {inp_path}"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"packmol failed with exit code {ret}")


def resolve_packmol_sphere_use(
    *,
    composition: str | None,
    packmol_radius: float | None = None,
    flat_bottom_radius: float | None = None,
    packmol_sphere: bool | None = None,
) -> bool:
    """Use spherical Packmol placement when explicitly requested or composition + a radius."""
    if packmol_sphere is True:
        return True
    if packmol_sphere is False:
        return False
    if composition is None:
        return False
    if packmol_radius is not None and float(packmol_radius) > 0.0:
        return True
    # Legacy: auto-enable when only --flat-bottom-radius was set (uses it as packmol R too).
    if flat_bottom_radius is not None and float(flat_bottom_radius) > 0.0:
        return True
    return False


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


def write_monomer_pdb_for_packmol(
    pdb_path: Path,
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
) -> None:
    """Write a centered monomer PDB for Packmol using true atomic numbers (e.g. Cl=17)."""
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import write

    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    coords_arr = np.asarray(coords, dtype=float)
    if int(z.shape[0]) != int(coords_arr.shape[0]):
        raise ValueError(
            f"atomic_numbers length ({z.shape[0]}) != coords rows ({coords_arr.shape[0]})"
        )
    symbols = [chemical_symbols[int(zi)] for zi in z]
    mol = Atoms(symbols=symbols, positions=coords_arr)
    mol.positions -= mol.get_center_of_mass()
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    write(pdb_path, mol)


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


def run_packmol_sphere_mixed(
    blocks: list[tuple[Path, int]],
    center: tuple[float, float, float],
    radius: float,
    *,
    output_pdb: str | Path = "pdb/init-packmol-sphere.pdb",
    tolerance: float = 2.0,
    seed: int | None = None,
) -> str:
    """Pack multiple structure types inside one sphere (composition order)."""
    if not blocks:
        raise ValueError("run_packmol_sphere_mixed: no structure blocks")
    if float(radius) <= 0.0:
        raise ValueError(f"sphere radius must be positive, got {radius}")

    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
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
            f"  inside sphere {cx} {cy} {cz} {float(radius)}\n"
            f"end structure"
        )

    if not structure_lines:
        raise ValueError("run_packmol_sphere_mixed: all molecule counts are zero")

    randint = int(seed) if seed is not None else int(np.random.randint(1_000_000))
    packmol_input = (
        f"seed {randint}\n"
        f"output {out}\n"
        f"filetype pdb\n"
        f"tolerance {float(tolerance)}\n\n"
        + "\n\n".join(structure_lines)
        + "\n"
    )
    inp_path = Path("packmol") / "packmol_sphere.inp"
    execute_packmol_script(packmol_input, inp_path)
    print(f"Generated {out}")
    return str(out)
