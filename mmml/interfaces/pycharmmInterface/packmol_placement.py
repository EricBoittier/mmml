"""Packmol input generation and spherical cluster placement (no PyCHARMM import)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

PACKMOL_PATH = Path("~/mmml/mmml/generate/packmol/packmol").expanduser()


def packmol_executable() -> str:
    from mmml.paths import bundled_file

    candidates = [
        bundled_file("generate", "packmol", "packmol"),
        Path(os.path.expanduser(str(PACKMOL_PATH))),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    found = shutil.which("packmol")
    if found:
        return found
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "packmol not found on PATH and no bundled binary "
        f"(tried {tried}). Install packmol or build mmml/generate/packmol."
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


def assign_packmol_pdb_to_psf_order(
    pdb_path: Path | str,
    psf_atom_names: list[str],
    atoms_per_list: list[int],
) -> np.ndarray:
    """Map Packmol-packed PDB coordinates onto CHARMM PSF atom order.

    Packmol output atom order does not generally match PSF ``atype`` order.  Match by
    ``(residue_index, atom_name)`` using the same recipe as ``mmml_ase.load_pdb_data``.
    """
    import MDAnalysis as mda

    atypes = [str(x) for x in psf_atom_names]
    n_atoms = len(atypes)
    if n_atoms != int(np.sum(atoms_per_list)):
        raise ValueError(
            f"PSF atom count ({n_atoms}) != sum(atoms_per_list) ({sum(atoms_per_list)})"
        )

    charmm_resids: list[int] = []
    for i, n_per in enumerate(atoms_per_list):
        charmm_resids.extend([int(i)] * int(n_per))

    u = mda.Universe(str(pdb_path))
    pdb_positions = np.asarray(u.atoms.positions, dtype=float)
    if int(pdb_positions.shape[0]) != n_atoms:
        raise RuntimeError(
            f"Packmol PDB atom count ({pdb_positions.shape[0]}) != PSF ({n_atoms})"
        )

    mda_names = [str(s) for s in u.atoms.names]
    mda_resids = [int(s) for s in u.atoms.resids]

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
    input_path: str | Path | None = None,
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
    inp_path = Path(input_path) if input_path is not None else Path("packmol") / "packmol_sphere.inp"
    execute_packmol_script(packmol_input, inp_path)
    print(f"Generated {out}", flush=True)
    return str(out)
