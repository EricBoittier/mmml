#!/usr/bin/env python3
"""``mmml npz2traj`` — convert MMML NPZ datasets to ASE trajectories.

Attaches energy, forces, dipole, charges, and other NPZ fields so they are
visible in ASE / MMML GUIs (``SinglePointCalculator`` + ``atoms.info`` /
``atoms.arrays``).

CLI::

    mmml npz2traj data.npz -o trajectory.traj
    mmml npz2traj data.npz -o subset.traj --max-structures 100 --stride 10
    mmml npz2traj data.npz -o frames.extxyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mmml.data.units import (
    DEBYE_TO_EANGSTROM,
    HARTREE_BOHR_TO_EV_ANGSTROM,
    HARTREE_TO_EV,
)

# Per-frame keys copied into atoms.info (scalars / small arrays).
_INFO_KEYS = (
    "id",
    "com",
    "polar",
    "quadrupole",
    "n_grid",
    "E_eV",
    "E_pred",
    "method",
    "basis_set",
)

# Per-atom keys copied into atoms.arrays (masked like R/Z).
_ARRAY_KEYS = (
    "mono",
    "Q",
    "charges",
    "ml_charges",
)

# NPZ metadata keys stored once on every frame under atoms.info["npz_*"].
_METADATA_KEYS = (
    "generation_date",
    "molpro_version",
    "basis_set",
    "method",
    "units",
    "source_files",
    "conversion_info",
    "_mmml_units",
)

_R_ALIASES = ("R", "coordinates", "positions", "coords")
_Z_ALIASES = ("Z", "atomic_numbers", "numbers")
_E_ALIASES = ("E", "energy", "energies")
_F_ALIASES = ("F", "forces")
_D_ALIASES = ("D", "Dxyz", "dipole", "dipoles", "d")
_N_ALIASES = ("N", "n_atoms", "natoms")
_CELL_ALIASES = ("cell", "cells", "lattice", "lattices", "box", "boxes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml npz2traj",
        description=(
            "Convert MMML NPZ datasets to ASE trajectories with energy, forces, "
            "dipole, charges, and extra fields attached for GUI inspection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mmml npz2traj data.npz -o trajectory.traj\n"
            "  mmml npz2traj data.npz -o subset.traj --max-structures 100 --stride 10\n"
            "  mmml npz2traj data.npz -o frames.extxyz\n"
            "  mmml npz2traj data.npz -o ase.traj --ase-units\n"
            "\n"
            "Schema keys: R/Z required; E, F, D/Dxyz, N, mono, cell optional.\n"
            "Default units follow NPZ schema (E Hartree, F Hartree/Bohr, D Debye).\n"
            "Use --ase-units to convert calculator + info energy/forces/dipole to\n"
            "ASE conventions (eV, eV/Å, e·Å)."
        ),
    )
    parser.add_argument("input", type=Path, help="Input NPZ file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output trajectory (.traj, .extxyz, .xyz, …)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to convert",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth structure (default: 1)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First structure index (default: 0)",
    )
    parser.add_argument(
        "--ase-units",
        action="store_true",
        help=(
            "Convert E/F/D from NPZ schema units to ASE calculator units "
            "(eV, eV/Å, e·Å). Without this flag, values stay in NPZ units "
            "and unit labels are stored in atoms.info."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser


def _first_key(data: Any, aliases: tuple[str, ...]) -> str | None:
    files = set(getattr(data, "files", data.keys()))
    for key in aliases:
        if key in files:
            return key
    return None


def _frame_slice(arr: np.ndarray | None, idx: int, n_structures: int) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.ndim == 0:
        return np.asarray(arr)
    if arr.shape[0] == n_structures:
        return np.asarray(arr[idx])
    # Shared across frames (e.g. single Z or cell).
    return np.asarray(arr)


def _atom_mask(z: np.ndarray, n_atoms: int | None) -> np.ndarray:
    z = np.asarray(z).reshape(-1)
    mask = z > 0
    if n_atoms is not None:
        n = int(n_atoms)
        if 0 < n <= z.size:
            n_mask = np.zeros(z.size, dtype=bool)
            n_mask[:n] = True
            mask = mask & n_mask
    return mask


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays to ASE-info-friendly Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 27:  # e.g. 3x3 tensors
            return value.tolist()
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _attach_calculator(
    atoms: Any,
    *,
    energy: float | None,
    forces: np.ndarray | None,
    dipole: np.ndarray | None,
    charges: np.ndarray | None,
) -> None:
    from ase.calculators.singlepoint import SinglePointCalculator

    kwargs: dict[str, Any] = {}
    if energy is not None:
        kwargs["energy"] = float(energy)
    if forces is not None:
        kwargs["forces"] = np.asarray(forces, dtype=np.float64)
    if dipole is not None:
        kwargs["dipole"] = np.asarray(dipole, dtype=np.float64).reshape(3)
    if charges is not None:
        kwargs["charges"] = np.asarray(charges, dtype=np.float64)
    if kwargs:
        atoms.calc = SinglePointCalculator(atoms, **kwargs)


def npz_to_atoms_list(
    npz_file: Path | str,
    *,
    max_structures: int | None = None,
    stride: int = 1,
    start: int = 0,
    ase_units: bool = False,
    verbose: bool = True,
) -> list[Any]:
    """Load an NPZ and return a list of ASE ``Atoms`` with properties attached."""
    from ase import Atoms

    npz_file = Path(npz_file)
    data = np.load(npz_file, allow_pickle=True)

    r_key = _first_key(data, _R_ALIASES)
    z_key = _first_key(data, _Z_ALIASES)
    if r_key is None or z_key is None:
        raise ValueError(
            "NPZ must contain positions (R/coordinates/positions) and "
            "atomic numbers (Z/atomic_numbers)"
        )

    R = np.asarray(data[r_key], dtype=np.float64)
    Z = np.asarray(data[z_key])
    if R.ndim == 2:
        R = R[None, ...]
    if Z.ndim == 1:
        Z = np.broadcast_to(Z[None, :], (R.shape[0], Z.shape[0])).copy()
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R must have shape (n_structures, n_atoms, 3), got {R.shape}")
    if Z.ndim != 2:
        raise ValueError(f"Z must have shape (n_structures, n_atoms), got {Z.shape}")

    n_structures = int(R.shape[0])

    e_key = _first_key(data, _E_ALIASES)
    f_key = _first_key(data, _F_ALIASES)
    d_key = _first_key(data, _D_ALIASES)
    n_key = _first_key(data, _N_ALIASES)
    cell_key = _first_key(data, _CELL_ALIASES)

    E = np.asarray(data[e_key]).reshape(-1) if e_key else None
    F = np.asarray(data[f_key], dtype=np.float64) if f_key else None
    D = np.asarray(data[d_key], dtype=np.float64) if d_key else None
    N = np.asarray(data[n_key]).reshape(-1) if n_key else None
    cell_arr = np.asarray(data[cell_key], dtype=np.float64) if cell_key else None

    if E is not None and E.shape[0] != n_structures:
        raise ValueError(f"E length {E.shape[0]} != n_structures {n_structures}")
    if F is not None and F.shape != R.shape:
        raise ValueError(f"F shape {F.shape} != R shape {R.shape}")
    if D is not None:
        D = np.asarray(D, dtype=np.float64)
        if D.ndim == 1 and D.size == 3:
            D = np.broadcast_to(D[None, :], (n_structures, 3)).copy()
        if D.shape != (n_structures, 3):
            raise ValueError(f"D must have shape (n_structures, 3), got {D.shape}")

    # Infer units from NPZ metadata when present; else MMML schema defaults.
    units_meta: dict[str, Any] = {}
    files = set(getattr(data, "files", data.keys()))
    for ukey in ("units", "_mmml_units"):
        if ukey not in files:
            continue
        raw = data[ukey]
        try:
            units_meta = dict(raw.item()) if getattr(raw, "ndim", 1) == 0 else dict(raw)
        except Exception:
            units_meta = {}
        break

    def _unit_is_ev(text: Any) -> bool:
        return str(text).strip().lower() in {"ev", "electronvolt"}

    def _unit_is_ev_angstrom(text: Any) -> bool:
        t = str(text).strip().lower().replace("å", "a").replace("angstrom", "a")
        return t in {"ev/a", "ev_a", "ev/ang", "ev_angstrom", "ev/angstrom"}

    if e_key == "E_eV":
        energy_is_ev = True
    elif "E" in units_meta or "energy" in units_meta:
        energy_is_ev = _unit_is_ev(units_meta.get("E", units_meta.get("energy")))
    else:
        energy_is_ev = False

    if "F" in units_meta or "forces" in units_meta:
        force_is_ev_a = _unit_is_ev_angstrom(units_meta.get("F", units_meta.get("forces")))
    else:
        force_is_ev_a = False

    indices = list(range(max(0, start), n_structures, max(1, stride)))
    if max_structures is not None:
        indices = indices[: max(0, int(max_structures))]

    if verbose:
        print(f"Converting NPZ → ASE ({len(indices)} / {n_structures} frames)")
        present = [k for k in (e_key, f_key, d_key, n_key, cell_key) if k]
        if present:
            print(f"  properties: {', '.join(present)}")

    # Shared metadata for every frame.
    shared_meta: dict[str, Any] = {
        "npz_source": str(npz_file.resolve()),
    }
    for key in _METADATA_KEYS:
        if key not in files:
            continue
        try:
            shared_meta[f"npz_{key}"] = _jsonable(data[key])
        except Exception:
            shared_meta[f"npz_{key}"] = str(data[key])

    atoms_list: list[Any] = []
    for out_i, idx in enumerate(indices):
        if verbose and (out_i + 1) % 200 == 0:
            print(f"  progress: {out_i + 1}/{len(indices)}")

        z_frame = _frame_slice(Z, idx, n_structures)
        assert z_frame is not None
        n_atoms = int(N[idx]) if N is not None else None
        mask = _atom_mask(z_frame, n_atoms)

        positions = R[idx][mask]
        numbers = np.asarray(z_frame[mask], dtype=int)

        atoms = Atoms(numbers=numbers, positions=positions)
        atoms.info.update(shared_meta)
        atoms.info["frame_index"] = int(idx)

        cell = _frame_slice(cell_arr, idx, n_structures) if cell_arr is not None else None
        if cell is not None:
            cell = np.asarray(cell, dtype=np.float64)
            if cell.shape == (3,):
                cell = np.diag(cell)
            if cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12:
                atoms.set_cell(cell)
                atoms.pbc = True
                atoms.info["cell"] = cell.tolist()

        energy_native: float | None = None
        forces_native: np.ndarray | None = None
        dipole_native: np.ndarray | None = None

        if E is not None:
            energy_native = float(E[idx])
        if F is not None:
            forces_native = np.asarray(F[idx][mask], dtype=np.float64)
        if D is not None:
            dipole_native = np.asarray(D[idx], dtype=np.float64).reshape(3)

        # Unit conversion for ASE convention when requested.
        energy_out = energy_native
        forces_out = forces_native
        dipole_out = dipole_native
        energy_unit = "eV" if energy_is_ev else "Hartree"
        forces_unit = "eV/Angstrom" if force_is_ev_a else "Hartree/Bohr"
        dipole_unit = "Debye"

        if ase_units:
            if energy_native is not None and not energy_is_ev:
                energy_out = float(energy_native * HARTREE_TO_EV)
                energy_unit = "eV"
            if forces_native is not None and not force_is_ev_a:
                forces_out = forces_native * HARTREE_BOHR_TO_EV_ANGSTROM
                forces_unit = "eV/Angstrom"
            if dipole_native is not None:
                dipole_out = dipole_native * DEBYE_TO_EANGSTROM
                dipole_unit = "e*Angstrom"

        if energy_out is not None:
            atoms.info["energy"] = float(energy_out)
            atoms.info["energy_unit"] = energy_unit
            if energy_native is not None and ase_units and not energy_is_ev:
                atoms.info["energy_hartree"] = float(energy_native)
        if forces_out is not None:
            atoms.arrays["forces"] = np.asarray(forces_out, dtype=np.float64)
            atoms.info["forces_unit"] = forces_unit
        if dipole_out is not None:
            atoms.info["dipole"] = np.asarray(dipole_out, dtype=np.float64).tolist()
            atoms.info["dipole_unit"] = dipole_unit
            if dipole_native is not None and ase_units:
                atoms.info["dipole_debye"] = np.asarray(dipole_native, dtype=np.float64).tolist()

        # Per-atom extras (charges / monopoles).
        charges_out: np.ndarray | None = None
        for akey in _ARRAY_KEYS:
            if akey not in files:
                continue
            arr = np.asarray(data[akey])
            frame = _frame_slice(arr, idx, n_structures)
            if frame is None:
                continue
            frame = np.asarray(frame).reshape(-1)
            if frame.size != z_frame.size:
                continue
            masked = frame[mask]
            # Canonical name for GUI / ASE.
            if akey in ("mono", "Q", "charges", "ml_charges"):
                atoms.arrays["charges"] = np.asarray(masked, dtype=np.float64)
                charges_out = atoms.arrays["charges"]
                if akey != "charges":
                    atoms.arrays[akey] = np.asarray(masked, dtype=np.float64)
            else:
                atoms.arrays[akey] = np.asarray(masked, dtype=np.float64)

        # Per-frame info extras.
        for ikey in _INFO_KEYS:
            if ikey not in files:
                continue
            frame = _frame_slice(np.asarray(data[ikey]), idx, n_structures)
            if frame is None:
                continue
            atoms.info[ikey] = _jsonable(frame)

        # Keep any other small per-frame arrays in info for the data inspector.
        for key in sorted(files):
            if key in {
                r_key,
                z_key,
                e_key,
                f_key,
                d_key,
                n_key,
                cell_key,
                *_ARRAY_KEYS,
                *_INFO_KEYS,
                *_METADATA_KEYS,
                "esp",
                "esp_grid",
                "espMask",
                "vdw_surface",
                "molpro_variables",
            }:
                continue
            try:
                arr = np.asarray(data[key])
            except Exception:
                continue
            if arr.dtype == object:
                continue
            if arr.shape[:1] == (n_structures,) and arr[idx].size <= 27:
                atoms.info[key] = _jsonable(arr[idx])

        _attach_calculator(
            atoms,
            energy=energy_out,
            forces=forces_out,
            dipole=dipole_out,
            charges=charges_out,
        )
        atoms_list.append(atoms)

    data.close()
    return atoms_list


def _write_atoms_list(output_file: Path, atoms_list: list[Any]) -> None:
    """Write frames, avoiding ASE extxyz conflicts between calc results and info.

    Note: ``Atoms.copy()`` in current ASE drops ``SinglePointCalculator``, so we
    mutate temporarily (or rebuild the calculator) rather than relying on copy().
    """
    from ase.io import write
    from ase.io.trajectory import Trajectory

    conflict_keys = ("energy", "forces", "dipole", "free_energy", "magmom", "stress", "charges")
    suffix = output_file.suffix.lower()
    # Only extxyz/xyz forbid overlapping calculator results vs atoms.info.
    strip_conflicts = suffix in {".extxyz", ".xyz"}

    def _prepare(atoms: Any) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
        """Optionally strip conflicting info/arrays; return (atoms, saved_info, saved_arrays)."""
        saved_info: dict[str, Any] = {}
        saved_arrays: dict[str, np.ndarray] = {}
        if strip_conflicts:
            saved_info = {k: atoms.info.pop(k) for k in conflict_keys if k in atoms.info}
            for key in ("forces", "charges"):
                if key in atoms.arrays:
                    saved_arrays[key] = atoms.arrays[key]
                    del atoms.arrays[key]
        return atoms, saved_info, saved_arrays

    def _restore(atoms: Any, saved_info: dict[str, Any], saved_arrays: dict[str, np.ndarray]) -> None:
        atoms.info.update(saved_info)
        for key, arr in saved_arrays.items():
            atoms.arrays[key] = arr

    prepared: list[tuple[Any, dict[str, Any], dict[str, np.ndarray]]] = [
        _prepare(atoms) for atoms in atoms_list
    ]
    try:
        frames = [item[0] for item in prepared]
        if suffix == ".traj":
            with Trajectory(str(output_file), "w") as traj:
                for atoms in frames:
                    traj.write(atoms)
        elif suffix in {".extxyz", ".xyz"}:
            fmt = "extxyz" if suffix == ".extxyz" else "xyz"
            write(str(output_file), frames, format=fmt)
        else:
            write(str(output_file), frames)
    finally:
        for atoms, saved_info, saved_arrays in prepared:
            _restore(atoms, saved_info, saved_arrays)


def npz_to_trajectory(
    npz_file: Path | str,
    output_file: Path | str,
    max_structures: int | None = None,
    stride: int = 1,
    start: int = 0,
    ase_units: bool = False,
    verbose: bool = True,
) -> int:
    """Convert NPZ → ASE trajectory file. Returns number of frames written."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = npz_to_atoms_list(
        npz_file,
        max_structures=max_structures,
        stride=stride,
        start=start,
        ase_units=ase_units,
        verbose=verbose,
    )
    if not atoms_list:
        raise ValueError("No structures selected (check --start / --stride / --max-structures)")

    _write_atoms_list(output_file, atoms_list)

    if verbose:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Wrote {len(atoms_list)} frame(s) → {output_file} ({size_mb:.2f} MB)")
    return len(atoms_list)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.stride < 1:
        print("Error: --stride must be >= 1", file=sys.stderr)
        return 1
    try:
        npz_to_trajectory(
            args.input,
            args.output,
            max_structures=args.max_structures,
            stride=args.stride,
            start=args.start,
            ase_units=args.ase_units,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Error during conversion: {exc}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
