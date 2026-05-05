#!/usr/bin/env python
"""Unwrap periodic trajectories and write ASE/XYZ outputs.

Examples:
  mmml unwrap-traj in.traj -o unwrapped.traj
  mmml unwrap-traj in.traj -o unwrapped.xyz --format xyz --fast
  mmml unwrap-traj coords.h5 -o unwrapped.extxyz --format extxyz --cell 25,25,25
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


_COORD_KEYS = ("R", "positions", "coordinates", "coords", "xyz")
_NUMBER_KEYS = ("Z", "atomic_numbers", "numbers")
_CELL_KEYS = ("cell", "cells", "lattice", "lattices", "box", "boxes")


def _cell_from_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"cell must have shape (3,) or (3, 3), got {arr.shape}")


def _parse_cell(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
    if values.size == 3:
        return np.diag(values)
    if values.size == 9:
        return values.reshape(3, 3)
    raise ValueError("--cell must contain 3 lengths or 9 cell-matrix values")


def _is_valid_cell(cell: np.ndarray) -> bool:
    try:
        return cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12
    except np.linalg.LinAlgError:
        return False


def _minimum_image_delta(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def unwrap_positions(
    positions: np.ndarray,
    cells: np.ndarray | None = None,
    cell: np.ndarray | None = None,
) -> np.ndarray:
    """Return unwrapped positions for an array shaped (n_frames, n_atoms, 3)."""
    coords = np.asarray(positions, dtype=float)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"positions must have shape (n_frames, n_atoms, 3), got {coords.shape}")
    if coords.shape[0] == 0:
        return coords.copy()

    out = np.empty_like(coords, dtype=float)
    out[0] = coords[0]
    prev_wrapped = coords[0]

    for i in range(1, coords.shape[0]):
        frame_cell = _frame_cell(i, cells, cell)
        if not _is_valid_cell(frame_cell):
            raise ValueError("a non-singular periodic cell is required for unwrapping")
        delta = _minimum_image_delta(coords[i] - prev_wrapped, frame_cell)
        out[i] = out[i - 1] + delta
        prev_wrapped = coords[i]
    return out


def _frame_cell(index: int, cells: np.ndarray | None, fallback: np.ndarray | None) -> np.ndarray | None:
    if cells is not None:
        arr = np.asarray(cells, dtype=float)
        if arr.ndim == 3:
            return _cell_from_array(arr[min(index, arr.shape[0] - 1)])
        if arr.ndim == 2 and arr.shape != (3, 3) and arr.shape[-1] == 3:
            return _cell_from_array(arr[min(index, arr.shape[0] - 1)])
        return _cell_from_array(arr)
    return fallback


def _find_dataset(group: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in group:
            return group[name]
    return None


def _read_h5(path: Path, coord_key: str | None, numbers_key: str | None, cell_key: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import h5py

    with h5py.File(path, "r") as handle:
        coords_ds = handle[coord_key] if coord_key else _find_dataset(handle, _COORD_KEYS)
        if coords_ds is not None and getattr(coords_ds, "shape", None) is not None:
            coords = np.asarray(coords_ds, dtype=float)
            if coords.ndim == 2 and coords.shape[-1] == 3:
                coords = coords[np.newaxis, ...]
            numbers_ds = handle[numbers_key] if numbers_key else _find_dataset(handle, _NUMBER_KEYS)
            numbers = _numbers_from_dataset(numbers_ds, coords.shape[0])
            cells_ds = handle[cell_key] if cell_key else _find_dataset(handle, _CELL_KEYS)
            cells = np.asarray(cells_ds, dtype=float) if cells_ds is not None else None
            return coords, numbers, cells

        frame_names = sorted(name for name, value in handle.items() if hasattr(value, "keys"))
        coords_list: list[np.ndarray] = []
        numbers_list: list[np.ndarray] = []
        cells_list: list[np.ndarray] = []
        for name in frame_names:
            group = handle[name]
            frame_coords_ds = group[coord_key] if coord_key else _find_dataset(group, _COORD_KEYS)
            if frame_coords_ds is None:
                continue
            coords_list.append(np.asarray(frame_coords_ds, dtype=float))
            frame_numbers_ds = group[numbers_key] if numbers_key else _find_dataset(group, _NUMBER_KEYS)
            if frame_numbers_ds is not None:
                numbers_list.append(np.asarray(frame_numbers_ds, dtype=int))
            frame_cells_ds = group[cell_key] if cell_key else _find_dataset(group, _CELL_KEYS)
            if frame_cells_ds is not None:
                cells_list.append(_cell_from_array(np.asarray(frame_cells_ds, dtype=float)))

    if not coords_list:
        raise ValueError(
            "could not find HDF5 coordinates; use --coord-key for a dataset shaped "
            "(n_frames, n_atoms, 3) or per-frame groups with positions"
        )
    coords = np.stack(coords_list, axis=0)
    if numbers_list:
        numbers = np.stack(numbers_list, axis=0)
    else:
        raise ValueError("HDF5 input needs atomic numbers; use --numbers-key if they are not named Z/atomic_numbers")
    cells = np.stack(cells_list, axis=0) if cells_list else None
    return coords, numbers, cells


def _numbers_from_dataset(dataset: Any | None, n_frames: int) -> np.ndarray:
    if dataset is None:
        raise ValueError("HDF5 input needs atomic numbers; use --numbers-key if they are not named Z/atomic_numbers")
    numbers = np.asarray(dataset, dtype=int)
    if numbers.ndim == 1:
        return numbers
    if numbers.ndim == 2:
        return numbers if numbers.shape[0] == n_frames else numbers[0]
    raise ValueError(f"atomic numbers must have shape (n_atoms,) or (n_frames, n_atoms), got {numbers.shape}")


def _iter_ase_frames(path: Path, index: str) -> Iterator[Any]:
    from ase.io import iread

    yield from iread(str(path), index=index)


def _iter_unwrapped_ase_frames(path: Path, index: str, override_cell: np.ndarray | None) -> Iterator[Any]:
    frames = _iter_ase_frames(path, index)
    try:
        first = next(frames)
    except StopIteration:
        return

    prev_wrapped = np.asarray(first.get_positions(), dtype=float)
    first_cell = override_cell if override_cell is not None else np.asarray(first.cell.array, dtype=float)
    if not _is_valid_cell(first_cell):
        raise ValueError("input has no usable cell; pass --cell a,b,c or a 3x3 cell")
    first.set_positions(prev_wrapped)
    if override_cell is not None:
        first.set_cell(override_cell)
        first.pbc = True
    yield first

    prev_unwrapped = prev_wrapped
    for atoms in frames:
        wrapped = np.asarray(atoms.get_positions(), dtype=float)
        frame_cell = override_cell if override_cell is not None else np.asarray(atoms.cell.array, dtype=float)
        if not _is_valid_cell(frame_cell):
            raise ValueError("input has no usable cell; pass --cell a,b,c or a 3x3 cell")
        atoms.set_positions(prev_unwrapped + _minimum_image_delta(wrapped - prev_wrapped, frame_cell))
        if override_cell is not None:
            atoms.set_cell(override_cell)
            atoms.pbc = True
        prev_wrapped = wrapped
        prev_unwrapped = np.asarray(atoms.get_positions(), dtype=float)
        yield atoms


def _symbols(numbers: np.ndarray) -> list[str]:
    from ase.data import chemical_symbols

    return [chemical_symbols[int(z)] for z in numbers]


def _write_fast_xyz(path: Path, frames: Iterator[Any], extended: bool) -> int:
    n_frames = 0
    with path.open("w", encoding="utf-8") as handle:
        for n_frames, atoms in enumerate(frames, start=1):
            numbers = atoms.get_atomic_numbers()
            positions = np.asarray(atoms.get_positions(), dtype=float)
            symbols = _symbols(numbers)
            handle.write(f"{len(symbols)}\n")
            if extended:
                cell = np.asarray(atoms.cell.array, dtype=float).reshape(-1)
                lattice = " ".join(f"{x:.16g}" for x in cell)
                pbc = " ".join("T" if flag else "F" for flag in atoms.pbc)
                handle.write(
                    f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="{pbc}" frame={n_frames - 1}\n'
                )
            else:
                handle.write(f"frame={n_frames - 1}\n")
            for sym, xyz in zip(symbols, positions):
                handle.write(f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}\n")
    return n_frames


def _h5_atoms_iter(path: Path, args: argparse.Namespace, override_cell: np.ndarray | None) -> Iterator[Any]:
    from ase import Atoms

    coords, numbers, cells = _read_h5(path, args.coord_key, args.numbers_key, args.cell_key)
    cell = override_cell
    if cell is None and cells is not None:
        cells_arr = np.asarray(cells, dtype=float)
        if cells_arr.ndim == 1 or cells_arr.shape == (3, 3):
            cell = _cell_from_array(cells_arr)
            cells = None
    unwrapped = unwrap_positions(coords, cells=cells, cell=cell)

    for i, positions in enumerate(unwrapped):
        frame_numbers = numbers[i] if numbers.ndim == 2 else numbers
        mask = frame_numbers > 0
        atoms_cell = _frame_cell(i, cells, cell)
        atoms = Atoms(numbers=frame_numbers[mask], positions=positions[mask])
        if atoms_cell is not None:
            atoms.set_cell(atoms_cell)
            atoms.pbc = True
        yield atoms


def _infer_format(output: Path, fmt: str | None) -> str:
    if fmt and fmt != "auto":
        return fmt
    suffix = output.suffix.lower()
    if suffix == ".xyz":
        return "xyz"
    if suffix == ".extxyz":
        return "extxyz"
    if suffix == ".traj":
        return "traj"
    return suffix.lstrip(".") or "traj"


def _write_frames(output: Path, frames: Iterator[Any], fmt: str, fast: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fast and fmt in {"xyz", "extxyz"}:
        return _write_fast_xyz(output, frames, extended=fmt == "extxyz")

    from ase.io import write
    from ase.io.trajectory import Trajectory

    n_frames = 0
    if fmt == "traj":
        traj = Trajectory(str(output), "w")
        try:
            for atoms in frames:
                traj.write(atoms)
                n_frames += 1
        finally:
            traj.close()
    else:
        for atoms in frames:
            write(str(output), atoms, format=fmt, append=n_frames > 0)
            n_frames += 1
    return n_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unwrap periodic ASE trajectories or HDF5 coordinate files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input trajectory (.traj/.xyz/.extxyz/etc.) or .h5/.hdf5 file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument("--format", choices=("auto", "traj", "xyz", "extxyz"), default="auto", help="Output format (default: infer from suffix)")
    parser.add_argument("--fast", action="store_true", help="Use direct streaming writer for xyz/extxyz outputs")
    parser.add_argument("--index", default=":", help="ASE input frame index/slice for non-HDF5 inputs (default: :)")
    parser.add_argument("--cell", help="Override cell as 'a,b,c' or 9 matrix values")
    parser.add_argument("--coord-key", help="HDF5 coordinate dataset key (default: R/positions/coordinates/coords/xyz)")
    parser.add_argument("--numbers-key", help="HDF5 atomic-number dataset key (default: Z/atomic_numbers/numbers)")
    parser.add_argument("--cell-key", help="HDF5 cell dataset key (default: cell/cells/lattice/lattices/box/boxes)")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output")

    args = parser.parse_args()
    t0 = time.perf_counter()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    try:
        override_cell = _parse_cell(args.cell)
        fmt = _infer_format(args.output, args.format)
        is_h5 = args.input.suffix.lower() in {".h5", ".hdf5"}
        if is_h5:
            frames = _h5_atoms_iter(args.input, args, override_cell)
        else:
            frames = _iter_unwrapped_ase_frames(args.input, args.index, override_cell)
        n_frames = _write_frames(args.output, frames, fmt, fast=args.fast)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        elapsed = time.perf_counter() - t0
        print(f"Wrote {n_frames} unwrapped frame(s) to {args.output}")
        print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
