#!/usr/bin/env python
"""Unwrap periodic trajectories and write ASE/XYZ outputs.

Examples:
  mmml unwrap-traj in.traj -o unwrapped.traj
  mmml unwrap-traj in.traj -o unwrapped.xyz --format xyz --fast
  mmml unwrap-traj coords.h5 -o unwrapped.extxyz --reference wrapped.traj --fast
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
_AtomGroups = list[np.ndarray]


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


def _is_valid_cell(cell: np.ndarray | None) -> bool:
    if cell is None:
        return False
    try:
        return cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12
    except np.linalg.LinAlgError:
        return False


def _minimum_image_delta(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def _contiguous_groups(n_atoms: int, group_size: int | None = None, n_groups: int | None = None) -> _AtomGroups:
    if group_size is not None and n_groups is not None:
        raise ValueError("use only one of --group-size or --n-groups")
    if group_size is None and n_groups is None:
        group_size = 1
    if n_groups is not None:
        if n_groups < 1:
            raise ValueError("--n-groups must be >= 1")
        if n_atoms % n_groups != 0:
            raise ValueError(f"{n_atoms} atoms cannot be split into {n_groups} equal contiguous groups")
        group_size = n_atoms // n_groups
    if group_size is None or group_size < 1:
        raise ValueError("--group-size must be >= 1")
    if n_atoms % group_size != 0:
        raise ValueError(f"{n_atoms} atoms is not divisible by group size {group_size}")
    return [np.arange(start, start + group_size, dtype=int) for start in range(0, n_atoms, group_size)]


def _infer_molecule_groups(numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray | None) -> _AtomGroups:
    from ase import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs

    frame_numbers = np.asarray(numbers, dtype=int)
    if frame_numbers.ndim != 1:
        raise ValueError(f"atomic numbers must have shape (n_atoms,), got {frame_numbers.shape}")

    atoms = Atoms(numbers=frame_numbers, positions=np.asarray(positions, dtype=float))
    if _is_valid_cell(cell):
        atoms.set_cell(cell)
        atoms.pbc = True

    cutoffs = natural_cutoffs(atoms, mult=1.2)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    neighbor_list.update(atoms)

    groups: _AtomGroups = []
    seen = np.zeros(len(atoms), dtype=bool)
    for start in range(len(atoms)):
        if seen[start]:
            continue
        stack = [start]
        component: list[int] = []
        seen[start] = True
        while stack:
            atom_index = stack.pop()
            component.append(atom_index)
            neighbors, _offsets = neighbor_list.get_neighbors(atom_index)
            for neighbor in neighbors:
                neighbor_index = int(neighbor)
                if not seen[neighbor_index]:
                    seen[neighbor_index] = True
                    stack.append(neighbor_index)
        groups.append(np.array(sorted(component), dtype=int))
    return groups


def _make_groups_whole(frame: np.ndarray, cell: np.ndarray, groups: _AtomGroups) -> np.ndarray:
    whole = np.asarray(frame, dtype=float).copy()
    for group in groups:
        anchor = frame[group[0]]
        whole[group] = anchor + _minimum_image_delta(frame[group] - anchor, cell)
    return whole


def _group_centers(frame: np.ndarray, groups: _AtomGroups) -> np.ndarray:
    return np.asarray([frame[group].mean(axis=0) for group in groups], dtype=float)


class _ContiguousGroupUnwrapper:
    def __init__(
        self,
        n_atoms: int,
        group_size: int | None = None,
        n_groups: int | None = None,
        groups: _AtomGroups | None = None,
    ) -> None:
        self.groups = groups if groups is not None else _contiguous_groups(n_atoms, group_size=group_size, n_groups=n_groups)
        if sum(len(group) for group in self.groups) != n_atoms:
            raise ValueError("atom groups must cover every atom exactly once")
        self.prev_wrapped_centers: np.ndarray | None = None
        self.prev_unwrapped_centers: np.ndarray | None = None

    def unwrap(self, wrapped: np.ndarray, cell: np.ndarray) -> np.ndarray:
        if not _is_valid_cell(cell):
            raise ValueError("a non-singular periodic cell is required for unwrapping")

        whole = _make_groups_whole(np.asarray(wrapped, dtype=float), cell, self.groups)
        wrapped_centers = _group_centers(whole, self.groups)
        if self.prev_wrapped_centers is None or self.prev_unwrapped_centers is None:
            self.prev_wrapped_centers = wrapped_centers
            self.prev_unwrapped_centers = wrapped_centers
            return whole

        center_delta = _minimum_image_delta(wrapped_centers - self.prev_wrapped_centers, cell)
        unwrapped_centers = self.prev_unwrapped_centers + center_delta

        out = whole.copy()
        for group, wrapped_center, unwrapped_center in zip(self.groups, wrapped_centers, unwrapped_centers):
            out[group] += unwrapped_center - wrapped_center

        self.prev_wrapped_centers = wrapped_centers
        self.prev_unwrapped_centers = unwrapped_centers
        return out


def unwrap_positions(
    positions: np.ndarray,
    cells: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    group_size: int | None = None,
    n_groups: int | None = None,
    groups: _AtomGroups | None = None,
) -> np.ndarray:
    """Return unwrapped positions for an array shaped (n_frames, n_atoms, 3)."""
    coords = np.asarray(positions, dtype=float)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"positions must have shape (n_frames, n_atoms, 3), got {coords.shape}")
    if coords.shape[0] == 0:
        return coords.copy()

    out = np.empty_like(coords, dtype=float)
    unwrapper = _ContiguousGroupUnwrapper(coords.shape[1], group_size=group_size, n_groups=n_groups, groups=groups)

    for i in range(coords.shape[0]):
        frame_cell = _frame_cell(i, cells, cell)
        out[i] = unwrapper.unwrap(coords[i], frame_cell)
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


def _read_h5(path: Path, coord_key: str | None, numbers_key: str | None, cell_key: str | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
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
        numbers = None
    cells = np.stack(cells_list, axis=0) if cells_list else None
    return coords, numbers, cells


def _numbers_from_dataset(dataset: Any | None, n_frames: int) -> np.ndarray | None:
    if dataset is None:
        return None
    numbers = np.asarray(dataset, dtype=int)
    if numbers.ndim == 1:
        return numbers
    if numbers.ndim == 2:
        return numbers if numbers.shape[0] == n_frames else numbers[0]
    raise ValueError(f"atomic numbers must have shape (n_atoms,) or (n_frames, n_atoms), got {numbers.shape}")


def _iter_ase_frames(path: Path, index: str) -> Iterator[Any]:
    from ase.io import iread

    yield from iread(str(path), index=index)


def _iter_unwrapped_ase_frames(
    path: Path,
    index: str,
    override_cell: np.ndarray | None,
    group_size: int | None,
    n_groups: int | None,
    infer_molecules: bool,
) -> Iterator[Any]:
    frames = _iter_ase_frames(path, index)
    try:
        first = next(frames)
    except StopIteration:
        return

    first_cell = override_cell if override_cell is not None else np.asarray(first.cell.array, dtype=float)
    groups = None
    if infer_molecules and group_size is None and n_groups is None:
        groups = _infer_molecule_groups(first.get_atomic_numbers(), first.get_positions(), first_cell)
    unwrapper = _ContiguousGroupUnwrapper(len(first), group_size=group_size, n_groups=n_groups, groups=groups)
    first.set_positions(unwrapper.unwrap(first.get_positions(), first_cell))
    if override_cell is not None:
        first.set_cell(override_cell)
        first.pbc = True
    yield first

    for atoms in frames:
        frame_cell = override_cell if override_cell is not None else np.asarray(atoms.cell.array, dtype=float)
        atoms.set_positions(unwrapper.unwrap(atoms.get_positions(), frame_cell))
        if override_cell is not None:
            atoms.set_cell(override_cell)
            atoms.pbc = True
        yield atoms


def _symbols(numbers: np.ndarray) -> list[str]:
    from ase.data import chemical_symbols

    return [chemical_symbols[int(z)] for z in numbers]


def _xyz_comment(atoms: Any, frame_index: int, extended: bool) -> str:
    cell = np.asarray(atoms.cell.array, dtype=float)
    if not _is_valid_cell(cell):
        return f"frame={frame_index}"

    lattice = " ".join(f"{x:.16g}" for x in cell.reshape(-1))
    pbc = " ".join("T" if flag else "F" for flag in atoms.pbc)
    fields = [f'Lattice="{lattice}"', f'pbc="{pbc}"', f"frame={frame_index}"]
    if extended:
        fields.insert(1, "Properties=species:S:1:pos:R:3")
    return " ".join(fields)


def _write_fast_xyz(path: Path, frames: Iterator[Any], extended: bool) -> int:
    n_frames = 0
    with path.open("w", encoding="utf-8") as handle:
        for n_frames, atoms in enumerate(frames, start=1):
            numbers = atoms.get_atomic_numbers()
            positions = np.asarray(atoms.get_positions(), dtype=float)
            symbols = _symbols(numbers)
            handle.write(f"{len(symbols)}\n")
            handle.write(f"{_xyz_comment(atoms, n_frames - 1, extended=extended)}\n")
            for sym, xyz in zip(symbols, positions):
                handle.write(f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}\n")
    return n_frames


def _read_psf_atomic_numbers(path: Path) -> np.ndarray:
    """Parse atomic numbers from a CHARMM PSF file based on atom masses."""
    masses = []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        natom = None
        atom_start_idx = None
        for idx, line in enumerate(lines):
            if "!NATOM" in line:
                natom = int(line.split()[0])
                atom_start_idx = idx + 1
                break
        if natom is None or atom_start_idx is None:
            raise ValueError(f"Could not find !NATOM section in PSF file {path}")

        for i in range(natom):
            line = lines[atom_start_idx + i]
            parts = line.split()
            if len(parts) >= 8:
                mass = float(parts[7])
                masses.append(mass)
            else:
                raise ValueError(f"Malformed atom line in PSF file: {line}")

    from ase.data import atomic_masses
    masses_arr = np.array(masses)
    diffs = np.abs(atomic_masses[1:, np.newaxis] - masses_arr)
    numbers = np.argmin(diffs, axis=0) + 1
    return np.asarray(numbers, dtype=int)


def _read_reference(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None:
        return None, None

    if path.suffix.lower() == ".psf":
        try:
            numbers = _read_psf_atomic_numbers(path)
            return numbers, None
        except Exception as e:
            raise ValueError(f"Failed to parse PSF reference file {path}: {e}")

    from ase.io import read

    atoms = read(str(path), index=0)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    cell = np.asarray(atoms.cell.array, dtype=float)
    return numbers, cell if _is_valid_cell(cell) else None


def _h5_atoms_iter(path: Path, args: argparse.Namespace, override_cell: np.ndarray | None) -> Iterator[Any]:
    from ase import Atoms

    coords, numbers, cells = _read_h5(path, args.coord_key, args.numbers_key, args.cell_key)
    ref_numbers, ref_cell = _read_reference(args.reference)
    if numbers is None:
        if ref_numbers is None:
            raise ValueError(
                "HDF5 input needs atomic numbers; use --numbers-key if present, "
                "or --reference with an ASE-readable structure/trajectory"
            )
        numbers = ref_numbers
    if np.asarray(numbers).shape[-1] != coords.shape[1]:
        raise ValueError(
            f"atomic-number count ({np.asarray(numbers).shape[-1]}) does not match HDF5 atom count ({coords.shape[1]})"
        )

    cell = override_cell
    if cell is None and cells is not None:
        cells_arr = np.asarray(cells, dtype=float)
        if cells_arr.ndim == 1 or cells_arr.shape == (3, 3):
            cell = _cell_from_array(cells_arr)
            cells = None
    if cell is None and cells is None:
        cell = ref_cell
    groups = None
    if not args.no_molecules and args.group_size is None and args.n_groups is None:
        first_numbers = numbers[0] if numbers.ndim == 2 else numbers
        first_cell = _frame_cell(0, cells, cell)
        groups = _infer_molecule_groups(first_numbers, coords[0], first_cell)
    unwrapped = unwrap_positions(
        coords,
        cells=cells,
        cell=cell,
        group_size=args.group_size,
        n_groups=args.n_groups,
        groups=groups,
    )

    for i, positions in enumerate(unwrapped):
        frame_numbers = numbers[i] if numbers.ndim == 2 else numbers
        mask = frame_numbers > 0
        atoms_cell = _frame_cell(i, cells, cell)
        atoms = Atoms(numbers=frame_numbers[mask], positions=positions[mask])
        if atoms_cell is not None:
            atoms.set_cell(atoms_cell)
            atoms.pbc = True
        yield atoms


def _dcd_atoms_iter(path: Path, args: argparse.Namespace, override_cell: np.ndarray | None) -> Iterator[Any]:
    from ase import Atoms
    from mmml.utils.dcd_reader import read_dcd_trajectory

    if args.reference is None:
        raise ValueError("Reading a .dcd file requires a topology reference file via --reference")

    ref_numbers, ref_cell = _read_reference(args.reference)
    if ref_numbers is None:
        raise ValueError(
            "Could not read atomic numbers from reference; supply an ASE-readable PDB/structure file via --reference"
        )

    coords, hdr = read_dcd_trajectory(path)
    if ref_numbers.shape[-1] != coords.shape[1]:
        raise ValueError(
            f"atomic-number count ({ref_numbers.shape[-1]}) does not match DCD atom count ({coords.shape[1]})"
        )

    cell = override_cell if override_cell is not None else ref_cell
    groups = None
    if not args.no_molecules and args.group_size is None and args.n_groups is None:
        first_numbers = ref_numbers[0] if ref_numbers.ndim == 2 else ref_numbers
        groups = _infer_molecule_groups(first_numbers, coords[0], cell)

    unwrapped = unwrap_positions(
        coords,
        cell=cell,
        group_size=args.group_size,
        n_groups=args.n_groups,
        groups=groups,
    )

    for i, positions in enumerate(unwrapped):
        frame_numbers = ref_numbers[i] if ref_numbers.ndim == 2 else ref_numbers
        mask = frame_numbers > 0
        atoms = Atoms(numbers=frame_numbers[mask], positions=positions[mask])
        if cell is not None:
            atoms.set_cell(cell)
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
    if suffix == ".dcd":
        return "dcd"
    return suffix.lstrip(".") or "traj"


def _write_frames(output: Path, frames: Iterator[Any], fmt: str, fast: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "xyz" or (fast and fmt == "extxyz"):
        return _write_fast_xyz(output, frames, extended=fmt == "extxyz")

    if fmt == "dcd":
        from mmml.utils.dcd_writer import save_trajectory_dcd
        atoms_list = list(frames)
        if not atoms_list:
            return 0
        first_atoms = atoms_list[0]
        positions = np.stack([atoms.get_positions() for atoms in atoms_list])
        boxes = []
        for atoms in atoms_list:
            if atoms.pbc.any():
                boxes.append(atoms.cell.array)
            else:
                boxes.append(None)
        has_boxes = any(b is not None for b in boxes)
        save_trajectory_dcd(
            path=output,
            positions=positions,
            atoms=first_atoms,
            boxes=boxes if has_boxes else None,
        )
        return len(atoms_list)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unwrap periodic ASE trajectories or HDF5 coordinate files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input trajectory (.traj/.xyz/.extxyz/etc.) or .h5/.hdf5 file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument("--format", choices=("auto", "traj", "xyz", "extxyz", "dcd"), default="auto", help="Output format (default: infer from suffix)")
    parser.add_argument("--fast", action="store_true", help="Use direct streaming writer for xyz/extxyz outputs")
    parser.add_argument("--index", default=":", help="ASE input frame index/slice for non-HDF5 inputs (default: :)")
    parser.add_argument("--cell", help="Override cell as 'a,b,c' or 9 matrix values")
    parser.add_argument("--group-size", type=int, help="Atoms per contiguous molecule/group for molecule-wise unwrapping")
    parser.add_argument("--n-groups", type=int, help="Number of equal contiguous molecule/groups for molecule-wise unwrapping")
    parser.add_argument("--no-molecules", action="store_true", help="Disable automatic bonded-fragment grouping; unwrap atoms independently")
    parser.add_argument("--reference", type=Path, help="ASE-readable file supplying atomic numbers and fallback cell for HDF5 inputs")
    parser.add_argument("--coord-key", help="HDF5 coordinate dataset key (default: R/positions/coordinates/coords/xyz)")
    parser.add_argument("--numbers-key", help="HDF5 atomic-number dataset key (default: Z/atomic_numbers/numbers)")
    parser.add_argument("--cell-key", help="HDF5 cell dataset key (default: cell/cells/lattice/lattices/box/boxes)")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1
    if args.reference is not None and not args.reference.exists():
        print(f"Error: reference not found: {args.reference}", file=sys.stderr)
        return 1

    try:
        override_cell = _parse_cell(args.cell)
        fmt = _infer_format(args.output, args.format)
        is_h5 = args.input.suffix.lower() in {".h5", ".hdf5"}
        is_dcd = args.input.suffix.lower() == ".dcd"
        if is_h5:
            frames = _h5_atoms_iter(args.input, args, override_cell)
        elif is_dcd:
            frames = _dcd_atoms_iter(args.input, args, override_cell)
        else:
            frames = _iter_unwrapped_ase_frames(
                args.input,
                args.index,
                override_cell,
                group_size=args.group_size,
                n_groups=args.n_groups,
                infer_molecules=not args.no_molecules,
            )
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
