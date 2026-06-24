#!/usr/bin/env python3
"""Reorder dimer structures to match a reference atom order.

This is intended for DFT dimer data whose coordinates are correct but whose atom
rows do not match the CHARMM/MLpot order used by the scan scripts. The matching
is geometry based and constrained by element: atoms are only assigned to
reference atoms with the same atomic number.

Examples
--------
  python scripts/reorder_dimer_atoms_to_reference.py \\
    --input dft_dimers.xyz \\
    --reference scan_geometries.xyz \\
    --output dft_dimers_psf_order.xyz

  python scripts/reorder_dimer_atoms_to_reference.py \\
    --input dft_dimers.npz \\
    --reference psf_order_reference.npz \\
    --output dft_dimers_psf_order.npz \\
    --positions-key R --numbers-key Z

  # DCM common case: input monomer order C,H,H,Cl,Cl -> CHARMM-like C,Cl,Cl,H,H
  python scripts/reorder_dimer_atoms_to_reference.py \\
    --input dft_dimers.npz \\
    --output dft_dimers_psf_order.npz \\
    --monomer-permutation 0,3,4,1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:
        pass

    import itertools

    n_rows, n_cols = cost.shape
    if n_rows != n_cols or n_rows > 9:
        raise RuntimeError(
            "scipy is required for non-square or >9 atom-per-element assignments"
        )
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n_cols)):
        value = float(sum(cost[i, perm[i]] for i in range(n_rows)))
        if value < best_cost:
            best_cost = value
            best_perm = perm
    assert best_perm is not None
    return np.arange(n_rows), np.asarray(best_perm, dtype=int)


def _as_frames(positions: np.ndarray) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[None, :, :]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"positions must have shape (N,3) or (F,N,3), got {arr.shape}")


def _as_numbers(numbers: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(numbers, dtype=np.int32)
    if arr.ndim == 1:
        return np.broadcast_to(arr[None, :], (n_frames, arr.shape[0])).copy()
    if arr.ndim == 2:
        if arr.shape[0] == 1 and n_frames > 1:
            return np.broadcast_to(arr, (n_frames, arr.shape[1])).copy()
        if arr.shape[0] == n_frames:
            return arr
    raise ValueError(f"numbers must have shape (N,) or (F,N), got {arr.shape}")


def _guess_key(data: Any, candidates: tuple[str, ...], label: str) -> str:
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not find {label} key. Tried: {', '.join(candidates)}")


def _read_npz(
    path: Path,
    *,
    positions_key: str | None,
    numbers_key: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str, str]:
    data = np.load(path, allow_pickle=True)
    pos_key = positions_key or _guess_key(
        data,
        ("R", "positions", "coords", "coordinates", "reference_positions_A"),
        "positions",
    )
    num_key = numbers_key or _guess_key(
        data,
        ("Z", "numbers", "atomic_numbers", "z"),
        "atomic numbers",
    )
    positions = _as_frames(data[pos_key])
    numbers = _as_numbers(data[num_key], positions.shape[0])
    extras = {key: data[key] for key in data.files if key not in {pos_key, num_key}}
    return positions, numbers, extras, pos_key, num_key


def _read_atoms_file(path: Path) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    from ase import io as ase_io

    atoms_list = ase_io.read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    positions = np.stack([atoms.get_positions() for atoms in atoms_list], axis=0)
    numbers = np.stack([atoms.get_atomic_numbers() for atoms in atoms_list], axis=0)
    return positions, numbers, atoms_list


def _write_atoms_file(
    path: Path,
    positions: np.ndarray,
    numbers: np.ndarray,
    template_atoms: list[Any] | None,
) -> None:
    from ase import Atoms
    from ase import io as ase_io

    atoms_out = []
    for frame_idx in range(positions.shape[0]):
        if template_atoms is not None and frame_idx < len(template_atoms):
            atoms = template_atoms[frame_idx].copy()
            atoms.set_atomic_numbers(numbers[frame_idx])
            atoms.set_positions(positions[frame_idx])
        else:
            atoms = Atoms(numbers=numbers[frame_idx], positions=positions[frame_idx])
        atoms_out.append(atoms)
    ase_io.write(path, atoms_out)


def _read_structures(
    path: Path,
    *,
    positions_key: str | None = None,
    numbers_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str | None, str | None, list[Any] | None]:
    if path.suffix.lower() == ".npz":
        positions, numbers, extras, pos_key, num_key = _read_npz(
            path, positions_key=positions_key, numbers_key=numbers_key
        )
        return positions, numbers, extras, pos_key, num_key, None
    positions, numbers, atoms = _read_atoms_file(path)
    return positions, numbers, {}, None, None, atoms


def _kabsch_rotation(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cov = moving.T @ reference
    u, _s, vt = np.linalg.svd(cov)
    det = np.linalg.det(u @ vt)
    fix = np.diag([1.0, 1.0, np.sign(det)])
    return u @ fix @ vt


def _assign_by_element(
    source_aligned: np.ndarray,
    source_z: np.ndarray,
    reference: np.ndarray,
    reference_z: np.ndarray,
) -> np.ndarray:
    if sorted(source_z.tolist()) != sorted(reference_z.tolist()):
        raise ValueError("source/reference atomic-number multisets differ")

    permutation = np.empty(reference_z.shape[0], dtype=int)
    for atomic_number in sorted(set(reference_z.tolist())):
        ref_idx = np.where(reference_z == atomic_number)[0]
        src_idx = np.where(source_z == atomic_number)[0]
        diff = reference[ref_idx, None, :] - source_aligned[src_idx][None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        rows, cols = _linear_sum_assignment(cost)
        permutation[ref_idx[rows]] = src_idx[cols]
    return permutation


def match_permutation(
    source_positions: np.ndarray,
    source_numbers: np.ndarray,
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    *,
    max_iter: int = 8,
) -> tuple[np.ndarray, float]:
    """Return ``perm`` such that ``source[perm]`` matches reference order."""
    src = np.asarray(source_positions, dtype=np.float64)
    ref = np.asarray(reference_positions, dtype=np.float64)
    src_z = np.asarray(source_numbers, dtype=np.int32)
    ref_z = np.asarray(reference_numbers, dtype=np.int32)
    if src.shape != ref.shape:
        raise ValueError(f"source/reference shape mismatch: {src.shape} vs {ref.shape}")
    if src_z.shape != ref_z.shape:
        raise ValueError(f"source/reference Z shape mismatch: {src_z.shape} vs {ref_z.shape}")

    src_centered = src - src.mean(axis=0)
    ref_centered = ref - ref.mean(axis=0)
    perm = _assign_by_element(src_centered, src_z, ref_centered, ref_z)

    for _ in range(max_iter):
        moving = src_centered[perm]
        rotation = _kabsch_rotation(moving, ref_centered)
        aligned_all = src_centered @ rotation
        next_perm = _assign_by_element(aligned_all, src_z, ref_centered, ref_z)
        if np.array_equal(next_perm, perm):
            break
        perm = next_perm

    moving = src_centered[perm]
    rotation = _kabsch_rotation(moving, ref_centered)
    aligned = moving @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=1))))
    return perm, rmsd


def _reference_frame(
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_positions.shape[0] == 1:
        return reference_positions[0], reference_numbers[0]
    if frame_idx >= reference_positions.shape[0]:
        raise ValueError(
            f"reference has {reference_positions.shape[0]} frames but input frame {frame_idx} was requested"
        )
    return reference_positions[frame_idx], reference_numbers[frame_idx]


def _write_output(
    path: Path,
    *,
    positions: np.ndarray,
    numbers: np.ndarray,
    extras: dict[str, Any],
    positions_key: str | None,
    numbers_key: str | None,
    template_atoms: list[Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        pos_key = positions_key or "R"
        num_key = numbers_key or "Z"
        np.savez_compressed(path, **extras, **{pos_key: positions, num_key: numbers})
    else:
        _write_atoms_file(path, positions, numbers, template_atoms)


def _active_count(numbers: np.ndarray, n_value: int | None) -> int:
    if n_value is not None:
        return int(n_value)
    nonzero = np.where(np.asarray(numbers) > 0)[0]
    if nonzero.size:
        return int(nonzero[-1] + 1)
    return int(len(numbers))


def _frame_counts(extras: dict[str, Any], key: str | None, n_frames: int) -> np.ndarray | None:
    if key is None or key not in extras:
        return None
    values = np.asarray(extras[key], dtype=int)
    if values.shape == (n_frames,):
        return values
    raise ValueError(f"{key!r} must have shape ({n_frames},), got {values.shape}")


def _reorder_per_atom_extras(
    extras: dict[str, Any],
    *,
    permutations: list[np.ndarray],
    active_counts: np.ndarray,
    max_atoms: int,
) -> dict[str, Any]:
    reordered: dict[str, Any] = {}
    n_frames = len(permutations)
    for key, value in extras.items():
        arr = np.asarray(value)
        if arr.shape[:2] != (n_frames, max_atoms):
            reordered[key] = value
            continue
        out = np.array(arr, copy=True)
        for frame_idx, perm in enumerate(permutations):
            n_active = int(active_counts[frame_idx])
            out[frame_idx, :n_active] = arr[frame_idx, perm]
        reordered[key] = out
    return reordered


def _parse_permutation(text: str) -> np.ndarray:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if sorted(values) != list(range(len(values))):
        raise ValueError(
            "--monomer-permutation must be a zero-based permutation, e.g. 0,3,4,1,2"
        )
    return np.asarray(values, dtype=int)


def _repeat_monomer_permutation(active_n: int, monomer_perm: np.ndarray) -> np.ndarray:
    monomer_size = int(len(monomer_perm))
    if active_n % monomer_size != 0:
        raise ValueError(
            f"active atom count {active_n} is not divisible by monomer permutation length {monomer_size}"
        )
    chunks = []
    for offset in range(0, active_n, monomer_size):
        chunks.append(monomer_perm + offset)
    return np.concatenate(chunks).astype(int)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="DFT dimer file (.npz/.xyz/.extxyz)")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference ordered structure/trajectory for geometry matching",
    )
    parser.add_argument("--output", required=True, type=Path, help="Reordered output path")
    parser.add_argument("--positions-key", default=None, help="NPZ positions key for --input")
    parser.add_argument("--numbers-key", default=None, help="NPZ atomic-number key for --input")
    parser.add_argument("--natoms-key", default="N", help="NPZ active atom-count key for padded arrays")
    parser.add_argument("--reference-positions-key", default=None, help="NPZ positions key for --reference")
    parser.add_argument("--reference-numbers-key", default=None, help="NPZ atomic-number key for --reference")
    parser.add_argument(
        "--reference-natoms-key",
        default="N",
        help="NPZ active atom-count key for padded reference arrays",
    )
    parser.add_argument(
        "--monomer-permutation",
        default=None,
        help=(
            "Apply a fixed zero-based permutation to every monomer instead of geometry matching, "
            "for example 0,3,4,1,2 for DCM C,H,H,Cl,Cl -> C,Cl,Cl,H,H."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Write JSON metadata with permutations/RMSDs. Default: <output>.reorder.json",
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=0.2,
        help="Warn if post-assignment aligned RMSD exceeds this value in Å.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    in_pos, in_z, extras, in_pos_key, in_num_key, in_atoms = _read_structures(
        args.input,
        positions_key=args.positions_key,
        numbers_key=args.numbers_key,
    )
    in_counts = _frame_counts(extras, args.natoms_key, in_pos.shape[0])
    if args.reference is None and args.monomer_permutation is None:
        raise SystemExit("Provide either --reference for geometry matching or --monomer-permutation")
    if args.reference is not None and args.monomer_permutation is not None:
        raise SystemExit("Use either --reference or --monomer-permutation, not both")

    ref_pos = ref_z = ref_counts = None
    if args.reference is not None:
        ref_pos, ref_z, _ref_extras, _ref_pos_key, _ref_num_key, _ref_atoms = _read_structures(
            args.reference,
            positions_key=args.reference_positions_key,
            numbers_key=args.reference_numbers_key,
        )
        ref_counts = _frame_counts(_ref_extras, args.reference_natoms_key, ref_pos.shape[0])

    monomer_perm = (
        _parse_permutation(args.monomer_permutation)
        if args.monomer_permutation is not None
        else None
    )

    reordered_pos = np.empty_like(in_pos)
    reordered_z = np.empty_like(in_z)
    permutations: list[np.ndarray] = []
    metadata_permutations = []
    active_counts = np.empty(in_pos.shape[0], dtype=int)
    rmsds = []
    for frame_idx in range(in_pos.shape[0]):
        in_n = _active_count(
            in_z[frame_idx],
            None if in_counts is None else int(in_counts[frame_idx]),
        )
        if monomer_perm is not None:
            perm = _repeat_monomer_permutation(in_n, monomer_perm)
            rmsd = float("nan")
        else:
            assert ref_pos is not None and ref_z is not None
            ref_frame_pos, ref_frame_z = _reference_frame(ref_pos, ref_z, frame_idx)
            if ref_pos.shape[0] == 1:
                ref_n_value = None if ref_counts is None else int(ref_counts[0])
            else:
                ref_n_value = None if ref_counts is None else int(ref_counts[frame_idx])
            ref_n = _active_count(ref_frame_z, ref_n_value)
            if in_n != ref_n:
                raise ValueError(
                    f"frame {frame_idx}: input active atom count {in_n} differs from reference {ref_n}"
                )
            perm, rmsd = match_permutation(
                in_pos[frame_idx, :in_n],
                in_z[frame_idx, :in_n],
                ref_frame_pos[:ref_n],
                ref_frame_z[:ref_n],
            )
        reordered_pos[frame_idx] = in_pos[frame_idx]
        reordered_z[frame_idx] = in_z[frame_idx]
        reordered_pos[frame_idx, :in_n] = in_pos[frame_idx, perm]
        reordered_z[frame_idx, :in_n] = in_z[frame_idx, perm]
        permutations.append(perm)
        metadata_permutations.append(perm.tolist())
        active_counts[frame_idx] = in_n
        rmsds.append(rmsd)

    extras = _reorder_per_atom_extras(
        extras,
        permutations=permutations,
        active_counts=active_counts,
        max_atoms=in_pos.shape[1],
    )

    _write_output(
        args.output,
        positions=reordered_pos,
        numbers=reordered_z,
        extras=extras,
        positions_key=in_pos_key,
        numbers_key=in_num_key,
        template_atoms=in_atoms,
    )

    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".reorder.json")
    payload = {
        "input": str(args.input),
        "reference": None if args.reference is None else str(args.reference),
        "output": str(args.output),
        "method": "monomer-permutation" if monomer_perm is not None else "geometry-match",
        "monomer_permutation": None if monomer_perm is None else monomer_perm.tolist(),
        "n_frames": int(in_pos.shape[0]),
        "n_atoms": int(in_pos.shape[1]),
        "active_atom_count_histogram": {
            str(int(n)): int(np.sum(active_counts == n)) for n in np.unique(active_counts)
        },
        "active_atom_counts": active_counts.tolist() if in_pos.shape[0] <= 1000 else None,
        "max_rmsd_A": float(np.nanmax(rmsds)) if np.any(np.isfinite(rmsds)) else None,
        "mean_rmsd_A": float(np.nanmean(rmsds)) if np.any(np.isfinite(rmsds)) else None,
        "permutations": metadata_permutations if monomer_perm is None else None,
        "rmsd_A": rmsds if monomer_perm is None else None,
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote reordered structures: {args.output}", flush=True)
    print(f"Wrote reorder metadata:    {metadata_path}", flush=True)
    if payload["max_rmsd_A"] is not None:
        print(
            f"RMSD after element-constrained assignment: max={payload['max_rmsd_A']:.6f} Å, "
            f"mean={payload['mean_rmsd_A']:.6f} Å",
            flush=True,
        )
    else:
        print("Applied fixed monomer permutation; RMSD check is not applicable.", flush=True)
    if payload["max_rmsd_A"] is not None and payload["max_rmsd_A"] > float(args.max_rmsd):
        print(
            f"WARNING: max RMSD exceeds --max-rmsd {float(args.max_rmsd):.3f} Å; "
            "the reference geometry may not correspond frame-by-frame to the DFT data.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
