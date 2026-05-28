#!/usr/bin/env python3
"""
Extract HR46 XYZ data into EF-training NPZ files.

Expected input layout (example):
  HR46/
    WFT/<basis>/<molecule>/<method>/*.xyz
    DFT/<basis>/<molecule>/<method>/*.xyz

This script selects:
  - WFT: /WFT/aug-cc-pvqz/*/CCSDT/
  - DFT: methods PBE0 and wB97M-V from larger basis sets

Per output NPZ, keys are:
  electric_field, Z, R, E, F
where F is filled with zeros.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "I": 53,
}

ENERGY_RE = re.compile(r"E\s*=\s*([-+0-9.eEdD]+)")


@dataclass
class Record:
    field: float
    energy: float
    z: np.ndarray
    r: np.ndarray


def parse_xyz_file(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Malformed XYZ (too short): {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Malformed XYZ atom-count line: {path}") from exc

    if len(lines) < natoms + 2:
        raise ValueError(f"Malformed XYZ (missing atom lines): {path}")

    m = ENERGY_RE.search(lines[1])
    if not m:
        raise ValueError(f"Could not parse energy from comment line: {path}")
    energy = float(m.group(1).replace("D", "E").replace("d", "e"))

    z = np.zeros((natoms,), dtype=np.int32)
    r = np.zeros((natoms, 3), dtype=np.float64)
    for i in range(natoms):
        parts = lines[2 + i].split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {i + 1} in {path}")
        symbol = parts[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported element '{symbol}' in {path}")
        z[i] = ATOMIC_NUMBERS[symbol]
        r[i, 0] = float(parts[1])
        r[i, 1] = float(parts[2])
        r[i, 2] = float(parts[3])
    return z, r, energy


def parse_field_from_filename(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"Cannot parse electric field from filename: {path}") from exc


def collect_records(method_dir: Path) -> List[Record]:
    records: List[Record] = []
    for xyz in sorted(method_dir.glob("*.xyz"), key=lambda p: float(p.stem)):
        field = parse_field_from_filename(xyz)
        z, r, e = parse_xyz_file(xyz)
        records.append(Record(field=field, energy=e, z=z, r=r))
    if not records:
        raise ValueError(f"No XYZ files found in {method_dir}")
    return records


def stack_records(records: List[Record]) -> Dict[str, np.ndarray]:
    natoms = records[0].z.shape[0]
    z0 = records[0].z
    for rec in records[1:]:
        if rec.z.shape[0] != natoms:
            raise ValueError("Mixed atom counts in one method directory.")
        if not np.array_equal(rec.z, z0):
            raise ValueError("Atomic numbers differ across field points.")

    n = len(records)
    electric_field = np.array([r.field for r in records], dtype=np.float64).reshape(n, 1)
    E = np.array([r.energy for r in records], dtype=np.float64).reshape(n, 1)
    Z = np.repeat(z0.reshape(1, natoms), n, axis=0)
    R = np.stack([r.r for r in records], axis=0)
    F = np.zeros_like(R, dtype=np.float64)

    return {
        "electric_field": electric_field,
        "Z": Z,
        "R": R,
        "E": E,
        "F": F,
    }


def concat_payloads(payloads: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("No payloads to concatenate.")
    max_atoms = max(p["R"].shape[1] for p in payloads)

    ef_all: List[np.ndarray] = []
    e_all: List[np.ndarray] = []
    z_all: List[np.ndarray] = []
    r_all: List[np.ndarray] = []
    f_all: List[np.ndarray] = []

    for p in payloads:
        n, natoms, _ = p["R"].shape
        ef_all.append(p["electric_field"])
        e_all.append(p["E"])
        if natoms == max_atoms:
            z_all.append(p["Z"])
            r_all.append(p["R"])
            f_all.append(p["F"])
            continue

        z_pad = np.zeros((n, max_atoms), dtype=p["Z"].dtype)
        r_pad = np.zeros((n, max_atoms, 3), dtype=p["R"].dtype)
        f_pad = np.zeros((n, max_atoms, 3), dtype=p["F"].dtype)
        z_pad[:, :natoms] = p["Z"]
        r_pad[:, :natoms, :] = p["R"]
        f_pad[:, :natoms, :] = p["F"]
        z_all.append(z_pad)
        r_all.append(r_pad)
        f_all.append(f_pad)

    return {
        "electric_field": np.concatenate(ef_all, axis=0),
        "Z": np.concatenate(z_all, axis=0),
        "R": np.concatenate(r_all, axis=0),
        "E": np.concatenate(e_all, axis=0),
        "F": np.concatenate(f_all, axis=0),
    }


def should_include(
    level: str,
    basis: str,
    method: str,
    dft_bases: set[str],
) -> bool:
    if level == "WFT":
        return basis == "aug-cc-pvqz" and method == "CCSDT"
    if level == "DFT":
        return basis in dft_bases and method in {"PBE0", "wB97M-V"}
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HR46 XYZ -> EF NPZ.")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to HR46 root (contains WFT/ and DFT/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where NPZ files are written.",
    )
    parser.add_argument(
        "--dft-bases",
        type=str,
        nargs="+",
        default=["aug-cc-pvtz", "aug-cc-pvqz"],
        help="DFT basis sets treated as larger basis sets.",
    )
    parser.add_argument(
        "--combine-by-theory",
        action="store_true",
        help=(
            "Combine all molecules into one NPZ per (level, basis, method). "
            "Example output: WFT_aug-cc-pvqz_CCSDT.npz"
        ),
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dft_bases = set(args.dft_bases)

    pattern = "*/*/*/*"
    method_dirs = [p for p in input_root.glob(pattern) if p.is_dir()]
    if not method_dirs:
        raise FileNotFoundError(
            f"No method directories found under {input_root} with pattern {pattern}"
        )

    written = 0
    grouped_payloads: Dict[Tuple[str, str, str], List[Dict[str, np.ndarray]]] = {}
    for method_dir in sorted(method_dirs):
        # Expected: <input_root>/<level>/<basis>/<molecule>/<method>
        rel = method_dir.relative_to(input_root)
        parts = rel.parts
        if len(parts) != 4:
            continue
        level, basis, molecule, method = parts
        if not should_include(level, basis, method, dft_bases):
            continue

        records = collect_records(method_dir)
        payload = stack_records(records)
        if args.combine_by_theory:
            grouped_payloads.setdefault((level, basis, method), []).append(payload)
        else:
            out_name = f"{level}_{basis}_{molecule}_{method}.npz".replace("/", "_")
            out_path = output_dir / out_name
            np.savez(out_path, **payload)
            written += 1
            print(f"Wrote {out_path} (n={payload['R'].shape[0]})")

    if args.combine_by_theory:
        for (level, basis, method), payloads in sorted(grouped_payloads.items()):
            combined = concat_payloads(payloads)
            out_name = f"{level}_{basis}_{method}.npz".replace("/", "_")
            out_path = output_dir / out_name
            np.savez(out_path, **combined)
            written += 1
            print(
                f"Wrote {out_path} (n={combined['R'].shape[0]}, natoms_max={combined['R'].shape[1]})"
            )

    if written == 0:
        raise RuntimeError(
            "No datasets matched selection. Check input path and basis/method filters."
        )
    print(f"Done. Wrote {written} NPZ files to {output_dir}")


if __name__ == "__main__":
    main()
