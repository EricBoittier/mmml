#!/usr/bin/env python3
"""Lightweight scanner to extract composition intel from an extxyz dataset.

Reports total structures, unique chemical formulas, atom count distributions,
info keys, arrays keys, and identifies dimer monomer pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase.io import iread


def inspect_dataset(extxyz_path: str | Path, max_structures: int | None = None, output_json: str | Path | None = None):
    extxyz_path = Path(extxyz_path).expanduser()
    if not extxyz_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {extxyz_path}")

    print(f"==================================================================")
    print(f" Dataset Intel Inspector: {extxyz_path}")
    print(f"==================================================================")

    formula_counts = Counter()
    atom_count_counts = Counter()
    info_keys = Counter()
    arrays_keys = Counter()
    total_structures = 0

    for atoms in iread(str(extxyz_path), index=":", format="extxyz"):
        total_structures += 1
        formula_counts[atoms.get_chemical_formula()] += 1
        atom_count_counts[len(atoms)] += 1
        
        for k in atoms.info:
            info_keys[k] += 1
        for k in atoms.arrays:
            arrays_keys[k] += 1

        if max_structures and total_structures >= max_structures:
            break

    print(f"\n[+] Total Structures Scanned: {total_structures}")
    print(f"\n[+] Atom Count Distribution:")
    for n_atoms, count in sorted(atom_count_counts.items()):
        print(f"    - {n_atoms} atoms: {count} structures")

    print(f"\n[+] Unique Chemical Formulas (Top 25):")
    for formula, count in formula_counts.most_common(25):
        print(f"    - {formula:<20}: {count} frames")

    print(f"\n[+] Info Keys Found:")
    for k, count in info_keys.most_common():
        print(f"    - {k:<25}: in {count} frames")

    print(f"\n[+] Arrays Keys Found:")
    for k, count in arrays_keys.most_common():
        print(f"    - {k:<25}: in {count} frames")

    intel = {
        "dataset_path": str(extxyz_path),
        "total_structures": total_structures,
        "atom_counts": dict(atom_count_counts),
        "top_formulas": dict(formula_counts.most_common(50)),
        "info_keys": dict(info_keys),
        "arrays_keys": dict(arrays_keys),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(intel, indent=2), encoding="utf-8")
        print(f"\n[+] Intel summary exported to: {out_path}")

    print(f"==================================================================")
    return intel


def main():
    parser = argparse.ArgumentParser(description="Extract composition intel from extxyz dataset")
    parser.add_argument("--extxyz", required=True, help="Path to extxyz file")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit for fast scan")
    parser.add_argument("--output-json", default="dataset_intel.json", help="Path to save output JSON intel")
    args = parser.parse_args()

    inspect_dataset(args.extxyz, max_structures=args.max_structures, output_json=args.output_json)


if __name__ == "__main__":
    main()
