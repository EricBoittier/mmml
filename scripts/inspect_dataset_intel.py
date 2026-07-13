#!/usr/bin/env python3
"""Lightweight scanner to extract composition intel from an Orbax data cache or extxyz dataset.

Fast Orbax cache reader for instant dataset inspection across hundreds of thousands of structures.
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

import numpy as np
import orbax.checkpoint as ocp
from ase.data import chemical_symbols


def formula_from_atomic_numbers(z_arr: np.ndarray) -> str:
    """Fast chemical formula string from atomic numbers array."""
    counts = Counter(z_arr)
    parts = []
    # Standard chemical ordering: C, H, then alphabetical
    order = []
    if 6 in counts:
        order.append(6)
    if 1 in counts:
        order.append(1)
    for z in sorted(counts.keys()):
        if z not in (1, 6):
            order.append(z)
    for z in order:
        sym = chemical_symbols[z]
        c = counts[z]
        parts.append(f"{sym}{c if c > 1 else ''}")
    return "".join(parts)


def inspect_orbax_cache(cache_path: str | Path, max_structures: int | None = None, output_json: str | Path | None = None):
    cache_path = Path(cache_path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache path not found: {cache_path}")

    print(f"==================================================================")
    print(f" Fast Orbax Cache Intel Inspector: {cache_path.name}")
    print(f" Path: {cache_path}")
    print(f"==================================================================")

    data = ocp.PyTreeCheckpointer().restore(cache_path)
    
    # Extract flat arrays
    Z = np.asarray(data["Z"]).reshape(-1)
    N = np.asarray(data["N"]).reshape(-1)
    offsets = np.asarray(data["mol_offsets"]).reshape(-1)
    E = np.asarray(data["E"]).reshape(-1)
    Q = np.asarray(data["Q"]).reshape(-1)

    n_total = len(N)
    if max_structures:
        n_total = min(n_total, max_structures)

    print(f"\n[+] Total Structures: {n_total:,}")
    print(f"[+] Total Atoms     : {offsets[n_total]:,}")

    atom_count_counts = Counter(N[:n_total].tolist())
    print(f"\n[+] Atom Count Distribution:")
    for n_atoms, count in sorted(atom_count_counts.items()):
        print(f"    - {n_atoms} atoms: {count:,} structures ({count / n_total * 100:.1f}%)")

    # Fast formula extraction across structures
    print(f"\n[+] Sampling Chemical Formulas across structures...")
    formula_counts = Counter()
    for i in range(n_total):
        start = offsets[i]
        end = offsets[i + 1]
        z_struct = Z[start:end]
        formula_counts[formula_from_atomic_numbers(z_struct)] += 1

    print(f"\n[+] Unique Chemical Formulas (Top 25):")
    for formula, count in formula_counts.most_common(25):
        print(f"    - {formula:<20}: {count:,} frames ({count / n_total * 100:.1f}%)")

    print(f"\n[+] Dataset Targets Summary:")
    print(f"    - Energy range : [{E[:n_total].min():.3f}, {E[:n_total].max():.3f}] (mean={E[:n_total].mean():.3f})")
    print(f"    - Charge range : [{Q[:n_total].min():.3f}, {Q[:n_total].max():.3f}] (mean={Q[:n_total].mean():.3f})")

    intel = {
        "cache_path": str(cache_path),
        "total_structures": int(n_total),
        "total_atoms": int(offsets[n_total]),
        "atom_counts": {int(k): int(v) for k, v in atom_count_counts.items()},
        "top_formulas": dict(formula_counts.most_common(50)),
        "energy_min": float(E[:n_total].min()),
        "energy_max": float(E[:n_total].max()),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(intel, indent=2), encoding="utf-8")
        print(f"\n[+] Intel summary exported to: {out_path}")

    print(f"==================================================================")
    return intel


def main():
    parser = argparse.ArgumentParser(description="Extract composition intel fast from Orbax cache or extxyz dataset")
    parser.add_argument("--cache-dir", required=True, help="Path to Orbax data cache directory")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--output-json", default="dataset_intel.json", help="Path to save output JSON intel")
    args = parser.parse_args()

    inspect_orbax_cache(args.cache_dir, max_structures=args.max_structures, output_json=args.output_json)


if __name__ == "__main__":
    main()
