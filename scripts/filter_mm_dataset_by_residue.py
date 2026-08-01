#!/usr/bin/env python3
"""Select the best-sampled part of a CGenFF-enriched dimer NPZ.

A broad set like the DES dimers reaches 90 CGenFF LJ types, but the tail is
thin: 25 of them appear in under 1,000 frames, and one appears in 140. A
trainable per-type sigma/epsilon scale on a type that thin will move under
gradient descent without being constrained by data -- the regime where the
sigma/epsilon degeneracy (docs/hybrid-mm-lj-scales.md) does real damage.

This script keeps only frames whose **both** monomers are in a residue
allowlist, chosen either explicitly (``--residues``) or as the ``--top N``
residues by frame count. It reports what that costs in frames and what it buys
in per-type sampling floor.

Requires ``cgenff_res_name`` -- written by ``mmml prepare-mm-dataset``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mmml.data.cgenff_dataset import load_reference


def residue_counts(res_name: np.ndarray) -> Counter:
    """Frames each residue appears in (a homodimer counts once for its residue)."""
    counts: Counter = Counter()
    for a, b in res_name:
        counts[str(a)] += 1
        if str(b) != str(a):
            counts[str(b)] += 1
    return counts


def type_counts(res_name: np.ndarray, keep: np.ndarray) -> Counter:
    """Frames touching each CGenFF LJ type, over the kept frames."""
    ref = load_reference()
    idx_to_name = {v: k for k, v in ref.nb_map.items()}
    per_resi = {
        name: sorted({idx_to_name[int(i)] for i in tmpl["type_indices"]})
        for name, tmpl in ref.residues.items()
    }
    counts: Counter = Counter()
    for (a, b), k in zip(res_name, keep):
        if not k:
            continue
        for t in set(per_resi[str(a)]) | set(per_resi[str(b)]):
            counts[t] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="CGenFF-enriched NPZ")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="filtered NPZ (omit for a dry run / report only)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--top", type=int, default=None,
                   help="keep the N best-sampled residues")
    g.add_argument("--residues", type=str, default=None,
                   help="explicit comma-separated RESI allowlist")
    ap.add_argument("--min-frames", type=int, default=None,
                    help="alternative cut: keep residues with >= this many frames")
    a = ap.parse_args(argv)

    data = dict(np.load(a.npz, allow_pickle=True))
    if "cgenff_res_name" not in data:
        raise SystemExit(
            f"{a.npz} has no 'cgenff_res_name'. Re-run `mmml prepare-mm-dataset` "
            "-- older enriched NPZs predate that field."
        )
    res_name = np.asarray(data["cgenff_res_name"]).astype(str)
    n = len(res_name)
    counts = residue_counts(res_name)

    if a.residues:
        allow = {r.strip() for r in a.residues.split(",") if r.strip()}
        unknown = allow - set(counts)
        if unknown:
            print(f"warning: not present in this dataset: {sorted(unknown)}",
                  file=sys.stderr)
    elif a.min_frames:
        allow = {r for r, c in counts.items() if c >= a.min_frames}
    elif a.top:
        allow = {r for r, _ in counts.most_common(a.top)}
    else:
        allow = set(counts)

    keep = np.array([str(x) in allow and str(y) in allow for x, y in res_name])
    n_keep = int(keep.sum())

    print(f"input : {n:,} frames, {len(counts)} residues")
    print(f"cut   : {len(allow)} residues -> {n_keep:,} frames "
          f"({100 * n_keep / n:.1f}%)")
    print("\nranked residues (kept marked *):")
    for i, (r, c) in enumerate(counts.most_common(), 1):
        mark = "*" if r in allow else " "
        print(f"  {mark}{i:>3} {r:8s} {c:>7,}")

    tc = type_counts(res_name, keep)
    if tc:
        vals = sorted(tc.values())
        thin = sorted(t for t, v in tc.items() if v < 500)
        print(f"\nLJ types reached: {len(tc)}   min {vals[0]:,}   "
              f"median {vals[len(vals) // 2]:,}   max {vals[-1]:,}")
        if thin:
            print(f"types under 500 frames ({len(thin)}): {', '.join(thin)}")
            print("  -> freeze or exclude these from the trainable scale set")
        else:
            print("no type is under 500 frames -- every trainable scale is constrained")

    if a.output is None:
        print("\n(dry run -- pass -o to write)")
        return 0

    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n else value
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **out)
    print(f"\nwrote {a.output}  ({n_keep:,} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
