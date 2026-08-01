#!/usr/bin/env python3
"""Measure separation coverage in an SO3LR/DES dimer HDF5.

The LJ sigma/epsilon fit is identifiable only if a monomer pair is represented
over a useful range of separations.  This script applies the same covalent-
component finder used by ``prepare-mm-dataset``, then reports mass-weighted COM
separation and the minimum inter-monomer atom distance globally and per pair.

Example (on scicore)::

    uv run python scripts/analyze_des_geometry_coverage.py \
        ~/qcell/qcell_dimers.h5 \
        -o artifacts/des_chemspace/geometry_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.data import atomic_masses

from mmml.data.cgenff_dataset import find_covalent_components, format_composition


def _summary(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    q = np.quantile(a, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "n": int(a.size),
        "min": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "max": float(q[6]),
        "q90_span": float(q[5] - q[1]),
    }


def frame_separations(z: np.ndarray, r: np.ndarray) -> tuple[str, float, float]:
    """Return ``(formula pair, COM distance, closest atom distance)``."""
    components = find_covalent_components(z, r)
    if len(components) != 2:
        raise ValueError(f"expected 2 covalent components, found {len(components)}")
    a, b = (np.asarray(c, dtype=np.int64) for c in components)
    forms = sorted((format_composition(z[a]), format_composition(z[b])))

    def com(idx: np.ndarray) -> np.ndarray:
        masses = atomic_masses[z[idx]]
        return np.average(r[idx], axis=0, weights=masses)

    com_distance = float(np.linalg.norm(com(a) - com(b)))
    delta = r[a, None, :] - r[None, b, :]
    closest_distance = float(np.sqrt(np.min(np.sum(delta * delta, axis=-1))))
    return " + ".join(forms), com_distance, closest_distance


def analyze(
    h5_path: Path, *, stride: int = 1, limit: int | None = None
) -> dict[str, object]:
    import h5py

    by_pair: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"com_distance_A": [], "closest_distance_A": []}
    )
    all_com: list[float] = []
    all_closest: list[float] = []
    skipped_components = 0

    with h5py.File(h5_path, "r") as fh:
        names = sorted(k for k in fh if k != "metadata")[::stride]
        if limit is not None:
            names = names[:limit]
        for i, name in enumerate(names, 1):
            group = fh[name]
            z = np.asarray(group["atomic_numbers"][()], dtype=np.int32).reshape(-1)
            r = np.asarray(group["positions"][()], dtype=np.float64).reshape(-1, 3)
            try:
                pair, com_d, closest_d = frame_separations(z, r)
            except ValueError:
                skipped_components += 1
                continue
            by_pair[pair]["com_distance_A"].append(com_d)
            by_pair[pair]["closest_distance_A"].append(closest_d)
            all_com.append(com_d)
            all_closest.append(closest_d)
            if i % 25000 == 0:
                print(f"  {i:,}/{len(names):,} frames", file=sys.stderr, flush=True)

    if not all_com:
        raise RuntimeError(f"no two-component dimers found in {h5_path}")

    pairs = {
        pair: {
            "com_distance_A": _summary(v["com_distance_A"]),
            "closest_distance_A": _summary(v["closest_distance_A"]),
        }
        for pair, v in sorted(by_pair.items(), key=lambda item: -len(item[1]["com_distance_A"]))
    }
    return {
        "source": str(h5_path),
        "stride": stride,
        "frames_selected": len(names),
        "frames_analyzed": len(all_com),
        "frames_skipped_not_two_components": skipped_components,
        "global": {
            "com_distance_A": _summary(all_com),
            "closest_distance_A": _summary(all_closest),
        },
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    result = analyze(args.h5.expanduser(), stride=args.stride, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    g = result["global"]
    print(f"analyzed {result['frames_analyzed']:,} frames across {len(result['pairs']):,} pairs")
    print(f"COM q05-q95: {g['com_distance_A']['q05']:.3f}-"
          f"{g['com_distance_A']['q95']:.3f} A")
    print(f"closest contact q05-q95: {g['closest_distance_A']['q05']:.3f}-"
          f"{g['closest_distance_A']['q95']:.3f} A")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
