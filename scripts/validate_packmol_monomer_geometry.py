#!/usr/bin/env python3
"""Measure how far a real Packmol + CHARMM MM cluster relax moves monomer skeletons.

Builds a Packmol cluster exactly as ``mmml liquid-box`` does (same builder, same
SD/ABNR defaults), then reports the per-monomer change in 1-2/1-3 distances
versus the monomer template Packmol placed. That distribution is what sets
``DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A`` in
``mmml/utils/monomer_internal_geometry.py``: the threshold must sit well above a
genuine relaxation and far below the >1 Å distortions a broken CHARMM build
produces.

Runs CHARMM — use a cluster node, not a laptop. The Packmol cache is bypassed so
every invocation is a real build.

Example
-------
  python scripts/validate_packmol_monomer_geometry.py \\
    --composition MEOH:327 --cube-side 28 --json meoh_327.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_composition(tokens: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^([A-Za-z0-9]+):(\d+)$", part)
            if not m:
                raise ValueError(f"Expected RES:COUNT, got {part!r}")
            out.append((m.group(1).upper(), int(m.group(2))))
    return out


class _GeometryStore:
    """Captures the monomer templates the builder placed."""

    _cluster_residue_geometries: dict[str, np.ndarray]


def _summarize(deviations: np.ndarray) -> dict[str, float]:
    finite = deviations[np.isfinite(deviations)]
    if finite.size == 0:
        return {}
    return {
        "n": int(finite.size),
        "max_A": float(np.max(finite)),
        "p99_A": float(np.percentile(finite, 99)),
        "p50_A": float(np.percentile(finite, 50)),
        "mean_A": float(np.mean(finite)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--composition",
        nargs="+",
        required=True,
        help="RES:COUNT tokens, e.g. MEOH:327 or 'MEOH:100 TIP3:200'",
    )
    parser.add_argument("--cube-side", type=float, default=None, help="Packmol cube side (Å)")
    parser.add_argument("--radius", type=float, default=None, help="Packmol sphere radius (Å)")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sd", type=int, default=50, help="CHARMM SD steps (liquid-box default 50)")
    parser.add_argument(
        "--abnr", type=int, default=100, help="CHARMM ABNR steps (liquid-box default 100)"
    )
    parser.add_argument("--scratch", type=Path, default=Path("packmol_geometry_validation"))
    parser.add_argument("--json", type=Path, default=None, help="Write the summary here")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Leave the production threshold armed (default: measure only)",
    )
    args = parser.parse_args(argv)

    composition = _parse_composition(args.composition)
    if args.cube_side is None and args.radius is None:
        parser.error("pass --cube-side or --radius")
    placement = "sphere" if args.radius is not None else "cube"
    if not args.enforce:
        # Measure the distribution even when it would trip the gate.
        os.environ["MMML_MAX_MONOMER_INTERNAL_DEVIATION_A"] = "0"

    # CHARMM must be live before ``cluster`` is imported: that module binds
    # ``pycharmm`` at import time and captures None while the session is cold.
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        ensure_charmm_session_ready,
    )

    ensure_charmm_session_ready()

    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.utils.monomer_internal_geometry import (
        DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A,
        scan_monomer_internal_geometry,
    )

    center = (
        (0.0, 0.0, 0.0)
        if placement == "sphere"
        else (args.cube_side / 2.0,) * 3
    )
    store = _GeometryStore()
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    z, positions, atoms_per_list, residue_names = build_packmol_composition_cluster(
        composition=composition,
        placement=placement,
        center=center,
        cube_side=args.cube_side,
        radius=args.radius,
        tolerance=float(args.tolerance),
        seed=int(args.seed),
        charmm_sd_steps=int(args.sd),
        charmm_abnr_steps=int(args.abnr),
        scratch_dir=scratch,
        verbose=True,
        reuse_packmol_cache=False,
        geometry_store=store,
    )

    templates_coords = getattr(store, "_cluster_residue_geometries", {})
    if not templates_coords:
        print("No monomer templates captured from the builder", file=sys.stderr)
        return 2

    # Atomic numbers per residue type from the first monomer of that type.
    offsets = np.concatenate([[0], np.cumsum(np.asarray(atoms_per_list, dtype=int))])
    numbers_by_residue: dict[str, np.ndarray] = {}
    for mi, residue in enumerate(residue_names):
        key = str(residue).upper()
        if key not in numbers_by_residue:
            numbers_by_residue[key] = np.asarray(
                z[int(offsets[mi]) : int(offsets[mi + 1])], dtype=int
            )
    templates = {
        key: (np.asarray(coords, dtype=float), numbers_by_residue[key])
        for key, coords in templates_coords.items()
        if key in numbers_by_residue
    }

    deviations, report = scan_monomer_internal_geometry(
        positions,
        atoms_per_list,
        residue_names=residue_names,
        templates=templates,
    )

    per_residue: dict[str, dict[str, float]] = {}
    residues = np.asarray([str(r).upper() for r in residue_names])
    for key in sorted(set(residues.tolist())):
        per_residue[key] = _summarize(deviations[residues == key])

    summary = {
        "composition": [[r, n] for r, n in composition],
        "placement": placement,
        "cube_side_A": args.cube_side,
        "radius_A": args.radius,
        "tolerance_A": float(args.tolerance),
        "seed": int(args.seed),
        "charmm_sd_steps": int(args.sd),
        "charmm_abnr_steps": int(args.abnr),
        "n_atoms": int(len(z)),
        "default_threshold_A": float(DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A),
        "overall": _summarize(deviations),
        "per_residue": per_residue,
        "residue_names": [str(r).upper() for r in residue_names],
        # Per-monomer max 1-2/1-3 deviation, for histograms (NaN -> null).
        "deviations_A": [None if not np.isfinite(d) else float(d) for d in deviations],
        "report": report.to_dict(),
        "headroom_vs_default": (
            float(DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A) / report.max_deviation_A
            if report.max_deviation_A > 0.0
            else None
        ),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json is not None:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
