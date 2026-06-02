#!/usr/bin/env python3
"""Validate sparse ML dimer cap vs near-neighbor count for a cluster geometry.

Counts dimers with COM distance < mm_switch_on (same rule as ``mmml_calculator``)
and compares to ``resolve_max_active_dimers``.

Examples
--------
From minimized MLpot CRD (DCM:90, 32 Å box):

  python scripts/validate_mlpot_sparse_dimers.py \\
    --crd artifacts/pycharmm_mlpot/dcm20_pbc/mini_full_mlpot_DCM90.crd \\
    --n-monomers 90 --atoms-per-monomer 10 --box-size 32

From numpy positions (shape N×3):

  python scripts/validate_mlpot_sparse_dimers.py \\
    --positions-npy coords.npy --n-monomers 90 --atoms-per-monomer 10 --box-size 32

Exit code 0 if cap is sufficient, 1 if saturated (near count > cap).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_composition(s: str) -> tuple[int, int]:
    m = re.match(r"^([A-Za-z0-9]+):(\d+)$", s.strip())
    if not m:
        raise ValueError(f"Expected RES:COUNT (e.g. DCM:90), got {s!r}")
    return int(m.group(2)), 10


def _load_positions_npy(path: Path) -> "np.ndarray":
    import numpy as np

    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _load_positions_crd(path: Path) -> "np.ndarray":
    """Minimal CHARMM CRD reader (EXT format with coords in Å)."""
    import numpy as np

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"CRD too short: {path}")
    try:
        n_atoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Cannot parse atom count from CRD header: {path}") from e
    coords = []
    for line in lines[2:]:
        if len(coords) >= n_atoms:
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        coords.append([x, y, z])
    if len(coords) != n_atoms:
        raise ValueError(f"Expected {n_atoms} coordinates in {path}, found {len(coords)}")
    return np.asarray(coords, dtype=np.float64)


def _find_crd_in_output_dir(out_dir: Path, tag: str | None) -> Path | None:
    if tag:
        p = out_dir / f"mini_full_mlpot_{tag}.crd"
        if p.is_file():
            return p
    candidates = sorted(out_dir.glob("mini_full_mlpot_*.crd"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--positions-npy", type=Path, help="Coordinates (N, 3) in Å")
    src.add_argument("--crd", type=Path, help="CHARMM CRD card from mini/MD")
    src.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact dir; uses mini_full_mlpot_*.crd (with optional --tag)",
    )
    parser.add_argument("--tag", type=str, default=None, help="Artifact tag for --output-dir")
    parser.add_argument("--composition", type=str, default=None, help="e.g. DCM:90 (uniform monomers)")
    parser.add_argument("--n-monomers", type=int, default=None)
    parser.add_argument("--atoms-per-monomer", type=int, default=None)
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=5.0,
        help="COM distance cutoff (Å), must match CutoffParameters.mm_switch_on",
    )
    parser.add_argument("--box-size", type=float, default=None, help="Cubic PBC side (Å)")
    parser.add_argument(
        "--ml-max-active-dimers",
        type=int,
        default=None,
        help="Cap to test (default: resolve_max_active_dimers policy)",
    )
    parser.add_argument(
        "--proposed-cap",
        type=int,
        default=None,
        help="Also report stats for a hypothetical cap (what-if)",
    )
    args = parser.parse_args()

    if args.composition:
        n_mol, n_apm = _parse_composition(args.composition)
        n_monomers = args.n_monomers or n_mol
        atoms_per = args.atoms_per_monomer or n_apm
    else:
        if args.n_monomers is None or args.atoms_per_monomer is None:
            parser.error("Provide --composition RES:N or both --n-monomers and --atoms-per-monomer")
        n_monomers = int(args.n_monomers)
        atoms_per = int(args.atoms_per_monomer)

    if args.positions_npy:
        pos = _load_positions_npy(args.positions_npy.expanduser())
    elif args.crd:
        pos = _load_positions_crd(args.crd.expanduser())
    else:
        out = args.output_dir.expanduser()
        crd = _find_crd_in_output_dir(out, args.tag)
        if crd is None:
            print(f"No mini_full_mlpot_*.crd under {out}", file=sys.stderr)
            return 2
        print(f"Using CRD: {crd}")
        pos = _load_positions_crd(crd)

    expected_atoms = n_monomers * atoms_per
    if pos.shape[0] != expected_atoms:
        print(
            f"WARN: position rows {pos.shape[0]} != n_monomers*atoms_per_monomer={expected_atoms}",
            file=sys.stderr,
        )

    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        resolve_max_active_dimers,
        validate_sparse_dimer_cap,
    )

    stats = validate_sparse_dimer_cap(
        pos,
        n_monomers,
        atoms_per,
        mm_switch_on=args.mm_switch_on,
        box_side_A=args.box_size,
        max_active_dimers=args.ml_max_active_dimers,
    )

    print("Sparse ML dimer validation")
    print(f"  n_monomers           = {stats['n_monomers']}")
    print(f"  n_dimers_total       = {stats['n_dimers_total']}  (C(n,2))")
    print(f"  mm_switch_on         = {stats['mm_switch_on_A']:.3f} Å")
    print(f"  near dimers (< cut)  = {stats['n_near_mm_switch_on']}")
    print(f"  cap (slots)          = {stats['max_active_dimers_cap']}")
    print(f"  cap margin           = {stats['cap_margin']}")
    print(f"  PhysNet padded batch = {stats['physnet_systems_padded_batch']} systems/step")
    print(f"  PhysNet if no trunc  = {stats['physnet_systems_per_step']} systems/step")
    default_cap = resolve_max_active_dimers(n_monomers, int(stats["n_dimers_total"]))
    if args.ml_max_active_dimers is None:
        print(f"  default policy cap   = {default_cap}")

    if args.proposed_cap is not None:
        alt = validate_sparse_dimer_cap(
            pos,
            n_monomers,
            atoms_per,
            mm_switch_on=args.mm_switch_on,
            box_side_A=args.box_size,
            max_active_dimers=args.proposed_cap,
        )
        print(f"\nProposed cap {args.proposed_cap}: {alt['verdict']}")

    print(f"\n{stats['verdict']}")
    if not stats["ok"]:
        print(
            "\nSuggest: raise --ml-max-active-dimers or export MMML_MLPOT_MAX_ACTIVE_DIMERS; "
            "re-run this script after mini. Watch GPU memory (cap raises PhysNet batch size).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
