#!/usr/bin/env python3
"""Extreme PBC neighbor-list cases (tight boxes, face wraps, orthorhombic cells).

Each named geometry stresses MIC pair generation under periodic boundaries. All
backends are compared to the Vesin / brute-force reference oracle.

Examples
--------
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py \\
      --case wrap_straddle_x --backends vesin,jax_md,ase,cell_list
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py \\
      --composition ACO:2 --box-side 18 --spacing 4
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from _common import (
    build_extreme_pbc_case,
    extreme_pbc_cases,
    print_fail,
    print_header,
    print_pass,
)
from mmml.interfaces.pycharmmInterface.nl_reference import compare_pair_sets, reference_mic_pairs

_DIR = Path(__file__).resolve().parent


def _load_compare_module():
    path = _DIR / "05_compare_nl_backends.py"
    spec = importlib.util.spec_from_file_location("nl_compare_backends", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_case(
    *,
    name: str,
    description: str,
    positions: np.ndarray,
    cell: np.ndarray,
    offsets: np.ndarray,
    monomer_id: np.ndarray,
    cutoff: float,
    backends: list[str],
    atomic_numbers: np.ndarray | None,
    charmm_geometry: bool,
) -> bool:
    nl05 = _load_compare_module()
    ref, ref_src = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )
    box_label = (
        f"L={float(cell[0, 0]):.1f}"
        if np.allclose(cell, np.diag(np.diag(cell)))
        else f"cell=({cell[0,0]:.0f},{cell[1,1]:.0f},{cell[2,2]:.0f})"
    )
    print(
        f"\n--- {name} ---\n"
        f"  {description}\n"
        f"  n_atoms={positions.shape[0]}  {box_label}  cutoff={cutoff:.2f} Å  "
        f"reference={ref_src} ({len(ref)} pairs)"
    )

    ok = True
    collected: dict[str, set[tuple[int, int]]] = {}
    for backend in backends:
        label, pairs, skip_reason = nl05._collect_backend(
            backend,
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            monomer_id=monomer_id,
            monomer_offsets=offsets,
            charmm_geometry=charmm_geometry,
            atomic_numbers=atomic_numbers,
        )
        if pairs is None:
            print(f"  SKIP {label}: {skip_reason or 'skipped'}")
            continue
        collected[label] = pairs

    if not collected:
        print_fail(f"{name}: no backends produced pair sets")
        return False

    print(f"  {'backend':<12} {'pairs':>6} {'only_ref':>9} {'only_be':>9}  status")
    for label, pairs in collected.items():
        cmp = compare_pair_sets(ref, pairs)
        status = "PASS" if cmp.match else "FAIL"
        if not cmp.match:
            ok = False
        print(
            f"  {label:<12} {cmp.n_b:>6} {len(cmp.only_a):>9} {len(cmp.only_b):>9}  {status}"
        )
        if not cmp.match:
            print(cmp.summary(label_a="reference", label_b=label, max_show=6))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only named case(s); default is all synthetic extreme cases",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list,pycharmm",
        help="Comma-separated backends (same as 05_compare_nl_backends.py)",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Optional CHARMM composition appended as extra case(s), e.g. ACO:2",
    )
    parser.add_argument("--box-side", type=float, default=18.0)
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument("--cutoff", type=float, default=13.0)
    args = parser.parse_args()

    print_header("Extreme PBC neighbor-list cases")
    nl05 = _load_compare_module()
    backends = nl05._parse_backends(args.backends)

    all_cases = {str(c["name"]): c for c in extreme_pbc_cases()}
    selected = list(all_cases.keys()) if not args.case else [str(n) for n in args.case]
    unknown = sorted(set(selected) - set(all_cases))
    if unknown:
        print_fail(f"unknown case(s): {', '.join(unknown)}")
        return 1

    ok = True
    for name in selected:
        positions, cell, offsets, monomer_id, cutoff, desc = build_extreme_pbc_case(
            all_cases[name]
        )
        case_ok = _run_case(
            name=name,
            description=desc,
            positions=positions,
            cell=cell,
            offsets=offsets,
            monomer_id=monomer_id,
            cutoff=cutoff,
            backends=backends,
            atomic_numbers=None,
            charmm_geometry=False,
        )
        ok = ok and case_ok

    if args.composition:
        try:
            positions, cell, offsets, monomer_id, atomic_numbers, eff_cutoff = (
                nl05._setup_composition_geometry(
                    args.composition,
                    cutoff=float(args.cutoff),
                    box_side=float(args.box_side),
                    spacing=float(args.spacing),
                )
            )
        except Exception as exc:
            print_fail(f"composition case: {exc}")
            return 1
        comp_name = f"charmm_{args.composition.replace(',', '_').replace(':', 'x')}"
        case_ok = _run_case(
            name=comp_name,
            description=(
                f"CGENFF {args.composition} tight PBC "
                f"(box={args.box_side:.0f} Å, spacing={args.spacing:.1f} Å)"
            ),
            positions=positions,
            cell=cell,
            offsets=offsets,
            monomer_id=monomer_id,
            cutoff=eff_cutoff,
            backends=backends,
            atomic_numbers=atomic_numbers,
            charmm_geometry=True,
        )
        ok = ok and case_ok

    if ok:
        print_pass("all extreme PBC cases match reference")
        return 0
    print_fail("one or more extreme PBC cases failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
