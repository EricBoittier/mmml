#!/usr/bin/env python3
"""Extreme PBC neighbor-list cases (tight boxes, face wraps, orthorhombic cells).

Each named geometry stresses MIC pair generation under periodic boundaries. All
backends are compared to the Vesin / brute-force reference oracle.

Synthetic cases (no CHARMM) cover toy clusters and orthorhombic cells. When
``pycharmm`` is included in ``--backends`` (or ``--with-charmm``), matching
CGENFF composition cases are run so PyCHARMM nbonds can participate.

Examples
--------
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py \\
      --case wrap_straddle_x --backends vesin,jax_md,ase,cell_list
  uv run python tests/functionality/neighbor_lists/06_extreme_pbc_nl.py \\
      --with-charmm --case charmm_high_cutoff_fraction
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from _common import (
    build_extreme_pbc_case,
    charmm_extreme_pbc_cases,
    extreme_pbc_cases,
    have_charmm_nl,
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
    cell_diag = np.diag(cell) if cell.ndim == 2 else None
    if cell_diag is not None and np.allclose(cell, np.diag(cell_diag)):
        box_label = f"L={float(cell_diag[0]):.1f}"
    elif cell.ndim == 2:
        box_label = f"cell=({cell[0,0]:.0f},{cell[1,1]:.0f},{cell[2,2]:.0f})"
    else:
        box_label = f"L={float(cell):.1f}"
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
        help="Run only named case(s); prefix charmm_ for CGENFF analog cases",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list,pycharmm",
        help="Comma-separated backends (same as 05_compare_nl_backends.py)",
    )
    parser.add_argument(
        "--with-charmm",
        action="store_true",
        help="Also run CGENFF CHARMM analog cases (auto when pycharmm is in --backends)",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Optional extra one-off CHARMM case, e.g. ACO:2",
    )
    parser.add_argument("--box-side", type=float, default=18.0)
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument("--cutoff", type=float, default=13.0)
    args = parser.parse_args()

    print_header("Extreme PBC neighbor-list cases")
    nl05 = _load_compare_module()
    backends = nl05._parse_backends(args.backends)
    run_charmm_suite = bool(args.with_charmm or "pycharmm" in backends)
    synthetic_backends = [b for b in backends if b != "pycharmm"]

    synthetic_all = {str(c["name"]): c for c in extreme_pbc_cases()}
    charmm_all = {str(c["name"]): c for c in charmm_extreme_pbc_cases()}

    if args.case:
        selected_synthetic = [n for n in args.case if n in synthetic_all]
        selected_charmm = [n for n in args.case if n in charmm_all]
        unknown = sorted(
            set(args.case) - set(selected_synthetic) - set(selected_charmm)
        )
        if unknown:
            print_fail(f"unknown case(s): {', '.join(unknown)}")
            return 1
        if selected_charmm:
            run_charmm_suite = True
    else:
        selected_synthetic = list(synthetic_all.keys())
        selected_charmm = list(charmm_all.keys()) if run_charmm_suite else []

    ok = True
    for name in selected_synthetic:
        positions, cell, offsets, monomer_id, cutoff, desc = build_extreme_pbc_case(
            synthetic_all[name]
        )
        case_ok = _run_case(
            name=name,
            description=desc,
            positions=positions,
            cell=cell,
            offsets=offsets,
            monomer_id=monomer_id,
            cutoff=cutoff,
            backends=synthetic_backends,
            atomic_numbers=None,
            charmm_geometry=False,
        )
        ok = ok and case_ok

    if run_charmm_suite and selected_charmm:
        if not have_charmm_nl():
            print("\nSKIP CHARMM extreme cases: PyCHARMM/CGENFF not available")
        else:
            print("\n=== CHARMM/CGENFF extreme PBC analogs ===")
            charmm_suite_skipped = False
            for name in selected_charmm:
                case = charmm_all[name]
                try:
                    positions, cell, offsets, monomer_id, atomic_numbers, eff_cutoff = (
                        nl05._setup_composition_geometry(
                            str(case["composition"]),
                            cutoff=float(case["cutoff"]),
                            box_side=float(case["box_side"]),
                            spacing=float(case["spacing"]),
                        )
                    )
                except Exception as exc:
                    if not charmm_suite_skipped:
                        print(f"SKIP CHARMM extreme cases: {exc}")
                        charmm_suite_skipped = True
                    break
                case_ok = _run_case(
                    name=name,
                    description=str(case["description"]),
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

    if args.composition:
        if not have_charmm_nl():
            print_fail(f"composition {args.composition}: PyCHARMM/CGENFF not available")
            return 1
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
