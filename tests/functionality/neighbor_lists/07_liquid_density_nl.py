#!/usr/bin/env python3
"""Neighbor-list parity at bulk liquid densities (PBC).

Builds clusters sized from experimental solvent densities (ACO, DCM) using
``box_sizing`` / ``bulk_density`` helpers, then compares NL backends to the
reference oracle. Synthetic toy monomers run without CHARMM; ``charmm_*`` cases
use CGENFF when PyCHARMM is available.

Examples
--------
  uv run python tests/functionality/neighbor_lists/07_liquid_density_nl.py
  uv run python tests/functionality/neighbor_lists/07_liquid_density_nl.py \\
      --case synthetic_aco_liquid_n16 --backends vesin,jax_md,ase,cell_list
  uv run python tests/functionality/neighbor_lists/07_liquid_density_nl.py \\
      --with-charmm --case charmm_aco_liquid_n16
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from _common import (
    build_liquid_density_synthetic_case,
    charmm_liquid_density_cases,
    have_charmm_nl,
    liquid_density_synthetic_cases,
    print_fail,
    print_header,
    print_pass,
    setup_charmm_liquid_density_cluster,
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
    box_side: float,
    rho_g_cm3: float,
) -> bool:
    nl05 = _load_compare_module()
    ref, ref_src = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )
    print(
        f"\n--- {name} ---\n"
        f"  {description}\n"
        f"  n_atoms={positions.shape[0]}  L={box_side:.2f} Å  "
        f"ρ_target={rho_g_cm3:.3f} g/cm³  cutoff={cutoff:.2f} Å  "
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
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list,pycharmm",
    )
    parser.add_argument(
        "--with-charmm",
        action="store_true",
        help="Run CGENFF liquid-density cases (auto when pycharmm is in --backends)",
    )
    args = parser.parse_args()

    print_header("Liquid-density PBC neighbor-list cases")
    nl05 = _load_compare_module()
    backends = nl05._parse_backends(args.backends)
    run_charmm_suite = bool(args.with_charmm or "pycharmm" in backends)
    synthetic_backends = [b for b in backends if b != "pycharmm"]

    synthetic_all = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    charmm_all = {str(c["name"]): c for c in charmm_liquid_density_cases()}

    if args.case:
        selected_synthetic = [n for n in args.case if n in synthetic_all]
        selected_charmm = [n for n in args.case if n in charmm_all]
        unknown = sorted(set(args.case) - set(selected_synthetic) - set(selected_charmm))
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
        positions, cell, offsets, monomer_id, cutoff, desc, side, rho = (
            build_liquid_density_synthetic_case(synthetic_all[name])
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
            box_side=side,
            rho_g_cm3=rho,
        )
        ok = ok and case_ok

    if run_charmm_suite and selected_charmm:
        if not have_charmm_nl():
            print("\nSKIP CHARMM liquid-density cases: PyCHARMM/CGENFF not available")
        else:
            print("\n=== CHARMM/CGENFF liquid-density cases ===")
            charmm_suite_skipped = False
            for name in selected_charmm:
                case = charmm_all[name]
                try:
                    (
                        positions,
                        cell,
                        offsets,
                        monomer_id,
                        atomic_numbers,
                        eff_cutoff,
                        side,
                        rho,
                    ) = setup_charmm_liquid_density_cluster(
                        case,
                        cutoff=float(case["cutoff"]),
                    )
                except Exception as exc:
                    if not charmm_suite_skipped:
                        print(f"SKIP CHARMM liquid-density cases: {exc}")
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
                    box_side=side,
                    rho_g_cm3=rho,
                )
                ok = ok and case_ok

    if ok:
        print_pass("all liquid-density PBC cases match reference")
        return 0
    print_fail("one or more liquid-density PBC cases failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
