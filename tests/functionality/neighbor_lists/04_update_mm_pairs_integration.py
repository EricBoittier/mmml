#!/usr/bin/env python3
"""Step 4: integration — real update_mm_pairs via build_mm_energy_forces_fn (PyCHARMM)."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import (
    npt_box_sequence,
    print_fail,
    print_header,
    print_pass,
    setup_charmm_aco_dimer_cluster,
)


def _build_update_fn(skip_charmm: bool, mm_nl_backend: str = "auto"):
    if skip_charmm:
        return None

    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    if CGENFF_PRM is None:
        raise RuntimeError("PyCHARMM/CGENFF not available")

    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    positions, cell, offsets, _mid = setup_charmm_aco_dimer_cluster()
    n_monomers = len(offsets) - 1
    atoms_per = int(offsets[1] - offsets[0])
    atoms_list = [atoms_per] * n_monomers

    result = build_mm_energy_forces_fn(
        positions,
        total_atoms=positions.shape[0],
        n_monomers=n_monomers,
        monomer_offsets=offsets,
        atoms_per_monomer_list=atoms_list,
        lambda_monomer=np.ones(n_monomers, dtype=np.float64),
        ml_switch_width=1.0,
        mm_switch_on=12.0,
        mm_switch_width=1.0,
        pbc_cell=float(cell[0, 0]),
        jax_md_skin_distance=0.0,
        jax_md_update_interval=3,
        defer_xla_gpu_warmup=True,
        mm_nl_backend=mm_nl_backend,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"expected (mm_fn, update_mm_pairs) from dynamic NL path ({mm_nl_backend})")
    _mm_fn, update_fn = result
    return update_fn, positions, cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mm-nl-backend",
        default="auto",
        choices=("auto", "vesin", "jax_md", "cell_list"),
        help="MM neighbor-list backend (default: auto → Vesin when installed)",
    )
    parser.add_argument(
        "--skip-charmm",
        action="store_true",
        help="Skip PyCHARMM integration (exit 0 with SKIP message)",
    )
    args = parser.parse_args()

    print_header("update_mm_pairs integration")
    if args.skip_charmm:
        print("SKIP: --skip-charmm (run locally with CHARMM_HOME / CGENFF)")
        return 0

    try:
        built = _build_update_fn(skip_charmm=False, mm_nl_backend=args.mm_nl_backend)
    except Exception as exc:
        print_fail(f"build_mm_energy_forces_fn: {exc}")
        print("Hint: run with --skip-charmm or set up PyCHARMM + CGENFF")
        return 1

    if built is None:
        print("SKIP: no update_fn")
        return 0

    update_fn, positions, cell = built
    box_diag = np.diagonal(cell).astype(np.float64)
    L0 = float(box_diag[0])

    update_fn(positions, box=box_diag)
    stats1 = update_fn.get_stats()
    calls_after_first = stats1["calls"]

    # interval skip
    update_fn(positions, box=box_diag)
    stats2 = update_fn.get_stats()
    if stats2["reused"] > stats1.get("reused", 0):
        print_pass("second call reused cache (skin=0, interval=3)")
    else:
        print_fail(f"expected reuse on call 2; stats={stats2}")
        return 1

    # box change must rebuild
    boxes = npt_box_sequence(L0)
    update_fn(positions, box=boxes[1])
    stats3 = update_fn.get_stats()
    if stats3["updates"] > stats2["updates"]:
        print_pass("NPT box change triggered rebuild")
    else:
        print_fail(f"box change should rebuild; stats={stats3}")
        return 1

    print(f"final stats: {stats3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
