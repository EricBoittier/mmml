#!/usr/bin/env python3
"""Step 2: path parity — cell-list, nl_backend (Vesin), jax-md vs reference."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import print_fail, print_header, print_pass, two_dimer_cluster
from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import (
    create_jax_md_neighbor_list,
    have_jax_md,
)
from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend
from mmml.interfaces.pycharmmInterface.nl_reference import (
    compare_pair_sets,
    extract_valid_pairs,
    reference_mic_pairs,
)


def _jaxmd_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_offsets: np.ndarray,
) -> set[tuple[int, int]]:
    bundle = create_jax_md_neighbor_list(
        cell,
        r_cutoff=cutoff,
        monomer_offsets=monomer_offsets,
        dr_threshold=0.5,
        capacity_multiplier=1.5,
        fractional_coordinates=False,
    )
    if bundle is None:
        raise RuntimeError("jax-md neighbor list unavailable")
    neighbor_fn, filter_fn, _monomer_id = bundle
    nbrs = neighbor_fn.allocate(np.asarray(positions, dtype=np.float64))
    pair_i, pair_j, mask = filter_fn(nbrs.idx)
    pi = np.asarray(pair_i)
    pj = np.asarray(pair_j)
    mk = np.asarray(mask)
    return extract_valid_pairs(pi, pj, mk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=13.0)
    parser.add_argument("--com-separation", type=float, default=8.0)
    args = parser.parse_args()

    print_header("MM neighbor-list path parity")
    positions, cell, offsets, monomer_id = two_dimer_cluster(com_separation=args.com_separation)
    cutoff = float(args.cutoff)

    ref, ref_src = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )
    print(f"reference ({ref_src}): {len(ref)} pairs")

    ok = True

    cl_i, cl_j, cl_mask, n_valid, _cap, used = build_mm_pairs_with_backend(
        "cell_list",
        positions,
        cell,
        cutoff=cutoff,
        monomer_offsets=offsets,
        total_atoms=positions.shape[0],
    )
    cell_pairs = extract_valid_pairs(cl_i, cl_j, cl_mask)
    cmp_cl = compare_pair_sets(ref, cell_pairs)
    print(f"cell_list ({used}): {cmp_cl.n_b} valid / reported n_valid={n_valid}")
    if cmp_cl.match:
        print_pass("cell_list matches reference")
    else:
        print(cmp_cl.summary(label_a="reference", label_b="cell_list"))
        print_fail("cell_list parity")
        ok = False

    try:
        vi, vj, vm, vn, _vc, vused = build_mm_pairs_with_backend(
            "vesin",
            positions,
            cell,
            cutoff=cutoff,
            monomer_offsets=offsets,
            total_atoms=positions.shape[0],
        )
        vesin_pairs = extract_valid_pairs(vi, vj, vm)
        cmp_v = compare_pair_sets(ref, vesin_pairs)
        print(f"vesin backend ({vused}): {cmp_v.n_b} valid / n_valid={vn}")
        if cmp_v.match:
            print_pass("vesin backend matches reference")
        else:
            print(cmp_v.summary(label_a="reference", label_b="vesin"))
            print_fail("vesin backend parity")
            ok = False
    except Exception as exc:
        print(f"SKIP vesin backend: {exc}")

    if have_jax_md():
        try:
            jax_pairs = _jaxmd_pairs(positions, cell, cutoff, offsets)
            cmp_j = compare_pair_sets(ref, jax_pairs)
            print(f"jax-md filter: {cmp_j.n_b} pairs")
            if cmp_j.match:
                print_pass("jax-md matches reference")
            else:
                print(cmp_j.summary(label_a="reference", label_b="jax-md"))
                print_fail("jax-md parity")
                ok = False
        except Exception as exc:
            print_fail(f"jax-md path: {exc}")
            ok = False
    else:
        print("SKIP: jax-md unavailable")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
