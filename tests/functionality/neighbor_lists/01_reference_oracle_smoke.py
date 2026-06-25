#!/usr/bin/env python3
"""Step 1: Vesin vs brute-force reference oracle on toy PBC clusters."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import print_fail, print_header, print_pass, two_dimer_cluster
from mmml.interfaces.pycharmmInterface.nl_reference import (
    brute_force_mic_pairs,
    compare_pair_sets,
    have_vesin,
    reference_mic_pairs,
    vesin_mic_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=13.0)
    parser.add_argument("--com-separation", type=float, default=8.0)
    args = parser.parse_args()

    print_header("Reference oracle smoke (Vesin vs brute-force)")
    positions, cell, offsets, monomer_id = two_dimer_cluster(com_separation=args.com_separation)
    cutoff = float(args.cutoff)

    brute = brute_force_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )
    print(f"brute-force: {len(brute)} pairs")

    ref, source = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
        prefer_vesin=True,
    )
    cmp = compare_pair_sets(brute, ref)
    print(f"reference ({source}): {cmp.n_b} pairs")
    if cmp.match:
        print_pass(f"brute matches {source} reference")
    else:
        print(cmp.summary(label_a="brute", label_b=source))
        print_fail("reference oracle mismatch")
        return 1

    if have_vesin():
        vesin_only = vesin_mic_pairs(
            positions,
            cell,
            cutoff,
            monomer_id,
            monomer_offsets=offsets,
        )
        cmp_v = compare_pair_sets(brute, vesin_only)
        if cmp_v.match:
            print_pass("vesin matches brute-force")
        else:
            print(cmp_v.summary(label_a="brute", label_b="vesin"))
            print_fail("vesin vs brute mismatch")
            return 1
    else:
        print("SKIP: vesin not installed")

    # Edge: smaller separation (fewer pairs)
    positions2, cell2, offsets2, mid2 = two_dimer_cluster(com_separation=20.0)
    ref2, _ = reference_mic_pairs(positions2, cell2, cutoff, mid2, monomer_offsets=offsets2)
    brute2 = brute_force_mic_pairs(positions2, cell2, cutoff, mid2, monomer_offsets=offsets2)
    if compare_pair_sets(brute2, ref2).match:
        print_pass("wide-separation geometry")
    else:
        print_fail("wide-separation geometry mismatch")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
