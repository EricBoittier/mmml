#!/usr/bin/env python3
"""Step 3: audit skin/interval cache predicate (no PyCHARMM required)."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import npt_box_sequence, print_fail, print_header, print_pass, two_dimer_cluster
from mmml.interfaces.pycharmmInterface.mm_energy_forces import neighbor_pair_cache_should_reuse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skin", type=float, default=0.0)
    parser.add_argument("--interval", type=int, default=3)
    args = parser.parse_args()

    print_header("Skin / interval cache audit")
    positions, _cell, _offsets, _mid = two_dimer_cluster()
    skin = float(args.skin)
    interval = int(args.interval)
    ok = True

    # NVT: small displacement, stable box — skin>0 should reuse
    last_R = positions.copy()
    R_small = positions + 0.01
    box = np.array([40.0, 40.0, 40.0])
    if skin > 0.0:
        reuse = neighbor_pair_cache_should_reuse(
            calls=2,
            interval=interval,
            skin=skin,
            R=R_small,
            last_R=last_R,
            box=box,
            last_box=box.copy(),
            have_cache=True,
        )
        if reuse:
            print_pass(f"skin={skin}: small displacement reuses cache")
        else:
            print_fail(f"skin={skin}: expected reuse on small displacement")
            ok = False

    # skin=0: interval skip with stable box
    if skin == 0.0:
        reuse_interval = neighbor_pair_cache_should_reuse(
            calls=2,
            interval=interval,
            skin=0.0,
            R=positions,
            last_R=last_R,
            box=box,
            last_box=box.copy(),
            have_cache=True,
        )
        if reuse_interval:
            print_pass(f"skin=0 interval={interval}: call 2 skips rebuild (stable box)")
        else:
            print_fail("skin=0 interval skip failed with stable box")
            ok = False

        no_reuse_on_rebuild_step = neighbor_pair_cache_should_reuse(
            calls=interval,
            interval=interval,
            skin=0.0,
            R=positions,
            last_R=last_R,
            box=box,
            last_box=box.copy(),
            have_cache=True,
        )
        if not no_reuse_on_rebuild_step:
            print_pass(f"skin=0: call {interval} triggers rebuild step")
        else:
            print_fail(f"skin=0: call {interval} should not reuse")
            ok = False

    # NPT: box change must invalidate skin=0 interval skip
    boxes = npt_box_sequence()
    npt_reuse = neighbor_pair_cache_should_reuse(
        calls=2,
        interval=interval,
        skin=0.0,
        R=positions,
        last_R=last_R,
        box=boxes[1],
        last_box=boxes[0],
        have_cache=True,
    )
    if not npt_reuse:
        print_pass("NPT box resize invalidates skin=0 interval reuse")
    else:
        print_fail("NPT box resize must not reuse cached pairs")
        ok = False

    # skin>0: box change must invalidate even if displacement small
    if skin > 0.0:
        npt_skin = neighbor_pair_cache_should_reuse(
            calls=2,
            interval=interval,
            skin=skin,
            R=R_small,
            last_R=last_R,
            box=boxes[1],
            last_box=boxes[0],
            have_cache=True,
        )
        if not npt_skin:
            print_pass("NPT box resize invalidates skin>0 reuse")
        else:
            print_fail("NPT box resize must invalidate skin>0 cache")
            ok = False

    # Large displacement with skin>0
    if skin > 0.0:
        R_large = positions + (skin + 0.1)
        large = neighbor_pair_cache_should_reuse(
            calls=2,
            interval=interval,
            skin=skin,
            R=R_large,
            last_R=last_R,
            box=box,
            last_box=box.copy(),
            have_cache=True,
        )
        if not large:
            print_pass("large displacement forces rebuild")
        else:
            print_fail("large displacement must not reuse")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
