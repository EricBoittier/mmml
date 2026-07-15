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

    last_R = positions.copy()
    R_small = positions + 0.01
    box = np.array([40.0, 40.0, 40.0])

    # skin=0: never reuse (Verlet-unsafe to skip rebuilds without a buffer)
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
        if not reuse_interval:
            print_pass(f"skin=0 interval={interval}: never reuses (always rebuild)")
        else:
            print_fail("skin=0 must not reuse cached pairs")
            ok = False

    # NVT: small displacement, stable box — skin>0 should reuse
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

        # Forced rebuild on interval boundary
        no_reuse_on_rebuild_step = neighbor_pair_cache_should_reuse(
            calls=interval,
            interval=interval,
            skin=skin,
            R=R_small,
            last_R=last_R,
            box=box,
            last_box=box.copy(),
            have_cache=True,
        )
        if interval > 1 and not no_reuse_on_rebuild_step:
            print_pass(f"skin>0: call {interval} forces rebuild")
        elif interval <= 1 and reuse:
            print_pass("skin>0 interval=1: skin check only")
        elif interval > 1:
            print_fail(f"skin>0: call {interval} should not reuse")
            ok = False

    # NPT: box change must invalidate
    boxes = npt_box_sequence()
    npt_reuse = neighbor_pair_cache_should_reuse(
        calls=2,
        interval=interval,
        skin=skin,
        R=positions if skin <= 0.0 else R_small,
        last_R=last_R,
        box=boxes[1],
        last_box=boxes[0],
        have_cache=True,
    )
    if not npt_reuse:
        print_pass("NPT box resize invalidates cache reuse")
    else:
        print_fail("NPT box resize must not reuse cached pairs")
        ok = False

    # Large displacement with skin>0 (beyond half-skin)
    if skin > 0.0:
        R_large = positions + (0.5 * skin + 0.05)
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
            print_pass("displacement beyond skin/2 forces rebuild")
        else:
            print_fail("displacement beyond skin/2 must not reuse")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
