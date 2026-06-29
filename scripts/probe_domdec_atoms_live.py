#!/usr/bin/env python3
"""Probe DOMDEC atom maps during a live dynamics step.

Run with mpirun on 8 ranks (NDIR 8 1 1):

    mpirun -np 8 python scripts/probe_domdec_atoms_live.py \\
        --psf  path/to/system.psf \\
        --crd  path/to/system.crd \\
        --box  <box_side_A>

The script runs ONE energy evaluation with DOMDEC active and prints:
  - Per-rank: natoml, natoml_tot, local atom indices, ghost atom indices
  - Verification that local + ghost = natoml_tot
  - Verification that local atom index sets are disjoint across ranks

Output is written to domdec_probe_rank<N>.txt so ranks don't interleave.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--psf",  required=True)
    p.add_argument("--crd",  required=True)
    p.add_argument("--box",  type=float, required=True, help="Cubic box side (Å)")
    p.add_argument("--ndir", type=int,   default=0,
                   help="NDIR along x (0 = auto from n_ranks)")
    p.add_argument("--cutnb", type=float, default=14.0, help="Nonbond cutoff (Å)")
    p.add_argument("--ctofnb", type=float, default=12.0)
    p.add_argument("--ctonnb", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ MPI
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.lingo as lingo

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nranks = comm.Get_size()
    except ImportError:
        rank = 0
        nranks = 1

    # --- sanity: DOMDEC needs domain_width >= cutnb
    # c47 axis rule: each axis must have 1 or >=8 domains
    ndir_requested = args.ndir if args.ndir > 0 else nranks
    domain_width = args.box / ndir_requested
    min_box_for_ndir = ndir_requested * args.cutnb
    if domain_width < args.cutnb:
        if rank == 0:
            print(
                f"WARNING: box={args.box:.1f} Å / ndir={ndir_requested} = "
                f"{domain_width:.1f} Å domain width < cutnb={args.cutnb:.1f} Å.\n"
                f"  DOMDEC requires domain_width >= cutnb  →  "
                f"min box for ndir={ndir_requested}: {min_box_for_ndir:.0f} Å.\n"
                f"  Falling back to ndir=1 (no DOMDEC decomposition, just 1 direct rank).\n"
                f"  To use ndir=8 you need box >= {min_box_for_ndir:.0f} Å  "
                f"(≈ {int(min_box_for_ndir**3 / (args.box**3 / 10)):.0f} DCM molecules "
                f"for the same density).",
                flush=True,
            )
        ndir = 1
    else:
        ndir = ndir_requested

    log = open(f"domdec_probe_rank{rank:02d}.txt", "w")

    def pr(*a, **kw):
        print(*a, **kw, file=log, flush=True)

    pr(f"=== rank {rank}/{nranks}  NDIR {ndir} 1 1 ===")

    # ------------------------------------------------------------------ locate RTF/PRM via MMML data dir
    import pathlib
    import mmml as _mmml

    _data = pathlib.Path(_mmml.__file__).parent / "data" / "charmm"
    rtf = os.environ.get("MMML_RTF") or str(_data / "top_all36_cgenff.rtf")
    prm = os.environ.get("MMML_PRM") or str(_data / "par_all36_cgenff.prm")

    pr(f"RTF : {rtf}")
    pr(f"PRM : {prm}")
    pr(f"PSF : {args.psf}")
    pr(f"CRD : {args.crd}")

    # ------------------------------------------------------------------ load
    lingo.charmm_script(f"""
read rtf  card name {rtf}
read param card name {prm} flex
read psf  card name {args.psf}
read coor card name {args.crd}
""")

    # ------------------------------------------------------------------ PBC
    L = args.box
    lingo.charmm_script(f"""
crystal define cubic {L} {L} {L} 90 90 90
crystal build cutoff {args.cutnb} noper 0
image byres xcen 0 ycen 0 zcen 0
""")

    # ------------------------------------------------------------------ DOMDEC energy
    # fftx/y/z must be >= box/spacing; use 32 for boxes ~40 Å, 64 for ~80 Å
    fft = max(32, int(args.box / 1.2 / 2) * 2)   # even integer, ~box/1.2
    lingo.charmm_script(f"""
faster on
energy cutnb {args.cutnb} ctofnb {args.ctofnb} ctonnb {args.ctonnb} -
    vfswitch atom fswitch -
    domd ndir {ndir} 1 1 -
    ewald kappa 0.32 order 6 fftx {fft} ffty {fft} fftz {fft}
""")

    # ------------------------------------------------------------------ probe
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
        domdec_summary,
        get_ghost_atom_count,
        get_ghost_atom_indices,
        get_local_atom_count,
        get_local_atom_indices,
        get_ndir,
        is_domdec_active,
    )

    pr(domdec_summary())
    pr()

    active  = is_domdec_active()
    nx, ny, nz = get_ndir()
    nlocal  = get_local_atom_count()
    nghost  = get_ghost_atom_count()
    local_i = get_local_atom_indices()
    ghost_i = get_ghost_atom_indices()

    pr(f"active={active}  NDIR=({nx},{ny},{nz})")
    pr(f"natoml={nlocal}  nghost={nghost}  sum={nlocal+nghost}")
    pr(f"local_idx[:10] = {local_i[:10].tolist()}")
    pr(f"ghost_idx[:10] = {ghost_i[:10].tolist()}")

    # Sanity checks (single-rank only — cross-rank disjoint check needs gather)
    if nlocal + nghost > 0:
        overlap = set(local_i.tolist()) & set(ghost_i.tolist())
        pr(f"local∩ghost overlap (should be 0): {len(overlap)}")

    # ------------------------------------------------------------------ MPI gather: verify disjoint local sets
    if nranks > 1:
        try:
            all_local = comm.gather(set(local_i.tolist()), root=0)
            if rank == 0 and all_local is not None:
                from functools import reduce
                union = reduce(lambda a, b: a | b, all_local)
                intersections = sum(
                    len(all_local[i] & all_local[j])
                    for i in range(nranks) for j in range(i + 1, nranks)
                )
                total_local = sum(len(s) for s in all_local)
                pr(f"\n[rank 0] Cross-rank union  size: {len(union)}")
                pr(f"[rank 0] Cross-rank total local: {total_local}")
                pr(f"[rank 0] Cross-rank pair overlaps: {intersections}  (should be 0)")
        except Exception as e:
            pr(f"MPI gather failed: {e}")

    log.close()


if __name__ == "__main__":
    main()
