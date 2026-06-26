#!/usr/bin/env python3
"""Compare neighbor lists across Vesin, jax-md, ASE, cell-list, and PyCHARMM.

Runs each backend on the same PBC geometry and reports pair counts plus symmetric
diffs against the reference oracle (Vesin when installed, else brute-force MIC).

Without ``--composition``, uses a synthetic two-dimer toy cluster (no CHARMM).
With ``--composition`` (md-system style, e.g. ``ACO:2`` or ``DCM:1,ACO:1``),
builds a CGENFF cluster in PyCHARMM so all backends including PyCHARMM can run.

Examples
--------
  uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py
  uv run python tests/functionality/neighbor_lists/05_compare_nl_backends.py \\
      --composition DCM:2 --backends vesin,jax_md,ase,pycharmm
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import (
    print_fail,
    print_header,
    print_pass,
    setup_charmm_composition_cluster,
    two_dimer_cluster,
)
from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import (
    create_jax_md_neighbor_list,
    have_jax_md,
)
from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend
from mmml.interfaces.pycharmmInterface.nl_reference import (
    compare_pair_sets,
    extract_valid_pairs,
    have_vesin,
    reference_mic_pairs,
    vesin_mic_pairs,
)


def _jax_md_pairs(
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
    return extract_valid_pairs(pair_i, pair_j, mask)


def _vesin_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    monomer_offsets: np.ndarray,
) -> set[tuple[int, int]]:
    return vesin_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=monomer_offsets,
    )


def _cell_list_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_offsets: np.ndarray,
) -> set[tuple[int, int]]:
    pi, pj, mask, _n_valid, _cap, _used = build_mm_pairs_with_backend(
        "cell_list",
        positions,
        cell,
        cutoff=cutoff,
        monomer_offsets=monomer_offsets,
        total_atoms=positions.shape[0],
    )
    return extract_valid_pairs(pi, pj, mask)


def _ase_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    *,
    atomic_numbers: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    from ase import Atoms
    from ase.neighborlist import NeighborList

    n = int(positions.shape[0])
    numbers = (
        np.asarray(atomic_numbers, dtype=int)
        if atomic_numbers is not None
        else np.ones(n, dtype=int)
    )
    atoms = Atoms(
        numbers=numbers,
        positions=np.asarray(positions, dtype=np.float64),
        cell=cell,
        pbc=True,
    )
    cutoffs = [float(cutoff) / 2.0] * n
    nl = NeighborList(
        cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)

    mid = np.asarray(monomer_id, dtype=np.int32)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        neighbors, _offsets = nl.get_neighbors(i)
        for j in neighbors:
            jj = int(j)
            if i >= jj or int(mid[i]) == int(mid[jj]):
                continue
            pairs.add((i, jj))
    return pairs


def _pycharmm_pairs(
    cutoff: float,
    monomer_id: np.ndarray,
) -> set[tuple[int, int]]:
    import pycharmm.nbonds as nbonds

    from mmml.interfaces.pycharmmInterface.import_pycharmm import capture_neighbour_list

    nbonds.update_bnbnd()
    nl_info = capture_neighbour_list()
    mid = np.asarray(monomer_id, dtype=np.int32)
    cutoff_f = float(cutoff)
    pairs: set[tuple[int, int]] = set()
    for (a, b), dist in nl_info["pair_distance_dict"].items():
        if float(dist) >= cutoff_f:
            continue
        i, j = int(a), int(b)
        if int(mid[i]) == int(mid[j]):
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return pairs


def _setup_composition_geometry(
    composition: str,
    *,
    cutoff: float,
    box_side: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    positions, cell, offsets, monomer_id, atomic_numbers = setup_charmm_composition_cluster(
        composition,
        box_side=box_side,
        spacing=spacing,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    box_a = float(cell[0, 0])
    prepare_charmm_pbc(box_a)
    cuts = apply_pbc_nbonds(nbxmod=5, cutnb=float(cutoff), cubic_box_side_A=box_a)
    effective_cutoff = float(cuts.cutnb)
    return positions, cell, offsets, monomer_id, atomic_numbers, effective_cutoff


def _parse_backends(raw: str) -> list[str]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    allowed = {"vesin", "jax_md", "ase", "cell_list", "pycharmm"}
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")
    return names


def _collect_backend(
    name: str,
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    monomer_offsets: np.ndarray,
    charmm_geometry: bool,
    atomic_numbers: np.ndarray | None,
) -> tuple[str, set[tuple[int, int]] | None, str | None]:
    try:
        if name == "vesin":
            if not have_vesin():
                return name, None, "vesin not installed (uv sync --extra nl-validation)"
            return name, _vesin_pairs(
                positions, cell, cutoff, monomer_id, monomer_offsets
            ), None
        if name == "jax_md":
            if not have_jax_md():
                return name, None, "jax-md unavailable"
            return name, _jax_md_pairs(positions, cell, cutoff, monomer_offsets), None
        if name == "ase":
            return (
                name,
                _ase_pairs(
                    positions,
                    cell,
                    cutoff,
                    monomer_id,
                    atomic_numbers=atomic_numbers,
                ),
                None,
            )
        if name == "cell_list":
            return name, _cell_list_pairs(positions, cell, cutoff, monomer_offsets), None
        if name == "pycharmm":
            if not charmm_geometry:
                return (
                    name,
                    None,
                    "requires --composition (PyCHARMM PSF + PBC nbonds)",
                )
            from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

            if CGENFF_PRM is None:
                return name, None, "PyCHARMM/CGENFF not available"
            return name, _pycharmm_pairs(cutoff, monomer_id), None
        raise ValueError(f"unsupported backend {name!r}")
    except Exception as exc:
        return name, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=13.0, help="MIC pair cutoff (Å)")
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help=(
            "CGENFF cluster composition (md-system style): comma-separated RES:N, "
            "e.g. ACO:2, DCM:2, or ACO:1,DCM:1. Builds PyCHARMM PSF + PBC for all backends."
        ),
    )
    parser.add_argument(
        "--charmm-geometry",
        action="store_true",
        help="Deprecated alias for --composition ACO:2",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=8.0,
        help="Minimum monomer COM spacing on placement grid (Å); used with --composition",
    )
    parser.add_argument("--box-side", type=float, default=40.0)
    parser.add_argument(
        "--com-separation",
        type=float,
        default=None,
        help="Deprecated alias for --spacing (synthetic geometry only)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list,pycharmm",
        help="Comma-separated backends to compare",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also print symmetric diffs between every collected backend pair",
    )
    args = parser.parse_args()

    print_header("Neighbor-list backend comparison")
    backends = _parse_backends(args.backends)
    cutoff = float(args.cutoff)
    composition = args.composition
    if composition is None and args.charmm_geometry:
        composition = "ACO:2"

    atomic_numbers: np.ndarray | None = None
    if composition:
        try:
            positions, cell, offsets, monomer_id, atomic_numbers, cutoff = (
                _setup_composition_geometry(
                    composition,
                    cutoff=cutoff,
                    box_side=float(args.box_side),
                    spacing=float(args.spacing),
                )
            )
        except Exception as exc:
            print_fail(f"composition cluster setup: {exc}")
            return 1
        geom_label = f"charmm:{composition}"
        charmm_geometry = True
    else:
        com_sep = float(args.com_separation if args.com_separation is not None else args.spacing)
        positions, cell, offsets, monomer_id = two_dimer_cluster(
            box_side=float(args.box_side),
            com_separation=com_sep,
        )
        geom_label = "synthetic_two_dimer"
        charmm_geometry = False

    ref, ref_src = reference_mic_pairs(
        positions,
        cell,
        cutoff,
        monomer_id,
        monomer_offsets=offsets,
    )
    print(
        f"geometry={geom_label}  n_atoms={positions.shape[0]}  "
        f"cutoff={cutoff:.3f} Å  reference={ref_src} ({len(ref)} pairs)"
    )

    collected: dict[str, set[tuple[int, int]]] = {}
    skipped: dict[str, str] = {}
    for name in backends:
        label, pairs, skip_reason = _collect_backend(
            name,
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            monomer_id=monomer_id,
            monomer_offsets=offsets,
            charmm_geometry=charmm_geometry,
            atomic_numbers=atomic_numbers,
        )
        if pairs is None:
            skipped[label] = skip_reason or "skipped"
            print(f"SKIP {label}: {skipped[label]}")
            continue
        collected[label] = pairs

    if not collected:
        print_fail("no backends produced pair sets")
        return 1

    ok = True
    print(f"\n{'backend':<12} {'pairs':>6} {'only_ref':>9} {'only_be':>9}  status")
    print("-" * 52)
    for name, pairs in collected.items():
        cmp = compare_pair_sets(ref, pairs)
        status = "PASS" if cmp.match else "FAIL"
        if not cmp.match:
            ok = False
        print(
            f"{name:<12} {cmp.n_b:>6} {len(cmp.only_a):>9} {len(cmp.only_b):>9}  {status}"
        )
        if not cmp.match:
            print(cmp.summary(label_a="reference", label_b=name, max_show=6))

    if args.pairwise and len(collected) >= 2:
        print("\nPairwise backend diffs:")
        names = list(collected)
        for i, a_name in enumerate(names):
            for b_name in names[i + 1 :]:
                cmp = compare_pair_sets(collected[a_name], collected[b_name])
                status = "match" if cmp.match else "DIFF"
                print(
                    f"  {a_name} vs {b_name}: {status} "
                    f"(only-{a_name}={len(cmp.only_a)}, only-{b_name}={len(cmp.only_b)})"
                )
                if not cmp.match:
                    ok = False
                    print(cmp.summary(label_a=a_name, label_b=b_name, max_show=4))

    if ok:
        print_pass("all collected backends match reference")
        return 0

    print_fail("one or more backends differ from reference")
    return 1


if __name__ == "__main__":
    sys.exit(main())
