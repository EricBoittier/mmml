#!/usr/bin/env python3
"""Benchmark neighbor-list backend wall time (cold build + optional jax-md update).

Reports median milliseconds per call for each backend on the same PBC geometry.
Use liquid-density cases for realistic pair counts; synthetic two-dimer for smoke.

Examples
--------
  uv run python tests/functionality/neighbor_lists/08_benchmark_nl_backends.py
  uv run python tests/functionality/neighbor_lists/08_benchmark_nl_backends.py \\
      --case synthetic_aco_liquid_n16 --repeat 30 --backends vesin,jax_md,cell_list
  uv run python tests/functionality/neighbor_lists/08_benchmark_nl_backends.py \\
      --case charmm_aco_liquid_n16 --with-charmm --repeat 10
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from _common import (
    build_liquid_density_synthetic_case,
    charmm_liquid_density_cases,
    have_charmm_nl,
    liquid_density_synthetic_cases,
    print_header,
    setup_charmm_liquid_density_cluster,
    two_dimer_cluster,
)
from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import (
    create_jax_md_neighbor_list,
    have_jax_md,
)
from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend
from mmml.interfaces.pycharmmInterface.nl_reference import (
    extract_valid_pairs,
    filter_pairs_under_cutoff,
    have_vesin,
    vesin_mic_pairs,
)

_DIR = Path(__file__).resolve().parent


def _load_compare_module():
    path = _DIR / "05_compare_nl_backends.py"
    spec = importlib.util.spec_from_file_location("nl_compare_backends", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _median_ms(fn: Callable[[], None], *, repeat: int, warmup: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def _resolve_geometry(
    case_name: str,
    *,
    with_charmm: bool,
) -> tuple[
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray | None,
    bool,
    float,
    float,
]:
    synthetic_all = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    charmm_all = {str(c["name"]): c for c in charmm_liquid_density_cases()}

    if case_name == "two_dimer":
        positions, cell, offsets, monomer_id = two_dimer_cluster()
        return (
            "two_dimer smoke cluster",
            positions,
            cell,
            offsets,
            monomer_id,
            13.0,
            None,
            False,
            float(cell[0, 0]),
            0.0,
        )

    if case_name in synthetic_all:
        positions, cell, offsets, monomer_id, cutoff, desc, side, rho = (
            build_liquid_density_synthetic_case(synthetic_all[case_name])
        )
        return desc, positions, cell, offsets, monomer_id, cutoff, None, False, side, rho

    if case_name in charmm_all:
        if not with_charmm:
            raise ValueError(f"{case_name} requires --with-charmm")
        if not have_charmm_nl():
            raise RuntimeError("PyCHARMM/CGENFF not available")
        case = charmm_all[case_name]
        (
            positions,
            cell,
            offsets,
            monomer_id,
            atomic_numbers,
            eff_cutoff,
            side,
            rho,
        ) = setup_charmm_liquid_density_cluster(case, cutoff=float(case["cutoff"]))
        return (
            str(case["description"]),
            positions,
            cell,
            offsets,
            monomer_id,
            eff_cutoff,
            atomic_numbers,
            True,
            side,
            rho,
        )

    raise ValueError(f"unknown case {case_name!r}")


def _benchmark_backend(
    name: str,
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_id: np.ndarray,
    monomer_offsets: np.ndarray,
    charmm_geometry: bool,
    atomic_numbers: np.ndarray | None,
    repeat: int,
    warmup: int,
) -> dict[str, float | str | None]:
    nl05 = _load_compare_module()
    label, pairs, skip = nl05._collect_backend(
        name,
        positions=positions,
        cell=cell,
        cutoff=cutoff,
        monomer_id=monomer_id,
        monomer_offsets=monomer_offsets,
        charmm_geometry=charmm_geometry,
        atomic_numbers=atomic_numbers,
    )
    if pairs is None:
        return {"backend": label, "build_ms": None, "update_ms": None, "n_pairs": None, "note": skip}

    n_pairs = len(pairs)

    if name == "vesin":
        if not have_vesin():
            return {"backend": label, "build_ms": None, "update_ms": None, "n_pairs": None, "note": "no vesin"}

        def build() -> None:
            vesin_mic_pairs(
                positions,
                cell,
                cutoff,
                monomer_id,
                monomer_offsets=monomer_offsets,
            )

        return {
            "backend": label,
            "build_ms": _median_ms(build, repeat=repeat, warmup=warmup),
            "update_ms": None,
            "n_pairs": n_pairs,
            "note": "rebuild each call",
        }

    if name == "cell_list":

        def build() -> None:
            build_mm_pairs_with_backend(
                "cell_list",
                positions,
                cell,
                cutoff=cutoff,
                monomer_offsets=monomer_offsets,
                total_atoms=positions.shape[0],
            )

        return {
            "backend": label,
            "build_ms": _median_ms(build, repeat=repeat, warmup=warmup),
            "update_ms": None,
            "n_pairs": n_pairs,
            "note": "rebuild each call",
        }

    if name == "jax_md":
        if not have_jax_md():
            return {"backend": label, "build_ms": None, "update_ms": None, "n_pairs": None, "note": "no jax-md"}

        bundle = create_jax_md_neighbor_list(
            cell,
            r_cutoff=cutoff,
            monomer_offsets=monomer_offsets,
            dr_threshold=0.5,
            capacity_multiplier=1.5,
            fractional_coordinates=False,
        )
        if bundle is None:
            return {"backend": label, "build_ms": None, "update_ms": None, "n_pairs": None, "note": "jax-md unavailable"}
        neighbor_fn, filter_fn, _ = bundle
        pos = np.asarray(positions, dtype=np.float64)
        pos_update = pos + 0.02 * np.random.default_rng(0).standard_normal(pos.shape)
        state = {"nbrs": neighbor_fn.allocate(pos)}

        def build() -> None:
            nbrs = neighbor_fn.allocate(pos)
            pi, pj, mask = filter_fn(nbrs.idx)
            pairs = extract_valid_pairs(pi, pj, mask)
            filter_pairs_under_cutoff(pairs, positions, cell, cutoff)

        def update() -> None:
            state["nbrs"] = neighbor_fn.update(pos_update, state["nbrs"])
            pi, pj, mask = filter_fn(state["nbrs"].idx)
            pairs = extract_valid_pairs(pi, pj, mask)
            filter_pairs_under_cutoff(pairs, positions, cell, cutoff)

        return {
            "backend": label,
            "build_ms": _median_ms(build, repeat=repeat, warmup=warmup),
            "update_ms": _median_ms(update, repeat=repeat, warmup=warmup),
            "n_pairs": n_pairs,
            "note": "allocate vs update (dr=0.02 Å)",
        }

    if name == "ase":
        from ase import Atoms
        from ase.neighborlist import NeighborList

        n = int(positions.shape[0])
        numbers = (
            np.asarray(atomic_numbers, dtype=int)
            if atomic_numbers is not None
            else np.ones(n, dtype=int)
        )
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
        cutoffs = [float(cutoff) / 2.0] * n

        def build() -> None:
            nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
            nl.update(atoms)

        return {
            "backend": label,
            "build_ms": _median_ms(build, repeat=repeat, warmup=warmup),
            "update_ms": None,
            "n_pairs": n_pairs,
            "note": "NeighborList rebuild each call",
        }

    if name == "pycharmm":
        if not charmm_geometry:
            return {
                "backend": label,
                "build_ms": None,
                "update_ms": None,
                "n_pairs": None,
                "note": "requires CHARMM liquid-density case",
            }

        def build() -> None:
            nl05._pycharmm_pairs(cutoff, monomer_id)

        return {
            "backend": label,
            "build_ms": _median_ms(build, repeat=repeat, warmup=warmup),
            "update_ms": None,
            "n_pairs": n_pairs,
            "note": "nbonds.update_bnbnd + capture",
        }

    return {"backend": name, "build_ms": None, "update_ms": None, "n_pairs": None, "note": "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=str,
        default="synthetic_aco_liquid_n16",
        help="Case name (liquid-density, charmm_*, or two_dimer)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list,pycharmm",
    )
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--with-charmm", action="store_true")
    args = parser.parse_args()

    print_header("Neighbor-list backend benchmark")
    nl05 = _load_compare_module()
    backends = nl05._parse_backends(args.backends)
    with_charmm = bool(args.with_charmm or "pycharmm" in backends)

    try:
        desc, positions, cell, offsets, monomer_id, cutoff, z, charmm_geom, side, rho = (
            _resolve_geometry(args.case, with_charmm=with_charmm)
        )
    except Exception as exc:
        print(f"FAIL: geometry setup: {exc}", file=sys.stderr)
        return 1

    rho_label = f"ρ={rho:.3f} g/cm³" if rho > 0 else "ρ=n/a"
    print(
        f"case={args.case}\n"
        f"  {desc}\n"
        f"  n_atoms={positions.shape[0]}  L={side:.2f} Å  {rho_label}  "
        f"cutoff={cutoff:.2f} Å  repeat={args.repeat} warmup={args.warmup}"
    )

    rows: list[dict[str, float | str | None]] = []
    for backend in backends:
        rows.append(
            _benchmark_backend(
                backend,
                positions=positions,
                cell=cell,
                cutoff=cutoff,
                monomer_id=monomer_id,
                monomer_offsets=offsets,
                charmm_geometry=charmm_geom,
                atomic_numbers=z,
                repeat=int(args.repeat),
                warmup=int(args.warmup),
            )
        )

    ranked = sorted(
        [r for r in rows if r["build_ms"] is not None],
        key=lambda r: float(r["build_ms"]),
    )

    print(f"\n{'backend':<12} {'build_ms':>10} {'update_ms':>10} {'pairs':>7}  note")
    print("-" * 72)
    for row in rows:
        build_s = f"{float(row['build_ms']):8.3f}" if row["build_ms"] is not None else "     n/a"
        upd_s = (
            f"{float(row['update_ms']):8.3f}"
            if row["update_ms"] is not None
            else "     n/a"
        )
        pairs_s = f"{int(row['n_pairs']):7d}" if row["n_pairs"] is not None else "    n/a"
        note = str(row.get("note") or "")
        print(f"{row['backend']:<12} {build_s:>10} {upd_s:>10} {pairs_s}  {note}")

    if ranked:
        fastest = ranked[0]
        print(
            f"\nFastest cold build: {fastest['backend']} "
            f"({float(fastest['build_ms']):.3f} ms median)"
        )
        if len(ranked) > 1:
            slowest = ranked[-1]
            ratio = float(slowest["build_ms"]) / float(fastest["build_ms"])
            print(
                f"Slowest: {slowest['backend']} ({float(slowest['build_ms']):.3f} ms, "
                f"{ratio:.2f}× slower)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
