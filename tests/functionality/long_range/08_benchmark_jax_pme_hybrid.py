#!/usr/bin/env python3
"""Benchmark hybrid jax-pme long-range correction wall time.

This script is intentionally a force-eval microbenchmark, not an MD driver.  By
default it times the host-side hybrid correction directly so it can run without
PyCHARMM.  Use ``--wrapped-mm`` on a CHARMM-ready machine to include the
``build_mm_energy_forces_fn`` wrapper and its ``jax.pure_callback`` round trip.

Examples
--------
  uv run python tests/functionality/long_range/08_benchmark_jax_pme_hybrid.py
  uv run python tests/functionality/long_range/08_benchmark_jax_pme_hybrid.py \
      --n-monomers 25 --atoms-per-monomer 5 --methods ewald,pme,p3m
  uv run python tests/functionality/long_range/08_benchmark_jax_pme_hybrid.py \
      --methods ewald,pme,p3m --coulomb-only --profile
  uv run python tests/functionality/long_range/08_benchmark_jax_pme_hybrid.py \
      --wrapped-mm --repeat 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable

import numpy as np

from _common import have_jax_pme_package, print_header
from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
    hybrid_jax_pme_mm_lr_correction,
)


def _truthy_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _median_ms(fn: Callable[[], None], *, repeat: int, warmup: int) -> tuple[float, float]:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples)), float(statistics.mean(samples))


def _synthetic_cluster(
    *,
    n_monomers: int,
    atoms_per_monomer: int,
    box_side: float,
    spacing: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_atoms = int(n_monomers) * int(atoms_per_monomer)
    offsets = np.arange(0, n_atoms + 1, int(atoms_per_monomer), dtype=np.int32)
    monomer_id = np.repeat(np.arange(int(n_monomers), dtype=np.int32), int(atoms_per_monomer))
    positions = np.zeros((n_atoms, 3), dtype=np.float64)

    grid_n = int(np.ceil(n_monomers ** (1.0 / 3.0)))
    starts = np.linspace(
        0.5 * spacing,
        max(0.5 * spacing, box_side - 0.5 * spacing),
        grid_n,
        dtype=np.float64,
    )
    centers = np.array(np.meshgrid(starts, starts, starts, indexing="ij")).reshape(3, -1).T
    centers = centers[:n_monomers]
    atom_template = rng.normal(scale=0.25, size=(atoms_per_monomer, 3))
    atom_template -= atom_template.mean(axis=0, keepdims=True)
    for mi, center in enumerate(centers):
        sl = slice(int(offsets[mi]), int(offsets[mi + 1]))
        positions[sl] = center + atom_template

    charges = rng.normal(size=n_atoms)
    charges -= np.repeat(
        [charges[int(offsets[m]) : int(offsets[m + 1])].mean() for m in range(n_monomers)],
        atoms_per_monomer,
    )
    c6_sqrt = np.abs(rng.normal(loc=0.20, scale=0.03, size=n_atoms))
    return positions, charges.astype(np.float64), c6_sqrt.astype(np.float64), offsets, monomer_id


def _direct_hybrid_fn(
    *,
    positions: np.ndarray,
    charges: np.ndarray,
    c6_sqrt: np.ndarray,
    offsets: np.ndarray,
    monomer_id: np.ndarray,
    box_side: float,
    method: str,
    sr_cutoff: float,
    include_dispersion: bool,
) -> Callable[[], None]:
    cell = np.eye(3, dtype=np.float64) * float(box_side)

    def eval_once() -> None:
        hybrid_jax_pme_mm_lr_correction(
            positions,
            charges,
            offsets,
            box_length_A=float(box_side),
            method=method,
            sr_cutoff_A=float(sr_cutoff),
            c6_sqrt=c6_sqrt if include_dispersion else None,
            monomer_id=monomer_id,
            lambda_monomer=np.ones(len(offsets) - 1, dtype=np.float64),
            pbc_cell=cell,
            ml_switch_width=1.0,
            mm_switch_on=6.0,
            mm_switch_width=4.0,
            include_dispersion=include_dispersion,
        )

    return eval_once


def _wrapped_mm_fn(
    *,
    positions: np.ndarray,
    offsets: np.ndarray,
    box_side: float,
    method: str,
    sr_cutoff: float,
    include_dispersion: bool,
) -> Callable[[], None]:
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

    n_monomers = len(offsets) - 1
    atoms_per = int(offsets[1] - offsets[0])
    result = build_mm_energy_forces_fn(
        positions,
        total_atoms=positions.shape[0],
        n_monomers=n_monomers,
        monomer_offsets=offsets,
        atoms_per_monomer_list=[atoms_per] * n_monomers,
        lambda_monomer=np.ones(n_monomers, dtype=np.float64),
        ml_switch_width=1.0,
        mm_switch_on=6.0,
        mm_switch_width=4.0,
        pbc_cell=float(box_side),
        defer_xla_gpu_warmup=True,
        mm_nl_backend="cell_list",
        use_jax_md_neighbor_list=False,
        lr_solver="jax_pme",
        jax_pme_method=method,
        jax_pme_sr_cutoff_A=float(sr_cutoff),
        jax_pme_dispersion=include_dispersion,
    )
    if isinstance(result, tuple):
        mm_fn, update_fn = result
        pair_idx, pair_mask = update_fn(positions, box=np.eye(3) * float(box_side))

        def eval_once() -> None:
            mm_fn(positions, pair_idx, pair_mask)

        return eval_once

    def eval_once() -> None:
        result(positions)

    return eval_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-monomers", type=int, default=8)
    parser.add_argument("--atoms-per-monomer", type=int, default=5)
    parser.add_argument("--box-side", type=float, default=28.0)
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument(
        "--methods",
        default="ewald,pme,p3m",
        help="comma-separated jax-pme methods; mesh methods may abort on unsupported hosts",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sr-cutoff", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--coulomb-only",
        action="store_true",
        help="skip r^-6 LJ-PME dispersion to measure Coulomb-only long range",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="set MMML_JAX_PME_PROFILE=per_call for component-level timings",
    )
    parser.add_argument(
        "--wrapped-mm",
        action="store_true",
        help="include build_mm_energy_forces_fn and pure_callback wrapper (requires PyCHARMM)",
    )
    args = parser.parse_args()

    if not have_jax_pme_package():
        print("SKIP: jax-pme is not installed")
        return 0
    if args.profile:
        os.environ["MMML_JAX_PME_PROFILE"] = "per_call"

    positions, charges, c6_sqrt, offsets, monomer_id = _synthetic_cluster(
        n_monomers=args.n_monomers,
        atoms_per_monomer=args.atoms_per_monomer,
        box_side=args.box_side,
        spacing=args.spacing,
        seed=args.seed,
    )
    mode = "wrapped-mm" if args.wrapped_mm else "direct-hybrid"
    long_range_mode = "coulomb-only" if args.coulomb_only else "coulomb+r^-6"
    print_header(f"jax-pme hybrid benchmark ({mode})")
    print(
        f"n_monomers={args.n_monomers} atoms_per={args.atoms_per_monomer} "
        f"n_atoms={positions.shape[0]} box={args.box_side:.1f} Å repeat={args.repeat} "
        f"long_range={long_range_mode}"
    )
    print("method        median_ms/eval   mean_ms/eval   note")

    for method in _truthy_csv(args.methods):
        try:
            if args.wrapped_mm:
                fn = _wrapped_mm_fn(
                    positions=positions,
                    offsets=offsets,
                    box_side=args.box_side,
                    method=method,
                    sr_cutoff=args.sr_cutoff,
                    include_dispersion=not args.coulomb_only,
                )
            else:
                fn = _direct_hybrid_fn(
                    positions=positions,
                    charges=charges,
                    c6_sqrt=c6_sqrt,
                    offsets=offsets,
                    monomer_id=monomer_id,
                    box_side=args.box_side,
                    method=method,
                    sr_cutoff=args.sr_cutoff,
                    include_dispersion=not args.coulomb_only,
                )
            median_ms, mean_ms = _median_ms(fn, repeat=args.repeat, warmup=args.warmup)
            print(f"{method:10s} {median_ms:15.2f} {mean_ms:14.2f}   ok")
        except Exception as exc:
            print(f"{method:10s} {'SKIP':>15s} {'SKIP':>14s}   {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
