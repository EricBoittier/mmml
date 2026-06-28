#!/usr/bin/env python3
"""Micro-benchmark hybrid jax-pme cost (full-box, intra, wrapped MM).

Run from repo root (PyCHARMM + CGENFF required for cluster setup):

    uv run python tests/functionality/long_range/08_jax_pme_hybrid_timing.py
    uv run python tests/functionality/long_range/08_jax_pme_hybrid_timing.py \\
        --composition DCM:25 --box-side 28 --method ewald --reps 20

The first jax-pme call often includes XLA compile; this script reports both
``first_call_ms`` (one eval after warmup) and ``steady_ms`` (mean over ``--reps``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from _common import have_jax_pme_package, print_fail, print_header, print_pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", default="DCM:25", help="RES:COUNT cluster")
    parser.add_argument("--box-side", type=float, default=28.0, help="Cubic box side (Å)")
    parser.add_argument("--spacing", type=float, default=4.0, help="Monomer placement spacing (Å)")
    parser.add_argument(
        "--method",
        choices=("ewald", "pme", "p3m"),
        default="ewald",
        help="jax-pme method",
    )
    parser.add_argument(
        "--sr-cutoff",
        type=float,
        default=6.0,
        help="jax-pme short-range cutoff (Å)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup repetitions per benchmark block",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=20,
        help="Timed repetitions for steady-state mean",
    )
    parser.add_argument(
        "--skip-wrapped-mm",
        action="store_true",
        help="Skip build_mm_energy_forces_fn MIC vs jax_pme comparison",
    )
    return parser.parse_args()


def _time_block(
    label: str,
    fn: Callable[[], Any],
    *,
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    """Return (first_call_ms, steady_mean_ms) after ``warmup`` untimed runs."""
    for _ in range(max(0, warmup)):
        fn()
    t0 = time.perf_counter()
    fn()
    first_ms = (time.perf_counter() - t0) * 1000.0
    if reps <= 0:
        return first_ms, first_ms
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    steady_ms = (time.perf_counter() - t0) * 1000.0 / reps
    return first_ms, steady_ms


def _print_row(label: str, first_ms: float, steady_ms: float) -> None:
    print(f"  {label:<32s}  first={first_ms:8.2f} ms   steady={steady_ms:8.2f} ms")


def main() -> int:
    args = _parse_args()
    print_header("Hybrid jax-pme timing breakdown")

    if not have_jax_pme_package():
        print_fail("jax-pme not installed")
        return 1

    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, psf
    except Exception:
        CGENFF_PRM = None
        psf = None  # type: ignore[assignment]
    if CGENFF_PRM is None or psf is None:
        print_fail("PyCHARMM/CGENFF not available")
        return 1

    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        hybrid_jax_pme_coulomb_correction,
        hybrid_jax_pme_mm_lr_correction,
        intra_monomer_jax_pme_coulomb,
    )
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_jax_pme_coulomb,
        jax_pme_host_device_name,
        per_atom_jax_pme_c6_sqrt_for_atoms,
    )
    from tests.functionality.neighbor_lists._common import setup_charmm_composition_cluster

    positions, cell, offsets, monomer_id, atomic_numbers = setup_charmm_composition_cluster(
        args.composition,
        box_side=float(args.box_side),
        spacing=float(args.spacing),
    )
    n_atoms = int(positions.shape[0])
    n_monomers = int(len(offsets) - 1)
    box_L = float(np.diag(cell)[0])
    charges = np.asarray(psf.get_charges(), dtype=np.float64)[:n_atoms]
    pbc_cell = np.diag([box_L, box_L, box_L])
    method = str(args.method)
    sr = float(args.sr_cutoff)

    # Dispersion timing uses uniform dummy LJ params (cost is weakly sensitive to c6).
    c6_sqrt = per_atom_jax_pme_c6_sqrt_for_atoms(
        np.full(n_atoms, 0.05, dtype=np.float64),
        np.full(n_atoms, 2.0, dtype=np.float64),
    )

    print(
        f"  system: {args.composition}  atoms={n_atoms}  monomers={n_monomers}  "
        f"L={box_L:.1f} Å  method={method}  sr_cutoff={sr:.1f} Å  "
        f"jax_pme_device={jax_pme_host_device_name()}  "
        f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '(default)')}"
    )
    print(f"  per hybrid callback ≈ {2 * (n_monomers + 1)} jax-pme solves (Coulomb + dispersion)")
    print()

    blocks: list[tuple[str, Callable[[], Any]]] = [
        (
            "full-box coulomb",
            lambda: compute_jax_pme_coulomb(
                positions,
                charges,
                box_length_A=box_L,
                method=method,
                sr_cutoff_A=sr,
            ),
        ),
        (
            "intra-monomer coulomb",
            lambda: intra_monomer_jax_pme_coulomb(
                positions,
                charges,
                offsets,
                box_length_A=box_L,
                method=method,
                sr_cutoff_A=sr,
            ),
        ),
        (
            "hybrid coulomb (full−intra)",
            lambda: hybrid_jax_pme_coulomb_correction(
                positions,
                charges,
                offsets,
                box_length_A=box_L,
                method=method,
                sr_cutoff_A=sr,
                pbc_cell=pbc_cell,
                ml_switch_width=1.0,
                mm_switch_on=6.0,
                mm_switch_width=4.0,
            ),
        ),
        (
            "hybrid MM LR (Coulomb+disp)",
            lambda: hybrid_jax_pme_mm_lr_correction(
                positions,
                charges,
                offsets,
                box_length_A=box_L,
                method=method,
                sr_cutoff_A=sr,
                c6_sqrt=c6_sqrt,
                monomer_id=monomer_id,
                pbc_cell=pbc_cell,
                ml_switch_width=1.0,
                mm_switch_on=6.0,
                mm_switch_width=4.0,
            ),
        ),
    ]

    for label, fn in blocks:
        first_ms, steady_ms = _time_block(
            label,
            fn,
            warmup=int(args.warmup),
            reps=int(args.reps),
        )
        _print_row(label, first_ms, steady_ms)

    if not args.skip_wrapped_mm:
        from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn

        atoms_per = int(offsets[1] - offsets[0])
        common_kw = dict(
            total_atoms=n_atoms,
            n_monomers=n_monomers,
            monomer_offsets=offsets,
            atoms_per_monomer_list=[atoms_per] * n_monomers,
            lambda_monomer=np.ones(n_monomers, dtype=np.float64),
            ml_switch_width=1.0,
            mm_switch_on=6.0,
            mm_switch_width=4.0,
            pbc_cell=box_L,
            defer_xla_gpu_warmup=True,
            mm_nl_backend="cell_list",
            use_jax_md_neighbor_list=False,
        )

        def _wrap_fn(lr: str) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
            out = build_mm_energy_forces_fn(
                positions,
                lr_solver=lr,
                jax_pme_method=method,
                jax_pme_sr_cutoff_A=sr,
                **common_kw,
            )
            return out[0] if isinstance(out, tuple) else out

        mic_fn = _wrap_fn("mic")
        pme_fn = _wrap_fn("jax_pme")
        for label, fn in (
            ("wrapped MM (mic)", mic_fn),
            (f"wrapped MM (jax_pme/{method})", pme_fn),
        ):
            first_ms, steady_ms = _time_block(
                label,
                lambda f=fn: f(positions),
                warmup=int(args.warmup),
                reps=int(args.reps),
            )
            _print_row(label, first_ms, steady_ms)

    print_pass("timing breakdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
