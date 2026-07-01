#!/usr/bin/env python3
"""Validate cross-monomer jax-pme vs legacy and profile both modes together.

    JAX_PLATFORMS=cpu uv run python tests/functionality/long_range/09_jax_pme_cross_validate.py
    MMML_JAX_PME_PROFILE=1 JAX_PLATFORMS=cpu uv run python tests/functionality/long_range/09_jax_pme_cross_validate.py --reps 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from _common import have_jax_pme_package, print_header, print_pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-monomers", type=int, default=18)
    parser.add_argument("--atoms-per-monomer", type=int, default=3)
    parser.add_argument("--box-side", type=float, default=28.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    return parser.parse_args()


def _time_hybrid(fn, *, warmup: int, reps: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    if reps <= 0:
        fn()
        return 0.0
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main() -> int:
    args = _parse_args()
    print_header("jax-pme cross-monomer validate + profile")

    if not have_jax_pme_package():
        print("jax-pme not installed", file=sys.stderr)
        return 1

    from jaxpme import prefactors as jpref

    from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
        compute_jax_pme_cross_monomer_power_law,
        consume_cross_monomer_profile,
    )
    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        _intra_monomer_jax_pme_power_law,
        hybrid_jax_pme_mm_lr_correction,
    )
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_jax_pme_power_law,
        per_atom_jax_pme_c6_sqrt_for_atoms,
    )

    rng = np.random.default_rng(7)
    n_mono = int(args.n_monomers)
    size = int(args.atoms_per_monomer)
    n = n_mono * size
    pos = rng.random((n, 3)) * float(args.box_side) * 0.7
    chg = rng.normal(0.0, 0.1, n)
    offsets = np.arange(0, n + 1, size, dtype=np.int64)
    box_L = float(args.box_side)
    cell = np.diag([box_L, box_L, box_L])
    c6 = per_atom_jax_pme_c6_sqrt_for_atoms(
        np.full(n, 0.05, dtype=np.float64),
        np.full(n, 2.0, dtype=np.float64),
    )
    sr = 6.0

    full = compute_jax_pme_power_law(
        pos, chg, box_length_A=box_L, method="ewald", sr_cutoff_A=sr,
        exponent=1, prefactor=float(jpref.kcalmol_A),
    )
    intra = _intra_monomer_jax_pme_power_law(
        pos, chg, offsets, box_length_A=box_L, method="ewald", sr_cutoff_A=sr,
        exponent=1, prefactor=float(jpref.kcalmol_A),
    )
    cross = compute_jax_pme_cross_monomer_power_law(
        pos, chg, offsets, box_length_A=box_L, method="ewald", sr_cutoff_A=sr,
        exponent=1, prefactor=float(jpref.kcalmol_A),
    )
    ref_e = full.energy_kcalmol - intra.energy_kcalmol
    ref_f = full.forces_kcalmol_A - intra.forces_kcalmol_A
    e_ok = abs(cross.energy_kcalmol - ref_e) < 1e-5
    f_ok = float(np.max(np.abs(cross.forces_kcalmol_A - ref_f))) < 1e-5
    print(f"  validation: energy_ok={e_ok}  max_force_err={np.max(np.abs(cross.forces_kcalmol_A-ref_f)):.3e}")
    if not (e_ok and f_ok):
        return 1

    def _run_hybrid(mode: str) -> float:
        os.environ["MMML_JAX_PME_INTRA_MODE"] = mode
        return _time_hybrid(
            lambda: hybrid_jax_pme_mm_lr_correction(
                pos,
                chg,
                offsets,
                box_length_A=box_L,
                method="ewald",
                sr_cutoff_A=sr,
                c6_sqrt=c6,
                pbc_cell=cell,
                mm_switch_on=6.0,
                mm_switch_width=4.0,
            ),
            warmup=int(args.warmup),
            reps=int(args.reps),
        )

    legacy_ms = _run_hybrid("full_minus_intra")
    cross_ms = _run_hybrid("cross")
    speedup = legacy_ms / cross_ms if cross_ms > 0 else float("inf")
    print(f"  hybrid_mm_lr steady: legacy={legacy_ms:.1f} ms  cross={cross_ms:.1f} ms  speedup={speedup:.2f}x")

    prof = consume_cross_monomer_profile()
    if prof:
        print("  cross kernel profile:")
        for label, stats in prof.items():
            print(f"    {label}: mean={stats['mean_ms']:.2f} ms  n={int(stats['n'])}")

    print_pass("cross-monomer validation + profile complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
