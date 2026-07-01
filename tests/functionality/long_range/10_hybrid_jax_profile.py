#!/usr/bin/env python3
"""Profile hybrid jax-pme LR (cProfile + JAX trace + MMML timer hooks).

No PyCHARMM required. Use for steady-state hybrid correction and compile
breakdown before/after ``MMML_JAX_PME_INTRA_MODE`` changes.

Examples
--------
  # Wall-clock + jax-pme component labels (stderr)
  MMML_JAX_PME_PROFILE=1 MMML_JAX_COMPILE_TIMERS=1 JAX_PLATFORMS=cpu \\
    uv run python tests/functionality/long_range/10_hybrid_jax_profile.py

  # Python cProfile (writes summary + .prof next to --out-dir)
  JAX_PLATFORMS=cpu uv run python tests/functionality/long_range/10_hybrid_jax_profile.py \\
    --cprofile --out-dir /tmp/hybrid_jax_prof

  # JAX device trace for TensorBoard (steady-state reps only)
  JAX_PLATFORMS=cpu uv run python tests/functionality/long_range/10_hybrid_jax_profile.py \\
    --jax-trace /tmp/jax_trace_hybrid --reps 5

  tensorboard --logdir /tmp/jax_trace_hybrid
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

from _common import have_jax_pme_package, print_header, print_pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-monomers", type=int, default=18)
    parser.add_argument("--atoms-per-monomer", type=int, default=3)
    parser.add_argument("--box-side", type=float, default=28.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--intra-mode",
        choices=("cross", "full_minus_intra", "both"),
        default="both",
        help="MMML_JAX_PME_INTRA_MODE to profile (default: compare both)",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Run steady-state block under cProfile; write .prof + top summary",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/hybrid_jax_profile"),
        help="Output directory for cProfile / metadata",
    )
    parser.add_argument(
        "--jax-trace",
        type=Path,
        default=None,
        help="If set, jax.profiler.start_trace() during steady-state reps",
    )
    parser.add_argument(
        "--enable-jax-pme-profile",
        action="store_true",
        help="Set MMML_JAX_PME_PROFILE=1 for component stderr labels",
    )
    parser.add_argument(
        "--enable-compile-timers",
        action="store_true",
        help="Set MMML_JAX_COMPILE_TIMERS=1 and print summary at end",
    )
    return parser.parse_args()


def _synthetic_system(args: argparse.Namespace):
    from jaxpme import prefactors as jpref

    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        hybrid_jax_pme_mm_lr_correction,
    )
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        per_atom_jax_pme_c6_sqrt_for_atoms,
    )

    rng = np.random.default_rng(11)
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

    def hybrid_eval():
        return hybrid_jax_pme_mm_lr_correction(
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
        )

    ctx = {
        "n_atoms": n,
        "n_monomers": n_mono,
        "box_L": box_L,
        "prefactor": float(jpref.kcalmol_A),
    }
    return hybrid_eval, ctx


def _time_steady(fn, *, warmup: int, reps: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    if reps <= 0:
        fn()
        return 0.0
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def _run_steady_with_optional_trace(fn, *, reps: int, trace_dir: Path | None) -> None:
    if trace_dir is None:
        for _ in range(reps):
            fn()
        return
    import jax

    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    try:
        for _ in range(reps):
            fn()
    finally:
        jax.profiler.stop_trace()


def _write_cprofile_summary(prof_path: Path, out_path: Path) -> None:
    stats = pstats.Stats(str(prof_path))
    stats.sort_stats("cumulative")
    with out_path.open("w", encoding="utf-8") as fh:
        stats.stream = fh
        stats.print_stats(50)


def _profile_mode(
    mode: str,
    hybrid_eval,
    args: argparse.Namespace,
    out_dir: Path,
) -> float:
    os.environ["MMML_JAX_PME_INTRA_MODE"] = mode
    from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
        consume_cross_monomer_profile,
    )
    from mmml.utils.jax_gpu_warmup import reset_jax_compile_timers

    if args.enable_compile_timers:
        reset_jax_compile_timers()

    for _ in range(max(0, args.warmup)):
        hybrid_eval()

    steady_ms = 0.0
    if args.cprofile:
        prof_path = out_dir / f"hybrid_{mode}.prof"
        prof = cProfile.Profile()
        prof.enable()
        t0 = time.perf_counter()
        _run_steady_with_optional_trace(
            hybrid_eval, reps=int(args.reps), trace_dir=args.jax_trace
        )
        steady_ms = (time.perf_counter() - t0) * 1000.0 / max(1, int(args.reps))
        prof.disable()
        prof.dump_stats(str(prof_path))
        _write_cprofile_summary(prof_path, out_dir / f"cprofile_top_{mode}.txt")
        print(f"  cProfile: {prof_path}  summary: {out_dir / f'cprofile_top_{mode}.txt'}")
    else:
        if args.jax_trace is not None:
            t0 = time.perf_counter()
            _run_steady_with_optional_trace(
                hybrid_eval, reps=int(args.reps), trace_dir=args.jax_trace
            )
            steady_ms = (time.perf_counter() - t0) * 1000.0 / max(1, int(args.reps))
            print(f"  jax trace dir: {args.jax_trace}")
        else:
            steady_ms = _time_steady(hybrid_eval, warmup=0, reps=int(args.reps))

    print(f"  {mode}: steady_mean={steady_ms:.2f} ms (reps={args.reps})")

    cross_prof = consume_cross_monomer_profile()
    if cross_prof:
        print(f"  cross kernel ({mode}):")
        for label, stats in cross_prof.items():
            print(f"    {label}: mean={stats['mean_ms']:.2f} ms  n={int(stats['n'])}")

    if args.enable_compile_timers:
        from mmml.utils.jax_gpu_warmup import maybe_log_jax_compile_timers

        maybe_log_jax_compile_timers()

    return steady_ms


def main() -> int:
    args = _parse_args()
    print_header("hybrid jax-pme profile (cProfile / JAX trace)")

    if not have_jax_pme_package():
        print("jax-pme not installed", file=sys.stderr)
        return 1

    if args.enable_jax_pme_profile:
        os.environ["MMML_JAX_PME_PROFILE"] = "1"
    if args.enable_compile_timers:
        os.environ["MMML_JAX_COMPILE_TIMERS"] = "1"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
        write_profile_git_metadata,
    )

    hybrid_eval, ctx = _synthetic_system(args)
    write_profile_git_metadata(
        args.out_dir,
        argv=sys.argv,
        extra={
            "profile_kind": "hybrid_jax_pme_lr",
            "n_monomers": ctx["n_monomers"],
            "n_atoms": ctx["n_atoms"],
            "box_side_A": ctx["box_L"],
            "intra_mode": args.intra_mode,
            "warmup": args.warmup,
            "reps": args.reps,
        },
    )

    modes = (
        ["cross", "full_minus_intra"]
        if args.intra_mode == "both"
        else [args.intra_mode]
    )
    timings: dict[str, float] = {}
    for mode in modes:
        print(f"\n--- intra_mode={mode} ---")
        timings[mode] = _profile_mode(mode, hybrid_eval, args, args.out_dir)

    if len(timings) == 2:
        legacy = timings["full_minus_intra"]
        cross = timings["cross"]
        if cross > 0:
            print(f"\n  speedup cross/legacy: {legacy / cross:.2f}x")

    print_pass(f"profile artifacts under {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
