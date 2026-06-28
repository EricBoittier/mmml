"""``mmml warmup-mlpot-jax`` — serial JAX JIT warmup for MLpot (outside ``mpirun``)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up MLpot PhysNet JAX compilation in serial Python (multithreaded XLA). "
            "Populates JAX_COMPILATION_CACHE_DIR for faster later runs under mpirun. "
            "Does not import PyCHARMM or call MPI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  export MMML_CKPT=/path/to/DESdimers_params.json
  mmml warmup-mlpot-jax --n-monomers 20 --ml-batch-size 128

  # Match spatial-mini production fingerprint (single-GPU PhysNet path):
  mmml warmup-mlpot-jax --checkpoint "$MMML_CKPT" --n-monomers 20 \\
    --box-side 32 --ml-batch-size 64 --ml-gpu-count 1 --do-mm

  # Then under MPI:
  MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 ./scripts/mmml-charmm-mpirun.sh md-system ...

Do **not** run under mpirun (compile threads are disabled there by design).
Clear stale launcher env if needed: unset OMPI_COMM_WORLD_SIZE PMI_SIZE PMIX_SIZE
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet checkpoint (default: MMML_CKPT or MMML_CHECKPOINT)",
    )
    parser.add_argument("--n-monomers", type=int, default=20, help="Monomer count (default 20)")
    parser.add_argument(
        "--atoms-per-monomer",
        type=int,
        default=10,
        help="Atoms per monomer for synthetic lattice (default 10)",
    )
    parser.add_argument("--box-side", type=float, default=32.0, help="PBC box side Å (0 = vacuum)")
    parser.add_argument("--spacing", type=float, default=5.0, help="Lattice spacing Å")
    parser.add_argument("--ml-batch-size", type=int, default=128, help="MLpot batch size")
    parser.add_argument("--ml-gpu-count", type=int, default=1, help="JAX pmap GPU count")
    parser.add_argument(
        "--do-mm",
        action="store_true",
        help="Include MM pair path in warmup (closer to production hybrid)",
    )
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=None,
        help="Override MMML_JAX_COMPILE_THREADS (default: min(16, ncpu) when unset)",
    )
    parser.add_argument(
        "--allow-under-mpirun",
        action="store_true",
        help="Allow running under mpirun (not recommended; compile threads usually off)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned warmup settings and exit",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _resolve_checkpoint(path: Path | None) -> Path:
    raw = path or os.environ.get("MMML_CKPT") or os.environ.get("MMML_CHECKPOINT")
    if not raw:
        raise SystemExit(
            "warmup-mlpot-jax: provide --checkpoint or set MMML_CKPT / MMML_CHECKPOINT"
        )
    candidate = Path(str(raw)).expanduser()
    if candidate.is_dir():
        for name in ("params.json", "DESdimers_params.json"):
            hit = candidate / name
            if hit.is_file():
                return hit.resolve()
        # Orbax experiment root (epoch-* subdirs) — same resolution as md-system.
        try:
            from mmml.cli.base import resolve_checkpoint_paths

            base, epoch = resolve_checkpoint_paths(candidate)
            return base.resolve()
        except SystemExit:
            pass
    if candidate.is_file():
        return candidate.resolve()
    raise SystemExit(f"warmup-mlpot-jax: checkpoint not found: {candidate}")


def _default_atomic_numbers(n_monomers: int, atoms_per: int) -> list[int]:
    pattern = [6, 1, 1, 1, 8, 1, 1, 1, 1, 1]
    if atoms_per != len(pattern):
        return [6] + [1] * (atoms_per - 1)
    return list(pattern) * int(n_monomers)


class _WarmupBuildArgs:
    def __init__(self, *, include_mm: bool) -> None:
        self.include_mm = include_mm


def run_warmup_mlpot_jax(args: argparse.Namespace) -> int:
    os.environ["MMML_WARMUP_MLPOT_JAX_ONLY"] = "1"
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        _under_mpirun,
        scrub_stale_openmpi_env,
    )
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        apply_jax_compile_xla_flags,
        jax_compile_threads_enabled,
        resolve_jax_compile_thread_count,
    )
    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
        apply_mlpot_jax_platform_env,
        mlpot_jax_compilation_cache_dir,
    )

    if _under_mpirun() and not args.allow_under_mpirun:
        print(
            "warmup-mlpot-jax: refuse to run under mpirun/PMI launcher env.\n"
            "  Run as serial: python -m mmml.cli.__main__ warmup-mlpot-jax ...\n"
            "  JAX compile threads are disabled under MPI (see jax_compile_threads.py).",
            file=sys.stderr,
        )
        return 2

    removed = scrub_stale_openmpi_env()
    if removed and not args.quiet:
        print(f"warmup-mlpot-jax: cleared {removed} stale OpenMPI/PMI env var(s)", flush=True)

    if args.compile_threads is not None:
        os.environ["MMML_JAX_COMPILE_THREADS"] = str(max(0, int(args.compile_threads)))
    elif not jax_compile_threads_enabled():
        os.environ.pop("MMML_NO_JAX_COMPILE_THREADS", None)

    ckpt = _resolve_checkpoint(args.checkpoint)
    n_monomers = int(args.n_monomers)
    atoms_per = int(args.atoms_per_monomer)
    box_side = float(args.box_side)
    cell = box_side if box_side > 0 else False

    cache_dir = mlpot_jax_compilation_cache_dir()
    compile_threads = resolve_jax_compile_thread_count()

    if args.dry_run:
        print("warmup-mlpot-jax dry-run:")
        print(f"  checkpoint: {ckpt}")
        print(f"  n_monomers: {n_monomers}  atoms_per_monomer: {atoms_per}")
        print(f"  box_side: {box_side}  do_mm: {bool(args.do_mm)}")
        print(f"  ml_batch_size: {args.ml_batch_size}  ml_gpu_count: {args.ml_gpu_count}")
        print(f"  compile_threads: {compile_threads}")
        print(f"  JAX_COMPILATION_CACHE_DIR: {cache_dir or '<disabled>'}")
        return 0

    apply_mlpot_jax_platform_env(quiet=args.quiet and not args.verbose)
    apply_jax_compile_xla_flags(quiet=args.quiet and not args.verbose)

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
        lattice_positions_cubic_pbc,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        build_decomposed_mlpot_model,
        warmup_decomposed_mlpot,
    )

    z = np.array(_default_atomic_numbers(n_monomers, atoms_per), dtype=int)
    per = [atoms_per] * n_monomers
    if cell:
        pos = lattice_positions_cubic_pbc(
            n_monomers, atoms_per, float(box_side), spacing_A=float(args.spacing), seed=11
        )
    else:
        rng = np.random.default_rng(11)
        pos = rng.standard_normal((len(z), 3)) * 2.0

    if not args.quiet:
        print(
            f"warmup-mlpot-jax: {n_monomers} monomers, {len(z)} atoms, "
            f"batch={args.ml_batch_size}, gpus={args.ml_gpu_count}, "
            f"compile_threads={compile_threads}",
            flush=True,
        )
        if cache_dir is not None:
            print(f"warmup-mlpot-jax: cache -> {cache_dir}", flush=True)

    t0 = time.perf_counter()
    build_args = _WarmupBuildArgs(include_mm=bool(args.do_mm))
    model = build_decomposed_mlpot_model(
        ckpt,
        z,
        per,
        n_monomers,
        ml_batch_size=int(args.ml_batch_size),
        ml_gpu_count=int(args.ml_gpu_count),
        cell=cell,
        verbose=args.verbose,
        args=build_args,
        defer_jax_until_mlpot_registered=False,
        defer_jax_until_after_sd=False,
    )
    warmup_decomposed_mlpot(
        model,
        pos,
        cell=cell,
        verbose=args.verbose and not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    from mmml.utils.jax_gpu_warmup import maybe_log_jax_compile_timers

    maybe_log_jax_compile_timers(quiet=args.quiet)

    print(
        f"warmup-mlpot-jax: done in {elapsed:.1f}s "
        f"(checkpoint={ckpt.name}, cache={cache_dir})",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_warmup_mlpot_jax(args)


if __name__ == "__main__":
    sys.exit(main())
