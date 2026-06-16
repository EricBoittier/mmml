"""Temporarily raise CPU thread limits during JAX/XLA compile (not during CHARMM ``upinb``).

CHARMM MPI builds pin ``OMP_NUM_THREADS=1`` for ``upinb`` safety. GPU JIT compile is
mostly CPU-bound (LLVM, Eigen, ``ptxas``); bumping BLAS/OpenMP during warmup only
can shorten first-compile wall time without touching MLpot registration.

Enable (default when unset): set ``MMML_JAX_COMPILE_THREADS`` to the desired count
(default: ``min(16, cpu_count)``). Disable with ``MMML_NO_JAX_COMPILE_THREADS=1``.

For XLA's Eigen pool, also call :func:`apply_jax_compile_xla_flags` before the first
``import jax`` (``md-system`` does this for the pycharmm backend).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_OPENMP_LIKE_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def jax_compile_threads_enabled() -> bool:
    return not _truthy("MMML_NO_JAX_COMPILE_THREADS")


def resolve_jax_compile_thread_count() -> int:
    """Thread budget for JAX compile warmup (0 when disabled)."""
    if not jax_compile_threads_enabled():
        return 0
    raw = (os.environ.get("MMML_JAX_COMPILE_THREADS") or "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        return max(0, n)
    cpu = os.cpu_count() or 8
    return max(1, min(16, int(cpu)))


def apply_jax_compile_xla_flags(*, quiet: bool = False) -> int:
    """Append XLA CPU thread flags before JAX backend init (no-op when disabled)."""
    n = resolve_jax_compile_thread_count()
    if n <= 0:
        return 0
    # ``xla_cpu_thread_pool_size`` is not in all JAX/XLA builds (gpu08 jaxlib 0.9.x
    # aborts on unknown flags). ``intra_op_parallelism_threads`` is widely supported.
    flags = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={n}"
    )
    existing = (os.environ.get("XLA_FLAGS") or "").strip()
    if "xla_cpu_multi_thread_eigen" in existing or "intra_op_parallelism_threads" in existing:
        if not quiet and not _truthy("MMML_QUIET"):
            print(
                "mmml: JAX compile XLA_FLAGS already set; not overriding",
                flush=True,
            )
        return n
    merged = f"{existing} {flags}".strip() if existing else flags
    os.environ["XLA_FLAGS"] = merged
    if not quiet and not _truthy("MMML_QUIET"):
        print(
            f"mmml: JAX compile XLA_FLAGS intra_op_parallelism_threads={n}",
            flush=True,
        )
    return n


@contextmanager
def jax_compile_threads_context(*, quiet: bool = False) -> Iterator[int]:
    """Raise OpenMP/BLAS thread caps during JAX compile; restore on exit."""
    n = resolve_jax_compile_thread_count()
    if n <= 0:
        yield 0
        return

    saved = {key: os.environ.get(key) for key in _OPENMP_LIKE_VARS}
    thread_s = str(n)
    for key in _OPENMP_LIKE_VARS:
        os.environ[key] = thread_s
    if not quiet and not _truthy("MMML_QUIET"):
        prev_omp = saved.get("OMP_NUM_THREADS")
        print(
            f"mmml: JAX compile threads={n} "
            f"(OMP {prev_omp!r} -> {thread_s}; restored after warmup)",
            flush=True,
        )
    try:
        yield n
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
