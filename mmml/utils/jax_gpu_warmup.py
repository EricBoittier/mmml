"""Warm up JAX/XLA on GPU before timed kernels run.

XLA's CUDA timer uses a short "delay" kernel to calibrate GPU timing. Without a
prior completed GPU execution, that kernel can time out and log::

    Delay kernel timed out: measured time has sub-optimal accuracy.
    There may be a missing warmup execution

Call :func:`ensure_xla_gpu_warmed` once per process before the first large
``jax.jit`` evaluation (e.g. hybrid MMML calculator).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_xla_gpu_warmed = False


def apply_xla_cuda_timer_log_filter() -> None:
    """Suppress XLA ``cuda_timer.cc`` delay-kernel timeout noise (harmless autotuner warnings).

  Set env ``MMML_SUPPRESS_XLA_CUDA_TIMER=1`` (default) to raise ``TF_CPP_MIN_LOG_LEVEL``
  to 3 when it is unset. Set ``MMML_SUPPRESS_XLA_CUDA_TIMER=0`` to leave logging unchanged.
    """
    if os.environ.get("MMML_SUPPRESS_XLA_CUDA_TIMER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def ensure_xla_gpu_warmed(*, force: bool = False) -> bool:
    """Run a tiny JITted reduction on GPU and block until complete.

    Returns True if a GPU warmup ran, False if JAX is missing or no GPU backend.
    Idempotent unless ``force=True``.
    """
    global _xla_gpu_warmed
    if _xla_gpu_warmed and not force:
        return False

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        _xla_gpu_warmed = True
        return False

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        _xla_gpu_warmed = True
        return False

    @jax.jit
    def _warmup_kernel(x: jnp.ndarray) -> jnp.ndarray:
        # Reduction + matmul: schedules real GPU work (not just a host alloc).
        y = jnp.sum(x * 1.001)
        m = jnp.ones((32, 32), dtype=x.dtype)
        return y + jnp.sum(m @ m)

    x = jnp.ones((256,), dtype=jnp.float32)
    # Two executions: first may compile; second satisfies delay-kernel calibration.
    for _ in range(2):
        out = _warmup_kernel(x)
        jax.block_until_ready(out)

    _xla_gpu_warmed = True
    logger.debug("XLA GPU delay-kernel warmup completed on %s", gpu_devices[0])
    return True


def block_jax_values(*values: Any) -> None:
    """Block until JAX array leaves are ready (no-op if JAX is unavailable)."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    for value in values:
        try:
            jax.block_until_ready(jnp.asarray(value))
        except Exception:
            pass


def warmup_hybrid_spherical_cutoff(
    spherical_cutoff_calculator: Any,
    *,
    atomic_numbers: Any,
    positions: Any,
    n_monomers: int,
    cutoff_params: Any,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    debug: bool = False,
    mm_pair_idx: Any = None,
    mm_pair_mask: Any = None,
    box: Any = None,
) -> None:
    """Compile and run one hybrid MMML eval; block until GPU work completes.

    Call after PyCHARMM/MM setup (e.g. CGENFF drudes) and before timed JAX-MD compiles.
    """
    ensure_xla_gpu_warmed(force=True)
    kwargs = dict(
        positions=positions,
        atomic_numbers=atomic_numbers,
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        doML=doML,
        doMM=doMM,
        doML_dimer=doML_dimer,
        debug=debug,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        box=box,
    )
    # Two untimed runs: first compiles/autotunes; second calibrates XLA's delay kernel.
    for _ in range(2):
        result = spherical_cutoff_calculator(**kwargs)
        block_jax_values(getattr(result, "energy", None), getattr(result, "forces", None))


def warmup_ase_mmml_energy_forces(atoms: Any, *, include_forces: bool = True) -> None:
    """JIT-warm an ASE calculator attached to ``atoms`` (energy, optionally forces)."""
    ensure_xla_gpu_warmed(force=True)
    energy = atoms.get_potential_energy()
    block_jax_values(energy)
    if include_forces:
        forces = atoms.get_forces()
        block_jax_values(forces)
