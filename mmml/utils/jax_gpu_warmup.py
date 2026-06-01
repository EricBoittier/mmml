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
from typing import Any

logger = logging.getLogger(__name__)

_xla_gpu_warmed = False


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


def warmup_ase_mmml_energy_forces(atoms: Any, *, include_forces: bool = True) -> None:
    """JIT-warm an ASE calculator attached to ``atoms`` (energy, optionally forces)."""
    ensure_xla_gpu_warmed()
    energy = atoms.get_potential_energy()
    try:
        import jax
        import jax.numpy as jnp

        jax.block_until_ready(jnp.asarray(energy))
    except Exception:
        pass
    if include_forces:
        forces = atoms.get_forces()
        try:
            import jax
            import jax.numpy as jnp

            jax.block_until_ready(jnp.asarray(forces))
        except Exception:
            pass
