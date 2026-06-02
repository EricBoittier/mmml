"""MLpot PhysNet batch chunk defaults (no JAX import)."""

from __future__ import annotations

import os
from typing import Optional

from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_name


def resolve_ml_batch_size(
    n_monomers: int,
    explicit: Optional[int] = None,
) -> Optional[int]:
    """Chunk size for PhysNet forward passes (limits XLA LLVM compile RAM).

    DCM:90 sparse path evaluates up to ~1090 systems (90 monomers + 1000 dimer slots) per step.
    GPU defaults use larger chunks (256) for throughput; CPU keeps smaller chunks (64)
    to limit JAX LLVM compile memory.
    """
    if explicit is not None:
        return int(explicit)
    env = (os.environ.get("MMML_MLPOT_ML_BATCH_SIZE") or "").strip()
    if env:
        return int(env)
    n = int(n_monomers)
    if n <= 10:
        return None
    on_gpu = mlpot_jax_device_name() == "gpu"
    if n >= 40:
        return 256 if on_gpu else 64
    if n >= 20:
        return 256 if on_gpu else 128
    return 512 if on_gpu else 256
