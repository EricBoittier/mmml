"""MLpot multi-GPU chunk policy (no JAX import)."""

from __future__ import annotations

import os
from typing import Optional

from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_local_gpu_count


def resolve_ml_gpu_count(explicit: Optional[int] = None) -> int:
    """Number of local GPUs to use for parallel PhysNet chunk evaluation (default 1)."""
    if explicit is not None:
        return max(1, int(explicit))
    env = (os.environ.get("MMML_MLPOT_N_GPUS") or "").strip()
    if env:
        return max(1, int(env))
    return 1


def effective_ml_gpu_count(
    requested: Optional[int],
    *,
    n_chunks: int,
) -> int:
    """Clamp requested GPU count to available devices and chunk count."""
    want = resolve_ml_gpu_count(requested)
    local = mlpot_local_gpu_count()
    if local <= 0 or n_chunks <= 1:
        return 1
    return max(1, min(want, local, int(n_chunks)))
