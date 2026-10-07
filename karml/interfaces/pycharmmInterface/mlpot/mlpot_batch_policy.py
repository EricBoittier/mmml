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


# Verlet skin (Å) of the MLpot MM pair list (static Vesin/cell-list rebuild backend).
DEFAULT_MLPOT_MM_SKIN_A = 0.25


def resolve_mlpot_mm_skin_A(args: object | None = None) -> float:
    """Verlet skin for the MLpot MM pair list.

    ``MMML_MLPOT_MM_SKIN_A`` wins, then an explicit ``--jax-md-skin-distance``,
    then ``DEFAULT_MLPOT_MM_SKIN_A``. The list radius grows by the skin and the
    list is reused until some atom has moved (minimum image) more than skin/2.
    """
    env = (os.environ.get("MMML_MLPOT_MM_SKIN_A") or "").strip()
    if env:
        return max(0.0, float(env))
    explicit = getattr(args, "_cli_explicit", None) or set()
    val = getattr(args, "jax_md_skin_distance", None) if args is not None else None
    if "jax_md_skin_distance" in explicit and val is not None:
        return max(0.0, float(val))
    return DEFAULT_MLPOT_MM_SKIN_A
