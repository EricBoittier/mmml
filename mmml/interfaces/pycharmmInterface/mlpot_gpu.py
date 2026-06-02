"""Multi-GPU chunk parallelism for hybrid ML PhysNet batches."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.ml_batching import prepare_batches_md

Array = Any


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
    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_local_gpu_count

    want = resolve_ml_gpu_count(requested)
    local = mlpot_local_gpu_count()
    if local <= 0 or n_chunks <= 1:
        return 1
    return max(1, min(want, local, int(n_chunks)))


def run_chunked_model_apply(
    *,
    R_chunks: Array,
    Z_chunks: Array,
    N_chunks: Array,
    n_chunks: int,
    effective_batch_size: int,
    chunk_size: int,
    max_atoms: int,
    n_gpus: int,
    apply_one_chunk: Callable[[Array, Array, Array], Tuple[Array, Array]],
) -> Tuple[Array, Array]:
    """Evaluate PhysNet chunks; use ``jax.pmap`` when ``n_gpus > 1``."""
    if n_gpus <= 1:
        e_list, f_list = jax.lax.map(
            lambda i: apply_one_chunk(R_chunks[i], Z_chunks[i], N_chunks[i]),
            jnp.arange(n_chunks),
        )
        e_out = jnp.reshape(e_list, -1)[:effective_batch_size]
        f_out = jnp.reshape(f_list, (-1, 3))[: effective_batch_size * max_atoms]
        return e_out, f_out

    n_padded = int(np.ceil(n_chunks / n_gpus) * n_gpus)
    if n_padded > n_chunks:
        pad_c = n_padded - n_chunks
        R_chunks = jnp.concatenate(
            [R_chunks, jnp.zeros((pad_c, chunk_size, max_atoms, 3), dtype=R_chunks.dtype)]
        )
        Z_chunks = jnp.concatenate(
            [
                Z_chunks,
                jnp.zeros((pad_c, chunk_size, max_atoms), dtype=Z_chunks.dtype),
            ]
        )
        N_chunks = jnp.concatenate(
            [N_chunks, jnp.ones((pad_c, chunk_size), dtype=N_chunks.dtype)]
        )

    n_waves = n_padded // n_gpus
    R_w = R_chunks.reshape(n_waves, n_gpus, chunk_size, max_atoms, 3)
    Z_w = Z_chunks.reshape(n_waves, n_gpus, chunk_size, max_atoms)
    N_w = N_chunks.reshape(n_waves, n_gpus, chunk_size)

    pmap_apply = jax.pmap(apply_one_chunk, in_axes=(0, 0, 0))

    def process_wave(w: Array) -> Tuple[Array, Array]:
        wi = w.astype(jnp.int32)
        return pmap_apply(R_w[wi], Z_w[wi], N_w[wi])

    e_waves, f_waves = jax.lax.map(process_wave, jnp.arange(n_waves))
    e_flat = jnp.reshape(e_waves, (-1))
    f_flat = jnp.reshape(f_waves, (-1, 3))

    e_out = e_flat[:effective_batch_size]
    f_out = f_flat[: effective_batch_size * max_atoms]
    return e_out, f_out
