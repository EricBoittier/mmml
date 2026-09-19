"""Multi-GPU chunk parallelism for hybrid ML PhysNet batches."""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import (
    effective_ml_gpu_count,
    resolve_ml_gpu_count,
)

Array = Any

__all__ = ["resolve_ml_gpu_count", "effective_ml_gpu_count", "run_chunked_model_apply"]


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
    has_aux: bool = False,
    n_valid: Array | int | None = None,
) -> tuple:
    """Evaluate PhysNet chunks; use ``jax.pmap`` when ``n_gpus > 1``.

    ``n_valid`` (may be traced): only the first ``n_valid`` batch slots hold
    systems whose output is used; the rest are padding (e.g. unused sparse
    dimer slots, which ``jnp.nonzero(..., size=cap)`` packs at the end). On a
    single GPU, chunks that lie entirely past ``n_valid`` are skipped with
    ``lax.cond`` and return zeros instead of running the model on padding.
    Chunks that are evaluated see exactly the same inputs, so used outputs are
    unchanged, and the chunk shape stays static (no recompiles).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
        get_mlpot_profile_stats,
        mlpot_profiling_enabled,
    )

    profile = mlpot_profiling_enabled()
    t0 = time.perf_counter() if profile else None

    if n_gpus <= 1:
        if n_valid is None:

            def one_chunk(i):
                return apply_one_chunk(R_chunks[i], Z_chunks[i], N_chunks[i])

        else:
            out_struct = jax.eval_shape(apply_one_chunk, R_chunks[0], Z_chunks[0], N_chunks[0])

            def _zeros_like_out():
                return jax.tree_util.tree_map(
                    lambda s: jnp.zeros(s.shape, s.dtype), out_struct
                )

            def one_chunk(i):
                return jax.lax.cond(
                    i * chunk_size < n_valid,
                    lambda: apply_one_chunk(R_chunks[i], Z_chunks[i], N_chunks[i]),
                    _zeros_like_out,
                )

        mapped = jax.lax.map(one_chunk, jnp.arange(n_chunks))
        if has_aux:
            e_list, f_list, aux_list = mapped
        else:
            e_list, f_list = mapped
        e_out = jnp.reshape(e_list, -1)[:effective_batch_size]
        f_out = jnp.reshape(f_list, (-1, 3))[: effective_batch_size * max_atoms]
        if has_aux:
            aux_out = jnp.reshape(aux_list, (-1, aux_list.shape[-1]))[
                : effective_batch_size * max_atoms
            ]
    else:
        n_padded = int(np.ceil(n_chunks / n_gpus) * n_gpus)
        if n_padded > n_chunks:
            pad_c = n_padded - n_chunks
            R_chunks = jnp.concatenate(
                [
                    R_chunks,
                    jnp.zeros(
                        (pad_c, chunk_size, max_atoms, 3), dtype=R_chunks.dtype
                    ),
                ]
            )
            Z_chunks = jnp.concatenate(
                [
                    Z_chunks,
                    jnp.zeros(
                        (pad_c, chunk_size, max_atoms), dtype=Z_chunks.dtype
                    ),
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

        mapped = jax.lax.map(process_wave, jnp.arange(n_waves))
        if has_aux:
            e_waves, f_waves, aux_waves = mapped
        else:
            e_waves, f_waves = mapped
        e_flat = jnp.reshape(e_waves, (-1))
        f_flat = jnp.reshape(f_waves, (-1, 3))

        e_out = e_flat[:effective_batch_size]
        f_out = f_flat[: effective_batch_size * max_atoms]
        if has_aux:
            aux_flat = jnp.reshape(aux_waves, (-1, aux_waves.shape[-1]))
            aux_out = aux_flat[: effective_batch_size * max_atoms]

    if profile and t0 is not None:
        e_out = jax.block_until_ready(e_out)
        f_out = jax.block_until_ready(f_out)
        get_mlpot_profile_stats().record_chunk_apply(
            time.perf_counter() - t0,
            n_gpus=int(n_gpus),
            n_chunks=int(n_chunks),
            chunk_size=int(chunk_size),
            effective_batch_size=int(effective_batch_size),
        )
    if has_aux:
        return e_out, f_out, aux_out
    return e_out, f_out
