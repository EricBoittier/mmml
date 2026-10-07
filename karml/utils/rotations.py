"""Utilities for SO(3) rotational data augmentation."""

from __future__ import annotations

import e3x
import jax
import jax.numpy as jnp


def sample_random_rotations(
    key: jax.Array,
    num: int,
    perturbation: float = 1.0,
) -> jax.Array:
    """Sample ``num`` random 3x3 rotation matrices."""
    return e3x.so3.rotations.random_rotation(
        key,
        perturbation=float(perturbation),
        num=int(num),
    )


def rotate_batched_vectors(vectors: jax.Array, rotations: jax.Array) -> jax.Array:
    """Rotate vectors with per-example rotations.

    Parameters
    ----------
    vectors
        Array with shape ``(B, ..., 3)``.
    rotations
        Rotation matrices with shape ``(B, 3, 3)``.
    """
    return jnp.einsum("bij,b...j->b...i", rotations, vectors)


def rotate_batched_rank2_tensors(
    tensors: jax.Array,
    rotations: jax.Array,
) -> jax.Array:
    """Rotate rank-2 tensors with ``Q' = R Q R^T``.

    Parameters
    ----------
    tensors
        Array with shape ``(B, 3, 3)``.
    rotations
        Rotation matrices with shape ``(B, 3, 3)``.
    """
    return jnp.einsum("bij,bjk,bkl->bil", rotations, tensors, jnp.swapaxes(rotations, -1, -2))
