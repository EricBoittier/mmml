"""Differentiable pairwise quantum-Drude-oscillator dispersion baseline."""

from __future__ import annotations

import jax.numpy as jnp


def qdo_pairwise_dispersion(
    positions: jnp.ndarray,
    dst_idx: jnp.ndarray,
    src_idx: jnp.ndarray,
    coefficients: jnp.ndarray,
    damping_radii: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate ``-sum(C6/(r6+Rd6) + C8/(r8+Rd8) + C10/(r10+Rd10))``.

    ``coefficients`` has shape ``(num_edges, 3)`` ordered as C6, C8, C10.
    Directed neighbor lists are accepted; only edges with ``dst_idx < src_idx``
    contribute, so every physical pair is counted once.
    """
    positions = jnp.asarray(positions)
    coefficients = jnp.asarray(coefficients)
    damping_radii = jnp.asarray(damping_radii)
    if coefficients.shape[-1] != 3:
        raise ValueError("coefficients must contain C6, C8, and C10")

    displacement = positions[src_idx] - positions[dst_idx]
    distance_squared = jnp.sum(jnp.square(displacement), axis=-1)
    powers = jnp.asarray((3, 4, 5), dtype=positions.dtype)
    distance_terms = distance_squared[:, None] ** powers
    damping_terms = damping_radii[:, None] ** (2 * powers)
    pair_energy = -jnp.sum(coefficients / (distance_terms + damping_terms), axis=-1)
    return jnp.sum(jnp.where(dst_idx < src_idx, pair_energy, 0.0))
