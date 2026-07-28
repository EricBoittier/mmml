"""Dihedral helpers for Dual-FFNet KerNN."""

from __future__ import annotations

import jax.numpy as jnp


def dihedral_angle(p0, p1, p2, p3):
    """Torsion angle (radians) for atoms p0-p1-p2-p3.

    Works for ``(..., 3)`` coordinate arrays (single or batched).
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
    v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1, v) * w, axis=-1)
    return jnp.arctan2(y, x)


def h2co_hcoh_dihedral(pos):
    """H2–C–O–H3 dihedral for ABCC atom order (C=0, O=1, H=2, H=3).

    Returns shape ``()`` or ``(B,)``; DualFFNet expects a trailing feature dim.
    """
    if pos.ndim == 2:
        return dihedral_angle(pos[2], pos[0], pos[1], pos[3])
    if pos.ndim == 3:
        return dihedral_angle(pos[:, 2], pos[:, 0], pos[:, 1], pos[:, 3])
    raise ValueError(f"positions must be (N,3) or (B,N,3); got {pos.shape}")
