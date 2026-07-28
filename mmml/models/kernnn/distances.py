"""JAX interatomic distance helpers for KerNN.

Default trained path uses H2CO / ABCC (C, O, H, H) with six pairwise distances.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_bond_length_abcc(pos, n_int_dist: int | None = None):
    """Interatomic distances for H2CO-style ABCC geometries.

    Atom order: C=0, O=1, H=2, H=3.

    Distance order:
      0: C–O, 1: C–H1, 2: C–H2, 3: O–H1, 4: O–H2, 5: H1–H2

    Parameters
    ----------
    pos :
        ``(4, 3)`` or ``(B, 4, 3)`` Cartesian coordinates (Å).
    n_int_dist :
        Expected number of distances (must be 6 when provided).
    """
    if n_int_dist is not None and int(n_int_dist) != 6:
        raise ValueError(f"ABCC expects n_int_dist=6, got {n_int_dist}")

    if pos.ndim == 2:
        if pos.shape[0] != 4:
            raise ValueError(f"ABCC expects 4 atoms, got shape {pos.shape}")
        pairs = (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        )
        return jnp.stack(
            [jnp.linalg.norm(pos[i] - pos[j]) for i, j in pairs],
            axis=0,
        )

    if pos.ndim == 3:
        if pos.shape[1] != 4:
            raise ValueError(f"ABCC expects 4 atoms, got shape {pos.shape}")
        pairs = (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        )
        return jnp.stack(
            [
                jnp.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
                for i, j in pairs
            ],
            axis=-1,
        )

    raise ValueError(
        f"positions must be (N,3) or (B,N,3); got shape {getattr(pos, 'shape', None)}"
    )
