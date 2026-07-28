"""JAX interatomic distance helpers for KerNN.

Default trained path uses H2CO / ABCC (C, O, H, H) with six pairwise distances.
``abcc_sym`` adds fundamental-invariant features for H↔H permutational symmetry.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_bond_length_abcc(pos, n_int_dist: int | None = None):
    """Interatomic distances for H2CO-style ABCC geometries.

    Atom order: C=0, O=1, H=2, H=3.

    Distance order:
      0: C–O, 1: C–H1, 2: C–H2, 3: O–H1, 4: O–H2, 5: H1–H2
    """
    if n_int_dist is not None and int(n_int_dist) != 6:
        raise ValueError(f"ABCC expects n_int_dist=6, got {n_int_dist}")

    if pos.ndim == 2:
        if pos.shape[0] != 4:
            raise ValueError(f"ABCC expects 4 atoms, got shape {pos.shape}")
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        return jnp.stack(
            [jnp.linalg.norm(pos[i] - pos[j]) for i, j in pairs],
            axis=0,
        )

    if pos.ndim == 3:
        if pos.shape[1] != 4:
            raise ValueError(f"ABCC expects 4 atoms, got shape {pos.shape}")
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
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


def get_bond_length_abcc_sym(pos, n_int_dist: int | None = None):
    """ABCC distances with H↔H fundamental invariants (7 features).

    Port of ``scripts/kernn/utils/distances.get_bond_length_ABCC_sym``.
    """
    if n_int_dist is not None and int(n_int_dist) != 6:
        raise ValueError(f"ABCC_sym expects n_int_dist=6, got {n_int_dist}")

    if pos.ndim == 2:
        if pos.shape[0] != 4:
            raise ValueError(f"ABCC_sym expects 4 atoms, got shape {pos.shape}")
        d_co = jnp.linalg.norm(pos[0] - pos[1])
        d_ch1 = jnp.linalg.norm(pos[0] - pos[2])
        d_ch2 = jnp.linalg.norm(pos[0] - pos[3])
        d_oh1 = jnp.linalg.norm(pos[1] - pos[2])
        d_oh2 = jnp.linalg.norm(pos[1] - pos[3])
        d_hh = jnp.linalg.norm(pos[2] - pos[3])
        return jnp.stack(
            [
                d_co,
                d_ch1 + d_ch2,
                d_oh1 + d_oh2,
                d_ch1**2 + d_ch2**2,
                d_oh1**2 + d_oh2**2,
                d_ch1 * d_oh1 + d_ch2 * d_oh2,
                d_hh,
            ],
            axis=0,
        )

    if pos.ndim == 3:
        if pos.shape[1] != 4:
            raise ValueError(f"ABCC_sym expects 4 atoms, got shape {pos.shape}")
        d_co = jnp.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=-1)
        d_ch1 = jnp.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=-1)
        d_ch2 = jnp.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=-1)
        d_oh1 = jnp.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=-1)
        d_oh2 = jnp.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=-1)
        d_hh = jnp.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=-1)
        return jnp.stack(
            [
                d_co,
                d_ch1 + d_ch2,
                d_oh1 + d_oh2,
                d_ch1**2 + d_ch2**2,
                d_oh1**2 + d_oh2**2,
                d_ch1 * d_oh1 + d_ch2 * d_oh2,
                d_hh,
            ],
            axis=-1,
        )

    raise ValueError(
        f"positions must be (N,3) or (B,N,3); got shape {getattr(pos, 'shape', None)}"
    )


DISTANCE_FNS = {
    "abcc": get_bond_length_abcc,
    "abcc_sym": get_bond_length_abcc_sym,
}


def n_features_for_scheme(scheme: str) -> int:
    if scheme == "abcc":
        return 6
    if scheme == "abcc_sym":
        return 7
    raise ValueError(f"unknown distance_scheme {scheme!r}; choose abcc or abcc_sym")
