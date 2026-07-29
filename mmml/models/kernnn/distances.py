"""JAX interatomic distance helpers for KerNN.

Schemes:
  abcc / abcc_sym — H2CO (4 atoms)
  form — formamide (6 atoms, all pairs)
  acem — acetamide (9 atoms, all pairs)
"""

from __future__ import annotations

import jax.numpy as jnp

SCHEME_N_ATOMS = {
    "abcc": 4,
    "abcc_sym": 4,
    "form": 6,
    "acem": 9,
}


def n_atoms_for_scheme(scheme: str) -> int:
    try:
        return SCHEME_N_ATOMS[scheme]
    except KeyError as exc:
        raise ValueError(
            f"unknown distance_scheme {scheme!r}; choose one of {sorted(SCHEME_N_ATOMS)}"
        ) from exc


def n_features_for_scheme(scheme: str) -> int:
    if scheme == "abcc":
        return 6
    if scheme == "abcc_sym":
        return 7
    n = n_atoms_for_scheme(scheme)
    return n * (n - 1) // 2


def _pair_list(n_atoms: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms))


def _pairwise(pos, n_atoms: int):
    pairs = _pair_list(n_atoms)
    if pos.ndim == 2:
        if pos.shape[0] != n_atoms:
            raise ValueError(f"expected {n_atoms} atoms, got shape {pos.shape}")
        return jnp.stack(
            [jnp.linalg.norm(pos[i] - pos[j]) for i, j in pairs],
            axis=0,
        )
    if pos.ndim == 3:
        if pos.shape[1] != n_atoms:
            raise ValueError(f"expected {n_atoms} atoms, got shape {pos.shape}")
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


def get_bond_length_abcc(pos, n_int_dist: int | None = None):
    """H2CO ABCC distances (C,O,H,H) — 6 pairs."""
    del n_int_dist
    return _pairwise(pos, 4)


def get_bond_length_abcc_sym(pos, n_int_dist: int | None = None):
    """ABCC with H↔H fundamental invariants (7 features)."""
    del n_int_dist
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


def get_bond_length_form(pos, n_int_dist: int | None = None):
    """Formamide all-pairs distances (H,C,N,H,H,O) — 15 features."""
    del n_int_dist
    return _pairwise(pos, 6)


def get_bond_length_acem(pos, n_int_dist: int | None = None):
    """Acetamide all-pairs distances (9 atoms) — 36 features."""
    del n_int_dist
    return _pairwise(pos, 9)


DISTANCE_FNS = {
    "abcc": get_bond_length_abcc,
    "abcc_sym": get_bond_length_abcc_sym,
    "form": get_bond_length_form,
    "acem": get_bond_length_acem,
}
