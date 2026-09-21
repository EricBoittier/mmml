"""Species-aware pooling of invariant atomic features.

Variable atom counts and compositions are handled by concatenating
per-element mean (and optional max) pools over a fixed species vocabulary,
plus a global mean pool.  Unpooled per-atom features are retained so local
environment coverage can be assessed instead of being silently destroyed by
pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PooledFeatures:
    """Pooled structure vectors plus the unpooled atoms they came from."""

    vectors: np.ndarray
    """``(n_structures, n_pool_features)``."""
    species: tuple[int, ...]
    n_atom_features: int
    atom_features: list[np.ndarray]
    """Per-structure ``(n_atoms_i, n_atom_features)`` arrays."""
    atom_species: list[np.ndarray]
    norms: np.ndarray
    """L2 of each pooled vector (pre-PCA magnitude diagnostic)."""


def pool_species_aware(
    atom_features: list[np.ndarray] | np.ndarray,
    atomic_numbers: list[np.ndarray] | np.ndarray,
    *,
    species: tuple[int, ...] | None = None,
    include_max: bool = True,
    include_global_mean: bool = True,
) -> PooledFeatures:
    """Pool invariant atomic features with explicit species channels.

    Parameters
    ----------
    atom_features
        Either a ragged list of ``(n_atoms_i, F)`` arrays or a padded
        ``(n_structures, pad, F)`` array (padding rows should be zeros with
        ``Z=0``).
    atomic_numbers
        Matching ``(n_atoms_i,)`` arrays or padded ``(n_structures, pad)``.
    species
        Vocabulary of atomic numbers.  Defaults to sorted unique Z>0.
    include_max
        Append per-species max-pool (helps unusual local environments survive
        mean pooling).
    include_global_mean
        Append a composition-agnostic global mean pool.
    """
    feats, zs = _as_ragged(atom_features, atomic_numbers)
    if not feats:
        raise ValueError("no structures to pool")
    n_f = int(feats[0].shape[-1])
    if species is None:
        present = sorted(
            {int(z) for zrow in zs for z in np.asarray(zrow).tolist() if int(z) > 0}
        )
        species = tuple(present)
    n_spec = len(species)
    # layout: [mean_Z1 | mean_Z2 | ... | (max_Z1 | ...) | (global_mean)]
    parts = n_spec  # means
    if include_max:
        parts += n_spec
    if include_global_mean:
        parts += 1
    dim = parts * n_f
    n = len(feats)
    vectors = np.zeros((n, dim), dtype=np.float64)
    for i, (phi, zrow) in enumerate(zip(feats, zs)):
        phi = np.asarray(phi, dtype=np.float64)
        zrow = np.asarray(zrow, dtype=np.int32)
        if phi.ndim != 2 or phi.shape[0] != zrow.shape[0]:
            raise ValueError(
                f"structure {i}: features {phi.shape} vs Z {zrow.shape}"
            )
        cursor = 0
        for spec in species:
            mask = zrow == spec
            if mask.any():
                vectors[i, cursor : cursor + n_f] = phi[mask].mean(axis=0)
            cursor += n_f
        if include_max:
            for spec in species:
                mask = zrow == spec
                if mask.any():
                    vectors[i, cursor : cursor + n_f] = phi[mask].max(axis=0)
                cursor += n_f
        if include_global_mean:
            real = zrow > 0
            if real.any():
                vectors[i, cursor : cursor + n_f] = phi[real].mean(axis=0)
            cursor += n_f
        if cursor != dim:
            raise RuntimeError("pooling layout mismatch")
    norms = np.linalg.norm(vectors, axis=1)
    return PooledFeatures(
        vectors=vectors,
        species=tuple(int(s) for s in species),
        n_atom_features=n_f,
        atom_features=[np.asarray(a, dtype=np.float64) for a in feats],
        atom_species=[np.asarray(z, dtype=np.int32) for z in zs],
        norms=norms,
    )


def _as_ragged(
    atom_features: list[np.ndarray] | np.ndarray,
    atomic_numbers: list[np.ndarray] | np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if isinstance(atom_features, list) and isinstance(atomic_numbers, list):
        if len(atom_features) != len(atomic_numbers):
            raise ValueError("features / Z list length mismatch")
        return atom_features, atomic_numbers
    feat = np.asarray(atom_features)
    z = np.asarray(atomic_numbers)
    if feat.ndim == 2:
        feat = feat[None, ...]
    if z.ndim == 1:
        z = z[None, ...]
    if feat.ndim != 3 or z.ndim != 2:
        raise ValueError(
            f"expected padded (n, pad, F) and (n, pad); got {feat.shape} {z.shape}"
        )
    feats: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for i in range(feat.shape[0]):
        n_i = int((z[i] > 0).sum())
        feats.append(feat[i, :n_i])
        zs.append(z[i, :n_i])
    return feats, zs


def local_coverage_indices(
    pooled: PooledFeatures,
    *,
    n_select: int,
    seed: int = 0,
) -> np.ndarray:
    """FPS on concatenated per-atom features (local-environment diagnostic).

    Returns structure indices (not atom indices).  A structure scores as
    the minimum distance of any of its atoms to the selected atom set; we
    greedily pick the structure that owns the currently farthest atom.
    This does not replace pooled-activation selection; it checks whether
    pooling hid unusual environments.
    """
    from mmml.acquisition.selection import farthest_point_sampling

    if n_select <= 0:
        return np.zeros((0,), dtype=np.int64)
    rows = []
    owners = []
    for i, (phi, zrow) in enumerate(zip(pooled.atom_features, pooled.atom_species)):
        real = np.asarray(zrow) > 0
        for row in np.asarray(phi)[real]:
            rows.append(row)
            owners.append(i)
    if not rows:
        return np.zeros((0,), dtype=np.int64)
    X = np.vstack(rows)
    atom_sel = farthest_point_sampling(X, min(n_select, len(X)), seed=seed)
    struct = []
    seen = set()
    for a in atom_sel:
        s = int(owners[int(a)])
        if s not in seen:
            seen.add(s)
            struct.append(s)
        if len(struct) >= n_select:
            break
    return np.asarray(struct, dtype=np.int64)
