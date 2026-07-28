"""Packed-batch ML + harmonic distance umbrella bias energies."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def cv_distance(positions: Any, atom_i: int, atom_j: int) -> Any:
    """Scalar distance between two atoms (Å)."""
    import jax.numpy as jnp

    r = positions[atom_j] - positions[atom_i]
    return jnp.sqrt(jnp.sum(r * r) + 1e-12)


def bias_energy(distance: Any, target: float, k_ev_A2: float) -> Any:
    """Harmonic umbrella bias ``0.5 * k * (r - ξ₀)²`` (eV)."""
    import jax.numpy as jnp

    return 0.5 * float(k_ev_A2) * jnp.square(distance - float(target))


def packed_cv_distances(
    positions_packed: Any,
    n_atoms: int,
    atom_i: int,
    atom_j: int,
    n_windows: int,
) -> Any:
    """Distances for each packed window copy. Shape ``(K,)``."""
    import jax.numpy as jnp

    pos = positions_packed.reshape(n_windows, n_atoms, 3)
    disp = pos[:, atom_j, :] - pos[:, atom_i, :]
    return jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1e-12)


def packed_bias_energies(
    positions_packed: Any,
    n_atoms: int,
    atom_i: int,
    atom_j: int,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
) -> Any:
    """Per-window 1D bias energies. Shape ``(K,)``."""
    import jax.numpy as jnp

    k = len(targets_A)
    dists = packed_cv_distances(positions_packed, n_atoms, atom_i, atom_j, k)
    targets = jnp.asarray(targets_A, dtype=dists.dtype)
    ks = jnp.asarray(k_ev_A2, dtype=dists.dtype)
    return 0.5 * ks * jnp.square(dists - targets)


def packed_bias_energies_nd(
    positions_packed: Any,
    n_atoms: int,
    atom_pairs: Sequence[tuple[int, int]],
    targets: Sequence[Sequence[float]],
    k_ev_A2: Sequence[Sequence[float]],
) -> Any:
    """Sum of harmonic biases over CVs. ``targets`` / ``k_ev_A2`` are ``(ndim, K)``-like."""
    import jax.numpy as jnp

    total = None
    for dim, (i, j) in enumerate(atom_pairs):
        term = packed_bias_energies(
            positions_packed,
            n_atoms,
            int(i),
            int(j),
            targets[dim],
            k_ev_A2[dim],
        )
        total = term if total is None else total + term
    assert total is not None
    return total


def build_packed_graph(n_atoms: int, n_windows: int) -> dict[str, Any]:
    """e3x pair indices / batch segments for ``K`` tiled copies of an ``N``-atom system.

    Layout matches ``physnet-md`` multi-replica JAX-MD packing.
    """
    import e3x
    import jax.numpy as jnp

    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1 (got {n_atoms})")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1 (got {n_windows})")

    dst_single, src_single = e3x.ops.sparse_pairwise_indices(n_atoms)
    dst_single = np.asarray(dst_single, dtype=np.int32)
    src_single = np.asarray(src_single, dtype=np.int32)
    offsets = np.arange(n_windows, dtype=np.int32) * n_atoms
    dst_idx = np.concatenate([dst_single + off for off in offsets])
    src_idx = np.concatenate([src_single + off for off in offsets])

    return {
        "dst_idx": jnp.asarray(dst_idx, dtype=jnp.int32),
        "src_idx": jnp.asarray(src_idx, dtype=jnp.int32),
        "batch_segments": jnp.repeat(jnp.arange(n_windows, dtype=jnp.int32), n_atoms),
        "batch_mask": jnp.ones(len(dst_idx), dtype=jnp.float32),
        "atom_mask": jnp.ones(n_windows * n_atoms, dtype=jnp.float32),
        "batch_size": int(n_windows),
        "n_atoms": int(n_atoms),
        "n_windows": int(n_windows),
    }


def pack_positions(positions: np.ndarray, n_windows: int) -> np.ndarray:
    """Tile ``(N, 3)`` → packed ``(K*N, 3)`` float64 array."""
    r = np.asarray(positions, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {r.shape}")
    return np.tile(r[None, :, :], (n_windows, 1, 1)).reshape(n_windows * r.shape[0], 3)


def make_packed_energy_fn(
    *,
    model_apply: Callable[..., dict[str, Any]],
    params: Any,
    atomic_numbers: Any,
    graph: dict[str, Any],
    atom_pairs: Sequence[tuple[int, int]],
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
) -> Callable[..., Any]:
    """Return ``energy_sum_fn(R_packed) = sum(E_ML) + sum_k W_k`` for NVT.

    ``targets_per_cv[d]`` and ``k_per_cv[d]`` are length-``K`` sequences for CV ``d``.
    """
    import jax.numpy as jnp

    n_atoms = int(graph["n_atoms"])
    n_windows = int(graph["n_windows"])
    z = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    if z.shape[0] == n_atoms:
        z_batched = jnp.tile(z, n_windows)
    elif z.shape[0] == n_atoms * n_windows:
        z_batched = z
    else:
        raise ValueError(
            f"atomic_numbers length {z.shape[0]} incompatible with "
            f"n_atoms={n_atoms}, n_windows={n_windows}"
        )

    pairs = tuple((int(i), int(j)) for i, j in atom_pairs)
    targets = tuple(tuple(float(x) for x in row) for row in targets_per_cv)
    ks = tuple(tuple(float(x) for x in row) for row in k_per_cv)
    if len(pairs) != len(targets) or len(pairs) != len(ks):
        raise ValueError("atom_pairs / targets_per_cv / k_per_cv length mismatch")
    for row in targets + ks:
        if len(row) != n_windows:
            raise ValueError("each CV target/k row must match graph n_windows")

    dst_idx = graph["dst_idx"]
    src_idx = graph["src_idx"]
    batch_segments = graph["batch_segments"]
    batch_mask = graph["batch_mask"]
    atom_mask = graph["atom_mask"]
    batch_size = graph["batch_size"]

    def energy_sum_fn(position, **kwargs):
        del kwargs
        out = model_apply(
            params,
            atomic_numbers=z_batched,
            positions=position,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=batch_size,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        e_ml = jnp.sum(jnp.asarray(out["energy"]).reshape(-1))
        e_bias = jnp.sum(
            packed_bias_energies_nd(position, n_atoms, pairs, targets, ks)
        )
        return e_ml + e_bias

    return energy_sum_fn


def make_single_ml_energy_fn(
    *,
    model_apply: Callable[..., dict[str, Any]],
    params: Any,
    atomic_numbers: Any,
    n_atoms: int,
) -> Callable[[Any], Any]:
    """Unbiased single-system ML energy (for MBAR re-evaluation)."""
    import e3x
    import jax.numpy as jnp

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
    z = jnp.asarray(atomic_numbers, dtype=jnp.int32)[:n_atoms]
    batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
    batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
    atom_mask = jnp.ones(n_atoms, dtype=jnp.float32)

    def energy_fn(position):
        out = model_apply(
            params,
            atomic_numbers=z,
            positions=position,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        return jnp.squeeze(jnp.asarray(out["energy"]))

    return energy_fn


def numpy_bias_matrix(
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
) -> np.ndarray:
    """Analytic 1D ``W_l(R)`` for one frame. Shape ``(K,)``."""
    r = np.asarray(positions, dtype=np.float64)
    dist = float(np.linalg.norm(r[atom_j] - r[atom_i]))
    targets = np.asarray(targets_A, dtype=np.float64)
    ks = np.asarray(k_ev_A2, dtype=np.float64)
    return 0.5 * ks * (dist - targets) ** 2


def numpy_bias_matrix_nd(
    positions: np.ndarray,
    atom_pairs: Sequence[tuple[int, int]],
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
) -> np.ndarray:
    """Analytic multi-CV ``W_l(R)`` for one frame. Shape ``(K,)``."""
    total = None
    for dim, (i, j) in enumerate(atom_pairs):
        term = numpy_bias_matrix(
            positions, int(i), int(j), targets_per_cv[dim], k_per_cv[dim]
        )
        total = term if total is None else total + term
    assert total is not None
    return total
