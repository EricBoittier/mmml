"""Packed-batch ML + harmonic umbrella bias energies.

Each CV dimension is a :class:`~mmml.md.restraints.LinearDistanceCV`, i.e. a
linear combination of interatomic distances. A bare ``(i, j)`` pair is accepted
everywhere a CV is and promoted to the plain-distance special case, so existing
distance-umbrella callers are unaffected; the general form is what makes an
antisymmetric-stretch reaction coordinate such as ``xi = r(C-Cl) - r(C-N)``
expressible.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from mmml.md.restraints import FlatBottomWall, LinearDistanceCV


def _as_cvs(specs: Sequence[Any]) -> tuple[LinearDistanceCV, ...]:
    """Promote ``(i, j)`` pairs / mappings / CV instances to CVs.

    Lets every CV-consuming helper accept either the legacy pair form or a
    general linear combination of distances, so an antisymmetric-stretch
    reaction coordinate (xi = r(C-Cl) - r(C-N)) is expressible without forking
    the sampler.
    """
    return tuple(LinearDistanceCV.from_spec(spec) for spec in specs)


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


def packed_cv_values(
    positions_packed: Any,
    n_atoms: int,
    cv: Any,
    n_windows: int,
) -> Any:
    """CV value for each packed window copy. Shape ``(K,)``.

    Generalises :func:`packed_cv_distances` to any
    :class:`~mmml.md.restraints.LinearDistanceCV` (or ``(i, j)`` pair).
    """
    return LinearDistanceCV.from_spec(cv).value_batched(
        positions_packed, n_atoms, n_windows
    )


def packed_cv_values_nd(
    positions_packed: Any,
    n_atoms: int,
    cvs: Sequence[Any],
    n_windows: int,
) -> Any:
    """All CV values per window. Shape ``(K, ndim)``."""
    import jax.numpy as jnp

    cols = [
        LinearDistanceCV.from_spec(cv).value_batched(positions_packed, n_atoms, n_windows)
        for cv in cvs
    ]
    return jnp.stack(cols, axis=-1)


def packed_bias_energies_cv(
    positions_packed: Any,
    n_atoms: int,
    cv: Any,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
) -> Any:
    """Per-window harmonic bias on one general CV. Shape ``(K,)``."""
    import jax.numpy as jnp

    k = len(targets_A)
    values = packed_cv_values(positions_packed, n_atoms, cv, k)
    targets = jnp.asarray(targets_A, dtype=values.dtype)
    ks = jnp.asarray(k_ev_A2, dtype=values.dtype)
    return 0.5 * ks * jnp.square(values - targets)


def packed_bias_energies_nd(
    positions_packed: Any,
    n_atoms: int,
    cvs: Sequence[Any],
    targets: Sequence[Sequence[float]],
    k_ev_A2: Sequence[Sequence[float]],
) -> Any:
    """Sum of harmonic biases over CVs. ``targets`` / ``k_ev_A2`` are ``(ndim, K)``-like.

    ``cvs`` entries may be ``(i, j)`` pairs or :class:`LinearDistanceCV` objects.
    """
    total = None
    for dim, cv in enumerate(_as_cvs(cvs)):
        term = packed_bias_energies_cv(
            positions_packed,
            n_atoms,
            cv,
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


def packed_bias_forces(
    positions_packed: Any,
    n_atoms: int,
    atom_i: int,
    atom_j: int,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
) -> Any:
    """ASE-style forces ``F = -∇W`` for 1D harmonic distance bias. Shape ``(K*N, 3)``."""
    import jax.numpy as jnp

    k = len(targets_A)
    pos = positions_packed.reshape(k, n_atoms, 3)
    disp = pos[:, atom_j, :] - pos[:, atom_i, :]
    dist = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1e-12)
    u = disp / dist[:, None]
    targets = jnp.asarray(targets_A, dtype=dist.dtype)
    ks = jnp.asarray(k_ev_A2, dtype=dist.dtype)
    # W = 0.5 k (r-r0)^2 → ∇_i W = -k(r-r0)u, ∇_j W = +k(r-r0)u
    # F = -∇W → F_i = k(r-r0)u, F_j = -k(r-r0)u
    scale = (ks * (dist - targets))[:, None]
    forces = jnp.zeros_like(pos)
    forces = forces.at[:, atom_i, :].add(scale * u)
    forces = forces.at[:, atom_j, :].add(-scale * u)
    return forces.reshape(k * n_atoms, 3)


def packed_bias_forces_cv(
    positions_packed: Any,
    n_atoms: int,
    cv: Any,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
) -> Any:
    """ASE-style bias forces ``F = -grad W`` for one general CV. Shape ``(K*N, 3)``.

    Uses the CV's analytic gradient so bias forces never differentiate through
    the ML model (nesting AD inside PhysNet's ``value_and_grad`` yields NaNs).
    """
    import jax.numpy as jnp

    resolved = LinearDistanceCV.from_spec(cv)
    k = len(targets_A)
    values = resolved.value_batched(positions_packed, n_atoms, k)
    grad = resolved.gradient_batched(positions_packed, n_atoms, k)  # (K, N, 3)
    targets = jnp.asarray(targets_A, dtype=values.dtype)
    ks = jnp.asarray(k_ev_A2, dtype=values.dtype)
    # W = 0.5 k (xi - xi0)^2 -> grad W = k (xi - xi0) * grad xi ; F = -grad W
    scale = (ks * (values - targets))[:, None, None]
    return (-scale * grad).reshape(k * n_atoms, 3)


def packed_bias_forces_nd(
    positions_packed: Any,
    n_atoms: int,
    cvs: Sequence[Any],
    targets: Sequence[Sequence[float]],
    k_ev_A2: Sequence[Sequence[float]],
) -> Any:
    """Sum of ASE-style bias forces over CVs. Shape ``(K*N, 3)``."""
    total = None
    for dim, cv in enumerate(_as_cvs(cvs)):
        term = packed_bias_forces_cv(
            positions_packed,
            n_atoms,
            cv,
            targets[dim],
            k_ev_A2[dim],
        )
        total = term if total is None else total + term
    assert total is not None
    return total


def make_packed_energy_fn(
    *,
    model_apply: Callable[..., dict[str, Any]],
    params: Any,
    atomic_numbers: Any,
    graph: dict[str, Any],
    atom_pairs: Sequence[Any] | None = None,
    targets_per_cv: Sequence[Sequence[float]] = (),
    k_per_cv: Sequence[Sequence[float]] = (),
    cvs: Sequence[Any] | None = None,
    walls: Sequence[Any] | None = None,
) -> Callable[..., Any]:
    """Return ``energy_sum_fn(R_packed) = sum(E_ML) + sum_k W_k + walls`` (forces off).

    Uses ``compute_forces=False`` so logging/MBAR-style energy evals do not nest
    autodiff through PhysNet's internal ``value_and_grad``.

    ``cvs`` (preferred) or ``atom_pairs`` selects the collective variables; both
    accept ``(i, j)`` pairs and :class:`LinearDistanceCV` objects.
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

    specs = cvs if cvs is not None else atom_pairs
    if not specs:
        raise ValueError("make_packed_energy_fn requires cvs (or atom_pairs)")
    resolved_cvs = _as_cvs(specs)
    resolved_walls = tuple(FlatBottomWall.from_spec(w) for w in (walls or ()))
    for wall in resolved_walls:
        wall.cv.validate_against(n_atoms)
    for cv in resolved_cvs:
        cv.validate_against(n_atoms)
    targets = tuple(tuple(float(x) for x in row) for row in targets_per_cv)
    ks = tuple(tuple(float(x) for x in row) for row in k_per_cv)
    if len(resolved_cvs) != len(targets) or len(resolved_cvs) != len(ks):
        raise ValueError("cvs / targets_per_cv / k_per_cv length mismatch")
    for row in targets + ks:
        if len(row) != n_windows:
            raise ValueError("each CV target/k row must match graph n_windows")

    dst_idx = graph["dst_idx"]
    src_idx = graph["src_idx"]
    batch_segments = graph["batch_segments"]
    batch_mask = graph["batch_mask"]
    atom_mask = graph["atom_mask"]
    batch_size = graph["batch_size"]

    def _apply(position, *, compute_forces: bool):
        return model_apply(
            params,
            atomic_numbers=z_batched,
            positions=position,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=batch_size,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
            compute_forces=compute_forces,
        )

    def per_window_energy_fn(position, **kwargs):
        """Per-window ``E_ML + W`` (eV). Shape ``(K,)``."""
        del kwargs
        out = _apply(position, compute_forces=False)
        e_ml = jnp.asarray(out["energy"]).reshape(-1)
        e_bias = packed_bias_energies_nd(position, n_atoms, resolved_cvs, targets, ks)
        for wall in resolved_walls:
            e_bias = e_bias + wall.energy_batched(position, n_atoms, n_windows)
        return e_ml + e_bias

    def energy_sum_fn(position, **kwargs):
        del kwargs
        return jnp.sum(per_window_energy_fn(position))

    def force_fn(position, **kwargs):
        """ASE/jax-md forces ``F = -∇E`` for ML + umbrella bias."""
        del kwargs
        out = _apply(position, compute_forces=True)
        f_ml = jnp.asarray(out["forces"]).reshape(-1, 3)
        f_bias = packed_bias_forces_nd(position, n_atoms, resolved_cvs, targets, ks)
        for wall in resolved_walls:
            f_bias = f_bias + wall.forces_batched(position, n_atoms, n_windows)
        return f_ml + f_bias

    energy_sum_fn.force_fn = force_fn  # type: ignore[attr-defined]
    energy_sum_fn.per_window_energy_fn = per_window_energy_fn  # type: ignore[attr-defined]
    return energy_sum_fn


def make_packed_force_fn(
    *,
    model_apply: Callable[..., dict[str, Any]],
    params: Any,
    atomic_numbers: Any,
    graph: dict[str, Any],
    atom_pairs: Sequence[Any] | None = None,
    targets_per_cv: Sequence[Sequence[float]] = (),
    k_per_cv: Sequence[Sequence[float]] = (),
    cvs: Sequence[Any] | None = None,
    walls: Sequence[Any] | None = None,
) -> Callable[..., Any]:
    """Return packed force_fn for jax-md (avoids nested AD through PhysNet)."""
    energy_fn = make_packed_energy_fn(
        model_apply=model_apply,
        params=params,
        atomic_numbers=atomic_numbers,
        graph=graph,
        atom_pairs=atom_pairs,
        targets_per_cv=targets_per_cv,
        k_per_cv=k_per_cv,
        cvs=cvs,
        walls=walls,
    )
    return energy_fn.force_fn  # type: ignore[attr-defined]


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


def numpy_bias_matrix_cv(
    positions: np.ndarray,
    cv: Any,
    targets_A: Sequence[float],
    k_ev_A2: Sequence[float],
    *,
    cell: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic ``W_l(R)`` for one frame and one general CV. Shape ``(K,)``."""
    value = LinearDistanceCV.from_spec(cv).value_numpy(positions, cell=cell)
    targets = np.asarray(targets_A, dtype=np.float64)
    ks = np.asarray(k_ev_A2, dtype=np.float64)
    return 0.5 * ks * (value - targets) ** 2


def numpy_bias_matrix_nd(
    positions: np.ndarray,
    cvs: Sequence[Any],
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
    *,
    cell: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic multi-CV ``W_l(R)`` for one frame. Shape ``(K,)``."""
    total = None
    for dim, cv in enumerate(_as_cvs(cvs)):
        term = numpy_bias_matrix_cv(
            positions, cv, targets_per_cv[dim], k_per_cv[dim], cell=cell
        )
        total = term if total is None else total + term
    assert total is not None
    return total
