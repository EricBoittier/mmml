"""Conservative linear energy student used for tests, smoke, and equivalence.

Energy::

    E(x) = sum_i [ w · φ_i(R, Z) + b_{Z_i} ]

``φ_i`` is a translation-invariant descriptor (species one-hot + sorted
pairwise distances) that does **not** depend on ``w``.  For this readout,
``∇_w E = sum_i φ_i`` — pooled activations and energy output-gradients are
the same vector.  Forces come from ``F = -∇_R E`` so they stay conservative.

A teacher is the same architecture with different ``(w, b)``.  Loss gradients
use teacher labels, never the student's own predictions (which would be zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mmml.acquisition.pooling import pool_species_aware

jax.config.update("jax_enable_x64", True)


def _atomic_features(R, Z, mask, max_z: int, k_dist: int):
    n = R.shape[0]
    onehot = jax.nn.one_hot(Z.astype(jnp.int32), max_z + 1, dtype=R.dtype)
    diff = R[:, None, :] - R[None, :, :]
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)
    pair = mask[:, None] * mask[None, :]
    d = jnp.where(pair > 0, d, 1.0e6)
    d = d + jnp.eye(n, dtype=R.dtype) * 1.0e6
    # Always emit k_dist columns (pad with zeros when n_atoms is small).
    k = int(k_dist)
    padded = jnp.full((n, k), 1.0e6, dtype=R.dtype)
    take = min(k, n)
    padded = padded.at[:, :take].set(jnp.sort(d, axis=1)[:, :take])
    nearest = jnp.where(padded > 1.0e5, 0.0, padded)
    phi = jnp.concatenate([onehot, nearest], axis=-1)
    return phi * mask[:, None]


def feature_dim(max_z: int, k_dist: int) -> int:
    return int(max_z) + 1 + int(k_dist)


def _energy_from_params(w, b, R, Z, mask, max_z: int, k_dist: int):
    phi = _atomic_features(R, Z, mask, max_z, k_dist)
    e_atom = phi @ w + b[Z.astype(jnp.int32)]
    return jnp.sum(e_atom * mask)


def _forces_from_params(w, b, R, Z, mask, max_z: int, k_dist: int):
    e_fn = lambda pos: _energy_from_params(w, b, pos, Z, mask, max_z, k_dist)
    neg_f = jax.grad(e_fn)(R)
    return -neg_f * mask[:, None]


@dataclass
class LinearStudent:
    """Tiny conservative student with a linear energy readout."""

    weights: np.ndarray
    bias: np.ndarray
    max_z: int
    k_dist: int = 4
    name: str = "linear_student"

    def n_readout_params(self) -> int:
        return int(self.weights.size + self.bias.size)

    def pack_readout(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.weights, dtype=np.float64).ravel(),
             np.asarray(self.bias, dtype=np.float64).ravel()]
        )

    def with_readout(self, packed: np.ndarray) -> "LinearStudent":
        packed = np.asarray(packed, dtype=np.float64).ravel()
        nw = int(self.weights.size)
        return LinearStudent(
            weights=packed[:nw].reshape(self.weights.shape),
            bias=packed[nw:].reshape(self.bias.shape),
            max_z=self.max_z,
            k_dist=self.k_dist,
            name=self.name,
        )

    def _arrays(self, positions, atomic_numbers):
        R = jnp.asarray(positions, dtype=jnp.float64)
        Z = jnp.asarray(atomic_numbers, dtype=jnp.int32)
        if R.ndim != 2:
            raise ValueError("positions must be (n_atoms, 3)")
        mask = (Z > 0).astype(R.dtype)
        return R, Z, mask

    def energy_forces(self, positions, atomic_numbers) -> tuple[float, np.ndarray]:
        R, Z, mask = self._arrays(positions, atomic_numbers)
        w = jnp.asarray(self.weights, dtype=jnp.float64)
        b = jnp.asarray(self.bias, dtype=jnp.float64)
        e = _energy_from_params(w, b, R, Z, mask, self.max_z, self.k_dist)
        f = _forces_from_params(w, b, R, Z, mask, self.max_z, self.k_dist)
        return float(np.asarray(e)), np.asarray(f)

    def atomic_features(self, positions, atomic_numbers) -> np.ndarray:
        R, Z, mask = self._arrays(positions, atomic_numbers)
        phi = _atomic_features(R, Z, mask, self.max_z, self.k_dist)
        n = int((np.asarray(atomic_numbers) > 0).sum())
        return np.asarray(phi)[:n]

    def pooled_activations(self, positions, atomic_numbers, *, species=None) -> np.ndarray:
        phi = self.atomic_features(positions, atomic_numbers)
        z = np.asarray(atomic_numbers, dtype=np.int32)
        n = int((z > 0).sum())
        pooled = pool_species_aware([phi], [z[:n]], species=species, include_max=False)
        return pooled.vectors[0]

    def energy_jacobian(self, positions, atomic_numbers) -> np.ndarray:
        """``∇_w E`` concatenated with ``∇_b E`` (readout parameters)."""
        R, Z, mask = self._arrays(positions, atomic_numbers)
        w = jnp.asarray(self.weights, dtype=jnp.float64)
        b = jnp.asarray(self.bias, dtype=jnp.float64)

        def e_of_wb(ww, bb):
            return _energy_from_params(ww, bb, R, Z, mask, self.max_z, self.k_dist)

        gw, gb = jax.grad(e_of_wb, argnums=(0, 1))(w, b)
        return np.concatenate([np.asarray(gw).ravel(), np.asarray(gb).ravel()])

    def force_jacobian(self, positions, atomic_numbers) -> np.ndarray:
        """Rows are ``∇_readout F_{iα}`` for real atoms (shape ``(3N, P)``)."""
        R, Z, mask = self._arrays(positions, atomic_numbers)
        w = jnp.asarray(self.weights, dtype=jnp.float64)
        b = jnp.asarray(self.bias, dtype=jnp.float64)
        n = int(np.asarray(mask).sum())

        def f_flat(ww, bb):
            f = _forces_from_params(ww, bb, R, Z, mask, self.max_z, self.k_dist)
            return f[:n].reshape(-1)

        jw = jax.jacrev(f_flat, argnums=0)(w, b)
        jb = jax.jacrev(f_flat, argnums=1)(w, b)
        return np.concatenate(
            [np.asarray(jw).reshape(3 * n, -1), np.asarray(jb).reshape(3 * n, -1)],
            axis=1,
        )

    def loss_gradient(
        self,
        positions,
        atomic_numbers,
        *,
        energy_target: float,
        forces_target: np.ndarray,
        energy_weight: float,
        forces_weight: float,
    ) -> np.ndarray:
        """``∇_readout`` of the training-style E/F MSE vs *teacher* labels."""
        R, Z, mask = self._arrays(positions, atomic_numbers)
        w = jnp.asarray(self.weights, dtype=jnp.float64)
        b = jnp.asarray(self.bias, dtype=jnp.float64)
        e_t = jnp.asarray(energy_target, dtype=jnp.float64)
        f_t = jnp.asarray(forces_target, dtype=jnp.float64)
        a_e = jnp.asarray(energy_weight, dtype=jnp.float64)
        a_f = jnp.asarray(forces_weight, dtype=jnp.float64)

        def loss(ww, bb):
            e = _energy_from_params(ww, bb, R, Z, mask, self.max_z, self.k_dist)
            f = _forces_from_params(ww, bb, R, Z, mask, self.max_z, self.k_dist)
            n_real = jnp.maximum(mask.sum(), 1.0)
            e_loss = 0.5 * (e - e_t) ** 2
            diff = (f - f_t) * mask[:, None]
            f_loss = 0.5 * jnp.sum(diff * diff) / n_real
            return a_e * e_loss + a_f * f_loss

        gw, gb = jax.grad(loss, argnums=(0, 1))(w, b)
        return np.concatenate([np.asarray(gw).ravel(), np.asarray(gb).ravel()])

    def information_block(
        self,
        positions,
        atomic_numbers,
        *,
        energy_weight: float,
        forces_weight: float,
        include_forces: bool = True,
    ) -> np.ndarray:
        """Weighted Jacobian block matching the fine-tuning loss scaling.

        ``J_x = [ √α_E ∇_w E^T ; √α_F ∇_w F_1^T ; ... ]`` with
        ``α_E = energy_weight`` and per-force-component
        ``α_F = forces_weight / N_atoms`` (the training force reduction).
        """
        gE = self.energy_jacobian(positions, atomic_numbers)
        rows = [np.sqrt(max(float(energy_weight), 0.0)) * gE]
        if include_forces:
            z = np.asarray(atomic_numbers)
            n = int((z > 0).sum())
            alpha_f = float(forces_weight) / max(n, 1)
            jF = self.force_jacobian(positions, atomic_numbers)
            rows.append(np.sqrt(max(alpha_f, 0.0)) * jF)
        return np.vstack(rows)

    def fingerprint(self) -> dict[str, Any]:
        w = np.asarray(self.weights)
        b = np.asarray(self.bias)
        return {
            "kind": "linear_student",
            "name": self.name,
            "max_z": int(self.max_z),
            "k_dist": int(self.k_dist),
            "n_readout_params": self.n_readout_params(),
            "weight_norm": float(np.linalg.norm(w)),
            "bias_norm": float(np.linalg.norm(b)),
            "weight_mean": float(w.mean()),
        }


def init_linear_student(
    *,
    max_z: int = 8,
    k_dist: int = 4,
    seed: int = 0,
    scale: float = 0.05,
    name: str = "linear_student",
) -> LinearStudent:
    rng = np.random.default_rng(int(seed))
    dim = feature_dim(max_z, k_dist)
    w = rng.normal(0.0, scale, size=(dim,))
    b = rng.normal(0.0, scale, size=(max_z + 1,))
    return LinearStudent(weights=w, bias=b, max_z=max_z, k_dist=k_dist, name=name)


def init_linear_teacher(student: LinearStudent, *, seed: int = 1, scale: float = 0.08) -> LinearStudent:
    rng = np.random.default_rng(int(seed))
    w = np.asarray(student.weights) + rng.normal(0.0, scale, size=student.weights.shape)
    b = np.asarray(student.bias) + rng.normal(0.0, 0.5 * scale, size=student.bias.shape)
    return LinearStudent(
        weights=w, bias=b, max_z=student.max_z, k_dist=student.k_dist, name="linear_teacher"
    )
