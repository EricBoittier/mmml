"""Periodic dihedral collective variable for umbrella sampling.

Values and window targets are in **degrees** (matching the TRIA φ/ψ scan
convention). Force constants are **eV/deg²**. The bias uses the periodic
shortest arc ``atan2(sin Δ, cos Δ)`` so windows near ±180° wrap correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["DihedralCV", "periodic_delta_deg", "harmonic_bias_energy_periodic_deg"]

_EPS = 1e-12


def periodic_delta_deg(value: Any, target: Any) -> Any:
    """Smallest signed angle difference ``value - target`` in degrees."""
    import jax.numpy as jnp

    d = jnp.deg2rad(value - target)
    return jnp.rad2deg(jnp.arctan2(jnp.sin(d), jnp.cos(d)))


def harmonic_bias_energy_periodic_deg(value: Any, target: float, k_ev_deg2: float) -> Any:
    """``0.5 * k * Δφ²`` with periodic Δφ in degrees (eV)."""
    import jax.numpy as jnp

    delta = periodic_delta_deg(value, float(target))
    return 0.5 * float(k_ev_deg2) * jnp.square(delta)


def _dihedral_angle_rad(r, atom_indices, xp):
    """Signed dihedral in radians (numpy or jax arrays)."""
    i0, i1, i2, i3 = (int(x) for x in atom_indices)
    p0, p1, p2, p3 = r[i0], r[i1], r[i2], r[i3]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (xp.linalg.norm(b1) + _EPS)
    v = b0 - xp.dot(b0, b1n) * b1n
    w = b2 - xp.dot(b2, b1n) * b1n
    x = xp.dot(v, w)
    y = xp.dot(xp.cross(b1n, v), w)
    return xp.arctan2(y, x)


@dataclass(frozen=True)
class DihedralCV:
    """Four-atom dihedral CV ``φ(i–j–k–l)`` in degrees."""

    atoms: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.atoms) != 4 or len(set(self.atoms)) != 4 or min(self.atoms) < 0:
            raise ValueError(
                f"DihedralCV needs four distinct non-negative indices; got {self.atoms}"
            )

    @classmethod
    def from_spec(cls, spec: Any) -> "DihedralCV":
        if isinstance(spec, DihedralCV):
            return spec
        if isinstance(spec, dict):
            atoms = spec.get("atoms") or spec.get("indices") or spec.get("dihedral")
            if atoms is None:
                raise ValueError(f"dihedral CV mapping needs 'atoms'; got {spec!r}")
            return cls(atoms=tuple(int(x) for x in atoms))
        atoms = tuple(int(x) for x in spec)
        if len(atoms) != 4:
            raise ValueError(f"cannot build DihedralCV from {spec!r}")
        return cls(atoms=atoms)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Legacy ``WindowSchedule.atom_pairs`` stub (first two atoms)."""
        return ((self.atoms[0], self.atoms[1]),)

    @property
    def is_plain_distance(self) -> bool:
        return False

    @property
    def is_periodic(self) -> bool:
        return True

    @property
    def max_atom_index(self) -> int:
        return max(self.atoms)

    def validate_against(self, n_atoms: int) -> None:
        if self.max_atom_index >= n_atoms:
            raise ValueError(
                f"DihedralCV references atom {self.max_atom_index} "
                f"but the system has {n_atoms} atoms"
            )

    def label(self) -> str:
        a, b, c, d = self.atoms
        return f"φ({a}-{b}-{c}-{d})"

    def value(self, positions: Any, *, cell: Any = None) -> Any:
        """Dihedral in degrees for one ``(N, 3)`` frame."""
        import jax.numpy as jnp

        del cell  # intramolecular; MIC not applied to backbone torsions
        return jnp.rad2deg(_dihedral_angle_rad(positions, self.atoms, jnp))

    def value_numpy(self, positions: Any, *, cell: Any = None) -> float:
        import numpy as np

        del cell
        r = np.asarray(positions, dtype=np.float64)
        return float(np.rad2deg(_dihedral_angle_rad(r, self.atoms, np)))

    def value_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """Per-window dihedral in degrees. Shape ``(K,)``."""
        import jax.numpy as jnp

        pos = positions.reshape(n_windows, n_atoms, 3)
        i0, i1, i2, i3 = self.atoms
        p0, p1, p2, p3 = pos[:, i0], pos[:, i1], pos[:, i2], pos[:, i3]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1n = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + _EPS)
        v = b0 - jnp.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
        w = b2 - jnp.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1n, v) * w, axis=-1)
        return jnp.rad2deg(jnp.arctan2(y, x))

    def gradient_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """Analytic ``∂φ/∂R`` in deg/Å. Shape ``(K, N, 3)``.

        Uses reverse-mode AD on the scalar dihedral (cheap four-atom graph).
        """
        import jax
        import jax.numpy as jnp

        pos = positions.reshape(n_windows, n_atoms, 3)

        def _one(frame):
            return self.value(frame)

        # vmap over windows
        vals, grads = jax.vmap(jax.value_and_grad(_one))(pos)
        del vals
        return grads


def cv_from_spec(spec: Any) -> Any:
    """Dispatch ``LinearDistanceCV`` vs ``DihedralCV`` from a YAML/Python spec."""
    from mmml.md.restraints.linear_distance import LinearDistanceCV

    if isinstance(spec, (DihedralCV, LinearDistanceCV)):
        return spec
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "")).strip().lower()
        if kind in {"dihedral", "torsion", "phi", "psi"} or "atoms" in spec or "dihedral" in spec:
            if "pairs" in spec and "atoms" not in spec and "dihedral" not in spec:
                return LinearDistanceCV.from_spec(spec)
            return DihedralCV.from_spec(spec)
        return LinearDistanceCV.from_spec(spec)
    # bare sequence
    try:
        seq = tuple(int(x) for x in spec)
    except TypeError as exc:
        raise ValueError(f"cannot build CV from {spec!r}") from exc
    if len(seq) == 4:
        return DihedralCV(atoms=seq)
    if len(seq) == 2:
        return LinearDistanceCV.distance(seq[0], seq[1])
    raise ValueError(f"cannot build CV from {spec!r}")
