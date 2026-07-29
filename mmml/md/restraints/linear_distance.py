"""Linear combinations of interatomic distances as collective variables.

A bare distance ``r(i, j)`` is the one-term special case; an antisymmetric
stretch such as ``xi = r(C-Cl) - r(C-N)`` is two terms with opposite-sign
coefficients. Flat-bottom walls on a sum of the same distances confine the
system so a difference CV cannot escape by dissociation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _as_pair(pair: Sequence[int]) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError(f"atom pair must have length 2 (got {pair!r})")
    i, j = int(pair[0]), int(pair[1])
    if i == j or i < 0 or j < 0:
        raise ValueError(f"atom pair requires two distinct non-negative indices (got {i}, {j})")
    return (i, j)


def _mic_disp_numpy(a: Any, b: Any, cell: Any | None) -> Any:
    import numpy as np

    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if cell is None:
        return d
    cell_arr = np.asarray(cell, dtype=np.float64)
    if cell_arr.shape == (3,):
        # Orthorhombic lengths
        box = cell_arr
        return d - box * np.round(d / box)
    if cell_arr.shape == (3, 3):
        inv = np.linalg.inv(cell_arr.T)
        frac = inv @ d
        frac = frac - np.round(frac)
        return cell_arr.T @ frac
    raise ValueError(f"cell must be shape (3,) or (3, 3), got {cell_arr.shape}")


def harmonic_bias_energy(value: Any, target: float, k_ev_A2: float) -> Any:
    """Harmonic umbrella bias ``0.5 * k * (ξ - ξ₀)²`` (eV)."""
    import jax.numpy as jnp

    return 0.5 * float(k_ev_A2) * jnp.square(value - float(target))


@dataclass(frozen=True)
class LinearDistanceCV:
    """Collective variable ``ξ = Σ_p c_p * r(i_p, j_p)`` (Å)."""

    pairs: tuple[tuple[int, int], ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("LinearDistanceCV requires at least one atom pair")
        if len(self.pairs) != len(self.coefficients):
            raise ValueError(
                f"pairs ({len(self.pairs)}) and coefficients "
                f"({len(self.coefficients)}) length mismatch"
            )
        object.__setattr__(
            self,
            "pairs",
            tuple(_as_pair(p) for p in self.pairs),
        )
        object.__setattr__(
            self,
            "coefficients",
            tuple(float(c) for c in self.coefficients),
        )

    @classmethod
    def distance(cls, atom_i: int, atom_j: int) -> LinearDistanceCV:
        """Plain interatomic distance ``r(i, j)``."""
        return cls(pairs=(_as_pair((atom_i, atom_j)),), coefficients=(1.0,))

    @classmethod
    def difference(
        cls,
        pair_a: Sequence[int],
        pair_b: Sequence[int],
    ) -> LinearDistanceCV:
        """Antisymmetric stretch ``ξ = r(a) - r(b)``."""
        return cls(
            pairs=(_as_pair(pair_a), _as_pair(pair_b)),
            coefficients=(1.0, -1.0),
        )

    @classmethod
    def from_spec(cls, spec: Any) -> LinearDistanceCV:
        """Accept a CV instance, ``(i, j)`` pair, or ``{pairs, coefficients}`` map."""
        if isinstance(spec, LinearDistanceCV):
            return spec
        if isinstance(spec, Mapping):
            pairs = spec.get("pairs")
            coeffs = spec.get("coefficients")
            if pairs is None or coeffs is None:
                raise ValueError(
                    "LinearDistanceCV mapping spec requires 'pairs' and 'coefficients'"
                )
            return cls(
                pairs=tuple(_as_pair(p) for p in pairs),
                coefficients=tuple(float(c) for c in coeffs),
            )
        if isinstance(spec, Sequence) and len(spec) == 2 and not isinstance(spec, (str, bytes)):
            # Bare (i, j) pair — but not a nested pair-list of length 2 without coeffs.
            if all(isinstance(x, (int, float)) for x in spec):
                return cls.distance(int(spec[0]), int(spec[1]))
        raise TypeError(
            "LinearDistanceCV.from_spec expects a LinearDistanceCV, (i, j) pair, "
            f"or {{'pairs', 'coefficients'}} mapping; got {type(spec).__name__}"
        )

    def validate_against(self, n_atoms: int) -> None:
        """Raise if any atom index is out of range for an ``n_atoms`` system."""
        n = int(n_atoms)
        if n < 1:
            raise ValueError(f"n_atoms must be >= 1 (got {n})")
        for i, j in self.pairs:
            if i >= n or j >= n:
                raise ValueError(
                    f"CV atom index out of range for n_atoms={n}: pair ({i}, {j}) "
                    f"in {self.label()}"
                )

    def label(self) -> str:
        """Human-readable expression, e.g. ``r(0,1)`` or ``1.0*r(2,0)+-1.0*r(2,1)``."""
        if len(self.pairs) == 1 and self.coefficients[0] == 1.0:
            i, j = self.pairs[0]
            return f"r({i},{j})"
        parts = [
            f"{c:g}*r({i},{j})" for (i, j), c in zip(self.pairs, self.coefficients)
        ]
        return "+".join(parts)

    def value_numpy(
        self,
        positions: Any,
        *,
        cell: Any | None = None,
    ) -> float:
        """Evaluate ``ξ`` for one frame (Å)."""
        import numpy as np

        r = np.asarray(positions, dtype=np.float64)
        total = 0.0
        for (i, j), c in zip(self.pairs, self.coefficients):
            disp = _mic_disp_numpy(r[i], r[j], cell)
            total += float(c) * float(np.linalg.norm(disp))
        return total

    def value_batched(
        self,
        positions_packed: Any,
        n_atoms: int,
        n_windows: int,
    ) -> Any:
        """CV value for each packed window copy. Shape ``(K,)``."""
        import jax.numpy as jnp

        pos = positions_packed.reshape(int(n_windows), int(n_atoms), 3)
        total = jnp.zeros((int(n_windows),), dtype=pos.dtype)
        for (i, j), c in zip(self.pairs, self.coefficients):
            disp = pos[:, j, :] - pos[:, i, :]
            dist = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1e-12)
            total = total + float(c) * dist
        return total

    def gradient_batched(
        self,
        positions_packed: Any,
        n_atoms: int,
        n_windows: int,
    ) -> Any:
        """Analytic ``∇ξ`` per window. Shape ``(K, N, 3)``.

        For each pair term ``c * r(i, j)`` with unit vector ``u = (r_j - r_i)/r``:
        ``∇_i = -c u``, ``∇_j = +c u``.
        """
        import jax.numpy as jnp

        k = int(n_windows)
        n = int(n_atoms)
        pos = positions_packed.reshape(k, n, 3)
        grad = jnp.zeros_like(pos)
        for (i, j), c in zip(self.pairs, self.coefficients):
            disp = pos[:, j, :] - pos[:, i, :]
            dist = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1e-12)
            u = disp / dist[:, None]
            coef = float(c)
            grad = grad.at[:, i, :].add(-coef * u)
            grad = grad.at[:, j, :].add(coef * u)
        return grad


@dataclass(frozen=True)
class FlatBottomWall:
    """One-sided (or two-sided) harmonic wall on a :class:`LinearDistanceCV`.

    Energy is zero while ``lower ≤ ξ ≤ upper`` (bounds that are ``None`` are
    inactive). Outside the flat region:

    * upper: ``0.5 * k * (ξ - upper)²`` when ``ξ > upper``
    * lower: ``0.5 * k * (lower - ξ)²`` when ``ξ < lower``
    """

    cv: LinearDistanceCV
    k: float = 50.0
    upper: float | None = None
    lower: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cv, LinearDistanceCV):
            object.__setattr__(self, "cv", LinearDistanceCV.from_spec(self.cv))
        if float(self.k) < 0:
            raise ValueError("wall force constant must be non-negative")
        if self.upper is not None and self.lower is not None:
            if float(self.lower) > float(self.upper):
                raise ValueError(
                    f"wall lower ({self.lower}) must be <= upper ({self.upper})"
                )
        if self.upper is None and self.lower is None:
            raise ValueError("FlatBottomWall requires at least one of lower/upper")

    @classmethod
    def from_spec(cls, spec: Any) -> FlatBottomWall:
        """Accept a wall instance or a mapping (``cv`` / ``pairs``+``coefficients``)."""
        if isinstance(spec, FlatBottomWall):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(
                "FlatBottomWall.from_spec expects a FlatBottomWall or mapping; "
                f"got {type(spec).__name__}"
            )
        if "cv" in spec and spec["cv"] is not None:
            cv = LinearDistanceCV.from_spec(spec["cv"])
        else:
            cv = LinearDistanceCV.from_spec(
                {"pairs": spec["pairs"], "coefficients": spec["coefficients"]}
            )
        return cls(
            cv=cv,
            k=float(spec.get("k", 50.0)),
            upper=None if spec.get("upper") is None else float(spec["upper"]),
            lower=None if spec.get("lower") is None else float(spec["lower"]),
        )

    def to_spec(self) -> dict[str, Any]:
        """JSON-serialisable description."""
        out: dict[str, Any] = {
            "cv": {
                "pairs": [list(p) for p in self.cv.pairs],
                "coefficients": list(self.cv.coefficients),
            },
            "k": float(self.k),
        }
        if self.upper is not None:
            out["upper"] = float(self.upper)
        if self.lower is not None:
            out["lower"] = float(self.lower)
        return out

    def label(self) -> str:
        bounds = []
        if self.lower is not None:
            bounds.append(f"ξ>={self.lower:g}")
        if self.upper is not None:
            bounds.append(f"ξ<={self.upper:g}")
        return f"wall({self.cv.label()}; {', '.join(bounds)}; k={self.k:g})"

    def _violation(self, value: Any) -> Any:
        """Signed overshoot used by energy/forces (JAX or NumPy array-compatible)."""
        import jax.numpy as jnp

        v = jnp.asarray(value)
        pen = jnp.zeros_like(v)
        if self.upper is not None:
            pen = pen + jnp.maximum(v - float(self.upper), 0.0)
        if self.lower is not None:
            pen = pen + jnp.maximum(float(self.lower) - v, 0.0)
        return pen

    def energy_batched(
        self,
        positions_packed: Any,
        n_atoms: int,
        n_windows: int,
    ) -> Any:
        """Per-window wall energy. Shape ``(K,)``."""
        values = self.cv.value_batched(positions_packed, n_atoms, n_windows)
        overshoot = self._violation(values)
        return 0.5 * float(self.k) * overshoot * overshoot

    def forces_batched(
        self,
        positions_packed: Any,
        n_atoms: int,
        n_windows: int,
    ) -> Any:
        """ASE-style wall forces ``F = -∇W``. Shape ``(K*N, 3)``.

        ``W = 0.5 k s²`` with ``s = max(0, ξ-upper) + max(0, lower-ξ)``.
        When only one side is active, ``dW/dξ = k * (ξ - bound)`` outside and
        zero inside; with both sides the two overshoots are mutually exclusive.
        """
        import jax.numpy as jnp

        k_win = int(n_windows)
        n = int(n_atoms)
        values = self.cv.value_batched(positions_packed, n_atoms, k_win)
        grad = self.cv.gradient_batched(positions_packed, n_atoms, k_win)
        dW_dxi = jnp.zeros_like(values)
        if self.upper is not None:
            dW_dxi = dW_dxi + float(self.k) * jnp.maximum(
                values - float(self.upper), 0.0
            )
        if self.lower is not None:
            dW_dxi = dW_dxi - float(self.k) * jnp.maximum(
                float(self.lower) - values, 0.0
            )
        # F = -∇W = -(dW/dξ) ∇ξ
        forces = (-dW_dxi[:, None, None]) * grad
        return forces.reshape(k_win * n, 3)


def linear_cvs_from_pairs(
    pairs: Sequence[Sequence[int]],
) -> tuple[LinearDistanceCV, ...]:
    """Promote a sequence of ``(i, j)`` pairs to plain-distance CVs."""
    return tuple(LinearDistanceCV.distance(int(i), int(j)) for i, j in pairs)
