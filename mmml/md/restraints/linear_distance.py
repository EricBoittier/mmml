"""Linear-combination-of-distances collective variable.

Generalises the single-pair :class:`~mmml.md.restraints.distance.DistanceRestraint`
CV to ``xi(R) = sum_d c_d * |r_{j_d} - r_{i_d}|``. The motivating case is the SN2
antisymmetric stretch used as the Menshutkin reaction coordinate,
``xi = r(C-X) - r(C-N)`` (Turan, Brickel & Meuwly, *J. Phys. Chem. B* **126**,
1951 (2022)), but plain distances (one pair, ``c = 1``) are the degenerate case,
so the same object drives both the gas-phase packed umbrella sampler
(:mod:`mmml.umbrella`) and the solvated ``rxncoor`` energy term. Keeping one
implementation is the point: a CV that disagrees between the two paths silently
produces two incomparable free-energy profiles.

Analytic gradients are provided because the packed sampler applies bias forces
explicitly rather than differentiating the total energy -- autodiff through
PhysNet's internal ``value_and_grad`` nests badly (see
``mmml.umbrella.sample.run_umbrella_nvt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "AngleWall",
    "BondRetentionWall",
    "FlatBottomWall",
    "LinearDistanceCV",
    "harmonic_bias_energy",
]

_EPS = 1e-12
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
    """``xi(R) = sum_d c_d * |r_{j_d} - r_{i_d}|`` over ``pairs`` / ``coefficients``.

    Atom indices may repeat across pairs (the SN2 CV shares the carbon between
    both distances); gradient contributions accumulate.
    """

    pairs: tuple[tuple[int, int], ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("LinearDistanceCV requires at least one atom pair")
        if len(self.pairs) != len(self.coefficients):
            raise ValueError(
                f"pairs ({len(self.pairs)}) and coefficients "
                f"({len(self.coefficients)}) must have the same length"
            )
        for i, j in self.pairs:
            if i == j or min(i, j) < 0:
                raise ValueError(
                    f"each pair needs two distinct non-negative atom indices; got ({i}, {j})"
                )
        if all(c == 0.0 for c in self.coefficients):
            raise ValueError("LinearDistanceCV needs at least one non-zero coefficient")

    # --- constructors -------------------------------------------------------

    @classmethod
    def distance(cls, i: int, j: int) -> "LinearDistanceCV":
        """Plain interatomic distance ``|r_j - r_i|``."""
        return cls(pairs=((int(i), int(j)),), coefficients=(1.0,))

    @classmethod
    def difference(
        cls, minuend: tuple[int, int], subtrahend: tuple[int, int]
    ) -> "LinearDistanceCV":
        """``|r_minuend| - |r_subtrahend|`` -- the antisymmetric-stretch SN2 CV.

        For the Menshutkin reaction pass ``minuend=(C, X)`` and
        ``subtrahend=(C, N)`` so reactants sit at negative ``xi`` and products at
        positive ``xi``, matching the Turan et al. convention.
        """
        return cls(
            pairs=(
                (int(minuend[0]), int(minuend[1])),
                (int(subtrahend[0]), int(subtrahend[1])),
            ),
            coefficients=(1.0, -1.0),
        )

    @classmethod
    def from_spec(cls, spec: Any) -> "LinearDistanceCV":
        """Build from an existing instance, a ``(i, j)`` pair, or a mapping.

        Accepts ``{"pairs": [[0, 2], [2, 1]], "coefficients": [1, -1]}`` so CVs
        can be declared in the YAML campaign configs.
        """
        if isinstance(spec, LinearDistanceCV):
            return spec
        if isinstance(spec, dict):
            return cls(
                pairs=tuple((int(a), int(b)) for a, b in spec["pairs"]),
                coefficients=tuple(float(c) for c in spec["coefficients"]),
            )
        pair = tuple(int(x) for x in spec)
        if len(pair) != 2:
            raise ValueError(f"cannot build a LinearDistanceCV from {spec!r}")
        return cls.distance(pair[0], pair[1])

    # --- properties ---------------------------------------------------------

    @property
    def atom_indices(self) -> tuple[int, ...]:
        """Every atom index the CV touches, ascending and deduplicated."""
        return tuple(sorted({idx for pair in self.pairs for idx in pair}))

    @property
    def is_plain_distance(self) -> bool:
        """True when this is a single pair with unit coefficient."""
        return len(self.pairs) == 1 and self.coefficients[0] == 1.0

    @property
    def max_atom_index(self) -> int:
        return max(idx for pair in self.pairs for idx in pair)

    def validate_against(self, n_atoms: int) -> None:
        if self.max_atom_index >= n_atoms:
            raise ValueError(
                f"CV references atom index {self.max_atom_index} "
                f"but the system has {n_atoms} atoms"
            )

    def label(self) -> str:
        """Compact human-readable form, e.g. ``r(1-0) - r(1-2)``."""
        out = ""
        for (i, j), c in zip(self.pairs, self.coefficients, strict=True):
            magnitude = abs(c)
            term = f"r({i}-{j})" if magnitude == 1.0 else f"{magnitude:g}*r({i}-{j})"
            if not out:
                out = term if c > 0 else f"-{term}"
            else:
                out += f" + {term}" if c > 0 else f" - {term}"
        return out

    # --- evaluation ---------------------------------------------------------

    def _displacements(self, positions, cell, xp):
        """Per-pair ``r_j - r_i``, minimum-image when ``cell`` is given."""
        if cell is None:
            return [positions[j] - positions[i] for i, j in self.pairs]
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

        return [mic_displacement(positions[i], positions[j], cell) for i, j in self.pairs]

    def value(self, positions: Any, *, cell: Any = None) -> Any:
        """CV value for one ``(N, 3)`` frame. Works with numpy or jax arrays."""
        import jax.numpy as jnp

        total = None
        for disp, c in zip(
            self._displacements(positions, cell, jnp), self.coefficients, strict=True
        ):
            r = jnp.sqrt(jnp.sum(disp * disp) + _EPS)
            term = c * r
            total = term if total is None else total + term
        return total

    def value_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """CV per window for a packed ``(K*N, 3)`` array. Shape ``(K,)``.

        The packed layout is the one :func:`mmml.umbrella.energy.build_packed_graph`
        produces: ``K`` tiled copies of an ``N``-atom system, window-major.
        """
        import jax.numpy as jnp

        pos = positions.reshape(n_windows, n_atoms, 3)
        total = None
        for (i, j), c in zip(self.pairs, self.coefficients, strict=True):
            disp = pos[:, j, :] - pos[:, i, :]
            r = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + _EPS)
            term = c * r
            total = term if total is None else total + term
        return total

    def gradient_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """``d(xi)/dR`` per window for a packed ``(K*N, 3)`` array. Shape ``(K, N, 3)``.

        Analytic, so bias forces never differentiate through the ML model.
        """
        import jax.numpy as jnp

        pos = positions.reshape(n_windows, n_atoms, 3)
        grad = jnp.zeros_like(pos)
        for (i, j), c in zip(self.pairs, self.coefficients, strict=True):
            disp = pos[:, j, :] - pos[:, i, :]
            r = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + _EPS)
            unit = disp / r[:, None]
            grad = grad.at[:, j, :].add(c * unit)
            grad = grad.at[:, i, :].add(-c * unit)
        return grad

    def value_numpy(self, positions, *, cell=None) -> float:
        """Host-side CV value for one frame (no jax), for analysis and MBAR."""
        import numpy as np

        r = np.asarray(positions, dtype=np.float64)
        total = 0.0
        for (i, j), c in zip(self.pairs, self.coefficients, strict=True):
            disp = r[j] - r[i]
            if cell is not None:
                box = np.asarray(cell, dtype=np.float64)
                lengths = np.diag(box) if box.ndim == 2 else box
                disp = disp - lengths * np.round(disp / lengths)
            total += c * float(np.linalg.norm(disp))
        return total

    def values_numpy(self, positions, *, cell=None):
        """Vectorised CV over a ``(T, N, 3)`` trajectory. Shape ``(T,)``."""
        import numpy as np

        r = np.asarray(positions, dtype=np.float64)
        if r.ndim != 3:
            raise ValueError(f"expected (T, N, 3) positions; got {r.shape}")
        total = np.zeros(r.shape[0], dtype=np.float64)
        for (i, j), c in zip(self.pairs, self.coefficients, strict=True):
            disp = r[:, j, :] - r[:, i, :]
            if cell is not None:
                box = np.asarray(cell, dtype=np.float64)
                lengths = np.diag(box) if box.ndim == 2 else box
                disp = disp - lengths * np.round(disp / lengths)
            total += c * np.linalg.norm(disp, axis=-1)
        return total


@dataclass(frozen=True)
class FlatBottomWall:
    """One- or two-sided flat-bottom restraint on a :class:`LinearDistanceCV`.

    Zero inside ``[lower, upper]`` and harmonic outside, so it does not bias the
    region being sampled and only prevents escape.

    This is not optional for a reactive umbrella run on a fitted surface. An
    antisymmetric stretch ``xi = r(C-X) - r(C-N)`` is degenerate: ``(1.8, 3.0)``
    and ``(8.1, 7.3)`` give nearly the same ``xi``, but the second is a
    dissociated system. On a physical potential the second costs bond energy and
    is never visited; on a neural-network fit the energy outside the training
    manifold is unbounded below, so the dissociated branch is *downhill* and the
    trajectory falls into it, converting hundreds of eV of spurious potential
    energy into kinetic energy. Walling the *sum* ``r(C-X) + r(C-N)`` removes the
    degenerate direction while leaving the reaction path untouched.
    """

    cv: "LinearDistanceCV"
    upper: float | None = None
    lower: float | None = None
    k: float = 50.0

    def __post_init__(self) -> None:
        if self.upper is None and self.lower is None:
            raise ValueError("FlatBottomWall needs at least one of lower/upper")
        if self.upper is not None and self.lower is not None and self.lower >= self.upper:
            raise ValueError(
                f"wall lower ({self.lower}) must be below upper ({self.upper})"
            )
        if self.k < 0:
            raise ValueError(f"wall force constant must be non-negative (got {self.k})")

    @classmethod
    def from_spec(cls, spec: Any) -> "FlatBottomWall":
        """Build from an instance or a ``{"cv": ..., "upper": ..., "k": ...}`` mapping."""
        if isinstance(spec, FlatBottomWall):
            return spec
        data = dict(spec)
        return cls(
            cv=LinearDistanceCV.from_spec(data["cv"]),
            upper=None if data.get("upper") is None else float(data["upper"]),
            lower=None if data.get("lower") is None else float(data["lower"]),
            k=float(data.get("k", 50.0)),
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "cv": {
                "pairs": [list(p) for p in self.cv.pairs],
                "coefficients": list(self.cv.coefficients),
            },
            "upper": self.upper,
            "lower": self.lower,
            "k": self.k,
        }

    def label(self) -> str:
        bounds = []
        if self.lower is not None:
            bounds.append(f">{self.lower:g}")
        if self.upper is not None:
            bounds.append(f"<{self.upper:g}")
        return f"{self.cv.label()} {' and '.join(bounds)} (k={self.k:g})"

    def _penalty(self, value, xp):
        """``0.5 k * excess^2`` where ``excess`` is the distance outside the box."""
        excess = 0.0
        if self.upper is not None:
            excess = excess + xp.maximum(value - self.upper, 0.0)
        if self.lower is not None:
            excess = excess + xp.minimum(value - self.lower, 0.0)
        return 0.5 * self.k * excess * excess

    def validate_against(self, n_atoms: int) -> None:
        """Delegate to the CV, so both wall kinds share one interface."""
        self.cv.validate_against(n_atoms)

    def energy(self, positions: Any, *, cell: Any = None) -> Any:
        import jax.numpy as jnp

        return self._penalty(self.cv.value(positions, cell=cell), jnp)

    def energy_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """Wall energy per packed window. Shape ``(K,)``."""
        import jax.numpy as jnp

        return self._penalty(self.cv.value_batched(positions, n_atoms, n_windows), jnp)

    def forces_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """``F = -grad W`` per packed window, flattened to ``(K*N, 3)``."""
        import jax.numpy as jnp

        value = self.cv.value_batched(positions, n_atoms, n_windows)
        grad = self.cv.gradient_batched(positions, n_atoms, n_windows)
        excess = jnp.zeros_like(value)
        if self.upper is not None:
            excess = excess + jnp.maximum(value - self.upper, 0.0)
        if self.lower is not None:
            excess = excess + jnp.minimum(value - self.lower, 0.0)
        scale = (self.k * excess)[:, None, None]
        return (-scale * grad).reshape(n_windows * n_atoms, 3)

    def value_numpy(self, positions, *, cell=None) -> float:
        return self.cv.value_numpy(positions, cell=cell)


def harmonic_bias_energy(cv_value: Any, target: float, k: float) -> Any:
    """``0.5 * k * (xi - xi0)^2``.

    Units follow the caller: eV/A^2 in the jax paths, kcal/mol/A^2 in the
    CHARMM-facing ones. Nothing here converts.
    """
    import jax.numpy as jnp

    return 0.5 * float(k) * jnp.square(cv_value - float(target))


def linear_cvs_from_pairs(
    atom_pairs: Sequence[tuple[int, int]],
) -> tuple[LinearDistanceCV, ...]:
    """Legacy shim: one plain-distance CV per ``(i, j)`` pair."""
    return tuple(LinearDistanceCV.distance(int(i), int(j)) for i, j in atom_pairs)


@dataclass(frozen=True)
class BondRetentionWall:
    """Keep a transferring group bonded to at least one of its partners.

    For a transfer coordinate ``xi = r(C-X) - r(C-N)`` the umbrella constrains
    only the difference, so the transferring group can drift away from *both*
    partners while xi sits exactly on its target. On a fitted potential that
    region is off the training manifold and typically unbounded below, so the
    trajectory falls into it.

    :class:`FlatBottomWall` on the sum ``r(C-X) + r(C-N)`` addresses the same
    degeneracy but needs a xi-dependent bound: the allowed sum is large where
    one bond is long and small near the transition state, so a single global
    bound is either useless at one end or wrong at the other. This restrains
    ``min(r_1, ..., r_n)`` instead, which is what the chemistry actually
    requires -- the group stays bonded to *something* -- and is close to
    constant in xi.

    Measured for NH3 + CH3Cl: across the whole training set
    ``min(r(C-Cl), r(C-N))`` has median 1.75 A, p99 2.03 A and max 2.18 A, with
    no systematic xi dependence, while the configuration that diverged sat at
    2.57 A.

    Zero while the shortest distance is within ``r_max``, harmonic beyond, so it
    does not bias sampling inside the physically bonded region.
    """

    pairs: tuple[tuple[int, int], ...]
    r_max: float
    k: float = 50.0

    def __post_init__(self) -> None:
        if len(self.pairs) < 2:
            raise ValueError(
                f"BondRetentionWall needs at least two competing pairs "
                f"(got {len(self.pairs)}); with one pair use a plain distance wall"
            )
        if self.r_max <= 0:
            raise ValueError(f"r_max must be positive (got {self.r_max})")
        if self.k < 0:
            raise ValueError(f"force constant must be non-negative (got {self.k})")

    @classmethod
    def from_spec(cls, spec: Any) -> "BondRetentionWall":
        if isinstance(spec, cls):
            return spec
        d = dict(spec)
        return cls(
            pairs=tuple((int(i), int(j)) for i, j in d["pairs"]),
            r_max=float(d["r_max"]),
            k=float(d.get("k", 50.0)),
        )

    def to_spec(self) -> dict[str, Any]:
        """Round-trips through :meth:`from_spec`; ``r_max`` is what marks the kind."""
        return {
            "pairs": [list(p) for p in self.pairs],
            "r_max": self.r_max,
            "k": self.k,
        }

    def label(self) -> str:
        """Compact form, e.g. ``min(r(2-0), r(2-1)) <= 2.25``."""
        inner = ", ".join(f"r({i}-{j})" for i, j in self.pairs)
        return f"min({inner}) <= {self.r_max:g}"

    def validate_against(self, n_atoms: int) -> None:
        for i, j in self.pairs:
            for a in (i, j):
                if not 0 <= a < n_atoms:
                    raise IndexError(
                        f"BondRetentionWall atom {a} outside system of {n_atoms}"
                    )

    def shortest(self, positions: Any, *, cell: Any = None) -> Any:
        import jax.numpy as jnp

        dists = []
        for i, j in self.pairs:
            d = positions[i] - positions[j]
            if cell is not None:
                lengths = jnp.diag(cell)
                d = d - lengths * jnp.round(d / lengths)
            dists.append(jnp.sqrt(jnp.sum(d * d) + 1e-12))
        return jnp.min(jnp.stack(dists))

    def energy(self, positions: Any, *, cell: Any = None) -> Any:
        import jax.numpy as jnp

        excess = jnp.clip(self.shortest(positions, cell=cell) - self.r_max, 0.0, None)
        return 0.5 * self.k * jnp.square(excess)

    def shortest_numpy(self, positions, *, cell=None) -> float:
        import numpy as _np

        best = None
        for i, j in self.pairs:
            d = _np.asarray(positions)[i] - _np.asarray(positions)[j]
            if cell is not None:
                lengths = _np.diag(_np.asarray(cell))
                d = d - lengths * _np.round(d / lengths)
            r = float(_np.linalg.norm(d))
            best = r if best is None else min(best, r)
        return float(best)

    def _shortest_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """``min_p r_p`` per packed window. Shape ``(K,)``.

        The packed layout used by :mod:`mmml.umbrella` stacks ``K`` copies of an
        ``N``-atom system into one ``(K*N, 3)`` array, so each window's atom
        ``i`` lives at ``k*N + i``. Gas-phase packing, hence no cell.
        """
        import jax.numpy as jnp

        pos = jnp.reshape(positions, (n_windows, n_atoms, 3))
        dists = []
        for i, j in self.pairs:
            d = pos[:, i, :] - pos[:, j, :]
            dists.append(jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-12))
        return jnp.min(jnp.stack(dists, axis=0), axis=0)

    def energy_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """Wall energy per packed window. Shape ``(K,)``."""
        import jax.numpy as jnp

        excess = jnp.clip(
            self._shortest_batched(positions, n_atoms, n_windows) - self.r_max,
            0.0, None,
        )
        return 0.5 * self.k * jnp.square(excess)

    def forces_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        """``F = -grad W``, flattened to ``(K*N, 3)``.

        Taken by autodiff rather than by hand: the gradient of a ``min`` routes
        to whichever pair is currently shortest, and writing that out invites a
        sign or indexing error for no gain -- this is a handful of atoms.
        """
        import jax

        def total(p):
            return self.energy_batched(p, n_atoms, n_windows).sum()

        return -jax.grad(total)(positions)


@dataclass(frozen=True)
class AngleWall:
    """Keep an attack angle inside the reaction channel.

    A transfer coordinate ``xi = r(C-X) - r(C-N)`` says nothing about the
    ``X-C-N`` angle, so the leaving group is free to swing around the product
    once the bond breaks. For NH3 + CH3Cl that is not a hypothetical: windows
    beyond xi = +1.3 sampled a mean angle of 70 deg, the chloride having
    reoriented to hydrogen-bond with the ammonium protons, while windows in the
    reaction region stayed at 165-173 deg. Both are real structures, but they
    are different basins, and a profile along xi alone merges them.

    Crucially those windows did not crash -- they produced finite numbers for
    the whole run. Without this restraint the failure is silent.

    Flat-bottomed above ``theta_min`` (degrees), harmonic below, so backside
    attack is sampled without biasing the angle within its allowed range. ``k``
    is per radian squared, matching the energy units of the rest of the stack.
    """

    atoms: tuple[int, int, int]          # (a, vertex, c); angle a-vertex-c
    theta_min_deg: float = 150.0
    k: float = 50.0

    def __post_init__(self) -> None:
        if len(self.atoms) != 3 or len(set(self.atoms)) != 3:
            raise ValueError(f"AngleWall needs three distinct atoms, got {self.atoms}")
        if not 0.0 < self.theta_min_deg < 180.0:
            raise ValueError(
                f"theta_min_deg must lie in (0, 180) (got {self.theta_min_deg})"
            )
        if self.k < 0:
            raise ValueError(f"force constant must be non-negative (got {self.k})")

    @classmethod
    def from_spec(cls, spec: Any) -> "AngleWall":
        if isinstance(spec, cls):
            return spec
        d = dict(spec)
        return cls(
            atoms=tuple(int(a) for a in d["atoms"]),
            theta_min_deg=float(d.get("theta_min_deg", 150.0)),
            k=float(d.get("k", 50.0)),
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "atoms": list(self.atoms),
            "theta_min_deg": self.theta_min_deg,
            "k": self.k,
        }

    def label(self) -> str:
        a, v, c = self.atoms
        return f"angle({a}-{v}-{c}) >= {self.theta_min_deg:g} deg"

    def validate_against(self, n_atoms: int) -> None:
        for a in self.atoms:
            if not 0 <= a < n_atoms:
                raise IndexError(f"AngleWall atom {a} outside system of {n_atoms}")

    def _theta(self, pos, xp, cell=None):
        a, v, c = self.atoms
        u1 = pos[a] - pos[v]
        u2 = pos[c] - pos[v]
        if cell is not None:
            lengths = xp.diag(cell)
            u1 = u1 - lengths * xp.round(u1 / lengths)
            u2 = u2 - lengths * xp.round(u2 / lengths)
        n1 = xp.sqrt(xp.sum(u1 * u1) + _EPS)
        n2 = xp.sqrt(xp.sum(u2 * u2) + _EPS)
        cos = xp.clip(xp.sum(u1 * u2) / (n1 * n2), -1.0 + 1e-9, 1.0 - 1e-9)
        return xp.arccos(cos)

    def energy(self, positions: Any, *, cell: Any = None) -> Any:
        import jax.numpy as jnp

        theta = self._theta(positions, jnp, cell)
        deficit = jnp.clip(jnp.deg2rad(self.theta_min_deg) - theta, 0.0, None)
        return 0.5 * self.k * jnp.square(deficit)

    def _theta_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        import jax.numpy as jnp

        a, v, c = self.atoms
        pos = jnp.reshape(positions, (n_windows, n_atoms, 3))
        u1 = pos[:, a, :] - pos[:, v, :]
        u2 = pos[:, c, :] - pos[:, v, :]
        n1 = jnp.sqrt(jnp.sum(u1 * u1, axis=-1) + _EPS)
        n2 = jnp.sqrt(jnp.sum(u2 * u2, axis=-1) + _EPS)
        cos = jnp.clip(jnp.sum(u1 * u2, axis=-1) / (n1 * n2), -1.0 + 1e-9, 1.0 - 1e-9)
        return jnp.arccos(cos)

    def energy_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        import jax.numpy as jnp

        theta = self._theta_batched(positions, n_atoms, n_windows)
        deficit = jnp.clip(jnp.deg2rad(self.theta_min_deg) - theta, 0.0, None)
        return 0.5 * self.k * jnp.square(deficit)

    def forces_batched(self, positions: Any, n_atoms: int, n_windows: int) -> Any:
        import jax

        def total(p):
            return self.energy_batched(p, n_atoms, n_windows).sum()

        return -jax.grad(total)(positions)

    def theta_deg_numpy(self, positions, *, cell=None) -> float:
        import numpy as _np

        return float(_np.degrees(self._theta(_np.asarray(positions), _np, cell)))
