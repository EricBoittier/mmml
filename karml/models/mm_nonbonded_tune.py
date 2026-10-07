"""Tune the CGenFF MM tail (per-type LJ scales + a charge scale) of an ML/MM hybrid.

The hybrid MLpot energy of a molecular liquid is

    E = sum_A P(A) + sum_{A<B} s_ML(r_AB) [P(AB) - P(A) - P(B)]
        + sum_{A<B} w_MM(r_AB) E_CGenFF(A, B)

with ``r_AB`` the monomer centroid separation, ``s_ML`` the ML taper and
``w_MM`` the MM handoff/taper (:mod:`mmml.interfaces.pycharmmInterface.calculator_utils`).
Only pair terms appear, so every many-body contribution of the teacher is
missing. This module fits a handful of MM parameters so the hybrid reproduces
teacher *interaction* energies/forces of whole liquid frames,
``E_int = E(box) - sum_A E(A)``, with a Gaussian prior on log-scales that pulls
back toward CGenFF.

Parameters (per CGenFF atom type ``t`` present in the monomer):

* ``eps_scale[t]``  -- ``eps_t -> eps_t * eps_scale[t]`` (geometric combining)
* ``rmin_scale[t]`` -- ``Rmin_t/2 -> Rmin_t/2 * rmin_scale[t]`` (== sigma scale)
* ``charge_scale``  -- ``q -> charge_scale * q`` for every MM charge
* optional *underlay* ``kappa_elec`` / ``kappa_disp``: CGenFF electrostatics /
  r^-6 dispersion added **inside** the ML region with weight ``kappa * s_ML``,
  a mean-field stand-in for the many-body (polarisation) energy that the pair
  ML cannot see. Off by default; needs MD support before it can be deployed.

The MM energy is *linear* in per-type-pair features once the scales are fixed:

    E_MM = sum_k c_k(theta) G_k

    G12_k = sum w_MM r^-12,  G6_k = sum w_MM r^-6   (atom pairs of type class k)
    C     = sum w_MM q_i q_j / r
    c12_k = sqrt(eps_a eps_b s_a s_b) Rmin_ab^12,  c6_k = -2 sqrt(...) Rmin_ab^6
    c_C   = 332.0637 * charge_scale^2

so a frame reduces to ``G`` (n_feat) plus, for forces, a thin SVD of
``J^T = U S V^T`` (``J = dG/dx``): ``|F_res + J^T c|^2 = |F_perp|^2 +
|U^T F_res + S V^T c|^2`` (:func:`force_blocks`; no cancellation between
huge r^-12 terms, unlike the normal equations). Fitting and
bootstrapping then never touch coordinates again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.models.cgenff_mm import COULOMB_CONSTANT, RMIN_HALF_TO_SIGMA
from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_BOUNDS,
    MM_LJ_SIGMA_SCALE_BOUNDS,
)

__all__ = [
    "CHARGE_SCALE_BOUNDS",
    "KAPPA_BOUNDS",
    "SwitchConfig",
    "MonomerNonbonded",
    "TuneParams",
    "FrameFeatures",
    "PriorConfig",
    "FitResult",
    "min_image_pairs",
    "frame_features",
    "ml_pair_energy_forces",
    "force_blocks",
    "mm_coefficients",
    "predict_energy",
    "fit_parameters",
    "bootstrap_fit",
    "cohesion_budget",
    "lj_sidecar_payload",
]

EV_TO_KCAL = 23.060549
CHARGE_SCALE_BOUNDS = (0.8, 1.3)
KAPPA_BOUNDS = (0.0, 2.0)


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchConfig:
    """ML/MM COM handoff, same convention as the MLpot calculator."""

    mm_switch_on: float = 6.0
    mm_switch_width: float = 5.0
    ml_switch_width: float = 1.5
    complementary_handoff: bool = True

    @property
    def mm_cutoff(self) -> float:
        """COM separation beyond which ``w_MM == 0``."""
        extra = 1.0 if self.complementary_handoff else 2.0
        return float(self.mm_switch_on + extra * self.mm_switch_width)

    def ml_weight(self, r_com):
        from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale

        return ml_switch_scale(
            r_com, mm_switch_on=self.mm_switch_on, ml_switch_width=self.ml_switch_width
        )

    def mm_weight(self, r_com):
        from mmml.interfaces.pycharmmInterface.calculator_utils import mm_switch_scale

        return mm_switch_scale(
            r_com,
            mm_switch_on=self.mm_switch_on,
            mm_switch_width=self.mm_switch_width,
            ml_switch_width=self.ml_switch_width,
            complementary_handoff=self.complementary_handoff,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mm_switch_on": float(self.mm_switch_on),
            "mm_switch_width": float(self.mm_switch_width),
            "ml_switch_width": float(self.ml_switch_width),
            "complementary_handoff": bool(self.complementary_handoff),
        }


# ---------------------------------------------------------------------------
# Monomer force field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonomerNonbonded:
    """CGenFF nonbonded parameters of one monomer, in the frames' atom order."""

    atom_types: tuple[str, ...]  # per atom
    charges: np.ndarray  # (a,) e
    type_names: tuple[str, ...]  # unique local types
    type_of_atom: np.ndarray  # (a,) index into type_names
    rmin_half: np.ndarray  # (T,) Angstrom
    epsilon: np.ndarray  # (T,) kcal/mol, >= 0
    residue: str = ""

    @property
    def n_atoms(self) -> int:
        return int(self.charges.shape[0])

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    @property
    def class_pairs(self) -> list[tuple[int, int]]:
        """Unordered type pairs ``(a <= b)``, the LJ feature classes."""
        return [(a, b) for a in range(self.n_types) for b in range(a, self.n_types)]

    @property
    def n_classes(self) -> int:
        return len(self.class_pairs)

    def atom_pair_class(self) -> np.ndarray:
        """``(a, a)`` class index of every (atom in A, atom in B) pair."""
        lookup = {p: k for k, p in enumerate(self.class_pairs)}
        t = self.type_of_atom
        out = np.empty((self.n_atoms, self.n_atoms), dtype=np.int32)
        for i in range(self.n_atoms):
            for j in range(self.n_atoms):
                a, b = sorted((int(t[i]), int(t[j])))
                out[i, j] = lookup[(a, b)]
        return out

    @classmethod
    def from_arrays(
        cls,
        atom_types: Sequence[str],
        charges: Sequence[float],
        rmin_half_by_type: dict[str, float],
        epsilon_by_type: dict[str, float],
        residue: str = "",
    ) -> "MonomerNonbonded":
        atom_types = tuple(str(t) for t in atom_types)
        names: list[str] = []
        for t in atom_types:
            if t not in names:
                names.append(t)
        idx = np.array([names.index(t) for t in atom_types], dtype=np.int32)
        return cls(
            atom_types=atom_types,
            charges=np.asarray(charges, dtype=np.float64),
            type_names=tuple(names),
            type_of_atom=idx,
            rmin_half=np.array([float(rmin_half_by_type[n]) for n in names]),
            epsilon=np.array([abs(float(epsilon_by_type[n])) for n in names]),
            residue=residue,
        )

    @classmethod
    def from_cgenff(cls, numbers: np.ndarray, positions: np.ndarray) -> "MonomerNonbonded":
        """Match a monomer geometry against the CGenFF RTF (graph isomorphism)."""
        from mmml.data.cgenff_dataset import load_reference, match_cgenff_template

        ref = load_reference()
        res, tidx, q = match_cgenff_template(ref, np.asarray(numbers), np.asarray(positions))
        inv = {int(v): k for k, v in ref.nb_map.items()}
        types = [inv[int(i)] for i in tidx]
        rmin = {inv[int(i)]: float(ref.sigmas[int(i)]) / RMIN_HALF_TO_SIGMA for i in tidx}
        eps = {inv[int(i)]: float(ref.epsilons[int(i)]) for i in tidx}
        return cls.from_arrays(types, q, rmin, eps, residue=str(res))

    def to_dict(self) -> dict[str, Any]:
        return {
            "residue": self.residue,
            "atom_types": list(self.atom_types),
            "charges": [float(x) for x in self.charges],
            "type_names": list(self.type_names),
            "rmin_half": [float(x) for x in self.rmin_half],
            "epsilon": [float(x) for x in self.epsilon],
        }


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class TuneParams:
    """Physical parameters (scales are multiplicative on CGenFF)."""

    eps_scale: np.ndarray
    rmin_scale: np.ndarray
    charge_scale: float = 1.0
    kappa_elec: float = 0.0
    kappa_disp: float = 0.0

    @classmethod
    def cgenff(cls, n_types: int) -> "TuneParams":
        return cls(np.ones(n_types), np.ones(n_types))

    def to_dict(self, type_names: Sequence[str]) -> dict[str, Any]:
        return {
            "eps_scale": {n: float(v) for n, v in zip(type_names, self.eps_scale)},
            "rmin_scale": {n: float(v) for n, v in zip(type_names, self.rmin_scale)},
            "charge_scale": float(self.charge_scale),
            "kappa_elec": float(self.kappa_elec),
            "kappa_disp": float(self.kappa_disp),
        }


def mm_coefficients(ff: MonomerNonbonded, params: TuneParams, *, xp=np):
    """Coefficient vector ``c`` with ``E_MM = c . G`` (layout of :func:`frame_features`)."""
    a_idx = np.array([p[0] for p in ff.class_pairs])
    b_idx = np.array([p[1] for p in ff.class_pairs])
    eps = xp.asarray(ff.epsilon) * params.eps_scale
    rh = xp.asarray(ff.rmin_half) * params.rmin_scale
    eps_ab = xp.sqrt(eps[a_idx] * eps[b_idx])
    r_ab = rh[a_idx] + rh[b_idx]
    c12 = eps_ab * r_ab**12
    c6 = -2.0 * eps_ab * r_ab**6
    c_q = COULOMB_CONSTANT * params.charge_scale**2
    # underlay (ML region): CGenFF electrostatics and r^-6 dispersion, unscaled.
    eps0_ab = np.sqrt(ff.epsilon[a_idx] * ff.epsilon[b_idx])
    r0_ab = ff.rmin_half[a_idx] + ff.rmin_half[b_idx]
    c_ul_q = COULOMB_CONSTANT * params.kappa_elec
    c_ul_6 = -2.0 * params.kappa_disp * xp.asarray(eps0_ab * r0_ab**6)
    return xp.concatenate(
        [c12, c6, xp.reshape(xp.asarray(c_q), (1,)), xp.reshape(xp.asarray(c_ul_q), (1,)), c_ul_6]
    )


def n_features(ff: MonomerNonbonded) -> int:
    return 3 * ff.n_classes + 2


# ---------------------------------------------------------------------------
# Geometry and features
# ---------------------------------------------------------------------------


def _mic(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def min_image_pairs(
    mols: np.ndarray, cell: np.ndarray | None, r_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Molecule pairs with centroid separation ``< r_max`` (minimum image).

    Returns ``(pairs (P, 2), shifts (P, 3), r_com (P,))``; molecule ``j`` of a
    pair is translated by ``shift`` to sit next to ``i``. Requires
    ``r_max < L/2`` (asserted) so each pair has a single image.
    """
    mols = np.asarray(mols, dtype=np.float64)
    c = mols.mean(axis=1)
    i, j = np.triu_indices(len(c), k=1)
    d = c[j] - c[i]
    if cell is not None and abs(np.linalg.det(cell)) > 1e-9:
        cell = np.asarray(cell, dtype=np.float64)
        heights = abs(np.linalg.det(cell)) / np.linalg.norm(
            np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
        )
        if r_max >= 0.5 * heights.min():
            raise ValueError(
                f"pair cutoff {r_max:.2f} A >= half the cell height {0.5 * heights.min():.2f} A"
            )
        dm = _mic(d, cell)
    else:
        dm = d
    r = np.linalg.norm(dm, axis=1)
    keep = r < float(r_max)
    return (
        np.stack([i[keep], j[keep]], axis=1).astype(np.int32),
        (dm - d)[keep],
        r[keep],
    )


@dataclass
class FrameFeatures:
    """Per-frame reduction used by the fitter (all energies kcal/mol)."""

    G: np.ndarray  # (F, n_feat)
    e_res: np.ndarray  # (F,) teacher E_int - student switched ML E_int
    n_mol: np.ndarray  # (F,)
    group: np.ndarray  # (F,) bootstrap/split group (seed)
    L: np.ndarray | None = None  # (F, n_feat, n_feat) Sigma V^T of J^T = U Sigma V^T
    y: np.ndarray | None = None  # (F, n_feat) U^T F_res
    f_perp_sq: np.ndarray | None = None  # (F,) |F_res - U U^T F_res|^2
    n_atoms: np.ndarray | None = None  # (F,)
    e_teacher: np.ndarray | None = None  # (F,) teacher E_int
    e_ml: np.ndarray | None = None  # (F,) student switched ML E_int
    meta: dict[str, Any] = field(default_factory=dict)

    def subset(self, idx: np.ndarray) -> "FrameFeatures":
        idx = np.asarray(idx)
        pick = lambda x: None if x is None else x[idx]  # noqa: E731
        return FrameFeatures(
            G=self.G[idx],
            e_res=self.e_res[idx],
            n_mol=self.n_mol[idx],
            group=self.group[idx],
            L=pick(self.L),
            y=pick(self.y),
            f_perp_sq=pick(self.f_perp_sq),
            n_atoms=pick(self.n_atoms),
            e_teacher=pick(self.e_teacher),
            e_ml=pick(self.e_ml),
            meta=dict(self.meta),
        )

    @property
    def has_forces(self) -> bool:
        return self.L is not None


def _feature_fn(ff: MonomerNonbonded, switch: SwitchConfig, pairs, shifts):
    """JAX ``X (M, a, 3) -> G (n_feat,)`` for a fixed pair list."""
    import jax
    import jax.numpy as jnp

    cls = jnp.asarray(ff.atom_pair_class().reshape(-1))
    qq = jnp.asarray(np.outer(ff.charges, ff.charges))
    n_cls = ff.n_classes
    pi = jnp.asarray(pairs[:, 0])
    pj = jnp.asarray(pairs[:, 1])
    sh = jnp.asarray(shifts)

    def features(X):
        xi = X[pi]
        xj = X[pj] + sh[:, None, :]
        dc = xj.mean(axis=1) - xi.mean(axis=1)
        r_com = jnp.sqrt(jnp.sum(dc * dc, axis=-1))
        w_mm = switch.mm_weight(r_com)
        w_ml = switch.ml_weight(r_com)
        d = xi[:, :, None, :] - xj[:, None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        inv2 = 1.0 / r2
        inv6 = inv2 * inv2 * inv2
        inv12 = inv6 * inv6
        invr = jnp.sqrt(inv2)
        g12 = jnp.einsum("p,pij->ij", w_mm, inv12).reshape(-1)
        g6 = jnp.einsum("p,pij->ij", w_mm, inv6).reshape(-1)
        g6_ml = jnp.einsum("p,pij->ij", w_ml, inv6).reshape(-1)
        cq = jnp.einsum("p,pij,ij->", w_mm, invr, qq)
        cq_ml = jnp.einsum("p,pij,ij->", w_ml, invr, qq)
        seg = lambda v: jax.ops.segment_sum(v, cls, num_segments=n_cls)  # noqa: E731
        return jnp.concatenate(
            [seg(g12), seg(g6), cq[None], cq_ml[None], seg(g6_ml)]
        )

    return features


def frame_features(
    ff: MonomerNonbonded,
    switch: SwitchConfig,
    mols: np.ndarray,
    cell: np.ndarray | None,
    *,
    with_jacobian: bool = True,
):
    """``G`` (and ``J = dG/dX`` of shape ``(n_feat, M*a*3)``) for one frame."""
    import jax
    import jax.numpy as jnp

    r_max = max(switch.mm_cutoff, switch.mm_switch_on)
    pairs, shifts, _ = min_image_pairs(mols, cell, r_max)
    fn = _feature_fn(ff, switch, pairs, shifts)
    X = jnp.asarray(np.asarray(mols, dtype=np.float64))
    G = np.asarray(jax.jit(fn)(X))
    if not with_jacobian:
        return G, None
    J = np.asarray(jax.jit(jax.jacrev(fn))(X)).reshape(G.shape[0], -1)
    return G, J


def force_blocks(J: np.ndarray, f_res: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(S V^T, U^T f_res, |f_perp|^2)`` from ``J`` (n_feat, 3N) and ``f_res`` (3N,).

    ``|f_res + J^T c|^2 == f_perp + |y + L c|^2`` for every ``c``.
    """
    U, S, Vt = np.linalg.svd(np.asarray(J).T, full_matrices=False)
    y = U.T @ f_res
    perp = f_res - U @ y
    return S[:, None] * Vt, y, float(perp @ perp)


def ml_pair_energy_forces(
    switch: SwitchConfig,
    n_mol: int,
    n_atoms_per_mol: int,
    pairs: np.ndarray,
    r_com: np.ndarray,
    shifts: np.ndarray,
    mols: np.ndarray,
    e_pair: np.ndarray,
    f_pair: np.ndarray | None,
) -> tuple[float, np.ndarray | None]:
    """Switched student ML interaction energy and forces from stored pair data.

    ``e_pair = P(AB) - P(A) - P(B)``; ``f_pair`` ``(P, 2a, 3)`` the matching
    interaction forces. Adds the switch-gradient force ``-e dS/dr dr/dx``.
    """
    import jax
    import jax.numpy as jnp

    r = jnp.asarray(r_com)
    s = np.asarray(switch.ml_weight(r))
    energy = float(np.sum(s * e_pair))
    if f_pair is None:
        return energy, None
    ds = np.asarray(jax.vmap(jax.grad(lambda x: switch.ml_weight(x)))(r))
    a = int(n_atoms_per_mol)
    F = np.zeros((n_mol, a, 3))
    fp = np.asarray(f_pair).reshape(-1, 2, a, 3) * s[:, None, None, None]
    np.add.at(F, pairs[:, 0], fp[:, 0])
    np.add.at(F, pairs[:, 1], fp[:, 1])
    c = np.asarray(mols).mean(axis=1)
    dvec = c[pairs[:, 1]] + shifts - c[pairs[:, 0]]
    unit = dvec / np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-12)
    # dr/dx_j(atom of B) = unit / a ; dr/dx_i(atom of A) = -unit / a
    g = (-(e_pair * ds))[:, None] * unit / a  # force on each B atom
    np.add.at(F, pairs[:, 1], np.repeat(g[:, None, :], a, axis=1))
    np.add.at(F, pairs[:, 0], -np.repeat(g[:, None, :], a, axis=1))
    return energy, F


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorConfig:
    """Gaussian prior on log-scales (width = relative change of ~1 sigma)."""

    tau_eps: float = 0.5
    tau_rmin: float = 0.03
    tau_charge: float = 0.1
    tau_kappa: float = 0.5
    sigma_e: float = 0.05  # kcal/mol per molecule
    sigma_f: float = 1.0  # kcal/mol/A per component
    force_weight: float = 1.0
    fit_eps: bool = True
    tie_eps: bool = False  # one epsilon scale shared by every type
    fit_rmin: bool = True
    fit_charge: bool = True
    fit_kappa_elec: bool = False
    fit_kappa_disp: bool = False
    eps_bounds: tuple[float, float] = MM_LJ_EPSILON_SCALE_BOUNDS
    rmin_bounds: tuple[float, float] = MM_LJ_SIGMA_SCALE_BOUNDS
    charge_bounds: tuple[float, float] = CHARGE_SCALE_BOUNDS
    kappa_bounds: tuple[float, float] = KAPPA_BOUNDS


@dataclass
class FitResult:
    params: TuneParams
    loss: float
    success: bool
    message: str
    at_bound: list[str]


def _unpack(z, ff: MonomerNonbonded, prior: PriorConfig, xp):
    T = ff.n_types
    k = 0
    ones = xp.ones(T)
    if prior.fit_eps and prior.tie_eps:
        eps = ones * xp.exp(z[k])
        k += 1
    elif prior.fit_eps:
        eps = xp.exp(z[k : k + T])
        k += T
    else:
        eps = ones
    if prior.fit_rmin:
        rmin = xp.exp(z[k : k + T])
        k += T
    else:
        rmin = ones
    if prior.fit_charge:
        q = xp.exp(z[k])
        k += 1
    else:
        q = 1.0
    if prior.fit_kappa_elec:
        ke = z[k]
        k += 1
    else:
        ke = 0.0
    if prior.fit_kappa_disp:
        kd = z[k]
        k += 1
    else:
        kd = 0.0
    return TuneParams(eps, rmin, q, ke, kd)


def _bounds(ff: MonomerNonbonded, prior: PriorConfig) -> list[tuple[float, float]]:
    T = ff.n_types
    out: list[tuple[float, float]] = []
    if prior.fit_eps:
        out += [tuple(np.log(prior.eps_bounds))] * (1 if prior.tie_eps else T)
    if prior.fit_rmin:
        out += [tuple(np.log(prior.rmin_bounds))] * T
    if prior.fit_charge:
        out.append(tuple(np.log(prior.charge_bounds)))
    if prior.fit_kappa_elec:
        out.append(tuple(prior.kappa_bounds))
    if prior.fit_kappa_disp:
        out.append(tuple(prior.kappa_bounds))
    return out


def _names(ff: MonomerNonbonded, prior: PriorConfig) -> list[str]:
    out: list[str] = []
    if prior.fit_eps and prior.tie_eps:
        out.append("eps_scale[*]")
    elif prior.fit_eps:
        out += [f"eps_scale[{t}]" for t in ff.type_names]
    if prior.fit_rmin:
        out += [f"rmin_scale[{t}]" for t in ff.type_names]
    if prior.fit_charge:
        out.append("charge_scale")
    if prior.fit_kappa_elec:
        out.append("kappa_elec")
    if prior.fit_kappa_disp:
        out.append("kappa_disp")
    return out


def _loss_fn(ff: MonomerNonbonded, data: FrameFeatures, prior: PriorConfig, weights=None):
    import jax.numpy as jnp

    G = jnp.asarray(data.G)
    e_res = jnp.asarray(data.e_res)
    n_mol = jnp.asarray(data.n_mol, dtype=G.dtype)
    w = jnp.ones(G.shape[0]) if weights is None else jnp.asarray(weights)
    wsum = jnp.sum(w)
    use_f = data.has_forces and prior.force_weight > 0
    if use_f:
        Lf = jnp.asarray(data.L)
        yf = jnp.asarray(data.y)
        fp = jnp.asarray(data.f_perp_sq)
        n3 = 3.0 * jnp.asarray(data.n_atoms, dtype=G.dtype)

    def loss(z):
        p = _unpack(z, ff, prior, jnp)
        c = mm_coefficients(ff, p, xp=jnp)
        de = (G @ c - e_res) / n_mol
        L = jnp.sum(w * de**2) / wsum / prior.sigma_e**2
        if use_f:
            sq = fp + jnp.sum((yf + Lf @ c) ** 2, axis=-1)
            L = L + prior.force_weight * jnp.sum(w * sq / n3) / wsum / prior.sigma_f**2
        reg = 0.0
        if prior.fit_eps:
            reg = reg + jnp.sum((jnp.log(p.eps_scale) / prior.tau_eps) ** 2) / (
                ff.n_types if prior.tie_eps else 1
            )
        if prior.fit_rmin:
            reg = reg + jnp.sum((jnp.log(p.rmin_scale) / prior.tau_rmin) ** 2)
        if prior.fit_charge:
            reg = reg + (jnp.log(p.charge_scale) / prior.tau_charge) ** 2
        if prior.fit_kappa_elec:
            reg = reg + (p.kappa_elec / prior.tau_kappa) ** 2
        if prior.fit_kappa_disp:
            reg = reg + (p.kappa_disp / prior.tau_kappa) ** 2
        # data terms are means, the prior is per-parameter: scale the prior by
        # 1/n_frames so it acts like one extra pseudo-frame per parameter.
        return L + reg / wsum

    return loss


def fit_parameters(
    ff: MonomerNonbonded,
    data: FrameFeatures,
    prior: PriorConfig = PriorConfig(),
    *,
    weights: np.ndarray | None = None,
    maxiter: int = 3000,
) -> FitResult:
    """MAP fit (L-BFGS-B on log-scales) of the MM parameters to ``data``."""
    import jax
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    loss = _loss_fn(ff, data, prior, weights)
    vg = jax.jit(jax.value_and_grad(loss))
    bounds = _bounds(ff, prior)
    z = np.zeros(len(bounds))
    opts = {"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10}
    best = float(vg(z)[0])
    res = None
    # L-BFGS-B tolerances are relative to the objective's magnitude, which
    # spans orders of magnitude between the CGenFF start and the optimum:
    # rescale by the current value and restart until it stops improving.
    for _ in range(12):
        scale = max(best, 1e-12)

        def f(zz, scale=scale):
            v, g = vg(np.asarray(zz))
            return float(v) / scale, np.asarray(g, dtype=np.float64) / scale

        res = minimize(f, z, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
        new = float(res.fun) * scale
        if new < best:
            z = np.asarray(res.x)
        if new > best * (1.0 - 1e-6):
            best = min(best, new)
            break
        best = new
    res.x = z
    res.fun = best
    p = _unpack(np.asarray(res.x), ff, prior, np)
    p = TuneParams(
        np.asarray(p.eps_scale, dtype=float),
        np.asarray(p.rmin_scale, dtype=float),
        float(p.charge_scale),
        float(p.kappa_elec),
        float(p.kappa_disp),
    )
    at_bound = [
        n
        for n, zi, (lo, hi) in zip(_names(ff, prior), res.x, bounds)
        if min(abs(zi - lo), abs(zi - hi)) < 1e-6
    ]
    return FitResult(p, float(res.fun), bool(res.success), str(res.message), at_bound)


def predict_energy(ff: MonomerNonbonded, params: TuneParams, data: FrameFeatures) -> np.ndarray:
    """MM energy per frame (kcal/mol)."""
    return data.G @ np.asarray(mm_coefficients(ff, params))


def predict_force_rmse(ff: MonomerNonbonded, params: TuneParams, data: FrameFeatures) -> float:
    """RMSE (kcal/mol/A, per component) of ``F_teacher_int - F_ML - F_MM``."""
    if not data.has_forces:
        return float("nan")
    c = np.asarray(mm_coefficients(ff, params))
    sq = data.f_perp_sq + np.sum((data.y + data.L @ c) ** 2, axis=-1)
    return float(np.sqrt(np.sum(sq) / np.sum(3.0 * data.n_atoms)))


def energy_rmse_per_mol(ff: MonomerNonbonded, params: TuneParams, data: FrameFeatures) -> float:
    de = (predict_energy(ff, params, data) - data.e_res) / data.n_mol
    return float(np.sqrt(np.mean(de**2)))


def bootstrap_fit(
    ff: MonomerNonbonded,
    data: FrameFeatures,
    prior: PriorConfig = PriorConfig(),
    *,
    n_boot: int = 10,
    seed: int = 0,
) -> list[FitResult]:
    """Group (seed) bootstrap: resample whole groups with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.unique(data.group)
    out = []
    for _ in range(int(n_boot)):
        draw = rng.choice(groups, size=len(groups), replace=True)
        counts = {g: int(np.sum(draw == g)) for g in groups}
        w = np.array([counts[g] for g in data.group], dtype=np.float64)
        keep = w > 0
        out.append(fit_parameters(ff, data.subset(np.flatnonzero(keep)), prior, weights=w[keep]))
    return out


def summarize_params(ff: MonomerNonbonded, fits: Sequence[FitResult]) -> dict[str, Any]:
    """Mean/std over fits, keyed by parameter name."""
    rows: dict[str, list[float]] = {}
    for fr in fits:
        p = fr.params
        for t, v in zip(ff.type_names, p.eps_scale):
            rows.setdefault(f"eps_scale[{t}]", []).append(float(v))
        for t, v in zip(ff.type_names, p.rmin_scale):
            rows.setdefault(f"rmin_scale[{t}]", []).append(float(v))
        rows.setdefault("charge_scale", []).append(float(p.charge_scale))
        rows.setdefault("kappa_elec", []).append(float(p.kappa_elec))
        rows.setdefault("kappa_disp", []).append(float(p.kappa_disp))
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "values": v}
        for k, v in rows.items()
    }


def cohesion_budget(
    ff: MonomerNonbonded, params: TuneParams, data: FrameFeatures
) -> dict[str, float]:
    """Mean per-molecule interaction energy decomposition (kcal/mol/molecule).

    ``mm_tail`` uses the fitted tail parameters with the underlay switched off;
    ``underlay`` is the ML-region term alone.
    """
    tail_only = TuneParams(params.eps_scale, params.rmin_scale, params.charge_scale, 0.0, 0.0)
    e_tail = predict_energy(ff, tail_only, data)
    e_all = predict_energy(ff, params, data)
    per = lambda x: float(np.mean(np.asarray(x) / data.n_mol))  # noqa: E731
    teacher = per(data.e_teacher)
    ml = per(data.e_ml)
    hybrid = ml + per(e_all)
    return {
        "teacher_total": teacher,
        "ml_pairs": ml,
        "mm_tail": per(e_tail),
        "underlay": per(e_all - e_tail),
        "hybrid_total": hybrid,
        "missing": teacher - hybrid,
    }


def lj_sidecar_payload(ff: MonomerNonbonded, params: TuneParams) -> dict[str, Any]:
    """``hybrid_mm.json``-compatible LJ-scale block (+ charge scale / switch info).

    ``sigma`` scale == ``rmin`` scale. Types outside the monomer stay at 1.
    :func:`mmml.models.mm_lj_scales.resolve_md_lj_scales` reads the LJ part.
    """
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm, mm_lj_scales_metadata

    names = cgenff_type_names_from_prm()
    sig = np.ones(len(names))
    eps = np.ones(len(names))
    for t, s_e, s_r in zip(ff.type_names, params.eps_scale, params.rmin_scale):
        k = names.index(t)
        sig[k] = float(s_r)
        eps[k] = float(s_e)
    out = mm_lj_scales_metadata(
        learn_mm_lj_scales=True, type_names=names, sigma_scale=sig, epsilon_scale=eps
    )
    out["mm_charge_scale"] = float(params.charge_scale)
    out["mm_underlay"] = {"kappa_elec": float(params.kappa_elec), "kappa_disp": float(params.kappa_disp)}
    return out


def save_features(path: Path, data: FrameFeatures) -> None:
    arrays = {k: getattr(data, k) for k in ("G", "e_res", "n_mol", "group", "L", "y",
                                             "f_perp_sq", "n_atoms", "e_teacher", "e_ml")}
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None},
             meta=json.dumps(data.meta))


def load_features(path: Path) -> FrameFeatures:
    d = np.load(path, allow_pickle=False)
    get = lambda k: d[k] if k in d.files else None  # noqa: E731
    return FrameFeatures(
        G=d["G"], e_res=d["e_res"], n_mol=d["n_mol"], group=d["group"], L=get("L"),
        y=get("y"), f_perp_sq=get("f_perp_sq"), n_atoms=get("n_atoms"),
        e_teacher=get("e_teacher"), e_ml=get("e_ml"),
        meta=json.loads(str(d["meta"])) if "meta" in d.files else {},
    )


def dimer_features(ff: MonomerNonbonded, switch: SwitchConfig, R: np.ndarray) -> np.ndarray:
    """Vectorised :func:`frame_features` (energy only) for gas-phase dimers.

    ``R`` is ``(n, 2a, 3)`` with monomer A first. Returns ``(n, n_feat)``.
    """
    import jax.numpy as jnp

    a = ff.n_atoms
    R = np.asarray(R, dtype=np.float64)
    xa, xb = R[:, :a], R[:, a : 2 * a]
    r_com = np.linalg.norm(xb.mean(axis=1) - xa.mean(axis=1), axis=-1)
    w_mm = np.asarray(switch.mm_weight(jnp.asarray(r_com)))
    w_ml = np.asarray(switch.ml_weight(jnp.asarray(r_com)))
    r2 = np.sum((xa[:, :, None, :] - xb[:, None, :, :]) ** 2, axis=-1)
    inv6 = r2**-3
    inv12 = inv6**2
    invr = r2**-0.5
    cls = ff.atom_pair_class().reshape(-1)
    K = ff.n_classes
    onehot = np.zeros((a * a, K))
    onehot[np.arange(a * a), cls] = 1.0
    qq = np.outer(ff.charges, ff.charges)
    g12 = (w_mm[:, None] * inv12.reshape(len(R), -1)) @ onehot
    g6 = (w_mm[:, None] * inv6.reshape(len(R), -1)) @ onehot
    g6_ml = (w_ml[:, None] * inv6.reshape(len(R), -1)) @ onehot
    cq = w_mm * np.einsum("nij,ij->n", invr, qq)
    cq_ml = w_ml * np.einsum("nij,ij->n", invr, qq)
    return np.concatenate([g12, g6, cq[:, None], cq_ml[:, None], g6_ml], axis=1)
