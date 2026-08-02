"""Soft-well dimer ``E_int`` auxiliary loss for hybrid ML/MM training.

The dense NVT overbind / soft underbind problem is not fixed by plain FT or
distillation at ``mm_switch_on=5``.  Those preserve ASE total-energy fits but
do not explicitly shape contact-ok soft wells.  This module is the real lever:

* Build DCM–DCM geometries in the soft COM window (default ``r ∈ [3.4, 6]`` Å).
* Keep only sterically plausible frames (``dmin ≥ DEFAULT_ORIENT_MIN_CONTACT_A``).
* Pull hybrid interaction energy ``E_int = E_total - E_A - E_B`` into the lit
  DCM window ``[-5, -3]`` kcal/mol, with a hard floor against deep wells.

Units: hybrid forward returns eV; lit targets are kcal/mol.  Convert with
:data:`mmml.data.units.EV_TO_KCAL_MOL`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mmml.analysis.dimer_scans import (
    DEFAULT_ORIENT_MIN_CONTACT_A,
    intermolecular_min_distance,
)
from mmml.data.units import EV_TO_KCAL_MOL

# Lit DCM soft-well window (kcal/mol). Matches eval_lever2_on5_* gates.
DEFAULT_TARGET_LO_KCAL = -5.0
DEFAULT_TARGET_HI_KCAL = -3.0
DEFAULT_TARGET_MID_KCAL = -4.0
# Start penalising before the deploy deepest floor (−15).
DEFAULT_DEEP_FLOOR_KCAL = -12.0
DEFAULT_HARD_FLOOR_KCAL = -15.0
# Soft metric is min E_int at r≥3.4, but at mm_switch_on=5 ML is already off
# for r≳5 (ml_s→0). Aux must act where ML still controls E_int (s≳0.5),
# i.e. r ≲ on − 0.75·width ≈ 4.25 Å — otherwise the lever is frozen MM.
DEFAULT_SOFT_R_MIN_A = 3.4
DEFAULT_SOFT_R_MAX_A = 4.25


def soft_well_e_int_loss(
    e_int_ev,
    *,
    target_lo_kcal: float = DEFAULT_TARGET_LO_KCAL,
    target_hi_kcal: float = DEFAULT_TARGET_HI_KCAL,
    target_mid_kcal: float = DEFAULT_TARGET_MID_KCAL,
    deep_floor_kcal: float = DEFAULT_DEEP_FLOOR_KCAL,
    hard_floor_kcal: float = DEFAULT_HARD_FLOOR_KCAL,
    center_weight: float = 0.25,
    deep_weight: float = 2.0,
    hard_weight: float = 4.0,
    per_sample_cap: float = 64.0,
    focus_max_kcal: float = 5.0,
):
    """Window + floor loss on hybrid ``E_int`` (eV in, scalar loss out).

    * ``e > target_hi`` (underbind / too shallow): quadratic pull down.
    * ``e < target_lo`` (overbind vs lit window): quadratic push up.
    * ``e < deep_floor`` / ``hard_floor``: stronger penalties (deploy gates).
    * Mild centre pull toward ``target_mid`` so soft median lands near −4.

    Samples with ``E_int > focus_max_kcal`` (far-repulsive pathologies; diag
    saw +137 kcal) are masked out of the mean so they cannot steal the
    underbind gradient.  Per-sample terms use a tanh soft-cap.
    """
    import jax.numpy as jnp

    e_kcal = jnp.asarray(e_int_ev, dtype=jnp.float32).reshape(-1) * EV_TO_KCAL_MOL
    under = jnp.maximum(e_kcal - float(target_hi_kcal), 0.0)
    over = jnp.maximum(float(target_lo_kcal) - e_kcal, 0.0)
    deep = jnp.maximum(float(deep_floor_kcal) - e_kcal, 0.0)
    hard = jnp.maximum(float(hard_floor_kcal) - e_kcal, 0.0)
    center = e_kcal - float(target_mid_kcal)
    # Only centre-pull samples already near the soft well; far-repulsive
    # outliers (E_int ≫ 0) must not dominate the gradient.
    near = jnp.exp(-((e_kcal - float(target_mid_kcal)) / 6.0) ** 2)
    per = (
        under * under
        + over * over
        + float(deep_weight) * deep * deep
        + float(hard_weight) * hard * hard
        + float(center_weight) * near * center * center
    )
    # Soft cap keeps a non-zero gradient (hard min() zeros grads past the cap).
    if per_sample_cap is not None and float(per_sample_cap) > 0.0:
        c = float(per_sample_cap)
        per = c * jnp.tanh(per / c)
    # Mask far-repulsive pathologies out of the batch mean.
    w = (e_kcal <= float(focus_max_kcal)).astype(per.dtype)
    denom = jnp.maximum(jnp.sum(w), 1.0)
    return jnp.sum(per * w) / denom


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def super_fibonacci(n: int) -> np.ndarray:
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    s = i / n
    t = s * n / phi
    d = 2.0 * np.pi * (t - np.floor(t))
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    t2 = i / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))
    return np.stack(
        [r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def extract_monomer_from_hybrid_frame(
    data: Mapping[str, Any],
    *,
    frame: int = 0,
) -> dict[str, np.ndarray]:
    """Take monomer A (``mol_id == 0``) from one hybrid training frame."""
    mol = np.asarray(data["mol_id"][frame])
    keep = mol == 0
    if not np.any(keep):
        # Fallback: first half of non-padding atoms (DCM dimers are 5+5).
        z = np.asarray(data["Z"][frame])
        n = int(np.sum(z > 0))
        keep = np.zeros_like(mol, dtype=bool)
        keep[: max(n // 2, 1)] = True
    R = np.asarray(data["R"][frame], dtype=np.float64)[keep]
    R = R - R.mean(axis=0)
    out = {
        "R": R,
        "Z": np.asarray(data["Z"][frame], dtype=np.int32)[keep],
        "cgenff_type_idx": np.asarray(data["cgenff_type_idx"][frame], dtype=np.int32)[
            keep
        ],
        "cgenff_charge": np.asarray(data["cgenff_charge"][frame], dtype=np.float64)[
            keep
        ],
    }
    return out


@dataclass(frozen=True)
class SoftWellConfig:
    """Host-side soft-well aux settings (not a jit static pytree)."""

    enabled: bool = False
    steps_per_epoch: int = 64
    every_n_train_batches: int = 20
    batch_size: int = 32
    n_directions: int = 20
    n_orientations: int = 16
    n_r: int = 10
    r_min: float = DEFAULT_SOFT_R_MIN_A
    r_max: float = DEFAULT_SOFT_R_MAX_A
    min_contact: float = DEFAULT_ORIENT_MIN_CONTACT_A
    target_lo_kcal: float = DEFAULT_TARGET_LO_KCAL
    target_hi_kcal: float = DEFAULT_TARGET_HI_KCAL
    target_mid_kcal: float = DEFAULT_TARGET_MID_KCAL
    deep_floor_kcal: float = DEFAULT_DEEP_FLOOR_KCAL
    hard_floor_kcal: float = DEFAULT_HARD_FLOOR_KCAL
    center_weight: float = 0.25
    loss_scale: float = 50.0
    seed: int = 0
    # Drop pool members whose frozen-teacher E_int is above this (kcal/mol).
    pool_max_e_int_kcal: float = 5.0

    @classmethod
    def coerce(cls, cfg) -> "SoftWellConfig | None":
        if cfg is None:
            return None
        if isinstance(cfg, cls):
            return cfg
        d = dict(cfg)
        d["enabled"] = bool(d.get("enabled", True))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class SoftWellGeometryPool:
    """Precomputed contact-ok soft dimer geometries + minibatch sampler."""

    def __init__(
        self,
        monomer: Mapping[str, np.ndarray],
        cfg: SoftWellConfig,
    ):
        self.cfg = cfg
        R1 = np.asarray(monomer["R"], dtype=np.float64)
        Z1 = np.asarray(monomer["Z"], dtype=np.int32)
        T1 = np.asarray(monomer["cgenff_type_idx"], dtype=np.int32)
        Q1 = np.asarray(monomer["cgenff_charge"], dtype=np.float64)
        n_mono = int(R1.shape[0])
        pad = 2 * n_mono
        self.n_mono = n_mono
        self.pad = pad

        dirs = fibonacci_sphere(int(cfg.n_directions))
        quats = super_fibonacci(int(cfg.n_orientations))
        rs = np.linspace(float(cfg.r_min), float(cfg.r_max), int(cfg.n_r))
        rows_R, rows_Z, rows_T, rows_Q, rows_M = [], [], [], [], []
        for dvec in dirs:
            for q in quats:
                Rb0 = R1 @ quat_to_matrix(q).T
                for r in rs:
                    Ra = R1 - 0.5 * r * dvec
                    Rb = Rb0 + 0.5 * r * dvec
                    dmin = intermolecular_min_distance(Ra, Rb)
                    if dmin < float(cfg.min_contact):
                        continue
                    R = np.zeros((pad, 3), dtype=np.float64)
                    Z = np.zeros((pad,), dtype=np.int32)
                    T = np.full((pad,), -1, dtype=np.int32)
                    Q = np.zeros((pad,), dtype=np.float64)
                    M = np.full((pad,), -1, dtype=np.int32)
                    R[:n_mono] = Ra
                    R[n_mono:pad] = Rb
                    Z[:n_mono] = Z1
                    Z[n_mono:pad] = Z1
                    T[:n_mono] = T1
                    T[n_mono:pad] = T1
                    Q[:n_mono] = Q1
                    Q[n_mono:pad] = Q1
                    M[:n_mono] = 0
                    M[n_mono:pad] = 1
                    rows_R.append(R)
                    rows_Z.append(Z)
                    rows_T.append(T)
                    rows_Q.append(Q)
                    rows_M.append(M)

        if not rows_R:
            raise ValueError(
                "SoftWellGeometryPool: no contact-ok soft geometries; "
                f"relax min_contact={cfg.min_contact} or r-window "
                f"[{cfg.r_min}, {cfg.r_max}]"
            )
        self.R = np.stack(rows_R, axis=0)
        self.Z = np.stack(rows_Z, axis=0)
        self.T = np.stack(rows_T, axis=0)
        self.Q = np.stack(rows_Q, axis=0)
        self.M = np.stack(rows_M, axis=0)
        self.n = int(self.R.shape[0])
        self._rng = np.random.default_rng(int(cfg.seed))
        self._batch_cache: list[dict] | None = None
        self._batch_i = 0

    def filter_by_teacher_e_int(
        self,
        model_apply,
        params,
        hybrid_mm,
        *,
        batch_size: int = 64,
        max_e_int_kcal: float | None = None,
    ) -> int:
        """Drop pool members with teacher ``E_int`` above ``max_e_int_kcal``.

        Far-repulsive orientations (+100 kcal) steal underbind gradients; keep
        only geometries the frozen teacher already treats as soft-ish.
        Returns the number of geometries retained.
        """
        import jax
        from mmml.data.units import EV_TO_KCAL_MOL as _EV2K
        from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS
        from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit
        from mmml.models.physnetjax.physnetjax.training.trainstep import _forward

        max_e = float(
            self.cfg.pool_max_e_int_kcal if max_e_int_kcal is None else max_e_int_kcal
        )
        bs = int(min(batch_size, max(self.n, 1)))
        n_pad = int(np.ceil(self.n / bs) * bs)
        idx = np.arange(n_pad) % self.n
        data = self._data_dict(idx)
        keys = [
            "R",
            "Z",
            "F",
            "E",
            "N",
            "D",
            "dst_idx",
            "src_idx",
            "batch_segments",
            "id",
        ] + list(HYBRID_MM_BATCH_KEYS)
        batches = prepare_batches_jit(
            jax.random.PRNGKey(0),
            data,
            bs,
            num_atoms=self.pad,
            data_keys=keys,
            include_id=True,
        )
        e_all = np.full(n_pad, np.nan, dtype=np.float64)
        for b in batches:
            out = _forward(model_apply, params, b, bs, hybrid_mm=hybrid_mm)
            e = np.asarray(out.get("e_int", out["energy"])).reshape(-1) * float(_EV2K)
            ids = np.asarray(b["id"]).reshape(-1)
            e_all[ids] = e
        keep = np.where(e_all[: self.n] <= max_e)[0]
        if keep.size < max(8, bs // 2):
            # Fall back to the most-bound half of the pool rather than failing.
            order = np.argsort(e_all[: self.n])
            keep = order[: max(bs, self.n // 2)]
        self.R = self.R[keep]
        self.Z = self.Z[keep]
        self.T = self.T[keep]
        self.Q = self.Q[keep]
        self.M = self.M[keep]
        self.n = int(self.R.shape[0])
        self._batch_cache = None
        self._batch_i = 0
        return self.n

    def _data_dict(self, idx: np.ndarray) -> dict:
        n = int(idx.shape[0])
        return {
            "R": self.R[idx],
            "Z": self.Z[idx],
            "F": np.zeros((n, self.pad, 3), dtype=np.float64),
            "E": np.zeros((n, 1), dtype=np.float64),
            "N": np.full((n,), self.pad, dtype=np.int32),
            "D": np.zeros((n, 3), dtype=np.float64),
            "cgenff_type_idx": self.T[idx],
            "cgenff_charge": self.Q[idx],
            "mol_id": self.M[idx],
            "id": np.arange(n, dtype=np.int32),
        }

    def next_batch(self, batch_size: int | None = None) -> dict:
        """Return one prepared hybrid batch (JAX arrays)."""
        import jax
        from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS
        from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

        bs = int(batch_size or self.cfg.batch_size)
        if bs > self.n:
            bs = self.n
        # Rebuild a shuffled epoch of batches when exhausted.
        if self._batch_cache is None or self._batch_i >= len(self._batch_cache):
            order = self._rng.permutation(self.n)
            # Drop remainder so every batch has identical shape (jit-friendly).
            n_use = (self.n // bs) * bs
            if n_use < bs:
                # Repeat geometries to fill one batch.
                order = np.resize(order, bs)
                n_use = bs
            order = order[:n_use]
            data = self._data_dict(order)
            keys = [
                "R",
                "Z",
                "F",
                "E",
                "N",
                "D",
                "dst_idx",
                "src_idx",
                "batch_segments",
                "id",
            ] + list(HYBRID_MM_BATCH_KEYS)
            self._batch_cache = list(
                prepare_batches_jit(
                    jax.random.PRNGKey(int(self._rng.integers(0, 2**31 - 1))),
                    data,
                    bs,
                    num_atoms=self.pad,
                    data_keys=keys,
                    include_id=True,
                )
            )
            self._batch_i = 0
        batch = self._batch_cache[self._batch_i]
        self._batch_i += 1
        return batch


def restore_mm_lj_scales(params, sigma_scale, epsilon_scale):
    """Hard-freeze LJ scale leaves after an optimizer step."""
    from mmml.models.mm_lj_scales import (
        MM_LJ_EPSILON_SCALE_KEY,
        MM_LJ_SIGMA_SCALE_KEY,
    )
    import jax.numpy as jnp

    if not isinstance(params, dict):
        return params
    if sigma_scale is None or epsilon_scale is None:
        return params
    out = dict(params)
    out[MM_LJ_SIGMA_SCALE_KEY] = jnp.asarray(sigma_scale, dtype=jnp.float32)
    out[MM_LJ_EPSILON_SCALE_KEY] = jnp.asarray(epsilon_scale, dtype=jnp.float32)
    return out


try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import optax  # type: ignore
    from optax import tree_utils as otu  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    optax = None  # type: ignore[assignment]
    otu = None  # type: ignore[assignment]


if jax is not None and jnp is not None and optax is not None and otu is not None:

    @functools.partial(
        jax.jit,
        static_argnames=(
            "model_apply",
            "optimizer_update",
            "batch_size",
            "hybrid_mm",
            "target_lo_kcal",
            "target_hi_kcal",
            "target_mid_kcal",
            "deep_floor_kcal",
            "hard_floor_kcal",
            "center_weight",
            "loss_scale",
            "update_scale",
        ),
    )
    def soft_well_train_step(
        model_apply,
        optimizer_update,
        batch,
        batch_size,
        opt_state,
        params,
        ema_params,
        *,
        hybrid_mm,
        target_lo_kcal: float = DEFAULT_TARGET_LO_KCAL,
        target_hi_kcal: float = DEFAULT_TARGET_HI_KCAL,
        target_mid_kcal: float = DEFAULT_TARGET_MID_KCAL,
        deep_floor_kcal: float = DEFAULT_DEEP_FLOOR_KCAL,
        hard_floor_kcal: float = DEFAULT_HARD_FLOOR_KCAL,
        center_weight: float = 0.25,
        loss_scale: float = 1.0,
        update_scale: float = 1.0,
        ema_decay: float = 0.999,
        frozen_sigma_scale=None,
        frozen_epsilon_scale=None,
    ):
        """One optimizer step on soft-well ``E_int`` window loss."""
        from mmml.models.mm_lj_scales import clip_mm_lj_scale_params
        from mmml.models.physnetjax.physnetjax.training.trainstep import _forward

        def loss_fn(p):
            out = _forward(model_apply, p, batch, batch_size, hybrid_mm=hybrid_mm)
            e_int = out.get("e_int", out["energy"])
            loss = soft_well_e_int_loss(
                e_int,
                target_lo_kcal=target_lo_kcal,
                target_hi_kcal=target_hi_kcal,
                target_mid_kcal=target_mid_kcal,
                deep_floor_kcal=deep_floor_kcal,
                hard_floor_kcal=hard_floor_kcal,
                center_weight=center_weight,
            )
            return float(loss_scale) * loss, e_int

        (loss, e_int), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer_update(grad, opt_state, params)
        updates = otu.tree_scalar_mul(float(update_scale), updates)
        params = optax.apply_updates(params, updates)
        params = clip_mm_lj_scale_params(
            params,
            sigma_bounds=getattr(hybrid_mm, "mm_lj_sigma_scale_bounds", (0.95, 1.05)),
            epsilon_bounds=getattr(
                hybrid_mm, "mm_lj_epsilon_scale_bounds", (0.25, 4.0)
            ),
            trainable_mask=getattr(hybrid_mm, "mm_lj_trainable_mask", None),
        )
        if frozen_sigma_scale is not None and frozen_epsilon_scale is not None:
            params = restore_mm_lj_scales(
                params, frozen_sigma_scale, frozen_epsilon_scale
            )
        ema_params = jax.tree_util.tree_map(
            lambda ema, new: ema_decay * ema + (1.0 - ema_decay) * new,
            ema_params,
            params,
        )
        e_med = jnp.median(jnp.asarray(e_int).reshape(-1) * EV_TO_KCAL_MOL)
        return params, ema_params, opt_state, loss, e_med

else:  # pragma: no cover

    def soft_well_train_step(*_args, **_kwargs):
        raise ModuleNotFoundError("jax and optax required for soft_well_train_step")
