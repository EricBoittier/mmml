"""Multitask output heads for SpookyPhysNet (charges, spins, HOMO / gap).

All heads read the final scalar atomic features ``x_i`` (the model's ``state``
output, shape ``(N, 1, 1, F)``) and are deliberately minimal (Dense layers +
per-element bias tables) so that a C++ re-implementation (mlip.cpp) can load the
same weights from a GGUF file. See SPOOKY_MULTITASK.md for the parameter table.

Constraints follow the PhysNet convention: a uniform per-molecule correction

    q_i <- q_i + (Q - sum_j q_j) / N        (sum over real atoms of the molecule)
    s_i <- s_i + ((mult - 1) - sum_j s_j) / N

so sum_i q_i == Q and sum_i s_i == 2S == mult - 1 hold to float rounding.

The functions in the second half (``prepare_multitask_cache_keys``,
``multitask_loss_terms``) are host/loss helpers used by
``scripts/train_so3lr_spooky_extxyz.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import ase.data
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Pure-JAX building blocks (model side)
# ---------------------------------------------------------------------------


def uniform_sum_constraint(
    raw: jnp.ndarray,
    target_per_atom: jnp.ndarray,
    atom_mask: jnp.ndarray,
    batch_segments: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """Shift per-atom values so each molecule sums to its target.

    ``target_per_atom`` is the molecular target broadcast to that molecule's atoms
    (how SpookyPhysNet receives Q and S); padding atoms are excluded via
    ``atom_mask`` and returned as exactly 0.
    """
    raw = raw.reshape(-1)
    mask = atom_mask.reshape(-1).astype(raw.dtype)
    target = target_per_atom.reshape(-1).astype(raw.dtype)
    n_mol = jax.ops.segment_sum(mask, batch_segments, num_segments=batch_size)
    sum_raw = jax.ops.segment_sum(raw * mask, batch_segments, num_segments=batch_size)
    n_i = jnp.maximum(jnp.take(n_mol, batch_segments), 1.0)
    corr = (target - jnp.take(sum_raw, batch_segments)) / n_i
    return (raw + corr) * mask


def masked_com_dipole(
    positions: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    batch_segments: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """mu = sum_i q_i (r_i - r_COM), COM from real atoms only. Units e*Angstrom.

    Referencing to the centre of mass makes mu origin-independent for charged
    molecules too (the existing ``molecular_dipole_from_charges`` lets padding
    atoms, Z=0 -> mass 1, shift the COM; this version masks them).
    """
    mask = atom_mask.reshape(-1).astype(positions.dtype)
    masses = jnp.take(jnp.asarray(ase.data.atomic_masses, dtype=positions.dtype), atomic_numbers) * mask
    m_tot = jax.ops.segment_sum(masses, batch_segments, num_segments=batch_size)
    mr = jax.ops.segment_sum(positions * masses[:, None], batch_segments, num_segments=batch_size)
    com = mr / jnp.maximum(m_tot, 1e-12)[:, None]
    rel = positions - jnp.take(com, batch_segments, axis=0)
    q = charges.reshape(-1) * mask
    return jax.ops.segment_sum(rel * q[:, None], batch_segments, num_segments=batch_size)


def masked_mean_pool(
    values: jnp.ndarray,
    atom_mask: jnp.ndarray,
    batch_segments: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """Per-molecule mean over real atoms of ``values`` (N, C) -> (B, C)."""
    mask = atom_mask.reshape(-1).astype(values.dtype)
    s = jax.ops.segment_sum(values * mask[:, None], batch_segments, num_segments=batch_size)
    n = jax.ops.segment_sum(mask, batch_segments, num_segments=batch_size)
    return s / jnp.maximum(n, 1.0)[:, None]


# ---------------------------------------------------------------------------
# Cache / batch helpers (host side, numpy)
# ---------------------------------------------------------------------------

#: per-atom optional cache keys, shape (n_atoms_total,), row-aligned with Z/R/F
MT_ATOM_KEYS = ("q_atom", "s_atom")
#: per-molecule optional cache keys, shape (n_mol, 1), row-aligned with E/Q/S
MT_MOL_KEYS = ("has_q", "has_spin", "has_orb", "homo", "gap")


def _per_mol_all(flags_atom: np.ndarray, mol_offsets: np.ndarray) -> np.ndarray:
    """True where every atom of the molecule satisfies ``flags_atom``."""
    bad = np.add.reduceat((~flags_atom).astype(np.int64), mol_offsets[:-1])
    return bad == 0


def prepare_multitask_cache_keys(
    data: Dict[str, np.ndarray],
    *,
    charges: bool,
    spins: bool,
    orbitals: bool,
) -> Dict[str, np.ndarray]:
    """Validate/complete the multitask label keys of a flat cache (in place).

    Missing masks are inferred (``has_q``/``has_spin``: all atom labels finite;
    ``has_spin`` additionally requires mult > 1 when absent from the cache;
    ``has_orb``: homo and gap finite). Non-finite labels are zeroed so the masked
    loss can never produce NaN gradients. Masks are stored as float32 (n_mol, 1).
    """
    off = np.asarray(data["mol_offsets"], dtype=np.int64)
    n_mol = off.shape[0] - 1
    n_at = int(off[-1])

    def _need(key, shape_last):
        if key not in data:
            raise KeyError(
                f"multitask head enabled but cache has no '{key}' "
                "(see SPOOKY_MULTITASK.md for the cache key spec)"
            )
        arr = np.asarray(data[key], dtype=np.float64)
        return arr.reshape(-1) if shape_last is None else arr.reshape(-1, shape_last)

    if charges:
        q = _need("q_atom", None)
        if q.shape[0] != n_at:
            raise ValueError(f"q_atom has {q.shape[0]} rows, expected n_atoms_total={n_at}")
        fin = np.isfinite(q)
        has_q = np.asarray(data["has_q"], dtype=bool).reshape(-1) if "has_q" in data else _per_mol_all(fin, off)
        data["q_atom"] = np.where(fin, q, 0.0)
        data["has_q"] = has_q.astype(np.float32).reshape(-1, 1)
    if spins:
        s = _need("s_atom", None)
        if s.shape[0] != n_at:
            raise ValueError(f"s_atom has {s.shape[0]} rows, expected n_atoms_total={n_at}")
        fin = np.isfinite(s)
        if "has_spin" in data:
            has_s = np.asarray(data["has_spin"], dtype=bool).reshape(-1)
        else:
            has_s = _per_mol_all(fin, off) & (np.asarray(data["S"]).reshape(-1) > 1.5)
        data["s_atom"] = np.where(fin, s, 0.0)
        data["has_spin"] = (has_s & _per_mol_all(fin, off)).astype(np.float32).reshape(-1, 1)
    if orbitals:
        homo = _need("homo", 1)
        gap = _need("gap", 1)
        if homo.shape[0] != n_mol or gap.shape[0] != n_mol:
            raise ValueError("homo/gap must have one row per structure")
        fin = (np.isfinite(homo) & np.isfinite(gap)).reshape(-1)
        has_o = np.asarray(data["has_orb"], dtype=bool).reshape(-1) & fin if "has_orb" in data else fin
        data["homo"] = np.where(np.isfinite(homo), homo, 0.0)
        data["gap"] = np.where(np.isfinite(gap), gap, 0.0)
        data["has_orb"] = has_o.astype(np.float32).reshape(-1, 1)
    return data


def orbital_offsets_from_data(data: Mapping[str, np.ndarray], indices: np.ndarray) -> tuple[float, float]:
    """Mean HOMO and gap over labelled training structures (used as fixed offsets)."""
    m = np.asarray(data["has_orb"]).reshape(-1)[indices] > 0
    if not np.any(m):
        return 0.0, 0.0
    h = np.asarray(data["homo"]).reshape(-1)[indices][m]
    g = np.asarray(data["gap"]).reshape(-1)[indices][m]
    return float(h.mean()), float(g.mean())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _masked_mse_mae(pred, ref, mask):
    mask = mask.astype(pred.dtype)
    d = (pred - ref) * mask
    n = jnp.sum(mask)
    denom = jnp.maximum(n, 1.0)
    return jnp.sum(d * d) / denom, jnp.sum(jnp.abs(d)) / denom, n


def multitask_loss_terms(
    out: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    charges: bool,
    spins: bool,
    orbitals: bool,
    charge_weight: float,
    spin_weight: float,
    dipole_weight: float,
    homo_weight: float,
    gap_weight: float,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted, masked multitask loss and its metrics.

    Per-atom terms average over labelled real atoms in the batch, molecular terms
    over labelled molecules; a batch with no labels for a head contributes 0 (not
    NaN). The dipole term compares mu(q_pred) with mu(q_ref) -- OMol25 has no DFT
    dipole, so the reference dipole is derived from the label charges with the same
    COM convention.
    """
    seg = batch["batch_segments"]
    atom_mask = batch["atom_mask"].reshape(-1)
    loss = jnp.asarray(0.0, dtype=jnp.float32)
    metrics: Dict[str, jnp.ndarray] = {}

    def _atom_mask_from_mol(key):
        return jnp.take(batch[key].reshape(-1), seg) * atom_mask

    if charges:
        m = _atom_mask_from_mol("has_q")
        mse, mae, _ = _masked_mse_mae(out["mt_charges"].reshape(-1), batch["q_atom"].reshape(-1), m)
        loss = loss + charge_weight * mse
        metrics.update(mt_charge_mse=mse, mt_charge_mae=mae)
        if dipole_weight > 0.0:
            mu_ref = masked_com_dipole(batch["R"], batch["Z"], batch["q_atom"].reshape(-1) * m,
                                       atom_mask, seg, batch_size)
            mol_m = batch["has_q"].reshape(-1, 1) * jnp.ones((1, 3), out["mt_dipole"].dtype)
            dmse, dmae, _ = _masked_mse_mae(out["mt_dipole"], mu_ref, mol_m)
            loss = loss + dipole_weight * dmse
            metrics.update(mt_dipole_mse=dmse, mt_dipole_mae=dmae)
    if spins:
        m = _atom_mask_from_mol("has_spin")
        mse, mae, n = _masked_mse_mae(out["mt_spins"].reshape(-1), batch["s_atom"].reshape(-1), m)
        loss = loss + spin_weight * mse
        # mt_spin_mae is a per-batch mean (0 on batches without open-shell
        # structures); mt_spin_abs_sum / mt_spin_labelled_atoms gives the
        # label-weighted MAE after averaging over batches (ratio of means).
        metrics.update(mt_spin_mse=mse, mt_spin_mae=mae, mt_spin_labelled_atoms=n,
                       mt_spin_abs_sum=mae * n)
    if orbitals:
        m = batch["has_orb"].reshape(-1)
        hmse, hmae, _ = _masked_mse_mae(out["mt_homo"].reshape(-1), batch["homo"].reshape(-1), m)
        gmse, gmae, _ = _masked_mse_mae(out["mt_gap"].reshape(-1), batch["gap"].reshape(-1), m)
        loss = loss + homo_weight * hmse + gap_weight * gmse
        metrics.update(mt_homo_mse=hmse, mt_homo_mae=hmae, mt_gap_mse=gmse, mt_gap_mae=gmae)
    return loss, metrics


__all__ = [
    "MT_ATOM_KEYS",
    "MT_MOL_KEYS",
    "masked_com_dipole",
    "masked_mean_pool",
    "multitask_loss_terms",
    "orbital_offsets_from_data",
    "prepare_multitask_cache_keys",
    "uniform_sum_constraint",
]
