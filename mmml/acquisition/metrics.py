"""Held-out energy / force metrics with composition offsets.

Energy errors are reported both raw and after subtracting per-atom reference
energies (when the table is available), plus a per-atom normalization
``ΔE / N``.  Force RMSE / MAE / tails are over real atoms only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from mmml.acquisition.ids import composition_key


def _real_mask(Z: np.ndarray, N: np.ndarray | None) -> np.ndarray:
    z = np.asarray(Z)
    if N is None:
        return z > 0
    n = np.asarray(N, dtype=np.int32).reshape(-1)
    mask = np.zeros(z.shape, dtype=bool)
    for i, ni in enumerate(n):
        mask[i, : int(ni)] = True
    return mask & (z > 0)


def energy_errors(
    e_pred: np.ndarray,
    e_ref: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray | None = None,
    *,
    subtract_atom_refs: bool = True,
    energy_unit: str = "ev",
) -> dict[str, Any]:
    ep = np.asarray(e_pred, dtype=np.float64).reshape(-1)
    er = np.asarray(e_ref, dtype=np.float64).reshape(-1)
    if ep.shape != er.shape:
        raise ValueError("prediction / reference energy length mismatch")
    finite = np.isfinite(ep) & np.isfinite(er)
    z = np.asarray(Z)
    n_atoms = np.asarray(N, dtype=np.float64).reshape(-1) if N is not None else (z > 0).sum(axis=-1).astype(np.float64)
    n_atoms = np.maximum(n_atoms, 1.0)
    delta = ep - er
    out: dict[str, Any] = {
        "n": int(finite.sum()),
        "n_total": int(len(ep)),
        "energy_mae": _mae(delta[finite]),
        "energy_rmse": _rmse(delta[finite]),
        "energy_mae_per_atom": _mae((delta / n_atoms)[finite]),
        "energy_rmse_per_atom": _rmse((delta / n_atoms)[finite]),
    }
    if subtract_atom_refs:
        try:
            from mmml.data.units import subtract_atom_refs as sub

            ep_s = sub(ep, z, energy_unit=energy_unit)
            er_s = sub(er, z, energy_unit=energy_unit)
            ds = ep_s - er_s
            out["energy_mae_atomref"] = _mae(ds[finite])
            out["energy_rmse_atomref"] = _rmse(ds[finite])
            out["energy_mae_atomref_per_atom"] = _mae((ds / n_atoms)[finite])
        except Exception as exc:  # table missing in some test envs
            out["atomref_error"] = repr(exc)
    return out


def force_errors(
    f_pred: np.ndarray,
    f_ref: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray | None = None,
) -> dict[str, Any]:
    fp = np.asarray(f_pred, dtype=np.float64)
    fr = np.asarray(f_ref, dtype=np.float64)
    mask = _real_mask(Z, N)
    if fp.shape != fr.shape:
        raise ValueError(f"force shape mismatch {fp.shape} vs {fr.shape}")
    diff = fp - fr
    abs_c = np.abs(diff[mask])
    sq = diff[mask] ** 2
    finite = np.isfinite(abs_c)
    vals = abs_c[finite]
    atom_mag = np.sqrt((diff ** 2).sum(axis=-1))
    atom_mask = mask if mask.ndim == 2 else mask.any(axis=-1)
    if atom_mag.ndim == atom_mask.ndim:
        atom_err = atom_mag[atom_mask]
    else:
        atom_err = atom_mag.reshape(-1)
    atom_err = atom_err[np.isfinite(atom_err)]
    return {
        "force_mae": float(vals.mean()) if vals.size else float("nan"),
        "force_rmse": float(np.sqrt(sq[finite].mean())) if finite.any() else float("nan"),
        "force_max_abs": float(vals.max()) if vals.size else float("nan"),
        "force_p95_abs": float(np.percentile(vals, 95)) if vals.size else float("nan"),
        "force_p99_abs": float(np.percentile(vals, 99)) if vals.size else float("nan"),
        "force_atom_rmse": float(np.sqrt(np.mean(atom_err ** 2))) if atom_err.size else float("nan"),
        "n_force_components": int(finite.sum()),
    }


def grouped_errors(
    e_pred: np.ndarray,
    e_ref: np.ndarray,
    f_pred: np.ndarray,
    f_ref: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray,
    *,
    compositions: list[str] | None = None,
    conditions: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    n = len(np.asarray(e_pred).reshape(-1))
    if compositions is None:
        compositions = [
            composition_key(Z[i], int(N[i])) for i in range(n)
        ]
    if conditions is None:
        conditions = ["unspecified"] * n
    buckets: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[f"comp:{compositions[i]}"].append(i)
        buckets[f"cond:{conditions[i]}"].append(i)
        buckets[f"comp+cond:{compositions[i]}|{conditions[i]}"].append(i)
    out: dict[str, dict[str, Any]] = {}
    ep = np.asarray(e_pred).reshape(-1)
    er = np.asarray(e_ref).reshape(-1)
    for name, idx in buckets.items():
        ii = np.asarray(idx, dtype=np.int64)
        metrics = energy_errors(ep[ii], er[ii], Z[ii], N[ii], subtract_atom_refs=False)
        metrics.update(force_errors(f_pred[ii], f_ref[ii], Z[ii], N[ii]))
        metrics["n_structures"] = int(len(ii))
        out[name] = metrics
    return out


def evaluate_predictions(
    pred: Mapping[str, np.ndarray],
    ref: Mapping[str, np.ndarray],
    *,
    compositions: list[str] | None = None,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    metrics = energy_errors(pred["E"], ref["E"], ref["Z"], ref.get("N"), subtract_atom_refs=True)
    metrics.update(force_errors(pred["F"], ref["F"], ref["Z"], ref.get("N")))
    metrics["by_group"] = grouped_errors(
        pred["E"], ref["E"], pred["F"], ref["F"], ref["Z"], ref["N"],
        compositions=compositions, conditions=conditions,
    )
    return metrics


def _mae(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else float("nan")


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")
