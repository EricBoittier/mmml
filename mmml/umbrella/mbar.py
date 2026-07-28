"""MBAR post-processing for distance umbrella windows (1D or 2D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mmml.umbrella.config import UmbrellaMbarConfig
from mmml.umbrella.energy import make_single_ml_energy_fn, numpy_bias_matrix_nd
from mmml.umbrella.io import (
    SNAPSHOTS_NPZ,
    SUMMARY_JSON,
    load_snapshots,
    load_summary,
    merge_mbar_into_summary,
)

_EV_TO_KCAL = 23.060547830619027
_K_B_EV = 8.617333262145e-5  # eV/K


def fill_u_kln(
    *,
    positions: np.ndarray,
    atom_pairs: Sequence[tuple[int, int]],
    targets_per_cv: Sequence[Sequence[float]],
    k_per_cv: Sequence[Sequence[float]],
    temperature_K: float,
    ml_energy_fn: Callable[[np.ndarray], float] | None = None,
    unbiased_energies: np.ndarray | None = None,
    box: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build reduced-potential tensor ``u_kln`` and sample counts ``N_k``.

    ``positions`` shape: ``(K, N_frames, N_atoms, 3)``.
    ``u_kln[k, l, n] = β (U_unbiased(R_k^n) + W_l(R_k^n))``.

    Pass either ``ml_energy_fn`` (gas-phase recomputation) or precomputed
    ``unbiased_energies`` with shape ``(K, N_frames)`` (hybrid mechanical
    embedding). When ``box`` is set, bias distances use minimum-image.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 4:
        raise ValueError(f"positions must be (K, N_frames, N, 3), got {pos.shape}")
    k_windows, n_frames, _, _ = pos.shape
    for row in list(targets_per_cv) + list(k_per_cv):
        if len(row) != k_windows:
            raise ValueError("targets / k rows must match K windows")
    if ml_energy_fn is None and unbiased_energies is None:
        raise ValueError("need ml_energy_fn or unbiased_energies")
    if unbiased_energies is not None:
        u_store = np.asarray(unbiased_energies, dtype=np.float64)
        if u_store.shape != (k_windows, n_frames):
            raise ValueError(
                f"unbiased_energies shape {u_store.shape} != {(k_windows, n_frames)}"
            )

    beta = 1.0 / (_K_B_EV * float(temperature_K))
    n_k = np.full(k_windows, n_frames, dtype=np.int64)
    u_kln = np.zeros((k_windows, k_windows, n_frames), dtype=np.float64)

    if box is not None:
        from mmml.umbrella.hybrid import mic_distance

        def _bias_row(r: np.ndarray) -> np.ndarray:
            out = np.zeros(k_windows, dtype=np.float64)
            for l in range(k_windows):
                total = 0.0
                for dim, (i, j) in enumerate(atom_pairs):
                    d = mic_distance(r, int(i), int(j), box)
                    total += 0.5 * float(k_per_cv[dim][l]) * (
                        d - float(targets_per_cv[dim][l])
                    ) ** 2
                out[l] = total
            return out
    else:
        def _bias_row(r: np.ndarray) -> np.ndarray:
            return numpy_bias_matrix_nd(r, atom_pairs, targets_per_cv, k_per_cv)

    for k in range(k_windows):
        for n in range(n_frames):
            r = pos[k, n]
            if unbiased_energies is not None:
                u_ml = float(u_store[k, n])
            else:
                assert ml_energy_fn is not None
                u_ml = float(ml_energy_fn(r))
            w_l = _bias_row(r)
            u_kln[k, :, n] = beta * (u_ml + w_l)
    return u_kln, n_k


def subsample_u_kln(
    u_kln: np.ndarray,
    n_k: np.ndarray,
    *,
    timeseries_module: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Subsample correlated frames along the diagonal self-potential."""
    if timeseries_module is None:
        from pymbar import timeseries as timeseries_module

    k_windows = int(u_kln.shape[0])
    g_k: list[float] = []
    selected: list[np.ndarray] = []
    for k in range(k_windows):
        u_self = u_kln[k, k, : int(n_k[k])]
        if u_self.size < 2:
            g_est = 1.0
            idx = np.arange(u_self.size, dtype=int)
        else:
            g_est = float(timeseries_module.statistical_inefficiency(u_self))
            g_est = max(1.0, g_est)
            idx = np.asarray(
                timeseries_module.subsample_correlated_data(u_self, g=g_est),
                dtype=int,
            )
            if idx.size == 0:
                idx = np.array([u_self.size - 1], dtype=int)
        g_k.append(g_est)
        selected.append(idx)

    n_k_eff = np.array([idx.size for idx in selected], dtype=np.int64)
    n_max_eff = int(n_k_eff.max()) if n_k_eff.size else 0
    u_eff = np.zeros((k_windows, k_windows, n_max_eff), dtype=np.float64)
    for k in range(k_windows):
        for j, n_old in enumerate(selected[k]):
            u_eff[k, :, j] = u_kln[k, :, int(n_old)]
    return u_eff, n_k_eff, g_k


def _snap_atom_pairs(snap: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = [(int(snap["atom_i"]), int(snap["atom_j"]))]
    if "atom_k" in snap and "atom_l" in snap:
        pairs.append(
            (
                int(np.asarray(snap["atom_k"]).reshape(-1)[0]),
                int(np.asarray(snap["atom_l"]).reshape(-1)[0]),
            )
        )
    return pairs


def run_umbrella_mbar(cfg: UmbrellaMbarConfig) -> dict[str, Any]:
    """Run pymbar MBAR on ``umbrella_snapshots.npz`` in ``cfg.run_dir``."""
    try:
        from pymbar import MBAR
    except ImportError as exc:
        raise SystemExit(
            "pymbar is required for umbrella MBAR. Install with: "
            "uv sync --extra mbar   or   pip install 'pymbar>=4.0'"
        ) from exc

    import jax
    import jax.numpy as jnp

    from mmml.umbrella.checkpoint import load_params_and_model

    jax.config.update("jax_enable_x64", True)

    run_dir = Path(cfg.run_dir).expanduser().resolve()
    snap_path = run_dir / SNAPSHOTS_NPZ
    if not snap_path.is_file():
        raise FileNotFoundError(
            f"Missing snapshots file: {snap_path} (run mmml umbrella-sample first)"
        )
    snap = load_snapshots(snap_path)

    temperature_K = cfg.temperature_K
    checkpoint = cfg.checkpoint
    if temperature_K is None or checkpoint is None:
        summary_path = run_dir / SUMMARY_JSON
        summary = load_summary(summary_path) if summary_path.is_file() else {}
        args = summary.get("args") or {}
        if temperature_K is None:
            temperature_K = float(
                snap.get("temperature_K")
                or args.get("temperature_K")
                or 300.0
            )
        if checkpoint is None:
            checkpoint = Path(
                str(snap.get("checkpoint") or args.get("checkpoint") or "")
            )
            if not str(checkpoint):
                raise ValueError(
                    "checkpoint required: pass --checkpoint or store it in snapshots/summary"
                )

    checkpoint = Path(checkpoint).expanduser().resolve()
    positions = np.asarray(snap["positions"], dtype=np.float64)
    z = np.asarray(snap["Z"], dtype=np.int32)
    n_atoms = int(z.shape[0])
    atom_pairs = _snap_atom_pairs(snap)
    xi0 = np.asarray(snap["xi0"], dtype=np.float64)
    k_x = np.asarray(snap["k_ev_A2"], dtype=np.float64)
    targets_per_cv: list[list[float]] = [xi0.tolist()]
    k_per_cv: list[list[float]] = [k_x.tolist()]
    yi0 = None
    if "yi0" in snap:
        yi0 = np.asarray(snap["yi0"], dtype=np.float64)
        k_y = np.asarray(snap.get("k_y_ev_A2", k_x), dtype=np.float64)
        targets_per_cv.append(yi0.tolist())
        k_per_cv.append(k_y.tolist())

    engine = "packed_ml"
    if "engine" in snap:
        eng = snap["engine"]
        engine = str(eng.item() if getattr(eng, "ndim", 1) == 0 else eng)
    summary_path = run_dir / SUMMARY_JSON
    if summary_path.is_file():
        engine = str(load_summary(summary_path).get("engine") or engine)

    box = None
    if "box" in snap:
        box = np.asarray(snap["box"], dtype=np.float64)

    if engine == "hybrid_jaxmd" or "energies_unbiased_ev" in snap:
        if "energies_unbiased_ev" not in snap:
            raise ValueError(
                "hybrid_jaxmd MBAR requires energies_unbiased_ev in umbrella_snapshots.npz"
            )
        u_kln, n_k = fill_u_kln(
            positions=positions,
            atom_pairs=atom_pairs,
            targets_per_cv=targets_per_cv,
            k_per_cv=k_per_cv,
            temperature_K=float(temperature_K),
            unbiased_energies=np.asarray(snap["energies_unbiased_ev"], dtype=np.float64),
            box=box,
        )
    else:
        params, model = load_params_and_model(checkpoint, natoms=n_atoms, prefer_ema=True)
        ml_fn = make_single_ml_energy_fn(
            model_apply=model.apply,
            params=params,
            atomic_numbers=z,
            n_atoms=n_atoms,
        )
        ml_fn_jit = jax.jit(ml_fn)

        def ml_energy_np(r: np.ndarray) -> float:
            return float(ml_fn_jit(jnp.asarray(r, dtype=jnp.float64)))

        u_kln, n_k = fill_u_kln(
            positions=positions,
            atom_pairs=atom_pairs,
            targets_per_cv=targets_per_cv,
            k_per_cv=k_per_cv,
            temperature_K=float(temperature_K),
            ml_energy_fn=ml_energy_np,
        )
    if np.any(n_k == 0):
        return {
            "error": "MBAR skipped: at least one umbrella window has no snapshots.",
            "N_k": n_k.tolist(),
        }

    u_eff, n_k_eff, g_k = subsample_u_kln(u_kln, n_k)
    mbar = MBAR(u_eff, n_k_eff, verbose=cfg.mbar_verbose)
    fe = mbar.compute_free_energy_differences(compute_uncertainty=True)

    delta_f = np.asarray(fe["Delta_f"], dtype=np.float64)
    d_delta_f = np.asarray(fe["dDelta_f"], dtype=np.float64)
    kbt = _K_B_EV * float(temperature_K)
    pmf_kt = delta_f[0, :].copy()
    pmf_kt -= pmf_kt.min()
    pmf_ev = pmf_kt * kbt
    d_pmf_ev = d_delta_f[0, :] * kbt

    grid_shape = None
    if "grid_shape" in snap:
        grid_shape = [int(x) for x in np.asarray(snap["grid_shape"]).tolist()]

    result: dict[str, Any] = {
        "temperature_K": float(temperature_K),
        "ndim": len(atom_pairs),
        "atom_pairs": [list(p) for p in atom_pairs],
        "xi0": xi0.tolist(),
        "yi0": None if yi0 is None else yi0.tolist(),
        "k_ev_A2": k_x.tolist(),
        "grid_shape": grid_shape,
        "Delta_f_kT": delta_f.tolist(),
        "dDelta_f_kT": d_delta_f.tolist(),
        "pmf_rel_kT": pmf_kt.tolist(),
        "pmf_rel_eV": pmf_ev.tolist(),
        "d_pmf_rel_eV": d_pmf_ev.tolist(),
        "pmf_rel_kcal_mol": (pmf_ev * _EV_TO_KCAL).tolist(),
        "d_pmf_rel_kcal_mol": (d_pmf_ev * _EV_TO_KCAL).tolist(),
        "N_k": n_k.tolist(),
        "N_k_effective": n_k_eff.tolist(),
        "g_k": g_k,
        "engine": engine,
        "note": (
            "PMF is F(window) − min F from MBAR; "
            "u_kln = β(U_unbiased + Σ_d W_d,l). For 2D, reshape pmf_* with grid_shape."
        ),
    }
    if grid_shape is not None and len(grid_shape) == 2:
        nx, ny = grid_shape
        if nx * ny == len(pmf_kt):
            result["pmf_rel_kcal_mol_2d"] = (
                (pmf_ev * _EV_TO_KCAL).reshape(nx, ny).tolist()
            )
            result["pmf_rel_eV_2d"] = pmf_ev.reshape(nx, ny).tolist()
    merge_mbar_into_summary(run_dir, result)
    return result
