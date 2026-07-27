"""MBAR post-processing for distance umbrella windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from mmml.umbrella.config import UmbrellaMbarConfig
from mmml.umbrella.energy import make_single_ml_energy_fn, numpy_bias_matrix
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
    atom_i: int,
    atom_j: int,
    xi0: np.ndarray,
    k_ev_A2: np.ndarray,
    temperature_K: float,
    ml_energy_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build reduced-potential tensor ``u_kln`` and sample counts ``N_k``.

    ``positions`` shape: ``(K, N_frames, N_atoms, 3)``.
    ``u_kln[k, l, n] = β (U_ML(R_k^n) + W_l(R_k^n))``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 4:
        raise ValueError(f"positions must be (K, N_frames, N, 3), got {pos.shape}")
    k_windows, n_frames, _, _ = pos.shape
    xi0 = np.asarray(xi0, dtype=np.float64).reshape(-1)
    k_arr = np.asarray(k_ev_A2, dtype=np.float64).reshape(-1)
    if xi0.shape[0] != k_windows or k_arr.shape[0] != k_windows:
        raise ValueError("xi0 / k_ev_A2 length must match K windows")

    beta = 1.0 / (_K_B_EV * float(temperature_K))
    n_k = np.full(k_windows, n_frames, dtype=np.int64)
    u_kln = np.zeros((k_windows, k_windows, n_frames), dtype=np.float64)

    for k in range(k_windows):
        for n in range(n_frames):
            r = pos[k, n]
            u_ml = float(ml_energy_fn(r))
            w_l = numpy_bias_matrix(r, atom_i, atom_j, xi0, k_arr)
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

    from mmml.models.physnetjax.physnetjax.restart.restart import get_last, get_params_model

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
    atom_i = int(snap["atom_i"])
    atom_j = int(snap["atom_j"])
    xi0 = np.asarray(snap["xi0"], dtype=np.float64)
    k_arr = np.asarray(snap["k_ev_A2"], dtype=np.float64)

    restart = get_last(str(checkpoint))
    params, model = get_params_model(str(restart), natoms=n_atoms, prefer_ema=True)
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
        atom_i=atom_i,
        atom_j=atom_j,
        xi0=xi0,
        k_ev_A2=k_arr,
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
    # PMF relative to window 0 along ξ₀
    pmf_kt = delta_f[0, :].copy()
    pmf_kt -= pmf_kt.min()
    pmf_ev = pmf_kt * kbt
    d_pmf_ev = d_delta_f[0, :] * kbt

    result = {
        "temperature_K": float(temperature_K),
        "xi0": xi0.tolist(),
        "k_ev_A2": k_arr.tolist(),
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
        "note": (
            "PMF is F(ξ₀) − min_k F(ξ₀) from MBAR window free energies; "
            "u_kln = β(U_ML + W_l)."
        ),
    }
    merge_mbar_into_summary(run_dir, result)
    return result
