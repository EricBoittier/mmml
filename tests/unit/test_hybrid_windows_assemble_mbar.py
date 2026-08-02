"""Assemble per-window checkpoints into snapshots and run MBAR on them.

Offline stand-in for the Snakemake ``window → assemble → mbar`` tail: synthetic
``windows/wXXX.npz`` files are packed exactly the way
:func:`mmml.umbrella.hybrid.run_umbrella_hybrid_nvt` packs them, then handed to
the same MBAR entry point the workflow calls, so the failed-window bookkeeping
and the antisymmetric CV round-trip are checked without a GPU or CHARMM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.umbrella.config import UmbrellaMbarConfig
from mmml.umbrella.hybrid_windows import load_all_window_arrays, save_window_checkpoint
from mmml.umbrella.io import SNAPSHOTS_NPZ, SUMMARY_JSON, save_snapshots
from mmml.umbrella.mbar import run_umbrella_mbar

pymbar = pytest.importorskip("pymbar")

K_B_EV = 8.617333262145e-5
TEMPERATURE_K = 300.0
# xi = r(C-Cl) - r(C-N), as in the NH3 + CH3Cl campaign.
CV_SPEC = [{"pairs": [[0, 2], [0, 1]], "coefficients": [1.0, -1.0]}]
Z = np.array([6, 7, 17], dtype=np.int32)
BOX = np.diag([30.0, 30.0, 30.0])
R_SUM = 4.0  # r(C-Cl) + r(C-N), held fixed so xi alone parameterises a frame
K_BIAS = 0.4  # eV/A**2, soft enough that neighbouring windows overlap
A_WELL = 0.2  # eV/A**2 curvature of the underlying (unbiased) well at xi = 0
N_FRAMES = 60
FAILED_WID = 7


def _positions_for_xi(xi: np.ndarray) -> np.ndarray:
    """Collinear Cl-C-N frames whose CV value is exactly ``xi``."""
    pos = np.zeros((xi.shape[0], 3, 3), dtype=np.float64)
    pos[:, 1, 0] = -(R_SUM - xi) / 2.0  # N
    pos[:, 2, 0] = (R_SUM + xi) / 2.0  # Cl
    return pos


def _sample_window(xi0: float, rng: np.random.Generator) -> np.ndarray:
    """Draw xi from exp(-beta[0.5*A*xi^2 + 0.5*K*(xi-xi0)^2])."""
    k_eff = A_WELL + K_BIAS
    center = K_BIAS * xi0 / k_eff
    sigma = np.sqrt(K_B_EV * TEMPERATURE_K / k_eff)
    return rng.normal(center, sigma, size=N_FRAMES)


def _write_windows(run_dir: Path, xi0: np.ndarray) -> None:
    rng = np.random.default_rng(20260731)
    for wid, x0 in enumerate(xi0):
        if wid == FAILED_WID:
            nan_pos = np.full((N_FRAMES, 3, 3), np.nan)
            nan_1d = np.full(N_FRAMES, np.nan)
            save_window_checkpoint(
                run_dir,
                wid,
                status="failed",
                positions=nan_pos,
                cv=nan_1d,
                energies=nan_1d,
                energies_unbiased=nan_1d,
                xi0=float(x0),
                k_ev_A2=K_BIAS,
                fail_reason="seed max|F| exceeds max_seed_force",
            )
            continue
        xi = _sample_window(float(x0), rng)
        unbiased = 0.5 * A_WELL * xi**2
        bias = 0.5 * K_BIAS * (xi - float(x0)) ** 2
        save_window_checkpoint(
            run_dir,
            wid,
            status="ok",
            positions=_positions_for_xi(xi),
            cv=xi,
            energies=unbiased + bias,
            energies_unbiased=unbiased,
            xi0=float(x0),
            k_ev_A2=K_BIAS,
        )


def _assemble(run_dir: Path, xi0: np.ndarray) -> list[int]:
    """Mirror the packing step of ``run_umbrella_hybrid_nvt``."""
    k = int(xi0.shape[0])
    positions, cv, energies, e_unb, failed, reasons = load_all_window_arrays(
        run_dir, k, n_frames=N_FRAMES, n_atoms=3
    )
    save_snapshots(
        run_dir / SNAPSHOTS_NPZ,
        positions=positions,
        Z=Z,
        atom_i=0,
        atom_j=2,
        xi0=xi0,
        k_ev_A2=np.full(k, K_BIAS),
        temperature_K=TEMPERATURE_K,
        dt_fs=0.25,
        cv_traj=cv[..., None],
        checkpoint=str(run_dir / "model_ext.json"),
        extra={
            "ndim": np.int32(1),
            "engine": np.asarray("hybrid_jaxmd"),
            "energies_ev": energies,
            "energies_unbiased_ev": e_unb,
            "cv_spec": np.asarray(json.dumps(CV_SPEC)),
            "failed_windows": np.asarray(failed, dtype=np.int32),
            "fail_reasons": np.asarray(
                json.dumps({str(w): r for w, r in reasons.items()})
            ),
            "box": BOX,
        },
    )
    (run_dir / SUMMARY_JSON).write_text(
        json.dumps({"engine": "hybrid_jaxmd", "n_windows": k}) + "\n"
    )
    return failed


@pytest.fixture(scope="module")
def mbar_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("umbrella_run")
    xi0 = np.linspace(-1.0, 1.0, 9)
    _write_windows(run_dir, xi0)
    failed = _assemble(run_dir, xi0)
    result = run_umbrella_mbar(UmbrellaMbarConfig(run_dir=run_dir))
    return run_dir, xi0, failed, result


def test_assemble_reports_the_failed_window(mbar_run):
    run_dir, xi0, failed, _ = mbar_run
    assert failed == [FAILED_WID]
    assert all(
        (run_dir / "windows" / f"w{w:03d}.npz").is_file() for w in range(len(xi0))
    )


def test_mbar_drops_failed_window_but_keeps_grid_indexing(mbar_run):
    _, xi0, _, result = mbar_run
    assert "error" not in result
    assert result["failed_windows"] == [FAILED_WID]
    assert result["n_windows_used"] == len(xi0) - 1

    pmf = np.asarray(result["pmf_rel_kcal_mol"], dtype=np.float64)
    assert pmf.shape == xi0.shape
    assert np.isnan(pmf[FAILED_WID])
    assert np.all(np.isfinite(np.delete(pmf, FAILED_WID)))


def test_mbar_recovers_a_well_centred_on_xi_zero(mbar_run):
    """Sign of the bias subtraction: the PMF minimum must sit at the well."""
    _, xi0, _, result = mbar_run
    pmf = np.asarray(result["pmf_rel_kcal_mol"], dtype=np.float64)
    assert xi0[int(np.nanargmin(pmf))] == pytest.approx(0.0, abs=0.26)
    assert np.nanmin(pmf) == pytest.approx(0.0, abs=1e-9)
    # Outermost kept windows are uphill from the minimum.
    assert pmf[0] > 0.1 and pmf[-1] > 0.1


def test_run_mbar_status_contract(mbar_run):
    """What scripts/run_mbar.sh reads out of umbrella_summary.json."""
    run_dir, _, _, _ = mbar_run
    summary = json.loads((run_dir / SUMMARY_JSON).read_text())
    mbar = summary["mbar"]
    assert "error" not in mbar
    assert "pmf_rel_kcal_mol" in mbar
    status = {
        "ok": True,
        "run_dir": str(run_dir),
        "n_windows_used": mbar.get("n_windows_used"),
        "failed_windows": mbar.get("failed_windows"),
    }
    assert json.loads(json.dumps(status))["failed_windows"] == [FAILED_WID]
