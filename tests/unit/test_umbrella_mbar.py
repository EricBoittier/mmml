"""Unit tests for umbrella MBAR reduced-potential assembly."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.io import (
    SNAPSHOTS_NPZ,
    load_snapshots,
    merge_mbar_into_summary,
    save_snapshots,
)
from mmml.umbrella.mbar import fill_u_kln, subsample_u_kln


def test_fill_u_kln_self_term_and_shape():
    # Two windows, two frames, diatomic along x
    k, n_frames, n_atoms = 2, 2, 2
    positions = np.zeros((k, n_frames, n_atoms, 3), dtype=np.float64)
    # window 0 samples near 1.0 Å; window 1 near 2.0 Å
    positions[0, :, 1, 0] = 1.0
    positions[1, :, 1, 0] = 2.0
    xi0 = np.array([1.0, 2.0])
    k_arr = np.array([2.0, 2.0])
    temperature_K = 300.0
    k_b = 8.617333262145e-5
    beta = 1.0 / (k_b * temperature_K)

    def ml_energy(_r: np.ndarray) -> float:
        return 0.25  # constant unbiased energy

    u_kln, n_k = fill_u_kln(
        positions=positions,
        atom_pairs=((0, 1),),
        targets_per_cv=(xi0.tolist(),),
        k_per_cv=(k_arr.tolist(),),
        temperature_K=temperature_K,
        ml_energy_fn=ml_energy,
    )
    assert u_kln.shape == (2, 2, 2)
    np.testing.assert_array_equal(n_k, [2, 2])

    # Self terms: W_k(R_k)=0 → u_kk = β U_ML
    assert u_kln[0, 0, 0] == pytest.approx(beta * 0.25)
    assert u_kln[1, 1, 0] == pytest.approx(beta * 0.25)

    # Cross: sample from window 0 (r=1) under window 1 bias W=0.5*2*(1-2)^2=1
    assert u_kln[0, 1, 0] == pytest.approx(beta * (0.25 + 1.0))
    # Sample from window 1 (r=2) under window 0 bias same
    assert u_kln[1, 0, 0] == pytest.approx(beta * (0.25 + 1.0))


def test_fill_u_kln_accepts_linear_distance_cv_with_box():
    """Hybrid MBAR passes LinearDistanceCV + PBC box (regression for unpack bug)."""
    from mmml.md.restraints import LinearDistanceCV

    k, n_frames, n_atoms = 2, 2, 2
    positions = np.zeros((k, n_frames, n_atoms, 3), dtype=np.float64)
    positions[0, :, 1, 0] = 1.5
    positions[1, :, 1, 0] = 2.5
    box = np.diag([20.0, 20.0, 20.0])
    u_unb = np.full((k, n_frames), 0.5, dtype=np.float64)
    cv = LinearDistanceCV.distance(0, 1)
    u_kln, n_k = fill_u_kln(
        positions=positions,
        atom_pairs=[cv],
        targets_per_cv=[[1.5, 2.5]],
        k_per_cv=[[4.0, 4.0]],
        temperature_K=300.0,
        unbiased_energies=u_unb,
        box=box,
    )
    assert u_kln.shape == (2, 2, 2)
    assert n_k.tolist() == [2, 2]
    assert np.all(np.isfinite(u_kln))
    # Self-bias vanishes at the window targets → u_kk = β U_unbiased
    beta = 1.0 / (8.617333262145e-5 * 300.0)
    assert u_kln[0, 0, 0] == pytest.approx(beta * 0.5)
    assert u_kln[1, 1, 0] == pytest.approx(beta * 0.5)


def test_subsample_u_kln_with_stub_timeseries():
    class _TS:
        @staticmethod
        def statistical_inefficiency(u_self):
            return 1.0

        @staticmethod
        def subsample_correlated_data(u_self, g=1.0):
            del g
            return np.arange(0, len(u_self), 2)

    u = np.zeros((2, 2, 4))
    u[0, 0] = np.arange(4)
    u[1, 1] = np.arange(4) + 10
    n_k = np.array([4, 4])
    u_eff, n_eff, g_k = subsample_u_kln(u, n_k, timeseries_module=_TS)
    assert n_eff.tolist() == [2, 2]
    assert g_k == [1.0, 1.0]
    assert u_eff.shape == (2, 2, 2)
    np.testing.assert_allclose(u_eff[0, 0], [0.0, 2.0])


def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / SNAPSHOTS_NPZ
    positions = np.random.randn(3, 5, 4, 3)
    save_snapshots(
        path,
        positions=positions,
        Z=np.array([6, 1, 1, 1], dtype=np.int32),
        atom_i=0,
        atom_j=1,
        xi0=np.array([1.0, 1.5, 2.0]),
        k_ev_A2=np.array([10.0, 10.0, 10.0]),
        temperature_K=298.15,
        dt_fs=0.5,
        cv_traj=np.ones((3, 5)),
        checkpoint="/tmp/ckpt",
    )
    snap = load_snapshots(path)
    assert snap["positions"].shape == (3, 5, 4, 3)
    assert snap["atom_i"] == 0
    assert snap["atom_j"] == 1
    assert snap["temperature_K"] == pytest.approx(298.15)
    assert snap["checkpoint"] == "/tmp/ckpt"

    summary_path = merge_mbar_into_summary(tmp_path, {"pmf_rel_eV": [0.0, 0.1]})
    assert summary_path.is_file()
