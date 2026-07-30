"""Unit tests for hybrid mechanical-embedding umbrella helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.hybrid import (
    find_atom_index_by_name,
    merge_ml_region_mol_id,
    mic_distance,
    resolve_ml_region_indices,
    stretch_distance_seed_mic,
)
from mmml.umbrella.io import load_snapshots, save_snapshots
from mmml.umbrella.mbar import fill_u_kln


def test_resolve_ml_region_indices():
    resnames = ["AMM1", "AMM1", "CH3CL", "TIP3", "TIP3", "TIP3"]
    idx = resolve_ml_region_indices(resnames, ("AMM1", "CH3CL"))
    assert idx.tolist() == [0, 1, 2]


def test_resolve_ml_region_accepts_truncated_ch3cl_alias():
    from mmml.md.ml_region import resolve_ml_region_indices

    resnames = ["AMM1", "AMM1", "CH3C", "CH3C", "TIP3"]
    idx = resolve_ml_region_indices(resnames, ("AMM1", "CH3CL"))
    assert idx.tolist() == [0, 1, 2, 3]


def test_resolve_ml_region_rejects_partial_match():
    from mmml.md.ml_region import resolve_ml_region_indices

    with pytest.raises(ValueError, match="missing residues"):
        resolve_ml_region_indices(["AMM1", "AMM1", "TIP3"], ("AMM1", "CH3CL"))


def test_merge_ml_region_mol_id_excludes_solute_solute():
    mol_id = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    ml = [0, 1, 2, 3]
    merged = merge_ml_region_mol_id(mol_id, ml)
    assert merged[0] == merged[1] == merged[2] == merged[3]
    assert merged[4] == 2
    # Solute–solute pairs share mol_id → intermolecular filter drops them
    assert merged[0] != merged[4]


def test_find_atom_index_by_name_scoped_to_ml_resnames():
    names = ["N1", "C1", "N1"]
    res = ["AMM1", "CH3CL", "TIP3"]
    assert find_atom_index_by_name(names, res, atom_name="N1", ml_resnames=("AMM1",)) == 0
    with pytest.raises(ValueError, match="ambiguous"):
        find_atom_index_by_name(names, res, atom_name="N1", ml_resnames=None)


def test_hybrid_config_requires_psf_or_composition():
    with pytest.raises(ValueError, match="from_psf"):
        UmbrellaConfig(
            checkpoint=Path("ckpt.json"),
            output_dir=Path("out"),
            atom_i=0,
            atom_j=1,
            engine="hybrid_jaxmd",
            xi_min=1.0,
            xi_max=2.0,
            n_windows=2,
        )


def test_hybrid_config_rejects_2d():
    with pytest.raises(ValueError, match="1D"):
        UmbrellaConfig(
            checkpoint=Path("ckpt.json"),
            output_dir=Path("out"),
            atom_i=0,
            atom_j=1,
            atom_k=2,
            atom_l=3,
            engine="hybrid_jaxmd",
            from_psf=Path("x.psf"),
            from_pdb=Path("x.pdb"),
            box_size=20.0,
            xi_min=1.0,
            xi_max=2.0,
            n_windows=2,
            yi_min=1.0,
            yi_max=2.0,
            n_windows_y=2,
        )


def test_hybrid_config_ok_from_psf():
    cfg = UmbrellaConfig(
        checkpoint=Path("ckpt.json"),
        output_dir=Path("out"),
        atom_i=0,
        atom_j=1,
        engine="hybrid_jaxmd",
        from_psf=Path("x.psf"),
        from_pdb=Path("x.pdb"),
        box_size=30.0,
        atom_name_i="C1",
        atom_name_j="N1",
        xi_min=2.0,
        xi_max=3.0,
        n_windows=3,
    )
    assert cfg.engine == "hybrid_jaxmd"
    assert cfg.resolve_schedule().n_windows == 3


def test_mic_stretch_matches_free_space_without_box():
    r = np.zeros((4, 3), dtype=np.float64)
    r[1] = [2.0, 0.0, 0.0]
    out = stretch_distance_seed_mic(r, 0, 1, 3.0, box=None, move_with=(2,))
    assert mic_distance(out, 0, 1, None) == pytest.approx(3.0)
    assert out[2, 0] == pytest.approx(1.0)


def test_snapshot_schema_hybrid_extras(tmp_path: Path):
    path = tmp_path / "umbrella_snapshots.npz"
    K, T, N = 2, 3, 5
    pos = np.zeros((K, T, N, 3), dtype=np.float64)
    save_snapshots(
        path,
        positions=pos,
        Z=np.arange(N, dtype=np.int32) + 1,
        atom_i=0,
        atom_j=1,
        xi0=np.array([1.5, 2.5]),
        k_ev_A2=np.array([5.0, 5.0]),
        temperature_K=300.0,
        dt_fs=0.5,
        cv_traj=np.ones((K, T, 1)),
        checkpoint="/tmp/ckpt.json",
        extra={
            "engine": np.asarray("hybrid_jaxmd"),
            "energies_unbiased_ev": np.zeros((K, T)),
            "ml_atom_indices": np.array([0, 1, 2], dtype=np.int32),
        },
    )
    snap = load_snapshots(path)
    assert str(snap["engine"].item() if hasattr(snap["engine"], "item") else snap["engine"]) == (
        "hybrid_jaxmd"
    )
    assert snap["energies_unbiased_ev"].shape == (K, T)
    assert snap["ml_atom_indices"].tolist() == [0, 1, 2]


def test_mbar_fill_u_kln_uses_stored_unbiased_energies():
    K, T, N = 2, 2, 3
    pos = np.zeros((K, T, N, 3), dtype=np.float64)
    # Place atom 1 along x so CV differs per frame a bit
    pos[0, 0, 1, 0] = 1.5
    pos[0, 1, 1, 0] = 1.6
    pos[1, 0, 1, 0] = 2.5
    pos[1, 1, 1, 0] = 2.6
    u_unb = np.array([[1.0, 1.1], [2.0, 2.1]], dtype=np.float64)
    u_kln, n_k = fill_u_kln(
        positions=pos,
        atom_pairs=[(0, 1)],
        targets_per_cv=[[1.5, 2.5]],
        k_per_cv=[[10.0, 10.0]],
        temperature_K=300.0,
        unbiased_energies=u_unb,
    )
    assert u_kln.shape == (2, 2, 2)
    assert n_k.tolist() == [2, 2]
    assert np.all(np.isfinite(u_kln))


def test_run_umbrella_nvt_dispatches_hybrid(monkeypatch):
    from mmml.umbrella import sample as sample_mod
    from mmml.umbrella.sample import UmbrellaResult

    called = {}

    def fake_hybrid(cfg):
        called["engine"] = cfg.engine
        return UmbrellaResult(
            output_dir=Path("out"),
            snapshots_path=Path("out/s.npz"),
            summary_path=Path("out/s.json"),
            n_windows=1,
            n_frames=1,
            paths={},
        )

    monkeypatch.setattr(
        "mmml.umbrella.hybrid.run_umbrella_hybrid_nvt",
        fake_hybrid,
    )
    cfg = UmbrellaConfig(
        checkpoint=Path("ckpt.json"),
        output_dir=Path("out"),
        atom_i=0,
        atom_j=1,
        engine="hybrid_jaxmd",
        from_psf=Path("x.psf"),
        from_pdb=Path("x.pdb"),
        box_size=30.0,
        xi_min=2.0,
        xi_max=3.0,
        n_windows=2,
    )
    result = sample_mod.run_umbrella_nvt(cfg)
    assert called["engine"] == "hybrid_jaxmd"
    assert result.n_windows == 1
