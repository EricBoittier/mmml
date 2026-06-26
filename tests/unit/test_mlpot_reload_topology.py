"""Tests for MLpot cluster reload topology PSF selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    resolve_topology_psf_for_mlpot_reload,
)
from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
    TopologyFingerprint,
    load_topology_sidecar,
    save_topology_sidecar,
    topology_fingerprint_path,
)


def test_resolve_topology_prefers_model_over_mini(tmp_path: Path):
    mini = tmp_path / "mini.psf"
    vmd = tmp_path / "model.psf"
    mini.write_text("mini", encoding="ascii")
    vmd.write_text("vmd", encoding="ascii")

    resolved = resolve_topology_psf_for_mlpot_reload(mini, tag="dcm_90")
    assert resolved == vmd.resolve()


def test_resolve_topology_prefers_cluster_for_vmd_legacy(tmp_path: Path):
    mini = tmp_path / "mini_full_mlpot_dcm_90.psf"
    vmd = tmp_path / "cluster_for_vmd_dcm_90.psf"
    mini.write_text("mini", encoding="ascii")
    vmd.write_text("vmd", encoding="ascii")

    resolved = resolve_topology_psf_for_mlpot_reload(mini, tag="dcm_90")
    assert resolved == vmd.resolve()


def test_resolve_topology_keeps_non_mini_psf(tmp_path: Path):
    psf = tmp_path / "cluster_for_vmd_dcm_90.psf"
    psf.write_text("vmd", encoding="ascii")
    assert resolve_topology_psf_for_mlpot_reload(psf) == psf.resolve()


def test_resolve_topology_raises_when_vmd_missing(tmp_path: Path):
    mini = tmp_path / "mini_full_mlpot_dcm_90.psf"
    mini.write_text("mini", encoding="ascii")
    with pytest.raises(FileNotFoundError, match="model.psf"):
        resolve_topology_psf_for_mlpot_reload(mini)


def test_fingerprint_sidecar_roundtrip_for_cluster_psf(tmp_path: Path):
    psf = tmp_path / "cluster_for_vmd_dcm_90.psf"
    psf.write_text("vmd", encoding="ascii")
    fp = TopologyFingerprint(
        natom=90,
        nres=10,
        nseg=1,
        atom_names=tuple(f"C{i}" for i in range(90)),
        resids=tuple((i % 10) + 1 for i in range(90)),
    )
    sidecar = topology_fingerprint_path(psf)
    save_topology_sidecar(sidecar, fp, pre_mlpot_iblo=[0] * 90, pre_mlpot_inb=[1, 2])
    loaded = load_topology_sidecar(sidecar)
    assert loaded is not None
    assert loaded.fingerprint.natom == 90
    assert len(loaded.pre_mlpot_iblo or ()) == 90
