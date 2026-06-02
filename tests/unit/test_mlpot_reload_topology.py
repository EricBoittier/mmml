"""Tests for MLpot cluster reload topology PSF selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    resolve_topology_psf_for_mlpot_reload,
)


def test_resolve_topology_prefers_cluster_for_vmd_over_mini(tmp_path: Path):
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
    with pytest.raises(FileNotFoundError, match="cluster_for_vmd"):
        resolve_topology_psf_for_mlpot_reload(mini)
