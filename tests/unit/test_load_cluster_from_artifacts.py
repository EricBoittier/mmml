"""Tests for PSF/CRD cluster reload from liquid-box artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import validate_cluster_geometry


def test_validate_cluster_geometry_rejects_empty_positions():
    with pytest.raises(ValueError, match="0 atoms"):
        validate_cluster_geometry(np.zeros((0, 3)), n_molecules=1)


def test_validate_cluster_geometry_rejects_placeholder_coordinates():
    with pytest.raises(ValueError, match="9999 sentinel"):
        validate_cluster_geometry(np.full((5, 3), 9999.0), n_molecules=1)


def test_load_cluster_from_artifacts_uses_xplor_psf_reader(tmp_path: Path, monkeypatch):
    psf = tmp_path / "model.psf"
    crd = tmp_path / "model.crd"
    psf.write_text("PSF EXT CMAP XPLOR\n", encoding="ascii")
    crd.write_text("* crd\n", encoding="ascii")

    calls: list[str] = []

    def _fake_read_psf(path: Path, **_kw):
        calls.append(str(path))

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.cgenff_bonded_reference.read_psf_card_file",
        _fake_read_psf,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config.read_cgenff_toppar",
        lambda **_kw: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup._import_pycharmm",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.resolve_topology_psf_for_mlpot_reload",
        lambda p, **_: Path(p).resolve(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.load_minimized_coordinates",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.utils.get_Z_from_psf",
        lambda: np.array([6, 1, 1, 17, 17], dtype=int),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: np.ones((5, 3), dtype=float),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.report_charmm_topology_summary",
        lambda **_kw: True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.reconcile_n_monomers_with_psf",
        lambda _a, _z, n: (1, [5]),
    )

    from mmml.interfaces.pycharmmInterface.mlpot.setup import load_cluster_from_artifacts

    args = SimpleNamespace(
        from_psf=str(psf),
        from_crd=str(crd),
        n_molecules=1,
        composition=None,
        tag=None,
        quiet=True,
        _cluster_atoms_per_list=None,
    )
    z, r, n_mol, _tag = load_cluster_from_artifacts(args)
    assert calls == [str(psf.resolve())]
    assert len(z) == 5
    assert r.shape == (5, 3)
    assert n_mol == 1
