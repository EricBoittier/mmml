"""Tests for PSF/CRD cluster reload from liquid-box artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    DEFAULT_MIN_MONOMER_EXTENT_A,
    validate_cluster_geometry,
)


def test_validate_cluster_geometry_rejects_empty_positions():
    with pytest.raises(ValueError, match="0 atoms"):
        validate_cluster_geometry(np.zeros((0, 3)), n_molecules=1)


def test_validate_cluster_geometry_rejects_placeholder_coordinates():
    with pytest.raises(ValueError, match="9999 sentinel"):
        validate_cluster_geometry(np.full((5, 3), 9999.0), n_molecules=1)


def test_validate_cluster_geometry_accepts_tip3_extent_below_legacy_threshold():
    tip3 = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.776, 0.437, 0.243],
            [-0.485, 0.582, 0.534],
        ]
    )

    stats = validate_cluster_geometry(tip3, n_molecules=1)

    assert stats["n_molecules"] == 1.0


def test_validate_cluster_geometry_rejects_collapsed_monomer():
    collapsed_extent_A = DEFAULT_MIN_MONOMER_EXTENT_A / 10.0
    collapsed = np.array(
        [
            [0.00, 0.00, 0.00],
            [collapsed_extent_A, 0.00, 0.00],
            [0.00, collapsed_extent_A, collapsed_extent_A],
        ]
    )

    with pytest.raises(ValueError, match="Monomer 1 extent"):
        validate_cluster_geometry(collapsed, min_axis_span=0.0, n_molecules=1)


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


def test_reconcile_n_monomers_uses_mixed_composition_when_psf_resids_unavailable():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import reconcile_n_monomers_with_psf

    args = SimpleNamespace(
        composition="MEOH:1,TIP3:1",
        quiet=True,
        _cluster_atoms_per_list=None,
    )

    n_mol, atoms_per = reconcile_n_monomers_with_psf(args, np.zeros(9, dtype=int), 2)

    assert n_mol == 2
    assert atoms_per == [6, 3]
    assert args._cluster_atoms_per_list == [6, 3]
    assert args._cluster_residue_labels == ["MEOH", "TIP3"]


def test_load_physnet_mlpot_uses_mixed_composition_when_psf_resids_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    from mmml.interfaces.pycharmmInterface.mlpot.setup import load_physnet_mlpot_bundle

    calls: list[tuple[list[int], int]] = []
    fake_hybrid = ModuleType("mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot")

    def _fake_build(_ckpt, _z, atoms_per_monomer, n_monomers, **_kwargs):
        calls.append((list(atoms_per_monomer), int(n_monomers)))
        return object()

    fake_hybrid.build_decomposed_mlpot_model = _fake_build  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot",
        fake_hybrid,
    )

    class _Atoms:
        def get_atomic_numbers(self):
            return np.array([6, 8, 1, 1, 1, 1, 8, 1, 1], dtype=int)

    args = SimpleNamespace(composition="MEOH:1,TIP3:1", _cluster_atoms_per_list=None)

    load_physnet_mlpot_bundle(
        tmp_path / "ckpt.json",
        9,
        _Atoms(),
        n_monomers=2,
        args=args,
    )

    assert calls == [([6, 3], 2)]
