"""Unit tests for PSF-based cluster monomer count reconciliation."""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.setup import reconcile_n_monomers_with_psf


def _stub_pycharmm_import(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "mmml.interfaces.pycharmmInterface.import_pycharmm",
        types.ModuleType("import_pycharmm"),
    )


def test_reconcile_n_monomers_with_psf_overrides_cli_count(monkeypatch) -> None:
    _stub_pycharmm_import(monkeypatch)
    import mmml.interfaces.pycharmmInterface.mlpot.trimer_scan as trimer_scan

    monkeypatch.setattr(
        trimer_scan,
        "atoms_per_monomer_from_psf",
        lambda: [5] * 103,
    )
    z = np.ones(515, dtype=np.int32)
    args = argparse.Namespace(quiet=True, residue="DCM")

    resolved, atoms_per = reconcile_n_monomers_with_psf(args, z, 10)

    assert resolved == 103
    assert atoms_per == [5] * 103
    assert getattr(args, "_cluster_atoms_per_list") == [5] * 103
    assert getattr(args, "_cluster_composition_summary") == {"DCM": 103}


def test_reconcile_n_monomers_with_psf_keeps_matching_cli_count(monkeypatch) -> None:
    _stub_pycharmm_import(monkeypatch)
    import mmml.interfaces.pycharmmInterface.mlpot.trimer_scan as trimer_scan

    monkeypatch.setattr(
        trimer_scan,
        "atoms_per_monomer_from_psf",
        lambda: [5] * 10,
    )
    z = np.ones(50, dtype=np.int32)
    args = argparse.Namespace(quiet=True)

    resolved, atoms_per = reconcile_n_monomers_with_psf(args, z, 10)

    assert resolved == 10
    assert atoms_per == [5] * 10


def test_reconcile_n_monomers_with_psf_no_psf_unchanged(monkeypatch) -> None:
    _stub_pycharmm_import(monkeypatch)
    import mmml.interfaces.pycharmmInterface.mlpot.trimer_scan as trimer_scan

    monkeypatch.setattr(
        trimer_scan,
        "atoms_per_monomer_from_psf",
        lambda: (_ for _ in ()).throw(ValueError("no PSF")),
    )
    z = np.ones(50, dtype=np.int32)
    args = argparse.Namespace(quiet=True)

    resolved, atoms_per = reconcile_n_monomers_with_psf(args, z, 10)

    assert resolved == 10
    assert atoms_per is None
