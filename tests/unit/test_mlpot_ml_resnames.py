"""Unit tests for PyCHARMM ml_resnames mechanical-embedding selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


def test_resolve_mlpot_selection_defaults_to_all_atoms(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import setup as setup_mod

    all_sel = object()
    monkeypatch.setattr(setup_mod, "select_all_atoms", lambda: all_sel)
    assert setup_mod.resolve_mlpot_selection_from_args(None) is all_sel
    assert setup_mod.resolve_mlpot_selection_from_args(SimpleNamespace()) is all_sel


def test_resolve_mlpot_selection_uses_resnames(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import setup as setup_mod

    called: list[tuple[str, ...]] = []

    def _fake_select(names):
        called.append(tuple(names))
        return SimpleNamespace(names=tuple(names))

    monkeypatch.setattr(setup_mod, "select_by_resnames", _fake_select)
    args = SimpleNamespace(ml_resnames=["AMM1", "CH3CL"])
    sel = setup_mod.resolve_mlpot_selection_from_args(args)
    assert called == [("AMM1", "CH3CL")]
    assert sel.names == ("AMM1", "CH3CL")


def test_select_by_resnames_expands_ch3cl_alias(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import setup as setup_mod

    names_seen: list[str] = []

    class _FakeSel:
        def __init__(self, res_name=""):
            names_seen.append(str(res_name))

        def __or__(self, other):
            return self

    monkeypatch.setattr(setup_mod, "_select_atoms_cls", lambda: _FakeSel)
    setup_mod.select_by_resnames(["AMM1", "CH3CL"])
    assert "AMM1" in names_seen
    assert "CH3CL" in names_seen
    assert "CH3C" in names_seen  # truncated PSF alias


def test_register_mlpot_context_refuses_jax_mic_with_ml_resnames(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow as rw

    args = SimpleNamespace(
        ml_resnames=["AMM1", "CH3CL"],
        mm_nonbond_mode="jax_mic",
        _cluster_residue_labels=["AMM1", "CH3CL", "TIP3"],
        mlpot_mm_internal_scale=0.0,
        verbose=False,
        mlpot_use_block=False,
        ml_spatial_mpi=None,
    )
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.resolve_mlpot_selection_from_args",
        lambda _a: sel,
    )
    monkeypatch.setattr(
        rw,
        "_atoms_per_monomer_list",
        lambda z, n, args=None: [4, 5, 3],
    )
    z = np.ones(12, dtype=int)
    r = np.zeros((12, 3), dtype=float)
    with pytest.raises(ValueError, match="periodic_external"):
        rw._register_mlpot_context(
            z,
            r,
            ckpt=mock.Mock(),
            n_atoms=12,
            n_monomers=3,
            atoms_per_monomer=[4, 5, 3],
            verbose=False,
            args=args,
            defer_jax_warmup=False,
        )
