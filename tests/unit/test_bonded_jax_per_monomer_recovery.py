"""Unit tests for per-monomer JAX bonded preflight."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_minimize_bonded_jax_per_monomer_uses_mixed_composition_offsets(
    monkeypatch,
) -> None:
    """MEOH:TIP3 (9 atoms / 2 monomers) must not use a uniform 4.5-atom split."""
    from mmml.interfaces.pycharmmInterface.mlpot import bonded_jax_recovery as mod

    positions = np.zeros((9, 3), dtype=np.float64)
    captured: dict[str, object] = {}

    def _fake_fire(pos, _system, **kwargs):
        freeze = set(int(i) for i in kwargs["freeze_indices"])
        # First monomer MEOH = atoms 0..5 → freeze should leave those free.
        assert 0 not in freeze
        assert 5 not in freeze
        assert 6 in freeze
        assert 8 in freeze
        captured["ok"] = True
        return pos, 0.1

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: positions,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        lambda _p: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.sync_charmm_lists_after_mini",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
        lambda _c: None,
    )
    monkeypatch.setattr(
        mod,
        "load_bonded_system_for_recovery",
        lambda *_a, **_k: (
            SimpleNamespace(
                topology=SimpleNamespace(bonds=np.array([[0, 1]], dtype=int))
            ),
            SimpleNamespace(cleanup=lambda: None),
        ),
    )
    monkeypatch.setattr(mod, "_run_jax_bonded_fire", _fake_fire)
    monkeypatch.setattr(mod, "_ml_atom_indices", lambda _c: ())

    ctx = SimpleNamespace(
        atoms_per_monomer=[6, 3],
        workflow_args=SimpleNamespace(composition="MEOH:1,TIP3:1"),
        ml_selection=None,
    )
    cfg = SimpleNamespace(
        nstep_jax=5,
        nstep_sd=5,
        nprint=1,
        tolgrd=1e-3,
        verbose=False,
        per_monomer_jax=True,
    )

    grms = mod.minimize_bonded_jax_per_monomer_recovery(
        ctx,
        cfg,
        n_monomers=2,
        monomer_indices=(0,),
    )
    assert grms == pytest.approx(0.1)
    assert captured.get("ok") is True


def test_maybe_run_per_monomer_preflight_skips_single_monomer() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _maybe_run_per_monomer_bonded_jax_preflight,
    )

    ctx = MagicMock()
    config = MagicMock(n_monomers=1, rescue=MagicMock())
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_per_monomer_recovery",
    ) as mini:
        _maybe_run_per_monomer_bonded_jax_preflight(ctx, config, context="test")
    mini.assert_not_called()


def test_maybe_run_per_monomer_preflight_calls_jax() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _maybe_run_per_monomer_bonded_jax_preflight,
    )

    ctx = MagicMock()
    config = MagicMock(
        n_monomers=4,
        monomer_health=MagicMock(enabled=False),
        rescue=MagicMock(
            nstep_sd=20,
            nprint=5,
            tolenr=1e-3,
            tolgrd=1e-3,
            verbose=False,
        ),
        topology_psf=None,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery.minimize_bonded_jax_per_monomer_recovery",
    ) as mini:
        _maybe_run_per_monomer_bonded_jax_preflight(ctx, config, context="test")
    mini.assert_called_once()
    assert mini.call_args.kwargs["n_monomers"] == 4
