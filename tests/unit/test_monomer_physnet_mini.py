"""Selective monomer PhysNet BFGS (mocked ASE / checkpoint)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini import (
    SelectiveMonomerPhysnetMiniConfig,
    monomer_physnet_mini_enabled,
    resolve_monomer_template_reference_positions,
    run_selective_monomer_physnet_mini,
    selective_monomer_physnet_mini_config_from_args,
)


def _ctx(
    *,
    positions: np.ndarray,
    z: np.ndarray | None = None,
    atoms_per: list[int] | None = None,
    mini_positions: np.ndarray | None = None,
) -> SimpleNamespace:
    atoms_per = atoms_per or [3, 3]
    z = z if z is not None else np.array([6, 1, 1, 6, 1, 1], dtype=int)
    ctx = SimpleNamespace(
        ml_Z=z,
        pyCModel=SimpleNamespace(_atoms_per_monomer=atoms_per),
        workflow_args=SimpleNamespace(
            checkpoint="/tmp/fake.ckpt",
            monomer_physnet_mini=True,
            pre_min_fmax=0.05,
            bfgs_maxstep=0.05,
            quiet_bfgs=True,
        ),
        use_pbc=False,
        geometry_mini_positions=mini_positions,
        _monomer_physnet_calc_cache={},
    )
    ctx._positions = np.asarray(positions, dtype=np.float64).copy()
    return ctx


class _FakeAtoms:
    def __init__(self, numbers, positions):
        self.numbers = np.asarray(numbers, dtype=int)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.calc = None

    def get_positions(self):
        return np.asarray(self.positions, dtype=np.float64)


class _FakeBFGS:
    def __init__(self, atoms, **kwargs):
        self.atoms = atoms

    def run(self, fmax=0.05, steps=60):
        self.atoms.positions = self.atoms.positions + 0.1


def test_monomer_physnet_mini_enabled_default():
    assert monomer_physnet_mini_enabled(None) is True
    assert monomer_physnet_mini_enabled(SimpleNamespace(monomer_physnet_mini=False)) is False


def test_config_from_args_inherits_pre_min():
    args = SimpleNamespace(
        monomer_physnet_mini_max_select=1,
        monomer_physnet_mini_min_grms=20.0,
        monomer_physnet_mini_min_ratio=3.0,
        monomer_physnet_mini_steps=40,
        monomer_physnet_mini_fmax=None,
        monomer_physnet_mini_maxstep=None,
        pre_min_fmax=0.02,
        bfgs_maxstep=0.03,
        quiet_bfgs=True,
    )
    cfg = selective_monomer_physnet_mini_config_from_args(args, verbose=False)
    assert cfg.max_select == 1
    assert cfg.min_abs_grms == pytest.approx(20.0)
    assert cfg.fmax_ev_a == pytest.approx(0.02)
    assert cfg.bfgs_maxstep == pytest.approx(0.03)


def test_resolve_monomer_template_reference_positions_uses_memory_mini():
    ref = np.arange(18, dtype=float).reshape(6, 3)
    ctx = _ctx(positions=np.zeros((6, 3)), mini_positions=ref)
    resolved = resolve_monomer_template_reference_positions(ctx, n_atoms=6)
    assert resolved is not None
    arr, source = resolved
    np.testing.assert_allclose(arr, ref)
    assert source.name == "<in-memory-mini>"


def test_run_selective_monomer_physnet_mini_skips_without_flagged(monkeypatch):
    ctx = _ctx(positions=np.zeros((6, 3)))
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: ctx._positions,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini.mlpot_hybrid_grms_from_calculator",
        lambda _ctx: 12.0,
    )
    empty_diag = SimpleNamespace(
        flagged=(),
        grms_per_monomer=np.array([5.0, 5.0]),
        cluster_grms=5.0,
    )
    monkeypatch.setattr(
        "mmml.utils.monomer_force_diag.resolve_selective_repack_monomers",
        lambda *a, **k: empty_diag,
    )
    result = run_selective_monomer_physnet_mini(ctx, context_prefix="test")
    assert result.ran is False
    assert result.grms == pytest.approx(12.0)


def test_run_selective_monomer_physnet_mini_runs_bfgs_on_flagged(monkeypatch):
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    mini = pos.copy()
    mini[3:6] += np.array([0.5, 0.0, 0.0])
    ctx = _ctx(positions=pos, mini_positions=mini)

    diag = SimpleNamespace(
        flagged=(1,),
        grms_per_monomer=np.array([5.0, 45.0]),
        cluster_grms=30.0,
    )
    monkeypatch.setattr(
        "mmml.utils.monomer_force_diag.resolve_selective_repack_monomers",
        lambda *a, **k: diag,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini.resolve_mlpot_checkpoint_path",
        lambda _ctx: __import__("pathlib").Path("/tmp/fake.ckpt"),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini._monomer_ase_calculator",
        lambda *a, **k: MagicMock(),
    )

    synced: list[np.ndarray] = []

    def _sync(arr):
        synced.append(np.asarray(arr, dtype=np.float64).copy())

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: ctx._positions,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        _sync,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
        lambda _ctx: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini.refresh_mlpot_energy_and_grms",
        lambda _ctx, context="": 8.5,
    )

    import ase
    import ase.optimize as ase_opt

    monkeypatch.setattr(ase, "Atoms", _FakeAtoms)
    monkeypatch.setattr(ase_opt, "BFGS", _FakeBFGS)

    cfg = SelectiveMonomerPhysnetMiniConfig(
        verbose=False,
        quiet_bfgs=True,
    )
    result = run_selective_monomer_physnet_mini(ctx, config=cfg, context_prefix="test")

    assert result.ran is True
    assert result.flagged == (1,)
    assert result.grms == pytest.approx(8.5)
    assert len(synced) == 1
    assert synced[0][3, 0] == pytest.approx(10.1)


def test_run_selective_monomer_physnet_mini_explicit_flagged(monkeypatch):
    pos = np.zeros((6, 3), dtype=np.float64)
    ctx = _ctx(positions=pos)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini.resolve_mlpot_checkpoint_path",
        lambda _ctx: __import__("pathlib").Path("/tmp/fake.ckpt"),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini._monomer_ase_calculator",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: ctx._positions,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        lambda arr: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
        lambda _ctx: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini.refresh_mlpot_energy_and_grms",
        lambda _ctx, context="": 4.0,
    )

    import ase
    import ase.optimize as ase_opt

    monkeypatch.setattr(ase, "Atoms", _FakeAtoms)
    monkeypatch.setattr(ase_opt, "BFGS", _FakeBFGS)

    result = run_selective_monomer_physnet_mini(
        ctx,
        config=SelectiveMonomerPhysnetMiniConfig(
            verbose=False,
            quiet_bfgs=True,
        ),
        flagged=(0,),
        context_prefix="test",
    )
    assert result.ran is True
    assert result.flagged == (0,)
