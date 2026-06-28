"""Unit tests for CHARMM lattice ABNR mini-stage helpers."""

from __future__ import annotations

import argparse
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def _fake_pycharmm_script_module(script_cls: MagicMock):
    """Inject a stub ``pycharmm.script`` without loading libcharmm."""
    fake_pycharmm = types.ModuleType("pycharmm")
    fake_script_mod = types.ModuleType("pycharmm.script")
    fake_script_mod.CommandScript = script_cls
    fake_pycharmm.script = fake_script_mod
    with patch.dict(
        sys.modules,
        {"pycharmm": fake_pycharmm, "pycharmm.script": fake_script_mod},
    ):
        yield


def test_should_run_mini_lattice_abnr_skips_fixed_box():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        should_run_mini_lattice_abnr,
    )

    args = argparse.Namespace(
        mini_lattice_abnr_steps=50,
        box_size=40.0,
        mini_lattice_abnr_allow_fixed_box=False,
    )
    assert not should_run_mini_lattice_abnr(
        args,
        charmm_pbc=True,
        stages=["mini", "heat"],
    )


def test_should_run_mini_lattice_abnr_true_for_pbc_mini():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        should_run_mini_lattice_abnr,
    )

    args = argparse.Namespace(
        mini_lattice_abnr_steps=25,
        box_size=None,
        mini_lattice_abnr_allow_fixed_box=False,
    )
    assert should_run_mini_lattice_abnr(
        args,
        charmm_pbc=True,
        stages=["mini", "heat"],
    )


def test_run_charmm_lattice_abnr_uses_script_command():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        run_charmm_lattice_abnr,
    )

    script_cls = MagicMock()
    script_inst = MagicMock()
    script_cls.return_value = script_inst
    with (
        _fake_pycharmm_script_module(script_cls),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
            return_value=(42.5, "pbound"),
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.apply_pbc_nbonds",
        ) as apply_nb,
        patch(
            "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
        ),
    ):
        side = run_charmm_lattice_abnr(
            nstep=30,
            tolenr=1e-3,
            tolgrd=1e-3,
            nocoords=True,
            verbose=False,
        )
    assert side == pytest.approx(42.5)
    script_cls.assert_called_once_with(
        "mini abnr",
        lattice=True,
        nstep=30,
        tolenr=1e-3,
        tolgrd=1e-3,
        nocoords=True,
    )
    script_inst.run.assert_called_once()
    apply_nb.assert_called_once_with(nbxmod=5, cubic_box_side_A=42.5)


def test_run_charmm_lattice_abnr_skips_zero_steps():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        run_charmm_lattice_abnr,
    )

    assert run_charmm_lattice_abnr(nstep=0, tolenr=1e-3, tolgrd=1e-3) is None


def test_run_charmm_lattice_abnr_uses_fallback_when_pbound_inactive():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        run_charmm_lattice_abnr,
    )

    script_cls = MagicMock()
    script_inst = MagicMock()
    script_cls.return_value = script_inst
    with (
        _fake_pycharmm_script_module(script_cls),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
            return_value=False,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.ensure_charmm_crystal_for_cpt",
        ) as ensure_crystal,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
            return_value=(35.0, "restart"),
        ) as resolve_side,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.apply_pbc_nbonds",
        ) as apply_nb,
        patch(
            "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
        ),
    ):
        side = run_charmm_lattice_abnr(
            nstep=20,
            tolenr=1e-3,
            tolgrd=1e-3,
            fallback_side_A=35.0,
            restart_path="/tmp/prod.res",
            verbose=False,
        )
    assert side == pytest.approx(35.0)
    ensure_crystal.assert_called_once_with(35.0, quiet=True)
    resolve_side.assert_called_once_with(
        fallback_side_A=35.0,
        restart_path="/tmp/prod.res",
    )
    apply_nb.assert_called_once_with(nbxmod=5, cubic_box_side_A=35.0)


def test_run_charmm_lattice_abnr_skips_restart_when_crystal_active():
    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        run_charmm_lattice_abnr,
    )

    script_cls = MagicMock()
    script_inst = MagicMock()
    script_cls.return_value = script_inst
    with (
        _fake_pycharmm_script_module(script_cls),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.ensure_charmm_crystal_for_cpt",
        ) as ensure_crystal,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
            return_value=(36.0, "pbound"),
        ) as resolve_side,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.apply_pbc_nbonds",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_quiet_output",
        ),
    ):
        side = run_charmm_lattice_abnr(
            nstep=10,
            tolenr=1e-3,
            tolgrd=1e-3,
            fallback_side_A=35.0,
            restart_path="/tmp/prod.res",
            verbose=False,
        )
    assert side == pytest.approx(36.0)
    ensure_crystal.assert_not_called()
    resolve_side.assert_called_once_with(
        fallback_side_A=35.0,
        restart_path=None,
    )
