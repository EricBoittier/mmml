"""Unit tests for CHARMM lattice ABNR mini-stage helpers."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


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
        patch("pycharmm.script.CommandScript", script_cls),
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

    with patch("pycharmm.script.CommandScript") as script_cls:
        assert run_charmm_lattice_abnr(nstep=0, tolenr=1e-3, tolgrd=1e-3) is None
    script_cls.assert_not_called()
