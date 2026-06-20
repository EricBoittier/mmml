"""Unit tests for CHARMM MM pretreat fixed-box vs NPT behavior."""

from __future__ import annotations

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
    _pretreat_use_fixed_box_nvt,
)


def test_pretreat_fixed_box_nvt_when_box_size_set() -> None:
    args = argparse.Namespace(box_size=32.0)
    assert _pretreat_use_fixed_box_nvt(args, use_pbc=True) is True


def test_pretreat_npt_when_box_size_unset() -> None:
    args = argparse.Namespace()
    assert _pretreat_use_fixed_box_nvt(args, use_pbc=True) is False


def test_pretreat_npt_when_not_pbc() -> None:
    args = argparse.Namespace(box_size=32.0)
    assert _pretreat_use_fixed_box_nvt(args, use_pbc=False) is False
