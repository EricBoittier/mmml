"""Tests for staged MLpot CLI stage/PBC resolution."""

from __future__ import annotations

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_md_stages,
    resolve_use_pbc,
)


def test_resolve_md_stages_pycharmm_full():
    args = argparse.Namespace(setup="pycharmm_full", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "heat", "nve", "equi", "prod"]


def test_resolve_md_stages_free_nve():
    args = argparse.Namespace(setup="free_nve", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "nve"]


def test_resolve_md_stages_override():
    args = argparse.Namespace(
        setup="pycharmm_full",
        md_stages="mini,heat",
        md_stage=None,
        phase="staged",
    )
    assert resolve_md_stages(args) == ["mini", "heat"]


def test_resolve_use_pbc_from_setup():
    args = argparse.Namespace(setup="pbc_nve", free_space=False, box_size=None)
    assert resolve_use_pbc(args) is True


def test_resolve_use_pbc_free_space():
    args = argparse.Namespace(setup="pbc_nve", free_space=True, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_box_size():
    args = argparse.Namespace(setup="free_nve", free_space=False, box_size=40.0)
    assert resolve_use_pbc(args) is True


def test_cubic_box_length_from_geometry():
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    side = cubic_box_length_from_geometry(pos, ml_cutoff=12.0, pad=10.0)
    assert side >= 2.0 * 12.0 + 10.0
    assert side >= 10.0 + 20.0
