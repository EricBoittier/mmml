"""Vacuum ``--from-pdb`` must not require CRYST1 / box.json / --box-size."""

from __future__ import annotations

import argparse

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    _vacuum_from_pdb_allows_missing_box,
)


@pytest.mark.parametrize(
    "kwargs, allowed",
    [
        ({"setup": "free_nvt"}, True),
        ({"setup": "free_nve"}, True),
        ({"setup": "free_thermalize"}, True),
        ({"free_space": True, "setup": "pbc_nvt"}, True),
        ({"setup": "pbc_nvt"}, False),
        ({"setup": "pbc_nve"}, False),
        ({}, False),
    ],
)
def test_vacuum_from_pdb_allows_missing_box(kwargs, allowed):
    args = argparse.Namespace(**kwargs)
    assert _vacuum_from_pdb_allows_missing_box(args) is allowed


def test_resolve_charmm_use_pbc_stays_off_when_vacuum_from_pdb_omits_box():
    """Inventing box_size for vacuum from-pdb would wrongly enable crystal."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_charmm_use_pbc

    args = argparse.Namespace(setup="free_nvt", free_space=False, box_size=None)
    assert resolve_charmm_use_pbc(args) is False
    # If a caller incorrectly set box_size without --free-space, PBC turns on:
    args.box_size = 40.0
    assert resolve_charmm_use_pbc(args) is True
    args.free_space = True
    assert resolve_charmm_use_pbc(args) is False
