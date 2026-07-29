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


def test_residue_sequence_from_pdb_preserves_resid_order(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _parse_pdb_atoms_whitespace,
        _residue_sequence_from_pdb,
    )

    pdb = tmp_path / "dimer.pdb"
    # Chain-less layout matching examples/m/_geometry.write_solute_pdb
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1 N1   AMM1     1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2 H11  AMM1     1       1.000   0.000   0.000  1.00  0.00           H",
                "TER",
                "ATOM      3 C1   CH3CL    2       2.000   0.000   0.000  1.00  0.00           C",
                "ATOM      4 CL1  CH3CL    2       3.800   0.000   0.000  1.00  0.00          Cl",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _residue_sequence_from_pdb(pdb) == ["AMM1", "CH3CL"]
    _n, resn, resids, xyz = _parse_pdb_atoms_whitespace(pdb)
    assert resn == ["AMM1", "AMM1", "CH3CL", "CH3CL"]
    assert resids == [1, 1, 2, 2]
    assert float(xyz[3, 0]) == pytest.approx(3.8)


def test_parse_pdb_atoms_whitespace_minimal_no_occupancy(tmp_path):
    """Serial/resid must not be taken as x/y when occ/tempFactor are omitted."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import _parse_pdb_atoms_whitespace

    pdb = tmp_path / "minimal.pdb"
    # ATOM serial name resname resid x y z  (exactly 5 numeric tokens)
    pdb.write_text(
        "ATOM 1 N1 AMM1 1 -2.699 1.081 -0.327\n"
        "ATOM 2 CL1 CH3CL 2 1.439 -0.600 0.174\n"
        "END\n",
        encoding="utf-8",
    )
    names, resn, resids, xyz = _parse_pdb_atoms_whitespace(pdb)
    assert names == ["N1", "CL1"]
    assert resn == ["AMM1", "CH3CL"]
    assert resids == [1, 2]
    assert xyz[0].tolist() == pytest.approx([-2.699, 1.081, -0.327])
    assert xyz[1].tolist() == pytest.approx([1.439, -0.600, 0.174])
    # Old float-harvest bug: floats[:3] or floats[-5:-2] → [1, 1, -2.699]
    assert xyz[0, 0] != pytest.approx(1.0)
