from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.cgenff_residues import (
    format_cgenff_residue_list,
    parse_cgenff_residue_line,
    parse_cgenff_residues,
)


def test_parse_cgenff_residue_line_with_comment() -> None:
    residue = parse_cgenff_residue_line(
        "RESI ACO           0.00 ! C3H6O, Acetone, adm, Oct 08\n"
    )
    assert residue is not None
    assert residue.name == "ACO"
    assert residue.charge == "0.00"
    assert "Acetone" in residue.comment


def test_parse_cgenff_residue_line_with_flags_before_comment() -> None:
    residue = parse_cgenff_residue_line(
        "RESI TIP3          0.00 NOANG NODIH ! H2O, tip3p water model\n"
    )
    assert residue is not None
    assert residue.name == "TIP3"
    assert residue.charge == "0.00"
    assert "tip3p" in residue.comment


def test_parse_cgenff_residues_includes_aco() -> None:
    residues = parse_cgenff_residues()
    names = {r.name for r in residues}
    assert "ACO" in names
    assert len(residues) >= 50


def test_format_cgenff_residue_list_columns() -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_residues import CgenffResidue

    text = format_cgenff_residue_list(
        [
            CgenffResidue("ACO", "0.00", "Acetone"),
            CgenffResidue("TIP3", "0.00", "Water"),
        ],
        rtf_path=Path("/tmp/top_all36_cgenff.rtf"),
    )
    assert "RESIDUE" in text
    assert "ACO" in text
    assert "Acetone" in text
    assert "mmml make-res --res RESIDUE" in text


def test_make_res_validate_args_list_residues() -> None:
    import argparse

    from mmml.cli.make.make_res import validate_args

    validate_args(argparse.Namespace(list_residues=True, res=None))


def test_make_res_validate_args_requires_res() -> None:
    import argparse

    from mmml.cli.make.make_res import validate_args

    with pytest.raises(SystemExit):
        validate_args(argparse.Namespace(list_residues=False, res=None))


def test_make_res_list_residues_cli(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.cli.misc import make_res_cli

    monkeypatch.setattr("sys.argv", ["mmml make-res", "--list-residues", "--no-pager"])
    rc = make_res_cli.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "ACO" in out
    assert "Acetone" in out
