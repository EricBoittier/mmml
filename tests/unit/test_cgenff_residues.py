from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.cgenff_residues import (
    format_cgenff_residue_list,
    is_cgenff_residue_name,
    normalize_cgenff_residue_name,
    parse_cgenff_residue_line,
    parse_cgenff_residues,
    require_cgenff_residue_name,
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


def test_normalize_and_require_cgenff_names() -> None:
    assert normalize_cgenff_residue_name("water") == "TIP3"
    assert is_cgenff_residue_name("ACO")
    assert require_cgenff_residue_name("octanol") == "OCOH"
    with pytest.raises(ValueError, match="Unknown CGenFF"):
        require_cgenff_residue_name("ZZZZZ")


def test_extra_rtf_env_registers_ch3cl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface import cgenff_residues as cr

    extra = tmp_path / "extra.rtf"
    extra.write_text(
        "* test\n*\n36 1\nRESI CH3CL 0.00 ! chloromethane\nEND\n",
        encoding="utf-8",
    )
    cr.cgenff_residue_name_set.cache_clear()
    monkeypatch.setenv("MMML_CGENFF_EXTRA_RTF", str(extra))
    assert cr.extra_cgenff_rtf_paths() == (extra.resolve(),)
    assert is_cgenff_residue_name("CH3CL")
    assert require_cgenff_residue_name("ch3cl") == "CH3CL"
    monkeypatch.delenv("MMML_CGENFF_EXTRA_RTF", raising=False)
    cr.cgenff_residue_name_set.cache_clear()
    assert not is_cgenff_residue_name("CH3CL")


def test_extra_prm_env_resolves_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface import cgenff_residues as cr

    prm = tmp_path / "extra.prm"
    prm.write_text("* test\n*\nBONDS\nEND\n", encoding="utf-8")
    monkeypatch.setenv("MMML_CGENFF_EXTRA_PRM", str(prm))
    assert cr.extra_cgenff_prm_paths() == (prm.resolve(),)
    monkeypatch.delenv("MMML_CGENFF_EXTRA_PRM", raising=False)
    assert cr.extra_cgenff_prm_paths() == ()


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
