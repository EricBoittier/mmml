"""Unit tests for pre-dynamics PyCHARMM lingo helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_pre_dynamics_lingo_from_args,
    normalize_pycharmm_pre_dynamics_lingo,
    parse_adumb_rc_wall_params,
    require_adumbrxncor_for_umbrella_rxncor,
    resolve_pre_dynamics_lingo_script,
    run_charmm_lingo_script,
    script_uses_umbrella_rxncor,
    split_charmm_lingo_commands,
    strip_mmfp_blocks_from_script,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ([], ""),
        ("cons fix sele resid 1 end", "cons fix sele resid 1 end"),
        (
            ["cons fix sele resid 1 end", "umbr", "end"],
            "cons fix sele resid 1 end\numbr\nend",
        ),
        (["  ", "umbr", ""], "umbr"),
    ],
)
def test_normalize_pycharmm_pre_dynamics_lingo(value, expected: str) -> None:
    assert normalize_pycharmm_pre_dynamics_lingo(value) == expected


def test_normalize_rejects_bad_types() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        normalize_pycharmm_pre_dynamics_lingo(123)
    with pytest.raises(ValueError, match="list items must be strings"):
        normalize_pycharmm_pre_dynamics_lingo(["ok", 2])


def test_resolve_merges_inline_and_file(tmp_path: Path) -> None:
    path = tmp_path / "extra.inp"
    path.write_text("umbr\nend\n", encoding="utf-8")
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="cons fix sele resid 1 end",
        pycharmm_pre_dynamics_lingo_file=path,
    )
    assert resolve_pre_dynamics_lingo_script(args) == (
        "cons fix sele resid 1 end\numbr\nend"
    )


def test_split_charmm_lingo_commands_joins_continuations() -> None:
    script = """
    ! comment
    cons hmcm force 2.0 -
      sele all end
    open unit 44 write card name adumb-wuni.dat
    """
    assert split_charmm_lingo_commands(script) == [
        "cons hmcm force 2.0 sele all end",
        "open unit 44 write card name adumb-wuni.dat",
    ]


def test_parse_adumb_rc_wall_params_reads_set_commands() -> None:
    script = """
    set adum_rcmax = 8.0
    set adum_rcwall = 500.0
    umbrella rxncor nresol 20 name rcl
    """
    assert parse_adumb_rc_wall_params(script) == (8.0, 500.0)
    assert parse_adumb_rc_wall_params("umbrella rxncor name rcl") is None


def test_strip_mmfp_blocks_from_script() -> None:
    script = """
    set adum_rcmax = 8.0
    MMFP -
    GEO sphere distance harmonic outside force 500 droff 8 -
      select atom * * CL1 end -
      select atom * * C1 end -
    END
    umbrella init nsim 4 update 50 equi 25 thresh 10 temp 300 wuni 44 ucun 50
    """
    stripped = strip_mmfp_blocks_from_script(script)
    assert "MMFP" not in stripped
    assert "GEO sphere" not in stripped
    assert "set adum_rcmax = 8.0" in stripped
    assert stripped.strip().endswith("ucun 50")


def test_run_charmm_lingo_installs_adumb_walls_before_umbrella(tmp_path: Path) -> None:
    script = """
    set adum_rcmax = 8.0
    set adum_rcwall = 500.0
    rxncor define rcl distance pcl pc
    MMFP -
    GEO sphere distance harmonic outside force @adum_rcwall droff @adum_rcmax -
      select atom * * CL1 end -
      select atom * * C1 end -
    END
    umbrella rxncor nresol 20 trig 0 poly 4 min 0.0 max @adum_rcmax name rcl
    umbrella init nsim 4 update 50 equi 25 thresh 10 temp 300 wuni 44 ucun 50
    """
    inp = tmp_path / "lingo.inp"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.require_adumbrxncor_for_umbrella_rxncor",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ) as install, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
        return_value=True,
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=inp, workdir=tmp_path)
    install.assert_called_once_with(rcmax=8.0, rcwall=500.0)
    assert script_fn.call_count == 5
    assert script_fn.call_args_list[0].args[0] == "set adum_rcmax = 8.0"
    assert script_fn.call_args_list[1].args[0] == "set adum_rcwall = 500.0"
    assert script_fn.call_args_list[2].args[0] == "rxncor define rcl distance pcl pc"
    assert script_fn.call_args_list[3].args[0].startswith("umbrella rxncor")
    assert script_fn.call_args_list[4].args[0].startswith("umbrella init")


def test_apply_pre_dynamics_lingo_no_op_when_empty() -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=False,
        output_dir=None,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_not_called()


def test_apply_pre_dynamics_lingo_runs_via_helper(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="cons fix sele resid 1 end",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=True,
        output_dir=tmp_path,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_called_once()
    assert run.call_args.args[0] == "cons fix sele resid 1 end"
    assert run.call_args.kwargs["workdir"] == tmp_path


def test_apply_pre_dynamics_lingo_from_file(tmp_path: Path) -> None:
    path = tmp_path / "umbr.inp"
    path.write_text("umbr\nend\n", encoding="utf-8")
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=path,
        quiet=True,
        output_dir=tmp_path,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_called_once()
    assert run.call_args.args[0] == "umbr\nend"
    assert run.call_args.kwargs["inp_path"] == path


def test_run_charmm_lingo_uses_line_commands_not_inp_api(tmp_path: Path) -> None:
    script = "cons fix sele resid 1 end\nopen unit 44 write card name adumb-wuni.dat"
    inp = tmp_path / "lingo.inp"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_inp_file",
    ) as invoke, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
        return_value=True,
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=inp, workdir=tmp_path)
    invoke.assert_not_called()
    assert script_fn.call_count == 2
    assert script_fn.call_args_list[0].args[0] == "cons fix sele resid 1 end"
    assert script_fn.call_args_list[1].args[0] == (
        "open unit 44 write card name adumb-wuni.dat"
    )
    assert inp.is_file()
    body = inp.read_text(encoding="utf-8")
    assert body.startswith("* MMML pre-dynamics")
    assert "cons fix sele resid 1 end" in body


def test_script_uses_umbrella_rxncor_detects_command() -> None:
    assert script_uses_umbrella_rxncor(
        "umbrella rxncor nresol 40 trig 0 poly 6 min 2.0 max 6.0 name r_nc"
    )
    assert script_uses_umbrella_rxncor(
        "  UMBRELLA   RXNCOR nresol 10 name dist\n"
    )
    assert not script_uses_umbrella_rxncor("umbrella dihe nresol 72")
    assert not script_uses_umbrella_rxncor("! umbrella rxncor commented")


def test_require_adumbrxncor_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    mock_lingo = mock.Mock()
    mock_lingo.get_charmm_builtins.return_value = {"ADUMBRXN": 0, "ADUMB": 1}
    fake = types.ModuleType("pycharmm")
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    with pytest.raises(RuntimeError, match="KEY_ADUMBRXNCOR"):
        require_adumbrxncor_for_umbrella_rxncor(
            "umbrella rxncor nresol 40 name r_nc"
        )


def test_require_adumbrxncor_raises_when_unset_but_adumb_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    mock_lingo = mock.Mock()
    mock_lingo.get_charmm_builtins.return_value = {"ADUMB": 1}
    fake = types.ModuleType("pycharmm")
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    with pytest.raises(RuntimeError, match="KEY_ADUMBRXNCOR"):
        require_adumbrxncor_for_umbrella_rxncor(
            "umbrella rxncor nresol 40 name r_nc"
        )


def test_require_adumbrxncor_skips_when_unqueryable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    import types

    mock_lingo = mock.Mock()
    mock_lingo.get_charmm_builtins.return_value = {}
    fake = types.ModuleType("pycharmm")
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    require_adumbrxncor_for_umbrella_rxncor("umbrella rxncor nresol 40 name r_nc")
    assert "skipping KEY_ADUMBRXNCOR preflight" in capsys.readouterr().out


def test_require_adumbrxncor_ok_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    mock_lingo = mock.Mock()
    mock_lingo.get_charmm_builtins.return_value = {"ADUMBRXN": 1, "ADUMB": 1}
    fake = types.ModuleType("pycharmm")
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    require_adumbrxncor_for_umbrella_rxncor("umbrella rxncor nresol 40 name r_nc")
    mock_lingo.get_charmm_builtins.assert_called()


def test_require_adumbrxncor_ignores_adum_energy_term_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_energy_value('ADUMBRXN') would hit energy term ADUM=0; builtins win."""
    import sys
    import types

    mock_lingo = mock.Mock()
    mock_lingo.get_energy_value.return_value = 0.0  # ADUM energy term trap
    mock_lingo.get_charmm_builtins.return_value = {"ADUMBRXN": 1, "ADUMB": 1}
    fake = types.ModuleType("pycharmm")
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    require_adumbrxncor_for_umbrella_rxncor("umbrella rxncor nresol 40 name r_nc")
    mock_lingo.get_energy_value.assert_not_called()