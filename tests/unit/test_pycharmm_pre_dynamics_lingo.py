"""Unit tests for pre-dynamics PyCHARMM lingo helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_pre_dynamics_lingo_from_args,
    normalize_pycharmm_pre_dynamics_lingo,
    parse_adumb_rc_params,
    parse_adumb_rc_wall_params,
    require_adumbrxncor_for_umbrella_rxncor,
    resolve_pre_dynamics_lingo_script,
    run_charmm_lingo_script,
    script_uses_umbrella_rxncor,
    split_charmm_lingo_commands,
    strip_mmfp_blocks_from_script,
    substitute_adumb_rc_tokens,
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


def test_mmfp_rcm_distance_wall_script_uses_outside_harmonic() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _mmfp_adumb_rc_distance_walls_script,
        adumb_rc_wall_droff,
    )

    droff = adumb_rc_wall_droff(8.0, margin=0.75)
    script = _mmfp_adumb_rc_distance_walls_script(((5, 4, droff, 500.0),))
    assert "GEO sphere RCM distance" in script
    assert "harmonic outside force 500 droff 7.25" in script
    assert "sele atom 6 end" in script
    assert "sele atom 5 end" in script
    assert script.strip().endswith("END")


def test_noe_adumb_rc_distance_wall_script_upper_bound() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _noe_adumb_rc_distance_wall_assign,
        _noe_adumb_rc_distance_walls_script,
    )

    assign = _noe_adumb_rc_distance_wall_assign(5, 4, rmax=7.25, kmax=500.0)
    assert len(assign) <= 78
    assert "-" not in assign
    assert "sele atom 6 end sele atom 5 end" in assign
    script = _noe_adumb_rc_distance_walls_script(((5, 4, 7.25, 500.0),))
    assert script.startswith("noe\nreset\n")
    assert assign in script
    assert script.strip().endswith("end")


def test_resd_adumb_rc_distance_wall_commands_positive_upper_bound() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _resd_adumb_rc_distance_wall_commands,
    )

    cmds = _resd_adumb_rc_distance_wall_commands(((5, 4, 7.25, 500.0),))
    assert cmds[0] == "RESDistance RESEt"
    assert cmds[1] == (
        "RESDistance KVAL 500 RVAL 7.25 POSITIVE 1.0 BYNU 6 5"
    )
    assert all(len(c) <= 78 for c in cmds)


def test_charmm_output_indicates_failure_detects_resd_syntax() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _charmm_output_indicates_failure,
        _resd_restraint_count_from_log,
    )

    log = "ERROR IN NXTATM: Unrecognizable SEGID or residue number"
    assert _charmm_output_indicates_failure(log) == (
        "CHARMM could not parse restraint atom tokens"
    )
    count_log = """
    RESDIST:  Current number of restraints=   0
    RESDIST:  Current number of restraints=   2
    """
    assert _resd_restraint_count_from_log(count_log) == 2


def test_resd_restraint_count_skips_when_capture_empty() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _resd_restraint_count_from_log,
    )

    assert _resd_restraint_count_from_log("") is None


def test_split_charmm_lingo_keeps_noe_block_together() -> None:
    script = """
    noe
    reset
    assi sele atom 5 end sele atom 4 end kmin 0 rmin 0 kmax 500 rmax 7.25
    end
    """
    cmds = split_charmm_lingo_commands(script)
    assert len(cmds) == 1
    assert cmds[0].startswith("noe\nreset\n")
    assert cmds[0].strip().endswith("end")


def test_parse_adumb_rc_wall_params_reads_set_commands() -> None:
    script = """
    set adumrcmax = 8.0
    set adumrcwall = 500.0
    umbrella rxncor nresol 20 name rcl
    """
    assert parse_adumb_rc_wall_params(script) == (8.0, 500.0)
    assert parse_adumb_rc_params(script) == (8.0, 500.0)
    assert parse_adumb_rc_wall_params("umbrella rxncor name rcl") is None


def test_parse_adumb_rc_wall_params_caps_at_umbrella_max() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_adumb_wall_rcmax,
    )

    script = """
    set adumrcmax = 8.0
    set adumrcwall = 500.0
    umbrella rxncor nresol 40 trig 0 poly 6 min 0.0 max 6.0 name rcl
    """
    assert resolve_adumb_wall_rcmax(8.0, script) == 6.0
    assert parse_adumb_rc_wall_params(script) == (6.0, 500.0)


def test_parse_adumb_rc_params_accepts_legacy_underscore_names() -> None:
    script = """
    set adum_rcmax = 8.0
    set adum_rcwall = 500.0
    """
    assert parse_adumb_rc_params(script) == (8.0, 500.0)


def test_substitute_adumb_rc_tokens_replaces_at_references() -> None:
    cmd = "umbrella rxncor min 0.0 max @adum_rcmax name rcl"
    assert (
        substitute_adumb_rc_tokens(cmd, rcmax=8.0, rcwall=None)
        == "umbrella rxncor min 0.0 max 8 name rcl"
    )
    assert (
        substitute_adumb_rc_tokens(
            "umbrella rxncor max @adumrcmax name rcn",
            rcmax=8.0,
            rcwall=None,
        )
        == "umbrella rxncor max 8 name rcn"
    )


def test_strip_mmfp_blocks_from_script() -> None:
    script = """
    set adumrcmax = 8.0
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
    assert "set adumrcmax = 8.0" in stripped
    assert stripped.strip().endswith("ucun 50")


def test_run_charmm_lingo_expands_adumb_tokens_before_umbrella(tmp_path: Path) -> None:
    script = """
    set adumrcmax = 8.0
    set adumrcwall = 500.0
    rxncor define rcl distance pcl pc
    umbrella rxncor nresol 20 trig 0 poly 4 min 0.0 max @adumrcmax name rcl
    umbrella init nsim 4 update 50 equi 25 thresh 10 temp 300 wuni 44 ucun 50
    """
    inp = tmp_path / "lingo.inp"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.require_adumbrxncor_for_umbrella_rxncor",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.adumb_rc_walls_enabled",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ) as install, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
        return_value=True,
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=inp, workdir=tmp_path)
    install.assert_not_called()
    assert script_fn.call_count == 5
    assert script_fn.call_args_list[0].args[0] == "set adumrcmax = 8.0"
    assert script_fn.call_args_list[1].args[0] == "set adumrcwall = 500.0"
    assert script_fn.call_args_list[2].args[0] == "rxncor define rcl distance pcl pc"
    assert script_fn.call_args_list[3].args[0] == (
        "umbrella rxncor nresol 20 trig 0 poly 4 min 0.0 max 8 name rcl"
    )
    assert script_fn.call_args_list[4].args[0].startswith("umbrella init")


def test_run_charmm_lingo_installs_adumb_walls_when_enabled(tmp_path: Path) -> None:
    script = """
    set adumrcmax = 8.0
    set adumrcwall = 500.0
    umbrella rxncor nresol 20 min 0.0 max @adumrcmax name rcl
    umbrella init nsim 4 update 50 equi 25 thresh 10 temp 300 wuni 44 ucun 50
    """
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.require_adumbrxncor_for_umbrella_rxncor",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.adumb_rc_walls_enabled",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ) as install, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
        return_value=True,
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=tmp_path / "lingo.inp", workdir=tmp_path)
    install.assert_called_once_with(rcmax=8.0, rcwall=500.0)
    assert script_fn.call_count == 4
    assert script_fn.call_args_list[-1].args[0].startswith("umbrella init")


def test_apply_pre_dynamics_lingo_sets_adumb_rc_guard(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="""
        set adumrcmax = 8.0
        set adumrcwall = 500.0
        umbrella rxncor nresol 20 min 0.0 max @adumrcmax name rcl
        umbrella init nsim 4 update 50 equi 25 thresh 10 temp 300 wuni 44 ucun 50
        """,
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=True,
        output_dir=tmp_path,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script",
    ):
        apply_pre_dynamics_lingo_from_args(args)
    guard = getattr(args, "_adumb_rc_guard", None)
    assert guard is not None
    assert guard.rcmax == 8.0
    assert guard.rcwall == 500.0
    assert guard.wall_droff() == pytest.approx(7.25)


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