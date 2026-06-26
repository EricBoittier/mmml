"""Unit tests for NPT pressure-tensor helpers (no CHARMM runtime)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmTrajectoryFiles,
    _apply_npt_cpt_kwargs,
    build_cpt_equilibration_dynamics,
)
from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
    NptPressureTensor,
    apply_npt_pressure_log_kwargs,
    apply_npt_pressure_reference,
    attach_pressure_tensor_log,
    maybe_configure_stage_pressure_tensor_io,
    maybe_report_instantaneous_pressure_tensor,
    parse_npt_pressure_tensor,
    report_instantaneous_pressure_tensor,
    resolve_npt_cpt_pressure_options,
)


def test_parse_isotropic_tensor_returns_none():
    assert parse_npt_pressure_tensor("1,1,1,0,0,0", isotropic_pref=1.0) is None
    assert parse_npt_pressure_tensor(None, isotropic_pref=1.0) is None


def test_parse_anisotropic_tensor():
    tensor = parse_npt_pressure_tensor("2,1,1,0,0,0", isotropic_pref=1.0)
    assert tensor == NptPressureTensor(2.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def test_parse_tensor_bad_length():
    with pytest.raises(ValueError, match="6 comma-separated"):
        parse_npt_pressure_tensor("1,1,1", isotropic_pref=1.0)


def test_apply_isotropic_pressure_reference():
    kw: dict = {}
    apply_npt_pressure_reference(kw, pref=2.5, pressure_tensor=None)
    assert kw["pint pconst pref"] == 2.5
    assert "PRXX" not in kw


def test_apply_anisotropic_pressure_reference():
    kw: dict = {}
    tensor = NptPressureTensor(2.0, 1.0, 1.0, 0.1, 0.0, -0.2)
    apply_npt_pressure_reference(kw, pref=1.0, pressure_tensor=tensor)
    assert kw["pint"] is True
    assert kw["pconst"] is True
    assert "pint pconst pref" not in kw
    assert kw["PRXX"] == 2.0
    assert kw["PRXY"] == 0.1
    assert kw["PRYZ"] == -0.2


def test_apply_pressure_log_kwargs():
    kw: dict = {}
    apply_npt_pressure_log_kwargs(kw, interval_steps=5, iupten_unit=29)
    assert kw["iptfrq"] == 5
    assert kw["iupten"] == 29
    apply_npt_pressure_log_kwargs(kw, interval_steps=0)
    assert "iptfrq" not in kw or kw.get("iptfrq") == 5


def test_build_cpt_equilibration_anisotropic():
    tensor = NptPressureTensor(3.0, 2.0, 1.0)
    kw = build_cpt_equilibration_dynamics(
        pref=1.0,
        pressure_tensor=tensor,
        pmass=10,
        tmass=100,
    )
    assert kw["PRXX"] == 3.0
    assert "pint pconst pref" not in kw


def test_resolve_npt_cpt_pressure_options():
    args = argparse.Namespace(
        npt_thermostat="hoover",
        npt_pressure=1.0,
        npt_pgamma=5.0,
        npt_pressure_tensor="2,1,1,0,0,0",
        npt_pressure_log_interval=10,
    )
    opts = resolve_npt_cpt_pressure_options(args)
    assert opts["pressure_tensor"].prxx == 2.0
    assert opts["pressure_log_interval"] == 10


def test_attach_pressure_tensor_log_sets_io_fields(tmp_path):
    io = CharmmTrajectoryFiles()
    log_path = tmp_path / "equi_pressure.dat"
    out = attach_pressure_tensor_log(io, log_path, unit=31)
    assert out == log_path
    assert io.pressure_tensor_log == log_path
    assert io.pressure_tensor_log_unit == 31


def test_maybe_configure_stage_pressure_tensor_io(tmp_path):
    io = CharmmTrajectoryFiles()
    kw = {"cpt": True}
    log_path = tmp_path / "prod_pressure.dat"
    maybe_configure_stage_pressure_tensor_io(
        io,
        kw,
        log_path=log_path,
        pressure_log_interval=3,
    )
    assert io.pressure_tensor_log == log_path
    assert kw["iptfrq"] == 3
    assert kw["iupten"] == 29


def test_maybe_report_skips_when_disabled():
    args = argparse.Namespace(skip_npt_pressure_report=True, quiet=True)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor.report_instantaneous_pressure_tensor"
    ) as report:
        maybe_report_instantaneous_pressure_tensor(
            stage="equi",
            temp=300.0,
            args=args,
            use_cpt=True,
        )
    report.assert_not_called()


def test_report_instantaneous_pressure_tensor_calls_charmm():
    ctx = MagicMock()
    mock_lingo = MagicMock()
    mock_pycharmm = MagicMock()
    mock_pycharmm.lingo = mock_lingo
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms"
    ) as refresh, patch.dict("sys.modules", {"pycharmm": mock_pycharmm}):
        report_instantaneous_pressure_tensor(
            280.0,
            context="EQUI",
            quiet=True,
            mlpot_ctx=ctx,
        )
    refresh.assert_called_once()
    mock_lingo.charmm_script.assert_called_once()
    assert "pressure instantaneous" in mock_lingo.charmm_script.call_args[0][0].lower()
    assert "280" in mock_lingo.charmm_script.call_args[0][0]


def test_apply_npt_cpt_kwargs_with_tensor():
    kw: dict = {}
    tensor = NptPressureTensor(1.5, 1.0, 0.5)
    _apply_npt_cpt_kwargs(
        kw,
        temp=300.0,
        pref=1.0,
        pmass=16,
        tmass=160,
        pressure_tensor=tensor,
    )
    assert kw["cpt"] is True
    assert kw["PRXX"] == 1.5
