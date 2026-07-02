from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface import packmol_placement


SAMPLE_SUCCESS_LOG = """
 Reading input file... (Control-C aborts)
################################################################################
                                 Success!
              Final objective function value:  1.23456E+02
              Maximum violation of target distance:   0.123456
              Maximum violation of the constraints:  1.00000E-03
################################################################################
"""

SAMPLE_ERROR_LOG = """
 Reading input file... (Control-C aborts)
 ERROR: Could not read any atom from file: missing.pdb
"""


def test_parse_packmol_log_success():
    parsed = packmol_placement.parse_packmol_log(SAMPLE_SUCCESS_LOG)
    assert parsed["success"] is True
    assert parsed["objective"] == pytest.approx(123.456)
    assert parsed["max_distance_violation"] == pytest.approx(0.123456)
    assert parsed["max_constraint_violation"] == pytest.approx(1e-3)
    assert parsed["error_message"] is None


def test_parse_packmol_log_error():
    parsed = packmol_placement.parse_packmol_log(SAMPLE_ERROR_LOG)
    assert parsed["success"] is False
    assert parsed["error_message"] is not None
    assert "ERROR:" in parsed["error_message"]


def test_execute_packmol_script_captures_output(tmp_path, monkeypatch):
    inp = tmp_path / "pack.inp"
    monkeypatch.setattr(
        packmol_placement,
        "packmol_executable",
        lambda: "/usr/bin/packmol",
    )

    proc = mock.Mock(
        returncode=0,
        stdout=SAMPLE_SUCCESS_LOG,
        stderr="",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.subprocess.run",
        return_value=proc,
    ) as run_mock:
        result = packmol_placement.execute_packmol_script("seed 1\n", inp)

    assert run_mock.call_args.kwargs["capture_output"] is True
    assert result.success is True
    assert result.inp_path == inp
    assert result.max_distance_violation == pytest.approx(0.123456)


def test_execute_packmol_script_raises_without_printing(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "pack.inp"
    monkeypatch.setattr(
        packmol_placement,
        "packmol_executable",
        lambda: "/usr/bin/packmol",
    )
    proc = mock.Mock(
        returncode=1,
        stdout=SAMPLE_ERROR_LOG,
        stderr="",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.subprocess.run",
        return_value=proc,
    ):
        with pytest.raises(RuntimeError, match="ERROR:"):
            packmol_placement.execute_packmol_script("seed 1\n", inp)

    captured = capsys.readouterr()
    assert "Running:" not in captured.out
    assert "Reading input file" not in captured.out


def test_emit_packmol_build_summary_plain(capsys):
    packmol_placement.emit_packmol_build_summary(
        placement="cube",
        composition=[("DCM", 10)],
        center=(19.0, 19.0, 19.0),
        tolerance=2.0,
        seed=42,
        output_pdb=Path("out.pdb"),
        cube_side=30.0,
        sim_cell_side=38.0,
        result=packmol_placement.PackmolRunResult(
            exit_code=0,
            log_text="",
            inp_path=Path("in.inp"),
            success=True,
            max_distance_violation=0.1,
        ),
        n_atoms=50,
        span_A=(10.0, 11.0, 12.0),
    )
    out = capsys.readouterr().out
    assert "Packmol" in out
    assert "DCM:10" in out
    assert "success" in out
