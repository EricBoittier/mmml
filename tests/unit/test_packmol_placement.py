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

    args, kwargs = run_mock.call_args
    assert args[0] == ["/usr/bin/packmol", "-i", str(inp.resolve())]
    assert kwargs["capture_output"] is True
    assert kwargs["cwd"] == str(inp.resolve().parent)
    assert result.success is True
    assert result.inp_path == inp.resolve()
    assert result.max_distance_violation == pytest.approx(0.123456)


def test_packmol_failure_message_uses_exit_label():
    result = packmol_placement.PackmolRunResult(
        exit_code=173,
        log_text="",
        inp_path=Path("x.inp"),
        success=False,
    )
    assert "failed to converge" in packmol_placement.packmol_failure_message(result)


def test_packmol_failure_message_uses_log_tail():
    result = packmol_placement.PackmolRunResult(
        exit_code=2,
        log_text="Reading input file...\nERROR: Could not read any atom from file: x.pdb\n",
        inp_path=Path("x.inp"),
        success=False,
        error_message="ERROR: Could not read any atom from file: x.pdb",
    )
    assert "Could not read any atom" in packmol_placement.packmol_failure_message(result)


def test_execute_packmol_script_raises_without_printing(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "pack.inp"
    monkeypatch.setattr(
        packmol_placement,
        "packmol_executable",
        lambda: "/usr/bin/packmol",
    )
    proc = mock.Mock(
        returncode=171,
        stdout=SAMPLE_ERROR_LOG,
        stderr="",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.subprocess.run",
        return_value=proc,
    ):
        with pytest.raises(RuntimeError, match="input error|ERROR:"):
            packmol_placement.execute_packmol_script("seed 1\n", inp)


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
