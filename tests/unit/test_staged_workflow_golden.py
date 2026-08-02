"""The characterization harness must be able to fail.

``scripts/ci/staged_workflow_golden.py`` is the safety net for decomposing
``run_staged_workflow``. A comparator that silently passes would be worse than
none at all -- it would license a refactor while proving nothing -- so every
class of divergence it claims to detect is asserted here in both directions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.ci.staged_workflow_golden import capture, compare, main


def _summary(**overrides) -> dict:
    stage = {
        "stage": "heat",
        "backend": "pycharmm",
        "setup": "pbc_nve",
        "status": "completed",
        "integrator": "hoover",
        "nsteps_requested": 100,
        "nsteps_completed": 100,
        "frames_written": 10,
        "record_every_steps": 10,
        "dt_fs": 0.5,
        "ps_requested": 0.05,
        "ps_completed": 0.05,
        "temperature_K": 300.0,
        "temperature_mean_K": 298.4,
        "temperature_final_K": 301.1,
        "temperature_first_K": 12.0,
        "pressure_atm": None,
        "pressure_mean_atm": None,
        "box_A_initial": 20.0,
        "box_A_final": 20.0,
        "volume_A3_final": 8000.0,
        "density_g_cm3_final": 0.99,
        "density_g_cm3_mean": 0.98,
        "wall_time_s": 123.4,
        "job_id": "run",
        "description": "free text that must not be compared",
        "artifacts": ["/abs/path/heat.res", "/abs/path/heat.dcd"],
    }
    stage.update(overrides)
    return {
        "job_id": "run",
        "backend": "pycharmm",
        "setup": "pbc_nve",
        "exit_code": 0,
        "wall_time_s": 999.0,
        "handoff": {},
        "stages": [stage],
    }


def _run_dir(tmp_path: Path, summary: dict, files: dict[str, bytes] | None = None) -> Path:
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    (d / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, blob in (files or {"heat.res": b"x" * 1000, "heat.dcd": b"y" * 10000}).items():
        (d / name).write_bytes(blob)
    return d


# --- capture ----------------------------------------------------------------


def test_capture_requires_a_stage_summary(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage_summary.json"):
        capture(tmp_path)


def test_capture_records_stage_order_and_artifacts(tmp_path):
    rec = capture(_run_dir(tmp_path, _summary()))
    assert rec["stage_order"] == ["heat"]
    assert rec["stages"][0]["_artifacts"] == ["heat.dcd", "heat.res"]
    assert "heat.res:1e3" in rec["manifest"]


def test_capture_strips_absolute_paths_from_artifacts(tmp_path):
    """The directory is run-specific; only the lineage of names matters."""
    rec = capture(_run_dir(tmp_path, _summary()))
    assert all("/" not in name for name in rec["stages"][0]["_artifacts"])


def test_capture_flags_an_empty_artifact(tmp_path):
    d = _run_dir(tmp_path, _summary(), files={"heat.res": b""})
    assert "heat.res:empty" in capture(d)["manifest"]


# --- fields that must NOT trip the comparator -------------------------------


@pytest.mark.parametrize("field", ["wall_time_s", "job_id", "description"])
def test_run_to_run_noise_is_ignored(tmp_path, field):
    golden = capture(_run_dir(tmp_path, _summary()))
    noisy = capture(_run_dir(tmp_path, _summary(**{field: "totally different"})))
    assert compare(golden, noisy, rtol=1e-6) == []


def test_identical_runs_compare_clean(tmp_path):
    golden = capture(_run_dir(tmp_path, _summary()))
    assert compare(golden, copy.deepcopy(golden), rtol=1e-6) == []


# --- divergences that MUST be caught ----------------------------------------


def _diff(tmp_path, **overrides) -> list[str]:
    golden = capture(_run_dir(tmp_path, _summary()))
    current = capture(_run_dir(tmp_path, _summary(**overrides)))
    return compare(golden, current, rtol=1e-6)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "skipped"),
        ("nsteps_completed", 50),
        ("frames_written", 1),
        ("record_every_steps", 5),
        ("integrator", "scale"),
        ("backend", "jaxmd"),
    ],
)
def test_a_changed_decision_is_reported(tmp_path, field, value):
    diffs = _diff(tmp_path, **{field: value})
    assert diffs and any(field in d for d in diffs)


def test_a_drifting_temperature_is_reported(tmp_path):
    """1% is far beyond refactor noise; a pure refactor moves nothing."""
    diffs = _diff(tmp_path, temperature_mean_K=301.4)
    assert any("temperature_mean_K" in d for d in diffs)


def test_a_tolerable_float_wobble_is_accepted(tmp_path):
    golden = capture(_run_dir(tmp_path, _summary()))
    current = capture(_run_dir(tmp_path, _summary(temperature_mean_K=298.4 + 1e-9)))
    assert compare(golden, current, rtol=1e-6) == []


def test_a_lost_artifact_is_reported(tmp_path):
    golden = capture(_run_dir(tmp_path, _summary()))
    current = capture(
        _run_dir(tmp_path, _summary(artifacts=["/abs/heat.res"]), files={"heat.res": b"x" * 1000})
    )
    diffs = compare(golden, current, rtol=1e-6)
    assert any("artifacts" in d for d in diffs)


def test_a_changed_stage_order_short_circuits(tmp_path):
    golden = capture(_run_dir(tmp_path, _summary()))
    current = copy.deepcopy(golden)
    current["stage_order"] = ["equi"]
    diffs = compare(golden, current, rtol=1e-6)
    assert len(diffs) == 1 and "stage order" in diffs[0]


def test_a_changed_exit_code_is_reported(tmp_path):
    golden = capture(_run_dir(tmp_path, _summary()))
    current = copy.deepcopy(golden)
    current["exit_code"] = 2
    assert any("exit_code" in d for d in compare(golden, current, rtol=1e-6))


def test_a_none_becoming_a_number_is_reported(tmp_path):
    """A pressure that starts being recorded means the ensemble changed."""
    diffs = _diff(tmp_path, pressure_mean_atm=1.0)
    assert any("pressure_mean_atm" in d for d in diffs)


# --- CLI --------------------------------------------------------------------


def test_cli_capture_then_compare_round_trips(tmp_path, capsys):
    run = _run_dir(tmp_path, _summary())
    golden = tmp_path / "golden.json"

    assert main(["capture", str(run), "-o", str(golden)]) == 0
    assert golden.is_file()
    assert main(["compare", str(run), str(golden)]) == 0
    assert "unchanged" in capsys.readouterr().out


def test_cli_reports_a_divergence_with_nonzero_status(tmp_path, capsys):
    run = _run_dir(tmp_path, _summary())
    golden = tmp_path / "golden.json"
    main(["capture", str(run), "-o", str(golden)])

    drifted = _run_dir(tmp_path, _summary(nsteps_completed=7))
    assert main(["compare", str(drifted), str(golden)]) == 1

    err = capsys.readouterr().err
    assert "diverged" in err and "nsteps_completed" in err
