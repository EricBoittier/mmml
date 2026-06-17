"""Unit tests for MD stage summary / campaign plan helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from mmml.cli.run.md_stage_summary import (
    MdJobSummary,
    MdStageSummary,
    build_pycharmm_plan_rows,
    build_single_leg_plan_row,
    dynamics_nstep_from_ps,
    ps_from_nsteps,
    write_campaign_plan,
    write_stage_summary_json,
)


def test_dynamics_nstep_from_ps() -> None:
    assert dynamics_nstep_from_ps(2.0, 0.25) == 8000
    assert ps_from_nsteps(8000, 0.25) == 2.0


def test_build_single_leg_plan_row() -> None:
    args = Namespace(
        setup="pbc_npt",
        ps=2.0,
        dt_fs=0.25,
        temperature=260.0,
        pressure=10.0,
        box_size=38.0,
        dcd_nsavc=800,
        steps_per_recording=800,
    )
    row = build_single_leg_plan_row("jaxmd_prod", args, "jaxmd")
    assert row.nsteps_requested == 8000
    assert row.pressure_atm == pytest.approx(10.0)


def test_build_pycharmm_plan_rows_expands_stages() -> None:
    args = Namespace(
        setup="pbc_npt",
        md_stages="mini,heat,equi",
        dt_fs=0.25,
        ps_heat=20.0,
        ps_equi=20.0,
        temperature=260.0,
        pressure=10.0,
        heat_firstt=52.0,
        heat_finalt=260.0,
        dcd_nsavc=800,
        mini_nstep=100,
    )
    rows = build_pycharmm_plan_rows("equil", args)
    stages = [r.stage for r in rows]
    assert stages == ["mini", "heat", "equi"]
    heat = next(r for r in rows if r.stage == "heat")
    assert heat.temperature_first_K == pytest.approx(52.0)
    assert heat.nsteps_requested == dynamics_nstep_from_ps(20.0, 0.25)


def test_write_stage_summary_json(tmp_path: Path) -> None:
    stage = MdStageSummary(
        stage="dynamics",
        job_id="jaxmd",
        backend="jaxmd",
        setup="pbc_nve",
        nsteps_requested=100,
        nsteps_completed=100,
        dt_fs=0.25,
        ps_requested=0.025,
        ps_completed=0.025,
        status="complete",
    )
    job = MdJobSummary(job_id="jaxmd", backend="jaxmd", setup="pbc_nve", stages=[stage])
    path = write_stage_summary_json(job, tmp_path)
    data = json.loads(path.read_text())
    assert data["job_id"] == "jaxmd"
    assert data["stages"][0]["stage"] == "dynamics"


def test_write_campaign_plan(tmp_path: Path) -> None:
    row = MdStageSummary(
        stage="heat",
        job_id="equil",
        backend="pycharmm",
        setup="pbc_npt",
        nsteps_requested=80000,
        nsteps_completed=0,
        dt_fs=0.25,
        ps_requested=20.0,
        ps_completed=0.0,
    )
    path = write_campaign_plan(tmp_path / "campaign_plan.json", [row])
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert payload[0]["stage"] == "heat"
