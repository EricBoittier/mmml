"""Unit tests for pbc_solvent_burst monitor_lib."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "pbc_solvent_burst"
    / "scripts"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from campaign_lib import RunCell  # noqa: E402
from monitor_lib import (  # noqa: E402
    grep_errors,
    inspect_run,
    parse_dyna_lines,
    summarize_dyna,
)


def test_parse_dyna_lines_fixed_width():
    text = "\n".join(
        [
            "DYNA>         100       0.0250   -1234.5678901    12.3456789   -1246.9135690   298.1234",
            "DYNA>         200       0.0500   -1234.5600000    12.3400000   -1246.9000000   150.0000",
        ]
    )
    rows = parse_dyna_lines(text)
    assert len(rows) == 2
    assert rows[0]["step"] == 100.0
    assert rows[1]["temperature_K"] == pytest.approx(150.0)


def test_summarize_dyna_drift():
    rows = parse_dyna_lines(
        "DYNA> 100 0.1 -1000.0 10.0 -1010.0 300.0\n"
        "DYNA> 200 0.2 -1001.0 10.0 -1011.0 305.0\n"
    )
    summary = summarize_dyna(rows)
    assert summary["n_frames"] == 2
    assert summary["total_energy_drift_kcal"] == pytest.approx(-1.0)
    assert summary["temperature_mean_K"] == pytest.approx(302.5)


def test_grep_errors_detects_segfault():
    text = "[gpu09:909494] Signal: Segmentation fault (11)\n"
    assert grep_errors(text)


def test_inspect_run_pending(tmp_path, monkeypatch):
    out_dir = tmp_path / "dcm_10"
    out_dir.mkdir()
    cfg = {
        "jaxmd_bursts": 2,
        "pycharmm_equi_legs": 2,
        "output_root": str(tmp_path),
        "cluster_sizes": [10],
        "bulk_density_fractions": None,
        "temperatures": [300.0],
        "box_sizes": [32.0],
    }
    cell = RunCell(solvent="DCM", n_monomers=10, temperature=300.0, box_size=32.0)
    monkeypatch.setattr(
        "monitor_lib.paths_for_run",
        lambda _c, _cell: {
            "out_dir": out_dir,
            "campaign_yaml": out_dir / "campaign.yaml",
            "campaign_summary": out_dir / "campaign_summary.json",
            "final_handoff": out_dir / "jaxmd_burst_02" / "handoff" / "state.npz",
            "done": out_dir / "done.txt",
        },
    )
    m = inspect_run(cfg, cell)
    assert m.status == "pending"
    assert m.run_tag == "dcm_10"


def test_inspect_run_running_with_dyna(tmp_path, monkeypatch):
    out_dir = tmp_path / "dcm_10"
    out_dir.mkdir()
    log = out_dir / "stdout.log"
    log.write_text(
        "MLpot SD minimize: 300 steps\n"
        "DYNA>         500       0.1000   -5000.0    100.0   -5100.0   295.0\n"
        "DYNA>        1000       0.2000   -4999.0    101.0   -5100.0   298.0\n",
        encoding="utf-8",
    )
    (out_dir / "pycharmm_init" / "handoff").mkdir(parents=True)
    (out_dir / "pycharmm_init" / "handoff" / "state.npz").write_bytes(b"stub")

    cfg = {
        "jaxmd_bursts": 2,
        "pycharmm_equi_legs": 2,
        "output_root": str(tmp_path),
        "cluster_sizes": [10],
        "bulk_density_fractions": None,
        "temperatures": [300.0],
        "box_sizes": [32.0],
    }
    cell = RunCell(solvent="DCM", n_monomers=10, temperature=300.0, box_size=32.0)

    def fake_paths(_cfg, _cell):
        return {
            "out_dir": out_dir,
            "campaign_yaml": out_dir / "campaign.yaml",
            "campaign_summary": out_dir / "campaign_summary.json",
            "final_handoff": out_dir / "jaxmd_burst_02" / "handoff" / "state.npz",
            "done": out_dir / "done.txt",
        }

    monkeypatch.setattr("monitor_lib.paths_for_run", fake_paths)
    m = inspect_run(cfg, cell)
    assert m.status in {"running", "partial", "started"}
    assert m.dyna["n_frames"] == 2
    assert m.dyna["temperature_last_K"] == pytest.approx(298.0)


def test_inspect_run_failed_from_summary(tmp_path, monkeypatch):
    out_dir = tmp_path / "dcm_10"
    out_dir.mkdir()
    (out_dir / "stdout.log").write_text(
        "dcm_10 burst campaign failed with exit code -11\n",
        encoding="utf-8",
    )
    summary = {
        "jobs": [
            {
                "job_id": "pycharmm_init",
                "exit_code": -11,
                "stages": [
                    {
                        "stage": "heat",
                        "ps_requested": 1.0,
                        "ps_completed": 0.3,
                    }
                ],
            }
        ]
    }
    (out_dir / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    cfg = {
        "jaxmd_bursts": 2,
        "pycharmm_equi_legs": 2,
        "output_root": str(tmp_path),
        "cluster_sizes": [10],
        "bulk_density_fractions": None,
        "temperatures": [300.0],
        "box_sizes": [32.0],
    }
    cell = RunCell(solvent="DCM", n_monomers=10, temperature=300.0, box_size=32.0)

    monkeypatch.setattr(
        "monitor_lib.paths_for_run",
        lambda _c, _cell: {
            "out_dir": out_dir,
            "campaign_yaml": out_dir / "campaign.yaml",
            "campaign_summary": out_dir / "campaign_summary.json",
            "final_handoff": out_dir / "jaxmd_burst_02" / "handoff" / "state.npz",
            "done": out_dir / "done.txt",
        },
    )
    m = inspect_run(cfg, cell)
    assert m.status == "failed"
    assert m.health == "BAD"
    assert m.campaign_failed_leg == "pycharmm_init"
