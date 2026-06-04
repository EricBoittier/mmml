"""Tests for DCM:5 benchmark result collector (fixture JSON/logs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKFLOW_ROOT = (
    Path(__file__).resolve().parents[2] / "workflows" / "dcm5_md_benchmark"
)
SCRIPTS = WORKFLOW_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from collect_benchmark import collect  # noqa: E402


def _write_ase_fixture(root: Path) -> None:
    job_dir = root / "ase_vac_nve"
    job_dir.mkdir(parents=True)
    summary = {
        "runs": {
            "vac_nve": {
                "log_samples": 17,
                "etot_drift_eV": 0.012,
                "temp_mean_K": 295.0,
                "temp_end_K": 298.0,
                "timings_s": {"md_integrator_loop_s": 1.23},
            }
        }
    }
    (job_dir / "suite_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (job_dir / "stdout.log").write_text("ok\n", encoding="utf-8")


def _write_jaxmd_fixture(root: Path) -> None:
    job_dir = root / "jaxmd_pbc_npt"
    job_dir.mkdir(parents=True)
    summary = {
        "status": "complete",
        "nsteps_completed": 7900,
        "nsteps_requested": 8000,
        "temperature_K": 301.0,
    }
    (job_dir / "suite_summary_jaxmd.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (job_dir / "stdout.log").write_text("ok\n", encoding="utf-8")


def _write_pycharmm_fixture(root: Path) -> None:
    job_dir = root / "pycharmm_vac_heat_hoover"
    job_dir.mkdir(parents=True)
    log = (
        "HEAT Hoover: 0.0 -> 240.0 K\n"
        "HEAT complete: restart_step=8000, dcd_frames=16\n"
        "AVER TEMP = 238.5\n"
    )
    (job_dir / "stdout.log").write_text(log, encoding="utf-8")


def test_collect_pycharmm_passes_when_restart_full_but_no_complete_line(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "pycharmm_vac_nve"
    job_dir.mkdir(parents=True)
    (job_dir / "nve_dcm_5.res").write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         100       10000         500         500          10        8000\n",
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text(
        "pycharmm_mlpot: error: DCD chunk truncated\n"
        "WRIDYN: RESTart file was written at step    8000\n",
        encoding="utf-8",
    )
    rows = collect(
        results_root=tmp_path,
        csv_path=tmp_path / "benchmark_summary.csv",
        md_path=tmp_path / "benchmark_report.md",
        config_path=WORKFLOW_ROOT / "config.yaml",
    )
    row = {r["job_id"]: r for r in rows}["pycharmm_vac_nve"]
    assert int(row["nsteps"]) == 8000
    assert row["status"] == "warn"
    assert "DCD" in row["notes"] or "complete" in row["notes"]


def test_collect_pycharmm_ignores_segment_nstep_in_log(tmp_path: Path) -> None:
    """Collector must not treat overlap chunk NSTEP=500 as total integrated steps."""
    job_dir = tmp_path / "pycharmm_vac_nve"
    job_dir.mkdir(parents=True)
    res = job_dir / "nve_dcm_5.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         100       10000         500         500          10        8000\n",
        encoding="utf-8",
    )
    log = (
        "overlap rescue SD=25\n"
        "NSTEP = 25\n"
        "NSTEP = 500\n"
        "WRIDYN: RESTart file was written at step    8000\n"
    )
    (job_dir / "stdout.log").write_text(log, encoding="utf-8")

    csv_path = tmp_path / "benchmark_summary.csv"
    md_path = tmp_path / "benchmark_report.md"
    rows = collect(
        results_root=tmp_path,
        csv_path=csv_path,
        md_path=md_path,
        config_path=WORKFLOW_ROOT / "config.yaml",
    )
    row = {r["job_id"]: r for r in rows}["pycharmm_vac_nve"]
    assert int(row["nsteps"]) == 8000
    assert row["status"] in {"pass", "warn"}


def test_collect_parses_fixture_outputs(tmp_path: Path) -> None:
    _write_ase_fixture(tmp_path)
    _write_jaxmd_fixture(tmp_path)
    _write_pycharmm_fixture(tmp_path)

    csv_path = tmp_path / "benchmark_summary.csv"
    md_path = tmp_path / "benchmark_report.md"
    rows = collect(
        results_root=tmp_path,
        csv_path=csv_path,
        md_path=md_path,
        config_path=WORKFLOW_ROOT / "config.yaml",
    )

    assert csv_path.is_file()
    assert md_path.is_file()

    by_id = {r["job_id"]: r for r in rows}
    assert by_id["ase_vac_nve"]["status"] == "pass"
    assert float(by_id["ase_vac_nve"]["energy_drift"]) == 0.012

    assert by_id["jaxmd_pbc_npt"]["status"] == "pass"
    assert int(by_id["jaxmd_pbc_npt"]["nsteps"]) == 7900

    assert by_id["pycharmm_vac_heat_hoover"]["status"] == "pass"
    assert int(by_id["pycharmm_vac_heat_hoover"]["nsteps"]) == 8000

    text = csv_path.read_text(encoding="utf-8")
    assert "ase_vac_nve" in text
    assert "jaxmd_pbc_npt" in text
