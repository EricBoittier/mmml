"""Validate DCM:5 benchmark config and md-system argv building."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

WORKFLOW_ROOT = (
    Path(__file__).resolve().parents[2] / "workflows" / "dcm5_md_benchmark"
)
SCRIPTS = WORKFLOW_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_lib import (  # noqa: E402
    build_md_system_argv,
    job_metadata,
    load_config,
    namespace_for_job,
)

EXPECTED_JOB_COUNT = 15


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(WORKFLOW_ROOT / "config.yaml")


def test_resolve_mmml_cmd_uses_md_system_subcommand() -> None:
    sys.path.insert(0, str(SCRIPTS))
    from run_job import _resolve_mmml_cmd  # noqa: E402

    cmd = _resolve_mmml_cmd(["--setup", "free_nve"])
    assert "md-system" in cmd
    idx = cmd.index("md-system")
    assert cmd[idx + 1] == "--setup"
    assert cmd[idx + 2] == "free_nve"


def test_validate_checkpoint_rejects_placeholder(tmp_path: Path) -> None:
    from benchmark_lib import validate_checkpoint  # noqa: E402

    bad = tmp_path / "path" / "to" / "dcm1-ckpt"
    bad.mkdir(parents=True)
    try:
        validate_checkpoint(bad)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "placeholder" in str(exc).lower()


def test_config_has_fifteen_jobs(cfg: dict) -> None:
    assert len(cfg["jobs"]) == EXPECTED_JOB_COUNT


def test_config_composition_is_dcm5(cfg: dict) -> None:
    assert cfg["composition"] == "DCM:5"
    assert cfg["ps"] == 2.0
    assert cfg["nsteps_target"] == 8000


@pytest.mark.parametrize(
    "job_id",
    [
        "ase_vac_nve",
        "ase_pbc_nve",
        "jaxmd_vac_nve",
        "jaxmd_pbc_nve",
        "pycharmm_vac_nve",
        "pycharmm_pbc_nve",
        "ase_vac_nvt_nhc",
        "ase_vac_nvt_langevin",
        "ase_pbc_nvt_nhc",
        "ase_pbc_nvt_langevin",
        "jaxmd_vac_nvt",
        "jaxmd_pbc_nvt",
        "pycharmm_vac_heat_scale",
        "pycharmm_vac_heat_hoover",
        "jaxmd_pbc_npt",
    ],
)
def test_build_md_system_argv_includes_core_flags(cfg: dict, job_id: str) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    argv = build_md_system_argv(cfg, job_id)
    assert "--composition" in argv
    assert "DCM:5" in argv
    assert "--dt-fs" in argv
    assert "--ps" in argv
    idx = argv.index("--ps")
    assert argv[idx + 1] == "2.0"


def test_pycharmm_heat_hoover_argv(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    argv = build_md_system_argv(cfg, "pycharmm_vac_heat_hoover")
    assert "--heat-thermostat" in argv
    idx = argv.index("--heat-thermostat")
    assert argv[idx + 1] == "hoover"
    idx = argv.index("--heat-ihtfrq")
    assert argv[idx + 1] == "0"
    assert "--ps-heat" in argv


def test_pycharmm_heat_scale_argv(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    argv = build_md_system_argv(cfg, "pycharmm_vac_heat_scale")
    idx = argv.index("--heat-thermostat")
    assert argv[idx + 1] == "scale"
    idx = argv.index("--heat-ihtfrq")
    assert argv[idx + 1] == "100"


def test_ase_vac_nve_warms_initial_temperature(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    argv = build_md_system_argv(cfg, "ase_vac_nve")
    assert "--extra-args" in argv
    extra_idx = argv.index("--extra-args")
    extra = argv[extra_idx + 1 :]
    assert "--nve-temp-K" in extra
    t_idx = extra.index("--nve-temp-K")
    assert extra[t_idx + 1] == "300"


def test_vacuum_jobs_use_packmol_sphere(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    for job_id, job in cfg["jobs"].items():
        if job.get("pbc"):
            continue
        argv = build_md_system_argv(cfg, job_id)
        assert "--packmol-sphere" in argv, job_id
        assert "--packmol-radius" in argv, job_id


def test_pbc_jobs_use_box_size(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    for job_id, job in cfg["jobs"].items():
        if not job.get("pbc"):
            continue
        argv = build_md_system_argv(cfg, job_id)
        assert "--box-size" in argv, job_id
        idx = argv.index("--box-size")
        expected = str(job.get("box_size", cfg["box_size"]))
        assert argv[idx + 1] == expected


def test_pbc_jobs_use_packmol_sphere(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    for job_id, job in cfg["jobs"].items():
        if not job.get("pbc"):
            continue
        argv = build_md_system_argv(cfg, job_id)
        assert "--packmol-sphere" in argv, job_id


def test_jaxmd_pbc_npt_uses_larger_box(cfg: dict) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    argv = build_md_system_argv(cfg, "jaxmd_pbc_npt")
    idx = argv.index("--box-size")
    assert argv[idx + 1] == "35.0"


@pytest.mark.parametrize("job_id", ["ase_vac_nve", "jaxmd_pbc_npt", "pycharmm_vac_nve"])
def test_namespace_builds_backend_argv(cfg: dict, job_id: str) -> None:
    os.environ.setdefault("MMML_CKPT", "/tmp/dcm5_test_ckpt")
    backend, backend_argv, args = namespace_for_job(cfg, job_id)
    job = cfg["jobs"][job_id]
    assert backend == job["backend"]
    assert backend_argv
    assert args.setup == job["setup"]


def test_job_metadata_fields(cfg: dict) -> None:
    meta = job_metadata(cfg, "jaxmd_pbc_npt")
    assert meta["backend"] == "jaxmd"
    assert meta["setup"] == "pbc_npt"
    assert meta["pbc"] is True
    assert meta["integrator"] == "npt_nhc"
