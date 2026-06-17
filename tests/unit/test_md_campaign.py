"""Unit tests for in-process MD campaign runner helpers."""

from __future__ import annotations

import pytest

from mmml.cli.run.md_config import (
    expand_repeated_jobs,
    merge_campaign_job_config,
    topological_job_order,
)


def _sample_campaign() -> dict:
    return {
        "defaults": {"composition": "DCM:5", "seed": 1, "dt_fs": 0.25},
        "runs": {
            "equil": {"backend": "pycharmm", "setup": "pbc_npt", "output_dir": "a"},
            "prod": {
                "backend": "jaxmd",
                "setup": "pbc_npt",
                "depends_on": "equil",
                "output_dir": "b",
            },
            "nve": {
                "backend": "jaxmd",
                "setup": "pbc_nve",
                "depends_on": "prod",
                "repeat": 2,
                "output_dir": "c",
            },
        },
    }


def test_topological_job_order() -> None:
    order = topological_job_order(_sample_campaign())
    assert order.index("equil") < order.index("prod")
    assert order.index("prod") < order.index("nve")


def test_topological_order_unknown_dep_raises() -> None:
    bad = {"runs": {"a": {"depends_on": "missing"}}}
    with pytest.raises(ValueError, match="unknown job"):
        topological_job_order(bad)


def test_expand_repeated_jobs() -> None:
    expanded = expand_repeated_jobs(_sample_campaign(), ["equil", "prod", "nve"])
    run_ids = [rid for _base, rid, _rep in expanded]
    assert run_ids == ["equil", "prod", "nve.0", "nve.1"]


def test_merge_campaign_job_config_defaults() -> None:
    merged = merge_campaign_job_config(_sample_campaign(), "prod")
    assert merged["composition"] == "DCM:5"
    assert merged["depends_on"] == "equil"
    assert merged["backend"] == "jaxmd"


def test_namespace_from_merged_keeps_extra_args_last() -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    merged = merge_campaign_job_config(_sample_campaign(), "prod")
    merged["continue_from"] = "/tmp/handoff/state.npz"
    merged["extra_args"] = ["--steps-per-recording", "800"]
    args = namespace_from_merged(merged)
    assert getattr(args, "continue_from", None) is None
    assert args.extra_args == ["--steps-per-recording", "800"]
    assert "--job-id" not in args.extra_args
    assert "--continue-from" not in args.extra_args


def test_build_command_filters_campaign_flags_from_extra_args() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="DCM:20",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.25,
        temperature=160.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=1,
        box_size=40.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol_sphere=True,
        packmol_radius=10.0,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        extra_args=[
            "--steps-per-recording",
            "800",
            "--job-id",
            "jaxmd_prod",
            "--continue-from",
            "/tmp/state.npz",
        ],
    )
    backend, argv = build_command(args)
    assert backend == "jaxmd"
    assert "--job-id" not in argv
    assert "--continue-from" not in argv
    assert "--steps-per-recording" in argv


def test_resolve_dcd_nsavc_caps_frames_for_short_legs() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_dcd_nsavc

    assert resolve_dcd_nsavc(dcd_nsavc=1, nstep=400, dcd_max_frames=25) == 16
    assert resolve_dcd_nsavc(dcd_nsavc=1, nstep=400, dcd_max_frames=0) == 1
    assert resolve_dcd_nsavc(dcd_nsavc=500, nstep=4000, dcd_max_frames=25) == 500
