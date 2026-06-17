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
