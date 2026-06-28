"""MCP build_smoke recipe and geometry stage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mmml.mcp.recipes import (
    _recipe_stage_names,
    configure_run,
    load_recipe,
    run_recipe_stage,
)


def test_build_smoke_recipe_stages() -> None:
    recipe = load_recipe("build_smoke")
    assert recipe["name"] == "build_smoke"
    names = _recipe_stage_names(recipe)
    assert "make_res" in names
    assert "box_build" in names
    assert "hybrid_md_ase" in names
    assert "hybrid_md_jaxmd" in names
    assert "hybrid_md_pycharmm" in names


def test_configure_build_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMML_MCP_RUNS_ROOT", str(tmp_path / "runs"))
    result = configure_run("build_test_001", recipe="build_smoke", mode="smoke")
    run_dir = Path(result["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "configs" / "build_box.yaml").is_file()
    assert (run_dir / "configs" / "hybrid_ase.yaml").is_file()
    assert (run_dir / "configs" / "hybrid_jaxmd.yaml").is_file()
    assert (run_dir / "configs" / "hybrid_pycharmm.yaml").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "hybrid_md_jaxmd" in manifest["stages"]
    hybrid_jax = yaml.safe_load((run_dir / "configs" / "hybrid_jaxmd.yaml").read_text())
    assert hybrid_jax["defaults"]["packmol_radius"] == 6.9


@pytest.mark.parametrize(
    "stage",
    [
        "make_res",
        "box_build",
        "hybrid_md_ase",
        "hybrid_md_jaxmd",
        "hybrid_md_pycharmm",
    ],
)
def test_build_smoke_stage_dry_run(tmp_path, monkeypatch, stage: str) -> None:
    monkeypatch.setenv("MMML_MCP_RUNS_ROOT", str(tmp_path / "runs"))
    configure_run("build_dry", recipe="build_smoke", mode="smoke")
    result = run_recipe_stage("build_dry", stage, mode="smoke", dry_run=True)
    assert result["state"] == "done"
    assert result["result"]["returncode"] == 0


def test_build_smoke_minimal_skips_box(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMML_MCP_RUNS_ROOT", str(tmp_path / "runs"))
    configure_run("build_min", recipe="build_smoke", mode="minimal")
    result = run_recipe_stage("build_min", "box_build", mode="minimal")
    assert result["state"] == "skipped"
