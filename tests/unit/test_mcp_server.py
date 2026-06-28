"""Tests for the MMML MCP server (no MD execution)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmml.mcp.allowlist import is_allowed_mmml_command, validate_cli_args
from mmml.mcp.env import repo_root, runs_root
from mmml.mcp.manifest import RunManifest, load_manifest, save_manifest
from mmml.mcp.recipes import configure_run, list_recipe_names, load_recipe
from mmml.mcp.server import list_capabilities


def test_allowlist_rejects_shell_metachar() -> None:
    with pytest.raises(ValueError):
        validate_cli_args(["foo; rm -rf /"])
    assert is_allowed_mmml_command("md-system")
    assert not is_allowed_mmml_command("rm")


def test_list_recipes() -> None:
    names = list_recipe_names()
    assert "dimer_smoke" in names
    assert "build_smoke" in names
    recipe = load_recipe("dimer_smoke")
    assert recipe["name"] == "dimer_smoke"


def test_configure_run_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MMML_MCP_RUNS_ROOT", str(tmp_path / "runs"))
    result = configure_run("test_smoke_001", recipe="dimer_smoke", mode="smoke")
    assert result["run_id"] == "test_smoke_001"
    run_dir = Path(result["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "configs" / "md_smoke.yaml").is_file()
    manifest = load_manifest(run_dir)
    assert manifest.stage("qm").state == "skipped"
    assert manifest.stage("train").state == "skipped"


def test_manifest_roundtrip(tmp_path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    manifest = RunManifest(run_id="run_a", recipe="dimer_smoke")
    manifest.stage("md").state = "running"
    save_manifest(run_dir, manifest)
    loaded = load_manifest(run_dir)
    assert loaded.run_id == "run_a"
    assert loaded.stage("md").state == "running"


def test_list_capabilities_json() -> None:
    text = list_capabilities()
    data = json.loads(text)
    assert "dimer_smoke" in data["recipes"]
    assert "md-system" in data["allowed_mmml_commands"]
    assert Path(data["repo_root"]) == repo_root()
    assert "mcp_runs" in data["runs_root"]
