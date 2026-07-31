"""Slurm resource wiring for workflows/hybrid_umbrella_windows.

snakemake-executor-plugin-slurm owns ``--gres`` / ``--gpus``: it derives them
from the job resources and aborts submission if it finds either flag in
``slurm_extra``. These tests pin the split so a GPU request cannot drift back
into the extra-flags string.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "hybrid_umbrella_windows"


def _load() -> ModuleType:
    name = "hybrid_umbrella_windows_campaign_lib"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, WORKFLOW / "scripts" / "campaign_lib.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load()


def test_slurm_extra_never_carries_a_gpu_request() -> None:
    cfg = {"slurm": {"nodelist": "gpu08,gpu09", "mail_user": "me@example.org"}}
    extra = cl.slurm_extra_string(cfg)
    assert "--gres" not in extra
    assert "--gpus" not in extra
    assert "--nodelist=gpu08,gpu09" in extra
    assert "--mail-user=me@example.org" in extra


def test_slurm_extra_is_empty_without_optional_settings() -> None:
    assert cl.slurm_extra_string({}) == ""
    assert cl.slurm_extra_string({"slurm": {"nodelist": "", "mail_user": ""}}) == ""


def test_gpu_request_defaults_to_gres() -> None:
    assert cl.gpu_request_resources({}) == {"gres": "gpu:1"}
    assert cl.gpu_request_resources({"slurm": {"gres": "gpu:tesla:1"}}) == {
        "gres": "gpu:tesla:1"
    }


def test_empty_gres_falls_back_to_the_plugin_gpu_resource() -> None:
    """Escape hatch for sites where --gres is rejected but --gpus works."""
    assert cl.gpu_request_resources({"slurm": {"gres": ""}}) == {"gpu": 1}


def test_gpu_request_never_yields_both_forms() -> None:
    """gres + gpu together would emit `--gres=gpu:1 --gpus=1` for one device."""
    for cfg in ({}, {"slurm": {"gres": "gpu:1"}}, {"slurm": {"gres": ""}}):
        assert len(set(cl.gpu_request_resources(cfg)) & {"gres", "gpu"}) == 1


def test_throttle_resource_is_not_named_gpu() -> None:
    """`--resources gpu=N` would be consumed by the plugin as --gpus=N."""
    cli = cl.slurm_resources_cli({"slurm": {"max_jobs": 8}})
    assert "gpu_slot=8" in cli
    assert "charmm_slot=8" in cli
    assert " gpu=" not in f" {cli}"


def test_shipped_configs_and_profiles_avoid_the_reserved_gpu_resource() -> None:
    for profile in ("slurm", "local"):
        text = (WORKFLOW / "profiles" / profile / "config.yaml").read_text()
        assert "- gpu=" not in text, f"profiles/{profile} sets the reserved `gpu` resource"
    snakefile = (WORKFLOW / "Snakefile").read_text()
    assert "--gres" not in snakefile
