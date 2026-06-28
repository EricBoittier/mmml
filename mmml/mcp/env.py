"""Resolve repo paths and MMML executables for the MCP server."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    env = os.environ.get("MMML_REPO_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    custom = os.environ.get("MMML_MCP_RUNS_ROOT", "").strip()
    if custom:
        return Path(custom).resolve()
    return repo_root() / "artifacts" / "mcp_runs"


def recipes_dir() -> Path:
    return Path(__file__).resolve().parent / "recipes"


def resolve_mmml_bin() -> Path:
    env = os.environ.get("MMML_BIN", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path.resolve()
    venv_mmml = repo_root() / ".venv" / "bin" / "mmml"
    if venv_mmml.is_file():
        return venv_mmml.resolve()
    found = shutil.which("mmml")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(
        "mmml executable not found; set MMML_BIN or create .venv (uv sync)"
    )


def resolve_python() -> Path:
    env = os.environ.get("MMML_PYTHON", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path.resolve()
    venv_py = repo_root() / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return venv_py.resolve()
    found = shutil.which("python3") or shutil.which("python")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError("Python interpreter not found")


def default_checkpoint() -> Path:
    env = os.environ.get("MMML_CKPT", "").strip()
    if env:
        return Path(env).resolve()
    bundled = repo_root() / "examples" / "ckpts_json" / "DESdimers_params.json"
    if bundled.is_file():
        return bundled.resolve()
    raise FileNotFoundError(
        "No checkpoint found; set MMML_CKPT or add examples/ckpts_json/DESdimers_params.json"
    )


def ensure_run_dir(run_id: str) -> Path:
    safe = run_id.strip().replace("/", "_")
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid run_id: {run_id!r}")
    path = runs_root() / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_under_runs(path: Path) -> Path:
    resolved = path.resolve()
    root = runs_root().resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    raise ValueError(f"path must be under {root}: {path}")
