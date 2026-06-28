"""Aggregate run and cluster status for MCP tools."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from mmml.mcp.env import ensure_run_dir, runs_root, repo_root
from mmml.mcp.manifest import RunManifest, load_manifest


def list_runs() -> list[dict[str, Any]]:
    root = runs_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        manifest_path = child / "manifest.json"
        if manifest_path.is_file():
            manifest = load_manifest(child)
            out.append(
                {
                    "run_id": manifest.run_id,
                    "recipe": manifest.recipe,
                    "updated_at": manifest.updated_at,
                    "stages": {
                        k: v.state for k, v in manifest.stages.items()
                    },
                }
            )
        else:
            out.append({"run_id": child.name, "manifest": None})
    return out


def get_run_status(run_id: str) -> dict[str, Any]:
    run_dir = ensure_run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "error": "manifest not found — call configure_run first",
        }
    manifest = load_manifest(run_dir)
    stage_summary = {
        name: {
            "state": rec.state,
            "log_path": rec.log_path,
            "job_id": rec.job_id,
            "error": rec.error,
        }
        for name, rec in manifest.stages.items()
    }
    artifacts = _scan_artifacts(run_dir)
    slurm = _slurm_snapshot()
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest.to_dict(),
        "stage_summary": stage_summary,
        "artifacts": artifacts,
        "slurm_queue": slurm,
    }


def _scan_artifacts(run_dir: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for rel in (
        "manifest.json",
        "configs/md_smoke.yaml",
        "md/results/jaxmd_smoke",
        "spectra",
        "configs/qm_pipeline",
    ):
        path = run_dir / rel
        if path.exists():
            if path.is_dir():
                files = [str(p.relative_to(run_dir)) for p in path.rglob("*") if p.is_file()]
                found[rel] = {"type": "dir", "files": files[:50]}
            else:
                found[rel] = {
                    "type": "file",
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
    done_markers = list(run_dir.rglob("done.txt"))
    if done_markers:
        found["done.txt"] = [str(p.relative_to(run_dir)) for p in done_markers]
    return found


def _slurm_snapshot() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["squeue", "-u", _username(), "-h", "-o", "%i %j %T %M %R"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        rows.append(
            {
                "jobid": parts[0],
                "name": parts[1],
                "state": parts[2],
                "time": parts[3],
                "reason": parts[4] if len(parts) > 4 else "",
            }
        )
    return rows


def _username() -> str:
    import getpass

    return getpass.getuser()


def tail_log(path: str, *, lines: int = 40) -> dict[str, Any]:
    log = Path(path)
    if not log.is_file():
        return {"path": path, "error": "not found"}
    try:
        log.resolve().relative_to(runs_root().resolve())
    except ValueError:
        try:
            log.resolve().relative_to(repo_root().resolve())
        except ValueError as exc:
            raise ValueError("log path must be under repo artifacts or mmml root") from exc
    text = log.read_text(encoding="utf-8", errors="replace")
    chunk = "\n".join(text.splitlines()[-lines:])
    return {"path": str(log), "lines": lines, "tail": chunk}
