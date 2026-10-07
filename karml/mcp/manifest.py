"""Run manifest persistence for MCP-orchestrated pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

StageState = Literal["pending", "running", "done", "failed", "skipped"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StageRecord:
    name: str
    state: StageState = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    command: str | None = None
    log_path: str | None = None
    job_id: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    run_id: str
    recipe: str | None = None
    preset: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    cluster: str | None = None
    stages: dict[str, StageRecord] = field(default_factory=dict)
    notes: str | None = None

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "recipe": self.recipe,
            "preset": self.preset,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cluster": self.cluster,
            "notes": self.notes,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        stages_raw = data.get("stages") or {}
        stages = {
            name: StageRecord(**payload) for name, payload in stages_raw.items()
        }
        return cls(
            run_id=str(data["run_id"]),
            recipe=data.get("recipe"),
            preset=data.get("preset"),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            cluster=data.get("cluster"),
            stages=stages,
            notes=data.get("notes"),
        )


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def load_manifest(run_dir: Path) -> RunManifest:
    path = manifest_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest.from_dict(data)


def save_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    manifest.touch()
    path = manifest_path(run_dir)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
