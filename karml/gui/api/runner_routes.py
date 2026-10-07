"""FastAPI routes exposing the job runner (mmml.gui.api.runner).

Mounted only when the GUI server is started with ``--enable-runner``. Provides:

* ``POST /api/jobs``            - launch an ``mmml md-system`` (or ``mmml ...``) run
* ``GET  /api/jobs``           - list jobs (status snapshots)
* ``GET  /api/jobs/{id}``      - one job's status snapshot
* ``GET  /api/jobs/{id}/logs`` - captured log lines (poll fallback, ``?since=seq``)
* ``GET  /api/jobs/{id}/files``- output-dir file manifest
* ``GET  /api/jobs/{id}/events`` - Server-Sent Events stream (log + file + status)
* ``POST /api/jobs/{id}/stop`` - signal the process group

The event stream is what the local UI tails for live progress; frame data is
served by the existing ``/api/file`` / ``/api/frame`` endpoints pointed at the
job's output dir.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .runner import JobManager


class CreateJobRequest(BaseModel):
    """Request body for launching a job.

    Provide either a full ``argv`` list, or a ``command`` string that is
    shell-split. The first token must be an allowed command (``mmml``).
    """

    argv: Optional[List[str]] = Field(default=None, description="Full argument vector")
    command: Optional[str] = Field(default=None, description="Shell-style command string")
    output_dir: Optional[str] = Field(default=None, description="Override --output-dir to watch")
    cwd: Optional[str] = Field(default=None, description="Working directory for the run")
    label: Optional[str] = Field(default=None, description="Human-friendly job label")
    env: Optional[dict[str, str]] = Field(default=None, description="Extra environment variables")


def register_runner_routes(app: FastAPI, manager: JobManager) -> None:
    """Attach runner endpoints and lifecycle hooks to ``app``."""
    app.state.job_manager = manager

    @app.on_event("shutdown")
    async def _stop_jobs_on_shutdown() -> None:
        await manager.shutdown()

    @app.get("/api/runner/config")
    async def runner_config() -> dict[str, Any]:
        return {
            "enabled": True,
            "default_cwd": str(manager.default_cwd),
            "output_root": str(manager.output_root) if manager.output_root else None,
            "allowed_commands": sorted(manager.ALLOWED_COMMANDS),
            "max_jobs": manager.max_jobs,
        }

    @app.post("/api/jobs")
    async def create_job(req: CreateJobRequest) -> dict[str, Any]:
        if not req.argv and not req.command:
            raise HTTPException(status_code=400, detail="Provide 'argv' or 'command'")
        argv: List[str] | str = req.argv if req.argv else (req.command or "")
        try:
            job = await manager.create(
                argv=argv,
                output_dir=req.output_dir,
                cwd=req.cwd,
                env=req.env,
                label=req.label,
                autostart=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        return job.summary()

    @app.get("/api/jobs")
    async def list_jobs() -> List[dict[str, Any]]:
        return manager.list()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job.summary()

    @app.get("/api/jobs/{job_id}/logs")
    async def get_job_logs(job_id: str, since: int = 0, limit: Optional[int] = None) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return {"job_id": job_id, "status": job.status, "lines": job.logs(since=since, limit=limit)}

    @app.get("/api/jobs/{job_id}/files")
    async def get_job_files(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return {
            "job_id": job_id,
            "output_dir": str(job.output_dir) if job.output_dir else None,
            "files": job.files(),
        }

    @app.post("/api/jobs/{job_id}/stop")
    async def stop_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        await job.stop()
        return job.summary()

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: str, replay: bool = True) -> StreamingResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        async def event_source():
            # Prelude comment keeps some proxies from buffering the stream.
            yield ": stream open\n\n"
            try:
                async for event in job.stream(replay=replay):
                    payload = json.dumps(event.to_dict())
                    yield f"event: {event.kind}\ndata: {payload}\n\n"
            except asyncio.CancelledError:  # client disconnected
                raise
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
