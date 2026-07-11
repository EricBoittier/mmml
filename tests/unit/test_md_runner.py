"""Unit tests for the GUI job runner (mmml.gui.api.runner).

These exercise the JobManager lifecycle without launching a real ``mmml``
process -- they whitelist the test interpreter so the tests stay fast and
hermetic. No pytest-asyncio dependency: each test drives its own event loop via
``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from mmml.gui.api.runner import Job, JobManager


def _make_manager(tmp_path: Path) -> JobManager:
    mgr = JobManager(default_cwd=tmp_path)
    # Allow launching the test interpreter directly (alongside the real ``mmml``).
    mgr.ALLOWED_COMMANDS = frozenset({Path(sys.executable).name, "mmml"})
    return mgr


async def _run_snippet(
    manager: JobManager,
    code: str,
    output_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> Job:
    argv = [sys.executable, "-u", "-c", code]
    job = await manager.create(
        argv=argv,
        output_dir=str(output_dir) if output_dir else None,
        env=env,
        autostart=True,
    )
    await job.wait()
    return job


def test_command_allowlist_rejects_unknown(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)

    async def go() -> None:
        with pytest.raises(ValueError, match="not allowed"):
            await manager.create(argv=["rm", "-rf", "/"], autostart=False)

    asyncio.run(go())


def test_log_capture_and_success(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    code = "import sys; print('hello'); print('world'); print('err', file=sys.stderr)"
    job = asyncio.run(_run_snippet(manager, code))

    assert job.status == "succeeded"
    assert job.exit_code == 0

    logs = job.logs()
    texts = [ln["text"] for ln in logs]
    assert "hello" in texts
    assert "world" in texts
    assert "err" in texts

    seqs = [ln["seq"] for ln in logs]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)

    err_lines = [ln for ln in logs if ln["stream"] == "stderr"]
    assert any(ln["text"] == "err" for ln in err_lines)


def test_logs_since_filters_by_seq(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    job = asyncio.run(_run_snippet(manager, "print('a'); print('b'); print('c')"))
    all_logs = job.logs()
    assert len(all_logs) >= 3
    mid = all_logs[0]["seq"]
    later = job.logs(since=mid)
    assert all(ln["seq"] > mid for ln in later)


def test_nonzero_exit_marks_failed(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    job = asyncio.run(_run_snippet(manager, "import sys; sys.exit(3)"))
    assert job.status == "failed"
    assert job.exit_code == 3


def test_file_watch_emits_output_files(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    out = tmp_path / "run"
    code = (
        "import pathlib,os;"
        "p=pathlib.Path(os.environ['OUT']);"
        "p.mkdir(parents=True, exist_ok=True);"
        "(p/'traj.npz').write_bytes(b'0'*128);"
        "(p/'summary.json').write_text('{}')"
    )
    job = asyncio.run(_run_snippet(manager, code, output_dir=out, env={"OUT": str(out)}))
    assert job.status == "succeeded"
    files = {f["relpath"] for f in job.files()}
    assert "traj.npz" in files
    assert "summary.json" in files
    npz = next(f for f in job.files() if f["relpath"] == "traj.npz")
    assert npz["size"] == 128


def test_stream_replays_history_for_late_subscriber(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)

    async def go() -> list:
        job = await _run_snippet(manager, "print('one'); print('two')")
        return [ev async for ev in job.stream(replay=True)]

    events = asyncio.run(go())
    kinds = [ev.kind for ev in events]
    assert "log" in kinds
    assert kinds[-1] == "status"
    log_texts = [ev.data["text"] for ev in events if ev.kind == "log"]
    assert "one" in log_texts and "two" in log_texts


def test_stop_terminates_long_job(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)

    async def go() -> Job:
        code = "import time\nwhile True:\n    print('tick', flush=True); time.sleep(0.1)"
        argv = [sys.executable, "-u", "-c", code]
        job = await manager.create(argv=argv, autostart=True)
        await asyncio.sleep(0.3)
        await job.stop()
        await job.wait()
        return job

    job = asyncio.run(go())
    assert job.status == "stopped"


def test_manager_list_and_get(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    job = asyncio.run(_run_snippet(manager, "print('x')"))
    assert manager.get(job.id) is job
    listing = manager.list()
    assert any(j["id"] == job.id for j in listing)
