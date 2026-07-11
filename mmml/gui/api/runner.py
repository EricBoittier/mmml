"""Job runner for launching and streaming ``mmml md-system`` runs.

This module lets the GUI backend act as a *runner*: it launches an
``mmml md-system`` (or arbitrary ``mmml``) invocation as a subprocess on the
host it runs on, captures stdout/stderr line-by-line, watches the job's
``--output-dir`` for new/changed files, and broadcasts three kinds of events to
subscribers so a local UI can stream progress live:

* ``log``    - one captured stdout/stderr line (with a monotonic ``seq``)
* ``file``   - an output file appeared or changed (path, size, mtime)
* ``status`` - the job changed state (running/succeeded/failed/stopped)

The design is transport-agnostic. On an HPC/remote host you run
``mmml gui --enable-runner`` and reach it from your laptop over an SSH
port-forward; the same code also works locally. Nothing here opens a network
port or trusts remote input by itself -- the FastAPI layer owns that.

Everything is stdlib + asyncio so it is cheap to unit-test without a running
event loop touching the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional

__all__ = [
    "JobEvent",
    "LogLine",
    "FileEntry",
    "Job",
    "JobManager",
]

# Terminal states -- a job in one of these will never emit further events.
_TERMINAL = frozenset({"succeeded", "failed", "stopped"})

# Cap on retained log lines per job so a chatty run cannot exhaust memory.
_DEFAULT_LOG_LIMIT = 50_000


@dataclass(frozen=True)
class JobEvent:
    """A single streamable event for a job (log/file/status)."""

    kind: str  # "log" | "file" | "status"
    seq: int
    time: float
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "seq": self.seq, "time": self.time, **self.data}


@dataclass
class LogLine:
    seq: int
    stream: str  # "stdout" | "stderr"
    text: str
    time: float

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "stream": self.stream, "text": self.text, "time": self.time}


@dataclass
class FileEntry:
    """Snapshot of an output file the runner is watching."""

    relpath: str
    size: int
    mtime: float

    def to_dict(self) -> Dict[str, Any]:
        return {"relpath": self.relpath, "size": self.size, "mtime": self.mtime}


class Job:
    """A single md-system subprocess plus its captured output and watchers.

    Lifecycle: ``pending`` -> ``running`` -> ``succeeded`` | ``failed`` |
    ``stopped``. Only the manager mutates a job; subscribers read snapshots.
    """

    def __init__(
        self,
        job_id: str,
        argv: List[str],
        output_dir: Optional[Path],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        label: Optional[str] = None,
        log_limit: int = _DEFAULT_LOG_LIMIT,
        watch_interval: float = 1.0,
    ) -> None:
        self.id = job_id
        self.argv = list(argv)
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.cwd = Path(cwd).resolve() if cwd else None
        self.env = env
        self.label = label or job_id
        self.watch_interval = float(watch_interval)

        self.status = "pending"
        self.exit_code: Optional[int] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self._log: Deque[LogLine] = deque(maxlen=log_limit)
        self._files: Dict[str, FileEntry] = {}
        self._seq = 0  # monotonic across ALL events (log/file/status) for ordering
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._tasks: List[asyncio.Task] = []
        self._pump_tasks: List[asyncio.Task] = []
        self._subscribers: List["asyncio.Queue[Optional[JobEvent]]"] = []
        self._done = asyncio.Event()

    # -- snapshots -----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Lightweight status snapshot (safe to poll frequently)."""
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "argv": self.argv,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_log_lines": len(self._log),
            "n_files": len(self._files),
            "last_seq": self._seq,
        }

    def logs(self, since: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return captured log lines with ``seq > since`` (for polling fallback)."""
        out = [ln.to_dict() for ln in self._log if ln.seq > since]
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        return out

    def files(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in sorted(self._files.values(), key=lambda e: e.relpath)]

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    # -- event plumbing ------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, kind: str, data: Dict[str, Any]) -> None:
        event = JobEvent(kind=kind, seq=self._next_seq(), time=time.time(), data=data)
        # Fan out to live subscribers; drop into unbounded per-subscriber queues.
        for q in list(self._subscribers):
            q.put_nowait(event)

    def subscribe(self) -> "asyncio.Queue[Optional[JobEvent]]":
        q: "asyncio.Queue[Optional[JobEvent]]" = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Optional[JobEvent]]") -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(q)

    def _close_subscribers(self) -> None:
        for q in list(self._subscribers):
            q.put_nowait(None)  # sentinel: stream complete

    # -- run ----------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the subprocess and its reader/watcher tasks."""
        if self.status != "pending":
            raise RuntimeError(f"job {self.id} already started (status={self.status})")

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        proc_env = dict(os.environ)
        if self.env:
            proc_env.update(self.env)
        # Keep child stdout unbuffered so lines stream promptly.
        proc_env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=str(self.cwd) if self.cwd else None,
                env=proc_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # New process group so we can signal the whole tree on stop().
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as exc:
            self.status = "failed"
            self.error = f"failed to launch: {exc}"
            self.finished_at = time.time()
            self._emit("status", {"status": self.status, "error": self.error})
            self._close_subscribers()
            self._done.set()
            return

        self.status = "running"
        self.started_at = time.time()
        self._emit("status", {"status": self.status, "pid": self._proc.pid})

        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._pump_tasks = [
            asyncio.create_task(self._pump(self._proc.stdout, "stdout")),
            asyncio.create_task(self._pump(self._proc.stderr, "stderr")),
        ]
        self._tasks = list(self._pump_tasks)
        self._tasks.append(asyncio.create_task(self._wait_and_finalize()))
        if self.output_dir is not None:
            self._tasks.append(asyncio.create_task(self._watch_files()))

    async def _pump(self, stream: asyncio.StreamReader, name: str) -> None:
        """Read a stream line-by-line, retaining and broadcasting each line."""
        while True:
            raw = await stream.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").rstrip("\n")
            line = LogLine(seq=0, stream=name, text=text, time=time.time())
            self._emit("log", {"stream": name, "text": text})
            line.seq = self._seq  # tie the retained line to the emitted event seq
            self._log.append(line)

    async def _watch_files(self) -> None:
        """Poll the output dir and emit ``file`` events on new/changed files."""
        assert self.output_dir is not None
        while not self._done.is_set():
            self._scan_files_once()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._done.wait(), timeout=self.watch_interval)
        # Final scan so the last-written frames/summary are captured.
        self._scan_files_once()

    def _scan_files_once(self) -> None:
        assert self.output_dir is not None
        root = self.output_dir
        if not root.exists():
            return
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            rel = str(path.relative_to(root))
            prev = self._files.get(rel)
            if prev is None or prev.size != st.st_size or prev.mtime != st.st_mtime:
                entry = FileEntry(relpath=rel, size=st.st_size, mtime=st.st_mtime)
                self._files[rel] = entry
                self._emit("file", entry.to_dict())

    async def _wait_and_finalize(self) -> None:
        assert self._proc is not None
        code = await self._proc.wait()
        # Let stdout/stderr pumps drain any buffered tail before finalizing.
        for t in self._pump_tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(t, timeout=5.0)

        self.exit_code = code
        self.finished_at = time.time()
        if self.status == "stopped":
            pass  # explicit stop already recorded
        elif code == 0:
            self.status = "succeeded"
        else:
            self.status = "failed"
        self._done.set()
        # Stop the watcher and do a last scan (only if we have an output dir).
        if self.output_dir is not None:
            self._scan_files_once()
        self._emit("status", {"status": self.status, "exit_code": code})
        self._close_subscribers()

    async def stop(self, sig: int = signal.SIGTERM, grace: float = 10.0) -> None:
        """Signal the process group, escalating to SIGKILL after ``grace``."""
        if self._proc is None or self.status in _TERMINAL:
            return
        self.status = "stopped"
        self._emit("status", {"status": self.status})
        pgid = None
        with contextlib.suppress(ProcessLookupError, OSError):
            pgid = os.getpgid(self._proc.pid)
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                self._proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._proc.wait(), timeout=grace)
        if self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self._proc.kill()

    async def wait(self) -> None:
        await self._done.wait()

    async def stream(self, replay: bool = True) -> AsyncIterator[JobEvent]:
        """Yield events for this job.

        When ``replay`` is True, first replays the retained log + file snapshot
        (as synthetic events) so a late subscriber sees full history, then live
        events until the job terminates and the stream closes.
        """
        q = self.subscribe()
        try:
            if replay:
                for ln in list(self._log):
                    yield JobEvent("log", ln.seq, ln.time, {"stream": ln.stream, "text": ln.text})
                for entry in sorted(self._files.values(), key=lambda e: e.relpath):
                    yield JobEvent("file", self._seq, time.time(), entry.to_dict())
                yield JobEvent(
                    "status", self._seq, time.time(),
                    {"status": self.status, "exit_code": self.exit_code},
                )
                if self.is_terminal:
                    return
            while True:
                event = await q.get()
                if event is None:  # sentinel
                    return
                yield event
        finally:
            self.unsubscribe(q)


class JobManager:
    """Owns the set of jobs launched by this server process."""

    #: allowlist of first-arg commands the runner may spawn
    ALLOWED_COMMANDS = frozenset({"mmml"})

    def __init__(
        self,
        default_cwd: Optional[Path] = None,
        output_root: Optional[Path] = None,
        max_jobs: int = 64,
    ) -> None:
        self.default_cwd = Path(default_cwd).resolve() if default_cwd else Path.cwd()
        self.output_root = Path(output_root).resolve() if output_root else None
        self.max_jobs = max_jobs
        self._jobs: "Dict[str, Job]" = {}

    def _validate_argv(self, argv: List[str]) -> None:
        if not argv:
            raise ValueError("empty argv")
        exe = Path(argv[0]).name
        if exe not in self.ALLOWED_COMMANDS:
            raise ValueError(
                f"command {exe!r} not allowed; runner may only launch: "
                f"{', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )

    def _resolve_output_dir(self, argv: List[str], explicit: Optional[str]) -> Optional[Path]:
        """Pull ``--output-dir`` from argv unless one is passed explicitly."""
        if explicit:
            return Path(explicit)
        for i, tok in enumerate(argv):
            if tok == "--output-dir" and i + 1 < len(argv):
                return Path(argv[i + 1])
            if tok.startswith("--output-dir="):
                return Path(tok.split("=", 1)[1])
        return None

    async def create(
        self,
        argv: List[str] | str,
        output_dir: Optional[str] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        label: Optional[str] = None,
        autostart: bool = True,
    ) -> Job:
        if isinstance(argv, str):
            argv = shlex.split(argv)
        self._validate_argv(argv)
        if len([j for j in self._jobs.values() if not j.is_terminal]) >= self.max_jobs:
            raise RuntimeError(f"too many active jobs (max {self.max_jobs})")

        out_dir = self._resolve_output_dir(argv, output_dir)
        job_cwd = Path(cwd) if cwd else self.default_cwd
        if out_dir is not None and not out_dir.is_absolute():
            out_dir = (job_cwd / out_dir).resolve()

        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            argv=argv,
            output_dir=out_dir,
            cwd=job_cwd,
            env=env,
            label=label,
        )
        self._jobs[job_id] = job
        if autostart:
            await job.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        return [j.summary() for j in sorted(self._jobs.values(), key=lambda j: j.created_at)]

    async def stop(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        await job.stop()
        return True

    async def shutdown(self) -> None:
        """Stop every still-running job (called on server shutdown)."""
        await asyncio.gather(
            *(j.stop() for j in self._jobs.values() if not j.is_terminal),
            return_exceptions=True,
        )
