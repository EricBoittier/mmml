"""Persistent JAX server for MMML ORCA external-tool calculations."""

from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mmml.interfaces.orca_external.runner import (
    OrcaPreparedJob,
    _cache_key,
    get_calculator,
    prepare_orca_job_from_arguments,
    run_prepared_jobs,
)
from mmml.interfaces.orca_external.settings import (
    MmmlOrcaSettings,
    add_model_arguments,
    settings_from_namespace,
)

DEFAULT_BIND = "127.0.0.1:8888"
DEFAULT_BATCH_SIZE = 16
DEFAULT_BATCH_WAIT_MS = 10


class CalculateRequest(BaseModel):
    arguments: list[str] = Field(min_length=1)
    directory: str


@dataclass
class _PendingRequest:
    prepared: OrcaPreparedJob
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class MmmlOrcaServer:
    """Handle ORCA external-tool requests with optional GPU micro-batching."""

    def __init__(
        self,
        default_settings: MmmlOrcaSettings | None = None,
        *,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
        batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS,
    ) -> None:
        self.default_settings = default_settings
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)
        self._queue: list[_PendingRequest] = []
        self._queue_cond = threading.Condition()
        self._shutdown = False
        self._worker = threading.Thread(target=self._batch_worker, name="mmml-orca-batch", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        """Stop the background batch worker."""
        with self._queue_cond:
            self._shutdown = True
            self._queue_cond.notify_all()
        self._worker.join(timeout=5.0)

    def warmup(self) -> None:
        """Load the default checkpoint into the process cache."""
        if self.default_settings is None:
            return
        get_calculator(self.default_settings)

    def handle(self, arguments: list[str], directory: str) -> dict[str, Any]:
        """Enqueue a request and block until the engrad file is written."""
        try:
            prepared = prepare_orca_job_from_arguments(
                arguments,
                directory,
                default_settings=self.default_settings,
            )
        except Exception as exc:
            return {
                "status": "Error",
                "error_message": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }

        pending = _PendingRequest(prepared=prepared)
        with self._queue_cond:
            self._queue.append(pending)
            self._queue_cond.notify()
        pending.done.wait()
        assert pending.result is not None
        return pending.result

    def _batch_worker(self) -> None:
        while True:
            try:
                batch = self._collect_batch()
            except Exception:
                logging.exception("ORCA batch worker failed while collecting requests")
                continue
            if batch is None:
                return
            if not batch:
                continue
            self._process_batch(batch)

    def _fail_pending(self, pending: _PendingRequest, exc: Exception) -> None:
        pending.result = {
            "status": "Error",
            "error_message": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        pending.done.set()

    def _collect_batch(self) -> list[_PendingRequest] | None:
        with self._queue_cond:
            while not self._shutdown and not self._queue:
                self._queue_cond.wait()

            if self._shutdown and not self._queue:
                return None

            deadline = time.monotonic() + self.batch_wait_s
            batch = [self._queue.pop(0)]

        try:
            settings_key = self._settings_key(batch[0])
        except Exception as exc:
            self._fail_pending(batch[0], exc)
            return []

        with self._queue_cond:
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                while not self._shutdown and not self._queue and remaining > 0:
                    self._queue_cond.wait(timeout=remaining)
                    remaining = deadline - time.monotonic()
                if not self._queue:
                    break
                try:
                    next_key = self._settings_key(self._queue[0])
                except Exception as exc:
                    orphan = self._queue.pop(0)
                    self._fail_pending(orphan, exc)
                    continue
                if next_key != settings_key:
                    break
                batch.append(self._queue.pop(0))

            return batch

    def _settings_key(self, pending: _PendingRequest) -> tuple[Any, ...]:
        return _cache_key(pending.prepared.settings)

    def _process_batch(self, batch: list[_PendingRequest]) -> None:
        prepared_jobs = [pending.prepared for pending in batch]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_prepared_jobs(prepared_jobs)

            stdout = buf.getvalue()
            for pending in batch:
                pending.result = {"status": "Success", "stdout": stdout}
                pending.done.set()
        except Exception as exc:
            error_payload = {
                "status": "Error",
                "error_message": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            for pending in batch:
                pending.result = dict(error_payload)
                pending.done.set()


def create_app(server: MmmlOrcaServer) -> FastAPI:
    app = FastAPI(title="mmml-orca-server")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "OK"}

    @app.post("/calculate")
    def calculate(payload: CalculateRequest) -> JSONResponse:
        try:
            result = server.handle(payload.arguments, payload.directory)
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse(
                {
                    "status": "Error",
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            )

    return app


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml-orca-server",
        description=(
            "Start a persistent MMML server for ORCA external-tool calculations. "
            "Point ORCA ProgExt at mmml-orca-client."
        ),
    )
    parser.add_argument(
        "-b",
        "--bind",
        metavar="hostname:port",
        default=DEFAULT_BIND,
        dest="host_port",
        help=f"Server bind address and port. Default: {DEFAULT_BIND}.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Load the default checkpoint at startup (recommended).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Maximum number of ORCA requests to evaluate in one GPU batch. "
            f"Default: {DEFAULT_BATCH_SIZE}. Use 1 to disable batching."
        ),
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=DEFAULT_BATCH_WAIT_MS,
        help=(
            "Milliseconds to wait for additional requests before launching a batch. "
            f"Default: {DEFAULT_BATCH_WAIT_MS}."
        ),
    )
    add_model_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_server_parser()
    args = parser.parse_args(argv)

    default_settings: MmmlOrcaSettings | None = None
    if args.checkpoint or os.environ.get("MMML_CHECKPOINT"):
        default_settings = settings_from_namespace(args)

    server = MmmlOrcaServer(
        default_settings=default_settings,
        max_batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    if args.warmup:
        logging.info("Warming up MMML checkpoint...")
        server.warmup()

    host, port = args.host_port.split(":", 1)
    app = create_app(server)
    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    main()
