"""Persistent JAX server for MMML ORCA external-tool calculations."""

from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mmml.interfaces.orca_external.runner import (
    MmmlOrcaExternalRunner,
    get_calculator,
    parse_runner_arguments,
)
from mmml.interfaces.orca_external.settings import (
    MmmlOrcaSettings,
    add_model_arguments,
    settings_from_namespace,
)

DEFAULT_BIND = "127.0.0.1:8888"


class CalculateRequest(BaseModel):
    arguments: list[str] = Field(min_length=1)
    directory: str


class MmmlOrcaServer:
    """Handle ORCA external-tool requests with a resident MMML calculator."""

    def __init__(self, default_settings: MmmlOrcaSettings | None = None) -> None:
        self.default_settings = default_settings
        self._lock = threading.Lock()

    def warmup(self) -> None:
        """Load the default checkpoint into the process cache."""
        if self.default_settings is None:
            return
        get_calculator(self.default_settings)

    def handle(self, arguments: list[str], directory: str) -> dict[str, Any]:
        working_dir = Path(directory).resolve()
        if not working_dir.is_dir():
            raise ValueError(f"Invalid directory: {working_dir}")

        inputfile, settings = parse_runner_arguments(
            arguments,
            default_settings=self.default_settings,
        )
        input_path = (working_dir / inputfile).resolve()

        buf = io.StringIO()
        with self._lock:
            with redirect_stdout(buf):
                MmmlOrcaExternalRunner(settings).run(input_path)

        return {"status": "Success", "stdout": buf.getvalue()}


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
    add_model_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_server_parser()
    args = parser.parse_args(argv)

    default_settings: MmmlOrcaSettings | None = None
    if args.checkpoint or os.environ.get("MMML_CHECKPOINT"):
        default_settings = settings_from_namespace(args)

    server = MmmlOrcaServer(default_settings=default_settings)
    if args.warmup:
        logging.info("Warming up MMML checkpoint...")
        server.warmup()

    host, port = args.host_port.split(":", 1)
    app = create_app(server)
    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    main()
