"""Subprocess helpers for MCP tools."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from mmml.mcp.allowlist import (
    is_allowed_console_script,
    is_allowed_mmml_command,
    validate_cli_args,
)
from mmml.mcp.env import (
    assert_under_runs,
    resolve_console_script,
    resolve_mmml_bin,
    repo_root,
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    log_path: str | None = None
    pid: int | None = None
    background: bool = False

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "log_path": self.log_path,
            "pid": self.pid,
            "background": self.background,
        }


def _logs_dir(run_dir: Path | None) -> Path:
    if run_dir is None:
        base = repo_root() / "artifacts" / "mcp_runs" / "_scratch"
    else:
        base = run_dir
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def run_mmml(
    command: str,
    args: list[str] | None = None,
    *,
    run_dir: Path | None = None,
    dry_run: bool = False,
    background: bool = False,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> CommandResult:
    if not is_allowed_mmml_command(command):
        raise ValueError(f"command not allowlisted: {command}")
    argv = validate_cli_args(list(args or []))
    mmml_bin = str(resolve_mmml_bin())
    full_cmd = [mmml_bin, command, *argv]
    cwd = str(run_dir.resolve()) if run_dir else str(repo_root())
    if run_dir is not None:
        assert_under_runs(run_dir)

    if dry_run:
        return CommandResult(
            command=full_cmd,
            cwd=cwd,
            returncode=0,
            stdout="(dry run — command not executed)",
            stderr="",
            background=False,
        )

    proc_env = os.environ.copy()
    proc_env.setdefault("JAX_ENABLE_X64", "1")
    if env:
        proc_env.update(env)

    log_path = _logs_dir(run_dir) / f"{command}-{uuid.uuid4().hex[:8]}.log"

    if background:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                full_cmd,
                cwd=cwd,
                env=proc_env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return CommandResult(
            command=full_cmd,
            cwd=cwd,
            returncode=0,
            stdout="",
            stderr="",
            log_path=str(log_path),
            pid=proc.pid,
            background=True,
        )

    completed = subprocess.run(
        full_cmd,
        cwd=cwd,
        env=proc_env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    log_path.write_text(
        "".join(
            [
                "$ " + " ".join(full_cmd) + "\n",
                f"cwd: {cwd}\n",
                f"exit: {completed.returncode}\n\n",
                "--- stdout ---\n",
                completed.stdout or "",
                "\n--- stderr ---\n",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return CommandResult(
        command=full_cmd,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        log_path=str(log_path),
        background=False,
    )


def run_console_script(
    script: str,
    args: list[str] | None = None,
    *,
    run_dir: Path | None = None,
    dry_run: bool = False,
    background: bool = False,
    timeout_s: int | None = None,
) -> CommandResult:
    if not is_allowed_console_script(script):
        raise ValueError(f"console script not allowlisted: {script}")
    argv = validate_cli_args(list(args or []))
    script_bin = str(resolve_console_script(script))
    full_cmd = [script_bin, *argv]
    cwd = str(run_dir.resolve()) if run_dir else str(repo_root())
    if run_dir is not None:
        assert_under_runs(run_dir)

    if dry_run:
        return CommandResult(
            command=full_cmd,
            cwd=cwd,
            returncode=0,
            stdout="(dry run — command not executed)",
            stderr="",
        )

    log_path = _logs_dir(run_dir) / f"{script}-{uuid.uuid4().hex[:8]}.log"
    if background:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                full_cmd,
                cwd=cwd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return CommandResult(
            command=full_cmd,
            cwd=cwd,
            returncode=0,
            stdout="",
            stderr="",
            log_path=str(log_path),
            pid=proc.pid,
            background=True,
        )

    completed = subprocess.run(
        full_cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    log_path.write_text(
        "".join(
            [
                "$ " + " ".join(full_cmd) + "\n",
                f"exit: {completed.returncode}\n\n",
                completed.stdout or "",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return CommandResult(
        command=full_cmd,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        log_path=str(log_path),
    )

