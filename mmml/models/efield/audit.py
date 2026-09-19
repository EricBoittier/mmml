"""Audit a running or finished efield-train job from logs + checkpoint dir.

A missing ``params-best.json`` during a run is **not** evidence that
validation never improved: older trainers only created that symlink on
process exit. Read ``best-valid-*.json``, ``history.jsonl``, and epoch
lines in the Slurm log instead.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mmml.models.efield.checkpointing import (
    HISTORY_NAME,
    RUN_META_NAME,
    is_orbax_checkpoint,
    read_history,
)

EPOCH_RE = re.compile(r"^epoch:\s+(\d+)\b")
POLAR_MAE_RE = re.compile(r"polar mae \[Bohr[³3]\]\s+(\S+)\s+(\S+)")
WEIGHTED_RE = re.compile(r"weighted total loss\s+(\S+)\s+(\S+)")
BEST_CKPT_RE = re.compile(r"Best valid checkpoint:.*epoch\s+(\d+)", re.I)
READY_RE = re.compile(
    r"ready:\s+n_train=(\d+)\s+n_valid=(\d+)\s+B=(\d+).*polar_weight=(\S+)"
)
CRASH_RE = re.compile(
    r"JaxRuntimeError|Autotuning failed|Traceback \(most recent call last\)|"
    r"ValueError: --batch_size|CUDA_ERROR|OOM|RESOURCE_EXHAUSTED",
    re.I,
)


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def parse_slurm_log(text: str) -> dict[str, Any]:
    """Pull epoch metrics and crash/compile markers from a trainer log."""
    epochs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    ready: dict[str, Any] | None = None
    crashes: list[str] = []
    last_nonempty = ""
    compiling = False
    step_compiled = False
    saw_valid_batch = False

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip():
            last_nonempty = line.strip()
        m_ready = READY_RE.search(line)
        if m_ready:
            ready = {
                "n_train": int(m_ready.group(1)),
                "n_valid": int(m_ready.group(2)),
                "batch_size": int(m_ready.group(3)),
                "polar_weight": m_ready.group(4),
            }
        if "compiling train_step" in line:
            compiling = True
        if "step 1 compiled" in line or "compiled train_step" in line:
            step_compiled = True
        if "Validation Batch" in line:
            saw_valid_batch = True
        if CRASH_RE.search(line):
            crashes.append(line.strip())
        m_epoch = EPOCH_RE.match(line.strip())
        if m_epoch:
            if current:
                epochs.append(current)
            current = {"epoch": int(m_epoch.group(1)), "source": "log"}
            continue
        if current is None:
            continue
        m_w = WEIGHTED_RE.search(line)
        if m_w:
            current["train_loss"] = _f(m_w.group(1))
            current["valid_loss"] = _f(m_w.group(2))
        m_p = POLAR_MAE_RE.search(line)
        if m_p:
            current["train_polar_mae_bohr3"] = _f(m_p.group(1))
            current["valid_polar_mae_bohr3"] = _f(m_p.group(2))
        m_best = BEST_CKPT_RE.search(line)
        if m_best:
            current["improved"] = True
            current["best_epoch"] = int(m_best.group(1))
    if current:
        epochs.append(current)

    return {
        "epochs": epochs,
        "ready": ready,
        "crashes": crashes,
        "last_line": last_nonempty,
        "compiling": compiling and not epochs,
        "step_compiled": step_compiled,
        "saw_valid_batch": saw_valid_batch,
    }


def load_best_valid(ckpt_dir: Path) -> dict[str, Any] | None:
    ckpt_dir = Path(ckpt_dir)
    candidates = []
    link = ckpt_dir / "best-valid.json"
    if link.is_file() or link.is_symlink():
        candidates.append(link)
    candidates.extend(sorted(ckpt_dir.glob("best-valid-*.json")))
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def load_run_meta(ckpt_dir: Path) -> dict[str, Any] | None:
    path = Path(ckpt_dir) / RUN_META_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_checkpoint_files(ckpt_dir: Path) -> list[dict[str, Any]]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    interesting = (
        "params-best",
        "best-valid",
        "history.jsonl",
        "run_meta.json",
        "config-",
        "params-epoch",
        "params-",
    )
    for path in sorted(ckpt_dir.iterdir(), key=lambda p: p.name):
        if path.name == "orbax" and path.is_dir():
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                rows.append(
                    {
                        "path": str(child),
                        "name": f"orbax/{child.name}",
                        "is_dir": child.is_dir(),
                        "mtime": child.stat().st_mtime,
                        "orbax": is_orbax_checkpoint(child),
                    }
                )
            continue
        if not any(path.name.startswith(p) or path.name == p for p in interesting):
            continue
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "is_dir": path.is_dir(),
                "is_symlink": path.is_symlink(),
                "target": str(path.readlink()) if path.is_symlink() else None,
                "mtime": path.stat().st_mtime if path.exists() else None,
                "size": path.stat().st_size if path.is_file() else None,
            }
        )
    return rows


def _query_sacct(job_id: str) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "-X",
                "-n",
                "-P",
                "--format=JobID,State,Elapsed,Start,End,ExitCode,NodeList",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return {"job_id": job_id, "raw": proc.stdout, "error": proc.stderr.strip() or None}
    parts = line[0].split("|")
    keys = ["job_id", "state", "elapsed", "start", "end", "exit_code", "node"]
    data = dict(zip(keys, parts, strict=False))
    data["raw"] = line[0]
    return data


def _merge_epochs(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            epoch = row.get("epoch")
            if epoch is None:
                continue
            key = int(epoch)
            by_epoch.setdefault(key, {})
            by_epoch[key].update(row)
            by_epoch[key]["epoch"] = key
    return [by_epoch[k] for k in sorted(by_epoch)]


def _polar_delta(epochs: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    polars = [e.get("valid_polar_mae_bohr3") for e in epochs if e.get("valid_polar_mae_bohr3") is not None]
    if not polars:
        return None, None, None
    first = _f(polars[0])
    last = _f(polars[-1])
    if first is None or last is None:
        return first, last, None
    return first, last, last - first


def verdict_from(
    *,
    epochs: list[dict[str, Any]],
    best: dict[str, Any] | None,
    log: dict[str, Any],
    job: dict[str, Any] | None,
    files: list[dict[str, Any]],
) -> tuple[str, str]:
    state = (job or {}).get("state") or ""
    if log.get("crashes") or state.upper() in {"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"}:
        return "crashed", "log or sacct reports a failure (not a polar plateau)"
    disk_best_epoch = None
    if best and best.get("best_epoch") is not None:
        disk_best_epoch = int(best["best_epoch"])
    has_weight_file = any(
        str(f.get("name", "")).startswith("params-best-") and not str(f.get("name")).endswith(".jsonl")
        for f in files
    )
    running = state.upper() in {"RUNNING", "COMPLETING"}

    if not epochs:
        # Disk beats an unflushed Slurm log. Epoch-1 ckpt + Validation Batch[0]
        # as the last line is the 22826285 case, not a first-compile hang.
        if disk_best_epoch is not None or has_weight_file:
            loss = (best or {}).get("best_valid_loss")
            if running:
                return (
                    "in_progress",
                    "slurm log has no epoch lines (stdout often unflushed) but "
                    f"disk has best_epoch={disk_best_epoch} best_valid_loss={loss}; "
                    "later epochs only rewrite params-best-* if valid loss improves",
                )
            return (
                "first_epoch",
                f"log never flushed epoch metrics; disk best_epoch={disk_best_epoch} "
                f"best_valid_loss={loss}",
            )
        if log.get("compiling") or (log.get("saw_valid_batch") and not log.get("step_compiled")):
            return "compiling", "first train_step / polar JVP is still compiling; no epoch yet"
        if log.get("saw_valid_batch") or log.get("step_compiled"):
            return "first_epoch", "epoch 1 is in progress or not flushed; missing params-best.json is normal"
        return "unknown", "no epoch lines, history.jsonl, or best-valid metrics yet"

    last = epochs[-1]
    last_epoch = int(last["epoch"])
    best_epoch = None
    if best and best.get("best_epoch") is not None:
        best_epoch = int(best["best_epoch"])
    elif last.get("best_epoch") is not None:
        best_epoch = int(last["best_epoch"])
    else:
        improved_epochs = [int(e["epoch"]) for e in epochs if e.get("improved")]
        if improved_epochs:
            best_epoch = improved_epochs[-1]
        else:
            best_epoch = last_epoch

    first_polar, last_polar, polar_delta = _polar_delta(epochs)
    if last.get("improved") or best_epoch == last_epoch:
        extra = ""
        if polar_delta is not None and abs(polar_delta) < 1e-3 and last_epoch > 1:
            extra = (
                f"; weighted valid loss improved but polar mae is flat "
                f"({first_polar} → {last_polar} Bohr³)"
            )
        return "improving", f"best_epoch={best_epoch} is the latest finished epoch{extra}"

    if last_epoch > (best_epoch or 0):
        note = f"best_epoch={best_epoch}, last_epoch={last_epoch}"
        if polar_delta is not None and abs(polar_delta) < 1e-3:
            note += f"; valid polar mae stalled ({first_polar} → {last_polar} Bohr³)"
        return "plateau", note

    has_symlink = any(f.get("name") == "params-best.json" for f in files)
    if not has_symlink and last_epoch == 1:
        return (
            "first_epoch",
            "only epoch 1 is on disk; params-best.json is created on improve "
            "(this branch) or at process exit (older trainers)",
        )
    return "unknown", f"last_epoch={last_epoch} best_epoch={best_epoch}"


@dataclass
class AuditReport:
    verdict: str
    reason: str
    last_epoch: int | None = None
    best_epoch: int | None = None
    best_valid_loss: float | None = None
    first_valid_polar_mae: float | None = None
    last_valid_polar_mae: float | None = None
    polar_delta: float | None = None
    n_epochs_logged: int = 0
    params_best_symlink: bool = False
    history_path: str | None = None
    job: dict[str, Any] | None = None
    ready: dict[str, Any] | None = None
    last_log_line: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    crashes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_job(
    *,
    ckpt_dir: Path | str | None = None,
    log_text: str | None = None,
    log_path: Path | str | None = None,
    job_id: str | None = None,
    job: dict[str, Any] | None = None,
) -> AuditReport:
    ckpt = Path(ckpt_dir) if ckpt_dir else None
    if log_text is None and log_path is not None:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    log = parse_slurm_log(log_text or "")
    history = read_history(ckpt) if ckpt else []
    best = load_best_valid(ckpt) if ckpt else None
    files = list_checkpoint_files(ckpt) if ckpt else []
    if job is None and job_id:
        job = _query_sacct(job_id)

    epochs = _merge_epochs(log.get("epochs") or [], history)
    first_polar, last_polar, polar_delta = _polar_delta(epochs)
    last_epoch = int(epochs[-1]["epoch"]) if epochs else None
    best_epoch = None
    best_loss = None
    if best:
        if best.get("best_epoch") is not None:
            best_epoch = int(best["best_epoch"])
        best_loss = _f(best.get("best_valid_loss"))
        if last_polar is None:
            last_polar = _f(best.get("valid_polar_mae_bohr3") or best.get("valid_polar_mae"))
    if best_epoch is None and epochs:
        improved = [int(e["epoch"]) for e in epochs if e.get("improved")]
        best_epoch = improved[-1] if improved else last_epoch

    verdict, reason = verdict_from(
        epochs=epochs, best=best, log=log, job=job, files=files
    )
    history_file = (ckpt / HISTORY_NAME) if ckpt and (ckpt / HISTORY_NAME).is_file() else None
    return AuditReport(
        verdict=verdict,
        reason=reason,
        last_epoch=last_epoch,
        best_epoch=best_epoch,
        best_valid_loss=best_loss,
        first_valid_polar_mae=first_polar,
        last_valid_polar_mae=last_polar,
        polar_delta=polar_delta,
        n_epochs_logged=len(epochs),
        params_best_symlink=any(f.get("name") == "params-best.json" for f in files),
        history_path=str(history_file) if history_file else None,
        job=job,
        ready=log.get("ready") or (load_run_meta(ckpt) if ckpt else None),
        last_log_line=log.get("last_line") or None,
        files=files,
        epochs=epochs,
        crashes=list(log.get("crashes") or []),
    )


def format_report(report: AuditReport) -> str:
    lines = [
        f"verdict: {report.verdict}",
        f"reason:  {report.reason}",
        f"epochs:  last={report.last_epoch} best={report.best_epoch} n={report.n_epochs_logged}",
        f"valid polar mae Bohr³: first={report.first_valid_polar_mae} "
        f"last={report.last_valid_polar_mae} delta={report.polar_delta}",
        f"best_valid_loss: {report.best_valid_loss}",
        f"params-best.json symlink: {report.params_best_symlink}",
        f"history.jsonl: {report.history_path or '(missing — parse slurm / best-valid only)'}",
    ]
    if report.ready:
        lines.append(f"ready/meta: {report.ready}")
    if report.job:
        lines.append(f"sacct: {report.job}")
    if report.last_log_line:
        lines.append(f"last log: {report.last_log_line}")
    if report.crashes:
        lines.append("crashes:")
        lines.extend(f"  {c}" for c in report.crashes[:8])
    if report.files:
        lines.append("checkpoint files:")
        for row in report.files:
            extra = f" -> {row['target']}" if row.get("target") else ""
            lines.append(f"  {row['name']}{extra}")
    if report.epochs:
        lines.append("epoch table:")
        for row in report.epochs:
            lines.append(
                f"  ep {row.get('epoch'):>4}  valid_loss={row.get('valid_loss')}  "
                f"polar={row.get('valid_polar_mae_bohr3')}  improved={row.get('improved')}"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit an efield-train / spice-α polar job. A missing "
            "params-best.json during the run does not mean valid loss never improved."
        )
    )
    p.add_argument("--ckpt", type=Path, default=None, help="Checkpoint directory")
    p.add_argument("--log", type=Path, default=None, help="slurm-*.out (or .err)")
    p.add_argument("--job", default=None, help="Slurm job id (optional sacct)")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ckpt is None and args.log is None and args.job is None:
        build_parser().error("provide --ckpt and/or --log (and optionally --job)")
    report = audit_job(ckpt_dir=args.ckpt, log_path=args.log, job_id=args.job)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(format_report(report), end="")
    return 0 if report.verdict != "crashed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
