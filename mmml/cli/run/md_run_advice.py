"""Post-run guidance: next ``md-system`` command, config snippet, restart pick."""

from __future__ import annotations

import json
import math
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
    BASELINE_RES,
    SNAPSHOTS_JSON,
    geometry_baseline_res,
    stage_restart,
)

_STAGE_ORDER = ("mini", "heat", "nve", "equi", "prod")
_SEGMENT_RES = re.compile(r"^(heat|nve|equi|prod)\.(\d+)\.res$", re.IGNORECASE)


@dataclass(frozen=True)
class RestartCandidate:
    path: Path
    label: str
    leg: str
    hybrid_grms: float | None
    source: str
    mtime: float
    is_restart: bool = True

    def display_grms(self) -> str:
        if self.hybrid_grms is None or not math.isfinite(self.hybrid_grms):
            return "—"
        return f"{self.hybrid_grms:.2f}"


@dataclass
class RunAdvice:
    exit_code: int
    headline: str
    restart: RestartCandidate | None = None
    md_stages: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    include_presets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    command: str = ""
    config_yaml: str = ""
    shell_script: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exit_code": self.exit_code,
            "headline": self.headline,
            "md_stages": self.md_stages,
            "config_overrides": dict(self.config_overrides),
            "include_presets": list(self.include_presets),
            "notes": list(self.notes),
            "command": self.command,
            "config_yaml": self.config_yaml,
            "shell_script": self.shell_script,
        }
        if self.restart is not None:
            payload["restart"] = {
                **asdict(self.restart),
                "path": str(self.restart.path),
            }
        return payload


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_usable_coordinate(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _is_valid_restart(path: Path) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

    return _valid_restart_file(path) is not None


def _add_candidate(
    out: list[RestartCandidate],
    *,
    path: Path | str | None,
    label: str,
    leg: str,
    hybrid_grms: float | None,
    source: str,
) -> None:
    if path is None:
        return
    p = Path(path).expanduser()
    if not p.is_file():
        return
    is_restart = p.suffix.lower() == ".res" and _is_valid_restart(p)
    is_crd = p.suffix.lower() == ".crd" and _is_usable_coordinate(p)
    if not is_restart and not is_crd:
        return
    out.append(
        RestartCandidate(
            path=p.resolve(),
            label=label,
            leg=leg,
            hybrid_grms=hybrid_grms,
            source=source,
            mtime=_file_mtime(p),
            is_restart=is_restart,
        )
    )


def _journal_steps(journal_path: Path) -> list[dict[str, Any]]:
    if not journal_path.is_file():
        return []
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    steps = payload.get("steps")
    return steps if isinstance(steps, list) else []


def _collect_journal_candidates(
    out: list[RestartCandidate],
    journal_path: Path,
    *,
    leg_prefix: str,
) -> None:
    for step in _journal_steps(journal_path):
        if not isinstance(step, dict):
            continue
        label = str(step.get("label") or step.get("slug") or "checkpoint")
        grms = step.get("hybrid_grms_kcalmol_A")
        hybrid_grms = float(grms) if grms is not None else None
        files = step.get("files") or {}
        if not isinstance(files, dict):
            continue
        stem = str(step.get("stem") or label)
        for key in ("restart", "res"):
            _add_candidate(
                out,
                path=files.get(key),
                label=f"{leg_prefix}: {label}",
                leg=leg_prefix,
                hybrid_grms=hybrid_grms,
                source=str(journal_path),
            )
        _add_candidate(
            out,
            path=files.get("crd"),
            label=f"{leg_prefix}: {label}",
            leg=leg_prefix,
            hybrid_grms=hybrid_grms,
            source=str(journal_path),
        )
        if journal_path.parent.is_dir():
            parent = journal_path.parent
            _add_candidate(
                out,
                path=parent / f"{stem}.crd",
                label=f"{leg_prefix}: {label}",
                leg=leg_prefix,
                hybrid_grms=hybrid_grms,
                source=str(journal_path),
            )
            _add_candidate(
                out,
                path=parent / f"{stem}.res",
                label=f"{leg_prefix}: {label}",
                leg=leg_prefix,
                hybrid_grms=hybrid_grms,
                source=str(journal_path),
            )


def _collect_snapshots(out: list[RestartCandidate], snapshots_path: Path) -> None:
    if not snapshots_path.is_file():
        return
    try:
        payload = json.loads(snapshots_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for snap in payload.get("snapshots") or []:
        if not isinstance(snap, dict):
            continue
        label = str(snap.get("label") or snap.get("stem") or "snapshot")
        grms = snap.get("grms_kcalmol_A")
        hybrid_grms = float(grms) if grms is not None else None
        files = snap.get("files") or {}
        if not isinstance(files, dict):
            continue
        for key in ("restart", "res", "crd"):
            _add_candidate(
                out,
                path=files.get(key),
                label=f"mini: {label}",
                leg="mini",
                hybrid_grms=hybrid_grms,
                source=str(snapshots_path),
            )


def collect_restart_candidates(output_dir: Path) -> list[RestartCandidate]:
    """Scan an ``md-system`` output directory for restartable legs with GRMS."""
    out_dir = Path(output_dir).expanduser().resolve()
    out: list[RestartCandidate] = []

    _collect_snapshots(out, out_dir / SNAPSHOTS_JSON)
    _collect_journal_candidates(
        out, out_dir / "prep_ladder" / "journal.json", leg_prefix="prep_ladder"
    )
    _collect_journal_candidates(
        out, out_dir / "cleanup" / "journal.json", leg_prefix="cleanup"
    )

    baseline = geometry_baseline_res(out_dir)
    _add_candidate(
        out,
        path=baseline,
        label="geometry baseline",
        leg="baseline",
        hybrid_grms=None,
        source="artifact_paths",
    )

    for stage in _STAGE_ORDER:
        if stage == "mini":
            for name, leg in (
                ("mini.crd", "mini"),
                ("03_bmm.crd", "bonded_mm"),
            ):
                _add_candidate(
                    out,
                    path=out_dir / name,
                    label=name,
                    leg=leg,
                    hybrid_grms=None,
                    source="staged",
                )
            continue
        res = stage_restart(out_dir, stage)
        _add_candidate(
            out,
            path=res,
            label=f"{stage} stage restart",
            leg=stage,
            hybrid_grms=None,
            source="staged",
        )
        for seg_path in sorted(out_dir.glob(f"{stage}.*.res")):
            m = _SEGMENT_RES.match(seg_path.name)
            if not m:
                continue
            seg_stage, seg_i = m.group(1).lower(), int(m.group(2))
            _add_candidate(
                out,
                path=seg_path,
                label=f"{seg_stage} segment {seg_i}",
                leg=seg_stage,
                hybrid_grms=None,
                source="segment",
            )

    pretreat = out_dir / "pretreat"
    if pretreat.is_dir():
        for stage in ("mini_box_equil", "heat", "equi", "prod"):
            _add_candidate(
                out,
                path=pretreat / f"{stage}.res",
                label=f"pretreat {stage}",
                leg=f"pretreat_{stage}",
                hybrid_grms=None,
                source="pretreat",
            )

    prep_root = out_dir / "prep_ladder"
    if prep_root.is_dir():
        for crd in sorted(prep_root.glob("*.crd")):
            _add_candidate(
                out,
                path=crd,
                label=crd.stem,
                leg="prep_ladder",
                hybrid_grms=None,
                source="prep_ladder",
            )

    # De-duplicate by resolved path, keeping the richest GRMS label.
    by_path: dict[Path, RestartCandidate] = {}
    for cand in out:
        prev = by_path.get(cand.path)
        if prev is None:
            by_path[cand.path] = cand
            continue
        prev_grms = prev.hybrid_grms
        new_grms = cand.hybrid_grms
        if new_grms is not None and (
            prev_grms is None or new_grms < prev_grms
        ):
            by_path[cand.path] = cand
        elif prev_grms == new_grms and cand.mtime > prev.mtime:
            by_path[cand.path] = cand
    return sorted(by_path.values(), key=lambda c: c.mtime)


def select_restart_candidate(
    candidates: list[RestartCandidate],
    *,
    failed: bool,
) -> RestartCandidate | None:
    """Pick restart leg: lowest hybrid GRMS on failure; latest restart on success."""
    if not candidates:
        return None

    restarts = [c for c in candidates if c.is_restart]
    pool = restarts if restarts else [c for c in candidates if c.path.suffix.lower() == ".crd"]
    if not pool:
        return None

    if failed:
        with_grms = [
            c
            for c in pool
            if c.hybrid_grms is not None and math.isfinite(c.hybrid_grms)
        ]
        if with_grms:
            min_grms = min(c.hybrid_grms for c in with_grms)  # type: ignore[type-var]
            tol = max(0.05 * float(min_grms), 0.5)
            tied = [
                c
                for c in with_grms
                if float(c.hybrid_grms) <= float(min_grms) + tol  # type: ignore[arg-type]
            ]
            return max(tied, key=lambda c: c.mtime)

    restart_pool = [c for c in pool if c.is_restart]
    if restart_pool:
        return max(restart_pool, key=lambda c: c.mtime)
    return max(pool, key=lambda c: c.mtime)


def _parse_md_stages(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    return []


def _load_stage_summary(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "stage_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _stage_leg_rank(leg: str) -> int:
    leg_l = leg.lower()
    for i, stage in enumerate(_STAGE_ORDER):
        if leg_l == stage or leg_l.startswith(stage):
            return i
    if leg_l == "baseline":
        return _STAGE_ORDER.index("mini")
    if leg_l.startswith("prep") or leg_l.startswith("pretreat"):
        return _STAGE_ORDER.index("mini")
    if leg_l == "bonded_mm":
        return _STAGE_ORDER.index("mini")
    return -1


def _infer_failed_stage(
    planned: list[str],
    summary: dict[str, Any] | None,
    restart: RestartCandidate | None,
) -> str | None:
    if summary:
        stages = summary.get("stages") or []
        for row in stages:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "")
            if status in {"error", "truncated"}:
                return str(row.get("stage") or "")
    if restart is not None:
        rank = _stage_leg_rank(restart.leg)
        if rank >= 0 and planned:
            for stage in reversed(planned):
                if _STAGE_ORDER.index(stage) <= rank:
                    return stage
    return planned[-1] if planned else None


def _completed_stages(summary: dict[str, Any] | None) -> set[str]:
    done: set[str] = set()
    if not summary:
        return done
    for row in summary.get("stages") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") == "complete":
            stage = str(row.get("stage") or "")
            if stage:
                done.add(stage)
    return done


def _remaining_stages(planned: list[str], completed: set[str]) -> list[str]:
    if not planned:
        return []
    for stage in planned:
        if stage not in completed:
            return planned[planned.index(stage) :]
    return []


def _failure_preset_hints(
    output_dir: Path,
    manifest_args: dict[str, Any],
) -> tuple[list[str], list[str]]:
    includes: list[str] = []
    notes: list[str] = []
    cleanup_journal = output_dir / "cleanup" / "journal.json"
    if cleanup_journal.is_file() and _journal_steps(cleanup_journal):
        includes.append("presets/dynamics-flyoff-strict.yaml")
        notes.append(
            "cleanup/ checkpoints present — overlap or fly-off rescue likely; "
            "strict fly-off guard preset included."
        )
    elif (output_dir / "cleanup").is_dir():
        includes.append("presets/dynamics-overlap-rescue.yaml")
        notes.append("cleanup/ folder present — overlap rescue preset suggested.")

    if not bool(manifest_args.get("no_echeck_heat", False)):
        notes.append("ECHECK was enabled for heat; set no_echeck_heat: true when GRMS > 30.")
    if str(manifest_args.get("heat_thermostat") or "") == "hoover":
        notes.append(
            "Consider heat-dt0.25-scale.yaml (velocity scaling) if Hoover CPT "
            "is not required."
        )
    return includes, notes


def _relative_path(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_run_advice(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    exit_code: int,
    repo_root: Path | None = None,
) -> RunAdvice | None:
    """Build next-run guidance from manifest + on-disk artifacts."""
    args = manifest.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    backend = str(manifest.get("backend") or args.get("backend") or "")
    if backend and backend != "pycharmm":
        return None

    out_dir = Path(output_dir).expanduser().resolve()
    if not out_dir.is_dir():
        return None

    failed = int(exit_code) != 0
    planned = _parse_md_stages(args.get("md_stages"))
    summary = _load_stage_summary(out_dir)
    candidates = collect_restart_candidates(out_dir)
    restart = select_restart_candidate(candidates, failed=failed)

    completed = _completed_stages(summary)
    remaining = _remaining_stages(planned, completed)

    config_path = args.get("config") or manifest.get("config")
    job_name = str(manifest.get("job_name") or args.get("job_name") or "run")
    resume_job = f"{job_name}_resume" if failed else job_name

    overrides: dict[str, Any] = {
        "output_dir": str(args.get("output_dir") or out_dir),
        "backend": "pycharmm",
    }
    include_presets: list[str] = []
    notes: list[str] = []

    if restart is not None:
        overrides["restart_from"] = _relative_path(restart.path, repo_root)
        if not restart.is_restart:
            notes.append(
                f"Selected checkpoint is {restart.path.suffix}; prefer a .res restart "
                f"when available (e.g. {BASELINE_RES})."
            )
    elif failed:
        notes.append("No valid restart file found — rebuild from liquid-box or pretreat.")

    if failed:
        failed_stage = _infer_failed_stage(planned, summary, restart)
        remaining_after_fail = _remaining_stages(planned, completed)
        if remaining_after_fail:
            overrides["md_stages"] = ",".join(remaining_after_fail)
        elif failed_stage and failed_stage in planned:
            idx = planned.index(failed_stage)
            overrides["md_stages"] = ",".join(planned[idx:])
        preset_includes, preset_notes = _failure_preset_hints(out_dir, args)
        include_presets.extend(preset_includes)
        notes.extend(preset_notes)
        headline = f"Job failed (exit {exit_code}) — resume from best checkpoint"
    else:
        if remaining:
            overrides["md_stages"] = ",".join(remaining)
            if restart is None:
                for stage in reversed(_STAGE_ORDER):
                    res = stage_restart(out_dir, stage)
                    if _is_valid_restart(res):
                        overrides["restart_from"] = _relative_path(res, repo_root)
                        notes.append(f"Restart from last completed stage: {stage}.res")
                        break
            headline = f"Job succeeded — continue remaining stages ({','.join(remaining)})"
        else:
            headline = "Job succeeded — staged PyCHARMM leg complete"
            notes.append(
                "Optional: hand off to JAX-MD with mmml md-system --backend jaxmd "
                "and --continue-from <handoff.npz>."
            )

    md_stages = str(overrides.get("md_stages") or args.get("md_stages") or "")

    yaml_lines: list[str] = [
        "# Suggested resume config (generated by mmml md-system)",
        f"# prior job: {job_name} exit={exit_code}",
    ]
    if config_path:
        yaml_lines.append(f"# base config: {config_path}")
        yaml_lines.append(f"include:")
        yaml_lines.append(f"  - {config_path}")
    elif include_presets:
        yaml_lines.append("include:")
        for preset in include_presets:
            yaml_lines.append(f"  - {preset}")
    elif failed:
        yaml_lines.append("include:")
        yaml_lines.append("  - presets/heat-dt0.25-scale.yaml")
        yaml_lines.append("  - presets/dynamics-flyoff-strict.yaml")

    yaml_lines.append("")
    yaml_lines.append("defaults:")
    for key in sorted(overrides):
        val = overrides[key]
        if isinstance(val, bool):
            yaml_lines.append(f"  {key}: {'true' if val else 'false'}")
        else:
            yaml_lines.append(f"  {key}: {val!r}")

    config_yaml = "\n".join(yaml_lines) + "\n"

    cmd_parts = ["mmml", "md-system"]
    if config_path:
        cmd_parts.extend(["--config", str(config_path)])
    cmd_parts.extend(["--job-id", resume_job])
    for key, val in overrides.items():
        flag = key.replace("_", "-")
        if isinstance(val, bool):
            cmd_parts.append(f"--{flag}" if val else f"--no-{flag}")
        else:
            cmd_parts.extend([f"--{flag}", str(val)])
    command = " ".join(shlex.quote(p) for p in cmd_parts)

    shell_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# Suggested resume (wrap with ./scripts/mmml-charmm-mpirun.sh on GPU nodes).",
            f"# prior job: {job_name} exit={exit_code}",
            "",
            command,
            "",
        ]
    )

    return RunAdvice(
        exit_code=int(exit_code),
        headline=headline,
        restart=restart,
        md_stages=md_stages or None,
        config_overrides=overrides,
        include_presets=include_presets,
        notes=notes,
        command=command,
        config_yaml=config_yaml,
        shell_script=shell_script,
    )


def write_run_advice_files(advice: RunAdvice, output_dir: Path) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "next_run_advice.json",
        "yaml": out_dir / "next_run.yaml",
        "sh": out_dir / "next_run.sh",
    }
    paths["json"].write_text(
        json.dumps(advice.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    paths["yaml"].write_text(advice.config_yaml, encoding="utf-8")
    paths["sh"].write_text(advice.shell_script, encoding="utf-8")
    paths["sh"].chmod(paths["sh"].stat().st_mode | 0o111)
    return paths


def emit_run_advice(
    advice: RunAdvice,
    *,
    output_dir: Path,
    quiet: bool = False,
) -> None:
    if quiet:
        return
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("Status", advice.headline)
        if advice.restart is not None:
            table.add_row("Restart", str(advice.restart.path))
            table.add_row("Leg", advice.restart.label)
            table.add_row("Hybrid GRMS", f"{advice.restart.display_grms()} kcal/mol/Å")
        if advice.md_stages:
            table.add_row("md_stages", advice.md_stages)
        for note in advice.notes:
            table.add_row("Note", note)
        table.add_row("Command", advice.command)
        table.add_row("Config", str(Path(output_dir) / "next_run.yaml"))
        table.add_row("Script", str(Path(output_dir) / "next_run.sh"))
        border = "red" if advice.exit_code != 0 else "green"
        console.print(Panel(table, title="[bold]Next md-system run[/bold]", border_style=border))
        console.print("[dim]Copy-paste:[/dim]")
        console.print(advice.command)
    except ImportError:
        print("\n--- Next md-system run ---", flush=True)
        print(advice.headline, flush=True)
        if advice.restart is not None:
            print(
                f"Restart: {advice.restart.path} "
                f"(GRMS {advice.restart.display_grms()} kcal/mol/Å)",
                flush=True,
            )
        if advice.md_stages:
            print(f"md_stages: {advice.md_stages}", flush=True)
        for note in advice.notes:
            print(f"Note: {note}", flush=True)
        print(f"Command: {advice.command}", flush=True)
        print(f"Wrote {Path(output_dir) / 'next_run.yaml'}", flush=True)


def maybe_emit_run_advice(
    args: Any,
    *,
    manifest: dict[str, Any],
    exit_code: int,
    repo_root: Path | None = None,
) -> RunAdvice | None:
    if getattr(args, "no_run_advice", False):
        return None
    if getattr(args, "run_all", False):
        return None
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return None
    backend = manifest.get("backend")
    if backend not in (None, "pycharmm"):
        return None
    advice = build_run_advice(
        manifest=manifest,
        output_dir=Path(output_dir),
        exit_code=int(exit_code),
        repo_root=repo_root,
    )
    if advice is None:
        return None
    write_run_advice_files(advice, Path(output_dir))
    emit_run_advice(
        advice,
        output_dir=Path(output_dir),
        quiet=bool(getattr(args, "quiet", False)),
    )
    return advice
