"""Per-stage and campaign observability for ``mmml md-system``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class MdStageSummary:
    stage: str
    job_id: str
    backend: str
    setup: str
    nsteps_requested: int
    nsteps_completed: int
    dt_fs: float
    ps_requested: float
    ps_completed: float
    temperature_K: float | None = None
    temperature_mean_K: float | None = None
    temperature_final_K: float | None = None
    temperature_first_K: float | None = None
    pressure_atm: float | None = None
    pressure_mean_atm: float | None = None
    density_g_cm3_mean: float | None = None
    density_g_cm3_final: float | None = None
    box_A_initial: float | None = None
    box_A_final: float | None = None
    volume_A3_final: float | None = None
    frames_written: int = 0
    record_every_steps: int = 0
    integrator: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    status: str = "complete"
    wall_time_s: float = 0.0
    description: str | None = None

    @property
    def truncated(self) -> bool:
        return self.nsteps_completed < self.nsteps_requested


@dataclass
class MdJobSummary:
    job_id: str
    backend: str
    setup: str
    stages: list[MdStageSummary] = field(default_factory=list)
    handoff: dict[str, str] = field(default_factory=dict)
    wall_time_s: float = 0.0
    exit_code: int = 0


def dynamics_nstep_from_ps(ps: float, dt_fs: float) -> int:
    return int(round(float(ps) * 1000.0 / float(dt_fs)))


def ps_from_nsteps(nsteps: int, dt_fs: float) -> float:
    return float(nsteps) * float(dt_fs) / 1000.0


def cubic_box_side_from_cell(cell: Any) -> float | None:
    if cell is None:
        return None
    import numpy as np

    arr = np.asarray(cell, dtype=float)
    if arr.shape == (3, 3):
        lengths = [float(np.linalg.norm(arr[k])) for k in range(3)]
        if all(l > 0 for l in lengths):
            return sum(lengths) / len(lengths)
    if arr.size >= 1:
        return float(arr.reshape(-1)[0])
    return None


def build_pycharmm_plan_rows(
    job_id: str,
    args: Any,
    *,
    description: str | None = None,
) -> list[MdStageSummary]:
    """Expand PyCHARMM ``md_stages`` into plan rows with step counts."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import dynamics_nstep_from_ps as _nstep

    dt = float(getattr(args, "dt_fs", 0.25))
    stages = str(getattr(args, "md_stages", "") or "").split(",")
    stages = [s.strip() for s in stages if s.strip()]
    if not stages:
        stages = ["dynamics"]
    rows: list[MdStageSummary] = []
    for stage in stages:
        if stage == "mini":
            nsteps = int(getattr(args, "mini_nstep", 0))
            ps = ps_from_nsteps(nsteps, dt)
        elif stage == "heat":
            ps = float(getattr(args, "ps_heat", getattr(args, "ps", 1.0)))
            nsteps = _nstep(ps, dt)
        elif stage == "nve":
            ps = float(getattr(args, "ps_nve", getattr(args, "ps", 1.0)))
            nsteps = _nstep(ps, dt)
        elif stage in {"equi", "prod"}:
            key = f"ps_{stage}"
            ps = float(getattr(args, key, getattr(args, "ps", 1.0)))
            nsteps = _nstep(ps, dt)
        else:
            ps = float(getattr(args, "ps", 1.0))
            nsteps = _nstep(ps, dt)
        t_first = None
        t_final = float(getattr(args, "temperature", 300.0) or 300.0)
        if stage == "heat":
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                resolve_heat_firstt_finalt,
            )

            t_first, t_final = resolve_heat_firstt_finalt(args, default_temp=t_final)
        p_atm = None
        if stage in {"equi", "prod"} and str(getattr(args, "setup", "")).endswith("npt"):
            raw_p = getattr(args, "npt_pressure", None)
            if raw_p is None:
                raw_p = getattr(args, "pressure", None)
            p_atm = float(raw_p) if raw_p is not None else None
        rows.append(
            MdStageSummary(
                stage=stage,
                job_id=job_id,
                backend="pycharmm",
                setup=str(getattr(args, "setup", "")),
                nsteps_requested=nsteps,
                nsteps_completed=0,
                dt_fs=dt,
                ps_requested=ps,
                ps_completed=0.0,
                temperature_K=t_final,
                temperature_first_K=t_first,
                pressure_atm=p_atm,
                box_A_initial=float(getattr(args, "box_size", 0) or 0) or None,
                record_every_steps=int(getattr(args, "dcd_nsavc", 1)),
                integrator=stage,
                status="planned",
                description=description,
            )
        )
    return rows


def build_single_leg_plan_row(job_id: str, args: Any, backend: str, *, description: str | None = None) -> MdStageSummary:
    dt = float(getattr(args, "dt_fs", 0.25))
    ps = float(getattr(args, "ps", 1.0))
    nsteps = dynamics_nstep_from_ps(ps, dt)
    setup = str(getattr(args, "setup", ""))
    ensemble = setup.split("_")[-1] if "_" in setup else "dynamics"
    record = 100
    if backend == "jaxmd":
        extra = list(getattr(args, "extra_args", []) or [])
        for i, tok in enumerate(extra):
            if tok == "--steps-per-recording" and i + 1 < len(extra):
                record = int(extra[i + 1])
    p_atm = float(getattr(args, "pressure", 1.0)) if ensemble == "npt" else None
    return MdStageSummary(
        stage=ensemble,
        job_id=job_id,
        backend=backend,
        setup=setup,
        nsteps_requested=nsteps,
        nsteps_completed=0,
        dt_fs=dt,
        ps_requested=ps,
        ps_completed=0.0,
        temperature_K=float(getattr(args, "temperature", 300.0)),
        pressure_atm=p_atm,
        box_A_initial=float(getattr(args, "box_size", 0) or 0) or None,
        record_every_steps=record,
        integrator=ensemble,
        status="planned",
        description=description,
    )


def write_stage_summary_json(job_summary: MdJobSummary, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stage_summary.json"
    payload = {
        "job_id": job_summary.job_id,
        "backend": job_summary.backend,
        "setup": job_summary.setup,
        "exit_code": job_summary.exit_code,
        "wall_time_s": job_summary.wall_time_s,
        "handoff": job_summary.handoff,
        "stages": [asdict(s) for s in job_summary.stages],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_campaign_plan(path: Path, rows: list[MdStageSummary]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_campaign_summary(path: Path, jobs: list[MdJobSummary]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "jobs": [
            {
                "job_id": j.job_id,
                "backend": j.backend,
                "setup": j.setup,
                "exit_code": j.exit_code,
                "wall_time_s": j.wall_time_s,
                "stages": [asdict(s) for s in j.stages],
            }
            for j in jobs
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def print_campaign_plan(rows: list[MdStageSummary], *, console: Console | None = None) -> None:
    c = console or Console()
    table = Table(title="[bold cyan]MD campaign plan[/bold cyan]", show_header=True)
    for col in ("Job", "Backend", "Stage", "Steps", "ps", "dt", "T", "P", "Box"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            row.job_id,
            row.backend,
            row.stage,
            str(row.nsteps_requested),
            f"{row.ps_requested:.2f}",
            f"{row.dt_fs}",
            f"{row.temperature_K}" if row.temperature_K is not None else "—",
            f"{row.pressure_atm}" if row.pressure_atm is not None else "—",
            f"{row.box_A_initial:.1f}" if row.box_A_initial else "—",
        )
    c.print(Panel(table, border_style="cyan"))


def print_stage_summary(stage: MdStageSummary, *, console: Console | None = None) -> None:
    c = console or Console()
    table = Table(title=f"[bold]{stage.job_id} / {stage.stage}[/bold]")
    table.add_column("Quantity")
    table.add_column("Value")
    table.add_row("Steps", f"{stage.nsteps_completed}/{stage.nsteps_requested}")
    table.add_row("ps", f"{stage.ps_completed:.3f}/{stage.ps_requested:.3f}")
    if stage.temperature_K is not None:
        table.add_row("T target (K)", f"{stage.temperature_K:.2f}")
    if stage.pressure_atm is not None:
        table.add_row("P target (atm)", f"{stage.pressure_atm:.2f}")
    if stage.density_g_cm3_final is not None:
        table.add_row("rho final (g/cm³)", f"{stage.density_g_cm3_final:.4f}")
    if stage.box_A_final is not None:
        table.add_row("box final (Å)", f"{stage.box_A_final:.3f}")
    table.add_row("frames", str(stage.frames_written))
    table.add_row("status", stage.status)
    c.print(Panel(table, border_style="green" if not stage.truncated else "yellow"))


def print_campaign_report(jobs: list[MdJobSummary], *, console: Console | None = None) -> None:
    c = console or Console()
    table = Table(title="[bold]Campaign summary[/bold]")
    table.add_column("Job")
    table.add_column("Backend")
    table.add_column("Stages")
    table.add_column("Steps ok")
    table.add_column("Exit")
    for job in jobs:
        ok = sum(1 for s in job.stages if not s.truncated and s.status == "complete")
        table.add_row(
            job.job_id,
            job.backend,
            str(len(job.stages)),
            f"{ok}/{len(job.stages)}",
            str(job.exit_code),
        )
    c.print(Panel(table, border_style="blue"))
