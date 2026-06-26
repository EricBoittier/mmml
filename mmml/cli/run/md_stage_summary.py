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


def pycharmm_trajectory_tag(args: Any) -> str:
    """Filename tag used by staged PyCHARMM artifacts (e.g. ``dcm_20``)."""
    comp = getattr(args, "composition", None)
    if comp:
        return str(comp).strip().lower().replace(":", "_")
    residue = str(getattr(args, "residue", "cluster") or "cluster").lower()
    n_mol = int(getattr(args, "n_molecules", 1) or 1)
    return f"{residue}_{n_mol}"


def pycharmm_stage_dcd_frames(output_dir: Path, stage: str, tag: str) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import stage_dcd
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        count_overlap_chunk_dcd_frames,
        count_readable_dcd_frames,
        overlap_chunk_dcd_paths,
    )

    out = Path(output_dir)
    for path in (stage_dcd(out, stage), out / "pretreat" / f"{stage}.dcd"):
        if path.is_file():
            if overlap_chunk_dcd_paths(path):
                _header, readable = count_overlap_chunk_dcd_frames(path)
                return int(readable)
            return int(count_readable_dcd_frames(path))
    chunk_paths = overlap_chunk_dcd_paths(stage_dcd(out, stage))
    if chunk_paths:
        return int(sum(count_readable_dcd_frames(p) for p in chunk_paths))
    return 0


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
    """Expand PyCHARMM ``md_stages`` / ``md_stage`` into plan rows with step counts."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        dynamics_nstep_from_ps as _nstep,
        opt_attr,
        resolve_dcd_nsavc,
        resolve_dt_fs,
        resolve_heat_firstt_finalt,
        resolve_md_stages,
        resolve_stage_pressure_atm,
        resolve_stage_ps,
        resolve_stage_temperature_K,
    )

    dt = resolve_dt_fs(args)
    timestep_ps = float(dt) / 1000.0
    stages = resolve_md_stages(args)
    rows: list[MdStageSummary] = []
    for stage in stages:
        if stage == "mini":
            nsteps = int(opt_attr(args, "mini_nstep", 0) or 0)
            ps = ps_from_nsteps(nsteps, dt)
        else:
            ps = resolve_stage_ps(args, stage)
            nsteps = _nstep(ps, dt)
        t_first = None
        t_final = resolve_stage_temperature_K(args)
        if stage == "heat":
            t_first, t_final = resolve_heat_firstt_finalt(args, default_temp=t_final)
        p_atm = resolve_stage_pressure_atm(args, stage)
        box_raw = opt_attr(args, "box_size", None)
        dcd_max = opt_attr(args, "dcd_max_frames", 25)
        record_every = resolve_dcd_nsavc(
            dcd_nsavc=int(opt_attr(args, "dcd_nsavc", 1) or 1),
            dcd_interval_ps=opt_attr(args, "dcd_interval_ps", None),
            timestep_ps=timestep_ps,
            nstep=nsteps if nsteps > 0 else None,
            dcd_max_frames=dcd_max,
        )
        rows.append(
            MdStageSummary(
                stage=stage,
                job_id=job_id,
                backend="pycharmm",
                setup=str(opt_attr(args, "setup", "") or ""),
                nsteps_requested=nsteps,
                nsteps_completed=0,
                dt_fs=dt,
                ps_requested=ps,
                ps_completed=0.0,
                temperature_K=t_final,
                temperature_first_K=t_first,
                pressure_atm=p_atm,
                box_A_initial=float(box_raw) if box_raw is not None else None,
                record_every_steps=record_every,
                integrator=stage,
                status="planned",
                description=description,
            )
        )
    return rows


def build_single_leg_plan_row(job_id: str, args: Any, backend: str, *, description: str | None = None) -> MdStageSummary:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        opt_attr,
        resolve_dt_fs,
        resolve_stage_temperature_K,
    )

    dt = resolve_dt_fs(args)
    ps = float(opt_attr(args, "ps", 1.0))
    nsteps = dynamics_nstep_from_ps(ps, dt)
    setup = str(opt_attr(args, "setup", "") or "")
    ensemble = setup.split("_")[-1] if "_" in setup else "dynamics"
    record = 100
    if backend == "jaxmd":
        extra = list(opt_attr(args, "extra_args", []) or [])
        for i, tok in enumerate(extra):
            if tok == "--steps-per-recording" and i + 1 < len(extra):
                record = int(extra[i + 1])
    p_raw = opt_attr(args, "pressure", None)
    p_atm = float(p_raw) if ensemble == "npt" and p_raw is not None else None
    box_raw = opt_attr(args, "box_size", None)
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
        temperature_K=resolve_stage_temperature_K(args),
        pressure_atm=p_atm,
        box_A_initial=float(box_raw) if box_raw is not None else None,
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
