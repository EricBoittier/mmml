"""Health monitoring helpers for pbc_liquid_density_dyn campaigns."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    RunCell,
    campaign_job_order,
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    paths_for_run,
    repo_root,
)

_DYNA_LINE = re.compile(
    r"^DYNA>\s+(\d+)\s+([\d.]+)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)

_ERROR_MARKERS = re.compile(
    r"Segmentation fault|Signal: Segmentation|exit code -?11|"
    r"pycharmm_mlpot: error:|intra-monomer|overlap|Traceback|"
    r"FAIL:|ERROR|CANCELLED|Killed|exit code [1-9]|"
    r"liquid-density dynamics campaign failed|dynamics campaign failed",
    re.IGNORECASE,
)

_SNAKEMAKE_NOISE = re.compile(
    r"Error in rule run_liquid_density_dyn|CalledProcessError in file|WorkflowError:|"
    r"slurmstepd: error:|srun: error:|Exiting because a job execution failed|"
    r"pmixp_client_v2\.c:",
    re.IGNORECASE,
)

_STAGE_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("init", re.compile(r"pycharmm_init|warmup-mlpot-jax|liquid_prep|density_prep", re.I)),
    ("pretreat", re.compile(r"CHARMM MM pretreat|charmm_mm_pretreat", re.I)),
    ("mini", re.compile(r"MLpot SD minimize|minimize_with_mlpot", re.I)),
    ("heat", re.compile(r"heat segment|MLpot heat|CHARMM MM pretreat heat", re.I)),
    ("equi", re.compile(r"md_stage.*equi|NPT equil|ps_equi", re.I)),
    ("jaxmd", re.compile(r"JAX-MD|jaxmd_burst|mmml md-system.*jaxmd", re.I)),
    ("packmol", re.compile(r"Packmol|packmol", re.I)),
]

_LOG_ACTIVITY = re.compile(
    r"MLpot SD minimize|heat segment|DYNA>|warmup-mlpot-jax|Running:|Campaign jobs|Packmol",
    re.IGNORECASE,
)

_FAILURE_BUCKETS: list[tuple[str, re.Pattern[str]]] = [
    ("A_grms_gate", re.compile(r"Pre-dynamics GRMS \d+.*> \d+", re.I)),
    ("B_heat_readyn", re.compile(r"READYN restart|__dynio_MOD_readyn|pretreat/charmm_mm_prod", re.I)),
    ("C_handoff_fortran", re.compile(r"Fortran runtime error: Bad value during floating point read|continue_seed\.res", re.I)),
    ("D_mlpot_reg_sigsegv", re.compile(r"MLpot registration:.*\n.*Signal: Segmentation|Signal: Segmentation.*exit code -11", re.I)),
    ("E_oom", re.compile(r"exit code -9|Killed", re.I)),
    ("F_heat_overlap", re.compile(r"dynamics aborted after chunk|intra-monomer close contact|echeck or CHARMM abort", re.I)),
    ("G_jax_compile", re.compile(r"MLpot USER active before MLpot SD minimize(?!.*SD pass)", re.I)),
    ("H_memory_handoff_ok", re.compile(r"memory handoff|in-memory coords after mini", re.I)),
]


def classify_failure(text: str) -> list[str]:
    """Return failure bucket labels (A–H) matched in campaign log text."""
    hits: list[str] = []
    for label, pat in _FAILURE_BUCKETS:
        if pat.search(text):
            hits.append(label)
    if not hits and _ERROR_MARKERS.search(text):
        hits.append("Z_other")
    return hits


def extract_campaign_markers(text: str) -> dict[str, str]:
    """Parse tier, mini-nstep scaling, and heat handoff markers from stdout."""
    out: dict[str, str] = {}
    m = re.search(r"tier=(\w+)\s+max_Npr=(\d+)", text)
    if m:
        out["tier"] = m.group(1)
        out["max_Npr"] = m.group(2)
    m = re.search(r"mini-nstep scaled (\d+) -> (\d+)", text)
    if m:
        out["mini_nstep"] = f"{m.group(1)}->{m.group(2)}"
    m = re.search(r"max_grms_before_dyn scaled [\d.]+ -> ([\d.]+)", text)
    if m:
        out["max_grms_limit"] = m.group(1)
    if re.search(r"in-memory coords after mini", text):
        out["heat_handoff"] = "memory"
    elif re.search(r"coords from .*/pretreat/charmm_mm_", text):
        out["heat_handoff"] = "pretreat_readyn"
    return out


def parse_dyna_lines(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        m = _DYNA_LINE.match(line.strip())
        if not m:
            continue
        rows.append(
            {
                "step": float(m.group(1)),
                "time_ps": float(m.group(2)),
                "total_energy_kcal": float(m.group(3)),
                "kinetic_energy_kcal": float(m.group(4)),
                "potential_energy_kcal": float(m.group(5)),
                "temperature_K": float(m.group(6)),
            }
        )
    return rows


def summarize_dyna(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"n_frames": 0}
    import numpy as np

    etot = np.array([r["total_energy_kcal"] for r in rows], dtype=np.float64)
    epot = np.array([r["potential_energy_kcal"] for r in rows], dtype=np.float64)
    temp = np.array([r["temperature_K"] for r in rows], dtype=np.float64)
    time_ps = np.array([r["time_ps"] for r in rows], dtype=np.float64)
    out: dict[str, Any] = {
        "n_frames": len(rows),
        "step_first": int(rows[0]["step"]),
        "step_last": int(rows[-1]["step"]),
        "time_ps_first": float(time_ps[0]),
        "time_ps_last": float(time_ps[-1]),
        "time_ps_span": float(time_ps[-1] - time_ps[0]) if len(time_ps) > 1 else 0.0,
        "total_energy_first_kcal": float(etot[0]),
        "total_energy_last_kcal": float(etot[-1]),
        "total_energy_drift_kcal": float(etot[-1] - etot[0]),
        "total_energy_std_kcal": float(np.std(etot)),
        "potential_energy_std_kcal": float(np.std(epot)),
        "temperature_first_K": float(temp[0]),
        "temperature_last_K": float(temp[-1]),
        "temperature_mean_K": float(np.mean(temp)),
        "temperature_std_K": float(np.std(temp)),
    }
    if len(etot) >= 2:
        out["max_step_delta_kcal"] = float(np.max(np.abs(np.diff(etot))))
    return out


def read_text_tail(path: Path, *, max_bytes: int = 512_000) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            raw = f.read()
    except OSError:
        return ""
    if b"\x00" in raw[:4096]:
        return ""
    return raw.decode("utf-8", errors="replace")


def grep_errors(text: str, *, limit: int = 8) -> list[str]:
    hits: list[str] = []
    noise: list[str] = []
    for line in text.splitlines():
        if not _ERROR_MARKERS.search(line):
            continue
        stripped = line.rstrip()[:200]
        if _SNAKEMAKE_NOISE.search(line):
            noise.append(stripped)
        else:
            hits.append(stripped)
        if len(hits) >= limit:
            break
    if not hits:
        return noise[:limit]
    return hits[:limit]


def last_log_stage(text: str) -> str | None:
    last: str | None = None
    for name, pat in _STAGE_MARKERS:
        if pat.search(text):
            last = name
    return last


def _read_restart_step(path: Path) -> int | None:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
            read_restart_last_step,
        )

        return read_restart_last_step(path)
    except Exception:
        return None


def _restart_has_bad_coords(path: Path) -> bool:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
            restart_has_nonfinite_coordinates,
        )

        return bool(restart_has_nonfinite_coordinates(path))
    except Exception:
        return False


def collect_restart_info(out_dir: Path, *, tag: str) -> list[dict[str, Any]]:
    """Restart files under the cell tree with integrated step when parseable."""
    patterns = [
        f"heat_{tag}*.res",
        f"equi_{tag}*.res",
        f"prod_{tag}*.res",
        f"nve_{tag}*.res",
        f"charmm_mm_*_{tag}.res",
        "*.res",
    ]
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for pat in patterns:
        for path in sorted(out_dir.rglob(pat)):
            if not path.is_file() or path in seen:
                continue
            if path.name.startswith("."):
                continue
            seen.add(path)
            step = _read_restart_step(path)
            rows.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "size_B": path.stat().st_size,
                    "step": step,
                    "bad_coords": _restart_has_bad_coords(path),
                    "mtime": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                }
            )
    rows.sort(key=lambda r: (r.get("step") or 0, r["path"]))
    return rows


def leg_handoff_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / job_id / "handoff" / "state.npz"


def completed_leg_ids(out_dir: Path, job_order: list[str]) -> list[str]:
    done: list[str] = []
    for job_id in job_order:
        if leg_handoff_path(out_dir, job_id).is_file():
            done.append(job_id)
    return done


def active_leg_id(out_dir: Path, job_order: list[str]) -> str | None:
    completed = completed_leg_ids(out_dir, job_order)
    if completed == job_order:
        return job_order[-1]
    if not completed:
        return job_order[0]
    last_done_idx = job_order.index(completed[-1])
    if last_done_idx + 1 < len(job_order):
        return job_order[last_done_idx + 1]
    return completed[-1]


def load_campaign_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def campaign_leg_progress(
    summary: dict[str, Any],
    job_order: list[str],
) -> tuple[str | None, str | None, float | None]:
    """Return (last_ok_leg, failed_leg, last_stage_ps_fraction)."""
    jobs = summary.get("jobs") or []
    if not jobs:
        return None, None, None
    last_ok: str | None = None
    failed: str | None = None
    stage_frac: float | None = None
    for job in jobs:
        jid = str(job.get("job_id", ""))
        ec = int(job.get("exit_code", 0))
        if ec == 0:
            last_ok = jid or last_ok
        else:
            failed = jid or failed
            stages = job.get("stages") or []
            if stages:
                st = stages[-1]
                req = float(st.get("ps_requested") or 0)
                done = float(st.get("ps_completed") or 0)
                if req > 0:
                    stage_frac = done / req
            break
    return last_ok, failed, stage_frac


@dataclass
class RunMonitor:
    run_tag: str
    solvent: str
    n_monomers: int
    temperature_target_K: float
    box_size_A: float
    out_dir: str
    status: str
    health: str
    active_leg: str | None = None
    legs_done: int = 0
    legs_total: int = 0
    progress_note: str = ""
    log_size_B: int = 0
    log_mtime: str | None = None
    log_stage: str | None = None
    dyna: dict[str, Any] = field(default_factory=dict)
    last_dyna_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    restarts: list[dict[str, Any]] = field(default_factory=list)
    campaign_failed_leg: str | None = None
    notes: str = ""

    def to_csv_row(self) -> dict[str, str]:
        d = asdict(self)
        d["dyna_n_frames"] = str(self.dyna.get("n_frames", 0))
        d["temperature_last_K"] = str(self.dyna.get("temperature_last_K", ""))
        d["temperature_mean_K"] = str(self.dyna.get("temperature_mean_K", ""))
        d["energy_drift_kcal"] = str(self.dyna.get("total_energy_drift_kcal", ""))
        d["last_dyna_lines"] = " | ".join(self.last_dyna_lines[-3:])
        d["errors"] = " | ".join(self.errors[:3])
        d.pop("dyna", None)
        d.pop("restarts", None)
        return {k: str(v) for k, v in d.items()}


def _format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1000:
        return f"{n / 1e3:.1f} kB"
    return f"{n} B"


def _classify_health(
    *,
    status: str,
    errors: list[str],
    dyna: dict[str, Any],
    target_temp_K: float,
    restarts: list[dict[str, Any]],
) -> str:
    if status == "failed":
        return "BAD"
    if any(r.get("bad_coords") for r in restarts):
        return "BAD"
    if errors and status != "running":
        return "BAD"
    if status == "pending":
        return "—"
    n = int(dyna.get("n_frames") or 0)
    if n >= 2:
        t_last = float(dyna.get("temperature_last_K", 0))
        t_std = float(dyna.get("temperature_std_K", 0))
        drift = abs(float(dyna.get("total_energy_drift_kcal", 0)))
        step_delta = float(dyna.get("max_step_delta_kcal", 0))
        temp_off = abs(t_last - target_temp_K) if target_temp_K > 0 else 0
        if step_delta > 500 or drift > 2000:
            return "WARN"
        if temp_off > max(80, 0.35 * target_temp_K) and target_temp_K > 20:
            return "WARN"
        if t_std > 100 and n > 5:
            return "WARN"
    if status in {"partial", "running"}:
        return "OK"
    if status == "done":
        return "OK"
    return "OK"


def _classify_status(
    *,
    done: bool,
    handoff: bool,
    summary: dict[str, Any],
    log_text: str,
    log_path: Path,
    legs_done: int,
    legs_total: int,
) -> str:
    if done and handoff:
        return "done"
    jobs = summary.get("jobs") or []
    for job in jobs:
        if int(job.get("exit_code", 0)) != 0:
            return "failed"
    if grep_errors(log_text):
        if "liquid-density dynamics campaign failed" in log_text.lower() or "dynamics campaign failed" in log_text.lower():
            return "failed"
        if "segmentation fault" in log_text.lower():
            return "failed"
        if legs_done == 0 and log_path.is_file() and log_path.stat().st_size > 0:
            return "failed"
    if log_path.is_file() and log_path.stat().st_size > 0:
        if _LOG_ACTIVITY.search(log_text):
            if legs_done < legs_total:
                return "running"
            return "partial"
        return "started"
    if legs_done > 0 or summary:
        return "partial"
    return "pending"


def inspect_run(cfg: dict[str, Any], cell: RunCell) -> RunMonitor:
    tag = cell_run_tag(cell, cfg)
    paths = paths_for_run(cfg, cell)
    out_dir = paths["out_dir"]
    job_order = campaign_job_order(cfg)
    legs_total = len(job_order)
    legs_done_list = completed_leg_ids(out_dir, job_order) if out_dir.is_dir() else []
    legs_done = len(legs_done_list)
    active = active_leg_id(out_dir, job_order) if out_dir.is_dir() else job_order[0]

    log_path = out_dir / "stdout.log"
    log_text = read_text_tail(log_path) if log_path.is_file() else ""
    summary = load_campaign_summary(paths["campaign_summary"])
    _, failed_leg, stage_frac = campaign_leg_progress(summary, job_order)

    done = paths["done"].is_file()
    handoff = paths["final_handoff"].is_file()
    status = _classify_status(
        done=done,
        handoff=handoff,
        summary=summary,
        log_text=log_text,
        log_path=log_path,
        legs_done=legs_done,
        legs_total=legs_total,
    )

    dyna_rows = parse_dyna_lines(log_text)
    dyna = summarize_dyna(dyna_rows)
    errors = grep_errors(log_text)
    restarts = collect_restart_info(out_dir, tag=tag) if out_dir.is_dir() else []

    progress_note = f"{legs_done}/{legs_total} legs"
    if failed_leg:
        progress_note += f" (failed @ {failed_leg})"
    elif stage_frac is not None and active:
        progress_note += f"; {active} ~{stage_frac * 100:.0f}%"
    elif active and legs_done < legs_total:
        progress_note += f"; active={active}"

    last_dyna = [
        ln.strip()
        for ln in log_text.splitlines()
        if ln.strip().startswith("DYNA>")
    ][-5:]

    notes = ""
    if restarts:
        latest = restarts[-1]
        if latest.get("step") is not None:
            notes = f"restart step {latest['step']} ({latest['path']})"

    health = _classify_health(
        status=status,
        errors=errors,
        dyna=dyna,
        target_temp_K=float(cell.temperature),
        restarts=restarts,
    )

    log_mtime = None
    log_size = 0
    if log_path.is_file():
        st = log_path.stat()
        log_size = st.st_size
        log_mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(
            timespec="minutes"
        )

    return RunMonitor(
        run_tag=tag,
        solvent=cell.solvent,
        n_monomers=int(cell.n_monomers),
        temperature_target_K=float(cell.temperature),
        box_size_A=float(cell.box_size),
        out_dir=str(out_dir),
        status=status,
        health=health,
        active_leg=active,
        legs_done=legs_done,
        legs_total=legs_total,
        progress_note=progress_note,
        log_size_B=log_size,
        log_mtime=log_mtime,
        log_stage=last_log_stage(log_text),
        dyna=dyna,
        last_dyna_lines=last_dyna,
        errors=errors,
        restarts=restarts[-8:],
        campaign_failed_leg=failed_leg,
        notes=notes,
    )


def plot_dyna_png(
    rows: list[dict[str, float]],
    path: Path,
    *,
    title: str,
    target_temp_K: float | None = None,
) -> bool:
    if len(rows) < 2:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    time_ps = [r["time_ps"] for r in rows]
    etot = [r["total_energy_kcal"] for r in rows]
    epot = [r["potential_energy_kcal"] for r in rows]
    temp = [r["temperature_K"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=11)
    axes[0].plot(time_ps, etot, color="#005384", lw=0.9, label="E_total")
    axes[0].set_ylabel("E_total (kcal/mol)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_ps, epot, color="#004D3D", lw=0.9, label="E_pot")
    axes[1].set_ylabel("E_pot (kcal/mol)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_ps, temp, color="#B80000", lw=0.9, label="T")
    if target_temp_K is not None:
        axes[2].axhline(target_temp_K, color="#666", ls="--", lw=0.8, label="target T")
    axes[2].set_ylabel("T (K)")
    axes[2].set_xlabel("time (ps)")
    axes[2].grid(True, alpha=0.25)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def iter_monitors(cfg: dict[str, Any]) -> Iterator[RunMonitor]:
    for cell in iter_matrix_cells(cfg):
        yield inspect_run(cfg, cell)


def print_summary_table(
    monitors: list[RunMonitor],
    cfg: dict[str, Any],
    *,
    verbose: bool = False,
) -> None:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_liquid_density_dyn"))
    counts: dict[str, int] = {}
    for m in monitors:
        counts[m.status] = counts.get(m.status, 0) + 1

    print(f"Artifact root: {root}")
    print(
        f"Jobs: {len(monitors)}  |  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    print()
    hdr = (
        f"{'run_tag':<22} {'status':<8} {'health':<5} {'leg':<16} "
        f"{'progress':<22} {'T_last':>7} {'T_tgt':>6}  notes"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in monitors:
        t_last = m.dyna.get("temperature_last_K")
        t_last_s = f"{t_last:.0f}" if t_last is not None else "—"
        leg = (m.active_leg or "—")[:16]
        note = m.notes or (m.errors[0][:40] if m.errors else "")
        print(
            f"{m.run_tag:<22} {m.status:<8} {m.health:<5} {leg:<16} "
            f"{m.progress_note:<22} {t_last_s:>7} {m.temperature_target_K:>6.0f}  {note}"
        )
        if verbose and m.last_dyna_lines:
            for ln in m.last_dyna_lines[-2:]:
                print(f"    DYNA> {ln[6:].strip()}")


def print_run_detail(m: RunMonitor, *, cfg: dict[str, Any] | None = None) -> None:
    print(f"=== {m.run_tag} ===")
    print(
        f"{m.solvent} N={m.n_monomers} T_target={m.temperature_target_K:.0f} K "
        f"L={m.box_size_A:.0f} Å"
    )
    print(f"out_dir: {m.out_dir}")
    print(f"status={m.status}  health={m.health}  {m.progress_note}")
    if m.log_mtime:
        print(f"stdout.log: {_format_bytes(m.log_size_B)}  mtime={m.log_mtime} UTC")
    if m.log_stage:
        print(f"last log stage marker: {m.log_stage}")
    if m.campaign_failed_leg:
        print(f"campaign failed leg: {m.campaign_failed_leg}")

    if m.dyna.get("n_frames"):
        print("\nDYNA summary (stdout.log):")
        for key in (
            "n_frames",
            "step_last",
            "time_ps_last",
            "temperature_mean_K",
            "temperature_last_K",
            "total_energy_drift_kcal",
            "total_energy_std_kcal",
            "max_step_delta_kcal",
        ):
            if key in m.dyna:
                val = m.dyna[key]
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")

    if m.last_dyna_lines:
        print("\nLast DYNA> lines:")
        for ln in m.last_dyna_lines:
            print(f"  {ln.rstrip()}")

    if m.restarts:
        print("\nRecent restart files:")
        for r in m.restarts[-6:]:
            step = r.get("step")
            step_s = str(step) if step is not None else "?"
            bad = " BAD_COORDS" if r.get("bad_coords") else ""
            print(f"  {r['path']}: step={step_s}  {_format_bytes(r['size_B'])}{bad}")

    if m.errors:
        print("\nLog errors / markers:")
        for line in m.errors:
            print(f"  {line}")

    # Leg handoffs
    if cfg is not None:
        out = Path(m.out_dir)
        order = campaign_job_order(cfg)
        print("\nLeg handoffs:")
        for jid in order:
            hp = leg_handoff_path(out, jid)
            mark = "✓" if hp.is_file() else "·"
            ss = out / jid / "stage_summary.json"
            extra = ""
            if ss.is_file():
                try:
                    payload = json.loads(ss.read_text(encoding="utf-8"))
                    ec = payload.get("exit_code")
                    stages = payload.get("stages") or []
                    if stages:
                        st = stages[-1]
                        extra = (
                            f" exit={ec} {st.get('stage')} "
                            f"{st.get('ps_completed', 0):.2f}/"
                            f"{st.get('ps_requested', 0):.2f} ps"
                        )
                except (json.JSONDecodeError, OSError, TypeError):
                    pass
            print(f"  [{mark}] {jid}{extra}")
