"""Generate md-system campaign YAML for one PBC solvent burst matrix cell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from cleanup_strategy import (
    jaxmd_job_flags,
    pretreat_job_flags,
    pycharmm_job_flags,
    resolve_cleanup_strategy,
)


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return workflow_root().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (workflow_root() / "config.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(raw: str) -> Path:
    if raw == "${MMML_CKPT}":
        env = os.environ.get("MMML_CKPT", "").strip()
        if not env:
            raise RuntimeError(
                "MMML_CKPT is not set (config checkpoint: ${MMML_CKPT}). "
                "Export your DES dimers PhysNet checkpoint directory before running."
            )
        path = Path(env).expanduser().resolve()
    else:
        path = Path(os.path.expandvars(raw)).expanduser().resolve()
    validate_checkpoint(path)
    return path


def validate_checkpoint(path: Path) -> None:
    text = str(path)
    placeholders = ("/path/to", "your/checkpoint", "REPLACE_ME")
    if any(p in text for p in placeholders):
        raise RuntimeError(
            f"Checkpoint path looks like a placeholder: {path}\n"
            "Set a real directory, e.g.\n"
            "  export MMML_CKPT=/mmhome/boittier/home/mmml/examples/ckpts_json/DESdimers_params.json"
        )
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")


def solvent_slug(solvent: str) -> str:
    return str(solvent).strip().upper()


@dataclass(frozen=True)
class RunCell:
    """One matrix point: solvent, size, temperature, box."""

    solvent: str
    n_monomers: int
    temperature: float
    box_size: float


def matrix_temperatures(cfg: dict[str, Any]) -> list[float]:
    if cfg.get("temperatures"):
        return [float(x) for x in cfg["temperatures"]]
    return [float(cfg.get("temperature", 300.0))]


def matrix_box_sizes(cfg: dict[str, Any]) -> list[float]:
    if cfg.get("box_sizes"):
        return [float(x) for x in cfg["box_sizes"]]
    return [float(cfg.get("box_size", 32.0))]


def iter_matrix_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    """Cartesian product of solvents × cluster_sizes × temperatures × box_sizes."""
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", [])]
    sizes = [int(n) for n in cfg.get("cluster_sizes", [])]
    for sol in solvents:
        for n in sizes:
            for temp in matrix_temperatures(cfg):
                for box in matrix_box_sizes(cfg):
                    yield RunCell(solvent=sol, n_monomers=n, temperature=temp, box_size=box)


def cell_run_tag(cell: RunCell) -> str:
    """Filesystem-safe run id, e.g. ``dcm_10_t300_l32``."""
    sol = solvent_slug(cell.solvent).lower()
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    return f"{sol}_{int(cell.n_monomers)}_t{t}_l{box}"


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_solvent_burst"))
    return (root / cell_run_tag(cell)).resolve()


def run_seed(cell: RunCell, *, seed_base: int = 123456) -> int:
    solvent_off = sum(ord(c) for c in solvent_slug(cell.solvent)) % 1000
    temp_off = int(round(cell.temperature)) % 100
    box_off = int(round(cell.box_size)) % 100
    return (
        int(seed_base)
        + int(cell.n_monomers) * 10000
        + solvent_off
        + temp_off * 17
        + box_off * 131
    )


def leg_output_dir(cell_root: Path, job_id: str) -> str:
    """Absolute path for one campaign leg (never repo-root ``results/``)."""
    return str((cell_root / job_id).resolve())


def _attach_leg_output_dir(job: dict[str, Any], cell_root: Path, job_id: str) -> dict[str, Any]:
    return {**job, "output_dir": leg_output_dir(cell_root, job_id)}


def campaign_job_order(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return ordered job IDs for the burst campaign (11 legs by default)."""
    cfg = cfg or {}
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    n_equi = int(cfg.get("pycharmm_equi_legs", 5))
    if n_equi != n_bursts:
        raise ValueError(
            f"pycharmm_equi_legs ({n_equi}) must equal jaxmd_bursts ({n_bursts})"
        )
    order = ["pycharmm_init", "pycharmm_equi_00"]
    for i in range(1, n_bursts + 1):
        order.append(f"jaxmd_burst_{i:02d}")
        if i < n_bursts:
            order.append(f"pycharmm_equi_{i:02d}")
    return order


def build_campaign(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    """Build full campaign dict for one matrix cell."""
    comp = composition_string(cell)
    tag = cell_run_tag(cell)
    seed = run_seed(cell, seed_base=int(cfg.get("seed_base", 123456)))
    burst_ps = float(cfg.get("jaxmd_burst_ps", 200.0))
    equi_ps = float(cfg.get("pycharmm_equi_ps", 10.0))
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    optional_sizes = {int(x) for x in (cfg.get("optional_sizes") or [])}
    cell_root = run_output_dir(cfg, cell)
    strategy = resolve_cleanup_strategy(cfg)

    defaults: dict[str, Any] = {
        "composition": comp,
        "checkpoint": str(cfg["checkpoint"]),
        "box_size": float(cell.box_size),
        "output_root": str(cell_root),
        "packmol_cache_dir": str(cell_root / ".packmol_cache"),
        "spacing": float(cfg.get("spacing", 5.0)),
        "packmol_tolerance": float(cfg.get("packmol_tolerance", 1.0)),
        "dt_fs": float(cfg.get("dt_fs", 0.25)),
        "temperature": float(cell.temperature),
        "pressure": float(cfg.get("pressure", 1.0)),
        "seed": seed,
        "mm_switch_on": float(cfg.get("mm_switch_on", 8.0)),
        "mm_switch_width": float(cfg.get("mm_switch_width", 5.0)),
        "ml_switch_width": float(cfg.get("ml_switch_width", 1.5)),
        "ml_gpu_count": int(cfg.get("ml_gpu_count", 1)),
        "ml_batch_size": int(cfg.get("ml_batch_size", 1024)),
        "handoff_write_res": True,
        "continue_velocities": True,
        "cleanup_strategy_name": strategy.name,
    }

    repair = pycharmm_job_flags(strategy)
    pretreat = pretreat_job_flags(strategy)
    jaxmd_extra = jaxmd_job_flags(strategy)

    init_desc = f"{comp} T={cell.temperature:.0f}K L={cell.box_size:.0f}Å init"
    if pretreat:
        init_desc += " (CHARMM MM pretreat + MLpot mini + heat)"
    else:
        init_desc += " (MLpot mini + heat)"

    runs: dict[str, Any] = {
        "pycharmm_init": _attach_leg_output_dir(
            {
                "description": init_desc,
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stages": "mini,heat",
                "ps_heat": float(cfg.get("ps_heat", 30.0)),
                "n_heat_segments": int(cfg.get("n_heat_segments", 8)),
                "heat_firstt": float(cfg.get("heat_firstt", 10.0)),
                "heat_finalt": float(cell.temperature),
                "heat_thermostat": str(cfg.get("heat_thermostat", "hoover")),
                **repair,
                **pretreat,
            },
            cell_root,
            "pycharmm_init",
        ),
        "pycharmm_equi_00": _attach_leg_output_dir(
            {
                "description": f"{comp} first NPT equil ({equi_ps} ps)",
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stage": "equi",
                "ps_equi": equi_ps,
                "depends_on": "pycharmm_init",
                **repair,
                **pretreat,
            },
            cell_root,
            "pycharmm_equi_00",
        ),
    }

    prev = "pycharmm_equi_00"
    for i in range(1, n_bursts + 1):
        burst_id = f"jaxmd_burst_{i:02d}"
        runs[burst_id] = _attach_leg_output_dir(
            {
                "description": f"{comp} JAX-MD burst {i}/{n_bursts} ({burst_ps} ps)",
                "backend": "jaxmd",
                "setup": "pbc_nvt",
                "ps": burst_ps,
                "depends_on": prev,
                **jaxmd_extra,
            },
            cell_root,
            burst_id,
        )
        if i < n_bursts:
            equi_id = f"pycharmm_equi_{i:02d}"
            equi_job: dict[str, Any] = _attach_leg_output_dir(
                {
                    "description": f"{comp} NPT equil after burst {i} ({equi_ps} ps)",
                    "backend": "pycharmm",
                    "setup": "pbc_npt",
                    "md_stage": "equi",
                    "ps_equi": equi_ps,
                    "depends_on": burst_id,
                    **repair,
                    **pretreat,
                },
                cell_root,
                equi_id,
            )
            if cell.n_monomers in optional_sizes:
                equi_job["optional"] = True
            runs[equi_id] = equi_job
            prev = equi_id
        else:
            if cell.n_monomers in optional_sizes:
                runs[burst_id]["optional"] = True
            prev = burst_id

    return {
        "defaults": defaults,
        "campaign_output": str(cell_root),
        "runs": runs,
    }


def write_campaign_yaml(
    cfg: dict[str, Any],
    cell: RunCell,
    *,
    out_dir: Path | None = None,
) -> Path:
    campaign = build_campaign(cfg, cell)
    root = out_dir or run_output_dir(cfg, cell)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(campaign, f, sort_keys=False, default_flow_style=False)
    return path


def build_md_system_campaign_argv(
    cfg: dict[str, Any],
    cell: RunCell,
    *,
    out_dir: Path | None = None,
) -> list[str]:
    root = out_dir or run_output_dir(cfg, cell)
    campaign_path = write_campaign_yaml(cfg, cell, out_dir=root)
    return [
        "--config",
        str(campaign_path),
        "--run-all",
        "--resume-campaign",
        "--campaign-output-dir",
        str(root),
    ]


def matrix_job_count(cfg: dict[str, Any]) -> int:
    return sum(1 for _ in iter_matrix_cells(cfg))


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    """Concurrent Snakemake/Slurm jobs (gpu pool == charmm_slot pool)."""
    cap = matrix_job_count(cfg)
    requested = int(cfg.get("slurm_max_concurrent", cap))
    return max(1, min(requested, cap))


def total_jaxmd_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("jaxmd_burst_ps", 200.0)) * int(cfg.get("jaxmd_bursts", 5))


def total_pycharmm_equi_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("pycharmm_equi_ps", 10.0)) * int(cfg.get("pycharmm_equi_legs", 5))


def paths_for_run(cfg: dict[str, Any], cell: RunCell) -> dict[str, Path]:
    out = run_output_dir(cfg, cell)
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "final_handoff": out / f"jaxmd_burst_{n_bursts:02d}" / "handoff" / "state.npz",
        "done": out / "done.txt",
    }


def cell_from_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    """Resolve a run tag back to a matrix cell (must exist in config matrix)."""
    for cell in iter_matrix_cells(cfg):
        if cell_run_tag(cell) == tag:
            return cell
    raise KeyError(f"run tag {tag!r} not in config matrix")


def cell_from_cli(
    cfg: dict[str, Any],
    solvent: str,
    n_monomers: int,
    *,
    temperature: float | None = None,
    box_size: float | None = None,
) -> RunCell:
    """Pick matrix cell matching solvent/N (and optional T, L)."""
    sol = solvent_slug(solvent)
    n = int(n_monomers)
    temps = matrix_temperatures(cfg) if temperature is None else [float(temperature)]
    boxes = matrix_box_sizes(cfg) if box_size is None else [float(box_size)]
    if len(temps) != 1 or len(boxes) != 1:
        raise ValueError("Specify --temperature and --box-size when matrix lists have multiple values")
    cell = RunCell(solvent=sol, n_monomers=n, temperature=temps[0], box_size=boxes[0])
    valid_tags = {cell_run_tag(c) for c in iter_matrix_cells(cfg)}
    if cell_run_tag(cell) not in valid_tags:
        raise ValueError(f"{cell} not in config matrix (valid tags: {len(valid_tags)})")
    return cell


# Backward-compatible helpers (single-T, single-L matrix)
def run_tag(solvent: str, n_monomers: int) -> str:
    cfg = load_config()
    cell = cell_from_cli(cfg, solvent, n_monomers)
    return cell_run_tag(cell)


def composition_string_legacy(solvent: str, n_monomers: int) -> str:
    return composition_string(cell_from_cli(load_config(), solvent, n_monomers))
