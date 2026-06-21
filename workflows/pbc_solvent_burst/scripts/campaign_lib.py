"""Generate md-system campaign YAML for one PBC solvent burst matrix cell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from bulk_density import matrix_cluster_sizes_for_cell, matrix_uses_bulk_density
from cleanup_strategy import (
    dense_cell_mlpot_overrides,
    jaxmd_job_flags,
    pretreat_job_flags,
    pycharmm_job_flags,
    resolve_cleanup_strategy,
    resolve_pycharmm_heat_thermostat,
)


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return workflow_root().parents[1]


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else (workflow_root() / "config.yaml")
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
    """Cartesian product of solvents × N × temperatures × box_sizes.

    ``N`` comes from ``cluster_sizes`` (legacy) or from ``bulk_density_fractions``
    × bulk liquid density at 298 K per solvent and box (see ``bulk_density.py``).
    """
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", [])]
    if matrix_uses_bulk_density(cfg):
        if cfg.get("cluster_sizes"):
            raise ValueError(
                "Set either bulk_density_fractions or cluster_sizes, not both."
            )
    elif not cfg.get("cluster_sizes"):
        raise ValueError("Matrix requires cluster_sizes or bulk_density_fractions.")
    skip = {str(t).strip() for t in (cfg.get("exclude_run_tags") or [])}
    seen_tags: set[str] = set()
    for sol in solvents:
        for box in matrix_box_sizes(cfg):
            sizes = matrix_cluster_sizes_for_cell(cfg, solvent=sol, box_size=box)
            for n in sizes:
                for temp in matrix_temperatures(cfg):
                    cell = RunCell(
                        solvent=sol,
                        n_monomers=n,
                        temperature=temp,
                        box_size=box,
                    )
                    tag = cell_run_tag(cell, cfg)
                    if tag in skip or tag in seen_tags:
                        continue
                    seen_tags.add(tag)
                    yield cell


def matrix_tag_includes_TL(cfg: dict[str, Any]) -> bool:
    """True when multiple temperatures or box sizes require T/L in the run tag."""
    return len(matrix_temperatures(cfg)) > 1 or len(matrix_box_sizes(cfg)) > 1


def cell_run_tag(cell: RunCell, cfg: dict[str, Any] | None = None) -> str:
    """Filesystem-safe run id.

    Single T and box in the matrix: ``dcm_10`` (backward compatible).
    Sweeps: ``dcm_10_t300_l32``.
    """
    sol = solvent_slug(cell.solvent).lower()
    base = f"{sol}_{int(cell.n_monomers)}"
    if cfg is not None and not matrix_tag_includes_TL(cfg):
        return base
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    return f"{base}_t{t}_l{box}"


def cell_run_tag_long(cell: RunCell) -> str:
    """Always include T/L suffix (for alias lookup)."""
    sol = solvent_slug(cell.solvent).lower()
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    return f"{sol}_{int(cell.n_monomers)}_t{t}_l{box}"


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_solvent_burst"))
    return (root / cell_run_tag(cell, cfg)).resolve()


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


def cell_is_optional(cell: RunCell, cfg: dict[str, Any]) -> bool:
    """True when the last JAX burst (or equi before it) may fail without aborting."""
    if int(cell.n_monomers) in {int(x) for x in (cfg.get("optional_sizes") or [])}:
        return True
    if not matrix_uses_bulk_density(cfg):
        return False
    from bulk_density import n_monomers_at_bulk_density

    min_n = int(cfg.get("bulk_density_n_min", 1))
    max_raw = cfg.get("bulk_density_n_max")
    max_n = int(max_raw) if max_raw is not None else None
    for frac in cfg.get("optional_bulk_fractions") or [1.0]:
        n = n_monomers_at_bulk_density(
            cell.solvent,
            cell.box_size,
            float(frac),
            min_n=min_n,
            max_n=max_n,
        )
        if n == int(cell.n_monomers):
            return True
    return False


def build_campaign(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    """Build full campaign dict for one matrix cell."""
    comp = composition_string(cell)
    tag = cell_run_tag(cell, cfg)
    seed = run_seed(cell, seed_base=int(cfg.get("seed_base", 123456)))
    burst_ps = float(cfg.get("jaxmd_burst_ps", 200.0))
    equi_ps = float(cfg.get("pycharmm_equi_ps", 10.0))
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    optional_cell = cell_is_optional(cell, cfg)
    cell_root = run_output_dir(cfg, cell)
    strategy = resolve_cleanup_strategy(cfg)
    heat_thermostat = resolve_pycharmm_heat_thermostat(cfg, strategy)

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
        "ml_batch_size": int(
            dense_cell_mlpot_overrides(cell, cfg).get(
                "ml_batch_size", cfg.get("ml_batch_size", 1024)
            )
        ),
        "handoff_write_res": True,
        "continue_velocities": True,
        "cleanup_strategy_name": strategy.name,
    }

    repair = pycharmm_job_flags(strategy)
    repair.update(dense_cell_mlpot_overrides(cell, cfg))
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
                "n_heat_segments": int(
                    dense_cell_mlpot_overrides(cell, cfg).get(
                        "n_heat_segments", cfg.get("n_heat_segments", 8)
                    )
                ),
                "heat_firstt": float(cfg.get("heat_firstt", 10.0)),
                "heat_finalt": float(cell.temperature),
                "heat_thermostat": heat_thermostat,
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
                },
                cell_root,
                equi_id,
            )
            if optional_cell:
                equi_job["optional"] = True
            runs[equi_id] = equi_job
            prev = equi_id
        else:
            if optional_cell:
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


def _slurm_node_list(cfg: dict[str, Any], key: str) -> list[str]:
    raw = cfg.get(key)
    if not raw:
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def slurm_gpu_nodes_fast(cfg: dict[str, Any]) -> list[str]:
    nodes = _slurm_node_list(cfg, "slurm_gpu_nodes_fast")
    if nodes:
        return nodes
    return _slurm_node_list(cfg, "slurm_gpu_nodes")


def slurm_gpu_nodes_slow(cfg: dict[str, Any]) -> list[str]:
    return _slurm_node_list(cfg, "slurm_gpu_nodes_slow")


def slurm_tier_enabled(cfg: dict[str, Any]) -> bool:
    return bool(slurm_gpu_nodes_slow(cfg)) and bool(slurm_gpu_nodes_fast(cfg))


def slurm_small_cluster_max_n(cfg: dict[str, Any]) -> int:
    return int(cfg.get("slurm_small_cluster_max_n", 30))


def cell_slurm_tier(cell: RunCell, cfg: dict[str, Any]) -> str:
    """``slow`` for small N (3080 pool); ``fast`` for large clusters."""
    if not slurm_tier_enabled(cfg):
        return "fast"
    if int(cell.n_monomers) <= slurm_small_cluster_max_n(cfg):
        return "slow"
    return "fast"


def slurm_nodelist_for_tier(cfg: dict[str, Any], tier: str) -> str:
    explicit = str(cfg.get("slurm_nodelist", "")).strip()
    if explicit and not slurm_tier_enabled(cfg):
        return explicit
    if tier == "slow":
        nodes = slurm_gpu_nodes_slow(cfg)
    else:
        nodes = slurm_gpu_nodes_fast(cfg)
    return ",".join(nodes)


def slurm_tier_gpu_pool(cfg: dict[str, Any], tier: str) -> int:
    tier_key = f"slurm_max_concurrent_{tier}"
    if tier_key in cfg:
        return max(1, int(cfg[tier_key]))
    if tier == "fast":
        return slurm_max_concurrent(cfg)
    # Default slow pool: ~2 GPUs per listed node (3080 nodes are gpu:2).
    return max(1, len(slurm_gpu_nodes_slow(cfg)) * 2)


def slurm_tier_resource_pools(cfg: dict[str, Any]) -> dict[str, int]:
    """Snakemake ``--resources`` pools for tiered or flat scheduling."""
    if not slurm_tier_enabled(cfg):
        n = slurm_max_concurrent(cfg)
        return {"gpu_fast": n, "gpu_slow": 0, "charmm_slot": n}
    fast = slurm_tier_gpu_pool(cfg, "fast")
    slow = slurm_tier_gpu_pool(cfg, "slow")
    return {"gpu_fast": fast, "gpu_slow": slow, "charmm_slot": fast + slow}


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    """Max Snakemake jobsteps in flight (-j)."""
    pools = slurm_tier_resource_pools(cfg)
    return int(pools["gpu_fast"]) + int(pools["gpu_slow"])


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    """Space-separated ``NAME=N`` for ``snakemake --resources``."""
    pools = slurm_tier_resource_pools(cfg)
    return " ".join(f"{key}={value}" for key, value in pools.items())


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
    """Resolve a run tag back to a matrix cell."""
    by_tag = {cell_run_tag(c, cfg): c for c in iter_matrix_cells(cfg)}
    if tag in by_tag:
        return by_tag[tag]
    # Accept long-form tags when matrix uses short tags (single T/L).
    if not matrix_tag_includes_TL(cfg):
        for cell in iter_matrix_cells(cfg):
            if cell_run_tag_long(cell) == tag:
                return cell
    raise KeyError(
        f"run tag {tag!r} not in config matrix "
        f"(examples: {', '.join(list(by_tag.keys())[:3])}…)"
    )


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
    valid_tags = {cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)}
    if cell_run_tag(cell, cfg) not in valid_tags:
        raise ValueError(f"{cell} not in config matrix (valid tags: {len(valid_tags)})")
    return cell


# Backward-compatible helpers (single-T, single-L matrix)
def run_tag(solvent: str, n_monomers: int) -> str:
    cfg = load_config()
    cell = cell_from_cli(cfg, solvent, n_monomers)
    return cell_run_tag(cell, cfg)


def composition_string_legacy(solvent: str, n_monomers: int) -> str:
    return composition_string(cell_from_cli(load_config(), solvent, n_monomers))
