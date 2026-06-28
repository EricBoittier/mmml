"""Generate md-system campaigns for liquid-density PBC cluster dynamics."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

_BURST_SCRIPTS = Path(__file__).resolve().parents[1].parent / "pbc_solvent_burst" / "scripts"
if str(_BURST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BURST_SCRIPTS))

from bulk_density import (  # noqa: E402
    BULK_SOLVENTS,
    matrix_cluster_sizes_for_cell,
    matrix_uses_bulk_density,
    n_monomers_at_bulk_density,
)
from cleanup_strategy import (  # noqa: E402
    dense_cell_mlpot_overrides,
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
            raise RuntimeError("MMML_CKPT is not set (config checkpoint: ${MMML_CKPT})")
        path = Path(env).expanduser().resolve()
    else:
        path = Path(os.path.expandvars(raw)).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")
    return path


def checkpoint_path_for_yaml(raw: str) -> str:
    """Resolve ``${MMML_CKPT}`` for campaign YAML; leave explicit paths as strings."""
    if str(raw).strip() == "${MMML_CKPT}":
        return str(resolve_checkpoint(str(raw)))
    return str(os.path.expandvars(str(raw)))


def solvent_slug(solvent: str) -> str:
    return str(solvent).strip().upper()


@dataclass(frozen=True)
class RunCell:
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


def matrix_density_fractions(cfg: dict[str, Any]) -> list[float]:
    raw = cfg.get("bulk_density_fractions")
    if not raw:
        return []
    return [float(x) for x in raw]


def cell_bulk_density_fraction(cell: RunCell, cfg: dict[str, Any]) -> float | None:
    if not matrix_uses_bulk_density(cfg):
        raw = cfg.get("bulk_density_fraction")
        return float(raw) if raw is not None else None
    min_n = int(cfg.get("bulk_density_n_min", 1))
    max_raw = cfg.get("bulk_density_n_max")
    max_n = int(max_raw) if max_raw is not None else None
    for frac in matrix_density_fractions(cfg):
        n = n_monomers_at_bulk_density(
            cell.solvent,
            cell.box_size,
            frac,
            min_n=min_n,
            max_n=max_n,
        )
        if n == int(cell.n_monomers):
            return float(frac)
    bulk_n = n_monomers_at_bulk_density(cell.solvent, cell.box_size, 1.0, min_n=1)
    return float(cell.n_monomers) / float(max(1, bulk_n))


def iter_matrix_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", [])]
    if matrix_uses_bulk_density(cfg):
        if cfg.get("cluster_sizes"):
            raise ValueError("Set either bulk_density_fractions or cluster_sizes, not both.")
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
    return len(matrix_temperatures(cfg)) > 1 or len(matrix_box_sizes(cfg)) > 1


def cell_run_tag(cell: RunCell, cfg: dict[str, Any] | None = None) -> str:
    sol = solvent_slug(cell.solvent).lower()
    base = f"{sol}_{int(cell.n_monomers)}"
    if cfg is not None and not matrix_tag_includes_TL(cfg):
        return base
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    return f"{base}_t{t}_l{box}"


def cell_run_tag_long(cell: RunCell) -> str:
    sol = solvent_slug(cell.solvent).lower()
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    return f"{sol}_{int(cell.n_monomers)}_t{t}_l{box}"


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_liquid_density_dyn"))
    return (root / cell_run_tag(cell, cfg)).resolve()


def run_seed(cell: RunCell, *, seed_base: int = 4242) -> int:
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
    return str((cell_root / job_id).resolve())


def _attach_leg_output_dir(job: dict[str, Any], cell_root: Path, job_id: str) -> dict[str, Any]:
    return {**job, "output_dir": leg_output_dir(cell_root, job_id)}


def campaign_job_order(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or {}
    n_equi = int(cfg.get("pycharmm_equi_legs", 5))
    n_prod = int(cfg.get("pycharmm_prod_legs", 10))
    order = ["pycharmm_init"]
    order.extend(f"pycharmm_equi_{i:02d}" for i in range(1, n_equi + 1))
    order.extend(f"pycharmm_prod_{i:02d}" for i in range(1, n_prod + 1))
    return order


def cell_is_optional(cell: RunCell, cfg: dict[str, Any]) -> bool:
    if int(cell.n_monomers) in {int(x) for x in (cfg.get("optional_sizes") or [])}:
        return True
    if not matrix_uses_bulk_density(cfg):
        return False
    frac = cell_bulk_density_fraction(cell, cfg)
    if frac is None:
        return False
    optional = [float(x) for x in (cfg.get("optional_bulk_fractions") or [1.0])]
    return any(abs(frac - o) < 1e-6 for o in optional)


def _liquid_prep_defaults(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    sol = solvent_slug(cell.solvent)
    rho = BULK_SOLVENTS[sol]["rho_g_cm3"]
    frac = cell_bulk_density_fraction(cell, cfg)
    flags: dict[str, Any] = {
        "liquid_prep": bool(cfg.get("liquid_prep", True)),
        "density_prep_ladder": bool(cfg.get("density_prep_ladder", True)),
        "density_prep_ladder_max_rounds": int(cfg.get("density_prep_ladder_max_rounds", 5)),
        "density_prep_lattice_abnr_steps": int(cfg.get("density_prep_lattice_abnr_steps", 200)),
        "mini_lattice_abnr_steps": int(cfg.get("mini_lattice_abnr_steps", 200)),
        "mini_lattice_abnr_allow_fixed_box": True,
        "mini_box_equil_ps": float(cfg.get("mini_box_equil_ps", 2.0)),
        "mini_box_equil_allow_fixed_box": True,
        "mini_box_equil_fixed_nvt": True,
        "mc_density_equalize": bool(cfg.get("mc_density_equalize", True)),
        "mc_density_steps": int(cfg.get("mc_density_steps", 80)),
        "min_intermonomer_atom_distance": float(cfg.get("min_intermonomer_atom_distance", 1.0)),
        "target_density_g_cm3": float(rho),
        "max_grms_before_dyn": float(cfg.get("max_grms_before_dyn", 50.0)),
    }
    if frac is not None:
        flags["bulk_density_fraction"] = float(frac)
    return flags


def build_campaign(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    comp = composition_string(cell)
    cell_root = run_output_dir(cfg, cell)
    strategy = resolve_cleanup_strategy(cfg)
    heat_thermostat = resolve_pycharmm_heat_thermostat(cfg, strategy)
    repair = pycharmm_job_flags(strategy)
    repair.update(dense_cell_mlpot_overrides(cell, cfg))
    pretreat = pretreat_job_flags(strategy)
    liquid = _liquid_prep_defaults(cfg, cell)
    equi_ps = float(cfg.get("pycharmm_equi_ps", 20.0))
    prod_ps = float(cfg.get("pycharmm_prod_ps", 50.0))
    n_equi = int(cfg.get("pycharmm_equi_legs", 5))
    n_prod = int(cfg.get("pycharmm_prod_legs", 10))
    prod_setup = str(cfg.get("prod_ensemble", "pbc_npt"))
    optional_cell = cell_is_optional(cell, cfg)

    defaults: dict[str, Any] = {
        "composition": comp,
        "checkpoint": checkpoint_path_for_yaml(str(cfg["checkpoint"])),
        "box_size": float(cell.box_size),
        "output_root": str(cell_root),
        "packmol_cache_dir": str(cell_root / ".packmol_cache"),
        "spacing": float(cfg.get("spacing", 4.0)),
        "packmol_tolerance": float(cfg.get("packmol_tolerance", 1.5)),
        "dt_fs": float(cfg.get("dt_fs", 0.25)),
        "temperature": float(cell.temperature),
        "pressure": float(cfg.get("pressure", 1.0)),
        "seed": run_seed(cell, seed_base=int(cfg.get("seed_base", 4242))),
        "mm_switch_on": float(cfg.get("mm_switch_on", 8.0)),
        "mm_switch_width": float(cfg.get("mm_switch_width", 5.0)),
        "ml_switch_width": float(cfg.get("ml_switch_width", 1.5)),
        "ml_gpu_count": int(cfg.get("ml_gpu_count", 1)),
        "ml_batch_size": int(
            dense_cell_mlpot_overrides(cell, cfg).get(
                "ml_batch_size", cfg.get("ml_batch_size", 2048)
            )
        ),
        "handoff_write_res": True,
        "continue_velocities": True,
        "cleanup_strategy_name": strategy.name,
        **liquid,
    }
    if bool(cfg.get("mlpot_profile", False)):
        defaults["mlpot_profile"] = True

    runs: dict[str, Any] = {
        "pycharmm_init": _attach_leg_output_dir(
            {
                "description": (
                    f"{comp} ρ≈liquid T={cell.temperature:.0f}K L={cell.box_size:.0f}Å "
                    "prep ladder + mini + heat"
                ),
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stages": "mini,heat",
                "ps_heat": float(cfg.get("ps_heat", 10.0)),
                "n_heat_segments": int(cfg.get("n_heat_segments", 5)),
                "heat_firstt": float(cfg.get("heat_firstt", 50.0)),
                "heat_finalt": float(cell.temperature),
                "heat_thermostat": heat_thermostat,
                **repair,
                **pretreat,
            },
            cell_root,
            "pycharmm_init",
        ),
    }

    prev = "pycharmm_init"
    for i in range(1, n_equi + 1):
        jid = f"pycharmm_equi_{i:02d}"
        job = _attach_leg_output_dir(
            {
                "description": f"{comp} NPT equil segment {i}/{n_equi} ({equi_ps} ps)",
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stage": "equi",
                "ps_equi": equi_ps,
                "depends_on": prev,
                **repair,
            },
            cell_root,
            jid,
        )
        if optional_cell and i == n_equi:
            job["optional"] = True
        runs[jid] = job
        prev = jid

    for i in range(1, n_prod + 1):
        jid = f"pycharmm_prod_{i:02d}"
        job = _attach_leg_output_dir(
            {
                "description": f"{comp} {prod_setup} production {i}/{n_prod} ({prod_ps} ps)",
                "backend": "pycharmm",
                "setup": prod_setup,
                "md_stage": "prod",
                "ps_prod": prod_ps,
                "depends_on": prev,
                **repair,
            },
            cell_root,
            jid,
        )
        if optional_cell and i == n_prod:
            job["optional"] = True
        runs[jid] = job
        prev = jid

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
        "--resume",
        "--campaign-output-dir",
        str(root),
    ]


def matrix_job_count(cfg: dict[str, Any]) -> int:
    return sum(1 for _ in iter_matrix_cells(cfg))


def paths_for_run(cfg: dict[str, Any], cell: RunCell) -> dict[str, Path]:
    out = run_output_dir(cfg, cell)
    n_prod = int(cfg.get("pycharmm_prod_legs", 10))
    final_id = f"pycharmm_prod_{n_prod:02d}"
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "prep_ladder": out / "prep_ladder",
        "final_handoff": out / final_id / "handoff" / "state.npz",
        "final_dcd": out / final_id / "dcd" / "traj.dcd",
        "done": out / "done.txt",
    }


def cell_from_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    by_tag = {cell_run_tag(c, cfg): c for c in iter_matrix_cells(cfg)}
    if tag in by_tag:
        return by_tag[tag]
    if not matrix_tag_includes_TL(cfg):
        for cell in iter_matrix_cells(cfg):
            if cell_run_tag_long(cell) == tag:
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
    temps = matrix_temperatures(cfg)
    boxes = matrix_box_sizes(cfg)
    t = float(temperature if temperature is not None else temps[0])
    box = float(box_size if box_size is not None else boxes[0])
    target = RunCell(
        solvent=str(solvent).strip().upper(),
        n_monomers=int(n_monomers),
        temperature=t,
        box_size=box,
    )
    for cell in iter_matrix_cells(cfg):
        if (
            solvent_slug(cell.solvent) == solvent_slug(target.solvent)
            and int(cell.n_monomers) == int(target.n_monomers)
            and abs(cell.temperature - target.temperature) < 0.5
            and abs(cell.box_size - target.box_size) < 0.5
        ):
            return cell
    raise KeyError(
        f"No matrix cell for {target.solvent}:{target.n_monomers} T={t} L={box}"
    )


# Slurm helpers (same contract as pbc_solvent_burst)
def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
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
    if not slurm_tier_enabled(cfg):
        return "fast"
    if int(cell.n_monomers) <= slurm_small_cluster_max_n(cfg):
        return "slow"
    return "fast"


def slurm_nodelist_for_tier(cfg: dict[str, Any], tier: str) -> str:
    explicit = str(cfg.get("slurm_nodelist", "")).strip()
    if explicit and not slurm_tier_enabled(cfg):
        return explicit
    nodes = slurm_gpu_nodes_slow(cfg) if tier == "slow" else slurm_gpu_nodes_fast(cfg)
    return ",".join(nodes)


def slurm_tier_gpu_pool(cfg: dict[str, Any], tier: str) -> int:
    tier_key = f"slurm_max_concurrent_{tier}"
    if tier_key in cfg:
        return max(1, int(cfg[tier_key]))
    if tier == "fast":
        return slurm_max_concurrent(cfg)
    return max(1, len(slurm_gpu_nodes_slow(cfg)) * 2)


def scheduler_mode(cfg: dict[str, Any]) -> str:
    """``gpu`` (default) or ``cpu`` (Slurm CPU partition, no GPU resources)."""
    return str(cfg.get("scheduler", "gpu")).strip().lower()


def cluster_name(cfg: dict[str, Any]) -> str:
    """Optional cluster label (e.g. ``pc-bach``) for job-environment prologs."""
    return str(cfg.get("cluster", "") or "").strip().lower()


def mlpot_device_name(cfg: dict[str, Any]) -> str:
    dev = str(cfg.get("mlpot_device", "") or "").strip().lower()
    if dev in ("cpu", "gpu"):
        return dev
    return "cpu" if scheduler_mode(cfg) == "cpu" else "gpu"


def slurm_tier_resource_pools(cfg: dict[str, Any]) -> dict[str, int]:
    if scheduler_mode(cfg) == "cpu":
        n = slurm_max_concurrent(cfg)
        return {"charmm_slot": n}
    if not slurm_tier_enabled(cfg):
        n = slurm_max_concurrent(cfg)
        return {"gpu_fast": n, "gpu_slow": 0, "charmm_slot": n}
    fast = slurm_tier_gpu_pool(cfg, "fast")
    slow = slurm_tier_gpu_pool(cfg, "slow")
    return {"gpu_fast": fast, "gpu_slow": slow, "charmm_slot": fast + slow}


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    pools = slurm_tier_resource_pools(cfg)
    if scheduler_mode(cfg) == "cpu":
        return int(pools["charmm_slot"])
    return int(pools["gpu_fast"]) + int(pools["gpu_slow"])


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    pools = slurm_tier_resource_pools(cfg)
    return " ".join(f"{key}={value}" for key, value in pools.items())


def local_gpu_count(cfg: dict[str, Any]) -> int:
    for key in ("local_gpu_count", "local_max_concurrent"):
        val = int(cfg.get(key) or 0)
        if val > 0:
            return val
    return 2


def local_max_concurrent(cfg: dict[str, Any]) -> int:
    val = int(cfg.get("local_max_concurrent") or 0)
    if val > 0:
        return val
    return local_gpu_count(cfg)


def local_resource_pools(cfg: dict[str, Any]) -> dict[str, int]:
    n = local_max_concurrent(cfg)
    if scheduler_mode(cfg) == "cpu":
        return {"charmm_slot": n}
    return {"gpu_fast": n, "gpu_slow": 0, "charmm_slot": n}


def local_launch_jobs(cfg: dict[str, Any]) -> int:
    return local_max_concurrent(cfg)


def local_resources_cli(cfg: dict[str, Any]) -> str:
    pools = local_resource_pools(cfg)
    return " ".join(f"{key}={value}" for key, value in pools.items())


def total_pycharmm_equi_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("pycharmm_equi_ps", 20.0)) * int(cfg.get("pycharmm_equi_legs", 5))


def total_pycharmm_prod_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("pycharmm_prod_ps", 50.0)) * int(cfg.get("pycharmm_prod_legs", 10))


def warmup_mlpot_argv(cfg: dict[str, Any], cell: RunCell) -> list[str]:
    dense = dense_cell_mlpot_overrides(cell, cfg)
    batch = int(
        cfg.get("warmup_ml_batch_size")
        or dense.get("ml_batch_size", cfg.get("ml_batch_size", 2048))
    )
    argv = [
        "warmup-mlpot-jax",
        "--checkpoint",
        str(checkpoint_path_for_yaml(str(cfg["checkpoint"]))),
        "--n-monomers",
        str(int(cell.n_monomers)),
        "--box-side",
        str(float(cell.box_size)),
        "--ml-batch-size",
        str(batch),
        "--ml-gpu-count",
        str(int(cfg.get("ml_gpu_count", 1))),
        "--quiet",
    ]
    compile_threads = cfg.get("warmup_compile_threads", cfg.get("jax_compile_threads"))
    if compile_threads is not None:
        argv.extend(["--compile-threads", str(int(compile_threads))])
    if bool(cfg.get("warmup_do_mm", True)):
        argv.append("--do-mm")
    return argv
