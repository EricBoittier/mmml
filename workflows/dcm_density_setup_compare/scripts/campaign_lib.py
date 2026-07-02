"""Generate md-system mini-only campaigns for DCM density × setup comparison."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

_BURST_SCRIPTS = Path(__file__).resolve().parents[1].parent / "pbc_solvent_burst" / "scripts"
if str(_BURST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BURST_SCRIPTS))

from bulk_density import (  # noqa: E402
    matrix_cluster_sizes_for_cell,
    matrix_uses_bulk_density,
    n_monomers_at_bulk_density,
)
from cleanup_strategy import (  # noqa: E402
    CleanupStrategy,
    dense_cell_mlpot_overrides,
    pretreat_job_flags,
    pycharmm_job_flags,
    resolve_cleanup_strategy,
)

from setup_variants import (  # noqa: E402
    merge_setup_into_config,
    resolve_setup_variant,
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
        path = Path(os.path.expandvars(str(raw))).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")
    return path


def checkpoint_path_for_yaml(raw: str) -> str:
    """Resolve ``${MMML_CKPT}`` when writing campaign YAML on the submit host."""
    if str(raw).strip() == "${MMML_CKPT}":
        return str(resolve_checkpoint(str(raw)))
    return str(os.path.expandvars(str(raw)))


_VALID_HEAT_THERMOSTATS = frozenset({"bussi", "hoover", "scale"})


@dataclass(frozen=True)
class RunCell:
    setup_id: str
    solvent: str
    n_monomers: int
    temperature: float
    box_size: float
    heat_thermostat: str | None = None


def matrix_setup_ids(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("setups")
    if not raw:
        raise ValueError("config requires setups: [minimal, ...]")
    return [str(x).strip() for x in raw if str(x).strip()]


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


def matrix_heat_thermostats(cfg: dict[str, Any]) -> list[str]:
    """Non-empty list enables mini+heat campaigns with one cell per thermostat."""
    raw = cfg.get("heat_thermostats")
    if not raw:
        return []
    thermostats: list[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if key not in _VALID_HEAT_THERMOSTATS:
            known = ", ".join(sorted(_VALID_HEAT_THERMOSTATS))
            raise ValueError(f"unknown heat_thermostat {item!r} (known: {known})")
        thermostats.append(key)
    return thermostats


def heat_compare_enabled(cfg: dict[str, Any]) -> bool:
    return bool(matrix_heat_thermostats(cfg))


def iter_heat_thermostats(cfg: dict[str, Any]) -> Iterator[str | None]:
    hts = matrix_heat_thermostats(cfg)
    if not hts:
        yield None
        return
    for ht in hts:
        yield ht


def iter_matrix_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", ["DCM"])]
    setups = matrix_setup_ids(cfg)
    if matrix_uses_bulk_density(cfg):
        if cfg.get("cluster_sizes"):
            raise ValueError("Set either bulk_density_fractions or cluster_sizes, not both.")
    elif not cfg.get("cluster_sizes"):
        raise ValueError("Matrix requires cluster_sizes or bulk_density_fractions.")
    skip = {str(t).strip() for t in (cfg.get("exclude_run_tags") or [])}
    seen_tags: set[str] = set()
    for setup_id in setups:
        resolve_setup_variant(setup_id)
        for sol in solvents:
            for box in matrix_box_sizes(cfg):
                sizes = matrix_cluster_sizes_for_cell(cfg, solvent=sol, box_size=box)
                for n in sizes:
                    for temp in matrix_temperatures(cfg):
                        for heat_thermostat in iter_heat_thermostats(cfg):
                            cell = RunCell(
                                setup_id=setup_id,
                                solvent=sol,
                                n_monomers=n,
                                temperature=temp,
                                box_size=box,
                                heat_thermostat=heat_thermostat,
                            )
                            tag = cell_run_tag(cell, cfg)
                            if tag in skip or tag in seen_tags:
                                continue
                            seen_tags.add(tag)
                            yield cell


def matrix_tag_includes_TL(cfg: dict[str, Any]) -> bool:
    return len(matrix_temperatures(cfg)) > 1 or len(matrix_box_sizes(cfg)) > 1


def solvent_slug(solvent: str) -> str:
    return str(solvent).strip().upper()


def cell_run_tag(cell: RunCell, cfg: dict[str, Any] | None = None) -> str:
    sol = solvent_slug(cell.solvent).lower()
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    base = f"{cell.setup_id}_{sol}_{int(cell.n_monomers)}_t{t}_l{box}"
    if cell.heat_thermostat:
        return f"{base}_ht_{cell.heat_thermostat}"
    return base


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/dcm_density_setup_compare"))
    return (root / cell_run_tag(cell, cfg)).resolve()


def run_seed(cell: RunCell, *, seed_base: int = 4242) -> int:
    setup_off = sum(ord(c) for c in cell.setup_id) % 1000
    solvent_off = sum(ord(c) for c in solvent_slug(cell.solvent)) % 1000
    temp_off = int(round(cell.temperature)) % 100
    box_off = int(round(cell.box_size)) % 100
    heat_off = 0
    if cell.heat_thermostat:
        heat_off = {"bussi": 11, "hoover": 22, "scale": 33}[cell.heat_thermostat]
    return (
        int(seed_base)
        + int(cell.n_monomers) * 10000
        + setup_off
        + solvent_off
        + temp_off * 17
        + box_off * 131
        + heat_off * 7
    )


def leg_output_dir(cell_root: Path, job_id: str) -> str:
    return str((cell_root / job_id).resolve())


def _attach_leg_output_dir(job: dict[str, Any], cell_root: Path, job_id: str) -> dict[str, Any]:
    return {**job, "output_dir": leg_output_dir(cell_root, job_id)}


def _mini_job_flags(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    variant = resolve_setup_variant(cell.setup_id)
    effective = merge_setup_into_config(cfg, cell.setup_id)
    flags: dict[str, Any] = dict(variant.job_overrides)
    if variant.use_cleanup_strategy:
        strategy = resolve_cleanup_strategy(effective)
        flags.update(pycharmm_job_flags(strategy))
        flags.update(dense_cell_mlpot_overrides(cell, effective))
        flags.update(pretreat_job_flags(strategy))
    else:
        flags.update(dense_cell_mlpot_overrides(cell, effective))
    # Workflow config.yaml overrides cleanup defaults (e.g. bonded_mm_mini: false).
    for key in (
        "bonded_mm_mini",
        "bonded_mm_mini_after",
        "bonded_mm_mini_steps",
        "no_echeck_heat",
    ):
        if key in cfg:
            flags[key] = cfg[key]
    return flags


def _resolve_cell_heat_thermostat(cfg: dict[str, Any], strategy: CleanupStrategy) -> str:
    """Resolve heat thermostat for campaign YAML (supports bussi, hoover, scale).

    When the cleanup strategy enables CHARMM MM pretreat, ``scale`` is coerced to
    ``hoover`` (same rule as ``pbc_solvent_burst``). ``bussi`` is preserved.
    """
    requested = str(cfg.get("heat_thermostat", "bussi") or "bussi").strip().lower()
    if requested not in _VALID_HEAT_THERMOSTATS:
        known = ", ".join(sorted(_VALID_HEAT_THERMOSTATS))
        raise ValueError(f"unknown heat_thermostat {requested!r} (known: {known})")
    pretreat = bool(strategy.charmm_mm.get("pretreat_on_pycharmm", False))
    if pretreat and requested == "scale":
        return "hoover"
    return requested


def _heat_job_overrides(cfg: dict[str, Any], cell: RunCell, effective: dict[str, Any]) -> dict[str, Any]:
    """Mini+heat leg flags when ``heat_thermostats`` is set in workflow config."""
    if not cell.heat_thermostat:
        return {}
    strategy = resolve_cleanup_strategy(effective)
    cell_effective = {**effective, "heat_thermostat": cell.heat_thermostat}
    heat_thermostat = _resolve_cell_heat_thermostat(cell_effective, strategy)
    dense = dense_cell_mlpot_overrides(cell, effective)
    return {
        "md_stages": "mini,heat",
        "ps_heat": float(effective.get("ps_heat", 5.0)),
        "n_heat_segments": int(
            dense.get("n_heat_segments", effective.get("n_heat_segments", 3))
        ),
        "heat_firstt": float(effective.get("heat_firstt", 10.0)),
        "heat_finalt": float(cell.temperature),
        "heat_thermostat": heat_thermostat,
    }


def campaign_job_order(cfg: dict[str, Any] | None = None) -> list[str]:
    return ["pycharmm_mini"]


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


def build_campaign(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    effective = merge_setup_into_config(cfg, cell.setup_id)
    comp = composition_string(cell)
    tag = cell_run_tag(cell, cfg)
    seed = run_seed(cell, seed_base=int(cfg.get("seed_base", 4242)))
    cell_root = run_output_dir(cfg, cell)
    variant = resolve_setup_variant(cell.setup_id)
    frac = cell_bulk_density_fraction(cell, cfg)
    frac_s = f"{frac:.2f}×bulk" if frac is not None else "custom N"

    defaults: dict[str, Any] = {
        "composition": comp,
        "checkpoint": checkpoint_path_for_yaml(str(effective["checkpoint"])),
        "box_size": float(cell.box_size),
        "output_root": str(cell_root),
        "packmol_cache_dir": str(cell_root / ".packmol_cache"),
        "spacing": float(effective.get("spacing", 4.0)),
        "packmol_tolerance": float(effective.get("packmol_tolerance", 1.0)),
        "dt_fs": float(effective.get("dt_fs", 0.25)),
        "temperature": float(cell.temperature),
        "pressure": float(effective.get("pressure", 1.0)),
        "seed": seed,
        "mm_switch_on": float(effective.get("mm_switch_on", 8.0)),
        "mm_switch_width": float(effective.get("mm_switch_width", 5.0)),
        "ml_switch_width": float(effective.get("ml_switch_width", 1.5)),
        "ml_gpu_count": int(effective.get("ml_gpu_count", 1)),
        "ml_batch_size": int(effective.get("ml_batch_size", 1024)),
        "setup_variant": cell.setup_id,
        "setup_description": variant.description,
        "bulk_density_fraction": frac,
    }
    for key in (
        "liquid_prep",
        "density_prep_ladder",
        "density_prep_ladder_max_rounds",
        "density_prep_lattice_abnr_steps",
        "mini_lattice_abnr_steps",
        "mc_density_equalize",
        "mc_density_steps",
        "min_intermonomer_atom_distance",
        "max_grms_before_dyn",
        "mini_box_equil_ps",
        "calculator_pre_minimize",
    ):
        if key in effective:
            defaults[key] = effective[key]

    mini_flags = _mini_job_flags(cfg, cell)
    mini_flags.update(_heat_job_overrides(cfg, cell, effective))
    stage_label = "mini+heat" if cell.heat_thermostat else "mini-only"
    ht_note = f" heat={cell.heat_thermostat}" if cell.heat_thermostat else ""
    runs: dict[str, Any] = {
        "pycharmm_mini": _attach_leg_output_dir(
            {
                "description": (
                    f"{comp} setup={cell.setup_id} {frac_s} "
                    f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å {stage_label}{ht_note}"
                ),
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stages": mini_flags.get("md_stages", "mini"),
                **mini_flags,
            },
            cell_root,
            "pycharmm_mini",
        ),
    }

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
    return max(1, len(slurm_gpu_nodes_slow(cfg)) * 2)


def slurm_tier_resource_pools(cfg: dict[str, Any]) -> dict[str, int]:
    if not slurm_tier_enabled(cfg):
        n = slurm_max_concurrent(cfg)
        return {"gpu_fast": n, "gpu_slow": 0, "charmm_slot": n}
    fast = slurm_tier_gpu_pool(cfg, "fast")
    slow = slurm_tier_gpu_pool(cfg, "slow")
    return {"gpu_fast": fast, "gpu_slow": slow, "charmm_slot": fast + slow}


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    pools = slurm_tier_resource_pools(cfg)
    return int(pools["gpu_fast"]) + int(pools["gpu_slow"])


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    """Space-separated ``NAME=N`` for ``snakemake --resources``."""
    pools = slurm_tier_resource_pools(cfg)
    return " ".join(f"{key}={value}" for key, value in pools.items())


def paths_for_run(cfg: dict[str, Any], cell: RunCell) -> dict[str, Path]:
    out = run_output_dir(cfg, cell)
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "final_handoff": out / "pycharmm_mini" / "handoff" / "state.npz",
        "done": out / "done.txt",
    }


def parse_run_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    for setup_id in sorted(matrix_setup_ids(cfg), key=len, reverse=True):
        prefix = f"{setup_id}_"
        if not tag.startswith(prefix):
            continue
        tail = tag[len(prefix) :]
        m = re.match(
            r"([a-z]+)_(\d+)_t(\d+)_l(\d+)(?:_ht_(bussi|hoover|scale))?$",
            tail,
        )
        if not m:
            break
        sol = m.group(1).upper()
        return RunCell(
            setup_id=setup_id,
            solvent=sol,
            n_monomers=int(m.group(2)),
            temperature=float(m.group(3)),
            box_size=float(m.group(4)),
            heat_thermostat=m.group(5),
        )
    raise KeyError(f"run tag {tag!r} not in config matrix")


def cell_from_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    by_tag = {cell_run_tag(c, cfg): c for c in iter_matrix_cells(cfg)}
    if tag in by_tag:
        return by_tag[tag]
    parsed = parse_run_tag(cfg, tag)
    if cell_run_tag(parsed, cfg) in by_tag:
        return by_tag[cell_run_tag(parsed, cfg)]
    raise KeyError(
        f"run tag {tag!r} not in config matrix "
        f"(examples: {', '.join(list(by_tag.keys())[:3])}…)"
    )


def cell_from_cli(
    cfg: dict[str, Any],
    setup_id: str,
    solvent: str,
    n_monomers: int,
    *,
    temperature: float | None = None,
    box_size: float | None = None,
    heat_thermostat: str | None = None,
) -> RunCell:
    sol = solvent_slug(solvent)
    n = int(n_monomers)
    temps = matrix_temperatures(cfg) if temperature is None else [float(temperature)]
    boxes = matrix_box_sizes(cfg) if box_size is None else [float(box_size)]
    hts = matrix_heat_thermostats(cfg)
    if heat_thermostat is None:
        if len(hts) == 1:
            heat_thermostat = hts[0]
        elif len(hts) > 1:
            raise ValueError(
                "Specify --heat-thermostat when config heat_thermostats lists multiple values"
            )
    elif hts and str(heat_thermostat).strip().lower() not in hts:
        raise ValueError(
            f"heat_thermostat {heat_thermostat!r} not in config heat_thermostats {hts}"
        )
    elif heat_thermostat is not None:
        heat_thermostat = str(heat_thermostat).strip().lower()
    if len(temps) != 1 or len(boxes) != 1:
        raise ValueError("Specify --temperature and --box-size when matrix lists have multiple values")
    cell = RunCell(
        setup_id=str(setup_id).strip(),
        solvent=sol,
        n_monomers=n,
        temperature=temps[0],
        box_size=boxes[0],
        heat_thermostat=heat_thermostat,
    )
    valid_tags = {cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)}
    if cell_run_tag(cell, cfg) not in valid_tags:
        raise ValueError(f"{cell} not in config matrix (valid tags: {len(valid_tags)})")
    return cell
