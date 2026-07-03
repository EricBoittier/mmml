"""Generate md-system campaigns for DCM density × setup comparison."""

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
    jaxmd_job_flags,
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


def default_checkpoint_path() -> Path:
    """Bundled DES dimer PhysNet JSON when ``MMML_CKPT`` is unset."""
    return repo_root() / "examples" / "ckpts_json" / "DESdimers_params.json"


def validate_checkpoint(path: Path) -> None:
    text = str(path)
    placeholders = ("/path/to", "your/checkpoint", "REPLACE_ME")
    if any(p in text for p in placeholders):
        raise RuntimeError(
            f"Checkpoint path looks like a placeholder: {path}\n"
            "Set a real file, e.g.\n"
            f"  export MMML_CKPT={default_checkpoint_path()}"
        )
    if not path.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {path}\n"
            "Export MMML_CKPT before launching Snakemake, e.g.\n"
            f"  export MMML_CKPT={default_checkpoint_path()}"
        )


def resolve_checkpoint(raw: str) -> Path:
    if raw == "${MMML_CKPT}":
        env = os.environ.get("MMML_CKPT", "").strip()
        if env:
            path = Path(env).expanduser().resolve()
        else:
            path = default_checkpoint_path()
    else:
        path = Path(os.path.expandvars(str(raw))).expanduser().resolve()
    validate_checkpoint(path)
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
    sweep_id: str | None = None


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


def temperature_ladder_enabled(cfg: dict[str, Any]) -> bool:
    """When true, higher-T cells continue from the prior successful lower-T handoff."""
    if prep_sweep_enabled(cfg):
        return False
    return bool(cfg.get("temperature_ladder", False))


def temperature_ladder_prior_temperature(
    cfg: dict[str, Any], temperature: float
) -> float | None:
    """Next lower matrix temperature (K), or ``None`` at the ladder base."""
    temps = sorted(matrix_temperatures(cfg))
    t = float(temperature)
    lower = [x for x in temps if x < t - 1e-6]
    if not lower:
        return None
    return float(max(lower))


def temperature_ladder_prior_cell(cell: RunCell, cfg: dict[str, Any]) -> RunCell | None:
    """Matching cell at the next lower matrix temperature, if ladder mode is on."""
    prior_t = temperature_ladder_prior_temperature(cfg, cell.temperature)
    if prior_t is None:
        return None
    if not temperature_ladder_enabled(cfg):
        return None
    return RunCell(
        setup_id=cell.setup_id,
        solvent=cell.solvent,
        n_monomers=cell.n_monomers,
        temperature=prior_t,
        box_size=cell.box_size,
        heat_thermostat=cell.heat_thermostat,
        sweep_id=cell.sweep_id,
    )


def temperature_ladder_prior_tag(cell: RunCell, cfg: dict[str, Any]) -> str | None:
    prior = temperature_ladder_prior_cell(cell, cfg)
    if prior is None:
        return None
    return cell_run_tag(prior, cfg)


def temperature_ladder_prior_handoff(cfg: dict[str, Any], cell: RunCell) -> Path | None:
    prior = temperature_ladder_prior_cell(cell, cfg)
    if prior is None:
        return None
    return paths_for_run(cfg, prior)["final_handoff"]


def _apply_temperature_ladder(
    cfg: dict[str, Any],
    cell: RunCell,
    defaults: dict[str, Any],
    init_flags: dict[str, Any],
) -> str | None:
    """Set ``continue_from`` and heat ramp for ladder continuation cells."""
    prior = temperature_ladder_prior_cell(cell, cfg)
    if prior is None:
        return None
    prior_paths = paths_for_run(cfg, prior)
    prior_tag = cell_run_tag(prior, cfg)
    defaults["continue_from"] = str(prior_paths["final_handoff"])
    defaults["temperature_ladder_from_tag"] = prior_tag
    defaults["temperature_ladder_from_temp_K"] = float(prior.temperature)
    prior_t = float(prior.temperature)
    defaults["heat_firstt"] = prior_t
    init_flags["heat_firstt"] = prior_t
    init_flags["heat_finalt"] = float(cell.temperature)
    return prior_tag


def iter_heat_thermostats(cfg: dict[str, Any]) -> Iterator[str | None]:
    if prep_sweep_enabled(cfg):
        stages = prep_sweep_stages(cfg)
        if stages == "mini":
            yield None
            return
        anchor = prep_sweep_section(cfg).get("anchor") or {}
        ht = anchor.get("heat_thermostat")
        if ht:
            key = str(ht).strip().lower()
            if key not in _VALID_HEAT_THERMOSTATS:
                known = ", ".join(sorted(_VALID_HEAT_THERMOSTATS))
                raise ValueError(f"unknown prep_sweep.anchor heat_thermostat {ht!r} (known: {known})")
            yield key
            return
    hts = matrix_heat_thermostats(cfg)
    if not hts:
        yield None
        return
    for ht in hts:
        yield ht


_PREP_SWEEP_VARIANT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def prep_sweep_config_path() -> Path:
    return workflow_root() / "config.prep_sweep.yaml"


def is_prep_sweep_run_tag(tag: str) -> bool:
    return "_sw_" in str(tag)


def config_for_run_tag(cfg: dict[str, Any], tag: str) -> dict[str, Any]:
    """Use ``config.prep_sweep.yaml`` when the tag is a sweep cell and the active config is not."""
    if is_prep_sweep_run_tag(tag) and not prep_sweep_enabled(cfg):
        sweep_path = prep_sweep_config_path()
        if sweep_path.is_file():
            return load_config(sweep_path)
    return cfg


def default_workflow_config_path(*, run_tag: str | None = None) -> Path:
    """Default config file for Snakemake / job_shell (sweep tags → prep_sweep config)."""
    if run_tag and is_prep_sweep_run_tag(run_tag):
        sweep_path = prep_sweep_config_path()
        if sweep_path.is_file():
            return sweep_path
    return workflow_root() / "config.yaml"


def prep_sweep_section(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("prep_sweep")
    return dict(raw) if isinstance(raw, dict) else {}


def prep_sweep_enabled(cfg: dict[str, Any]) -> bool:
    return bool(prep_sweep_section(cfg).get("enabled", False))


def prep_sweep_stages(cfg: dict[str, Any]) -> str:
    stages = str(prep_sweep_section(cfg).get("stages", "mini")).strip().lower()
    if stages not in {"mini", "mini,heat"}:
        raise ValueError(
            f"prep_sweep.stages must be 'mini' or 'mini,heat', got {stages!r}"
        )
    return stages


def prep_sweep_variant_ids(cfg: dict[str, Any]) -> list[str]:
    variants = prep_sweep_section(cfg).get("variants") or {}
    if not isinstance(variants, dict) or not variants:
        raise ValueError("prep_sweep.enabled requires non-empty prep_sweep.variants")
    ids: list[str] = []
    for key in sorted(variants.keys()):
        vid = str(key).strip().lower()
        if not _PREP_SWEEP_VARIANT_KEY.match(vid):
            raise ValueError(
                f"invalid prep_sweep variant id {key!r} "
                "(use lowercase letters, digits, underscore; max 32 chars)"
            )
        ids.append(vid)
    return ids


def prep_sweep_variant_overrides(cfg: dict[str, Any], sweep_id: str) -> dict[str, Any]:
    variants = prep_sweep_section(cfg).get("variants") or {}
    key = str(sweep_id).strip().lower()
    if key not in variants:
        known = ", ".join(prep_sweep_variant_ids(cfg))
        raise KeyError(f"unknown prep_sweep variant {sweep_id!r} (known: {known})")
    raw = variants[key]
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"prep_sweep.variants[{key!r}] must be a mapping")
    return dict(raw)


def _resolve_prep_sweep_n_monomers(cfg: dict[str, Any], anchor: dict[str, Any]) -> int:
    if anchor.get("n_monomers") is not None:
        return int(anchor["n_monomers"])
    frac = anchor.get("bulk_density_fraction")
    if frac is None:
        raise ValueError(
            "prep_sweep.anchor requires n_monomers or bulk_density_fraction"
        )
    solvent = str(anchor.get("solvent") or cfg.get("solvents", ["DCM"])[0]).strip().upper()
    box = float(anchor["box_size"])
    return n_monomers_at_bulk_density(
        solvent,
        box,
        float(frac),
        min_n=int(cfg.get("bulk_density_n_min", 1)),
        max_n=int(anchor["bulk_density_n_max"])
        if anchor.get("bulk_density_n_max") is not None
        else None,
    )


def prep_sweep_anchor_cell(cfg: dict[str, Any]) -> RunCell:
    anchor = dict(prep_sweep_section(cfg).get("anchor") or {})
    setup_id = str(anchor.get("setup_id") or matrix_setup_ids(cfg)[0]).strip()
    resolve_setup_variant(setup_id)
    solvent = str(anchor.get("solvent") or cfg.get("solvents", ["DCM"])[0]).strip().upper()
    temperature = float(anchor.get("temperature", matrix_temperatures(cfg)[0]))
    box_size = float(anchor.get("box_size", matrix_box_sizes(cfg)[0]))
    n_monomers = _resolve_prep_sweep_n_monomers(cfg, anchor)
    heat_thermostat: str | None = None
    if prep_sweep_stages(cfg) == "mini,heat":
        ht_raw = anchor.get("heat_thermostat") or cfg.get("default_heat_thermostat", "bussi")
        heat_thermostat = str(ht_raw).strip().lower()
        if heat_thermostat not in _VALID_HEAT_THERMOSTATS:
            known = ", ".join(sorted(_VALID_HEAT_THERMOSTATS))
            raise ValueError(f"unknown prep_sweep.anchor heat_thermostat {ht_raw!r} (known: {known})")
    return RunCell(
        setup_id=setup_id,
        solvent=solvent,
        n_monomers=n_monomers,
        temperature=temperature,
        box_size=box_size,
        heat_thermostat=heat_thermostat,
        sweep_id=None,
    )


def iter_prep_sweep_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    anchor = prep_sweep_anchor_cell(cfg)
    for sweep_id in prep_sweep_variant_ids(cfg):
        yield RunCell(
            setup_id=anchor.setup_id,
            solvent=anchor.solvent,
            n_monomers=anchor.n_monomers,
            temperature=anchor.temperature,
            box_size=anchor.box_size,
            heat_thermostat=anchor.heat_thermostat,
            sweep_id=sweep_id,
        )


def cell_workflow_cfg(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    """Per-cell config: prep_sweep variant overrides and sweep stage scope."""
    if not cell.sweep_id:
        return cfg
    out = dict(cfg)
    out.update(prep_sweep_variant_overrides(cfg, cell.sweep_id))
    sweep = prep_sweep_section(cfg)
    if not sweep.get("full_dynamics", False):
        out["dynamics_legs"] = {
            "pycharmm_equi": False,
            "pycharmm_prod": False,
            "jaxmd": False,
            "ase": False,
        }
    if prep_sweep_stages(cfg) == "mini":
        out["heat_thermostats"] = []
    elif cell.heat_thermostat:
        out["heat_thermostats"] = [cell.heat_thermostat]
    return out


def iter_matrix_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    if prep_sweep_enabled(cfg):
        skip = {str(t).strip() for t in (cfg.get("exclude_run_tags") or [])}
        seen_tags: set[str] = set()
        for cell in iter_prep_sweep_cells(cfg):
            tag = cell_run_tag(cell, cfg)
            if tag in skip or tag in seen_tags:
                continue
            seen_tags.add(tag)
            yield cell
        return
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
        base = f"{base}_ht_{cell.heat_thermostat}"
    if cell.sweep_id:
        base = f"{base}_sw_{cell.sweep_id}"
    return base


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/dcm_density_setup_compare"))
    return (root / cell_run_tag(cell, cfg)).resolve()


def prep_sweep_placement_seed_ignore_heat(cfg: dict[str, Any]) -> bool:
    """When true, heat thermostat does not shift Packmol seed for prep_sweep cells."""
    if not prep_sweep_enabled(cfg):
        return False
    return bool(prep_sweep_section(cfg).get("placement_seed_ignore_heat", True))


def placement_seed_ignore_heat(cfg: dict[str, Any]) -> bool:
    """When true, heat thermostat name does not shift Packmol / placement seed."""
    if bool(cfg.get("placement_seed_ignore_heat", False)):
        return True
    return prep_sweep_placement_seed_ignore_heat(cfg)


def run_seed(cell: RunCell, *, seed_base: int = 4242, cfg: dict[str, Any] | None = None) -> int:
    setup_off = sum(ord(c) for c in cell.setup_id) % 1000
    solvent_off = sum(ord(c) for c in solvent_slug(cell.solvent)) % 1000
    temp_off = int(round(cell.temperature)) % 100
    box_off = int(round(cell.box_size)) % 100
    heat_off = 0
    ignore_heat_seed = cfg is not None and placement_seed_ignore_heat(cfg)
    if cell.heat_thermostat and not ignore_heat_seed:
        heat_off = {"bussi": 11, "hoover": 22, "scale": 33}[cell.heat_thermostat]
    sweep_off = 0
    if cell.sweep_id:
        sweep_off = sum(ord(c) for c in cell.sweep_id) % 997
    return (
        int(seed_base)
        + int(cell.n_monomers) * 10000
        + setup_off
        + solvent_off
        + temp_off * 17
        + box_off * 131
        + heat_off * 7
        + sweep_off * 13
    )


def leg_output_dir(cell_root: Path, job_id: str) -> str:
    return str((cell_root / job_id).resolve())


def _attach_leg_output_dir(job: dict[str, Any], cell_root: Path, job_id: str) -> dict[str, Any]:
    return {**job, "output_dir": leg_output_dir(cell_root, job_id)}


# Workflow config.yaml overrides cleanup / dense-cell defaults (applied last).
_WORKFLOW_JOB_OVERRIDE_KEYS = (
    "bonded_mm_mini",
    "bonded_mm_mini_after",
    "bonded_mm_mini_steps",
    "no_echeck_heat",
    "no_scale_max_grms",
    "allow_high_grms",
    "mini_nstep",
    "charmm_sd_steps",
    "charmm_abnr_steps",
    "charmm_pre_minimize",
    "pre_min_steps",
    "pre_min_fmax",
    "dynamics_overlap_check_interval",
    "dynamics_overlap_charmm_sd_steps",
    "dynamics_overlap_charmm_abnr_steps",
    "dynamics_overlap_min_distance",
    "dynamics_intra_min_distance",
    "dynamics_overlap_action",
    "overlap_run_state_every_chunks",
    "mm_nonbond_mode",
    "periodic_charmm_vdw",
    "charmm_mm_pretreat",
    "dt_fs",
    "spacing",
    "packmol_tolerance",
    "packmol_box_padding",
    "mm_switch_on",
    "mm_switch_width",
    "ml_switch_width",
    "dcd_nsavc",
    "dyn_inbfrq",
    "dyn_nprint",
    "geometry_packing_fire_bfgs_crossover_grms",
)


def _apply_workflow_job_overrides(flags: dict[str, Any], effective: dict[str, Any]) -> None:
    for key in _WORKFLOW_JOB_OVERRIDE_KEYS:
        if key in effective:
            flags[key] = effective[key]


def _mini_job_flags(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    cell_cfg = cell_workflow_cfg(cfg, cell)
    variant = resolve_setup_variant(cell.setup_id)
    effective = merge_setup_into_config(cell_cfg, cell.setup_id)
    flags: dict[str, Any] = dict(variant.job_overrides)
    if variant.use_cleanup_strategy:
        strategy = resolve_cleanup_strategy(effective)
        flags.update(pycharmm_job_flags(strategy))
        flags.update(dense_cell_mlpot_overrides(cell, effective))
        flags.update(pretreat_job_flags(strategy))
    else:
        flags.update(dense_cell_mlpot_overrides(cell, effective))
    _apply_workflow_job_overrides(flags, effective)
    # Global workflow yaml (resilient-focused) must not override setup-specific disables.
    for key in (
        "cleanup",
        "calculator_pre_minimize",
        "liquid_prep",
        "density_prep_ladder",
        "charmm_mm_pretreat",
        "charmm_pre_minimize",
        "bonded_mm_mini",
    ):
        if key in variant.job_overrides:
            flags[key] = variant.job_overrides[key]
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


def parse_dynamics_legs(cfg: dict[str, Any]) -> dict[str, bool]:
    """Which post-prep dynamics legs to append (PyCHARMM equi/prod, JAX-MD, ASE)."""
    raw = cfg.get("dynamics_legs")
    if raw is None:
        return {
            "pycharmm_equi": False,
            "pycharmm_prod": False,
            "jaxmd": False,
            "ase": False,
        }
    if isinstance(raw, bool):
        return {
            "pycharmm_equi": bool(raw),
            "pycharmm_prod": bool(raw),
            "jaxmd": bool(raw),
            "ase": bool(raw),
        }
    return {
        "pycharmm_equi": bool(raw.get("pycharmm_equi", False)),
        "pycharmm_prod": bool(raw.get("pycharmm_prod", False)),
        "jaxmd": bool(raw.get("jaxmd", False)),
        "ase": bool(raw.get("ase", False)),
    }


def dynamics_campaign_enabled(cfg: dict[str, Any]) -> bool:
    return any(parse_dynamics_legs(cfg).values())


def init_job_id(cfg: dict[str, Any]) -> str:
    """First PyCHARMM leg id (``pycharmm_init`` when dynamics legs are enabled)."""
    return "pycharmm_init" if dynamics_campaign_enabled(cfg) else "pycharmm_mini"


def campaign_job_order(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or {}
    legs = parse_dynamics_legs(cfg)
    order = [init_job_id(cfg)]
    if legs["pycharmm_equi"]:
        n_equi = max(1, int(cfg.get("pycharmm_equi_legs", 1)))
        order.extend(f"pycharmm_equi_{i:02d}" for i in range(1, n_equi + 1))
    if legs["pycharmm_prod"]:
        n_prod = max(1, int(cfg.get("pycharmm_prod_legs", 1)))
        order.extend(f"pycharmm_prod_{i:02d}" for i in range(1, n_prod + 1))
    if legs["jaxmd"]:
        order.append("jaxmd_prod")
    if legs["ase"]:
        order.append("ase_prod")
    return order


def campaign_final_job_id(cfg: dict[str, Any]) -> str:
    return campaign_job_order(cfg)[-1]


def _init_stage_overrides(
    cfg: dict[str, Any], cell: RunCell, effective: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ``md_stages`` / heat kwargs for the first PyCHARMM leg."""
    cell_cfg = cell_workflow_cfg(cfg, cell)
    heat = _heat_job_overrides(cell_cfg, cell, effective)
    if heat:
        return heat
    if dynamics_campaign_enabled(cell_cfg):
        strategy = resolve_cleanup_strategy(effective)
        default_ht = str(cell_cfg.get("default_heat_thermostat", "bussi")).strip().lower()
        heat_thermostat = _resolve_cell_heat_thermostat(
            {**effective, "heat_thermostat": default_ht},
            strategy,
        )
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
    return {"md_stages": "mini"}


def _equi_prod_flags(mini_flags: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "md_stages",
        "ps_heat",
        "n_heat_segments",
        "heat_firstt",
        "heat_finalt",
        "heat_thermostat",
    }
    return {k: v for k, v in mini_flags.items() if k not in skip}


def _append_pycharmm_equi_prod_legs(
    runs: dict[str, Any],
    *,
    cfg: dict[str, Any],
    cell: RunCell,
    cell_root: Path,
    comp: str,
    repair: dict[str, Any],
    prev: str,
) -> str:
    legs = parse_dynamics_legs(cfg)
    equi_ps = float(cfg.get("pycharmm_equi_ps", 10.0))
    prod_ps = float(cfg.get("pycharmm_prod_ps", 10.0))
    prod_setup = str(cfg.get("prod_ensemble", "pbc_npt"))
    optional = {str(x) for x in (cfg.get("optional_legs") or [])}

    if legs["pycharmm_equi"]:
        n_equi = max(1, int(cfg.get("pycharmm_equi_legs", 1)))
        for i in range(1, n_equi + 1):
            jid = f"pycharmm_equi_{i:02d}"
            job = _attach_leg_output_dir(
                {
                    "description": (
                        f"{comp} NPT equil {i}/{n_equi} ({equi_ps} ps) "
                        f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å"
                    ),
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
            if jid in optional:
                job["optional"] = True
            runs[jid] = job
            prev = jid

    if legs["pycharmm_prod"]:
        n_prod = max(1, int(cfg.get("pycharmm_prod_legs", 1)))
        for i in range(1, n_prod + 1):
            jid = f"pycharmm_prod_{i:02d}"
            job = _attach_leg_output_dir(
                {
                    "description": (
                        f"{comp} {prod_setup} prod {i}/{n_prod} ({prod_ps} ps) "
                        f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å"
                    ),
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
            if jid in optional:
                job["optional"] = True
            runs[jid] = job
            prev = jid
    return prev


def _append_jaxmd_leg(
    runs: dict[str, Any],
    *,
    cfg: dict[str, Any],
    cell: RunCell,
    cell_root: Path,
    comp: str,
    effective: dict[str, Any],
    prev: str,
) -> str:
    strategy = resolve_cleanup_strategy(effective)
    jaxmd_extra = jaxmd_job_flags(strategy)
    ps = float(cfg.get("jaxmd_ps", 10.0))
    setup = str(cfg.get("jaxmd_setup", "pbc_nvt"))
    optional = {str(x) for x in (cfg.get("optional_legs") or [])}
    jid = "jaxmd_prod"
    job = _attach_leg_output_dir(
        {
            "description": (
                f"{comp} JAX-MD {setup} ({ps} ps) "
                f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å"
            ),
            "backend": "jaxmd",
            "setup": setup,
            "ps": ps,
            "depends_on": prev,
            **jaxmd_extra,
        },
        cell_root,
        jid,
    )
    if jid in optional:
        job["optional"] = True
    runs[jid] = job
    return jid


def _append_ase_leg(
    runs: dict[str, Any],
    *,
    cfg: dict[str, Any],
    cell: RunCell,
    cell_root: Path,
    comp: str,
    prev: str,
) -> str:
    ps = float(cfg.get("ase_ps", 10.0))
    setup = str(cfg.get("ase_setup", "pbc_nvt"))
    raw_integrator = str(cfg.get("ase_integrator", "nvt_nhc")).strip().lower()
    if raw_integrator in {"nvt_nhc", "nhc"}:
        nvt_integrator = "nhc"
    elif raw_integrator in {"nvt_langevin", "langevin"}:
        nvt_integrator = "langevin"
    elif raw_integrator == "auto":
        nvt_integrator = "auto"
    else:
        raise ValueError(
            f"unknown ase_integrator {raw_integrator!r} "
            "(use nvt_nhc, nvt_langevin, or auto)"
        )
    optional = {str(x) for x in (cfg.get("optional_legs") or [])}
    extra = cfg.get("ase_extra_args")
    if not extra:
        extra = ["--log-every", "100", "--traj-every", "100"]
    jid = "ase_prod"
    job = _attach_leg_output_dir(
        {
            "description": (
                f"{comp} ASE {raw_integrator} {setup} ({ps} ps) "
                f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å"
            ),
            "backend": "ase",
            "setup": setup,
            "nvt_integrator": nvt_integrator,
            "ps": ps,
            "depends_on": prev,
            "extra_args": list(extra),
        },
        cell_root,
        jid,
    )
    if jid in optional:
        job["optional"] = True
    runs[jid] = job
    return jid


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
    cell_cfg = cell_workflow_cfg(cfg, cell)
    effective = merge_setup_into_config(cell_cfg, cell.setup_id)
    comp = composition_string(cell)
    tag = cell_run_tag(cell, cfg)
    seed = run_seed(cell, seed_base=int(cfg.get("seed_base", 4242)), cfg=cfg)
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
    if cell.sweep_id:
        defaults["prep_sweep_id"] = cell.sweep_id
        defaults["prep_sweep_overrides"] = prep_sweep_variant_overrides(cfg, cell.sweep_id)
    for key in (
        "liquid_prep",
        "density_prep_ladder",
        "density_prep_ladder_max_rounds",
        "density_prep_lattice_abnr_steps",
        "mini_lattice_abnr_steps",
        "mc_density_equalize",
        "mc_density_steps",
        "geometry_packing_fire_bfgs_crossover_grms",
        "bonded_mm_mini_steps",
        "min_intermonomer_atom_distance",
        "max_grms_before_dyn",
        "no_scale_max_grms",
        "allow_high_grms",
        "mini_box_equil_ps",
        "calculator_pre_minimize",
        "periodic_charmm_vdw",
        "dcd_nsavc",
        "dyn_inbfrq",
        "dynamics_overlap_check_interval",
        "bonded_mm_mini",
        "charmm_mm_pretreat",
    ):
        if key in effective:
            defaults[key] = effective[key]

    if dynamics_campaign_enabled(cell_cfg):
        defaults["handoff_write_res"] = bool(cell_cfg.get("handoff_write_res", True))
        defaults["continue_velocities"] = bool(cell_cfg.get("continue_velocities", True))

    mini_flags = _mini_job_flags(cfg, cell)
    mini_flags.update(_init_stage_overrides(cfg, cell, effective))
    ladder_from = _apply_temperature_ladder(cfg, cell, defaults, mini_flags)
    init_id = init_job_id(cell_cfg)
    md_stages = str(mini_flags.get("md_stages", "mini"))
    ht = mini_flags.get("heat_thermostat")
    ht_note = f" heat={ht}" if ht else ""
    if ladder_from is not None:
        stage_label = f"heat ladder from {defaults['temperature_ladder_from_temp_K']:.0f}K"
    elif "heat" in md_stages:
        stage_label = "mini+heat"
    else:
        stage_label = "mini-only"
    runs: dict[str, Any] = {
        init_id: _attach_leg_output_dir(
            {
                "description": (
                    f"{comp} setup={cell.setup_id} {frac_s} "
                    f"T={cell.temperature:.0f}K L={cell.box_size:.0f}Å {stage_label}{ht_note}"
                    + (f" ← {ladder_from}" if ladder_from else "")
                ),
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stages": md_stages,
                **mini_flags,
            },
            cell_root,
            init_id,
        ),
    }

    prev = init_id
    legs = parse_dynamics_legs(cell_cfg)
    if legs["pycharmm_equi"] or legs["pycharmm_prod"]:
        repair = _equi_prod_flags(mini_flags)
        prev = _append_pycharmm_equi_prod_legs(
            runs,
            cfg=cell_cfg,
            cell=cell,
            cell_root=cell_root,
            comp=comp,
            repair=repair,
            prev=prev,
        )
    if legs["jaxmd"]:
        prev = _append_jaxmd_leg(
            runs,
            cfg=cell_cfg,
            cell=cell,
            cell_root=cell_root,
            comp=comp,
            effective=effective,
            prev=prev,
        )
    if legs["ase"]:
        _append_ase_leg(
            runs,
            cfg=cell_cfg,
            cell=cell,
            cell_root=cell_root,
            comp=comp,
            prev=prev,
        )

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
    cell_cfg = cell_workflow_cfg(cfg, cell)
    out = run_output_dir(cfg, cell)
    final_job = campaign_final_job_id(cell_cfg)
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "final_handoff": out / final_job / "handoff" / "state.npz",
        "done": out / "done.txt",
    }


def parse_run_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    setup_ids = list(matrix_setup_ids(cfg))
    try:
        from setup_variants import all_setup_variants

        for sid in all_setup_variants():
            if sid not in setup_ids:
                setup_ids.append(sid)
    except Exception:
        pass
    for setup_id in sorted(setup_ids, key=len, reverse=True):
        prefix = f"{setup_id}_"
        if not tag.startswith(prefix):
            continue
        tail = tag[len(prefix) :]
        m = re.match(
            r"([a-z]+)_(\d+)_t(\d+)_l(\d+)(?:_ht_(bussi|hoover|scale))?(?:_sw_([a-z0-9_]+))?$",
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
            sweep_id=m.group(6),
        )
    raise KeyError(f"run tag {tag!r} not in config matrix")


def cell_from_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    cfg = config_for_run_tag(cfg, tag)
    by_tag = {cell_run_tag(c, cfg): c for c in iter_matrix_cells(cfg)}
    if tag in by_tag:
        return by_tag[tag]
    parsed = parse_run_tag(cfg, tag)
    if parsed.sweep_id:
        if not prep_sweep_enabled(cfg):
            raise KeyError(
                f"run tag {tag!r} is a prep sweep tag but prep_sweep.enabled is false "
                f"in {default_workflow_config_path(run_tag=tag)}. "
                "Set MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml or use "
                "bash scripts/snakemake_prep_sweep.sh"
            )
        variant_ids = prep_sweep_variant_ids(cfg)
        if parsed.sweep_id not in variant_ids:
            raise KeyError(
                f"run tag {tag!r} uses unknown prep_sweep variant {parsed.sweep_id!r} "
                f"(known: {', '.join(variant_ids)})"
            )
        return parsed
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
