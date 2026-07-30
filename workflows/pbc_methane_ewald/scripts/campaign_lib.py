"""Generate md-system campaigns for liquid-methane PBC Ewald JSON sweeps."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

_SCRIPTS = Path(__file__).resolve().parent
_BURST_SCRIPTS = Path(__file__).resolve().parents[1].parent / "pbc_solvent_burst" / "scripts"
# Prefer this workflow's scripts/ for ``campaign_lib``, then reuse burst helpers.
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_BURST_SCRIPTS) not in sys.path:
    sys.path.append(str(_BURST_SCRIPTS))

from bulk_density import (  # noqa: E402
    matrix_cluster_sizes_for_cell,
    matrix_uses_bulk_density,
    n_monomers_at_bulk_density,
)
from cleanup_strategy import (  # noqa: E402
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "include":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else (workflow_root() / "config.yaml")
    path = path.expanduser()
    if not path.is_absolute():
        cand = (workflow_root() / path).resolve()
        path = cand if cand.is_file() else path.resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    includes = raw.get("include") or []
    if isinstance(includes, str):
        includes = [includes]
    cfg: dict[str, Any] = {}
    for rel in includes:
        inc_path = Path(rel)
        if not inc_path.is_absolute():
            inc_path = (path.parent / inc_path).resolve()
        cfg = _deep_merge(cfg, load_config(inc_path))
    cfg = _deep_merge(cfg, {k: v for k, v in raw.items() if k != "include"})
    if ckpt := os.environ.get("MMML_CKPT", "").strip():
        # Keep named checkpoint matrix; also expose a default single checkpoint.
        cfg.setdefault("checkpoint", ckpt)
    if output_root := os.environ.get("MMML_PBC_OUTPUT_ROOT", "").strip():
        cfg["output_root"] = output_root
    return cfg


def checkpoint_map(cfg: dict[str, Any]) -> dict[str, str]:
    """Return slug → checkpoint path for the JSON sweep."""
    raw = cfg.get("checkpoints")
    if isinstance(raw, dict) and raw:
        return {str(k): str(v) for k, v in raw.items()}
    single = cfg.get("checkpoint")
    if single:
        return {"default": str(single)}
    raise ValueError("config requires checkpoints: {slug: path} or checkpoint:")


def resolve_checkpoint_path(raw: str) -> Path:
    env = os.environ.get("MMML_CKPT", "").strip()
    if env and raw in {"${MMML_CKPT}", "$MMML_CKPT"}:
        path = Path(env).expanduser().resolve()
        if path.exists():
            return path
        raise RuntimeError(f"MMML_CKPT not found: {path}")
    expanded = Path(os.path.expandvars(str(raw))).expanduser()
    if expanded.exists():
        return expanded.resolve()
    fallback = (repo_root() / str(raw)).resolve()
    if fallback.exists():
        return fallback
    raise RuntimeError(f"Checkpoint not found: {raw}")


def validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")
    if path.suffix.lower() != ".json":
        raise RuntimeError(f"Expected a portable JSON checkpoint, got: {path}")


def checkpoint_path_for_yaml(raw: str) -> str:
    return str(resolve_checkpoint_path(raw))


def solvent_slug(solvent: str) -> str:
    return str(solvent).strip().upper()


def ckpt_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "ckpt"


# Mechanical ≈ fixed CGenFF q for E_MM Coulomb; electrostatic ≈ ML Q⁰ (q0).
_EMBEDDING_ALIASES: dict[str, tuple[str, str]] = {
    "mechanical": ("mechanical", "fixed"),
    "mech": ("mechanical", "fixed"),
    "fixed": ("mechanical", "fixed"),
    "electrostatic": ("electrostatic", "q0"),
    "es": ("electrostatic", "q0"),
    "elec": ("electrostatic", "q0"),
    "q0": ("electrostatic", "q0"),
}

_EMBEDDING_TAG: dict[str, str] = {
    "mechanical": "mech",
    "electrostatic": "es",
}


@dataclass(frozen=True)
class RunCell:
    """One matrix point: methane box × T × checkpoint × embedding × backend."""

    solvent: str
    n_monomers: int
    temperature: float
    box_size: float
    checkpoint_slug: str
    checkpoint: str
    backend: str
    embedding: str = "mechanical"
    mm_charge_mode: str = "fixed"


def matrix_temperatures(cfg: dict[str, Any]) -> list[float]:
    if cfg.get("temperatures"):
        return [float(x) for x in cfg["temperatures"]]
    return [float(cfg.get("temperature", 100.0))]


def matrix_box_sizes(cfg: dict[str, Any]) -> list[float]:
    if cfg.get("box_sizes"):
        return [float(x) for x in cfg["box_sizes"]]
    return [float(cfg.get("box_size", 20.0))]


def matrix_backends(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("backends") or ["pycharmm", "jaxmd"]
    out = []
    for b in raw:
        key = str(b).strip().lower()
        if key not in {"pycharmm", "jaxmd"}:
            raise ValueError(f"Unsupported backend {b!r}; use pycharmm or jaxmd")
        out.append(key)
    return out


def resolve_embedding(raw: str) -> tuple[str, str]:
    """Return ``(embedding_name, mm_charge_mode)`` for a config token."""
    key = str(raw).strip().lower().replace("-", "_")
    if key not in _EMBEDDING_ALIASES:
        raise ValueError(
            f"Unsupported embedding {raw!r}; use mechanical/electrostatic "
            f"(or fixed/q0). Known: {sorted(_EMBEDDING_ALIASES)}"
        )
    return _EMBEDDING_ALIASES[key]


def matrix_embeddings(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Embedding axis: mechanical (fixed) and/or electrostatic (q0)."""
    raw = cfg.get("embeddings")
    if raw is None:
        raw = cfg.get("mm_charge_modes")
    if raw is None:
        single = cfg.get("mm_charge_mode") or cfg.get("embedding")
        raw = [single] if single is not None else ["mechanical"]
    if isinstance(raw, str):
        raw = [raw]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        emb, mode = resolve_embedding(str(item))
        if emb in seen:
            continue
        seen.add(emb)
        out.append((emb, mode))
    if not out:
        raise ValueError("Matrix requires at least one embedding / mm_charge_mode")
    return out


def checkpoint_predicts_charges(raw: str) -> bool:
    """True when the portable JSON checkpoint enables an ML charge head."""
    import json

    path = resolve_checkpoint_path(raw)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("config", data) if isinstance(data, dict) else {}
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("charges") or cfg.get("predict_charges"))


def mm_charge_mode_needs_ml_charges(mode: str) -> bool:
    return str(mode).strip().lower() in {
        "q0",
        "latent",
        "q1",
        "fixed_plus_latent",
        "latent_dynamic",
    }


def iter_matrix_cells(cfg: dict[str, Any]) -> Iterator[RunCell]:
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", ["METH"])]
    if not solvents:
        raise ValueError("Matrix requires solvents (expected METH)")
    if matrix_uses_bulk_density(cfg):
        if cfg.get("cluster_sizes"):
            raise ValueError("Set either bulk_density_fractions or cluster_sizes, not both.")
    elif not cfg.get("cluster_sizes"):
        raise ValueError("Matrix requires cluster_sizes or bulk_density_fractions.")

    skip = {str(t).strip() for t in (cfg.get("exclude_run_tags") or [])}
    skip_charge_mismatch = bool(cfg.get("skip_embedding_without_charges", True))
    force_charge_mismatch = {
        str(t).strip() for t in (cfg.get("force_embedding_without_charges") or [])
    }
    seen: set[str] = set()
    for sol in solvents:
        for box in matrix_box_sizes(cfg):
            sizes = matrix_cluster_sizes_for_cell(cfg, solvent=sol, box_size=box)
            for n in sizes:
                for temp in matrix_temperatures(cfg):
                    for slug, ckpt in checkpoint_map(cfg).items():
                        for embedding, mm_mode in matrix_embeddings(cfg):
                            for backend in matrix_backends(cfg):
                                cell = RunCell(
                                    solvent=sol,
                                    n_monomers=int(n),
                                    temperature=float(temp),
                                    box_size=float(box),
                                    checkpoint_slug=ckpt_slug(slug),
                                    checkpoint=str(ckpt),
                                    backend=backend,
                                    embedding=embedding,
                                    mm_charge_mode=mm_mode,
                                )
                                tag = cell_run_tag(cell, cfg)
                                if tag in skip or tag in seen:
                                    continue
                                if (
                                    skip_charge_mismatch
                                    and tag not in force_charge_mismatch
                                    and mm_charge_mode_needs_ml_charges(mm_mode)
                                    and not checkpoint_predicts_charges(ckpt)
                                ):
                                    # DES dimers (charges: False) cannot do q0/ES.
                                    continue
                                seen.add(tag)
                                yield cell


def cell_run_tag(cell: RunCell, cfg: dict[str, Any] | None = None) -> str:
    del cfg  # tag always includes T/L/ckpt/embedding/backend for this workflow
    sol = solvent_slug(cell.solvent).lower()
    t = int(round(cell.temperature))
    box = int(round(cell.box_size))
    emb = _EMBEDDING_TAG.get(cell.embedding, ckpt_slug(cell.embedding))
    return (
        f"{sol}_{int(cell.n_monomers)}_t{t}_l{box}"
        f"_{cell.checkpoint_slug}_{emb}_{cell.backend}"
    )


def composition_string(cell: RunCell) -> str:
    return f"{solvent_slug(cell.solvent)}:{int(cell.n_monomers)}"


def cell_ml_atoms(cell: RunCell) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms

    return estimate_ml_atoms(cell.n_monomers, solvent=cell.solvent)


def cell_bulk_total(cell: RunCell, fraction: float = 1.0) -> int:
    """Monomer count at ``fraction`` × bulk liquid density for this cell's box."""
    return n_monomers_at_bulk_density(
        cell.solvent, cell.box_size, fraction, min_n=1
    )


def dense_cell_mlpot_overrides(cell: RunCell, cfg: dict[str, Any]) -> dict[str, Any]:
    """Size/density-aware MLpot flags (local copy; avoids burst campaign_lib import)."""
    n = int(cell.n_monomers)
    overrides: dict[str, Any] = {}
    bulk_n = cell_bulk_total(cell, 1.0)
    bulk_fraction = float(n) / float(max(1, bulk_n))
    dense = n >= 150 or bulk_fraction >= 0.75
    if dense:
        base_segments = int(cfg.get("n_heat_segments", 10))
        overrides["n_heat_segments"] = min(4, base_segments)
        base_interval = int(cfg.get("dynamics_overlap_check_interval", 250))
        overrides["dynamics_overlap_check_interval"] = max(500, base_interval)
        overrides["dynamics_overlap_memory_handoff"] = True
    if n >= 200:
        overrides["ml_batch_size"] = min(int(cfg.get("ml_batch_size", 512)), 128)
    return overrides


def run_output_dir(cfg: dict[str, Any], cell: RunCell) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_methane_ewald"))
    return (root / cell_run_tag(cell, cfg)).resolve()


def run_seed(cell: RunCell, *, seed_base: int = 4242) -> int:
    solvent_off = sum(ord(c) for c in solvent_slug(cell.solvent)) % 1000
    ckpt_off = sum(ord(c) for c in cell.checkpoint_slug) % 997
    backend_off = 17 if cell.backend == "jaxmd" else 0
    embedding_off = 29 if cell.embedding == "electrostatic" else 0
    return (
        int(seed_base)
        + int(cell.n_monomers) * 10000
        + solvent_off
        + int(round(cell.temperature)) * 19
        + int(round(cell.box_size)) * 131
        + ckpt_off
        + backend_off
        + embedding_off
    )


def leg_output_dir(cell_root: Path, job_id: str) -> str:
    return str((cell_root / job_id).resolve())


def _attach_leg_output_dir(job: dict[str, Any], cell_root: Path, job_id: str) -> dict[str, Any]:
    return {**job, "output_dir": leg_output_dir(cell_root, job_id)}


def _liquid_prep_defaults(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "liquid_prep": bool(cfg.get("liquid_prep", True)),
        "density_prep_ladder": bool(cfg.get("density_prep_ladder", True)),
        "density_prep_ladder_max_rounds": int(cfg.get("density_prep_ladder_max_rounds", 4)),
        "mc_density_equalize": bool(cfg.get("mc_density_equalize", True)),
        "mc_density_steps": int(cfg.get("mc_density_steps", 60)),
        "min_intermonomer_atom_distance": float(
            cfg.get("min_intermonomer_atom_distance", 1.0)
        ),
        "max_grms_before_dyn": float(cfg.get("max_grms_before_dyn", 50.0)),
    }
    if cfg.get("max_fmax_before_dyn_ev_A") is not None:
        flags["max_fmax_before_dyn_ev_A"] = float(cfg["max_fmax_before_dyn_ev_A"])
    if matrix_uses_bulk_density(cfg):
        # Match the fraction used to size N for this cell (smoke may be < 1.0).
        min_n = int(cfg.get("bulk_density_n_min", 1))
        max_raw = cfg.get("bulk_density_n_max")
        max_n = int(max_raw) if max_raw is not None else None
        frac_match: float | None = None
        for frac in (float(x) for x in (cfg.get("bulk_density_fractions") or [])):
            n = n_monomers_at_bulk_density(
                cell.solvent,
                cell.box_size,
                frac,
                min_n=min_n,
                max_n=max_n,
            )
            if n == int(cell.n_monomers):
                frac_match = float(frac)
                break
        if frac_match is None:
            bulk_n = n_monomers_at_bulk_density(
                cell.solvent, cell.box_size, 1.0, min_n=1
            )
            frac_match = float(cell.n_monomers) / float(max(1, bulk_n))
        flags["bulk_density_fraction"] = float(frac_match)
    return flags


def _ewald_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mm_nonbond_mode": str(cfg.get("mm_nonbond_mode", "periodic_external")),
        "lr_solver": str(cfg.get("lr_solver", "ewald")),
    }


def campaign_job_order(cfg: dict[str, Any], cell: RunCell | None = None) -> list[str]:
    backend = cell.backend if cell is not None else "pycharmm"
    if backend == "jaxmd":
        return ["pycharmm_init", "jaxmd_equi", "jaxmd_prod"]
    n_equi = int(cfg.get("pycharmm_equi_legs", 2))
    n_prod = int(cfg.get("pycharmm_prod_legs", 2))
    order = ["pycharmm_init"]
    order.extend(f"pycharmm_equi_{i:02d}" for i in range(1, n_equi + 1))
    order.extend(f"pycharmm_prod_{i:02d}" for i in range(1, n_prod + 1))
    return order


def build_campaign(cfg: dict[str, Any], cell: RunCell) -> dict[str, Any]:
    comp = composition_string(cell)
    cell_root = run_output_dir(cfg, cell)
    strategy = resolve_cleanup_strategy(cfg)
    heat_thermostat = resolve_pycharmm_heat_thermostat(cfg, strategy)
    repair = pycharmm_job_flags(strategy)
    repair.update(dense_cell_mlpot_overrides(cell, cfg))
    pretreat = pretreat_job_flags(strategy)
    jaxmd_extra = jaxmd_job_flags(strategy)
    liquid = _liquid_prep_defaults(cfg, cell)
    ewald = _ewald_defaults(cfg)
    ensemble = str(cfg.get("ensemble", "pbc_nvt"))
    ckpt = checkpoint_path_for_yaml(cell.checkpoint)

    defaults: dict[str, Any] = {
        "composition": comp,
        "checkpoint": ckpt,
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
        "ml_compute_dtype": str(cfg.get("ml_compute_dtype", "float64")),
        "ml_batch_size": int(
            dense_cell_mlpot_overrides(cell, cfg).get(
                "ml_batch_size", cfg.get("ml_batch_size", 512)
            )
        ),
        # mechanical → fixed CGenFF q; electrostatic → ML Q⁰ for E_MM Coulomb
        "mm_charge_mode": str(cell.mm_charge_mode),
        "handoff_write_res": True,
        "continue_velocities": True,
        "cleanup_strategy_name": strategy.name,
        "include_mm": True,
        **ewald,
        **liquid,
    }
    if bool(cfg.get("jax_mm_spoof", False)):
        defaults["jax_mm_spoof"] = True

    runs: dict[str, Any] = {
        "pycharmm_init": _attach_leg_output_dir(
            {
                "description": (
                    f"{comp} METH liquid T={cell.temperature:.0f}K "
                    f"L={cell.box_size:.0f}Å ckpt={cell.checkpoint_slug} "
                    f"emb={cell.embedding}/{cell.mm_charge_mode} "
                    f"backend={cell.backend} ewald init"
                ),
                "backend": "pycharmm",
                "setup": ensemble,
                "md_stages": "mini,heat",
                "ps_heat": float(cfg.get("ps_heat", 5.0)),
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

    if cell.backend == "jaxmd":
        runs["jaxmd_equi"] = _attach_leg_output_dir(
            {
                "description": f"{comp} JAX-MD NVT equil ({cfg.get('jaxmd_equi_ps', 10.0)} ps)",
                "backend": "jaxmd",
                "setup": ensemble,
                "ps": float(cfg.get("jaxmd_equi_ps", 10.0)),
                "depends_on": "pycharmm_init",
                **jaxmd_extra,
            },
            cell_root,
            "jaxmd_equi",
        )
        runs["jaxmd_prod"] = _attach_leg_output_dir(
            {
                "description": f"{comp} JAX-MD NVT production ({cfg.get('jaxmd_prod_ps', 20.0)} ps)",
                "backend": "jaxmd",
                "setup": ensemble,
                "ps": float(cfg.get("jaxmd_prod_ps", 20.0)),
                "depends_on": "jaxmd_equi",
                **jaxmd_extra,
            },
            cell_root,
            "jaxmd_prod",
        )
    else:
        prev = "pycharmm_init"
        n_equi = int(cfg.get("pycharmm_equi_legs", 2))
        equi_ps = float(cfg.get("pycharmm_equi_ps", 10.0))
        for i in range(1, n_equi + 1):
            jid = f"pycharmm_equi_{i:02d}"
            runs[jid] = _attach_leg_output_dir(
                {
                    "description": f"{comp} PyCHARMM equil {i}/{n_equi} ({equi_ps} ps)",
                    "backend": "pycharmm",
                    "setup": ensemble,
                    "md_stage": "equi",
                    "ps_equi": equi_ps,
                    "depends_on": prev,
                    **repair,
                },
                cell_root,
                jid,
            )
            prev = jid
        n_prod = int(cfg.get("pycharmm_prod_legs", 2))
        prod_ps = float(cfg.get("pycharmm_prod_ps", 20.0))
        for i in range(1, n_prod + 1):
            jid = f"pycharmm_prod_{i:02d}"
            runs[jid] = _attach_leg_output_dir(
                {
                    "description": f"{comp} PyCHARMM production {i}/{n_prod} ({prod_ps} ps)",
                    "backend": "pycharmm",
                    "setup": ensemble,
                    "md_stage": "prod",
                    "ps_prod": prod_ps,
                    "depends_on": prev,
                    **repair,
                },
                cell_root,
                jid,
            )
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
    final_id = campaign_job_order(cfg, cell)[-1]
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "final_handoff": out / final_id / "handoff" / "state.npz",
        "done": out / "done.txt",
    }


def cell_from_tag(cfg: dict[str, Any], tag: str) -> RunCell:
    by_tag = {cell_run_tag(c, cfg): c for c in iter_matrix_cells(cfg)}
    if tag not in by_tag:
        raise KeyError(f"run tag {tag!r} not in config matrix")
    return by_tag[tag]


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    cap = matrix_job_count(cfg)
    requested = int(cfg.get("slurm_max_concurrent", min(8, max(1, cap))))
    return max(1, min(requested, cap or 1))


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    n = slurm_max_concurrent(cfg)
    return f"gpu={n} charmm_slot={n}"


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    return slurm_max_concurrent(cfg)


def slurm_nodelist(cfg: dict[str, Any]) -> str:
    nodes = [str(x).strip() for x in (cfg.get("slurm_gpu_nodes_fast") or []) if str(x).strip()]
    explicit = str(cfg.get("slurm_nodelist", "") or "").strip()
    if explicit:
        return explicit
    return ",".join(nodes)


def methane_n_bulk(box_side_A: float, fraction: float = 1.0) -> int:
    return n_monomers_at_bulk_density("METH", box_side_A, fraction, min_n=1)
