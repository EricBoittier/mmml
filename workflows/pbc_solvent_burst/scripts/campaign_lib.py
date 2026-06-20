"""Generate md-system campaign YAML for one PBC solvent burst matrix cell."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


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


def run_tag(solvent: str, n_monomers: int) -> str:
    return f"{solvent_slug(solvent).lower()}_{int(n_monomers)}"


def composition_string(solvent: str, n_monomers: int) -> str:
    return f"{solvent_slug(solvent)}:{int(n_monomers)}"


def run_output_dir(cfg: dict[str, Any], solvent: str, n_monomers: int) -> Path:
    root = repo_root() / str(cfg.get("output_root", "artifacts/pbc_solvent_burst"))
    return (root / run_tag(solvent, n_monomers)).resolve()


def run_seed(solvent: str, n_monomers: int, *, seed_base: int = 123456) -> int:
    solvent_off = sum(ord(c) for c in solvent_slug(solvent)) % 1000
    return int(seed_base) + int(n_monomers) * 10000 + solvent_off


def _pycharmm_repair_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "dynamics_overlap_action": str(cfg.get("dynamics_overlap_action", "rescue")),
        "dynamics_overlap_min_distance": float(cfg.get("dynamics_overlap_min_distance", 1.5)),
        "dynamics_intra_min_distance": float(cfg.get("dynamics_intra_min_distance", 0.5)),
        "dynamics_overlap_check_interval": int(cfg.get("dynamics_overlap_check_interval", 500)),
        "dynamics_overlap_charmm_sd_steps": int(
            cfg.get("dynamics_overlap_charmm_sd_steps", 200)
        ),
        "dynamics_overlap_charmm_abnr_steps": int(
            cfg.get("dynamics_overlap_charmm_abnr_steps", 400)
        ),
        "bonded_mm_mini": bool(cfg.get("bonded_mm_mini", True)),
        "bonded_mm_mini_after": str(cfg.get("bonded_mm_mini_after", "mini,heat")),
        "bonded_mm_mini_steps": int(cfg.get("bonded_mm_mini_steps", 100)),
        "charmm_pre_minimize": bool(cfg.get("charmm_pre_minimize", True)),
        "charmm_sd_steps": int(cfg.get("charmm_sd_steps", 200)),
        "charmm_abnr_steps": int(cfg.get("charmm_abnr_steps", 400)),
        "mini_nstep": int(cfg.get("mini_nstep", 150)),
        "dcd_nsavc": int(cfg.get("dcd_nsavc", 500)),
        "dyn_nprint": int(cfg.get("dyn_nprint", 500)),
    }


def _charmm_mm_pretreat_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Optional CGENFF + CHARMM-only heat/equi/prod before MLpot on ``pycharmm_init``."""
    if not bool(cfg.get("charmm_mm_pretreat", False)):
        return {}
    block: dict[str, Any] = {
        "charmm_mm_pretreat": True,
        "charmm_mm_pretreat_ps_heat": float(
            cfg.get("charmm_mm_pretreat_ps_heat", cfg.get("ps_heat", 30.0))
        ),
    }
    ps_equi = cfg.get("charmm_mm_pretreat_ps_equi")
    if ps_equi is not None:
        block["charmm_mm_pretreat_ps_equi"] = float(ps_equi)
    ps_prod = cfg.get("charmm_mm_pretreat_ps_prod")
    if ps_prod is not None:
        block["charmm_mm_pretreat_ps_prod"] = float(ps_prod)
    return block


def _jaxmd_burst_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "handoff_quality_gate": bool(cfg.get("handoff_quality_gate", True)),
        "handoff_quality_fmax_eVA": float(cfg.get("handoff_quality_fmax_eVA", 1.0)),
        "handoff_quality_action": str(cfg.get("handoff_quality_action", "minimize")),
        "jaxmd_minimize_steps": int(cfg.get("jaxmd_minimize_steps", 200)),
        "jaxmd_pbc_minimize_steps": int(cfg.get("jaxmd_pbc_minimize_steps", 200)),
        "dynamics_overlap_action": str(cfg.get("jaxmd_dynamics_overlap_action", "warn")),
        "dynamics_overlap_charmm_sd_steps": int(
            cfg.get("dynamics_overlap_charmm_sd_steps", 200)
        ),
        "dynamics_overlap_charmm_abnr_steps": int(
            cfg.get("dynamics_overlap_charmm_abnr_steps", 400)
        ),
        "extra_args": [
            "--steps-per-recording",
            str(int(cfg.get("steps_per_recording", 800))),
            "--jax-md-update-interval",
            str(int(cfg.get("jax_md_update_interval", 1))),
        ],
    }


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


def build_campaign(
    cfg: dict[str, Any],
    solvent: str,
    n_monomers: int,
) -> dict[str, Any]:
    """Build full campaign dict for one matrix cell."""
    sol = solvent_slug(solvent)
    n = int(n_monomers)
    comp = composition_string(sol, n)
    tag = run_tag(sol, n)
    seed = run_seed(sol, n, seed_base=int(cfg.get("seed_base", 123456)))
    burst_ps = float(cfg.get("jaxmd_burst_ps", 200.0))
    equi_ps = float(cfg.get("pycharmm_equi_ps", 10.0))
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    optional_sizes = {int(x) for x in (cfg.get("optional_sizes") or [])}

    defaults: dict[str, Any] = {
        "composition": comp,
        "checkpoint": str(cfg["checkpoint"]),
        "box_size": float(cfg["box_size"]),
        "spacing": float(cfg.get("spacing", 5.0)),
        "packmol_tolerance": float(cfg.get("packmol_tolerance", 1.0)),
        "dt_fs": float(cfg.get("dt_fs", 0.25)),
        "temperature": float(cfg.get("temperature", 300.0)),
        "pressure": float(cfg.get("pressure", 1.0)),
        "seed": seed,
        "mm_switch_on": float(cfg.get("mm_switch_on", 8.0)),
        "mm_switch_width": float(cfg.get("mm_switch_width", 5.0)),
        "ml_switch_width": float(cfg.get("ml_switch_width", 1.5)),
        "ml_gpu_count": int(cfg.get("ml_gpu_count", 1)),
        "ml_batch_size": int(cfg.get("ml_batch_size", 1024)),
        "handoff_write_res": True,
        "continue_velocities": True,
    }

    repair = _pycharmm_repair_block(cfg)
    pretreat = _charmm_mm_pretreat_block(cfg)
    jaxmd_extra = _jaxmd_burst_block(cfg)

    init_desc = f"{comp} PBC init: MLpot mini + gentle heat"
    if pretreat:
        init_desc = (
            f"{comp} PBC init: CHARMM MM pretreat + MLpot mini + gentle heat"
        )

    runs: dict[str, Any] = {
        "pycharmm_init": {
            "description": init_desc,
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "md_stages": "mini,heat",
            "ps_heat": float(cfg.get("ps_heat", 30.0)),
            "n_heat_segments": int(cfg.get("n_heat_segments", 8)),
            "heat_firstt": float(cfg.get("heat_firstt", 10.0)),
            "heat_finalt": float(cfg.get("heat_finalt", 300.0)),
            "heat_thermostat": str(cfg.get("heat_thermostat", "hoover")),
            **repair,
            **pretreat,
        },
        "pycharmm_equi_00": {
            "description": f"{comp} first NPT equil segment ({equi_ps} ps)",
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "md_stage": "equi",
            "ps_equi": equi_ps,
            "depends_on": "pycharmm_init",
            **repair,
        },
    }

    prev = "pycharmm_equi_00"
    for i in range(1, n_bursts + 1):
        burst_id = f"jaxmd_burst_{i:02d}"
        runs[burst_id] = {
            "description": f"{comp} JAX-MD burst {i}/{n_bursts} ({burst_ps} ps)",
            "backend": "jaxmd",
            "setup": "pbc_nvt",
            "ps": burst_ps,
            "depends_on": prev,
            **jaxmd_extra,
        }
        if i < n_bursts:
            equi_id = f"pycharmm_equi_{i:02d}"
            equi_job: dict[str, Any] = {
                "description": f"{comp} NPT equil after burst {i} ({equi_ps} ps)",
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "md_stage": "equi",
                "ps_equi": equi_ps,
                "depends_on": burst_id,
                **repair,
            }
            if n in optional_sizes:
                equi_job["optional"] = True
            runs[equi_id] = equi_job
            prev = equi_id
        else:
            burst_job = runs[burst_id]
            if n in optional_sizes:
                burst_job["optional"] = True
            prev = burst_id

    return {
        "defaults": defaults,
        "campaign_output": str(run_output_dir(cfg, sol, n)),
        "runs": runs,
    }


def write_campaign_yaml(
    cfg: dict[str, Any],
    solvent: str,
    n_monomers: int,
    *,
    out_dir: Path | None = None,
) -> Path:
    campaign = build_campaign(cfg, solvent, n_monomers)
    root = out_dir or run_output_dir(cfg, solvent, n_monomers)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(campaign, f, sort_keys=False, default_flow_style=False)
    return path


def build_md_system_campaign_argv(
    cfg: dict[str, Any],
    solvent: str,
    n_monomers: int,
    *,
    out_dir: Path | None = None,
) -> list[str]:
    root = out_dir or run_output_dir(cfg, solvent, n_monomers)
    campaign_path = write_campaign_yaml(cfg, solvent, n_monomers, out_dir=root)
    return [
        "--config",
        str(campaign_path),
        "--run-all",
        "--resume-campaign",
        "--campaign-output-dir",
        str(root),
    ]


def matrix_job_count(cfg: dict[str, Any]) -> int:
    return len(cfg.get("solvents", [])) * len(cfg.get("cluster_sizes", []))


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    """Concurrent Snakemake/Slurm jobs (gpu pool == charmm_slot pool)."""
    cap = matrix_job_count(cfg)
    requested = int(cfg.get("slurm_max_concurrent", cap))
    return max(1, min(requested, cap))


def total_jaxmd_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("jaxmd_burst_ps", 200.0)) * int(cfg.get("jaxmd_bursts", 5))


def total_pycharmm_equi_ps(cfg: dict[str, Any]) -> float:
    return float(cfg.get("pycharmm_equi_ps", 10.0)) * int(cfg.get("pycharmm_equi_legs", 5))


def paths_for_run(
    cfg: dict[str, Any],
    solvent: str,
    n_monomers: int,
) -> dict[str, Path]:
    out = run_output_dir(cfg, solvent, n_monomers)
    n_bursts = int(cfg.get("jaxmd_bursts", 5))
    return {
        "out_dir": out,
        "campaign_yaml": out / "campaign.yaml",
        "campaign_summary": out / "campaign_summary.json",
        "final_handoff": out / f"jaxmd_burst_{n_bursts:02d}" / "handoff" / "state.npz",
        "done": out / "done.txt",
    }
