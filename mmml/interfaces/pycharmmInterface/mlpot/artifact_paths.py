"""Short on-disk artifact names (CHARMM Fortran path limits).

Each run uses its own output directory, so filenames omit composition tags.
VMD scripts reference basenames only (run ``vmd -e view.vmd.tcl`` from the job dir).
"""

from __future__ import annotations

import re
from pathlib import Path

PathLike = str | Path

MODEL_STEM = "model"
MINI_STEM = "mini"
PREP_LADDER_SUBDIR = "prep_ladder"
CLEANUP_SUBDIR = "cleanup"
SNAPSHOTS_JSON = "snapshots.json"
VMD_TCL = "view.vmd.tcl"
BASELINE_RES = "baseline.res"

_STAGE_NAMES = ("heat", "nve", "equi", "prod")


def model_psf(out_dir: PathLike) -> Path:
    return Path(out_dir) / f"{MODEL_STEM}.psf"


def model_pdb(out_dir: PathLike) -> Path:
    return Path(out_dir) / f"{MODEL_STEM}.pdb"


def stage_restart(out_dir: PathLike, stage: str) -> Path:
    return Path(out_dir) / f"{stage}.res"


def stage_dcd(out_dir: PathLike, stage: str) -> Path:
    return Path(out_dir) / f"{stage}.dcd"


def stage_segment_restart(out_dir: PathLike, stage: str, seg_i: int) -> Path:
    return Path(out_dir) / f"{stage}.{seg_i}.res"


def stage_pressure_tensor(out_dir: PathLike, stage: str) -> Path:
    return Path(out_dir) / f"{stage}.ptn"


def geometry_baseline_res(out_dir: PathLike) -> Path:
    return Path(out_dir) / BASELINE_RES


def pretreat_restart(pretreat_dir: PathLike, stage: str) -> Path:
    return Path(pretreat_dir) / f"{stage}.res"


def pretreat_dcd(pretreat_dir: PathLike, stage: str) -> Path:
    return Path(pretreat_dir) / f"{stage}.dcd"


def mini_paths(out_dir: PathLike) -> dict[str, Path]:
    """Canonical mini-stage mirror paths (``mini.*``)."""
    out = Path(out_dir)
    return {
        "mini_crd": out / f"{MINI_STEM}.crd",
        "mini_psf": out / f"{MINI_STEM}.psf",
        "mini_pdb": out / f"{MINI_STEM}.pdb",
        "mini_dcd": out / f"{MINI_STEM}.dcd",
        "mini_xyz": out / f"{MINI_STEM}.xyz",
        "mini_energy_json": out / f"{MINI_STEM}_energy.json",
    }


def overlap_restart_slot_paths(final_restart: PathLike) -> tuple[Path, Path]:
    """Alternating scratch restarts: ``heat.a.res`` / ``heat.b.res``."""
    p = Path(final_restart)
    base = p.stem
    return p.parent / f"{base}.a.res", p.parent / f"{base}.b.res"


def is_overlap_scratch_restart_name(name: str) -> bool:
    n = name.lower()
    if n.endswith(".overlap_a.res") or n.endswith(".overlap_b.res"):
        return True
    return bool(re.fullmatch(r".+\.[ab]\.res", n))


def is_overlap_scratch_restart_path(path: PathLike) -> bool:
    return is_overlap_scratch_restart_name(Path(path).name)


def alternate_overlap_scratch(path: PathLike) -> Path | None:
    """Return the paired A/B scratch restart, if ``path`` is overlap scratch."""
    p = Path(path)
    name = p.name
    if ".overlap_a." in name:
        return p.with_name(name.replace(".overlap_a.", ".overlap_b."))
    if ".overlap_b." in name:
        return p.with_name(name.replace(".overlap_b.", ".overlap_a."))
    if name.endswith(".a.res"):
        return p.with_name(f"{name[:-6]}.b.res")
    if name.endswith(".b.res"):
        return p.with_name(f"{name[:-6]}.a.res")
    return None


def overlap_chunk_trajectory_path(trajectory: PathLike, chunk_index: int) -> Path:
    """Per-chunk DCD: ``heat.0000.dcd`` (not ``heat.chunk.0000.dcd``)."""
    p = Path(trajectory)
    return p.with_name(f"{p.stem}.{chunk_index:04d}{p.suffix}")


def _is_numbered_chunk_dcd_name(name: str, stem: str, suffix: str) -> bool:
    return bool(re.fullmatch(re.escape(stem) + r"\.\d{4}" + re.escape(suffix), name))


def overlap_chunk_dcd_paths(dcd_path: PathLike) -> list[Path]:
    """Sorted per-chunk DCD siblings for an overlap stage trajectory."""
    p = Path(dcd_path)
    stem, suffix = p.stem, p.suffix
    new_paths = [
        cp
        for cp in sorted(p.parent.glob(f"{stem}.*{suffix}"))
        if _is_numbered_chunk_dcd_name(cp.name, stem, suffix)
    ]
    if new_paths:
        return new_paths
    return sorted(p.parent.glob(f"{stem}.chunk.*{suffix}"))


def overlap_chunk_dcd_glob_pattern(trajectory_stem: str) -> str:
    """Glob for cleanup of per-chunk DCDs (new + legacy)."""
    return f"{trajectory_stem}.*.dcd"


def staged_artifact_paths(out_dir: PathLike, tag: str) -> dict[str, Path]:
    """Standard staged-workflow artifact paths under ``out_dir``."""
    from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
        BONDED_MM_AFTER_HEAT,
        BONDED_MM_AFTER_MINI,
        CHARMM_MM_PRE,
        MLPOT_MMML,
        legacy_charmm_mm_dcd,
        snapshot_file_paths,
    )

    out = Path(out_dir)
    pretreat_dir = out / "pretreat"
    legacy = mini_paths(out)
    mm = snapshot_file_paths(pretreat_dir, CHARMM_MM_PRE, tag)
    mmml = snapshot_file_paths(out, MLPOT_MMML, tag)
    bonded_mini = snapshot_file_paths(out, BONDED_MM_AFTER_MINI, tag)
    bonded_heat = snapshot_file_paths(out, BONDED_MM_AFTER_HEAT, tag)
    return {
        **legacy,
        "mini_crd": legacy["mini_crd"],
        "mini_psf": legacy["mini_psf"],
        "mini_pdb": legacy["mini_pdb"],
        "mini_charmm_dcd": legacy_charmm_mm_dcd(pretreat_dir, tag),
        "mini_dcd": legacy["mini_dcd"],
        "charmm_mm_crd": mm["crd"],
        "charmm_mm_pdb": mm["pdb"],
        "charmm_mm_psf": mm["psf"],
        "charmm_mm_energy_json": mm["energy_json"],
        "mlpot_mmml_crd": mmml["crd"],
        "mlpot_mmml_pdb": mmml["pdb"],
        "mlpot_mmml_psf": mmml["psf"],
        "mlpot_mmml_dcd": mmml["dcd"],
        "mlpot_mmml_xyz": mmml["xyz"],
        "mlpot_mmml_energy_json": mmml["energy_json"],
        "bonded_mm_after_mini_crd": bonded_mini["crd"],
        "bonded_mm_after_mini_pdb": bonded_mini["pdb"],
        "bonded_mm_after_heat_crd": bonded_heat["crd"],
        "bonded_mm_after_heat_pdb": bonded_heat["pdb"],
        "charmm_mm_heat_res": pretreat_restart(pretreat_dir, "heat"),
        "charmm_mm_heat_dcd": pretreat_dcd(pretreat_dir, "heat"),
        "charmm_mm_equi_res": pretreat_restart(pretreat_dir, "equi"),
        "charmm_mm_equi_dcd": pretreat_dcd(pretreat_dir, "equi"),
        "charmm_mm_prod_res": pretreat_restart(pretreat_dir, "prod"),
        "charmm_mm_prod_dcd": pretreat_dcd(pretreat_dir, "prod"),
        "mini_box_equil_res": pretreat_restart(pretreat_dir, "mini_box_equil"),
        "mini_box_equil_dcd": pretreat_dcd(pretreat_dir, "mini_box_equil"),
        "geometry_baseline_res": geometry_baseline_res(out),
        "heat_res": stage_restart(out, "heat"),
        "heat_dcd": stage_dcd(out, "heat"),
        "nve_res": stage_restart(out, "nve"),
        "nve_dcd": stage_dcd(out, "nve"),
        "equi_res": stage_restart(out, "equi"),
        "equi_dcd": stage_dcd(out, "equi"),
        "equi_pressure_ptn": stage_pressure_tensor(out, "equi"),
        "prod_res": stage_restart(out, "prod"),
        "prod_dcd": stage_dcd(out, "prod"),
        "prod_pressure_ptn": stage_pressure_tensor(out, "prod"),
        "vmd_psf": model_psf(out),
    }


def resolve_topology_psf_candidates(
    psf_path: PathLike,
    *,
    tag: str | None = None,
) -> list[Path]:
    """PSF paths to try before MLpot re-registration (bond-safe topology)."""
    psf = Path(psf_path).expanduser().resolve()
    parent = psf.parent
    tags: list[str] = []
    if tag:
        tags.append(str(tag))
    if psf.name.startswith("mini_full_mlpot_"):
        derived = psf.name.replace("mini_full_mlpot_", "").replace(".psf", "")
        if derived and derived not in tags:
            tags.append(derived)

    candidates: list[Path] = [model_psf(parent)]
    for t in tags:
        legacy = parent / f"cluster_for_vmd_{t}.psf"
        if legacy not in candidates:
            candidates.append(legacy)
    if psf.name not in {c.name for c in candidates} and "mini" in psf.name.lower():
        pass  # already covered by model + legacy
    return candidates
