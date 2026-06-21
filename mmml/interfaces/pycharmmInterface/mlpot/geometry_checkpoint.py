"""Geometry checkpoint ladder for pretreat resume and overlap/extent recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles


def geometry_baseline_path(out_dir: Path, tag: str) -> Path:
    return Path(out_dir) / f"geometry_baseline_{tag}.res"


def resolve_geometry_checkpoint_ladder(
    paths: dict[str, Path],
    tag: str,
    *,
    n_heat_segments: int = 1,
) -> list[Path]:
    """Ordered restart candidates for fly-off / early-abort recovery (pretreat included)."""
    out_dir = Path(paths.get("heat_res", Path("."))).parent
    candidates: list[Path] = []

    if n_heat_segments > 1:
        for seg_i in range(n_heat_segments - 1, -1, -1):
            candidates.append(out_dir / f"heat_{tag}.{seg_i}.res")
    else:
        heat_res = paths.get("heat_res")
        if heat_res is not None:
            candidates.append(Path(heat_res))

    for key in (
        "charmm_mm_prod_res",
        "charmm_mm_equi_res",
        "charmm_mm_heat_res",
        "geometry_baseline_res",
    ):
        p = paths.get(key)
        if p is not None:
            candidates.append(Path(p))

    bonded_crd = paths.get("bonded_mm_after_mini_crd")
    if bonded_crd is not None:
        candidates.append(Path(bonded_crd))

    seen: set[str] = set()
    ordered: list[Path] = []
    for cand in candidates:
        key = str(cand.expanduser().resolve()) if cand else ""
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(Path(cand))
    return ordered


def first_valid_restart_path(candidates: list[Path] | tuple[Path, ...]) -> Path | None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

    for cand in candidates:
        valid = _valid_restart_file(cand)
        if valid is not None:
            return valid
    return None


def write_geometry_baseline_restart(out_dir: Path, tag: str) -> Path | None:
    """Persist post-pretreat/post-mini CHARMM state for extent recovery."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )

    path = geometry_baseline_path(out_dir, tag)
    if not rewrite_dynamics_restart_validated(path):
        return None
    return path


@dataclass(frozen=True)
class PretreatResumeState:
    skip_entire_pretreat: bool = False
    skip_minimize: bool = False
    skip_heat: bool = False
    skip_equi: bool = False
    restart_read: Path | None = None


def _pretreat_expected_nstep(args: Any, *, timestep_ps: float, ps: float) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import ps_to_nsteps

    return max(1, ps_to_nsteps(timestep_ps, ps))


def _resolve_charmm_mm_pretreat_heat_nstep(args: Any, *, timestep_ps: float) -> int:
    ps_heat = getattr(args, "charmm_mm_pretreat_ps_heat", None)
    if ps_heat is not None and float(ps_heat) > 0.0:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import ps_to_nsteps

        return max(1, ps_to_nsteps(timestep_ps, float(ps_heat)))
    return max(1, int(getattr(args, "charmm_mm_pretreat_heat_nstep", 2000)))


def pretreat_stage_complete(
    restart_path: Path | None,
    *,
    expected_nstep: int,
    min_step_fraction: float = 0.95,
) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        resolve_integrated_restart_step,
    )

    if restart_path is None:
        return False
    valid = _valid_restart_file(restart_path)
    if valid is None:
        return False
    step = resolve_integrated_restart_step(
        valid,
        expected_nstep=expected_nstep,
        min_step_fraction=min_step_fraction,
    )
    if step is None:
        return False
    min_steps = max(1, int(expected_nstep * min_step_fraction))
    return int(step) >= min_steps - 1


def resume_charmm_mm_pretreat_if_available(
    paths: dict[str, Path],
    args: Any,
    *,
    timestep_ps: float,
) -> PretreatResumeState:
    """Determine which CHARMM MM pretreat legs can be skipped on retry."""
    ps_equi = float(getattr(args, "charmm_mm_pretreat_ps_equi", 0.0) or 0.0)
    ps_prod = float(getattr(args, "charmm_mm_pretreat_ps_prod", 0.0) or 0.0)
    n_heat = _resolve_charmm_mm_pretreat_heat_nstep(args, timestep_ps=timestep_ps)

    prod_res = paths.get("charmm_mm_prod_res")
    equi_res = paths.get("charmm_mm_equi_res")
    heat_res = paths.get("charmm_mm_heat_res")

    if ps_prod > 0.0 and pretreat_stage_complete(
        Path(prod_res) if prod_res else None,
        expected_nstep=_pretreat_expected_nstep(args, timestep_ps=timestep_ps, ps=ps_prod),
    ):
        return PretreatResumeState(
            skip_entire_pretreat=True,
            skip_minimize=True,
            skip_heat=True,
            skip_equi=True,
            restart_read=first_valid_restart_path([Path(prod_res)]),
        )

    if ps_equi > 0.0 and pretreat_stage_complete(
        Path(equi_res) if equi_res else None,
        expected_nstep=_pretreat_expected_nstep(args, timestep_ps=timestep_ps, ps=ps_equi),
    ):
        return PretreatResumeState(
            skip_minimize=True,
            skip_heat=True,
            skip_equi=True,
            restart_read=first_valid_restart_path([Path(equi_res)]),
        )

    if pretreat_stage_complete(
        Path(heat_res) if heat_res else None,
        expected_nstep=n_heat,
    ):
        return PretreatResumeState(
            skip_minimize=True,
            skip_heat=True,
            restart_read=first_valid_restart_path([Path(heat_res)]),
        )

    return PretreatResumeState()


def discover_resume_restart(
    out_dir: Path,
    tag: str,
    *,
    paths: dict[str, Path] | None = None,
    n_heat_segments: int = 1,
) -> Path | None:
    """Best on-disk restart for Snakemake / staged-workflow retry."""
    if paths is None:
        pretreat_dir = out_dir / "pretreat"
        paths = {
            "heat_res": out_dir / f"heat_{tag}.res",
            "charmm_mm_prod_res": pretreat_dir / f"charmm_mm_prod_{tag}.res",
            "charmm_mm_equi_res": pretreat_dir / f"charmm_mm_equi_{tag}.res",
            "charmm_mm_heat_res": pretreat_dir / f"charmm_mm_heat_{tag}.res",
            "geometry_baseline_res": geometry_baseline_path(out_dir, tag),
        }

    ladder = resolve_geometry_checkpoint_ladder(
        paths,
        tag,
        n_heat_segments=n_heat_segments,
    )
    found = first_valid_restart_path(ladder)
    if found is not None:
        return found

    summary = out_dir / "stage_summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            last_restart = payload.get("last_restart")
            if last_restart:
                valid = first_valid_restart_path([Path(last_restart)])
                if valid is not None:
                    return valid
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return None


def restore_geometry_from_ladder(
    candidates: list[Path] | tuple[Path, ...],
    *,
    label: str = "geometry recovery",
) -> Path:
    """Load the first valid restart in ``candidates`` into CHARMM."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    path = first_valid_restart_path(candidates)
    if path is None:
        raise RuntimeError(f"{label}: no valid restart in geometry checkpoint ladder")
    restore_charmm_state_from_restart(path)
    print(f"{label}: restored CHARMM state from {path.name}", flush=True)
    return path


def attach_geometry_checkpoints_to_overlap(
    overlap: Any,
    *,
    paths: dict[str, Path],
    tag: str,
    n_heat_segments: int = 1,
) -> Any:
    """Set baseline + fallback ladder on a ``DynamicsOverlapConfig``."""
    from dataclasses import replace

    out_dir = Path(paths.get("heat_res", Path("."))).parent
    baseline = paths.get("geometry_baseline_res")
    if baseline is None:
        baseline = geometry_baseline_path(out_dir, tag)
    ladder = tuple(
        resolve_geometry_checkpoint_ladder(paths, tag, n_heat_segments=n_heat_segments)
    )
    return replace(
        overlap,
        geometry_baseline_restart=Path(baseline) if baseline else None,
        geometry_fallback_restarts=ladder,
    )


def attempt_overlap_early_abort_recovery(
    overlap: Any,
    *,
    chunk_nstep: int,
    steps_done: int,
    steps_before_chunk: int,
    overlap_context: str,
    overlap_run_state_dir: Path | None = None,
) -> bool:
    """Reload geometry after a short chunk abort; return True when CHARMM was restored."""
    if overlap is None or overlap.action != "rescue":
        return False
    integrated = int(steps_done) - int(steps_before_chunk)
    threshold = max(1, int(chunk_nstep * 0.1))
    if integrated >= threshold:
        return False

    candidates: list[Path] = list(overlap.geometry_fallback_restarts)
    if overlap.geometry_baseline_restart is not None:
        candidates.append(Path(overlap.geometry_baseline_restart))
    if overlap.prior_segment_restart is not None:
        candidates.insert(0, Path(overlap.prior_segment_restart))

    label = f"early-abort recovery ({overlap_context})"
    try:
        restore_geometry_from_ladder(candidates, label=label)
        return True
    except RuntimeError:
        pass

    if overlap_run_state_dir is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
            restore_positions_from_overlap_run_state,
        )

        if restore_positions_from_overlap_run_state(overlap_run_state_dir, label=label):
            return True
    return False


def ensure_restartable_before_overlap_chunk(
    restart_path: Path | None,
    overlap: Any,
    *,
    overlap_context: str,
    overlap_run_state_dir: Path | None = None,
) -> None:
    """Reload from the geometry ladder when the latest restart is unusable."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        restart_has_nonfinite_coordinates,
    )

    if restart_path is None:
        return
    path = Path(restart_path)
    if _valid_restart_file(path) is not None and not restart_has_nonfinite_coordinates(path):
        return

    candidates: list[Path] = list(overlap.geometry_fallback_restarts)
    if overlap.geometry_baseline_restart is not None:
        candidates.append(Path(overlap.geometry_baseline_restart))
    if overlap.prior_segment_restart is not None:
        candidates.insert(0, Path(overlap.prior_segment_restart))

    label = f"pre-chunk geometry reload ({overlap_context})"
    try:
        restore_geometry_from_ladder(candidates, label=label)
        return
    except RuntimeError:
        pass

    if overlap_run_state_dir is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
            restore_positions_from_overlap_run_state,
        )

        if restore_positions_from_overlap_run_state(overlap_run_state_dir, label=label):
            return

    raise RuntimeError(
        f"{label}: latest restart {path.name} is not restartable and geometry ladder "
        "recovery failed"
    )
