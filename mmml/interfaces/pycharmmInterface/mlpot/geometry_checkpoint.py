"""Geometry checkpoint ladder for pretreat resume and overlap/extent recovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import alternate_overlap_scratch
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles


def geometry_baseline_path(out_dir: Path, tag: str) -> Path:
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import geometry_baseline_res

    return geometry_baseline_res(out_dir)


def resolve_geometry_checkpoint_ladder(
    paths: dict[str, Path],
    tag: str,
    *,
    n_heat_segments: int = 1,
    prefer_heat_segments: bool = False,
) -> list[Path]:
    """Ordered restart candidates for fly-off recovery or campaign resume.

  When ``prefer_heat_segments`` is true (Snakemake retry), completed heat
  segment files rank above ``geometry_baseline``.  Otherwise baseline and
  bonded-mini snapshots precede pretreat MM checkpoints for MLpot recovery.
    """
    out_dir = Path(paths.get("heat_res", Path("."))).parent
    candidates: list[Path] = []

    heat_paths: list[Path] = []
    if n_heat_segments > 1:
        from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import stage_segment_restart

        for seg_i in range(n_heat_segments - 1, -1, -1):
            heat_paths.append(stage_segment_restart(out_dir, "heat", seg_i))
    else:
        heat_res = paths.get("heat_res")
        if heat_res is not None:
            heat_paths.append(Path(heat_res))

    baseline = paths.get("geometry_baseline_res")
    if baseline is not None:
        candidates.append(Path(baseline))

    mlpot_crd = paths.get("mlpot_mmml_crd") or paths.get("mini_crd")
    if mlpot_crd is not None:
        candidates.append(Path(mlpot_crd))

    legacy_mini_crd = paths.get("mini_crd")
    if legacy_mini_crd is not None:
        candidates.append(Path(legacy_mini_crd))

    bonded_crd = paths.get("bonded_mm_after_mini_crd")
    if bonded_crd is not None:
        candidates.append(Path(bonded_crd))

    if prefer_heat_segments:
        candidates = heat_paths + candidates
    else:
        candidates.extend(heat_paths)

    for key in (
        "charmm_mm_prod_res",
        "charmm_mm_equi_res",
        "charmm_mm_heat_res",
    ):
        p = paths.get(key)
        if p is not None:
            candidates.append(Path(p))

    # Removed peer directory fallback logic as per user request

    seen: set[str] = set()
    ordered: list[Path] = []
    for cand in candidates:
        key = str(cand.expanduser().resolve()) if cand else ""
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(Path(cand))
    return ordered


def is_overlap_scratch_restart_path(path: Path | str) -> bool:
    """True for alternating overlap chunk scratch files (not stage segment restarts)."""
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
        is_overlap_scratch_restart_path as _is_scratch,
    )

    return _is_scratch(path)


def is_pretreat_mm_restart_path(path: Path | str) -> bool:
    """True for CHARMM MM pretreat leg checkpoints (pre-MLpot topology/forces)."""
    p = Path(path)
    name = p.name
    if name.startswith("charmm_mm_") and name.endswith(".res"):
        return True
    return p.parent.name == "pretreat" and name.endswith(".res")


def is_handoff_seed_restart_path(path: Path | str) -> bool:
    """True for JAX/PyCHARMM handoff seeds (not valid overlap READYN sources)."""
    name = Path(path).name.lower()
    return name.startswith("continue_seed") and name.endswith(".res")


def is_heat_segment_restart_path(path: Path | str) -> bool:
    """True for multi-segment heat checkpoints (``heat.N.res``), not finals or scratch."""
    if is_overlap_scratch_restart_path(path):
        return False
    name = Path(path).name.lower()
    if name == "heat.res":
        return False
    if re.fullmatch(r"heat\.\d+\.res", name):
        return True
    return bool(re.fullmatch(r"heat_.+\.\d+\.res", name))


def build_geometry_recovery_candidates(overlap: Any) -> list[Path]:
    """Ordered restart ladder for overlap recovery (baseline before segment tails).

    Scratch ``.overlap_a/.b.res`` files are excluded: they carry stale CPT/image
    internals and must not be used for early-abort or pre-chunk geometry reload.
    """
    seen: set[str] = set()
    ordered: list[Path] = []

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        p = Path(path)
        if is_overlap_scratch_restart_path(p):
            return
        if is_pretreat_mm_restart_path(p):
            return
        if is_handoff_seed_restart_path(p):
            return
        key = str(p.expanduser())
        if key in seen:
            return
        seen.add(key)
        ordered.append(p)

    if overlap.geometry_baseline_restart is not None:
        add(overlap.geometry_baseline_restart)
    if overlap.prior_segment_restart is not None:
        add(overlap.prior_segment_restart)
    for cand in overlap.geometry_fallback_restarts:
        add(cand)
    return ordered


def build_extent_recovery_candidates(overlap: Any) -> list[Path]:
    """Recovery ladder for monomer fly-off: baseline/mini CRD, not heat segment tails.

    Heat segment checkpoints (``heat.N.res``) may still be valid CHARMM restarts after
    a dynamics blow-up; they must not rank above ``baseline.res`` or bonded-mini CRDs.
    """
    seen: set[str] = set()
    ordered: list[Path] = []

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        p = Path(path)
        if is_overlap_scratch_restart_path(p):
            return
        if is_pretreat_mm_restart_path(p):
            return
        if is_handoff_seed_restart_path(p):
            return
        if is_heat_segment_restart_path(p):
            return
        key = str(p.expanduser())
        if key in seen:
            return
        seen.add(key)
        ordered.append(p)

    if overlap.geometry_baseline_restart is not None:
        add(overlap.geometry_baseline_restart)

    for cand in overlap.geometry_fallback_restarts:
        p = Path(cand)
        if is_geometry_recovery_crd_path(p):
            add(p)

    prior = getattr(overlap, "prior_segment_restart", None)
    if prior is not None and not is_heat_segment_restart_path(prior):
        add(prior)

    for cand in overlap.geometry_fallback_restarts:
        p = Path(cand)
        if is_geometry_recovery_crd_path(p):
            continue
        add(cand)

    return ordered


def resolve_extent_recovery_source(
    candidates: list[Path] | tuple[Path, ...],
) -> Path | None:
    """First usable fly-off source for logging (restart preferred over CRD)."""
    return first_valid_restart_path(candidates) or first_valid_geometry_crd_path(candidates)


def build_early_abort_recovery_candidates(
    overlap: Any,
    *,
    overlap_restart_read: Path | str | None = None,
    segment_restart_read: Path | str | None = None,
) -> list[Path]:
    """Recovery ladder after a short overlap chunk abort.

    Unlike :func:`build_geometry_recovery_candidates`, the upcoming overlap
    ``READYN`` source (``.overlap_a/.b.res`` from the previous good chunk) is
    tried first so equi/prod does not rewind to an earlier stage (e.g. heat).
    ``segment_restart_read`` is the stage handoff checkpoint (e.g. equi ``.res``
    before prod overlap chunk 0).
    """
    seen: set[str] = set()
    ordered: list[Path] = []

    def add(path: Path | str | None, *, allow_scratch: bool = False) -> None:
        if path is None:
            return
        p = Path(path)
        if is_overlap_scratch_restart_path(p) and not allow_scratch:
            return
        if is_pretreat_mm_restart_path(p):
            return
        if is_handoff_seed_restart_path(p):
            return
        key = str(p.expanduser())
        if key in seen:
            return
        seen.add(key)
        ordered.append(p)

    if overlap_restart_read is not None:
        read = Path(overlap_restart_read)
        add(read, allow_scratch=True)
        if is_overlap_scratch_restart_path(read):
            alt = alternate_overlap_scratch(read)
            if alt is not None:
                add(alt, allow_scratch=True)

    if segment_restart_read is not None:
        add(segment_restart_read)

    for cand in build_geometry_recovery_candidates(overlap):
        add(cand)
    return ordered


def _early_abort_trust_in_memory(
    overlap: Any,
    *,
    integrated: int,
    chunk_nstep: int,
    chunk_index: int,
    cpt: bool,
    mlpot_ctx: Any | None,
) -> bool:
    """Whether in-memory CHARMM state is safe to continue after a mid-chunk abort."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        charmm_memory_coordinates_usable,
    )

    if not charmm_memory_coordinates_usable():
        return False

    if mlpot_ctx is None:
        if cpt and chunk_index == 0 and integrated <= 4:
            return False
        return integrated >= max(2, chunk_nstep // 10)

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_mlpot_grms_kcalmol_A,
    )

    n_mon = max(1, int(getattr(overlap, "n_monomers", 1) or 1))
    limit = max(50.0, 50.0 * (float(n_mon) ** 0.5))
    grms = resolve_mlpot_grms_kcalmol_A(
        mlpot_ctx,
        context="early-abort memory gate",
    )
    if grms > limit:
        print(
            f"early-abort memory gate: hybrid GRMS {grms:.2f} kcal/mol/Å > "
            f"{limit:.0f} — not continuing from in-memory state",
            flush=True,
        )
        return False
    return True


def is_geometry_recovery_crd_path(path: Path | str) -> bool:
    """True for CHARMM coordinate cards used in the geometry recovery ladder."""
    return Path(path).suffix.lower() == ".crd"


def first_valid_restart_path(candidates: list[Path] | tuple[Path, ...]) -> Path | None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import restart_has_nonfinite_coordinates
    import sys

    for cand in candidates:
        if is_geometry_recovery_crd_path(cand):
            continue
        valid = _valid_restart_file(cand)
        if valid is not None:
            if restart_has_nonfinite_coordinates(valid):
                print(f"WARN: Invalidating corrupt restart file {valid} (non-finite or all zero coords)", file=sys.stderr, flush=True)
                valid.rename(valid.with_suffix(".res.corrupt"))
                continue
            return valid
    return None


def first_valid_geometry_crd_path(candidates: list[Path] | tuple[Path, ...]) -> Path | None:
    """First non-empty ``.crd`` candidate in the recovery ladder."""
    for cand in candidates:
        p = Path(cand)
        if not is_geometry_recovery_crd_path(p):
            continue
        if p.is_file() and p.stat().st_size > 0:
            return p.resolve()
    return None


def write_geometry_baseline_restart(out_dir: Path, tag: str) -> Path | None:
    """Persist post-pretreat/post-mini CHARMM state for extent recovery."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

    path = geometry_baseline_path(out_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rewrite_dynamics_restart_validated(path):
        path.unlink(missing_ok=True)
        return None
    valid = _valid_restart_file(path)
    if valid is None:
        path.unlink(missing_ok=True)
        return None
    return valid


@dataclass(frozen=True)
class PretreatResumeState:
    skip_entire_pretreat: bool = False
    skip_minimize: bool = False
    skip_heat: bool = False
    skip_equi: bool = False
    restart_read: Path | None = None
    heat_integrated_step: int = 0


def _pretreat_expected_nstep(args: Any, *, timestep_ps: float, ps: float) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_settings,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import ps_to_nsteps

    pretreat = resolve_charmm_mm_pretreat_settings(args)
    dt_ps = float(pretreat.timestep_ps) if pretreat.timestep_ps > 0.0 else float(timestep_ps)
    return max(1, ps_to_nsteps(dt_ps, ps))


def _resolve_charmm_mm_pretreat_heat_nstep(args: Any, *, timestep_ps: float) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_heat_nstep,
        resolve_charmm_mm_pretreat_settings,
    )

    pretreat = resolve_charmm_mm_pretreat_settings(args)
    if pretreat.timestep_ps > 0.0:
        return resolve_charmm_mm_pretreat_heat_nstep(args, settings=pretreat)
    settings = pretreat.__class__(
        dt_fs=pretreat.dt_fs,
        timestep_ps=float(timestep_ps),
        temperature_K=pretreat.temperature_K,
        pressure_atm=pretreat.pressure_atm,
        ps_heat=pretreat.ps_heat,
        ps_equi=pretreat.ps_equi,
        ps_prod=pretreat.ps_prod,
    )
    return resolve_charmm_mm_pretreat_heat_nstep(args, settings=settings)


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

    if heat_res is not None:
        valid = first_valid_restart_path([Path(heat_res)])
        if valid is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                read_restart_last_step,
                resolve_integrated_restart_step,
            )

            step = resolve_integrated_restart_step(
                valid,
                expected_nstep=n_heat,
                min_step_fraction=0.0,
            )
            if step is None:
                step = read_restart_last_step(valid)
            min_complete = max(1, int(n_heat * 0.95)) - 1
            if step is not None and 0 < int(step) < min_complete:
                return PretreatResumeState(
                    skip_minimize=True,
                    restart_read=valid,
                    heat_integrated_step=int(step),
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
        from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
            geometry_baseline_res,
            pretreat_restart,
            stage_restart,
        )

        pretreat_dir = out_dir / "pretreat"
        paths = {
            "heat_res": stage_restart(out_dir, "heat"),
            "charmm_mm_prod_res": pretreat_restart(pretreat_dir, "prod"),
            "charmm_mm_equi_res": pretreat_restart(pretreat_dir, "equi"),
            "charmm_mm_heat_res": pretreat_restart(pretreat_dir, "heat"),
            "geometry_baseline_res": geometry_baseline_res(out_dir),
        }

    ladder = resolve_geometry_checkpoint_ladder(
        paths,
        tag,
        n_heat_segments=n_heat_segments,
        prefer_heat_segments=True,
    )
    found = first_valid_restart_path(ladder)
    if found is not None:
        baseline = paths.get("geometry_baseline_res")
        if is_handoff_seed_restart_path(found):
            if baseline is not None and first_valid_restart_path([Path(baseline)]) is not None:
                return Path(baseline)
            found = None
        elif (
            is_pretreat_mm_restart_path(found)
            and baseline is not None
            and first_valid_restart_path([Path(baseline)]) is not None
        ):
            return Path(baseline)
        if found is not None:
            return found

    summary = out_dir / "stage_summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            last_restart = payload.get("last_restart")
            if last_restart:
                valid = first_valid_restart_path([Path(last_restart)])
                if valid is not None and not is_handoff_seed_restart_path(valid):
                    return valid
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return None


def restore_geometry_from_ladder(
    candidates: list[Path] | tuple[Path, ...],
    *,
    label: str = "geometry recovery",
    allow_in_memory: bool = False,
) -> Path:
    """Load the first usable geometry source in ``candidates`` into CHARMM.

    Tries valid ``.res`` restarts first, then ``.crd`` coordinate cards (e.g.
    ``03_bonded_mm_after_mini_*.crd``), then optionally the active in-memory
    CHARMM coordinates when ``allow_in_memory`` is true.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        charmm_memory_coordinates_usable,
        restore_charmm_state_from_crd,
        restore_charmm_state_from_restart,
    )

    path = first_valid_restart_path(candidates)
    if path is not None:
        restore_charmm_state_from_restart(path)
        print(f"{label}: restored CHARMM state from {path.name}", flush=True)
        return path

    crd = first_valid_geometry_crd_path(candidates)
    if crd is not None:
        restore_charmm_state_from_crd(crd)
        print(
            f"{label}: restored CHARMM coordinates from {crd.name} (CRD fallback)",
            flush=True,
        )
        return crd

    if allow_in_memory and charmm_memory_coordinates_usable():
        print(
            f"{label}: using in-memory CHARMM coordinates "
            "(no valid restart/CRD on disk)",
            flush=True,
        )
        return Path("<in-memory>")

    raise RuntimeError(f"{label}: no valid restart/CRD in geometry checkpoint ladder")


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


GeometryRecoverySource = Literal["restart", "crd", "memory", "run_state"]


@dataclass(frozen=True)
class GeometryRecoveryResult:
    """Outcome of overlap geometry reload after a short dynamics abort."""

    ok: bool
    source: GeometryRecoverySource | None = None


def _geometry_recovery_source_from_path(path: Path) -> GeometryRecoverySource:
    if path == Path("<in-memory>"):
        return "memory"
    if path.suffix.lower() == ".crd":
        return "crd"
    return "restart"


def attempt_overlap_early_abort_recovery(
    overlap: Any,
    *,
    chunk_nstep: int,
    steps_done: int,
    steps_before_chunk: int,
    overlap_context: str,
    overlap_run_state_dir: Path | None = None,
    overlap_restart_read: Path | str | None = None,
    segment_restart_read: Path | str | None = None,
    mlpot_ctx: Any | None = None,
    cpt: bool = False,
    chunk_index: int = 0,
) -> GeometryRecoveryResult:
    """Reload geometry after a short chunk abort.

    When ``source`` is ``memory``, CHARMM still holds coordinates, velocities,
    and barostat state from the aborted chunk; callers must not Boltzmann-
    reassign velocities before retrying the chunk.

    ``overlap_restart_read`` is the scratch/file path the failed chunk would
  have ``READYN`` from (end of the previous overlap chunk).
    """
    if overlap is None or overlap.action != "rescue":
        return GeometryRecoveryResult(False)
    integrated = int(steps_done) - int(steps_before_chunk)
    if integrated <= 0:
        return GeometryRecoveryResult(False)

    candidates = build_early_abort_recovery_candidates(
        overlap,
        overlap_restart_read=overlap_restart_read,
        segment_restart_read=segment_restart_read,
    )

    label = f"early-abort recovery ({overlap_context})"
    try:
        path = restore_geometry_from_ladder(candidates, label=label, allow_in_memory=False)
        return GeometryRecoveryResult(True, _geometry_recovery_source_from_path(path))
    except RuntimeError:
        tried = ", ".join(p.name for p in candidates) or "(none)"
        print(
            f"{label}: no valid restart among {len(candidates)} candidate(s): {tried}",
            flush=True,
        )

    if overlap_run_state_dir is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
            restore_positions_from_overlap_run_state,
        )

        if restore_positions_from_overlap_run_state(overlap_run_state_dir, label=label):
            return GeometryRecoveryResult(True, "run_state")

    if _early_abort_trust_in_memory(
        overlap,
        integrated=integrated,
        chunk_nstep=int(chunk_nstep),
        chunk_index=int(chunk_index),
        cpt=bool(cpt),
        mlpot_ctx=mlpot_ctx,
    ):
        print(
            f"{label}: using in-memory CHARMM coordinates "
            "(no valid restart/CRD on disk; hybrid GRMS gate passed)",
            flush=True,
        )
        return GeometryRecoveryResult(True, "memory")

    return GeometryRecoveryResult(False)


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

    candidates = build_geometry_recovery_candidates(overlap)

    label = f"pre-chunk geometry reload ({overlap_context})"
    try:
        restore_geometry_from_ladder(candidates, label=label, allow_in_memory=True)
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
