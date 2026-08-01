"""Restart / resume policy for the staged MD workflow.

Extracted from ``staged_workflow.py`` so it can be tested without CHARMM.

Everything here answers one question: **which restart file does a stage resume
from, and should the cold-start force gate apply?**  The answers are pure
functions of paths, filenames and flags -- no dynamics, no live CHARMM state --
but getting one wrong is expensive and silent:

* resuming ``equi`` from ``heat.res`` when ``nve.res`` exists throws away the
  whole NVE leg,
* applying the cold-start |F|max ceiling (meant for Packmol clash geometries)
  to an equilibrated liquid FIRE-minimises a valid finite-temperature structure
  into a different one,
* a segmented stage that resumes ``equi.res`` instead of ``equi.{N-1}.res``
  silently restarts from the first segment.

None of those raise; the trajectory is just wrong.  ``staged_workflow`` re-exports
these names, so existing imports keep working.

Covered by ``tests/unit/test_staged_workflow_helpers.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

MdStage = Literal["mini", "heat", "nve", "equi", "prod"]

__all__ = [
    "should_auto_resume_failed_staged_run",
]


def _equi_restart_name(tag: str, n_equi_segments: int) -> str:
    if n_equi_segments > 1:
        return f"equi.{n_equi_segments - 1}.res"
    return "equi.res"



# Cold-start |F|max≈2 eV/Å is for Packmol/clash geometries, not finite-T liquids
# that already completed heat/NVE.
_POST_DYNAMICS_RESUME_STAGES = frozenset({"equi", "nve", "prod"})


def _is_dynamics_stage_restart_path(path: Path | str | None) -> bool:
    """True for heat/nve/equi/prod stage restarts (not handoff/pretreat seeds)."""
    if path is None:
        return False
    p = Path(path)
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        is_handoff_seed_restart_path,
        is_heat_segment_restart_path,
        is_pretreat_mm_restart_path,
    )

    if is_handoff_seed_restart_path(p) or is_pretreat_mm_restart_path(p):
        return False
    name = p.name.lower()
    if name in {"heat.res", "nve.res", "equi.res", "prod.res"}:
        return True
    if is_heat_segment_restart_path(p):
        return True
    # Segmented equi/prod: equi.0.res, prod.3.res
    import re

    return bool(re.fullmatch(r"(equi|prod|nve)\.\d+\.res", name))


def _should_skip_pre_dyn_fmax_gate(
    *,
    seeded_from_dynamics_restart: bool,
    dyn_stages: list[str] | tuple[str, ...],
    restart_from: Path | str | None = None,
    handoff_coords_in_memory: bool = False,
) -> bool:
    """True when equi/NVE/prod resumes a finished dynamics restart.

    Skip even when offline coord seeding fails: the cold-start 2 eV/Å ceiling
    still must not FIRE a post-heat liquid, and EQUI CPT start loads coords
    from the restart before ``dyna``.

    Memory-handoff legs already placed finite-T coordinates in CHARMM (often
    with ``restart_from`` rewritten to ``continue_seed.res`` or a local
    ``baseline.res`` after prep). Treat that the same as a dynamics resume.
    """
    if not dyn_stages or dyn_stages[0] not in _POST_DYNAMICS_RESUME_STAGES:
        return False
    if seeded_from_dynamics_restart or handoff_coords_in_memory:
        return True
    if _is_dynamics_stage_restart_path(restart_from):
        return True
    if restart_from is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            is_handoff_seed_restart_path,
        )

        if is_handoff_seed_restart_path(restart_from):
            return True
    return False


def _restart_coord_read_candidates(path: Path) -> list[Path]:
    """User path plus CHARMM IO staging alias (may hold the only full copy)."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path | None) -> None:
        if candidate is None:
            return
        try:
            key = str(candidate.expanduser().resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(Path(candidate))

    _add(path)
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
            _staging_alias_for_restart,
        )

        _add(_staging_alias_for_restart(Path(path)))
    except Exception:
        pass
    return candidates


def _heat_restart_path(paths: dict[str, Path], tag: str, n_heat_segments: int) -> Path:
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import stage_segment_restart

    if n_heat_segments > 1:
        return stage_segment_restart(paths["heat_res"].parent, "heat", n_heat_segments - 1)
    return paths["heat_res"]


def _prior_restart_for_stage(
    stage: MdStage,
    paths: dict[str, Path],
    *,
    restart_from: Path | None,
    tag: str | None = None,
    n_heat_segments: int = 1,
) -> Path | None:
    if stage == "heat":
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            is_handoff_seed_restart_path,
            is_pretreat_mm_restart_path,
        )

        baseline = paths.get("geometry_baseline_res")
        if baseline is not None and Path(baseline).is_file():
            return Path(baseline)
        if restart_from is not None and not is_pretreat_mm_restart_path(restart_from):
            if not is_handoff_seed_restart_path(restart_from):
                return restart_from
        return None
    if restart_from is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            is_handoff_seed_restart_path,
            is_pretreat_mm_restart_path,
        )

        if is_handoff_seed_restart_path(restart_from) or is_pretreat_mm_restart_path(
            restart_from
        ):
            return None
        return restart_from
    if stage == "nve":
        heat_restart = _heat_restart_path(paths, tag or "", n_heat_segments)
        if heat_restart.is_file():
            return heat_restart
        if paths["heat_res"].is_file():
            return paths["heat_res"]
        return None
    if stage == "equi":
        if paths["nve_res"].is_file():
            return paths["nve_res"]
        # Segmented heat writes heat.{N-1}.res, not heat.res.
        heat_restart = _heat_restart_path(paths, tag or "", n_heat_segments)
        if heat_restart.is_file():
            return heat_restart
        if paths["heat_res"].is_file():
            return paths["heat_res"]
        return None
    if stage == "prod":
        return paths["equi_res"] if paths["equi_res"].is_file() else None
    return None


def _heat_in_place_restart(io: CharmmTrajectoryFiles) -> bool:
    """True when heat reads and writes the same ``.res`` (resume interrupted heat)."""
    if io.restart_read is None or io.restart_write is None:
        return False
    return Path(io.restart_read).resolve() == Path(io.restart_write).resolve()


def _trajectory_outputs(path: Path | None) -> list[Path]:
    """Existing non-empty DCD output for a stage (including overlap chunk files)."""
    if path is None:
        return []
    stage_path = Path(path)
    outputs: list[Path] = []
    if stage_path.is_file() and stage_path.stat().st_size > 0:
        outputs.append(stage_path)
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        overlap_chunk_dcd_paths,
    )

    for chunk_path in overlap_chunk_dcd_paths(stage_path):
        if chunk_path.is_file() and chunk_path.stat().st_size > 0:
            outputs.append(chunk_path)
    return outputs


def _should_seed_heat_prior_restart(
    *,
    seg_i: int,
    prev_restart_is_current_state: bool,
    use_memory: bool,
    memory_handoff_next: bool,
) -> bool:
    """True when heat starts from in-memory coords and needs a fly-off checkpoint."""
    if seg_i == 0 and prev_restart_is_current_state:
        return True
    return bool(use_memory and (seg_i == 0 or memory_handoff_next))


def _equi_in_place_restart(io: CharmmTrajectoryFiles) -> bool:
    """True when equi reads and writes the same ``.res`` (resume interrupted equi)."""
    if io.restart_read is None or io.restart_write is None:
        return False
    return Path(io.restart_read).resolve() == Path(io.restart_write).resolve()


def _valid_restart_file_lazy(path: Path):
    """Lazy shim for ``dynamics._valid_restart_file``.

    ``dynamics`` pulls in the CHARMM stack at import time; importing it at module
    scope here would undo the point of the extraction.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

    return _valid_restart_file(path)


def _can_seed_stage_from_memory(
    rread: Path | None,
    *,
    prev_restart: Path | None,
    prev_restart_is_current_state: bool,
) -> bool:
    """True when an invalid prior-stage restart can be replaced from live CHARMM state."""
    return (
        rread is not None
        and prev_restart is not None
        and prev_restart_is_current_state
        and Path(rread) == Path(prev_restart)
        and Path(rread).is_file()
        and _valid_restart_file_lazy(rread) is None
    )


def should_auto_resume_failed_staged_run(
    args: argparse.Namespace,
    *,
    out_dir: Path,
) -> bool:
    """Return True when a prior failed ``stage_summary.json`` should set ``restart_from``.

    ``--rebuild-packmol`` must not inherit a stale ``baseline.res`` from a previous
    failed attempt — that discards the freshly packed geometry and often flies off
    in CHARMM MM pretreat heat.
    """
    if getattr(args, "restart_from", None):
        return False
    if bool(getattr(args, "rebuild_packmol", False)):
        if not getattr(args, "quiet", False):
            summary_path = Path(out_dir) / "stage_summary.json"
            if summary_path.is_file():
                print(
                    "Skipping auto-resume: --rebuild-packmol requested "
                    "(ignoring prior stage_summary / baseline.res)",
                    flush=True,
                )
        return False
    summary_path = Path(out_dir) / "stage_summary.json"
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(payload.get("exit_code", 0)) != 0
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
