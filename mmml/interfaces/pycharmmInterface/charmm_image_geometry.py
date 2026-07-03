"""CHARMM IMAGE (MKIMAT2) min-distance gates before MLpot USER/ENER."""

from __future__ import annotations

import argparse
import io
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

from mmml.utils.intermonomer_geometry import DEFAULT_PRE_MLPOT_OVERLAP_MIN_A

_MKIMAT2_MARKER = "<MKIMAT2>"
_MKIMAT2_ROW_RE = re.compile(
    r"^\s*\d+\s+\S+\s+has\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CharmmImageMinDistanceReport:
    distances: tuple[float, ...]
    worst: float | None


def parse_mkimat2_min_distances(charmm_log: str) -> list[float]:
    """Parse ``Min-Distance`` values from the latest ``<MKIMAT2>`` block."""
    text = str(charmm_log)
    marker = text.rfind(_MKIMAT2_MARKER)
    if marker < 0:
        return []
    block = text[marker:]
    distances: list[float] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            if distances:
                break
            continue
        if stripped.startswith("<") and _MKIMAT2_MARKER not in stripped:
            if distances:
                break
            continue
        if stripped.startswith("Transformation") or stripped.startswith("Total of"):
            continue
        match = _MKIMAT2_ROW_RE.match(line)
        if match is None:
            continue
        distances.append(float(match.group(1)))
    return distances


def summarize_mkimat2_min_distances(charmm_log: str) -> CharmmImageMinDistanceReport:
    distances = tuple(parse_mkimat2_min_distances(charmm_log))
    worst = min(distances) if distances else None
    return CharmmImageMinDistanceReport(distances=distances, worst=worst)


def assert_charmm_image_min_distance(
    charmm_log: str,
    *,
    min_distance_A: float = DEFAULT_PRE_MLPOT_OVERLAP_MIN_A,
    context: str = "MLpot PBC",
) -> float:
    """Abort when any ``<MKIMAT2>`` transformation reports an unsafe Min-Distance."""
    report = summarize_mkimat2_min_distances(charmm_log)
    if not report.distances:
        raise RuntimeError(
            f"{context}: no <MKIMAT2> Min-Distance rows in CHARMM output. "
            "IMAGE neighbor lists were not built — cannot verify ML-safe geometry "
            "before USER/ENER."
        )
    worst = float(report.worst)
    floor = float(min_distance_A)
    if worst + 1.0e-9 < floor:
        bad = [d for d in report.distances if d + 1.0e-9 < floor]
        raise RuntimeError(
            f"{context}: CHARMM IMAGE Min-Distance {worst:.2f} Å < prep floor "
            f"{floor:.2f} Å ({len(bad)} transformation(s) below floor; "
            f"values={', '.join(f'{d:.2f}' for d in sorted(bad))}). "
            "Repack/expand the box and delete stale prep_ladder checkpoints "
            "before MLpot registration."
        )
    return worst


def run_charmm_update_capture_image_log() -> str:
    """Run ``UPDATE`` and capture Fortran stdout (includes ``<MKIMAT2>``)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        _restore_charmm_levels,
        _set_charmm_levels,
    )

    buf = io.StringIO()
    old = _set_charmm_levels(prnlev=2, warnlev=5, bomlev=-2)
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            pycharmm.lingo.charmm_script("UPDATE")
    finally:
        _restore_charmm_levels(old)
    return buf.getvalue()


def resolve_charmm_image_min_distance_A(
    workflow_args: argparse.Namespace | None,
) -> float:
    if workflow_args is None:
        return float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance

    return float(resolve_pre_mlpot_overlap_min_distance(workflow_args))


def assert_charmm_image_min_distance_after_update(
    *,
    min_distance_A: float | None = None,
    workflow_args: argparse.Namespace | None = None,
    context: str = "MLpot PBC",
    charmm_log: str | None = None,
) -> float:
    """Run ``UPDATE`` (unless ``charmm_log`` given) and enforce IMAGE Min-Distance."""
    floor = (
        float(min_distance_A)
        if min_distance_A is not None
        else resolve_charmm_image_min_distance_A(workflow_args)
    )
    log = charmm_log if charmm_log is not None else run_charmm_update_capture_image_log()
    worst = assert_charmm_image_min_distance(
        log,
        min_distance_A=floor,
        context=context,
    )
    print(
        f"{context}: CHARMM IMAGE Min-Distance OK "
        f"(worst {worst:.2f} Å, prep floor {floor:.2f} Å)",
        flush=True,
    )
    return worst
