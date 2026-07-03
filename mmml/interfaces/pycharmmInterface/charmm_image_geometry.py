"""CHARMM IMAGE (MKIMAT2) min-distance gates before MLpot USER/ENER."""

from __future__ import annotations

import argparse
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from mmml.utils.intermonomer_geometry import (
    DEFAULT_CHARMM_IMAGE_MLPOT_MIN_A,
    DEFAULT_PRE_MLPOT_OVERLAP_MIN_A,
)

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


def _resolve_atoms_per_for_image_gate(
    workflow_args: argparse.Namespace | None,
) -> list[int] | None:
    if workflow_args is not None:
        atoms_per = getattr(workflow_args, "_cluster_atoms_per_list", None)
        if atoms_per is not None:
            return [int(x) for x in atoms_per]
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
            atoms_per_monomer_from_psf,
        )

        return atoms_per_monomer_from_psf()
    except Exception:
        return None


def assert_charmm_image_mic_fallback(
    *,
    workflow_args: argparse.Namespace | None,
    box_side_A: float | None,
    min_distance_A: float,
    context: str,
) -> float:
    """MIC prep gate when ``<MKIMAT2>`` is unavailable (MPI / cached IMAGE lists)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array
    from mmml.utils.intermonomer_geometry import assert_pre_mlpot_mic_geometry

    atoms_per = _resolve_atoms_per_for_image_gate(workflow_args)
    if not atoms_per or box_side_A is None or float(box_side_A) <= 0.0:
        raise RuntimeError(
            f"{context}: cannot run MIC image fallback "
            "(missing atoms_per_monomer or cubic box side)."
        )
    pos = get_charmm_positions_array()
    z_arr = (
        getattr(workflow_args, "_cluster_atomic_numbers", None)
        if workflow_args is not None
        else None
    )
    worst = assert_pre_mlpot_mic_geometry(
        pos,
        atoms_per,
        box_side=float(box_side_A),
        use_pbc=True,
        args=workflow_args,
        atomic_numbers=z_arr,
        context=f"{context} (MIC fallback)",
    )
    print(
        f"{context}: MIC image fallback OK "
        f"(worst inter-monomer {worst:.2f} Å, prep floor {float(min_distance_A):.2f} Å)",
        flush=True,
    )
    return float(worst)


def _force_charmm_image_remap_for_probe() -> None:
    """Re-run ``image byres`` so the next UPDATE/ENER rebuilds ``<MKIMAT2>`` tables."""
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _image_setup_byres_all

    _image_setup_byres_all(0.0, 0.0, 0.0)


def _run_charmm_script_capture_fortran(script: str, *, replay: bool = True) -> str:
    """Run a CHARMM script and return captured Fortran stdout/stderr (fd-level)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        _restore_charmm_levels,
        _set_charmm_levels,
        capture_fortran_stdio,
    )

    old = _set_charmm_levels(prnlev=2, warnlev=5, bomlev=-2)
    tmp_path = ""
    try:
        with capture_fortran_stdio() as tmp_path:
            pycharmm.lingo.charmm_script(script)
        text = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        if replay and text:
            print(text, end="", flush=True)
        return text
    finally:
        _restore_charmm_levels(old)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _probe_command_via_charmm_log_file(command: str) -> str:
    """Run one CHARMM command with ``OUTU`` redirected to a temp file (MPI-safe)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        _restore_charmm_levels,
        _set_charmm_levels,
        charmm_relaxed_bomlev,
    )

    log_path = (
        Path(os.environ.get("TMPDIR", "/tmp"))
        / f"mmml-mkimat-{os.getpid()}-{uuid.uuid4().hex[:8]}.log"
    )
    path_quoted = str(log_path)
    script = (
        f'open unit 99 write form name "{path_quoted}"\n'
        "outu 99\n"
        "prnlev 2\n"
        "wrnlev 5\n"
        f"{command.strip()}\n"
        "close unit 99\n"
        "outu 6\n"
    )
    old = _set_charmm_levels(prnlev=2, warnlev=5, bomlev=-2)
    try:
        with charmm_relaxed_bomlev():
            pycharmm.lingo.charmm_script(script)
    finally:
        _restore_charmm_levels(old)
    if not log_path.is_file():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            log_path.unlink()
        except OSError:
            pass
    if text.strip():
        print(text, end="", flush=True)
    return text


def run_charmm_post_bimag_image_probe_log() -> str:
    """Collect ``<MKIMAT2>`` after ``update_bimag`` (ENER rebuilds IMAGE tables)."""
    chunks: list[str] = []
    for command in ("ener", "update"):
        file_log = _probe_command_via_charmm_log_file(command)
        if file_log:
            chunks.append(file_log)
        if parse_mkimat2_min_distances("\n".join(chunks)):
            return "\n".join(chunks)
        capture_log = _run_charmm_script_capture_fortran(command.upper(), replay=True)
        if capture_log:
            chunks.append(capture_log)
        if parse_mkimat2_min_distances("\n".join(chunks)):
            return "\n".join(chunks)
    return "\n".join(chunks)


def run_charmm_image_probe_log(*, post_bimag: bool = False) -> str:
    """Collect ``<MKIMAT2>`` from CHARMM output (file redirect + fd capture)."""
    if post_bimag:
        return run_charmm_post_bimag_image_probe_log()
    _force_charmm_image_remap_for_probe()
    chunks: list[str] = []

    for command in ("update", "ener"):
        file_log = _probe_command_via_charmm_log_file(command)
        if file_log:
            chunks.append(file_log)
        if parse_mkimat2_min_distances("\n".join(chunks)):
            return "\n".join(chunks)

    capture_log = _run_charmm_script_capture_fortran("UPDATE", replay=True)
    if capture_log:
        chunks.append(capture_log)
    if parse_mkimat2_min_distances("\n".join(chunks)):
        return "\n".join(chunks)

    ener_log = _run_charmm_script_capture_fortran("ENER", replay=True)
    if ener_log:
        chunks.append(ener_log)
    return "\n".join(chunks)


def run_charmm_update_capture_image_log() -> str:
    """Backward-compatible alias for :func:`run_charmm_image_probe_log`."""
    return run_charmm_image_probe_log()


def resolve_charmm_image_min_distance_A(
    workflow_args: argparse.Namespace | None,
) -> float:
    if workflow_args is None:
        return float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance

    return float(resolve_pre_mlpot_overlap_min_distance(workflow_args))


def resolve_mkimat2_min_distance_A(
    workflow_args: argparse.Namespace | None,
) -> float:
    """Floor for ``<MKIMAT2>`` group Min-Distance (≥ MIC prep; default 3.5 Å)."""
    if workflow_args is not None:
        explicit = getattr(workflow_args, "charmm_image_mlpot_min_distance_A", None)
        if explicit is not None:
            return float(explicit)
    mic_floor = resolve_charmm_image_min_distance_A(workflow_args)
    return max(float(mic_floor), float(DEFAULT_CHARMM_IMAGE_MLPOT_MIN_A))


def assert_charmm_image_min_distance_after_update(
    *,
    min_distance_A: float | None = None,
    workflow_args: argparse.Namespace | None = None,
    context: str = "MLpot PBC",
    charmm_log: str | None = None,
    cubic_box_side_A: float | None = None,
    post_bimag: bool = False,
) -> float:
    """Enforce IMAGE Min-Distance before MLpot USER/ENER (MKIMAT2 or MIC fallback)."""
    mkimat_floor = (
        float(min_distance_A)
        if min_distance_A is not None
        else resolve_mkimat2_min_distance_A(workflow_args)
    )
    mic_floor = resolve_charmm_image_min_distance_A(workflow_args)
    log = (
        charmm_log
        if charmm_log is not None
        else run_charmm_image_probe_log(post_bimag=post_bimag)
    )
    if parse_mkimat2_min_distances(log):
        worst = assert_charmm_image_min_distance(
            log,
            min_distance_A=mkimat_floor,
            context=context,
        )
        print(
            f"{context}: CHARMM IMAGE Min-Distance OK "
            f"(worst {worst:.2f} Å, MKIMAT2 floor {mkimat_floor:.2f} Å)",
            flush=True,
        )
        return worst

    print(
        f"{context}: <MKIMAT2> not emitted (MPI/cached IMAGE); "
        "using MIC prep gate fallback",
        flush=True,
    )
    return assert_charmm_image_mic_fallback(
        workflow_args=workflow_args,
        box_side_A=cubic_box_side_A,
        min_distance_A=mic_floor,
        context=context,
    )
