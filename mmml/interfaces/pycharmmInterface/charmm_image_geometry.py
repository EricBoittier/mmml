"""CHARMM IMAGE (MKIMAT2) min-distance gates before MLpot USER/ENER."""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmml.utils.intermonomer_geometry import (
    DEFAULT_CHARMM_IMAGE_MLPOT_MIN_A,
    DEFAULT_MIC_MKIMAT2_REGISTRATION_SLACK_A,
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


@dataclass(frozen=True)
class CharmmImageNbStats:
    """Sizes of CHARMM image nonbond exclusion buffers (post-UPIMNB/MKIMNB)."""

    natom: int
    natim: int
    ntrans: int
    nnb: int
    niminb: int
    iminb_capacity: int
    nimnb: int
    imjnb_capacity: int
    niming: int
    mlpot_active: bool

    @property
    def iminb_headroom(self) -> int:
        return max(0, int(self.iminb_capacity) - int(self.niminb))

    @property
    def iminb_tight(self) -> bool:
        cap = int(self.iminb_capacity)
        return cap > 0 and int(self.niminb) >= cap - 1


def fetch_charmm_image_nb_stats() -> CharmmImageNbStats | None:
    """Read image list sizes via ``pycharmm.image.get_iminb_stats()``."""
    try:
        import pycharmm.image as charmm_image
    except (ImportError, OSError):
        return None
    raw = charmm_image.get_iminb_stats()
    if raw is None:
        return None
    return CharmmImageNbStats(
        natom=int(raw["natom"]),
        natim=int(raw["natim"]),
        ntrans=int(raw["ntrans"]),
        nnb=int(raw["nnb"]),
        niminb=int(raw["niminb"]),
        iminb_capacity=int(raw["iminb_capacity"]),
        nimnb=int(raw["nimnb"]),
        imjnb_capacity=int(raw["imjnb_capacity"]),
        niming=int(raw["niming"]),
        mlpot_active=bool(raw["mlpot_active"]),
    )


def format_charmm_image_nb_stats(
    stats: CharmmImageNbStats,
    *,
    prefix: str = "CHARMM image NB",
) -> str:
    tight = " tight" if stats.iminb_tight else ""
    mlpot = " MLpot" if stats.mlpot_active else ""
    return (
        f"{prefix}: natom={stats.natom} natim={stats.natim} ntrans={stats.ntrans} "
        f"nnb={stats.nnb} niminb={stats.niminb}/{stats.iminb_capacity}{tight} "
        f"nimnb={stats.nimnb}/{stats.imjnb_capacity} niming={stats.niming}{mlpot}"
    )


def log_charmm_image_nb_stats(
    *,
    context: str = "CHARMM image NB",
    verbose: bool = True,
) -> CharmmImageNbStats | None:
    """Emit one-line image exclusion buffer stats when the C API is available."""
    if not verbose:
        return fetch_charmm_image_nb_stats()
    stats = fetch_charmm_image_nb_stats()
    if stats is None:
        return None
    print(format_charmm_image_nb_stats(stats, prefix=context), flush=True)
    return stats


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



def _resolve_atomic_numbers_for_image_gate(
    workflow_args: argparse.Namespace | None,
) -> np.ndarray | None:
    if workflow_args is not None:
        z = getattr(workflow_args, "_cluster_atomic_numbers", None)
        if z is not None:
            return np.asarray(z, dtype=int).reshape(-1)
        z = getattr(workflow_args, "ml_Z", None)
        if z is not None:
            return np.asarray(z, dtype=int).reshape(-1)
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

        return np.asarray(get_Z_from_psf(), dtype=int).reshape(-1)
    except Exception:
        return None


def assert_charmm_image_mic_fallback(
    *,
    workflow_args: argparse.Namespace | None,
    box_side_A: float | None,
    min_distance_A: float,
    context: str,
) -> float:
    """MIC prep gate when ``<MKIMAT2>`` is unavailable (MPI / cached IMAGE lists).

    Uses the same element-pair prep floors as Packmol/MC certification (≈2.3–2.9 Å),
    not the CHARMM ``<MKIMAT2>`` group Min-Distance margin (3.5–4.5 Å).
    """
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array
    from mmml.utils.intermonomer_geometry import (
        assert_pre_mlpot_mic_geometry,
        summarize_worst_intermonomer_contact,
    )

    atoms_per = _resolve_atoms_per_for_image_gate(workflow_args)
    if not atoms_per or box_side_A is None or float(box_side_A) <= 0.0:
        raise RuntimeError(
            f"{context}: cannot run MIC image fallback "
            "(missing atoms_per_monomer or cubic box side)."
        )
    prep_global = resolve_charmm_image_min_distance_A(workflow_args)
    pos = get_charmm_positions_array()
    z_arr = _resolve_atomic_numbers_for_image_gate(workflow_args)
    if z_arr is None:
        print(
            f"{context}: MIC fallback untyped (no element data); "
            f"using prep global floor {prep_global:.2f} Å only",
            flush=True,
        )
    worst = assert_pre_mlpot_mic_geometry(
        pos,
        atoms_per,
        box_side=float(box_side_A),
        use_pbc=True,
        args=workflow_args,
        atomic_numbers=z_arr,
        context=f"{context} (MIC prep element-pair floors)",
    )
    summary = summarize_worst_intermonomer_contact(
        pos,
        atoms_per,
        box_side=float(box_side_A),
        use_pbc=True,
        threshold_A=float(min_distance_A) if min_distance_A > 0 else prep_global,
        atomic_numbers=z_arr,
    )
    print(
        f"{context}: MIC image fallback OK ({summary.format_log_line()})",
        flush=True,
    )
    return float(worst)


_MKIMAT2_STASH_ATTR = "_charmm_mkimat2_registration_log"


def stash_mkimat2_registration_log(
    workflow_args: argparse.Namespace | None,
    charmm_log: str,
) -> None:
    """Retain a pre-MLpot ``<MKIMAT2>`` block for post-registration gates."""
    if workflow_args is None or not str(charmm_log).strip():
        return
    if not parse_mkimat2_min_distances(charmm_log):
        return
    setattr(workflow_args, _MKIMAT2_STASH_ATTR, str(charmm_log))


def _get_stashed_mkimat2_log(
    workflow_args: argparse.Namespace | None,
) -> str | None:
    if workflow_args is None:
        return None
    log = getattr(workflow_args, _MKIMAT2_STASH_ATTR, None)
    if log is None:
        return None
    text = str(log).strip()
    return text if text else None


def _mlpot_active_in_charmm() -> bool:
    stats = fetch_charmm_image_nb_stats()
    return bool(stats is not None and stats.mlpot_active)


def _image_probe_skip_hybrid_ener(
    workflow_args: argparse.Namespace | None,
    *,
    explicit: bool | None = None,
) -> bool:
    """True when CHARMM ``ENER`` would invoke the hybrid MLpot callback (avoid at registration)."""
    if explicit is not None:
        return bool(explicit)
    if _mlpot_active_in_charmm():
        return True
    return bool(getattr(workflow_args, "_mlpot_image_probe_skip_hybrid_ener", False))


def _force_charmm_image_remap_for_probe() -> None:
    """Re-run ``image byres`` so the next UPDATE/ENER rebuilds ``<MKIMAT2>`` tables."""
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _image_setup_byres_all

    _image_setup_byres_all(0.0, 0.0, 0.0)


def capture_charmm_script_output(script: str, *, replay: bool = True) -> str:
    """Run a CHARMM script and return captured Fortran stdout/stderr (fd-level)."""
    return _run_charmm_script_capture_fortran(script, replay=replay)


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
        if replay and text and not sys.platform.startswith("linux"):
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


def run_charmm_post_bimag_image_probe_log(*, skip_ener: bool = False) -> str:
    """Collect ``<MKIMAT2>`` after ``update_bimag`` (UPDATE emits IMAGE tables on fd 1)."""
    chunks: list[str] = []
    commands = ("update",) if skip_ener else ("update", "ener")
    for command in commands:
        capture_log = _run_charmm_script_capture_fortran(command.upper(), replay=True)
        if capture_log:
            chunks.append(capture_log)
        log_charmm_image_nb_stats(context=f"CHARMM image NB after {command.upper()}")
        if parse_mkimat2_min_distances("\n".join(chunks)):
            return "\n".join(chunks)
        file_log = _probe_command_via_charmm_log_file(command)
        if file_log:
            chunks.append(file_log)
        if parse_mkimat2_min_distances("\n".join(chunks)):
            return "\n".join(chunks)
    return "\n".join(chunks)


def run_charmm_image_probe_log(*, post_bimag: bool = False, skip_ener: bool = False) -> str:
    """Collect ``<MKIMAT2>`` from CHARMM output (file redirect + fd capture).

    When ``skip_ener`` is set (MLpot registered), never run CHARMM ``ENER`` — that
    would trigger a full hybrid JAX compile just to scrape IMAGE tables.
    """
    if post_bimag:
        return run_charmm_post_bimag_image_probe_log(skip_ener=skip_ener)
    _force_charmm_image_remap_for_probe()
    chunks: list[str] = []

    for command in ("update",):
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

    if skip_ener:
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
    """Floor for ``<MKIMAT2>`` group Min-Distance (default 1.0 Å)."""
    if workflow_args is not None:
        explicit = getattr(workflow_args, "charmm_image_mlpot_min_distance", None)
        if explicit is not None:
            return float(explicit)
    return float(DEFAULT_CHARMM_IMAGE_MLPOT_MIN_A)


def resolve_mic_registration_fallback_min_A(
    workflow_args: argparse.Namespace | None,
) -> float:
    """Global prep MIC floor for logging when ``<MKIMAT2>`` is unavailable.

    Atom-pair abort logic uses :func:`assert_pre_mlpot_mic_geometry` (element-aware
    prep floors), not the CHARMM ``<MKIMAT2>`` group margin from
    :func:`resolve_mkimat2_min_distance_A`.
    """
    return resolve_charmm_image_min_distance_A(workflow_args)


def run_mlpot_pbc_image_registration_gate(
    *,
    cubic_box_side_A: float,
    workflow_args: argparse.Namespace | None = None,
    context: str = "MLpot PBC registration (post-MLpot)",
    verbose: bool = False,
) -> float:
    """IMAGE gate after MLpot USER is registered (MKIMAT2 tables are built)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        rewrap_charmm_coords_for_mlpot_pbc,
    )

    side = float(cubic_box_side_A)
    rewrap_charmm_coords_for_mlpot_pbc(
        cubic_box_side_A=side,
        workflow_args=workflow_args,
        verbose=verbose,
    )
    with charmm_relaxed_bomlev():
        pycharmm.image.update_bimag()
        log_charmm_image_nb_stats(context="MLpot PBC registration image NB")
        update_log = capture_charmm_script_output("UPDATE", replay=False)
    if not parse_mkimat2_min_distances(update_log):
        file_log = _probe_command_via_charmm_log_file("update")
        if file_log:
            update_log = "\n".join(filter(None, [update_log, file_log]))
    stashed = _get_stashed_mkimat2_log(workflow_args)
    if not parse_mkimat2_min_distances(update_log) and stashed:
        if verbose:
            print(
                f"{context}: reusing pre-MLpot <MKIMAT2> log for post-registration gate",
                flush=True,
            )
        update_log = stashed
    pycharmm.image.update_bimag()
    image_log = update_log.strip()
    return assert_charmm_image_min_distance_after_update(
        workflow_args=workflow_args,
        context=context,
        cubic_box_side_A=side,
        charmm_log=image_log if parse_mkimat2_min_distances(image_log) else None,
        post_bimag=False,
        skip_hybrid_ener_probe=True,
    )


def assert_charmm_image_min_distance_after_update(
    *,
    min_distance_A: float | None = None,
    workflow_args: argparse.Namespace | None = None,
    context: str = "MLpot PBC",
    charmm_log: str | None = None,
    cubic_box_side_A: float | None = None,
    post_bimag: bool = False,
    skip_hybrid_ener_probe: bool | None = None,
) -> float:
    """Enforce IMAGE Min-Distance before MLpot USER/ENER (MKIMAT2 or MIC fallback)."""
    mkimat_floor = (
        float(min_distance_A)
        if min_distance_A is not None
        else resolve_mkimat2_min_distance_A(workflow_args)
    )
    skip_ener = _image_probe_skip_hybrid_ener(
        workflow_args,
        explicit=skip_hybrid_ener_probe,
    )
    if charmm_log is None:
        stashed = _get_stashed_mkimat2_log(workflow_args)
        if stashed and parse_mkimat2_min_distances(stashed):
            charmm_log = stashed
    log = (
        charmm_log
        if charmm_log is not None
        else run_charmm_image_probe_log(post_bimag=post_bimag, skip_ener=skip_ener)
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
        f"using MIC prep element-pair fallback "
        f"(global floor {resolve_mic_registration_fallback_min_A(workflow_args):.2f} Å)",
        flush=True,
    )
    return assert_charmm_image_mic_fallback(
        workflow_args=workflow_args,
        box_side_A=cubic_box_side_A,
        min_distance_A=resolve_mic_registration_fallback_min_A(workflow_args),
        context=context,
    )
