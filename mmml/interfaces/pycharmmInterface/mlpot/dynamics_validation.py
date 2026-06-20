"""Post-dynamics checks so truncated runs do not print 'workflow OK'."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

_FMT_I = "<i"


def count_dcd_frames(path: Path) -> int:
    """Return frame count from a CHARMM/NAMD DCD header (0 if missing/invalid)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size < 16:
        return 0
    with p.open("rb") as f:
        f.seek(4)
        if f.read(4) != b"CORD":
            return 0
        return int(struct.unpack(_FMT_I, f.read(4))[0])


def count_readable_dcd_frames(path: Path) -> int:
    """Return frames actually readable from coordinate records (not header alone)."""
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        from mmml.utils.dcd_reader import scan_dcd_frame_count

        readable, _, _ = scan_dcd_frame_count(p)
        return int(readable)
    except (ValueError, struct.error, OSError):
        return 0


def overlap_chunk_dcd_paths(dcd_path: Path) -> list[Path]:
    """Sorted ``{stem}.chunk.*.dcd`` siblings for an overlap stage trajectory."""
    p = Path(dcd_path)
    return sorted(p.parent.glob(f"{p.stem}.chunk.*{p.suffix}"))


def count_overlap_chunk_dcd_frames(dcd_path: Path) -> tuple[int, int]:
    """Return ``(header_frames, readable_frames)`` summed over overlap chunk DCDs."""
    total_header = 0
    total_readable = 0
    for chunk_path in overlap_chunk_dcd_paths(dcd_path):
        header = count_dcd_frames(chunk_path)
        readable = count_readable_dcd_frames(chunk_path)
        total_header += header
        total_readable += min(header, readable) if header else readable
    return total_header, total_readable


def expected_dcd_frame_count(*, nstep: int, nsavc: int) -> int:
    """Minimum frames CHARMM should write (step 0 plus every ``nsavc`` steps)."""
    n = max(1, int(nstep))
    if n <= 1:
        # nstep=1 with nsavc=1 writes a single coordinate set (step 1), not step 0+1.
        return 1
    sav = max(1, min(int(nsavc), n - 1))
    return 1 + n // sav


def harmonize_nsavc_frequency(value: int, chunk_nstep: int) -> int:
    """Trajectory save interval: strictly less than ``nstep`` and (when possible) divides it."""
    n = max(1, int(chunk_nstep))
    if n <= 1:
        return max(1, int(value))
    cap = n - 1
    val = max(1, min(int(value), cap))
    if n % val == 0:
        return val
    for d in range(val, 0, -1):
        if n % d == 0:
            return d
    return 1


def expected_overlap_chunk_dcd_frame_count(
    *,
    total_nstep: int,
    nsavc: int,
    n_chunks: int,
    cold_start_first_chunk: bool = False,
) -> int:
    """Expected readable frames across per-chunk overlap DCDs.

    Overlap segments restart between chunks; CHARMM usually omits a duplicate
    origin frame on continuation chunks, so expect ``nstep // nsavc`` per chunk
    rather than ``1 + nstep // nsavc`` for every chunk.
    """
    n_chunks = max(1, int(n_chunks))
    total = max(1, int(total_nstep))
    chunk_nstep = max(1, total // n_chunks)
    # Use the configured nsavc (capped per chunk), not the harmonized divisor.
    # Overlap chunks restart without a duplicate origin frame on each segment.
    sav = max(1, min(int(nsavc), chunk_nstep - 1))
    per_restart = max(1, chunk_nstep // sav)
    if cold_start_first_chunk and n_chunks > 1:
        per_cold = expected_dcd_frame_count(nstep=chunk_nstep, nsavc=sav)
        return per_cold + per_restart * (n_chunks - 1)
    if cold_start_first_chunk:
        return expected_dcd_frame_count(nstep=chunk_nstep, nsavc=sav)
    return per_restart * n_chunks


def _parse_fortran_d_float(token: str) -> float:
    return float(token.upper().replace("D", "E"))


_FORTRAN_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+)[DEde][+-]?\d+",
    re.IGNORECASE,
)


def read_restart_natom(path: Path) -> int | None:
    """Return atom count from a CHARMM restart ``!NATOM`` header line."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    for i, raw in enumerate(lines):
        tag = raw.strip().split()[0] if raw.strip() else ""
        if not (tag.startswith("!NATOM") or tag.startswith("NATOM")):
            continue
        if i + 1 >= len(lines):
            return None
        fields = lines[i + 1].split()
        if not fields:
            return None
        try:
            return int(fields[0])
        except ValueError:
            return None
    return None


def _restart_coordinate_values(path: Path) -> list[float]:
    """Parse Cartesian values from the ``!X, Y, Z`` section of a restart file."""
    return _restart_section_values(path, "!X, Y, Z")


def _restart_section_values(path: Path, section_marker: str) -> list[float]:
    """Parse Fortran floats from a named restart section until the next ``!`` header."""
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()
    except OSError:
        return []
    values: list[float] = []
    in_section = False
    marker = section_marker.strip()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(marker) or (
            marker.upper() in line.upper() and line.startswith("!")
        ):
            in_section = True
            continue
        if in_section and line.startswith("!"):
            break
        if not in_section:
            continue
        for match in _FORTRAN_FLOAT_RE.finditer(line):
            values.append(_parse_fortran_d_float(match.group()))
    return values


def read_restart_velocities(path: Path) -> np.ndarray | None:
    """Return ``(N, 3)`` velocities from ``!VELOCITIES`` when present."""
    p = Path(path)
    natom = read_restart_natom(p)
    if natom is None or natom <= 0:
        return None
    flat = _restart_section_values(p, "!VELOCITIES")
    if len(flat) < 3 * natom:
        return None
    vel = np.asarray(flat[: 3 * natom], dtype=float).reshape(natom, 3)
    if not np.all(np.isfinite(vel)):
        return None
    return vel


def read_restart_coordinates(path: Path) -> np.ndarray | None:
    """Return ``(N, 3)`` Cartesian coordinates from a CHARMM restart file."""
    p = Path(path)
    natom = read_restart_natom(p)
    if natom is None or natom <= 0:
        return None
    flat = _restart_coordinate_values(p)
    if len(flat) < 3 * natom:
        return None
    pos = np.asarray(flat[: 3 * natom], dtype=float).reshape(natom, 3)
    if not np.all(np.isfinite(pos)):
        return None
    return pos


def restart_has_nonfinite_coordinates(path: Path | None) -> bool:
    """Return True when a restart file contains non-finite Cartesian coordinates."""
    if path is None:
        return False
    p = Path(path)
    if not p.is_file():
        return False
    if read_restart_coordinates(p) is not None:
        return False
    try:
        text = p.read_text(errors="ignore")
    except OSError:
        return False
    upper = text.upper()
    if "NAN" in upper or "INF" in upper:
        marker = "!X, Y, Z"
        idx = upper.find(marker)
        if idx >= 0 and ("NAN" in upper[idx:] or "INF" in upper[idx:]):
            return True
    for value in _restart_coordinate_values(p):
        if not np.isfinite(value):
            return True
    return False


def read_restart_last_step(path: Path) -> int | None:
    """Return the accumulated dynamics step from a CHARMM restart file.

    On the ``!NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,...`` line, use **JHSTRT**
    (6th integer, index 5) as the global step counter. ``NSTEP`` (index 2) is
    often the last segment length (e.g. 500 when overlap chunks use
    ``nstep=500``), not the total integrated steps — reading it caused false
    "restart step 500 < 8000" failures when ``JHSTRT`` was already 8000.

    Falls back to ``NSTEP`` when ``JHSTRT`` is absent, then legacy ``REST``.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    if not lines:
        return None

    for i, raw in enumerate(lines):
        tag = raw.strip().split()[0] if raw.strip() else ""
        if not (tag.startswith("!NATOM") or tag.startswith("NATOM")):
            continue
        if i + 1 >= len(lines):
            break
        fields = lines[i + 1].split()
        if len(fields) >= 6:
            try:
                jhstrt = int(fields[5])
                if jhstrt > 0:
                    return jhstrt
                # Coordinate-history restarts often leave JHSTRT=0; for a single
                # segment the NSTEP field still records the integrated length.
                nstep_field = int(fields[2])
                if nstep_field > 0:
                    return nstep_field
            except ValueError:
                pass
        if len(fields) >= 3:
            try:
                return int(fields[2])
            except ValueError:
                break

    header = lines[0].split()
    if len(header) >= 3 and header[0].upper() == "REST":
        try:
            return int(header[2])
        except ValueError:
            return None
    return None


def resolve_integrated_restart_step(
    path: Path | None,
    *,
    expected_nstep: int,
    min_step_fraction: float = 0.95,
) -> int | None:
    """Integrated dynamics step from restart, tolerating stale ``NSTEP`` sub-chunk fields."""
    if path is None:
        return None
    raw = read_restart_last_step(path)
    if raw is None:
        return None
    expected = max(1, int(expected_nstep))
    step = int(raw)
    min_steps = max(1, int(expected * min_step_fraction))
    if step >= min_steps - 1:
        return step
    if expected > step and step > 0 and expected % step == 0 and expected // step >= 2:
        return expected
    return step


def _field_span(line: str, index: int) -> tuple[int, int] | None:
    """Character span ``(start, end)`` of the ``index``-th whitespace-delimited field."""
    field = -1
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            return None
        start = i
        while i < n and not line[i].isspace():
            i += 1
        field += 1
        if field == index:
            return start, i
    return None


def _replace_field_preserve_width(
    line: str, index: int, value: int, *, min_width: int = 1
) -> str:
    """Replace one field in-place, keeping its original column width."""
    span = _field_span(line, index)
    if span is None:
        return line
    start, end = span
    width = max(min_width, end - start)
    new = f"{int(value):>{width}d}"
    if len(new) > width:
        new = new[-width:]
    return line[:start] + new + line[end:]


def _replace_i10_field(line: str, index: int, value: int) -> str:
    """Replace one Fortran ``I10`` field without disturbing trailing formatted data."""
    start = index * 10
    end = start + 10
    if len(line) < end:
        line = line.ljust(end)
    return line[:start] + f"{int(value):>10d}" + line[end:]


def patch_restart_global_step(path: Path, global_step: int) -> bool:
    """Set ``JHSTRT`` (and legacy ``REST`` header) to the integrated dynamics step.

    Intra overlap rescue reloads the PSF and clears CHARMM's in-memory step
    counter; ``WRITe restart`` then leaves ``JHSTRT=0``.  Patching restores
    correct global step accounting on the next ``READYN`` handoff.

    CHARMM restart files use fixed-width Fortran integers; never rewrite lines
    with ``" ".join(fields)`` or ``READYN`` fails with "Bad value during
    integer read".
    """
    p = Path(path)
    step = max(0, int(global_step))
    if not p.is_file():
        return False
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return False
    if not lines:
        return False

    patched = False
    if lines[0].strip().upper().startswith("REST"):
        # CHARMM REST line: A4 title + I10 step counter at column 11 (0-based offset 10).
        rest = lines[0]
        if len(rest) >= 20:
            lines[0] = _replace_i10_field(rest, 1, step)
        else:
            lines[0] = _replace_field_preserve_width(rest, 2, step, min_width=10)
        patched = True

    for i, raw in enumerate(lines):
        tag = raw.strip().split()[0] if raw.strip() else ""
        if not (tag.startswith("!NATOM") or tag.startswith("NATOM")):
            continue
        if i + 1 >= len(lines):
            break
        lines[i + 1] = _replace_i10_field(lines[i + 1], 5, step)
        patched = True
        break

    if not patched:
        return False
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def assert_stage_dynamics_completed(
    *,
    stage: str,
    expected_nstep: int,
    nsavc: int,
    dcd_path: Path | None,
    restart_path: Path | None = None,
    min_step_fraction: float = 0.95,
    min_frame_fraction: float = 0.90,
    allow_incomplete: bool = False,
) -> None:
    """Fail if dynamics stopped early or the stage DCD is nearly empty.

    CHARMM often exits quietly when ``echeck`` is exceeded (e.g. H dissociation /
    huge energy spikes during heating). Previously ``run_staged_workflow`` still
    printed 'Staged workflow OK'.
    """
    if allow_incomplete:
        return

    expected_nstep = max(1, int(expected_nstep))
    nsavc = max(1, int(nsavc))
    chunk_paths = overlap_chunk_dcd_paths(dcd_path) if dcd_path is not None else []
    if chunk_paths:
        expected_frames = expected_overlap_chunk_dcd_frame_count(
            total_nstep=expected_nstep,
            nsavc=nsavc,
            n_chunks=len(chunk_paths),
        )
    else:
        expected_frames = expected_dcd_frame_count(nstep=expected_nstep, nsavc=nsavc)
    min_frames = max(1, int(expected_frames * min_frame_fraction))
    if expected_frames >= 2:
        min_frames = max(2, min_frames)
    min_steps = max(1, int(expected_nstep * min_step_fraction))

    restart_step: int | None = None
    if restart_path is not None:
        restart_step = resolve_integrated_restart_step(
            restart_path,
            expected_nstep=expected_nstep,
            min_step_fraction=min_step_fraction,
        )

    header_frames = count_dcd_frames(dcd_path) if dcd_path is not None else 0
    readable_frames = (
        count_readable_dcd_frames(dcd_path) if dcd_path is not None else 0
    )
    if dcd_path is not None and header_frames == 0 and readable_frames == 0:
        chunk_header, chunk_readable = count_overlap_chunk_dcd_frames(dcd_path)
        if chunk_header or chunk_readable:
            header_frames = chunk_header
            readable_frames = chunk_readable
    n_frames = min(header_frames, readable_frames) if header_frames else readable_frames

    problems: list[str] = []
    if restart_step is not None and restart_step < min_steps:
        problems.append(
            f"restart step {restart_step} < {min_steps} "
            f"(expected ~{expected_nstep}; likely echeck or CHARMM abort)"
        )
    if restart_path is not None and restart_has_nonfinite_coordinates(restart_path):
        problems.append(
            f"restart {Path(restart_path).name} contains non-finite coordinates "
            f"(NaN/Inf; dynamics blew up despite completing {restart_step or '?'} steps)"
        )
    if dcd_path is not None and header_frames > 0 and readable_frames < header_frames:
        problems.append(
            f"DCD {dcd_path.name} header claims {header_frames} frame(s) but only "
            f"{readable_frames} are readable (truncated or corrupt file)"
        )
    if dcd_path is not None and n_frames < min_frames:
        label = (
            f"{len(chunk_paths)} overlap chunk DCD(s) for {dcd_path.name}"
            if chunk_paths
            else f"DCD {dcd_path.name}"
        )
        chunk_note = (
            f" ({len(chunk_paths)} chunks × ~{expected_frames // max(1, len(chunk_paths))} "
            f"frames/chunk at nsavc={nsavc})"
            if chunk_paths
            else f" at nsavc={nsavc}"
        )
        if restart_step is not None and restart_step >= min_steps:
            print(
                f"WARN: {stage.upper()} {label} has {n_frames} readable frame(s), "
                f"expected >= {min_frames} (~{expected_frames} total{chunk_note}), "
                f"but restart step {restart_step} >= {min_steps}; accepting segment "
                "from checkpoint (common after overlap/extent rescue in-memory refresh)",
                flush=True,
            )
        else:
            problems.append(
                f"{label} has {n_frames} readable frame(s), "
                f"expected >= {min_frames} (~{expected_frames} total{chunk_note})"
            )

    if not problems:
        print(
            f"{stage.upper()} complete: "
            f"restart_step={restart_step if restart_step is not None else '?'}, "
            f"dcd_frames={n_frames} readable"
            + (
                f" (header {header_frames})"
                if header_frames and header_frames != n_frames
                else ""
            )
            + f" (expected ~{expected_frames})",
            flush=True,
        )
        return

    hint = (
        "Check the log for 'ENERGY CHANGE TOLERANCE' / echeck and the last DYNA> line. "
        "ML USER-only NVE (no SHAKE) often needs --no-echeck or echeck >> 500 kcal/mol; "
        "inspect VMD for dissociation if the run continues but physics look wrong."
    )
    raise RuntimeError(
        f"{stage.upper()} dynamics incomplete: " + "; ".join(problems) + f". {hint}"
    )
