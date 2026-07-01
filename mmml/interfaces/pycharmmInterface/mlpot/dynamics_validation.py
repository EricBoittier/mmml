"""Post-dynamics checks so truncated runs do not print 'workflow OK'."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

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
    """Sorted per-chunk DCD siblings for an overlap stage trajectory."""
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import overlap_chunk_dcd_paths as _paths

    return _paths(dcd_path)


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


def resolve_target_dcd_nsavc(kw: dict) -> int | None:
    """Stage-level DCD save interval in integration steps (before per-chunk harmonization)."""
    if kw.get("_target_dcd_nsavc") is not None:
        return max(1, int(kw["_target_dcd_nsavc"]))
    interval_ps = kw.get("dcd_interval_ps")
    if interval_ps is None:
        interval_ps = kw.get("_dcd_interval_ps")
    timestep = kw.get("timestep")
    if interval_ps is not None and timestep is not None and float(timestep) > 0:
        return max(1, int(round(float(interval_ps) / float(timestep))))
    return None


def install_target_dcd_metadata(kw: dict) -> None:
    """Record user-intended DCD cadence so overlap/CPT chunks can normalize against it."""
    if kw.get("_target_dcd_nsavc") is not None:
        target = max(1, int(kw["_target_dcd_nsavc"]))
    else:
        interval_ps = kw.get("dcd_interval_ps")
        timestep = kw.get("timestep")
        if interval_ps is not None and timestep is not None and float(timestep) > 0:
            target = max(1, int(round(float(interval_ps) / float(timestep))))
        elif "nsavc" in kw:
            target = max(1, int(kw["nsavc"]))
        else:
            target = None
    if target is not None:
        kw["_target_dcd_nsavc"] = int(target)
    interval_ps = kw.get("dcd_interval_ps")
    if interval_ps is None:
        interval_ps = kw.get("_dcd_interval_ps")
    timestep = kw.get("timestep")
    if interval_ps is None and target is not None and timestep is not None:
        kw["_dcd_interval_ps"] = float(timestep) * int(target)
    elif interval_ps is not None:
        kw["_dcd_interval_ps"] = float(interval_ps)


def effective_dcd_interval_ps(*, nsavc: int, timestep_ps: float) -> float:
    """Physical time between DCD frames for a given ``nsavc``."""
    return max(0.0, float(timestep_ps)) * max(1, int(nsavc))


def nsavc_for_chunk_preserving_interval(
    target_nsavc: int,
    chunk_nstep: int,
    global_step_start: int = 0,
) -> int | None:
    """Per-chunk ``nsavc`` aligned to a global target cadence (IR / spectroscopy).

    CHARMM requires ``nsavc < nstep``. When a short overlap or CPT sub-chunk
    cannot honor the target interval without spurious frames, returns ``None`` so
    the caller can skip trajectory I/O for that segment.
    """
    target = max(1, int(target_nsavc))
    n = max(1, int(chunk_nstep))
    start = max(0, int(global_step_start))
    cap = max(1, n - 1)

    if target <= cap and n % target == 0:
        return target

    end = start + n
    aligned = [
        g
        for g in range(target, end + target, target)
        if start < g <= end
    ]

    if len(aligned) == 1:
        rel = aligned[0] - start
        if rel > cap:
            return None
        if n % rel == 0:
            return rel
        harmonized = harmonize_nsavc_frequency(rel, n)
        if harmonized == rel or (n // harmonized == 1 and start + harmonized == aligned[0]):
            return harmonized
        return None

    if not aligned:
        return None

    if target <= cap:
        harmonized = harmonize_nsavc_frequency(target, n)
        if harmonized == target or abs(harmonized - target) <= max(1, target // 10):
            return harmonized

    return harmonize_nsavc_frequency(min(target, cap), n)


def expected_overlap_chunk_dcd_frame_count(
    *,
    total_nstep: int,
    nsavc: int,
    n_chunks: int,
    cold_start_first_chunk: bool = False,
    integrated_step: int | None = None,
    n_completed_chunks: int | None = None,
    per_chunk_nsavc: Sequence[int] | None = None,
) -> int:
    """Expected readable frames across per-chunk overlap DCDs.

    Overlap segments restart between chunks; CHARMM usually omits a duplicate
    origin frame on continuation chunks, so expect ``nstep // nsavc`` per chunk
    rather than ``1 + nstep // nsavc`` for every chunk.

    When ``integrated_step`` is set (partial stage), frame expectations use the
    integrated step count, not the full ``total_nstep``.
    """
    n_chunks = max(1, int(n_chunks))
    total = max(1, int(total_nstep))
    if integrated_step is not None:
        integrated = max(0, min(int(integrated_step), total))
        if integrated <= 0:
            return 0
        sav = max(1, int(nsavc))
        return expected_dcd_frame_count(nstep=integrated, nsavc=sav)

    chunk_nstep = max(1, total // n_chunks)
    completed = n_completed_chunks if n_completed_chunks is not None else n_chunks
    completed = max(0, min(int(completed), n_chunks))
    if completed == 0:
        return 0

    if per_chunk_nsavc is not None and len(per_chunk_nsavc) >= completed:
        frames = 0
        for i in range(completed):
            sav_i = max(1, int(per_chunk_nsavc[i]))
            n_i = chunk_nstep if i < completed - 1 else max(
                1, total - chunk_nstep * (completed - 1)
            )
            if sav_i < n_i:
                frames += max(1, n_i // sav_i)
        return frames

    sav = max(1, int(nsavc))
    if sav >= chunk_nstep:
        return 0
    per_restart = max(1, chunk_nstep // sav)
    if cold_start_first_chunk and completed > 1:
        per_cold = expected_dcd_frame_count(nstep=chunk_nstep, nsavc=sav)
        return per_cold + per_restart * (completed - 1)
    if cold_start_first_chunk and completed == 1:
        return expected_dcd_frame_count(nstep=chunk_nstep, nsavc=sav)
    return per_restart * completed


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
    velocities = "VELOC" in marker.upper()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not in_section:
            if line.startswith(marker) or (
                marker.upper() in line.upper() and line.startswith("!")
            ):
                in_section = True
                continue
            if marker.upper().startswith("!X") and _is_restart_xyz_header(
                line, velocities=velocities
            ):
                in_section = True
                continue
            continue
        if line.startswith("!"):
            break
        for match in _FORTRAN_FLOAT_RE.finditer(line):
            values.append(_parse_fortran_d_float(match.group()))
    return values


def read_restart_velocities(path: Path) -> np.ndarray | None:
    """Return ``(N, 3)`` velocities from a CHARMM restart when present.

    CHARMM ``WRIDYN`` uses ``!VX, VY, VZ``; mmml-written restarts may use ``!VELOCITIES``.
    """
    p = Path(path)
    natom = read_restart_natom(p)
    if natom is None or natom <= 0:
        return None
    for marker in ("!VELOCITIES", "!VX, VY, VZ", "!VX,VY,VZ"):
        flat = _restart_section_values(p, marker)
        if len(flat) < 3 * natom:
            continue
        vel = np.asarray(flat[: 3 * natom], dtype=float).reshape(natom, 3)
        if not np.all(np.isfinite(vel)):
            continue
        return vel
    return None


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


def _parse_crd_xyz(parts: Sequence[str]) -> tuple[float, float, float] | None:
    # PyCHARMM write.coor_card EXT: index resid resname atomname x y z ...
    if len(parts) >= 7:
        try:
            return float(parts[4]), float(parts[5]), float(parts[6])
        except ValueError:
            pass
    if len(parts) >= 5:
        try:
            return float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            pass
    return None


def _is_restart_xyz_header(line: str, *, velocities: bool = False) -> bool:
    """Match ``!X, Y, Z`` / ``!X,Y,Z`` (and velocity variant) restart section headers."""
    u = line.strip().upper().replace(" ", "")
    if not u.startswith("!"):
        return False
    if velocities:
        return u.startswith("!VX") or "VELOC" in u
    if "OLD" in u or u.startswith("!VX") or u.startswith("!V"):
        return False
    return u.startswith("!X") and "Y" in u and "Z" in u


def read_crd_coordinates(path: Path) -> np.ndarray | None:
    """Return ``(N, 3)`` Cartesian coordinates from a CHARMM CRD card (EXT format)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None

    n_atoms: int | None = None
    header_idx: int | None = None
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 2 and parts[-1].upper() == "EXT":
            try:
                n_atoms = int(parts[0])
                header_idx = idx
                break
            except ValueError:
                continue
    if n_atoms is None or header_idx is None or n_atoms <= 0:
        return None

    coords: list[tuple[float, float, float]] = []
    for line in lines[header_idx + 1 :]:
        if len(coords) >= n_atoms:
            break
        parts = line.split()
        if not parts or parts[0].startswith("*"):
            continue
        xyz = _parse_crd_xyz(parts)
        if xyz is None:
            continue
        coords.append(xyz)
    if len(coords) != n_atoms:
        return None
    pos = np.asarray(coords, dtype=float)
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


def restart_coordinates_are_unsafe(
    path: Path | None,
    *,
    max_abs_A: float = 2000.0,
) -> bool:
    """True when restart coordinates are non-finite or unphysically large."""
    if path is None:
        return False
    p = Path(path)
    if not p.is_file():
        return False
    pos = read_restart_coordinates(p)
    if pos is None:
        return restart_has_nonfinite_coordinates(p)
    if not np.all(np.isfinite(pos)):
        return True
    return float(np.max(np.abs(pos))) > float(max_abs_A)


def charmm_coordinates_are_finite() -> bool:
    """True when all CHARMM Cartesian coordinates are finite."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    pos = coor.get_positions()
    arr = pos[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return True
    return bool(np.all(np.isfinite(arr)))


def charmm_coordinates_are_nontrivial(*, min_span_A: float = 1.0e-6) -> bool:
    """False when every atom is at the origin (blow-up / cleared CRD)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    pos = coor.get_positions()
    arr = pos[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return False
    span = float(np.max(arr) - np.min(arr))
    return span > float(min_span_A)


def charmm_dynamics_energy_is_finite() -> bool:
    """True when the current CHARMM energy row has no NaN/Inf scalars."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.energy as energy

    try:
        row = energy.get_energy().iloc[0]
    except Exception:
        return False
    for value in row.to_dict().values():
        if isinstance(value, (int, float, np.floating)):
            if not np.isfinite(float(value)):
                return False
    return True


# Integration blow-up can leave finite but unphysical energies (e.g. TOTKe ~1e11
# kcal/mol) while coordinates still sit inside the primary cell — must abort
# before ENER/UPDATE/mlpot_update (Fortran image rebuild segfault risk).
_DYNAMICS_ENERGY_ABS_MAX_KCALMOL = 1.0e8


def charmm_dynamics_energy_is_plausible(
    *,
    max_abs_kcalmol: float = _DYNAMICS_ENERGY_ABS_MAX_KCALMOL,
) -> bool:
    """False when any CHARMM energy scalar exceeds a sane magnitude."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.energy as energy

    limit = float(max_abs_kcalmol)
    if limit <= 0.0:
        return True
    try:
        row = energy.get_energy().iloc[0]
    except Exception:
        return False
    for value in row.to_dict().values():
        if isinstance(value, (int, float, np.floating)):
            v = float(value)
            if not np.isfinite(v) or abs(v) > limit:
                return False
    return True


def charmm_coordinates_are_bounded(*, max_abs_A: float = 2000.0) -> bool:
    """False when any atom coordinate magnitude exceeds a sane bound."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    pos = coor.get_positions()
    arr = pos[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return False
    return float(np.max(np.abs(arr))) <= float(max_abs_A)


def charmm_dynamics_state_is_finite() -> bool:
    """Coordinates and energy row are finite and physically plausible after dynamics."""
    return (
        charmm_coordinates_are_finite()
        and charmm_coordinates_are_nontrivial()
        and charmm_coordinates_are_bounded()
        and charmm_dynamics_energy_is_finite()
        and charmm_dynamics_energy_is_plausible()
    )


def validate_charmm_dynamics_state_after_chunk(
    *,
    context: str,
    restart_path: Path | None = None,
) -> None:
    """Raise when coordinates or energies are non-finite (barostat / MLpot blow-up)."""
    if restart_path is not None and restart_has_nonfinite_coordinates(Path(restart_path)):
        raise RuntimeError(
            f"{context}: restart {Path(restart_path).name} has non-finite coordinates "
            "after dynamics (MLpot integration blow-up). Do not continue integration — "
            "Fortran image updates can segfault on bad coordinates."
        )
    if restart_path is not None and restart_coordinates_are_unsafe(Path(restart_path)):
        raise RuntimeError(
            f"{context}: restart {Path(restart_path).name} has unphysical coordinates "
            f"(>|{2000.0:.0f}| Å) after dynamics (MLpot integration blow-up). "
            "Do not continue integration — Fortran image updates can segfault on "
            "fly-off coordinates."
        )
    if charmm_coordinates_are_finite() and charmm_dynamics_energy_is_finite():
        if not charmm_coordinates_are_nontrivial():
            raise RuntimeError(
                f"{context}: CHARMM coordinates are all zero after dynamics "
                "(MLpot integration blow-up). Refuse overlap rescue — fix heat "
                "thermostat (scale heat needs ihtfrq < nstep per overlap chunk; "
                "prefer --heat-thermostat hoover with overlap), shorten timestep, "
                "or run bonded-MM mini before heat."
            )
        if not charmm_coordinates_are_bounded():
            raise RuntimeError(
                f"{context}: CHARMM coordinates exceed {2000.0:.0f} Å after dynamics "
                "(MLpot integration blow-up). Refuse overlap rescue — ensure scale heat "
                "passes ihtfrq/teminc on the dynamics line (not after), cap ihtfrq "
                "< chunk nstep, or use --heat-thermostat hoover."
            )
        if not charmm_dynamics_energy_is_plausible():
            raise RuntimeError(
                f"{context}: CHARMM energies exceed "
                f"{_DYNAMICS_ENERGY_ABS_MAX_KCALMOL:.0e} kcal/mol after dynamics "
                "(MLpot integration blow-up). Shorten the timestep, tighten echeck, "
                "or verify MLpot timestep is applied (not leftover CHARMM pretreat dt)."
            )
        return
    raise RuntimeError(
        f"{context}: CHARMM dynamics produced non-finite coordinates or energy "
        "(NPT barostat instability or MLpot blow-up). Use a shorter timestep, "
        "tighter echeck, or rely on CPT stability chunking; do not continue "
        "integration — Fortran image updates can segfault on NaN coordinates."
    )


def assert_charmm_dynamics_chunk_safe(
    *,
    context: str,
    restart_path: Path | None = None,
) -> None:
    """Log a warning when state looks corrupt, then raise before list rebuild / mlpot_update."""
    memory_bad = not charmm_dynamics_state_is_finite()
    restart_bad = restart_path is not None and (
        restart_has_nonfinite_coordinates(Path(restart_path))
        or restart_coordinates_are_unsafe(Path(restart_path))
    )
    if memory_bad or restart_bad:
        print(
            f"WARN: {context}: unsafe CHARMM coordinates/energy after dynamics chunk",
            flush=True,
        )
    validate_charmm_dynamics_state_after_chunk(
        context=context,
        restart_path=restart_path,
    )


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

    # Check for negative step counter on the REST line first (indicates an abort)
    if lines[0].strip().upper().startswith("REST"):
        header = lines[0].split()
        if len(header) >= 3:
            try:
                val = int(header[2])
                if val < 0:
                    return val
            except ValueError:
                pass

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


@dataclass(frozen=True)
class ChunkOutcome:
    """Trusted chunk integration outcome for overlap abort decisions."""

    kind: Literal["ok", "charmm_aborted", "geometry_violation"]
    integrated_step: int
    charmm_aborted: bool
    geometry_violation: bool
    header_misread: bool = False


def classify_chunk_outcome(
    *,
    steps_before_chunk: int,
    chunk_nstep: int,
    reported_steps: int,
    chunk_state_corrupt: bool,
    restart_path: Path | None,
    overlap_context: str = "dynamics",
    chunk_index: int = 0,
    n_chunks: int = 1,
) -> ChunkOutcome:
    """Classify overlap chunk result using corroborated CHARMM abort signals.

    Restart-header shortfall alone is **not** treated as ``charmm_aborted``; stale
    ``JHSTRT``/``NSTEP`` on scratch ``.a/.b.res`` files caused false recovery.
    """
    expected_after = int(steps_before_chunk) + int(chunk_nstep)
    reported = max(0, int(reported_steps))
    header_step: int | None = None
    if restart_path is not None:
        header_step = read_restart_last_step(Path(restart_path))

    charmm_aborted = bool(chunk_state_corrupt)
    if header_step is not None and header_step < 0:
        charmm_aborted = True

    header_misread = False
    shortfall = reported < expected_after - 1

    if charmm_aborted and shortfall:
        integrated = reported
    elif shortfall:
        print(
            f"overlap ({overlap_context}): chunk {chunk_index + 1}/{n_chunks} "
            f"restart header reports step {reported} < {expected_after - 1} "
            f"without CHARMM abort signal; trusting completed chunk accounting",
            flush=True,
        )
        integrated = expected_after
        header_misread = True
    elif reported >= expected_after - 1:
        integrated = max(reported, expected_after)
    else:
        integrated = reported

    return ChunkOutcome(
        kind="charmm_aborted" if charmm_aborted else "ok",
        integrated_step=integrated,
        charmm_aborted=charmm_aborted,
        geometry_violation=False,
        header_misread=header_misread,
    )


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
    segment_note: str | None = None,
    integrated_step: int | None = None,
    salvaged_partial: bool = False,
) -> None:
    """Fail if dynamics stopped early or the stage DCD is nearly empty.

    CHARMM often exits quietly when ``echeck`` is exceeded (e.g. H dissociation /
    huge energy spikes during heating). Previously ``run_staged_workflow`` still
    printed 'Staged workflow OK'.
    """
    if allow_incomplete:
        return

    if salvaged_partial and integrated_step is not None:
        raise RuntimeError(
            f"{stage.upper()} dynamics incomplete: salvaged partial segment at step "
            f"{integrated_step}/{expected_nstep} — equi/prod blocked unless "
            "--allow-incomplete-dynamics"
        )

    expected_nstep = max(1, int(expected_nstep))
    nsavc = max(1, int(nsavc))
    chunk_paths = overlap_chunk_dcd_paths(dcd_path) if dcd_path is not None else []
    step_for_frames = integrated_step if integrated_step is not None else expected_nstep
    if chunk_paths:
        expected_frames = expected_overlap_chunk_dcd_frame_count(
            total_nstep=expected_nstep,
            nsavc=nsavc,
            n_chunks=len(chunk_paths),
            integrated_step=step_for_frames if step_for_frames < expected_nstep else None,
            n_completed_chunks=(
                max(1, (int(step_for_frames) * len(chunk_paths)) // expected_nstep)
                if step_for_frames < expected_nstep
                else None
            ),
        )
    else:
        expected_frames = expected_dcd_frame_count(
            nstep=step_for_frames, nsavc=nsavc
        )
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
    effective_step = integrated_step
    if effective_step is None and restart_path is not None:
        effective_step = restart_step
    if effective_step is not None and effective_step < min_steps:
        problems.append(
            f"integrated step {effective_step} < {min_steps} "
            f"(expected ~{expected_nstep}; likely echeck or CHARMM abort)"
        )
    elif restart_step is not None and restart_step < min_steps and integrated_step is None:
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
        if integrated_step is not None and integrated_step >= min_steps:
            print(
                f"WARN: {stage.upper()} {label} has {n_frames} readable frame(s), "
                f"expected >= {min_frames} (~{expected_frames} total{chunk_note}), "
                f"but trusted integrated step {integrated_step} >= {min_steps}; "
                "accepting segment from completed dynamics accounting",
                flush=True,
            )
        elif restart_step is not None and restart_step >= min_steps:
            if integrated_step is not None and integrated_step < min_steps:
                problems.append(
                    f"{label} has {n_frames} readable frame(s) but integrated step "
                    f"{integrated_step} < {min_steps}"
                )
            else:
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
        stage_label = stage.upper()
        if segment_note:
            stage_label = f"{stage_label} {segment_note}"
        print(
            f"{stage_label} complete: "
            f"restart_step={restart_step if restart_step is not None else '?'}, "
            f"integrated_step={integrated_step if integrated_step is not None else effective_step if effective_step is not None else '?'}, "
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
