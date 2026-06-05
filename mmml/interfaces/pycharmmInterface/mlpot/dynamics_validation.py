"""Post-dynamics checks so truncated runs do not print 'workflow OK'."""

from __future__ import annotations

import struct
from pathlib import Path

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


def expected_dcd_frame_count(*, nstep: int, nsavc: int) -> int:
    """Minimum frames CHARMM should write (step 0 plus every ``nsavc`` steps)."""
    n = max(1, int(nstep))
    sav = max(1, min(int(nsavc), n - 1))
    return 1 + n // sav


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


def patch_restart_global_step(path: Path, global_step: int) -> bool:
    """Set ``JHSTRT`` (and legacy ``REST`` header) to the integrated dynamics step.

    Intra overlap rescue reloads the PSF and clears CHARMM's in-memory step
    counter; ``WRITe restart`` then leaves ``JHSTRT=0``.  Patching restores
    correct global step accounting on the next ``READYN`` handoff.
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
        header = lines[0].split()
        if len(header) >= 3:
            header[2] = str(step)
            lines[0] = " ".join(header)
            patched = True

    for i, raw in enumerate(lines):
        tag = raw.strip().split()[0] if raw.strip() else ""
        if not (tag.startswith("!NATOM") or tag.startswith("NATOM")):
            continue
        if i + 1 >= len(lines):
            break
        fields = lines[i + 1].split()
        if len(fields) < 6:
            break
        fields[5] = str(step)
        lines[i + 1] = " ".join(fields)
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
    expected_frames = expected_dcd_frame_count(nstep=expected_nstep, nsavc=nsavc)
    min_frames = max(2, int(expected_frames * min_frame_fraction))
    min_steps = max(1, int(expected_nstep * min_step_fraction))

    restart_step: int | None = None
    if restart_path is not None:
        restart_step = read_restart_last_step(restart_path)

    header_frames = count_dcd_frames(dcd_path) if dcd_path is not None else 0
    readable_frames = (
        count_readable_dcd_frames(dcd_path) if dcd_path is not None else 0
    )
    n_frames = min(header_frames, readable_frames) if header_frames else readable_frames

    problems: list[str] = []
    if restart_step is not None and restart_step < min_steps:
        problems.append(
            f"restart step {restart_step} < {min_steps} "
            f"(expected ~{expected_nstep}; likely echeck or CHARMM abort)"
        )
    if dcd_path is not None and header_frames > 0 and readable_frames < header_frames:
        problems.append(
            f"DCD {dcd_path.name} header claims {header_frames} frame(s) but only "
            f"{readable_frames} are readable (truncated or corrupt file)"
        )
    if dcd_path is not None and n_frames < min_frames:
        problems.append(
            f"DCD {dcd_path.name} has {n_frames} readable frame(s), "
            f"expected >= {min_frames} ({expected_frames} at nsavc={nsavc})"
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
