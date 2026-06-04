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
                return int(fields[5])
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

    n_frames = count_dcd_frames(dcd_path) if dcd_path is not None else 0

    problems: list[str] = []
    if restart_step is not None and restart_step < min_steps:
        problems.append(
            f"restart step {restart_step} < {min_steps} "
            f"(expected ~{expected_nstep}; likely echeck or CHARMM abort)"
        )
    if dcd_path is not None and n_frames < min_frames:
        problems.append(
            f"DCD {dcd_path.name} has {n_frames} frame(s), "
            f"expected >= {min_frames} ({expected_frames} at nsavc={nsavc})"
        )

    if not problems:
        print(
            f"{stage.upper()} complete: "
            f"restart_step={restart_step if restart_step is not None else '?'}, "
            f"dcd_frames={n_frames} (expected ~{expected_frames})",
            flush=True,
        )
        return

    hint = (
        "Check the log for 'ENERGY CHANGE TOLERANCE' / echeck. "
        "For heating tests try --no-echeck or a larger --echeck; "
        "for dissociation inspect VMD (ML USER-only dynamics has no SHAKE)."
    )
    raise RuntimeError(
        f"{stage.upper()} dynamics incomplete: " + "; ".join(problems) + f". {hint}"
    )
