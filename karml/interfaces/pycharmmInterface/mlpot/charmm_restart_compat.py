"""Make CHARMM dynamics restarts from another CHARMM version safe to READYN.

The restart header line after ``!NATOM,NPRIV,...`` ends with
``numnod nrand seed_1 .. seed_nrand``. c49b1 writes 8 RNG seeds, c52a1 has
``nrand = 4``. c52a1 ``READYN`` (dynio.F90) then does
``rngseeds(1:nrandlr) = allseeds(1:nrandlr)`` with ``nrandlr = 8`` into arrays
of length 4: a heap overflow that surfaces later as
``free(): invalid next size`` / SIGABRT (rc 134) or SIGSEGV at exit.

:func:`restart_with_seed_count` returns a copy with the seed list cut to the
library's ``nrand`` when the file holds more seeds than that; otherwise the
original path. Only the RNG seeds change; positions, velocities, box and
thermostat/barostat state are untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

_SEED_LINE_TAG = "!NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED"
# '(7I12,D22.15,I12,2I22,<n>I22)': fixed part = 7*12 + 22 + 12 characters.
_FIXED = 7 * 12 + 22 + 12
_W = 22


def charmm_lib_nrand() -> int | None:
    """``rndnum::nrand`` of the loaded CHARMM library, or None."""
    try:
        import pycharmm.dynamics as dyn

        n = int(dyn.get_nrand())
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return None
    return n if n > 0 else None


def _rewrite_seed_line(line: str, nrand: int) -> str | None:
    """Seed line cut to ``nrand`` seeds, or None if it needs no (or no safe) change."""
    body = line.rstrip("\n")
    head, tail = body[:_FIXED], body[_FIXED:].split()
    if len(head.split()) != 9 or len(tail) < 2:
        return None
    try:
        numnod, nseeds = int(tail[0]), int(tail[1])
        seeds = [int(s) for s in tail[2:]]
    except ValueError:
        return None
    if numnod != 1 or nseeds <= nrand or len(seeds) < nseeds:
        return None
    out = head + f"{numnod:{_W}d}{nrand:{_W}d}" + "".join(f"{s:{_W}d}" for s in seeds[:nrand])
    return out + "\n"


def restart_with_seed_count(
    path: str | Path,
    nrand: int | None = None,
    *,
    verbose: bool = True,
) -> Path:
    """Return ``path``, or a temp copy whose RNG seed list fits ``nrand``."""
    p = Path(path)
    if nrand is None:
        nrand = charmm_lib_nrand()
    if not nrand:
        return p
    try:
        lines = p.read_text().splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return p
    for i, line in enumerate(lines[:-1]):
        if _SEED_LINE_TAG not in line:
            continue
        new = _rewrite_seed_line(lines[i + 1], int(nrand))
        if new is None:
            return p
        lines[i + 1] = new
        out_dir = Path(tempfile.mkdtemp(prefix="karml-res-seeds-"))
        out = out_dir / p.name
        out.write_text("".join(lines))
        if verbose:
            print(
                f"CHARMM restart {p}: RNG seed list cut to this library's nrand={nrand} "
                f"(written by a CHARMM with more seeds; READYN would overflow) -> {out}",
                flush=True,
            )
        return out
    return p
