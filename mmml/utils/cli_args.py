"""Helpers for argparse patterns that must coexist with Jupyter/IPython ``sys.argv``."""

from __future__ import annotations

import sys
from typing import Sequence


def exit_if_unknown_long_options(
    unknown: Sequence[str],
    *,
    prog: str | None = None,
    exit_code: int = 2,
) -> None:
    """Fail after ``parse_known_args`` if any leftover token looks like a user flag.

    Long options (``--something``) are treated as mistakes: they are almost never
    injected by kernels, unlike short flags such as ``-m`` or ``-f``.

    The standalone ``--`` end-of-options marker is ignored.
    """
    bad = [a for a in unknown if a.startswith("--") and a != "--"]
    if not bad:
        return
    prefix = f"{prog}: " if prog else ""
    print(
        f"{prefix}error: unrecognized arguments: {' '.join(bad)}",
        file=sys.stderr,
    )
    print(
        "Hint: check spelling; only flags listed in --help are accepted. "
        "In notebooks, avoid stray ``--`` flags on ``sys.argv`` or pass options via "
        "``get_args(...)`` keywords.",
        file=sys.stderr,
    )
    raise SystemExit(exit_code)
