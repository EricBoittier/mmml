#!/usr/bin/env python3
"""Summarize liquid-density dynamics campaign progress and DYNA health."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def main() -> int:
    diag = _SCRIPTS / "collect_diagnostics.py"
    argv = ["matrix", *sys.argv[1:]]
    if not any(a.startswith("--config") for a in sys.argv):
        argv = ["matrix", "--config", str(_SCRIPTS.parent / "config.yaml"), *sys.argv[1:]]
    return subprocess.call([sys.executable, str(diag), *argv])


if __name__ == "__main__":
    raise SystemExit(main())
