#!/usr/bin/env python3
"""Deprecated shim: use ``mmml md-system`` or ``python -m mmml.cli.run.md_pbc_suite.ase``."""

from __future__ import annotations

import warnings

from mmml.cli.run.md_pbc_suite.ase import main

warnings.warn(
    "scripts/md_10mer_mmml_pbc_suite.py is deprecated; "
    "use mmml md-system or python -m mmml.cli.run.md_pbc_suite.ase",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(main())
