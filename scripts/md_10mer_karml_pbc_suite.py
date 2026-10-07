#!/usr/bin/env python3
"""Deprecated shim: use ``karml md-system`` or ``python -m karml.cli.run.md_pbc_suite.ase``."""

from __future__ import annotations

import warnings

from karml.cli.run.md_pbc_suite.ase import main

warnings.warn(
    "scripts/md_10mer_karml_pbc_suite.py is deprecated; "
    "use karml md-system or python -m karml.cli.run.md_pbc_suite.ase",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(main())
