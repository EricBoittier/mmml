#!/usr/bin/env python3
"""Deprecated shim: use ``python -m karml.cli.run.md_pbc_suite.check_fd``."""

from __future__ import annotations

import warnings

from karml.cli.run.md_pbc_suite.check_fd import main

warnings.warn(
    "scripts/check_fd_forces_karml_pbc.py is deprecated; "
    "use python -m karml.cli.run.md_pbc_suite.check_fd",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(main())
