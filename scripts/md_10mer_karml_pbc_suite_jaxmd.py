#!/usr/bin/env python3
"""Deprecated shim: use ``karml md-system --backend jaxmd`` or ``python -m karml.cli.run.md_pbc_suite.jaxmd``."""

from __future__ import annotations

import warnings

from karml.cli.run.md_pbc_suite.jaxmd import main

warnings.warn(
    "scripts/md_10mer_karml_pbc_suite_jaxmd.py is deprecated; "
    "use karml md-system --backend jaxmd or python -m karml.cli.run.md_pbc_suite.jaxmd",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(main())
