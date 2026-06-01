#!/usr/bin/env python3
"""Deprecated shim: use ``mmml md-system --backend jaxmd`` or ``python -m mmml.cli.run.md_pbc_suite.jaxmd``."""

from __future__ import annotations

import warnings

from mmml.cli.run.md_pbc_suite.jaxmd import main

warnings.warn(
    "scripts/md_10mer_mmml_pbc_suite_jaxmd.py is deprecated; "
    "use mmml md-system --backend jaxmd or python -m mmml.cli.run.md_pbc_suite.jaxmd",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(main())
