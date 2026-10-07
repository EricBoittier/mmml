"""Deprecated import path — use :mod:`karml.models.efield.evaluate` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "karml.models.EF.evaluate is deprecated; use karml.models.efield.evaluate",
    DeprecationWarning,
    stacklevel=2,
)

from karml.models.efield.evaluate import *  # noqa: F403
