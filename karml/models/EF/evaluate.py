"""Deprecated import path — use :mod:`mmml.models.efield.evaluate` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "mmml.models.EF.evaluate is deprecated; use mmml.models.efield.evaluate",
    DeprecationWarning,
    stacklevel=2,
)

from mmml.models.efield.evaluate import *  # noqa: F403
