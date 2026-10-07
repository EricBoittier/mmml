"""Deprecated import path — use :mod:`karml.models.efield.training` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "karml.models.EF.training is deprecated; use karml.models.efield.training",
    DeprecationWarning,
    stacklevel=2,
)

from karml.models.efield.training import *  # noqa: F403
