"""Deprecated import path — use :mod:`mmml.models.efield.training` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "mmml.models.EF.training is deprecated; use mmml.models.efield.training",
    DeprecationWarning,
    stacklevel=2,
)

from mmml.models.efield.training import *  # noqa: F403
