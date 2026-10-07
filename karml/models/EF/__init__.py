"""Deprecated import path — use :mod:`mmml.models.efield` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "mmml.models.EF is deprecated; use mmml.models.efield instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mmml.models.efield.training import (  # noqa: E402
    EFieldPhysNet,
    MessagePassingModel,
)

__all__ = ["EFieldPhysNet", "MessagePassingModel"]
