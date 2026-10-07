"""Deprecated import path — use :mod:`karml.models.efield` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "karml.models.EF is deprecated; use karml.models.efield instead.",
    DeprecationWarning,
    stacklevel=2,
)

from karml.models.efield.training import (  # noqa: E402
    EFieldPhysNet,
    MessagePassingModel,
)

__all__ = ["EFieldPhysNet", "MessagePassingModel"]
