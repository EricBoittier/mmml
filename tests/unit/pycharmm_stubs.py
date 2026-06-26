"""CI-safe PyCHARMM stubs (no libcharmm.so required)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any
from unittest import mock


@contextmanager
def fake_pycharmm_modules(
    *,
    coor: Any | None = None,
    energy: Any | None = None,
    extra: dict[str, Any] | None = None,
):
    """Install minimal ``pycharmm`` submodules in ``sys.modules`` for unit tests."""
    fake_coor = coor if coor is not None else mock.MagicMock()
    fake_energy = energy if energy is not None else mock.MagicMock()
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.coor = fake_coor
    fake_pycharmm.energy = fake_energy
    modules = {
        "pycharmm": fake_pycharmm,
        "pycharmm.coor": fake_coor,
        "pycharmm.energy": fake_energy,
        "mmml.interfaces.pycharmmInterface.import_pycharmm": mock.MagicMock(),
    }
    if extra:
        modules.update(extra)
    with mock.patch.dict(sys.modules, modules, clear=False):
        yield fake_pycharmm, fake_coor, fake_energy
