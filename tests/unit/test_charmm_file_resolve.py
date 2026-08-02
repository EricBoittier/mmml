"""Resolve CharmmFile when top-level pycharmm export is missing."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def test_resolve_charmm_file_cls_uses_top_level_export():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _resolve_charmm_file_cls

    sentinel = object()
    mod = SimpleNamespace(CharmmFile=sentinel)
    assert _resolve_charmm_file_cls(mod) is sentinel


def test_resolve_charmm_file_cls_falls_back_to_submodule():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _resolve_charmm_file_cls

    sentinel = object()
    stub_sub = ModuleType("pycharmm.charmm_file")
    stub_sub.CharmmFile = sentinel  # type: ignore[attr-defined]
    bare = ModuleType("pycharmm")
    # No CharmmFile attribute — mirrors namespace-package shadowing.
    with patch.dict("sys.modules", {"pycharmm.charmm_file": stub_sub}):
        assert _resolve_charmm_file_cls(bare) is sentinel


def test_resolve_charmm_file_cls_raises_when_unavailable():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _resolve_charmm_file_cls

    bare = ModuleType("pycharmm")
    stub_sub = ModuleType("pycharmm.charmm_file")  # no CharmmFile attr
    with patch.dict("sys.modules", {"pycharmm.charmm_file": stub_sub}):
        with pytest.raises(AttributeError, match="CharmmFile"):
            _resolve_charmm_file_cls(bare)


def test_open_minimize_dcd_uses_resolved_charmm_file(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot import dynamics as dyn

    opened: dict[str, object] = {}

    class FakeCharmmFile:
        def __init__(self, **kwargs):
            opened.update(kwargs)

    pycharmm = SimpleNamespace()  # no CharmmFile
    with patch.object(
        dyn,
        "_import_pycharmm_modules",
        return_value=(pycharmm, None, None, None, None, None),
    ), patch.object(dyn, "_resolve_charmm_file_cls", return_value=FakeCharmmFile):
        out = dyn.open_minimize_dcd(tmp_path / "mini.dcd", unit=51)
    assert isinstance(out, FakeCharmmFile)
    assert opened["file_unit"] == 51
    assert opened["formatted"] is False
    assert str(opened["file_name"]).endswith("mini.dcd")
