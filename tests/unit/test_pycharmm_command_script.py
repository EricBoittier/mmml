"""Regression tests for generated CHARMM command syntax."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = Path("setup/charmm/tool/pycharmm/pycharmm/script.py")


def _load_command_script(monkeypatch) -> type:
    """Load the serializer with minimal PyCHARMM stubs; no libcharmm needed."""
    package = ModuleType("pycharmm")
    package.__path__ = []  # type: ignore[attr-defined]
    lingo = ModuleType("pycharmm.lingo")
    atoms = ModuleType("pycharmm.select_atoms")
    package.lingo = lingo  # type: ignore[attr-defined]
    package.select_atoms = atoms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycharmm", package)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", lingo)
    monkeypatch.setitem(sys.modules, "pycharmm.select_atoms", atoms)

    spec = importlib.util.spec_from_file_location("pycharmm.command_script_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CommandScript


def test_command_script_uppercases_option_keywords(monkeypatch) -> None:
    CommandScript = _load_command_script(monkeypatch)

    script = CommandScript("dynamics", firstt=300.0, iasvel=1, start=True).create_script_string()

    assert script.startswith("DYNAMICS ")
    assert "FIRSTT 300.0" in script
    assert "IASVEL 1" in script
    assert "START" in script
    assert "firstt" not in script
    assert "iasvel" not in script
