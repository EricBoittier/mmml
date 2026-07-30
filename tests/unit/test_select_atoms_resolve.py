"""SelectAtoms resolution across PyCHARMM package layouts."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace


def test_select_atoms_cls_falls_back_to_submodule(monkeypatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot import setup as setup_mod

    class _FakeSelectAtoms:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    pkg = ModuleType("pycharmm")
    sub = ModuleType("pycharmm.select_atoms")
    sub.SelectAtoms = _FakeSelectAtoms  # type: ignore[attr-defined]
    pkg.select_atoms = sub  # type: ignore[attr-defined]
    # Intentionally no pkg.SelectAtoms — matches cluster AttributeError hint.

    monkeypatch.setitem(__import__("sys").modules, "pycharmm", pkg)
    monkeypatch.setitem(__import__("sys").modules, "pycharmm.select_atoms", sub)
    monkeypatch.setattr(setup_mod, "_import_pycharmm", lambda: pkg)

    cls = setup_mod._select_atoms_cls()
    assert cls is _FakeSelectAtoms
    assert isinstance(cls(), _FakeSelectAtoms)


def test_select_atoms_cls_prefers_package_attribute(monkeypatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot import setup as setup_mod

    class _PkgSelectAtoms:
        pass

    class _SubSelectAtoms:
        pass

    pkg = SimpleNamespace(SelectAtoms=_PkgSelectAtoms)
    monkeypatch.setattr(setup_mod, "_import_pycharmm", lambda: pkg)

    assert setup_mod._select_atoms_cls() is _PkgSelectAtoms
