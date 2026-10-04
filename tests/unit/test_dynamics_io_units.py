"""c52a1: DCD/restart units opened by dynamics_set_iun* must reach the DYNA line."""

from __future__ import annotations

import sys
import types

from mmml.interfaces.pycharmmInterface.mlpot import dynamics as mdyn


def _fake_dyn(monkeypatch, opened):
    mod = types.ModuleType("pycharmm.dynamics")
    mod.set_iuncrd = lambda p: opened.append(("iuncrd", p)) or True
    mod.set_iunwri = lambda p: opened.append(("iunwri", p)) or True
    mod.set_iunrea = lambda p: True
    pkg = sys.modules.get("pycharmm") or types.ModuleType("pycharmm")
    monkeypatch.setitem(sys.modules, "pycharmm", pkg)
    monkeypatch.setitem(sys.modules, "pycharmm.dynamics", mod)
    monkeypatch.setattr(pkg, "dynamics", mod, raising=False)


def test_opened_units_are_passed_on_dyna_line(monkeypatch):
    opened: list = []
    _fake_dyn(monkeypatch, opened)
    units = {"iuncrd": 91, "iunwri": 90}
    monkeypatch.setattr(mdyn, "_charmm_reawri_unit", lambda key: units[key])
    kw = {"iuncrd": "/tmp/x.dcd", "iunwri": "/tmp/x.res", "nsavc": 1}
    mdyn._apply_dynamics_io_setters(kw)
    assert opened == [("iunwri", "/tmp/x.res"), ("iuncrd", "/tmp/x.dcd")]
    assert kw["iuncrd"] == 91 and kw["iunwri"] == 90
    assert mdyn._dynamics_writes_dcd(kw)


def test_unknown_unit_drops_keyword(monkeypatch):
    _fake_dyn(monkeypatch, [])
    monkeypatch.setattr(mdyn, "_charmm_reawri_unit", lambda key: None)
    kw = {"iuncrd": "/tmp/x.dcd"}
    mdyn._apply_dynamics_io_setters(kw)
    assert "iuncrd" not in kw


def test_api_seeds_go_on_script_line_for_c52(monkeypatch):
    monkeypatch.setattr(mdyn, "_charmm_dyna_script_parses_iseed", lambda: True)
    monkeypatch.setattr(mdyn, "_pending_api_rngseeds", [11, 22, 33, 44])
    kw: dict = {}
    mdyn._put_api_rngseeds_on_script_line(kw, use_c_api=False)
    assert kw["iseed"] == "11 22 33 44"
    # consumed: the next DYNA without a fresh set_rngseeds gets nothing (as c49)
    kw2: dict = {}
    mdyn._put_api_rngseeds_on_script_line(kw2, use_c_api=False)
    assert "iseed" not in kw2


def test_api_seeds_not_on_line_for_c_api_or_c49(monkeypatch):
    monkeypatch.setattr(mdyn, "_pending_api_rngseeds", [1, 2])
    monkeypatch.setattr(mdyn, "_charmm_dyna_script_parses_iseed", lambda: True)
    kw: dict = {}
    mdyn._put_api_rngseeds_on_script_line(kw, use_c_api=True)
    assert kw == {}
    monkeypatch.setattr(mdyn, "_pending_api_rngseeds", [1, 2])
    monkeypatch.setattr(mdyn, "_charmm_dyna_script_parses_iseed", lambda: False)
    mdyn._put_api_rngseeds_on_script_line(kw, use_c_api=False)
    assert kw == {}
