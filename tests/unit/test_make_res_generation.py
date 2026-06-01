from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_IMPORT_PYCHARMM = "mmml.interfaces.pycharmmInterface.import_pycharmm"
_SETUP_RES = "mmml.interfaces.pycharmmInterface.setupRes"
_PYCHARMM_INTERFACE = "mmml.interfaces.pycharmmInterface"


class _FakeAtoms:
    pass


def _make_fake_utils(calls: list[str]) -> types.ModuleType:
    fake = types.ModuleType("mmml.interfaces.pycharmmInterface.utils")
    fake.get_Z_from_psf = lambda: [1]
    fake.set_up_directories = lambda: calls.append("set_up_directories")
    return fake


def _make_fake_import_pycharmm(calls: list[str]) -> types.ModuleType:
    """Stub import_pycharmm with every symbol setupRes / make_res import at load time."""
    fake = types.ModuleType(_IMPORT_PYCHARMM)
    fake.CGENFF_RTF = "fake.rtf"
    fake.CGENFF_PRM = "fake.prm"
    fake.CHARMM_HOME = "/tmp/charmm"
    fake.CHARMM_LIB_DIR = "/tmp/charmm/lib"
    fake.pycharmm_quiet = lambda: calls.append("quiet")
    fake.pycharmm_loud = lambda: calls.append("loud")
    fake.reset_block = lambda: calls.append("reset_block")
    fake.reset_block_no_internal = lambda: calls.append("reset_block_no_internal")
    fake.safe_energy_show = lambda: calls.append("safe_energy_show")
    return fake


def _clear_pycharmm_interface_import_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (_SETUP_RES, _IMPORT_PYCHARMM):
        monkeypatch.delitem(sys.modules, name, raising=False)
    parent = sys.modules.get(_PYCHARMM_INTERFACE)
    if parent is not None:
        parent.__dict__.pop("setupRes", None)
        parent.__dict__.pop("import_pycharmm", None)


def _install_fake_setupres_dependencies(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    fake_import_pycharmm = _make_fake_import_pycharmm(calls)
    monkeypatch.setitem(sys.modules, _IMPORT_PYCHARMM, fake_import_pycharmm)

    monkeypatch.setitem(
        sys.modules,
        "mmml.interfaces.pycharmmInterface.utils",
        _make_fake_utils(calls),
    )

    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda script: calls.append(f"script:{script.strip().splitlines()[0]}")

    class FakeNonBondedScript:
        def __init__(self, **kwargs):
            calls.append(f"nbonds_init:{kwargs['nbxmod']}")

        def run(self):
            calls.append("nbonds_run")

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.NonBondedScript = FakeNonBondedScript
    fake_pycharmm.lingo = fake_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)

    fake_generate = types.ModuleType("pycharmm.generate")
    fake_generate.new_segment = lambda **kwargs: calls.append(f"new_segment:{kwargs['seg_name']}")
    monkeypatch.setitem(sys.modules, "pycharmm.generate", fake_generate)

    fake_ic = types.ModuleType("pycharmm.ic")
    fake_ic.prm_fill = lambda **kwargs: calls.append("prm_fill")
    fake_ic.build = lambda: calls.append("ic_build")
    monkeypatch.setitem(sys.modules, "pycharmm.ic", fake_ic)

    fake_coor = types.ModuleType("pycharmm.coor")
    monkeypatch.setitem(sys.modules, "pycharmm.coor", fake_coor)

    fake_minimize = types.ModuleType("pycharmm.minimize")
    fake_minimize.run_abnr = lambda **kwargs: calls.append("run_abnr")
    monkeypatch.setitem(sys.modules, "pycharmm.minimize", fake_minimize)

    fake_read = types.ModuleType("pycharmm.read")
    fake_read.rtf = lambda path: calls.append(f"read_rtf:{path}")
    fake_read.prm = lambda path: calls.append(f"read_prm:{path}")
    fake_read.sequence_string = lambda resid: calls.append(f"sequence:{resid}")
    monkeypatch.setitem(sys.modules, "pycharmm.read", fake_read)

    fake_write = types.ModuleType("pycharmm.write")
    fake_write.coor_pdb = lambda path: calls.append(f"coor_pdb:{path}")
    fake_write.psf_card = lambda path: calls.append(f"psf_card:{path}")
    monkeypatch.setitem(sys.modules, "pycharmm.write", fake_write)

    fake_settings = types.ModuleType("pycharmm.settings")
    fake_settings.set_bomb_level = lambda level: 0
    fake_settings.set_warn_level = lambda level: 0
    monkeypatch.setitem(sys.modules, "pycharmm.settings", fake_settings)


def _import_setupres_with_fakes(monkeypatch: pytest.MonkeyPatch, calls: list[str]):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    _clear_pycharmm_interface_import_cache(monkeypatch)
    _install_fake_setupres_dependencies(monkeypatch, calls)
    return importlib.import_module(_SETUP_RES)


def test_mini_sets_nbonds_before_abnr_and_uses_safe_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    setup_res = _import_setupres_with_fakes(monkeypatch, calls)

    setup_res.mini(nbxmod=1, skip_energy_show=False)

    assert calls[-4:] == ["nbonds_init:1", "nbonds_run", "run_abnr", "safe_energy_show"]


def test_mini_can_skip_energy_show(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    setup_res = _import_setupres_with_fakes(monkeypatch, calls)

    setup_res.mini(nbxmod=5, skip_energy_show=True)

    assert calls[-3:] == ["nbonds_init:5", "nbonds_run", "run_abnr"]
    assert "safe_energy_show" not in calls


def test_setupres_main_retries_from_fresh_residue_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    setup_res = _import_setupres_with_fakes(monkeypatch, calls)
    atoms = _FakeAtoms()
    attempts = {"count": 0}

    def fake_generate_residue(resid: str) -> None:
        calls.append(f"generate_residue:{resid}")

    def fake_generate_coordinates(*, skip_energy_show: bool) -> _FakeAtoms:
        calls.append(f"generate_coordinates:{skip_energy_show}")
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("unresolved geometry")
        return atoms

    monkeypatch.setattr(setup_res, "generate_residue", fake_generate_residue)
    monkeypatch.setattr(setup_res, "generate_coordinates", fake_generate_coordinates)
    monkeypatch.setattr(setup_res, "write_psf", lambda resid: calls.append(f"write_psf:{resid}"))
    monkeypatch.setattr(setup_res.shutil, "copy", lambda src, dst: calls.append(f"copy:{src}->{dst}"))
    monkeypatch.setattr(setup_res.ase.io, "write", lambda path, obj: calls.append(f"ase_write:{path}"))

    result = setup_res.main("tip3", skip_energy_show=True, max_attempts=2)

    assert result is atoms
    assert calls[:4] == [
        "generate_residue:TIP3",
        "generate_coordinates:True",
        "generate_residue:TIP3",
        "generate_coordinates:True",
    ]
    assert "write_psf:TIP3" in calls


def test_make_res_main_loop_uses_single_checked_setup_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    atoms = _FakeAtoms()

    _clear_pycharmm_interface_import_cache(monkeypatch)

    fake_setup_res = types.ModuleType(_SETUP_RES)

    def fake_main(resid: str, *, skip_energy_show: bool, max_attempts: int) -> _FakeAtoms:
        calls.append(f"setupRes.main:{resid}:{skip_energy_show}:{max_attempts}")
        return atoms

    fake_setup_res.main = fake_main
    monkeypatch.setitem(sys.modules, _SETUP_RES, fake_setup_res)
    monkeypatch.setitem(sys.modules, _IMPORT_PYCHARMM, _make_fake_import_pycharmm(calls))
    monkeypatch.setitem(
        sys.modules,
        "mmml.interfaces.pycharmmInterface.utils",
        _make_fake_utils(calls),
    )

    import mmml.interfaces.pycharmmInterface as pycharmm_interface

    monkeypatch.setattr(pycharmm_interface, "setupRes", fake_setup_res, raising=False)

    import ase.io

    monkeypatch.setattr(ase.io, "write", lambda path, obj: calls.append(f"ase_write:{path}"))

    from mmml.cli.make import make_res

    result = make_res.main_loop(types.SimpleNamespace(res="TIP3", skip_energy_show=True))

    assert result is atoms
    assert calls.count("setupRes.main:TIP3:True:2") == 1
    assert not hasattr(fake_setup_res, "generate_coordinates")
    assert calls[-5:] == [
        "reset_block",
        "reset_block_no_internal",
        "reset_block",
        "ase_write:xyz/initial.xyz",
        "ase_write:xyz/tip3.xyz",
    ]
