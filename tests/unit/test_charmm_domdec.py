"""Tests for once-only domdec disable on DOMDEC CHARMM builds."""

from __future__ import annotations

from contextlib import nullcontext
from unittest import mock

from pathlib import Path


def _prepare_domdec_module(monkeypatch):
    """Use the real DOMDEC helper functions with PyCHARMM side effects mocked."""
    from karml.interfaces.pycharmmInterface import import_pycharmm as mod

    mod._domdec_vacuum_disabled = False
    mod._domdec_disabled_early = False
    monkeypatch.delenv("KARML_FORCE_DOMDEC_OFF", raising=False)
    monkeypatch.delenv("KARML_NO_CHARMM_DOMDEC_OFF", raising=False)
    monkeypatch.setattr(
        "karml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(mod, "pycharmm", mock.Mock())
    mod.pycharmm.lingo.charmm_script = mock.Mock()
    return mod


def test_disable_charmm_domdec_skips_by_default(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    assert mod.disable_charmm_domdec() is False
    mod.pycharmm.lingo.charmm_script.assert_not_called()


def test_disable_charmm_domdec_force_hatch_runs_once(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    monkeypatch.setenv("KARML_FORCE_DOMDEC_OFF", "1")
    assert mod.disable_charmm_domdec(when="mlpot_energy") is True
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    assert mod.pycharmm.lingo.charmm_script.call_count == 1
    assert mod.pycharmm.lingo.charmm_script.call_args[0][0] == "domdec off"
    assert mod._domdec_vacuum_disabled is True
    assert mod._domdec_disabled_early is False


def test_disable_charmm_domdec_no_hatch_wins_over_force(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    monkeypatch.setenv("KARML_FORCE_DOMDEC_OFF", "1")
    monkeypatch.setenv("KARML_NO_CHARMM_DOMDEC_OFF", "1")
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    mod.pycharmm.lingo.charmm_script.assert_not_called()


def test_ensure_domdec_off_recovers_mpi_only_after_success(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    recover = mock.Mock()
    monkeypatch.setattr(
        "karml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
        recover,
    )

    assert mod.ensure_domdec_off_for_mlpot_energy(context="unit default") is False
    recover.assert_not_called()

    monkeypatch.setenv("KARML_FORCE_DOMDEC_OFF", "1")
    assert mod.ensure_domdec_off_for_mlpot_energy(context="unit forced") is True
    recover.assert_called_once_with(phase="after domdec off (unit forced)")


def test_disable_charmm_domdec_skipped_by_default():
    path = (
        Path(__file__).resolve().parents[2]
        / "karml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _should_run_domdec_off()" in source
    assert "KARML_FORCE_DOMDEC_OFF" in source
    block = source.split("def disable_charmm_domdec(")[1].split("\ndef ")[0]
    assert "if not _should_run_domdec_off():" in block


def test_init_vacuum_charmm_state_does_not_disable_domdec():
    path = (
        Path(__file__).resolve().parents[2]
        / "karml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _init_vacuum_charmm_state() -> None:" in source
    block = source.split("def _init_vacuum_charmm_state() -> None:")[1].split("\ndef ")[0]
    code_lines = [
        ln
        for ln in block.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert not any("disable_charmm_domdec(" in ln for ln in code_lines)


def test_setup_charmm_environment_defers_domdec_off():
    path = (
        Path(__file__).resolve().parents[2]
        / "karml/interfaces/pycharmmInterface/mlpot/pbc_env.py"
    )
    source = path.read_text(encoding="utf-8")
    block = source.split("def setup_charmm_environment(")[1].split("\ndef ")[0]
    assert "disable_charmm_domdec" not in block


def test_ensure_vendored_pycharmm_on_path_prefers_repo_root():
    import sys

    from karml.interfaces.pycharmmInterface.import_pycharmm import (
        _ensure_vendored_pycharmm_on_path,
        _vendored_pycharmm_sys_path_entries,
    )

    repo = Path(__file__).resolve().parents[2]
    entries = _vendored_pycharmm_sys_path_entries()
    assert entries, "expected at least setup/charmm/tool/pycharmm on path"
    # Repo-root sibling only when it has a real package __init__.py
    repo_pkg = repo / "pycharmm" / "__init__.py"
    if repo_pkg.is_file():
        assert entries[0] == str(repo)
    else:
        assert str(repo) not in entries
        assert entries[0].endswith(str(Path("setup/charmm/tool/pycharmm")))

    _ensure_vendored_pycharmm_on_path()
    assert sys.path[0] == entries[0]


def test_vendored_pycharmm_entries_skip_initless_namespace(tmp_path, monkeypatch):
    from karml.interfaces.pycharmmInterface import import_pycharmm as ip

    fake_repo = tmp_path / "karml"
    (fake_repo / "pycharmm").mkdir(parents=True)  # no __init__.py → namespace trap
    tool = fake_repo / "setup" / "charmm" / "tool" / "pycharmm" / "pycharmm"
    tool.mkdir(parents=True)
    (tool / "__init__.py").write_text("name = 'pycharmm'\n", encoding="utf-8")

    monkeypatch.setattr(ip, "_REPO_ROOT", fake_repo)
    entries = ip._vendored_pycharmm_sys_path_entries()
    assert str(fake_repo) not in entries
    assert entries == [str(fake_repo / "setup" / "charmm" / "tool" / "pycharmm")]


def _fake_lib(lib_dir, version):
    lib_dir.mkdir(parents=True, exist_ok=True)
    (lib_dir / "libcharmm.so").write_bytes(b"\0junk\0Developmental Version " + version + b"\0more\0")


def _fake_vendored_repo(tmp_path):
    repo = tmp_path / "karml"
    tool = repo / "setup" / "charmm" / "tool" / "pycharmm" / "pycharmm"
    tool.mkdir(parents=True)
    (tool / "__init__.py").write_text("", encoding="utf-8")
    src = repo / "setup" / "charmm" / "source" / "charmm"
    src.mkdir(parents=True)
    (src / "charmm_main.F90").write_text(
        "  !      Version 52 - Developmental Version (c52a1) - August 15, 2026\n", encoding="utf-8"
    )
    return repo


def test_charmm_version_detection(tmp_path):
    from karml.interfaces.pycharmmInterface import import_pycharmm as ip

    _fake_lib(tmp_path / "lib49", b"49b1")
    assert ip._charmm_lib_major(tmp_path / "lib49") == 49
    assert ip._charmm_lib_major(tmp_path / "missing") is None
    assert ip._vendored_pycharmm_major(_fake_vendored_repo(tmp_path)) == 52


def test_mismatched_library_prefers_charmm_home_pycharmm(tmp_path, monkeypatch):
    import sys

    from karml.interfaces.pycharmmInterface import import_pycharmm as ip

    repo = _fake_vendored_repo(tmp_path)
    home = tmp_path / "c49"
    (home / "tool" / "pycharmm" / "pycharmm").mkdir(parents=True)
    (home / "tool" / "pycharmm" / "pycharmm" / "__init__.py").write_text("", encoding="utf-8")
    _fake_lib(home / "lib", b"49b1")
    monkeypatch.setattr(ip, "_REPO_ROOT", repo)
    monkeypatch.setattr(ip, "CHARMM_HOME", str(home))
    monkeypatch.setattr(ip, "CHARMM_LIB_DIR", str(home / "lib"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    ip._ensure_vendored_pycharmm_on_path()
    assert sys.path[0] == str(home / "tool" / "pycharmm")
    assert str(repo / "setup" / "charmm" / "tool" / "pycharmm") not in sys.path

    _fake_lib(home / "lib", b"52a1")
    ip._ensure_vendored_pycharmm_on_path()
    assert sys.path[0] == str(repo / "setup" / "charmm" / "tool" / "pycharmm")
