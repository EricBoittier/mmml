"""Tests for CHARMM path discovery."""

from __future__ import annotations

import os

from mmml.interfaces.pycharmmInterface import charmm_paths


def test_discover_repo_default_charmm_home(tmp_path):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    chm.mkdir(parents=True)
    (chm / "libcharmm.dylib").write_bytes(b"stub")

    home, lib = charmm_paths.resolve_charmm_paths(repo_root=repo, env={})

    assert home == str(chm)
    assert lib == str(chm)


def test_discover_lib_in_setup_charmm_lib_subdir(tmp_path):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    lib_dir = chm / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcharmm.so").write_bytes(b"stub")

    home, lib = charmm_paths.resolve_charmm_paths(repo_root=repo, env={})

    assert home == str(chm)
    assert lib == str(chm)


def test_explicit_env_overrides_repo_default(tmp_path):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    chm.mkdir(parents=True)
    (chm / "libcharmm.dylib").write_bytes(b"stub")
    tier_lib = tmp_path / "tier" / "lib"
    tier_lib.mkdir(parents=True)
    (tier_lib / "libcharmm.so").write_bytes(b"tier")

    home, lib = charmm_paths.resolve_charmm_paths(
        repo_root=repo,
        env={"CHARMM_LIB_DIR": str(tier_lib)},
    )

    assert lib == str(tier_lib)
    assert home == str(chm)


def test_stale_explicit_lib_dir_falls_back_to_repo_default(tmp_path):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    lib_dir = chm / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcharmm.so").write_bytes(b"stub")

    home, lib = charmm_paths.resolve_charmm_paths(
        repo_root=repo,
        env={"CHARMM_LIB_DIR": "/path/to/charmm"},
    )

    assert home == str(chm)
    assert lib == str(chm)


def test_normalize_charmm_lib_dir_so_file(tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    assert charmm_paths.normalize_charmm_lib_dir(str(lib)) == str(tmp_path)


def test_resolve_lib_dir_accepts_lib_subdir(tmp_path):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    lib_dir = chm / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcharmm.so").write_bytes(b"stub")

    home, lib = charmm_paths.resolve_charmm_paths(
        repo_root=repo,
        env={"CHARMM_LIB_DIR": str(lib_dir)},
    )

    assert lib == str(lib_dir)
    assert home == str(chm)


def test_charmmsetup_legacy_export_format(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    custom = tmp_path / "custom"
    custom.mkdir()
    (repo / "CHARMMSETUP").write_text(
        f"export CHARMM_HOME={custom}\nexport CHARMM_LIB_DIR={custom}\n",
        encoding="utf-8",
    )

    home, lib = charmm_paths.resolve_charmm_paths(repo_root=repo, env={})

    assert home == str(custom)
    assert lib == str(custom)


def test_charmmsetup_legacy_plain_format(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    custom = tmp_path / "custom"
    custom.mkdir()
    (repo / "CHARMMSETUP").write_text(
        f"CHARMM_HOME={custom}\nCHARMM_LIB_DIR={custom}\n",
        encoding="utf-8",
    )

    home, lib = charmm_paths.resolve_charmm_paths(repo_root=repo, env={})

    assert home == str(custom)
    assert lib == str(custom)


def test_env_beats_charmmsetup(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    setup_dir = tmp_path / "from-setup"
    env_dir = tmp_path / "from-env"
    setup_dir.mkdir()
    env_dir.mkdir()
    (repo / "CHARMMSETUP").write_text(
        f"CHARMM_HOME={setup_dir}\nCHARMM_LIB_DIR={setup_dir}\n",
        encoding="utf-8",
    )

    home, lib = charmm_paths.resolve_charmm_paths(
        repo_root=repo,
        env={"CHARMM_HOME": str(env_dir), "CHARMM_LIB_DIR": str(env_dir)},
    )

    assert home == str(env_dir)
    assert lib == str(env_dir)


def test_bootstrap_charmm_env_sets_os_environ(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    chm.mkdir(parents=True)
    (chm / "libcharmm.so").write_bytes(b"stub")
    monkeypatch.delenv("CHARMM_HOME", raising=False)
    monkeypatch.delenv("CHARMM_LIB_DIR", raising=False)

    home, lib = charmm_paths.bootstrap_charmm_env(repo_root=repo)

    assert home == str(chm)
    assert lib == str(chm)
    assert os.environ["CHARMM_HOME"] == str(chm)
    assert os.environ["CHARMM_LIB_DIR"] == str(chm)


def test_charmm_lib_available_without_explicit_env(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

    repo = tmp_path / "repo"
    chm = repo / "setup" / "charmm"
    chm.mkdir(parents=True)
    (chm / "libcharmm.so").write_bytes(b"stub")
    monkeypatch.delenv("CHARMM_HOME", raising=False)
    monkeypatch.delenv("CHARMM_LIB_DIR", raising=False)
    monkeypatch.setattr(charmm_paths, "mmml_repo_root", lambda start=None: repo)

    assert charmm_mpi.charmm_lib_available() is True
    assert os.environ["CHARMM_HOME"] == str(chm)
    assert os.environ["CHARMM_LIB_DIR"] == str(chm)


def test_fortran_path_needs_alias_detects_uppercase(tmp_path):
    upper = tmp_path / "DCM60_L32" / "pretreat" / "mini_box_equil.res"
    lower = tmp_path / "dcm60_l32" / "pretreat" / "mini_box_equil.res"
    assert charmm_paths.fortran_path_needs_alias(upper)
    assert not charmm_paths.fortran_path_needs_alias(lower)


def test_charmm_io_alias_read_symlink(tmp_path):
    upper_dir = tmp_path / "boxes" / "dcm60_L32" / "pretreat"
    upper_dir.mkdir(parents=True)
    original = upper_dir / "mini_box_equil.res"
    original.write_text("restart\n", encoding="ascii")

    alias = charmm_paths.charmm_io_alias(original, for_write=False)
    assert alias is not None
    assert alias.alias.is_symlink()
    assert alias.alias.resolve() == original.resolve()
    assert alias.fortran_path == alias.fortran_path.lower()
    assert alias.alias.read_text(encoding="ascii") == "restart\n"


def test_charmm_io_alias_write_copy_back(tmp_path):
    upper_dir = tmp_path / "boxes" / "dcm60_L32" / "pretreat"
    target = upper_dir / "mini_box_equil.res"
    staging = tmp_path / "staging"

    alias = charmm_paths.charmm_io_alias(target, for_write=True, staging_root=staging)
    assert alias is not None
    alias.alias.write_text("written via alias\n", encoding="ascii")
    assert not target.is_file()

    alias.finalize()
    assert target.is_file()
    assert target.read_text(encoding="ascii") == "written via alias\n"


def test_charmm_fortran_path_noop_for_lowercase(tmp_path):
    path = tmp_path / "pretreat" / "mini_box_equil.res"
    path.parent.mkdir()
    path.write_text("x", encoding="ascii")

    fortran_path, alias = charmm_paths.charmm_fortran_path(path, for_write=True)
    assert alias is None
    assert fortran_path == str(path.resolve())
