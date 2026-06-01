"""Regression tests for files shipped inside the installed mmml wheel."""

from __future__ import annotations

from pathlib import Path

import importlib.util

from mmml.paths import (
    _package_dir,
    bundled_file,
    crystal_image_str_source,
    default_meoh_template_pdb,
)


def test_atomic_reference_json_is_bundled() -> None:
    candidates = (
        bundled_file("data", "qcml", "atomic_reference_energies.json"),
        bundled_file("data", "atomic_reference_energies.json"),
    )
    assert any(p.is_file() for p in candidates), (
        "missing bundled atomic reference table: " + ", ".join(str(p) for p in candidates)
    )


def test_default_meoh_template_pdb_is_bundled() -> None:
    path = default_meoh_template_pdb()
    assert path.is_file(), f"missing bundled template PDB: {path}"


def test_crystal_image_str_is_bundled() -> None:
    path = crystal_image_str_source()
    assert path.is_file(), f"missing bundled CHARMM helper: {path}"


def test_pycharmm_interface_compat_alias() -> None:
    import mmml.pycharmmInterface  # noqa: F401 — registers lazy submodule proxies

    from mmml.pycharmmInterface import cutoffs as alias_mod
    from mmml.interfaces.pycharmmInterface import cutoffs as canonical_mod

    assert alias_mod is canonical_mod


def test_generate_sample_module_is_packaged() -> None:
    spec = importlib.util.find_spec("mmml.generate.sample.sample_diverse_xyz")
    assert spec is not None and spec.origin
    assert Path(spec.origin).is_file()


def test_mmml_package_root_is_directory() -> None:
    root = _package_dir()
    assert root.is_dir()
    assert (root / "__init__.py").is_file()
