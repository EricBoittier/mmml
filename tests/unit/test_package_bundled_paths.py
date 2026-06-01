"""Regression tests for files shipped inside the installed mmml wheel."""

from __future__ import annotations

from pathlib import Path

import mmml
from mmml.data.atomic_references import _DATA_PATH
from mmml.paths import crystal_image_str_source, default_meoh_template_pdb


def test_atomic_reference_json_is_bundled() -> None:
    assert _DATA_PATH.is_file(), f"missing bundled reference table: {_DATA_PATH}"


def test_default_meoh_template_pdb_is_bundled() -> None:
    path = default_meoh_template_pdb()
    assert path.is_file(), f"missing bundled template PDB: {path}"


def test_crystal_image_str_is_bundled() -> None:
    path = crystal_image_str_source()
    assert path.is_file(), f"missing bundled CHARMM helper: {path}"


def test_pycharmm_interface_compat_alias() -> None:
    from mmml.pycharmmInterface import mmml_calculator
    from mmml.interfaces.pycharmmInterface import mmml_calculator as canonical

    assert mmml_calculator is canonical


def test_generate_sample_importable_from_installed_package() -> None:
    from mmml.generate.sample import sample_diverse_xyz

    assert callable(getattr(sample_diverse_xyz, "main", None))


def test_mmml_package_root_is_directory() -> None:
    root = Path(mmml.__file__).resolve().parent
    assert root.is_dir()
