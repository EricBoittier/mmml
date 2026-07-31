"""Regression tests for files shipped inside the installed mmml wheel."""

from __future__ import annotations

from pathlib import Path

import importlib.util

import pytest

from mmml.paths import (
    _package_dir,
    bundled_file,
    crystal_image_str_source,
    default_aco_template_pdb,
    default_benzene_crystal_cif,
    default_dcm_crystal_cif,
    default_dcm_molecule_xyz,
    default_meoh_template_pdb,
    default_tip3_template_pdb,
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


def test_default_aco_template_pdb_is_bundled() -> None:
    path = default_aco_template_pdb()
    assert path.is_file(), f"missing bundled acetone template PDB: {path}"
    text = path.read_text(encoding="utf-8")
    assert "O1" in text and "ACO" in text


def test_default_tip3_template_pdb_is_bundled() -> None:
    path = default_tip3_template_pdb()
    assert path.is_file(), f"missing bundled TIP3 template PDB: {path}"
    text = path.read_text(encoding="utf-8")
    assert "OH2" in text and "TIP3" in text


def test_default_template_pdb_for_residue_tip3():
    from mmml.cli.run.md_pbc_suite.cluster import _default_template_pdb_for_residue

    path = _default_template_pdb_for_residue("TIP3")
    assert path is not None and path.is_file()
    assert "OH2" in path.read_text(encoding="utf-8")


def test_crystal_image_str_is_bundled() -> None:
    path = crystal_image_str_source()
    assert path.is_file(), f"missing bundled CHARMM helper: {path}"


def test_default_dcm_molecule_xyz_is_bundled() -> None:
    path = default_dcm_molecule_xyz()
    assert path.is_file(), f"missing bundled DCM monomer XYZ: {path}"
    text = path.read_text(encoding="utf-8")
    assert "Cl" in text and "DCM" in text.splitlines()[1]


def test_default_dcm_crystal_cif_is_bundled() -> None:
    path = default_dcm_crystal_cif()
    assert path.is_file(), f"missing bundled DCM crystal CIF: {path}"
    text = path.read_text(encoding="utf-8")
    assert "P b c n" in text or "Pbcn" in text
    assert "_cell_formula_units_Z" in text


def test_both_dcm_pressure_points_are_bundled() -> None:
    """The default must stay the 1.63 GPa entry: presets and doc tables use it."""
    from mmml.paths import DCM_CRYSTAL_CIFS

    assert set(DCM_CRYSTAL_CIFS) == {"pbcn_133gpa", "pbcn_163gpa"}
    for phase in DCM_CRYSTAL_CIFS:
        path = default_dcm_crystal_cif(phase)
        assert path.is_file(), f"missing bundled DCM CIF for {phase}: {path}"
        assert "_cell_measurement_pressure" in path.read_text(encoding="utf-8")
    assert default_dcm_crystal_cif() == default_dcm_crystal_cif("pbcn_163gpa")


def test_unknown_dcm_phase_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="pbcn_133gpa"):
        default_dcm_crystal_cif("ambient")


def test_default_benzene_crystal_cif_is_bundled() -> None:
    path = default_benzene_crystal_cif()
    assert path.is_file(), f"missing bundled benzene crystal CIF: {path}"
    text = path.read_text(encoding="utf-8")
    assert "P 1 21/c" in text or "P21/c" in text.replace(" ", "")
    assert "_cell_formula_units_Z" in text


def test_generate_sample_module_is_packaged() -> None:
    spec = importlib.util.find_spec("mmml.generate.sample.sample_diverse_xyz")
    assert spec is not None and spec.origin
    assert Path(spec.origin).is_file()


def test_mmml_package_root_is_directory() -> None:
    root = _package_dir()
    assert root.is_dir()
    assert (root / "__init__.py").is_file()
