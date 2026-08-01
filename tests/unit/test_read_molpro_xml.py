"""Parsing of Molpro XML output into arrays.

This parser is the front door for quantum-chemistry reference data: whatever it
returns becomes training targets. It is also the worst place for a silent
failure, because a selector that quietly matches nothing returns ``None`` and a
geometry read from the wrong ``<molecule>`` block is still a perfectly valid
array of numbers -- neither raises, and both poison a dataset.

It had 10.6% coverage. The fixtures below are hand-written XML small enough to
verify by eye, so every assertion is against a value visible in the source
document rather than against whatever the parser happens to produce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.parse_molpro.read_molden import (
    MolproData,
    MolproXMLParser,
    read_molpro_xml,
)

_CML = "http://www.xml-cml.org/schema"
_MOLPRO = "http://www.molpro.net/schema/molpro-output"


def _doc(body: str, *, namespaced: bool = True) -> str:
    if namespaced:
        return (
            f'<molpro xmlns="{_MOLPRO}" xmlns:cml="{_CML}">\n{body}\n</molpro>'
        )
    return f"<molpro>\n{body}\n</molpro>"


def _write(tmp_path: Path, body: str, *, namespaced: bool = True, name: str = "out.xml") -> str:
    path = tmp_path / name
    path.write_text(_doc(body, namespaced=namespaced), encoding="utf-8")
    return str(path)


# Two geometries: the optimiser's first step and its converged result.
_TWO_GEOMETRIES = """
  <cml:molecule xmlns:cml="{cml}">
    <cml:atomArray>
      <cml:atom elementType="O" x3="0.0" y3="0.0" z3="0.0"/>
      <cml:atom elementType="H" x3="0.9" y3="0.0" z3="0.0"/>
    </cml:atomArray>
  </cml:molecule>
  <cml:molecule xmlns:cml="{cml}">
    <cml:atomArray>
      <cml:atom elementType="O" x3="0.0" y3="0.0" z3="0.0"/>
      <cml:atom elementType="H" x3="0.96" y3="0.0" z3="0.0"/>
    </cml:atomArray>
  </cml:molecule>
""".format(cml=_CML)


# --- geometry ---------------------------------------------------------------


def test_geometry_maps_element_symbols_to_atomic_numbers(tmp_path):
    parser = MolproXMLParser(_write(tmp_path, _TWO_GEOMETRIES))
    z, r = parser.parse_geometry()
    assert z.tolist() == [8, 1]
    assert r.shape == (2, 3)


def test_geometry_defaults_to_the_last_block(tmp_path):
    """An optimisation writes one <molecule> per step; the converged one is last."""
    parser = MolproXMLParser(_write(tmp_path, _TWO_GEOMETRIES))
    _, r = parser.parse_geometry()
    assert r[1, 0] == pytest.approx(0.96)


def test_geometry_can_take_the_first_block(tmp_path):
    parser = MolproXMLParser(_write(tmp_path, _TWO_GEOMETRIES))
    _, r = parser.parse_geometry(use_last=False)
    assert r[1, 0] == pytest.approx(0.9)


def test_geometry_accepts_element_numbers_instead_of_symbols(tmp_path):
    body = f"""
      <cml:molecule xmlns:cml="{_CML}">
        <cml:atomArray>
          <cml:atom elementNumber="6" x3="0.0" y3="0.0" z3="0.0"/>
        </cml:atomArray>
      </cml:molecule>
    """
    z, r = MolproXMLParser(_write(tmp_path, body)).parse_geometry()
    assert z.tolist() == [6]


def test_geometry_is_none_when_absent(tmp_path):
    z, r = MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_geometry()
    assert z is None and r is None


def test_geometry_parses_a_document_without_namespaces(tmp_path):
    body = """
      <molecule>
        <atom elementNumber="1" x3="0.0" y3="0.0" z3="0.0"/>
        <atom elementNumber="1" x3="0.74" y3="0.0" z3="0.0"/>
      </molecule>
    """
    z, r = MolproXMLParser(_write(tmp_path, body, namespaced=False)).parse_geometry()
    assert z.tolist() == [1, 1]
    assert r[1, 0] == pytest.approx(0.74)


# --- energies ---------------------------------------------------------------


def test_energies_are_keyed_by_method(tmp_path):
    body = """
      <jobstep>
        <property name="Energy" method="RHF-SCF" value="-76.02" />
        <property name="Energy" method="MP2" value="-76.23" />
      </jobstep>
    """
    got = MolproXMLParser(_write(tmp_path, body)).parse_energies()
    assert got == {"RHF-SCF": pytest.approx(-76.02), "MP2": pytest.approx(-76.23)}


def test_non_energy_properties_are_ignored(tmp_path):
    body = """
      <jobstep>
        <property name="Energy" method="MP2" value="-76.23" />
        <property name="Dipole moment" method="MP2" value="0.0 0.0 0.8" />
      </jobstep>
    """
    assert set(MolproXMLParser(_write(tmp_path, body)).parse_energies()) == {"MP2"}


def test_energies_is_empty_not_none_when_absent(tmp_path):
    assert MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_energies() == {}


# --- dipole -----------------------------------------------------------------


def test_dipole_is_read_as_three_components(tmp_path):
    body = '<jobstep><property name="Dipole moment" value="0.1 -0.2 0.8"/></jobstep>'
    got = MolproXMLParser(_write(tmp_path, body)).parse_dipole()
    assert got == pytest.approx([0.1, -0.2, 0.8])


def test_dipole_with_wrong_component_count_is_rejected(tmp_path):
    """A two-component 'dipole' is a malformed file, not a 2D dipole."""
    body = '<jobstep><property name="Dipole moment" value="0.1 -0.2"/></jobstep>'
    assert MolproXMLParser(_write(tmp_path, body)).parse_dipole() is None


def test_dipole_is_none_when_absent(tmp_path):
    assert MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_dipole() is None


# --- gradient ---------------------------------------------------------------


def test_gradient_reshapes_to_n_atoms_by_three(tmp_path):
    body = "<jobstep><gradient>0.1 0.2 0.3 -0.1 -0.2 -0.3</gradient></jobstep>"
    got = MolproXMLParser(_write(tmp_path, body)).parse_gradient()
    assert got.shape == (2, 3)
    assert got[1].tolist() == pytest.approx([-0.1, -0.2, -0.3])


def test_gradient_defaults_to_the_last_block(tmp_path):
    body = """
      <jobstep><gradient>1.0 0.0 0.0</gradient></jobstep>
      <jobstep><gradient>2.0 0.0 0.0</gradient></jobstep>
    """
    parser = MolproXMLParser(_write(tmp_path, body))
    assert parser.parse_gradient()[0, 0] == pytest.approx(2.0)
    assert parser.parse_gradient(use_last=False)[0, 0] == pytest.approx(1.0)


def test_gradient_is_none_when_absent(tmp_path):
    assert MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_gradient() is None


# --- hessian ----------------------------------------------------------------


def test_hessian_is_reshaped_to_a_square_matrix(tmp_path):
    body = "<jobstep><hessian>1 2 3 4 5 6 7 8 9</hessian></jobstep>"
    got = MolproXMLParser(_write(tmp_path, body)).parse_hessian()
    assert got.shape == (3, 3)
    assert got[2, 2] == pytest.approx(9.0)


def test_non_square_hessian_is_rejected(tmp_path):
    """Silently reshaping 8 values would misalign every force constant."""
    body = "<jobstep><hessian>1 2 3 4 5 6 7 8</hessian></jobstep>"
    assert MolproXMLParser(_write(tmp_path, body)).parse_hessian() is None


def test_hessian_is_none_when_absent(tmp_path):
    assert MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_hessian() is None


# --- variables --------------------------------------------------------------


def test_scalar_variable_is_a_float(tmp_path):
    body = '<variables><variable name="ENERGY" length="1"><value>-76.02</value></variable></variables>'
    got = MolproXMLParser(_write(tmp_path, body)).parse_variables()
    assert got["ENERGY"] == pytest.approx(-76.02)


def test_array_variable_is_an_array(tmp_path):
    body = '<variables><variable name="DIP" length="3"><value>0.1 0.2 0.3</value></variable></variables>'
    got = MolproXMLParser(_write(tmp_path, body)).parse_variables()
    assert isinstance(got["DIP"], np.ndarray)
    assert got["DIP"] == pytest.approx([0.1, 0.2, 0.3])


def test_non_numeric_variable_falls_back_to_its_string(tmp_path):
    body = '<variables><variable name="BASIS" length="1"><value>aug-cc-pVTZ</value></variable></variables>'
    got = MolproXMLParser(_write(tmp_path, body)).parse_variables()
    assert got["BASIS"] == "aug-cc-pVTZ"


def test_variables_is_empty_when_absent(tmp_path):
    assert MolproXMLParser(_write(tmp_path, "<jobstep/>")).parse_variables() == {}


# --- cube files -------------------------------------------------------------


def _cube_text(nx: int, ny: int, nz: int, values: list[float], n_atoms: int = 1) -> str:
    lines = ["comment one", "comment two", f"{n_atoms} 0.0 0.0 0.0",
             f"{nx} 0.1 0.0 0.0", f"{ny} 0.0 0.1 0.0", f"{nz} 0.0 0.0 0.1"]
    lines += ["1 1.0 0.0 0.0 0.0"] * n_atoms
    lines += [" ".join(f"{v}" for v in values)]
    return "\n".join(lines) + "\n"


def test_cube_reader_returns_the_declared_grid_shape(tmp_path):
    path = tmp_path / "esp.cube"
    values = [float(i) for i in range(2 * 3 * 4)]
    path.write_text(_cube_text(2, 3, 4, values))

    parser = MolproXMLParser(_write(tmp_path, "<jobstep/>"))
    got = parser._read_cube_file(str(path))

    assert got.shape == (2, 3, 4)
    # Cube data is written x-slowest, z-fastest.
    assert got[0, 0, 0] == pytest.approx(0.0)
    assert got[1, 2, 3] == pytest.approx(23.0)


def test_cube_reader_skips_the_declared_atom_block(tmp_path):
    """Miscounting atom lines shifts every voxel; pin it with 3 atoms."""
    path = tmp_path / "d.cube"
    path.write_text(_cube_text(1, 1, 2, [7.0, 8.0], n_atoms=3))

    got = MolproXMLParser(_write(tmp_path, "<jobstep/>"))._read_cube_file(str(path))

    assert got.ravel().tolist() == pytest.approx([7.0, 8.0])


# --- parse_all / read_molpro_xml -------------------------------------------


def test_parse_all_populates_the_container(tmp_path):
    body = _TWO_GEOMETRIES + """
      <jobstep>
        <property name="Energy" method="MP2" value="-76.23"/>
        <property name="Dipole moment" value="0.0 0.0 0.8"/>
        <gradient>0.0 0.0 0.1 0.0 0.0 -0.1</gradient>
      </jobstep>
    """
    data = MolproXMLParser(_write(tmp_path, body)).parse_all(load_cubes=False)

    assert isinstance(data, MolproData)
    assert data.atomic_numbers.tolist() == [8, 1]
    assert data.energies["MP2"] == pytest.approx(-76.23)
    assert data.dipole_moment == pytest.approx([0.0, 0.0, 0.8])
    assert data.gradient.shape == (2, 3)


def test_read_molpro_xml_matches_the_parser(tmp_path):
    body = _TWO_GEOMETRIES + '<jobstep><property name="Energy" method="MP2" value="-1.0"/></jobstep>'
    path = _write(tmp_path, body)

    via_helper = read_molpro_xml(path, load_cubes=False)
    via_parser = MolproXMLParser(path).parse_all(load_cubes=False)

    assert via_helper.energies == via_parser.energies
    assert via_helper.atomic_numbers.tolist() == via_parser.atomic_numbers.tolist()


def test_read_molpro_xml_honours_use_last_geometry(tmp_path):
    path = _write(tmp_path, _TWO_GEOMETRIES)
    assert read_molpro_xml(path, use_last_geometry=True, load_cubes=False).coordinates[1, 0] == pytest.approx(0.96)
    assert read_molpro_xml(path, use_last_geometry=False, load_cubes=False).coordinates[1, 0] == pytest.approx(0.9)


def test_empty_document_yields_an_empty_container_not_a_crash(tmp_path):
    data = read_molpro_xml(_write(tmp_path, "<jobstep/>"), load_cubes=False)
    assert data.atomic_numbers is None
    assert data.energies == {}
    assert data.dipole_moment is None


def test_malformed_xml_raises(tmp_path):
    """A truncated Molpro job must not parse to an empty-but-valid result."""
    import xml.etree.ElementTree as ET

    path = tmp_path / "broken.xml"
    path.write_text("<molpro><jobstep>")
    with pytest.raises(ET.ParseError):
        MolproXMLParser(str(path))


def test_molpro_data_defaults_are_independent_between_instances():
    """Mutable dataclass defaults shared across instances would cross-contaminate
    every parsed file in a batch."""
    a, b = MolproData(), MolproData()
    a.energies["scf"] = -1.0
    a.variables["x"] = 2.0
    assert b.energies == {} and b.variables == {}
