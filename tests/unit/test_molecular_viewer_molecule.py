"""Unit tests for the GL-free structure/trajectory parsers in
mmml.gui.molecular_viewer.molecule. The GL renderer/viewer/VR loop needs a
real display context and is left to manual testing."""

from __future__ import annotations

import pytest

from mmml.gui.molecular_viewer.molecule import (
    Atom,
    _cell_from_cryst1,
    _cell_from_lattice,
    _element_from_name,
    _forces_col_from_properties,
    center_and_scale,
    compute_angles,
    compute_bonds,
    compute_dihedrals,
    load_pdb,
    load_structure,
    load_xyz,
    load_xyz_trajectory,
)


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


def test_atom_pos_and_force():
    a = Atom(element="O", x=1.0, y=2.0, z=3.0)
    assert a.pos == (1.0, 2.0, 3.0)
    assert a.force is None

    b = Atom(element="O", x=0.0, y=0.0, z=0.0, fx=0.1, fy=0.2, fz=0.3)
    assert b.force == (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# _element_from_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    # This helper always takes the leading 1-2 letters verbatim (it doesn't
    # know PDB atom-naming conventions like "CA" == alpha carbon), so "CA"
    # resolves to "Ca" and "HB1" to "Hb", not the biologically intended C/H.
    "name,expected",
    [("CA", "Ca"), ("FE", "Fe"), ("HB1", "Hb"), ("", "?"), ("na", "Na")],
)
def test_element_from_name(name, expected):
    assert _element_from_name(name) == expected


# ---------------------------------------------------------------------------
# CRYST1 / Lattice cell parsing
# ---------------------------------------------------------------------------


def test_cell_from_cryst1_orthorhombic():
    line = "CRYST1   10.000   20.000   30.000  90.00  90.00  90.00 P 1           1\n"
    cell = _cell_from_cryst1(line)
    assert cell is not None
    a, b, c = cell
    assert a == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)
    assert b == pytest.approx((0.0, 20.0, 0.0), abs=1e-6)
    assert c == pytest.approx((0.0, 0.0, 30.0), abs=1e-6)


def test_cell_from_cryst1_malformed_returns_none():
    assert _cell_from_cryst1("CRYST1 not-a-number\n") is None


def test_cell_from_lattice_parses_nine_values():
    comment = 'Lattice="10.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 30.0" Properties=species:S:1:pos:R:3\n'
    cell = _cell_from_lattice(comment)
    assert cell == ((10.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 30.0))


def test_cell_from_lattice_missing_returns_none():
    assert _cell_from_lattice("just a plain comment\n") is None


# ---------------------------------------------------------------------------
# Extended-XYZ Properties= parsing
# ---------------------------------------------------------------------------


def test_forces_col_from_properties_finds_forces_block():
    comment = "Properties=species:S:1:pos:R:3:forces:R:3\n"
    assert _forces_col_from_properties(comment) == 4


def test_forces_col_from_properties_absent():
    comment = "Properties=species:S:1:pos:R:3\n"
    assert _forces_col_from_properties(comment) is None


# ---------------------------------------------------------------------------
# load_pdb
# ---------------------------------------------------------------------------


def _water_pdb_text() -> str:
    return (
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H\n"
        "ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H\n"
        "END\n"
    )


def test_load_pdb_parses_atoms_and_cell(tmp_path):
    p = tmp_path / "water.pdb"
    p.write_text(_water_pdb_text())
    atoms, cell = load_pdb(p)
    assert [a.element for a in atoms] == ["O", "H", "H"]
    assert atoms[0].residue == "HOH"
    assert atoms[1].x == pytest.approx(0.957)
    assert cell is not None
    assert cell[0] == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)


def test_load_pdb_falls_back_to_element_from_name_when_column_missing(tmp_path):
    line = "ATOM      1  O   ALA A   1       1.000   2.000   3.000  1.00  0.00\n"
    p = tmp_path / "no_elem_col.pdb"
    p.write_text(line)
    atoms, cell = load_pdb(p)
    assert len(atoms) == 1
    assert atoms[0].element == "O"
    assert cell is None


# ---------------------------------------------------------------------------
# load_xyz / load_xyz_trajectory
# ---------------------------------------------------------------------------


def _water_xyz_text() -> str:
    return "3\ncomment\nO 0.0 0.0 0.0\nH 0.957 0.0 0.0\nH -0.240 0.927 0.0\n"


def test_load_xyz_basic(tmp_path):
    p = tmp_path / "water.xyz"
    p.write_text(_water_xyz_text())
    atoms, cell = load_xyz(p)
    assert len(atoms) == 3
    assert atoms[0].element == "O"
    assert cell is None
    assert atoms[0].force is None


def test_load_xyz_with_forces(tmp_path):
    text = (
        '2\nProperties=species:S:1:pos:R:3:forces:R:3 Lattice="5 0 0 0 5 0 0 0 5"\n'
        "O 0.0 0.0 0.0 0.1 0.2 0.3\n"
        "H 1.0 0.0 0.0 0.4 0.5 0.6\n"
    )
    p = tmp_path / "with_forces.xyz"
    p.write_text(text)
    atoms, cell = load_xyz(p)
    assert atoms[0].force == pytest.approx((0.1, 0.2, 0.3))
    assert atoms[1].force == pytest.approx((0.4, 0.5, 0.6))
    assert cell == ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0))


def test_load_xyz_trajectory_multi_frame(tmp_path):
    text = _water_xyz_text() + _water_xyz_text()
    p = tmp_path / "traj.xyz"
    p.write_text(text)
    frames, cells = load_xyz_trajectory(p)
    assert len(frames) == 2
    assert len(cells) == 2
    assert all(len(frame) == 3 for frame in frames)


# ---------------------------------------------------------------------------
# load_structure: format auto-detection
# ---------------------------------------------------------------------------


def test_load_structure_detects_pdb(tmp_path):
    p = tmp_path / "water.pdb"
    p.write_text(_water_pdb_text())
    frames, cells, meta = load_structure(p)
    assert len(frames) == 1
    assert len(frames[0]) == 3
    assert meta == [{}]


def test_load_structure_detects_single_frame_xyz(tmp_path):
    p = tmp_path / "water.xyz"
    p.write_text(_water_xyz_text())
    frames, cells, meta = load_structure(p)
    assert len(frames) == 1
    assert len(frames[0]) == 3


def test_load_structure_detects_multi_frame_xyz(tmp_path):
    p = tmp_path / "traj.xyz"
    p.write_text(_water_xyz_text() * 3)
    frames, cells, meta = load_structure(p)
    assert len(frames) == 3


def test_load_structure_rejects_unknown_extension(tmp_path):
    p = tmp_path / "water.foo"
    p.write_text("garbage")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_structure(p)


# ---------------------------------------------------------------------------
# compute_bonds / compute_angles / compute_dihedrals
# ---------------------------------------------------------------------------


def _water_atoms() -> list[Atom]:
    return [
        Atom(element="O", x=0.0, y=0.0, z=0.0),
        Atom(element="H", x=0.957, y=0.0, z=0.0),
        Atom(element="H", x=-0.240, y=0.927, z=0.0),
    ]


def test_compute_bonds_finds_both_oh_bonds():
    # default max_bond=2.0 is looser than a real O-H bond and also catches
    # water's ~1.5 A H...H distance, so use a realistic cutoff here.
    bonds = compute_bonds(_water_atoms(), max_bond=1.2)
    assert set(bonds) == {(0, 1), (0, 2)}


def test_compute_bonds_default_cutoff_also_links_the_two_hydrogens():
    bonds = compute_bonds(_water_atoms())
    assert set(bonds) == {(0, 1), (0, 2), (1, 2)}


def test_compute_bonds_no_bond_between_distant_atoms():
    atoms = [
        Atom(element="O", x=0.0, y=0.0, z=0.0),
        Atom(element="O", x=20.0, y=0.0, z=0.0),
    ]
    assert compute_bonds(atoms) == []


def test_compute_angles_water_hoh_angle_close_to_real_value():
    atoms = _water_atoms()
    bonds = compute_bonds(atoms, max_bond=1.2)
    angles = compute_angles(atoms, bonds)
    assert len(angles) == 1
    i, j, k, angle_deg = angles[0]
    assert {i, k} == {1, 2}
    assert j == 0
    assert angle_deg == pytest.approx(104.5, abs=1.0)


def test_compute_dihedrals_empty_for_water():
    atoms = _water_atoms()
    bonds = compute_bonds(atoms, max_bond=1.2)
    assert compute_dihedrals(atoms, bonds) == []


def test_compute_dihedrals_four_atom_chain():
    # a simple staggered A-B-C-D chain with a known 90 degree torsion.
    atoms = [
        Atom(element="C", x=0.0, y=0.0, z=1.0),
        Atom(element="C", x=0.0, y=0.0, z=0.0),
        Atom(element="C", x=1.0, y=0.0, z=0.0),
        Atom(element="C", x=1.0, y=1.0, z=0.0),
    ]
    bonds = [(0, 1), (1, 2), (2, 3)]
    dihedrals = compute_dihedrals(atoms, bonds)
    assert len(dihedrals) == 1
    i, j, k, l, angle_deg = dihedrals[0]
    assert (i, j, k, l) == (0, 1, 2, 3)
    assert abs(angle_deg) == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# center_and_scale
# ---------------------------------------------------------------------------


def test_center_and_scale_centers_at_origin():
    atoms = _water_atoms()
    centered = center_and_scale(atoms)
    cx = sum(a.x for a in centered) / len(centered)
    cy = sum(a.y for a in centered) / len(centered)
    cz = sum(a.z for a in centered) / len(centered)
    assert (cx, cy, cz) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_center_and_scale_applies_scale_factor():
    atoms = [Atom(element="C", x=1.0, y=0.0, z=0.0), Atom(element="C", x=-1.0, y=0.0, z=0.0)]
    scaled = center_and_scale(atoms, scale=2.0)
    assert scaled[0].x == pytest.approx(2.0)
    assert scaled[1].x == pytest.approx(-2.0)


def test_center_and_scale_empty_list_is_noop():
    assert center_and_scale([]) == []
