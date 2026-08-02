"""Tests for ``mmml npz2traj``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.io import read
from ase.io.trajectory import Trajectory

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc import convert_npz_traj
from mmml.cli.parser_utils import parser_available
from mmml.cli.registry import command_by_name
from mmml.data.units import DEBYE_TO_EANGSTROM, HARTREE_BOHR_TO_EV_ANGSTROM, HARTREE_TO_EV


def _write_sample_npz(path: Path, *, with_pad: bool = True) -> None:
    n_real = 3
    n_pad = 2 if with_pad else 0
    n_atoms = n_real + n_pad
    R = np.zeros((2, n_atoms, 3), dtype=np.float64)
    R[0, :n_real] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    R[1, :n_real] = [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]]
    Z = np.zeros((2, n_atoms), dtype=np.int64)
    Z[:, :n_real] = [8, 1, 1]
    E = np.array([-76.0, -76.1], dtype=np.float64)
    F = np.zeros_like(R)
    F[:, :n_real] = 0.01
    D = np.array([[1.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float64)
    N = np.array([n_real, n_real], dtype=np.int64)
    mono = np.zeros((2, n_atoms), dtype=np.float64)
    mono[:, :n_real] = [-0.8, 0.4, 0.4]
    np.savez(
        path,
        R=R,
        Z=Z,
        E=E,
        F=F,
        D=D,
        N=N,
        mono=mono,
        id=np.array(["a", "b"], dtype=object),
        method="MP2",
        units={"E": "Hartree", "F": "Hartree/Bohr", "D": "Debye", "R": "Angstrom"},
    )


def test_npz2traj_cli_is_registered():
    spec = command_by_name("npz2traj")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.convert_npz_traj"
    assert parser_available("npz2traj")
    assert convert_npz_traj.build_parser().prog == "mmml npz2traj"


def test_npz2traj_help_is_reachable(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mmml", "npz2traj", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Convert MMML NPZ" in out
    assert "--ase-units" in out


def test_npz_to_traj_attaches_energy_forces_dipole_charges(tmp_path: Path) -> None:
    npz = tmp_path / "data.npz"
    out = tmp_path / "out.traj"
    _write_sample_npz(npz)

    n = convert_npz_traj.npz_to_trajectory(npz, out, verbose=False)
    assert n == 2

    frames = read(str(out), index=":")
    assert len(frames) == 2
    assert len(frames[0]) == 3  # padding stripped via N / Z>0
    assert frames[0].info["energy_unit"] == "Hartree"
    assert frames[0].info["dipole_unit"] == "Debye"
    assert frames[0].info["id"] == "a"
    assert frames[0].calc is not None
    assert frames[0].get_potential_energy() == pytest.approx(-76.0)
    np.testing.assert_allclose(frames[0].get_forces(), 0.01)
    np.testing.assert_allclose(frames[0].get_dipole_moment(), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(frames[0].get_charges(), [-0.8, 0.4, 0.4])
    # Unit labels and extras survive in atoms.info for the data inspector.
    assert "npz_source" in frames[0].info


def test_npz_to_traj_ase_units_conversion(tmp_path: Path) -> None:
    npz = tmp_path / "data.npz"
    out = tmp_path / "out.traj"
    _write_sample_npz(npz, with_pad=False)

    convert_npz_traj.npz_to_trajectory(npz, out, ase_units=True, verbose=False)
    atoms = read(str(out), index=0)
    assert atoms.info["energy_unit"] == "eV"
    assert atoms.info["energy_hartree"] == pytest.approx(-76.0)
    assert atoms.info["dipole_unit"] == "e*Angstrom"
    assert atoms.get_potential_energy() == pytest.approx(-76.0 * HARTREE_TO_EV)
    np.testing.assert_allclose(
        atoms.get_forces(),
        np.full((3, 3), 0.01 * HARTREE_BOHR_TO_EV_ANGSTROM),
    )
    np.testing.assert_allclose(
        atoms.get_dipole_moment(),
        np.array([1.0, 0.0, 0.0]) * DEBYE_TO_EANGSTROM,
    )
    np.testing.assert_allclose(
        atoms.info["dipole_debye"],
        [1.0, 0.0, 0.0],
    )


def test_npz_to_extxyz_and_stride(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    npz = tmp_path / "data.npz"
    out = tmp_path / "out.extxyz"
    _write_sample_npz(npz)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml npz2traj",
            str(npz),
            "-o",
            str(out),
            "--stride",
            "2",
            "--quiet",
        ],
    )
    assert convert_npz_traj.main() == 0
    frames = read(str(out), index=":")
    assert len(frames) == 1
    assert frames[0].info["frame_index"] == 0


def test_md_npz_to_dcd_with_resname_split(tmp_path: Path) -> None:
    """jaxmd-unified trajectory.npz → PSF+DCD (+ TRIA / TIP3 copies)."""
    psf = tmp_path / "model.psf"
    psf.write_text(
        "\n".join(
            [
                "PSF",
                "",
                "       5 !NATOM",
                "       1 A    1    TRIA N    N     -0.470000       14.0070           0",
                "       2 A    1    TRIA HN   H      0.310000        1.0080           0",
                "       3 WAT  1    TIP3 OH2  OT    -0.834000       15.9994           0",
                "       4 WAT  1    TIP3 H1   HT     0.417000        1.0080           0",
                "       5 WAT  1    TIP3 H2   HT     0.417000        1.0080           0",
                "",
                "       3 !NBOND: bonds",
                "       1       2       3       4       3       5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    npz = tmp_path / "trajectory.npz"
    positions = np.zeros((3, 5, 3), dtype=np.float64)
    positions[:, :, 0] = np.arange(5)[None, :]
    boxes = np.broadcast_to(np.eye(3)[None, ...] * 20.0, (3, 3, 3)).copy()
    np.savez(
        npz,
        positions=positions,
        Z=np.array([7, 1, 8, 1, 1], dtype=np.int32),
        boxes=boxes,
        energies=np.array([-1.0, -1.1, -1.2]),
    )
    out = tmp_path / "all.dcd"
    n = convert_npz_traj.export_md_npz(
        npz,
        out,
        psf=psf,
        split_resnames="TRIA,TIP3",
        verbose=False,
    )
    assert n == 3
    assert out.is_file()
    assert out.with_suffix(".psf").is_file()
    tria = tmp_path / "all.TRIA.dcd"
    tip3 = tmp_path / "all.TIP3.dcd"
    assert tria.is_file() and tip3.is_file()
    assert tria.with_suffix(".psf").is_file()
    assert "TIP3" not in tria.with_suffix(".psf").read_text(encoding="utf-8")
    assert "TRIA" not in tip3.with_suffix(".psf").read_text(encoding="utf-8")


def test_gui_ase_frame_reads_dipole_and_charges(tmp_path: Path) -> None:
    from mmml.gui.api.parsers import MolecularFileParser

    npz = tmp_path / "data.npz"
    traj = tmp_path / "out.traj"
    _write_sample_npz(npz)
    convert_npz_traj.npz_to_trajectory(npz, traj, verbose=False)

    parser = MolecularFileParser(traj)
    meta = parser.get_metadata()
    assert "dipole" in meta.available_properties
    assert "charges" in meta.available_properties
    assert "forces" in meta.available_properties

    frame = parser.get_frame(0)
    assert frame.energy == pytest.approx(-76.0)
    assert frame.dipole == pytest.approx([1.0, 0.0, 0.0])
    assert frame.charges is not None
    np.testing.assert_allclose(frame.charges, [-0.8, 0.4, 0.4])

    props = parser.get_all_properties()
    assert "dipole_magnitude" in props
    assert props["dipole_magnitude"][0] == pytest.approx(1.0)
