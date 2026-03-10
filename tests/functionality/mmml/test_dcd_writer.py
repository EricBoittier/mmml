"""
Unit tests for the DCD trajectory writer (CHARMM-compatible format).

Tests the pure Python DCD writer without MDAnalysis or other optional dependencies.
"""
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DCD_WRITER_PATH = PROJECT_ROOT / "mmml" / "utils" / "dcd_writer.py"


def _load_dcd_writer():
    """Load dcd_writer module without triggering mmml package imports (avoids jax etc)."""
    spec = importlib.util.spec_from_file_location("dcd_writer", DCD_WRITER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _can_import_ase() -> bool:
    try:
        __import__("ase")
        return True
    except Exception:
        return False


@pytest.fixture
def dcd_writer():
    return _load_dcd_writer()


@pytest.fixture
def water_atoms():
    """Minimal ASE Atoms for water (3 atoms)."""
    if not _can_import_ase():
        pytest.skip("ase not available")
    from ase import Atoms
    return Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.87, 0]])


def _read_dcd_header(path):
    """Read DCD header and return key fields."""
    with open(path, "rb") as f:
        rec1 = struct.unpack("<i", f.read(4))[0]
        cord = f.read(4)
        n_frames = struct.unpack("<i", f.read(4))[0]
        istart = struct.unpack("<i", f.read(4))[0]
        nsavc = struct.unpack("<i", f.read(4))[0]
        f.read(4 * 6)  # skip numsteps, zeros
        dt = struct.unpack("<f", f.read(4))[0]
        iscell = struct.unpack("<i", f.read(4))[0]
        f.read(4 * 9)  # skip zeros, charmm version
        rec2 = struct.unpack("<i", f.read(4))[0]
        # Title block
        block_size = struct.unpack("<i", f.read(4))[0]
        n_titles = struct.unpack("<i", f.read(4))[0]
        f.read(block_size - 4)
        f.read(4)
        # Natoms block: 4 (size) + 4 (n_atoms) + 4 (size)
        struct.unpack("<i", f.read(4))[0]  # 4
        n_atoms = struct.unpack("<i", f.read(4))[0]
        struct.unpack("<i", f.read(4))[0]  # trailing 4
        first_frame_byte = f.tell()
    return {
        "rec1": rec1,
        "cord": cord,
        "n_frames": n_frames,
        "istart": istart,
        "nsavc": nsavc,
        "dt": dt,
        "iscell": iscell,
        "rec2": rec2,
        "n_atoms": n_atoms,
        "first_frame_byte": first_frame_byte,
    }


def _read_dcd_frame_coords(f, n_atoms, has_unitcell):
    """Read one frame of coordinates from DCD file."""
    if has_unitcell:
        rec = struct.unpack("<i", f.read(4))[0]
        assert rec == 48
        uc = np.fromfile(f, dtype=np.float64, count=6)
        struct.unpack("<i", f.read(4))[0]
    block_size = n_atoms * 4
    struct.unpack("<i", f.read(4))[0]  # block start
    x = np.fromfile(f, dtype=np.float32, count=n_atoms)
    struct.unpack("<i", f.read(4))[0]  # block end
    struct.unpack("<i", f.read(4))[0]  # block start
    y = np.fromfile(f, dtype=np.float32, count=n_atoms)
    struct.unpack("<i", f.read(4))[0]
    struct.unpack("<i", f.read(4))[0]
    z = np.fromfile(f, dtype=np.float32, count=n_atoms)
    struct.unpack("<i", f.read(4))[0]
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_basic(dcd_writer, water_atoms, tmp_path):
    """Test writing a minimal DCD file without PBC."""
    positions = np.random.rand(5, 3, 3).astype(np.float32)
    path = tmp_path / "test.dcd"
    dcd_writer.save_trajectory_dcd(path, positions, water_atoms)

    assert path.exists()
    header = _read_dcd_header(path)
    assert header["rec1"] == 84
    assert header["cord"] == b"CORD"
    assert header["n_frames"] == 5
    assert header["n_atoms"] == 3
    assert header["iscell"] == 0
    assert header["nsavc"] == 1
    assert header["dt"] == 1.0


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_with_boxes(dcd_writer, water_atoms, tmp_path):
    """Test writing DCD with unit cell (NPT-style boxes)."""
    positions = np.random.rand(3, 3, 3).astype(np.float32)
    boxes = [np.array([10.0, 10.0, 10.0])] * 3
    path = tmp_path / "test_pbc.dcd"
    dcd_writer.save_trajectory_dcd(
        path, positions, water_atoms, boxes=boxes, dt_ps=0.001, steps_per_frame=100
    )

    header = _read_dcd_header(path)
    assert header["iscell"] == 1
    assert header["n_frames"] == 3
    assert header["n_atoms"] == 3
    assert header["nsavc"] == 100
    assert abs(header["dt"] - 0.001) < 1e-6


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_coordinates_roundtrip(dcd_writer, water_atoms, tmp_path):
    """Test that written coordinates can be read back correctly."""
    positions = np.array(
        [[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=np.float32,
    )
    path = tmp_path / "roundtrip.dcd"
    dcd_writer.save_trajectory_dcd(path, positions, water_atoms)

    header = _read_dcd_header(path)
    n_atoms = header["n_atoms"]
    has_unitcell = header["iscell"] == 1

    with open(path, "rb") as f:
        f.seek(header["first_frame_byte"])
        read_coords = _read_dcd_frame_coords(f, n_atoms, has_unitcell)

    np.testing.assert_allclose(read_coords, positions[0], atol=1e-5)


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_multiframe_roundtrip(dcd_writer, water_atoms, tmp_path):
    """Test multi-frame trajectory coordinate roundtrip."""
    positions = np.random.rand(4, 3, 3).astype(np.float32)
    path = tmp_path / "multiframe.dcd"
    dcd_writer.save_trajectory_dcd(path, positions, water_atoms)

    header = _read_dcd_header(path)
    n_atoms = header["n_atoms"]
    n_frames = header["n_frames"]

    with open(path, "rb") as f:
        f.seek(header["first_frame_byte"])
        for i in range(n_frames):
            read_coords = _read_dcd_frame_coords(f, n_atoms, False)
            np.testing.assert_allclose(read_coords, positions[i], atol=1e-5)


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_reshape(dcd_writer, water_atoms, tmp_path):
    """Test that positions with different shapes are reshaped correctly."""
    # Shape (2, 9) = 2 frames * (3 atoms * 3 coords) -> reshapes to (2, 3, 3)
    positions_flat = np.random.rand(2, 9).astype(np.float32)
    path = tmp_path / "reshaped.dcd"
    dcd_writer.save_trajectory_dcd(path, positions_flat, water_atoms)

    header = _read_dcd_header(path)
    assert header["n_frames"] == 2
    assert header["n_atoms"] == 3


def test_box_to_dcd_unitcell_orthorhombic(dcd_writer):
    """Test _box_to_dcd_unitcell for orthorhombic (Lx, Ly, Lz) box."""
    uc = dcd_writer._box_to_dcd_unitcell(np.array([10.0, 20.0, 30.0]))
    expected = np.array([10.0, 0.0, 20.0, 0.0, 0.0, 30.0])
    np.testing.assert_allclose(uc, expected)


def test_box_to_dcd_unitcell_diagonal_matrix(dcd_writer):
    """Test _box_to_dcd_unitcell for 3x3 diagonal matrix."""
    box = np.diag([5.0, 6.0, 7.0])
    uc = dcd_writer._box_to_dcd_unitcell(box)
    expected = np.array([5.0, 0.0, 6.0, 0.0, 0.0, 7.0])
    np.testing.assert_allclose(uc, expected)


def test_box_to_dcd_unitcell_triclinic(dcd_writer):
    """Test _box_to_dcd_unitcell for triclinic box (non-orthogonal vectors)."""
    # Box with a=10, b=10, c=10 and gamma=60° (cos=0.5)
    a = np.array([10.0, 0.0, 0.0])
    b = np.array([5.0, 8.660254, 0.0])  # |b|=10, angle with a = 60°
    c = np.array([0.0, 0.0, 10.0])
    box = np.array([a, b, c])
    uc = dcd_writer._box_to_dcd_unitcell(box)
    assert uc[0] == pytest.approx(10.0)
    assert uc[2] == pytest.approx(10.0)
    assert uc[5] == pytest.approx(10.0)
    assert uc[1] == pytest.approx(0.5, abs=0.01)  # cos(gamma)


@pytest.mark.skipif(not _can_import_ase(), reason="ase not available")
def test_save_trajectory_dcd_creates_parent_dir(dcd_writer, water_atoms, tmp_path):
    """Test that parent directory is created when it does not exist."""
    path = tmp_path / "subdir" / "nested" / "traj.dcd"
    positions = np.random.rand(1, 3, 3).astype(np.float32)
    dcd_writer.save_trajectory_dcd(path, positions, water_atoms)
    assert path.exists()
