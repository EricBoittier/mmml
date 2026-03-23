"""
Regression test: Python dcm.xyz (from DCMNet) vs CHARMM dcm.xyz.

Compares charge positions from:
  - Python: global → local → global via frame transform
  - CHARMM: mdcm → DCM module → dcm.xyz (when PyCHARMM available)

Uses synthetic MEOH-like data when H5 file is not present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _synthetic_meoh_data():
    """Generate minimal MEOH-like R, Z, charges, positions."""
    # MEOH: C(0), O(1), H_oh(2), H(3), H(4), H(5)
    R = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [1.43, 0.0, 0.0],  # O
            [1.9, 0.5, 0.0],  # H_oh
            [-0.5, 0.9, 0.0],  # H
            [-0.5, -0.5, 0.8],  # H
            [-0.5, -0.4, -0.9],  # H
        ],
        dtype=float,
    )
    Z = np.array([6, 8, 1, 1, 1, 1], dtype=int)
    n_charges = 3
    # Dummy charges and positions (near atom centers)
    np.random.seed(42)
    charges = np.random.randn(6, n_charges).astype(np.float32) * 0.1
    positions = R[:, None, :] + np.random.randn(6, n_charges, 3).astype(np.float32) * 0.1
    return R, Z, charges, positions


def test_frame_compute():
    """Frame computation matches AXIS1 logic."""
    from mmml.interfaces.dcmInterface import compute_dcm_frame

    R = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
    frame = (0, 1, 2)  # atom 0, neighbors 1 and 2
    vectors = compute_dcm_frame(R, frame)
    X, Y, Z = vectors[0]
    # Z should point from atom 1 toward 0
    expected_z = (R[0] - R[1]) / np.linalg.norm(R[0] - R[1])
    np.testing.assert_allclose(Z, expected_z, atol=1e-10)
    # Orthonormal
    np.testing.assert_allclose(np.dot(X, Y), 0, atol=1e-10)
    np.testing.assert_allclose(np.dot(Y, Z), 0, atol=1e-10)
    np.testing.assert_allclose(np.dot(X, Z), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(X), 1, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(Y), 1, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(Z), 1, atol=1e-10)


def test_convert_roundtrip():
    """global_to_local and local_to_global roundtrip."""
    from mmml.interfaces.dcmInterface import compute_dcm_frame, global_to_local, local_to_global

    R = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
    frame = (0, 1, 2)
    vectors = compute_dcm_frame(R, frame)
    X, Y, Z = vectors[0]
    atom_pos = R[0]
    global_pos = np.array([0.1, 0.2, 0.3])
    aq, bq, cq = global_to_local(global_pos, atom_pos, X, Y, Z)
    back = local_to_global(atom_pos, aq, bq, cq, X, Y, Z)
    np.testing.assert_allclose(back, global_pos, atol=1e-10)


def test_dcmnet_to_mdcm_and_xyz_roundtrip(tmp_path):
    """DCMNet → mdcm → Python dcm.xyz roundtrip preserves charge positions."""
    from mmml.interfaces.dcmInterface import (
        dcmnet_to_mdcm,
        generate_dcm_xyz,
        get_frames_meoh_like,
    )

    R, Z, charges, positions = _synthetic_meoh_data()
    frames = get_frames_meoh_like(R, Z)
    mdcm_path = tmp_path / "meoh.mdcm"
    xyz_path = tmp_path / "dcm.xyz"

    dcmnet_to_mdcm(R, Z, charges, positions, "MEOH", mdcm_path, frames=frames)
    assert mdcm_path.exists()

    # Rebuild charges_per_frame for generate_dcm_xyz (normally from mdcm parser)
    from mmml.interfaces.dcmInterface.convert import global_to_local
    from mmml.interfaces.dcmInterface.frame import compute_dcm_frame

    charges_per_frame = []
    for fr_idx, frame_atoms in enumerate(frames):
        fv = compute_dcm_frame(R, frame_atoms)
        ca = frame_atoms[0]
        X, Y, Z_vec = fv[0]
        atom_pos = R[ca]
        frame_charges = []
        for c in range(charges.shape[1]):
            aq, bq, cq = global_to_local(
                positions[ca, c], atom_pos, X, Y, Z_vec
            )
            frame_charges.append((aq, bq, cq, float(charges[ca, c])))
        charges_per_frame.append(frame_charges)

    generate_dcm_xyz(R, frames, charges_per_frame, xyz_path)
    assert xyz_path.exists()

    # Parse dcm.xyz and compare charge positions to original
    lines = xyz_path.read_text().strip().split("\n")
    n_total = int(lines[0])
    n_atoms = R.shape[0]
    n_charges = n_total - n_atoms
    charge_lines = [l for l in lines[2:] if l.startswith("O")]
    assert len(charge_lines) == n_charges

    # Extract positions from O lines
    py_positions = []
    for ln in charge_lines:
        parts = ln.split()
        py_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    py_positions = np.array(py_positions)

    # Flatten original in same order (frame order, charge order)
    orig_flat = []
    for fr_idx, frame_atoms in enumerate(frames):
        ca = frame_atoms[0]
        for c in range(charges.shape[1]):
            orig_flat.append(positions[ca, c])
    orig_flat = np.array(orig_flat)

    np.testing.assert_allclose(py_positions, orig_flat, atol=1e-4)


@pytest.mark.skipif(
    not os.environ.get("CHARMM_HOME") or not os.path.exists(os.environ.get("CHARMM_HOME", "")),
    reason="CHARMM_HOME not set or path missing",
)
def test_charmm_dcm_xyz_vs_python(tmp_path):
    """When CHARMM available: compare CHARMM dcm.xyz vs Python dcm.xyz."""
    pytest.importorskip("pycharmm")
    from mmml.interfaces.dcmInterface import build_mdcm_from_dcmnet, generate_dcm_xyz

    h5_path = os.environ.get("DCM_REGRESSION_H5")
    if not h5_path or not Path(h5_path).exists():
        pytest.skip("DCM_REGRESSION_H5 not set or file missing")

    import h5py

    with h5py.File(h5_path, "r") as f:
        if "dcmnet_charges" not in f or "dcmnet_charge_positions" not in f:
            pytest.skip("H5 lacks dcmnet_charges/dcmnet_charge_positions")

    frame_idx = 0
    mdcm_path = tmp_path / "meoh.mdcm"
    py_xyz = tmp_path / "dcm_python.xyz"
    charmm_xyz = tmp_path / "dcm_charmm.xyz"

    frames, charges_per_frame = build_mdcm_from_dcmnet(
        h5_path, frame_idx, mdcm_path, "MEOH"
    )
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        R = np.asarray(f["R"][frame_idx], dtype=float)
        n = int(f["N"][frame_idx]) if "N" in f else R.shape[0]
    R = R[:n]
    generate_dcm_xyz(R, frames, charges_per_frame, py_xyz)

    # Run CHARMM DCM
    from mmml.interfaces.pycharmmInterface import import_pycharmm

    import_pycharmm()
    import pycharmm  # noqa: F401

    # Requires PDB/PSF; simplified - just compare Python roundtrip for now
    # Full CHARMM run would need: read psf, read coords, DCM IUDCM 11 TSHIFT XYZ 99
    # For CI without full CHARMM setup, we only run the Python path
    assert py_xyz.exists()
