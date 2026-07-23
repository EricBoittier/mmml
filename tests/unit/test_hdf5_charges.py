"""HDF5 trajectory recording of per-atom MM Coulomb charges."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.calculator_utils import ModelOutput
from mmml.utils.hdf5_reporter import load_hdf5_trajectory, make_jaxmd_reporter


def test_model_output_mm_charges_default_and_field():
    n = 4
    forces = np.zeros((n, 3), dtype=np.float64)
    q = np.array([0.1, -0.05, -0.05, 0.0], dtype=np.float64)
    out = ModelOutput(
        energy=1.0,
        forces=forces,
        dH=0.0,
        internal_E=0.0,
        internal_F=forces,
        mm_E=0.0,
        mm_F=forces,
        ml_2b_E=0.0,
        ml_2b_F=forces,
        hybrid_energy=1.0,
        flat_bottom_E=0.0,
        com_restraint_E=0.0,
        com=np.zeros(3),
        com_dist=0.0,
        com_restraint_min_dist=0.0,
        mm_charges=q,
    )
    assert np.allclose(out.mm_charges, q)

    bare = ModelOutput(
        energy=0.0,
        forces=forces,
        dH=0.0,
        internal_E=0.0,
        internal_F=forces,
        mm_E=0.0,
        mm_F=forces,
        ml_2b_E=0.0,
        ml_2b_F=forces,
        hybrid_energy=0.0,
        flat_bottom_E=0.0,
        com_restraint_E=0.0,
        com=np.zeros(3),
        com_dist=0.0,
        com_restraint_min_dist=0.0,
    )
    assert bare.mm_charges == 0.0


def test_make_jaxmd_reporter_accepts_zero_buffer_size(tmp_path: Path):
    """Short smokes can compute buffer_size=min(100, total_records)=0."""
    path = tmp_path / "traj_zero_buf.h5"
    reporter = make_jaxmd_reporter(
        path,
        n_atoms=2,
        buffer_size=0,
        include_positions=True,
        include_velocities=False,
        include_charges=False,
    )
    assert reporter._buffer_size == 1
    reporter.report(
        potential_energy=-1.0,
        kinetic_energy=0.1,
        temperature=300.0,
        invariant=-0.9,
        positions=np.zeros((2, 3), dtype=np.float32),
    )
    reporter.close()
    assert path.is_file()


def test_make_jaxmd_reporter_writes_charges(tmp_path: Path):
    path = tmp_path / "traj.h5"
    n_atoms = 3
    q0 = np.array([0.4, -0.2, -0.2], dtype=np.float32)
    q1 = np.array([0.5, -0.25, -0.25], dtype=np.float32)
    reporter = make_jaxmd_reporter(
        path,
        n_atoms=n_atoms,
        buffer_size=2,
        include_positions=True,
        include_velocities=False,
        include_charges=True,
        attrs={"mm_charge_mode": "q0", "charges_units": "e"},
    )
    for i, q in enumerate((q0, q1)):
        reporter.report(
            potential_energy=-1.0 * i,
            kinetic_energy=0.1,
            temperature=300.0,
            invariant=-0.9,
            positions=np.zeros((n_atoms, 3), dtype=np.float32),
            charges=q,
        )
    reporter.close()

    data = load_hdf5_trajectory(path, datasets=["charges", "potential_energy"])
    assert data["charges"].shape == (2, n_atoms)
    np.testing.assert_allclose(data["charges"][0], q0, rtol=1e-5)
    np.testing.assert_allclose(data["charges"][1], q1, rtol=1e-5)

    import h5py

    with h5py.File(path, "r") as f:
        assert f.attrs["mm_charge_mode"] == "q0"
        assert f.attrs["charges_units"] == "e"
