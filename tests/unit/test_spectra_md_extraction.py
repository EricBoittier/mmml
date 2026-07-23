"""Unit tests for mmml.spectra.spectra_md's ASE-trajectory/HDF5 extraction
and ML-calculator-dependent batched inference paths.

These don't need a real trained model or PyCHARMM/GPU: the batched-inference
tests use a tiny hand-written linear-response stand-in for `model.apply`
(same call signature as a real `MessagePassingModel`) to check the batching,
chunking/padding, and jacrev-based polarizability plumbing in isolation from
any actual physics, which lives in the real models and is covered elsewhere.
"""

from __future__ import annotations

import ase
import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.spectra.spectra_md import (
    compute_polarizability_batched,
    extract_dipoles_batched,
    extract_properties,
    extract_properties_hdf5,
    load_hdf5_trajectory,
)
import mmml.spectra.spectra_md as spectra_md


# ---------------------------------------------------------------------------
# extract_properties (ASE trajectory frames)
# ---------------------------------------------------------------------------


def _make_frames(n_frames=4, n_atoms=3, with_velocities=True):
    frames = []
    rng = np.random.default_rng(0)
    for t in range(n_frames):
        atoms = ase.Atoms(
            numbers=[8, 1, 1][:n_atoms],
            positions=rng.normal(size=(n_atoms, 3)),
        )
        if with_velocities:
            atoms.set_velocities(rng.normal(size=(n_atoms, 3)) * 0.01)
        frames.append(atoms)
    return frames


def test_extract_properties_reads_stored_ml_dipole_and_charges_without_calculator():
    frames = _make_frames(n_frames=3, n_atoms=3)
    for t, atoms in enumerate(frames):
        atoms.info["ml_dipole"] = np.array([1.0, 2.0, 3.0]) * (t + 1)
        atoms.arrays["ml_charges"] = np.array([-0.8, 0.4, 0.4])

    positions, velocities, dipoles, charges = extract_properties(frames, calc=None)

    assert positions.shape == (3, 3, 3)
    assert velocities.shape == (3, 3, 3)
    np.testing.assert_allclose(dipoles[1], [2.0, 4.0, 6.0])
    for t in range(3):
        np.testing.assert_allclose(charges[t], [-0.8, 0.4, 0.4])


def test_extract_properties_charges_fall_back_to_atomic_numbers_when_absent():
    frames = _make_frames(n_frames=2, n_atoms=3)
    # no ml_dipole / ml_charges stored, and no calculator available
    _, _, dipoles, charges = extract_properties(frames, calc=None)

    # dipoles stay zero-initialised (no calc to fill them, only a warning)
    np.testing.assert_allclose(dipoles, 0.0)
    # charges default to nuclear atomic numbers per the documented fallback
    for t in range(2):
        np.testing.assert_allclose(charges[t], [8.0, 1.0, 1.0])


def test_extract_properties_prints_warning_when_ml_properties_missing(capsys):
    frames = _make_frames(n_frames=2, n_atoms=3)
    extract_properties(frames, calc=None)
    out = capsys.readouterr().out
    assert "no calculator" in out


def test_extract_properties_no_velocities_warns_and_zero_fills(capsys):
    frames = _make_frames(n_frames=2, n_atoms=3, with_velocities=False)
    for atoms in frames:
        atoms.info["ml_dipole"] = np.zeros(3)
        atoms.arrays["ml_charges"] = np.zeros(len(atoms))

    _, velocities, _, _ = extract_properties(frames, calc=None)

    np.testing.assert_allclose(velocities, 0.0)
    out = capsys.readouterr().out
    assert "no velocities" in out


# ---------------------------------------------------------------------------
# load_hdf5_trajectory
# ---------------------------------------------------------------------------


def _write_hdf5_trajectory(path, T=5, N=3, with_velocities=True, dt_ps=None, time_ps=None):
    rng = np.random.default_rng(1)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("positions", data=rng.normal(size=(T, N, 3)).astype(np.float32))
        if with_velocities:
            f.create_dataset("velocities", data=rng.normal(size=(T, N, 3)).astype(np.float32) * 0.01)
        if time_ps is not None:
            f.create_dataset("time_ps", data=np.asarray(time_ps, dtype=np.float64))
        if dt_ps is not None:
            f.attrs["dt_ps"] = dt_ps


def test_load_hdf5_trajectory_reads_positions_velocities_and_dt(tmp_path):
    path = tmp_path / "traj.h5"
    _write_hdf5_trajectory(path, T=5, N=3, with_velocities=True, dt_ps=0.001)

    positions, velocities, dt_fs, metadata = load_hdf5_trajectory(path)

    assert positions.shape == (5, 3, 3)
    assert velocities.shape == (5, 3, 3)
    assert dt_fs == pytest.approx(1.0)  # 0.001 ps -> 1 fs
    assert "attr_dt_ps" in metadata


def test_load_hdf5_trajectory_no_velocities_returns_none(tmp_path):
    path = tmp_path / "traj_novel.h5"
    _write_hdf5_trajectory(path, T=4, N=2, with_velocities=False)

    positions, velocities, dt_fs, metadata = load_hdf5_trajectory(path)

    assert positions.shape == (4, 2, 3)
    assert velocities is None


def test_load_hdf5_trajectory_infers_dt_from_time_ps_when_no_dt_attr(tmp_path):
    path = tmp_path / "traj_time.h5"
    time_ps = np.array([0.0, 0.002, 0.004, 0.006])
    _write_hdf5_trajectory(path, T=4, N=2, with_velocities=False, time_ps=time_ps)

    _, _, dt_fs, metadata = load_hdf5_trajectory(path)

    assert dt_fs == pytest.approx(2.0)  # 0.002 ps step -> 2 fs
    np.testing.assert_allclose(metadata["time_ps"], time_ps)


def test_load_hdf5_trajectory_divides_dt_by_steps_per_recording(tmp_path):
    path = tmp_path / "traj_spr.h5"
    _write_hdf5_trajectory(path, T=3, N=2, with_velocities=False, dt_ps=0.001)
    with h5py.File(str(path), "a") as f:
        f.attrs["steps_per_recording"] = 5

    _, _, dt_fs, _ = load_hdf5_trajectory(path)

    assert dt_fs == pytest.approx(0.2)  # 1 fs / 5


def test_load_hdf5_trajectory_raises_without_h5py(tmp_path, monkeypatch):
    monkeypatch.setattr(spectra_md, "_HAS_H5PY", False)
    with pytest.raises(ImportError, match="h5py"):
        load_hdf5_trajectory(tmp_path / "does_not_matter.h5")


# ---------------------------------------------------------------------------
# extract_properties_hdf5
# ---------------------------------------------------------------------------


def test_extract_properties_hdf5_no_calc_charges_are_atomic_numbers():
    T, N = 3, 4
    positions = np.random.default_rng(2).normal(size=(T, N, 3))
    velocities = np.zeros((T, N, 3))
    Z = np.array([8, 1, 1, 1])

    pos_out, vel_out, dipoles, charges = extract_properties_hdf5(
        positions, velocities, calc=None, atomic_numbers=Z
    )

    assert pos_out.shape == (T, N, 3)
    np.testing.assert_allclose(dipoles, 0.0)
    for t in range(T):
        np.testing.assert_allclose(charges[t], Z.astype(np.float32))


def test_extract_properties_hdf5_missing_velocities_are_zero_filled(capsys):
    T, N = 2, 3
    positions = np.zeros((T, N, 3))
    Z = np.array([6, 1, 1])

    _, velocities, _, _ = extract_properties_hdf5(positions, None, calc=None, atomic_numbers=Z)

    assert velocities.shape == (T, N, 3)
    np.testing.assert_allclose(velocities, 0.0)
    assert "no velocities" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Batched ML-calculator inference: extract_dipoles_batched /
# compute_polarizability_batched, exercised with a fake linear-response model
# so the test doesn't depend on any trained checkpoint.
# ---------------------------------------------------------------------------


class _FakeLinearDipoleModel:
    """Stand-in for a MessagePassingModel: dipole(Ef) = alpha_true @ Ef,
    constant across atoms/positions so jacrev(dipole)(Ef) == alpha_true
    exactly, independent of the (unused) geometry -- enough to validate the
    batching/chunking/padding plumbing without a real trained model."""

    def __init__(self, alpha_true: np.ndarray, n_atoms: int):
        self.alpha_true = jnp.asarray(alpha_true, dtype=jnp.float32)
        self.n_atoms = n_atoms
        self.fake_charge_value = 0.25

    def apply(
        self,
        params,
        Z_batch,
        pos_batch,
        Ef_batch,
        *,
        dst_idx_flat=None,
        src_idx_flat=None,
        batch_segments=None,
        batch_size=None,
        dst_idx=None,
        src_idx=None,
        mutable=None,
    ):
        B = pos_batch.shape[0]
        dipole = jnp.einsum("ij,bj->bi", self.alpha_true, Ef_batch)
        energy = jnp.zeros((B,), dtype=jnp.float32)
        if mutable:
            charges = jnp.full((B, self.n_atoms), self.fake_charge_value, dtype=jnp.float32)
            atomic_dipoles = jnp.zeros((B, self.n_atoms, 3), dtype=jnp.float32)
            state = {
                "intermediates": {
                    "atomic_charges": (charges,),
                    "atomic_dipoles": (atomic_dipoles,),
                }
            }
            return (energy, dipole), state
        return energy, dipole


def test_extract_dipoles_batched_shapes_and_linear_response():
    N = 3
    T = 5  # deliberately not a multiple of chunk_size to exercise padding
    alpha_true = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    model = _FakeLinearDipoleModel(alpha_true, n_atoms=N)
    positions = np.zeros((T, N, 3), dtype=np.float32)
    Z = np.array([8, 1, 1])
    Ef = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    dipoles, charges, atomic_dipoles = extract_dipoles_batched(
        positions, Z, Ef, model, params=None, chunk_size=2
    )

    assert dipoles.shape == (T, 3)
    assert charges.shape == (T, N)
    assert atomic_dipoles.shape == (T, N, 3)

    expected_dipole = alpha_true @ Ef
    for t in range(T):
        np.testing.assert_allclose(dipoles[t], expected_dipole, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(charges[t], np.full(N, 0.25), rtol=1e-5)


def test_compute_polarizability_batched_recovers_linear_alpha():
    N = 4
    T = 3
    alpha_true = np.array(
        [[1.5, 0.2, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 2.0]], dtype=np.float32
    )
    model = _FakeLinearDipoleModel(alpha_true, n_atoms=N)
    positions = np.zeros((T, N, 3), dtype=np.float32)
    Z = np.array([6, 1, 1, 1])
    Ef = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    alpha = compute_polarizability_batched(
        positions, Z, Ef, model, params=None, chunk_size=2, field_scale=1.0
    )

    assert alpha.shape == (T, 3, 3)
    for t in range(T):
        np.testing.assert_allclose(alpha[t], alpha_true, rtol=1e-4, atol=1e-5)


def test_compute_polarizability_batched_applies_field_scale():
    N = 2
    T = 1
    alpha_true = np.eye(3, dtype=np.float32)
    model = _FakeLinearDipoleModel(alpha_true, n_atoms=N)
    positions = np.zeros((T, N, 3), dtype=np.float32)
    Z = np.array([1, 1])
    Ef = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    alpha = compute_polarizability_batched(
        positions, Z, Ef, model, params=None, chunk_size=1, field_scale=0.5
    )

    # jac (== alpha_true here) is divided by field_scale to convert to
    # physical units, per the module's documented convention.
    np.testing.assert_allclose(alpha[0], alpha_true / 0.5, rtol=1e-4)
