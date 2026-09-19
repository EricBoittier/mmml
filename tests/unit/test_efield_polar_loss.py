"""Zero-field polarizability loss (synthetic; no datasets)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.data.units import (
    ANGSTROM_TO_BOHR,
    E_ANGSTROM2_PER_VOLT_TO_BOHR3,
    polar_bohr3_to_e_angstrom2_per_volt,
    polar_e_angstrom2_per_volt_to_bohr3,
)
from mmml.models.efield.args import build_train_parser
from mmml.models.efield.model_functions import predicted_polarizability_bohr3
from mmml.models.efield.training import (
    load_ef_npz,
    polarizability_loss_and_mae,
    prepare_batches,
    require_drop_last_batches,
)
from mmml.utils.rotations import rotate_batched_rank2_tensors

pytestmark = pytest.mark.data_loading


def test_polar_unit_roundtrip():
    raw = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    au = polar_e_angstrom2_per_volt_to_bohr3(raw)
    assert au[0, 0] == pytest.approx(E_ANGSTROM2_PER_VOLT_TO_BOHR3)
    back = polar_bohr3_to_e_angstrom2_per_volt(au)
    np.testing.assert_allclose(back, raw)


def test_efield_parser_polar_weight_defaults_off():
    args = build_train_parser().parse_args([])
    assert args.polar_weight == 0.0
    assert args.polar_at_zero_field is True
    args = build_train_parser().parse_args(["--polar_weight", "1.5", "--no-polar-at-zero-field"])
    assert args.polar_weight == pytest.approx(1.5)
    assert args.polar_at_zero_field is False


def test_require_drop_last_batches_rejects_valid_smaller_than_batch():
    require_drop_last_batches(n_train=230, n_valid=13, batch_size=8)
    with pytest.raises(ValueError, match="BATCH_SIZE<=13"):
        require_drop_last_batches(n_train=230, n_valid=13, batch_size=64)


def _dummy_apply(params, atomic_numbers, positions, Ef, **_kwargs):
    energy = jnp.zeros((Ef.shape[0],))
    dipole = params["scale"] * Ef
    return energy, dipole


def test_predicted_polar_at_zero_field_is_scaled_identity():
    params = {"scale": jnp.float32(2.0)}
    field_scale = 0.001
    B, N = 2, 3
    pred = predicted_polarizability_bohr3(
        _dummy_apply,
        params,
        jnp.ones((B, N), dtype=jnp.int32),
        jnp.zeros((B, N, 3), dtype=jnp.float32),
        jnp.zeros((B * N * (N - 1),), dtype=jnp.int32),
        jnp.zeros((B * N * (N - 1),), dtype=jnp.int32),
        jnp.repeat(jnp.arange(B), N),
        B,
        field_scale=field_scale,
        ef_shared=jnp.zeros((3,), dtype=jnp.float32),
    )
    expected = (2.0 / field_scale) * ANGSTROM_TO_BOHR * np.eye(3)
    np.testing.assert_allclose(np.asarray(pred[0]), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(pred[1]), expected, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_polar_loss_gradients_flow(batch_size):
    """``dL/d(scale)`` through ``dμ/dEf`` must be finite and nonzero for B=1 and B>1."""
    params = {"scale": jnp.float32(2.0)}
    field_scale = 0.001
    n_atoms = 2
    target = jnp.eye(3, dtype=jnp.float32)
    batch = {
        "atomic_numbers": jnp.ones((batch_size * n_atoms,), dtype=jnp.int32),
        "positions": jnp.zeros((batch_size * n_atoms, 3), dtype=jnp.float32),
        "electric_field": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "dst_idx_flat": jnp.zeros((batch_size * n_atoms * (n_atoms - 1),), dtype=jnp.int32),
        "src_idx_flat": jnp.zeros((batch_size * n_atoms * (n_atoms - 1),), dtype=jnp.int32),
        "batch_segments": jnp.repeat(jnp.arange(batch_size), n_atoms),
        "polar": jnp.stack([target] * batch_size),
    }

    def loss(params_):
        mse, _mae = polarizability_loss_and_mae(
            _dummy_apply, params_, batch, batch_size, field_scale=field_scale, at_zero_field=True
        )
        return mse

    grads = jax.grad(loss)(params)
    scale_grad = float(grads["scale"])
    assert np.isfinite(scale_grad)
    assert abs(scale_grad) > 0.0


def test_polarizability_loss_zero_when_target_matches():
    params = {"scale": jnp.float32(2.0)}
    field_scale = 0.001
    B, N = 2, 2
    alpha = (2.0 / field_scale) * ANGSTROM_TO_BOHR * np.eye(3, dtype=np.float32)
    batch = {
        "atomic_numbers": jnp.ones((B * N,), dtype=jnp.int32),
        "positions": jnp.zeros((B * N, 3), dtype=jnp.float32),
        "electric_field": jnp.zeros((B, 3), dtype=jnp.float32),
        "dst_idx_flat": jnp.zeros((B * N * (N - 1),), dtype=jnp.int32),
        "src_idx_flat": jnp.zeros((B * N * (N - 1),), dtype=jnp.int32),
        "batch_segments": jnp.repeat(jnp.arange(B), N),
        "polar": jnp.stack([alpha, alpha]),
    }
    mse, mae = polarizability_loss_and_mae(
        _dummy_apply, params, batch, B, field_scale=field_scale, at_zero_field=True
    )
    assert float(mse) == pytest.approx(0.0, abs=1e-3)
    assert float(mae) == pytest.approx(0.0, abs=1e-3)


def test_load_ef_npz_defaults_zero_field_and_keeps_polar(tmp_path):
    n, natoms = 3, 2
    polar = np.eye(3)[None].repeat(n, 0).astype(np.float32)
    path = tmp_path / "ef.npz"
    np.savez_compressed(
        path,
        R=np.zeros((n, natoms, 3), np.float32),
        Z=np.ones((n, natoms), np.int32),
        E=np.array([-1.0, -1.1, -1.2], np.float32),
        F=np.zeros((n, natoms, 3), np.float32),
        D=np.zeros((n, 3), np.float32),
        polar=polar,
    )
    loaded = load_ef_npz(path)
    np.testing.assert_allclose(np.asarray(loaded["electric_field"]), 0.0)
    assert loaded["polar"].shape == (n, 3, 3)


def test_prepare_batches_rotates_polar():
    key = jax.random.PRNGKey(0)
    polar = jnp.array(
        [
            [[2.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 0.5]],
            [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.8]],
        ],
        dtype=jnp.float32,
    )
    data = {
        "atomic_numbers": jnp.array([[1, 8], [6, 1]], dtype=jnp.int32),
        "positions": jnp.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "electric_field": jnp.zeros((2, 3), dtype=jnp.float32),
        "energies": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "forces": jnp.zeros((2, 2, 3), dtype=jnp.float32),
        "polar": polar,
    }
    plain = prepare_batches(key, data, batch_size=2, shuffle=False, rot_augment=False)[0]
    aug = prepare_batches(
        key, data, batch_size=2, shuffle=False, rot_augment=True, rot_perturbation=0.0
    )[0]
    assert plain["polar"].shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(plain["polar"]), np.asarray(aug["polar"]), atol=1e-5)
    rot = jnp.array(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    rotated = rotate_batched_rank2_tensors(polar, rot)
    assert rotated.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(rotated[0]), np.asarray(rotated[0]).T, atol=1e-5)
