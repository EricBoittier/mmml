"""Tiny EFieldPhysNet training on synthetic SPICE-α efield NPZ (no datasets)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mmml.data.spice_alpha import convert_spice_alpha_hdf5, split_npz
from mmml.models.efield.training import (
    EFieldPhysNet,
    eval_step,
    load_ef_npz,
    prepare_batches,
    train_model,
    train_step,
)
from spice_alpha_fixtures import write_water_spice_h5

BATCH_SIZE = 2


def _tiny_model() -> EFieldPhysNet:
    return EFieldPhysNet(
        features=4,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=4,
        cutoff=5.0,
        include_pseudotensors=False,
        dipole_field_coupling=False,
        field_scale=0.001,
        zbl=False,
        electrostatics_damping_sigma=0.0,
    )


def _spice_efield_splits(tmp_path, *, n_confs: int = 8) -> dict[str, object]:
    src = write_water_spice_h5(tmp_path / "water.hdf5", n_confs=n_confs)
    data = convert_spice_alpha_hdf5(
        [src],
        tmp_path / "all.npz",
        write_efield=True,
        polar_units="bohr3",
    )
    return split_npz(
        data,
        tmp_path / "splits",
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
        seed=0,
    )


def _init_params(model: EFieldPhysNet, train_data: dict, key):
    import e3x

    num_atoms = int(train_data["positions"].shape[1])
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
    batch_segments0 = jnp.repeat(jnp.arange(1, dtype=jnp.int32), num_atoms)
    offsets0 = jnp.arange(1, dtype=jnp.int32) * num_atoms
    return model.init(
        key,
        atomic_numbers=train_data["atomic_numbers"][0:1],
        positions=train_data["positions"][0:1],
        Ef=train_data["electric_field"][0:1],
        dst_idx_flat=(dst_idx[None, :] + offsets0[:, None]).reshape(-1),
        src_idx_flat=(src_idx[None, :] + offsets0[:, None]).reshape(-1),
        batch_segments=batch_segments0,
        batch_size=1,
        dst_idx=dst_idx,
        src_idx=src_idx,
    )


def test_spice_alpha_efield_npz_feeds_load_and_batches(tmp_path):
    written = _spice_efield_splits(tmp_path)
    train = load_ef_npz(written["train"])
    valid = load_ef_npz(written["valid"])
    assert train["positions"].shape[0] >= BATCH_SIZE
    assert valid["positions"].shape[0] >= BATCH_SIZE
    np.testing.assert_allclose(np.asarray(train["electric_field"]), 0.0)
    assert train["polar"].shape[-2:] == (3, 3)
    assert "D" in train
    batches = prepare_batches(
        jax.random.PRNGKey(0),
        train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        rot_augment=False,
    )
    assert batches
    batch = batches[0]
    assert batch["polar"].shape == (BATCH_SIZE, 3, 3)
    assert batch["electric_field"].shape == (BATCH_SIZE, 3)
    assert "dipoles" in batch


def test_spice_alpha_efield_train_step_polar_finite(tmp_path):
    written = _spice_efield_splits(tmp_path)
    train = load_ef_npz(written["train"])
    model = _tiny_model()
    params = _init_params(model, train, jax.random.PRNGKey(1))
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(1e-3))
    opt_state = optimizer.init(params)
    transform = optax.contrib.reduce_on_plateau(
        patience=5,
        cooldown=5,
        factor=0.9,
        rtol=1e-4,
        accumulation_size=5,
        min_scale=0.01,
    )
    transform_state = transform.init(params)
    batch = prepare_batches(
        jax.random.PRNGKey(2),
        train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        rot_augment=False,
    )[0]
    out = train_step(
        model_apply=model.apply,
        optimizer_update=optimizer.update,
        batch=batch,
        batch_size=BATCH_SIZE,
        opt_state=opt_state,
        params=params,
        ema_params=params,
        transform_state=transform_state,
        ema_decay=0.5,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=0.1,
        charge_weight=1.0,
        polar_weight=1.0,
        field_scale=0.001,
        polar_at_zero_field=True,
    )
    new_params, _ema, _opt, loss, *_rest, polar_loss, polar_mae = out
    assert np.isfinite(float(loss))
    assert np.isfinite(float(polar_loss))
    assert np.isfinite(float(polar_mae))
    assert float(polar_loss) > 0.0
    eval_out = eval_step(
        model_apply=model.apply,
        batch=batch,
        batch_size=BATCH_SIZE,
        params=new_params,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=0.1,
        charge_weight=1.0,
        polar_weight=1.0,
        field_scale=0.001,
        polar_at_zero_field=True,
    )
    eval_loss, *_eval_rest, eval_polar_loss, eval_polar_mae = eval_out
    assert np.isfinite(float(eval_loss))
    assert np.isfinite(float(eval_polar_loss))
    assert np.isfinite(float(eval_polar_mae))


def test_spice_alpha_efield_train_model_one_epoch(tmp_path, capsys):
    written = _spice_efield_splits(tmp_path)
    train = load_ef_npz(written["train"])
    valid = load_ef_npz(written["valid"])
    params = train_model(
        key=jax.random.PRNGKey(3),
        model=_tiny_model(),
        train_data=train,
        valid_data=valid,
        num_epochs=1,
        learning_rate=1e-3,
        batch_size=BATCH_SIZE,
        clip_norm=10.0,
        ema_decay=0.5,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=0.1,
        charge_weight=1.0,
        polar_weight=1.0,
        field_scale=0.001,
        polar_at_zero_field=True,
        verbose=False,
        save_best=False,
        save_every_n_epochs=0,
    )
    leaves = jax.tree_util.tree_leaves(params)
    assert leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    logged = capsys.readouterr().out
    assert "polar mae" in logged
    assert "polar MSE" in logged
