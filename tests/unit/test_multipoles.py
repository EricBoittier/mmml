from __future__ import annotations

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.multipoles import E3xMultipoleModel, irrep_blocks_to_traceless
from scripts.cache_qcml_multipoles_orbax import preprocess_examples
from scripts.analyze_qcml_multipoles import error_metrics, generate_report
from scripts.train_qcml_multipoles import (
    TrainConfig,
    bucket_indices,
    build_steps,
    create_state,
    filter_indices_by_target_thresholds,
    target_rms_from_arrays,
    target_rms_vector,
    target_component_scale_from_arrays,
    eligible_indices,
    limit_cache,
    make_batch,
    multipole_loss,
)


def test_irrep_blocks_to_traceless_shapes_and_traces() -> None:
    converted = irrep_blocks_to_traceless(jnp.arange(16, dtype=jnp.float32))

    assert converted["l0_irrep"].shape == (1,)
    assert converted["l1_dipole_tensor"].shape == (3,)
    assert converted["l2_quadrupole_tensor"].shape == (3, 3)
    assert converted["l3_octupole_tensor"].shape == (3, 3, 3)
    np.testing.assert_allclose(
        jnp.trace(converted["l2_quadrupole_tensor"]),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.trace(converted["l3_octupole_tensor"], axis1=-2, axis2=-1),
        jnp.zeros(3),
        atol=1e-6,
    )


def test_irrep_blocks_reject_wrong_width() -> None:
    with pytest.raises(ValueError, match="Expected 16"):
        irrep_blocks_to_traceless(jnp.zeros(15))


def test_e3x_multipole_model_output_shapes() -> None:
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.2, 0.8, 0.0]]
    )
    atomic_numbers = jnp.array([8, 1, 1])
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(3)
    model = E3xMultipoleModel(
        features=8,
        num_iterations=1,
        num_basis_functions=4,
    )
    arguments = (
        positions,
        atomic_numbers,
        jnp.array([0.0]),
        jnp.array([1.0]),
        dst_idx,
        src_idx,
    )

    variables = model.init(jax.random.key(0), *arguments)
    prediction = model.apply(variables, *arguments)

    assert prediction["multipoles"].shape == (1, 16)


def test_qcml_pair_preprocessing_joins_and_pads() -> None:
    irreps = {
        degree: jnp.arange(2 * degree + 1, dtype=jnp.float32)
        for degree in range(1, 4)
    }
    geometry = {
        "atomic_numbers": np.array([8, 1], dtype=np.uint8),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "charge": np.array(-1, dtype=np.int64),
        "multiplicity": np.array(2, dtype=np.int64),
        "key_hash": np.array(b"same-key"),
    }
    moments = {
        "key_hash": np.array(b"same-key"),
        "pbe0_dipole": np.asarray(irreps[1]),
        "pbe0_quadrupole": np.asarray(irreps[2]),
        "pbe0_octupole": np.asarray(irreps[3]),
    }

    cache = preprocess_examples([(geometry, moments)])

    assert cache["R"].shape == (1, 2, 3)
    assert cache["atom_mask"].shape == (1, 2)
    assert cache["multipoles"].shape == (1, 16)
    assert cache["multipoles"][0, 0] == -1
    np.testing.assert_allclose(cache["multipoles"][0, 1:4], irreps[1], atol=1e-6)


def test_training_batch_and_step_are_padding_safe() -> None:
    cache = {
        "R": np.array(
            [
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.2, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "Z": np.array([[8, 1, 0], [6, 1, 1]], dtype=np.int32),
        "Q": np.array([0.0, 0.0], dtype=np.float32),
        "S": np.array([1.0, 1.0], dtype=np.float32),
        "atom_mask": np.array([[1, 1, 0], [1, 1, 1]], dtype=np.float32),
        "multipoles": np.ones((2, 16), dtype=np.float32),
    }
    batch = make_batch(cache, np.array([0, 1]), np.ones(2, dtype=np.float32))
    config = TrainConfig(features=4, num_iterations=1, num_basis_functions=3)
    model = E3xMultipoleModel(**vars(config))
    state = create_state(model, batch, seed=0, learning_rate=1e-3, weight_decay=0.0)
    train_step, _ = build_steps(model, batch_size=2)

    updated_state, loss, degree_losses = train_step(state, batch)

    assert int(updated_state.step) == 1
    assert np.isfinite(loss)
    assert set(degree_losses) == {"l0", "l1", "l2", "l3"}
    assert batch["edge_mask"].sum() == 8


def test_multipole_loss_balances_degrees() -> None:
    target = jnp.zeros((1, 16))
    prediction = target.at[:, 0].set(2.0).at[:, 9:16].set(2.0)
    loss, degree_losses = multipole_loss(prediction, target, jnp.ones(1))

    assert degree_losses["l0"] == pytest.approx(4.0)
    assert degree_losses["l3"] == pytest.approx(4.0)
    assert loss == pytest.approx(2.0)


def test_multipole_loss_uses_rms_and_charge_constraint() -> None:
    target = jnp.zeros((1, 16))
    prediction = target.at[:, 0].set(2.0).at[:, 9:16].set(4.0)
    target_rms = jnp.ones(16).at[9:16].set(2.0)

    loss, losses = multipole_loss(
        prediction,
        target,
        jnp.ones(1),
        charge=jnp.array([1.0]),
        target_rms=target_rms,
        charge_weight=0.5,
    )

    assert losses["l0"] == pytest.approx(4.0)
    assert losses["l3"] == pytest.approx(4.0)
    assert losses["charge"] == pytest.approx(1.0)
    assert loss == pytest.approx(2.5)


def test_multipole_loss_can_use_huber_on_normalized_errors() -> None:
    target = jnp.zeros((1, 16))
    prediction = target.at[:, 0].set(4.0)
    target_rms = jnp.ones(16).at[0].set(2.0)

    loss, losses = multipole_loss(
        prediction,
        target,
        jnp.ones(1),
        target_rms=target_rms,
        huber_delta=1.0,
    )

    assert losses["l0"] == pytest.approx(1.5)
    assert loss == pytest.approx(0.375)


def test_target_rms_helpers_expand_degree_blocks() -> None:
    targets = np.zeros((2, 16), dtype=np.float32)
    targets[:, 1:4] = 2.0
    target_rms = target_rms_from_arrays(targets)
    vector = target_rms_vector(target_rms)

    assert target_rms["l0"] == pytest.approx(1e-6)
    assert target_rms["l1"] == pytest.approx(2.0)
    assert vector.shape == (16,)
    np.testing.assert_allclose(vector[1:4], 2.0)


def test_target_quantile_scale_and_outlier_filtering() -> None:
    targets = np.zeros((3, 16), dtype=np.float32)
    targets[:, 1] = [1.0, 2.0, 100.0]
    scales = target_component_scale_from_arrays(targets, quantile=0.5)
    cache = {
        "multipoles": targets,
    }

    filtered = filter_indices_by_target_thresholds(
        cache,
        np.array([0, 1, 2]),
        {"l0": 1.0, "l1": 10.0, "l2": 1.0, "l3": 1.0},
    )

    assert scales["l1"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_equal(filtered, np.array([0, 1]))


def test_training_cache_limit_preserves_aligned_fields() -> None:
    cache = {
        "R": np.zeros((5, 2, 3)),
        "Z": np.zeros((5, 2)),
        "metadata": np.array(17),
    }

    limited = limit_cache(cache, 3)

    assert limited["R"].shape[0] == 3
    assert limited["Z"].shape[0] == 3
    assert limited["metadata"] == 17


def test_multipole_atom_buckets_crop_batch_shapes() -> None:
    cache = {
        "R": np.zeros((3, 9, 3), dtype=np.float32),
        "Z": np.zeros((3, 9), dtype=np.int32),
        "Q": np.zeros(3),
        "S": np.ones(3),
        "atom_mask": np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.float32,
        ),
        "multipoles": np.zeros((3, 16), dtype=np.float32),
    }
    indices = eligible_indices(cache, max_atoms=5)
    buckets = bucket_indices(cache, indices, bucket_width=4)
    batch = make_batch(
        cache,
        np.array([0]),
        np.ones(1, dtype=np.float32),
        max_atoms=4,
    )

    np.testing.assert_array_equal(indices, np.array([0, 1]))
    assert set(buckets) == {4, 8}
    assert batch["positions"].shape == (4, 3)
    assert batch["dst_idx"].shape == (12,)


def test_analysis_metrics_and_report_outputs(tmp_path) -> None:
    target = np.zeros((2, 16), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 1:4] = 1.0
    num_atoms = np.array([2.0, 4.0])
    scales = {degree: 1.0 for degree in range(4)}
    units = {degree: "native" for degree in range(4)}

    metrics = error_metrics(target, prediction, num_atoms, scales, units)
    report = generate_report(
        tmp_path,
        np.array([3, 7]),
        target,
        prediction,
        num_atoms,
        scales,
        units,
    )

    assert metrics["l1"]["spherical_traceless"]["component_mae"] == pytest.approx(0.5)
    assert metrics["l1"]["spherical_traceless"]["tensor_norm_mae"] == pytest.approx(
        np.sqrt(3) / 2
    )
    assert report == metrics
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_structure_errors.csv").exists()
    assert (tmp_path / "scatter_spherical_l3.png").exists()
    assert (tmp_path / "scatter_cartesian_l3.png").exists()
    assert (tmp_path / "error_distributions.png").exists()
    assert (tmp_path / "error_vs_num_atoms.png").exists()
