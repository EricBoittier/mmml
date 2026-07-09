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
    build_steps,
    create_state,
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
    assert prediction["l1_dipole_tensor"].shape == (1, 3)
    assert prediction["l2_quadrupole_tensor"].shape == (1, 3, 3)
    assert prediction["l3_octupole_tensor"].shape == (1, 3, 3, 3)


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
