from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.models.zbl import (
    ZBLRepulsion,
    geometric_pair_distances,
)
from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.models.physnetjax.physnetjax.models.model_charge_spin import PhysNetChargeSpin
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from scripts.train_so3lr_spooky_extxyz import build_parser as build_spooky_parser


def _inputs():
    return dict(
        atomic_numbers=jnp.asarray([1, 1]),
        distances=jnp.asarray([1.0, 1.0]),
        switch_off=None,
        eshift=None,
        idx_i=jnp.asarray([0, 1]),
        idx_j=jnp.asarray([1, 0]),
        atom_mask=jnp.ones(2),
        batch_mask=jnp.ones(2),
        batch_segments=jnp.zeros(2, dtype=jnp.int32),
        batch_size=1,
    )


def test_zbl_is_fixed_by_default_and_trainable_only_by_opt_in():
    fixed = ZBLRepulsion(cutoff=6.0)
    trainable = ZBLRepulsion(cutoff=6.0, trainable=True)
    fixed_variables = fixed.init(jax.random.PRNGKey(0), **_inputs())
    trainable_variables = trainable.init(jax.random.PRNGKey(0), **_inputs())

    assert "params" not in fixed_variables
    assert set(trainable_variables["params"]) == {
        "a_coefficient",
        "a_exponent",
        "phi_coefficients",
        "phi_exponents",
    }
    fixed_energy = fixed.apply(fixed_variables, **_inputs())
    trainable_energy = trainable.apply(trainable_variables, **_inputs())
    np.testing.assert_allclose(fixed_energy, trainable_energy, rtol=1e-7)


def test_physnet_family_defaults_to_fixed_zbl():
    assert PhysNet().trainable_zbl is False
    assert SpookyPhysNet().trainable_zbl is False
    assert PhysNetChargeSpin().trainable_zbl is False


def test_spooky_training_requires_explicit_trainable_zbl_flag():
    parser = build_spooky_parser()
    defaults = parser.parse_args([])
    assert defaults.trainable_zbl is False
    assert defaults.zbl_cuton == 0.8
    assert defaults.zbl_cutoff == 1.5
    assert parser.parse_args(["--trainable-zbl"]).trainable_zbl is True
    fixed = parser.parse_args(["--fixed-zbl"])
    assert fixed.trainable_zbl is False
    assert fixed.force_fixed_zbl is True


def test_zbl_receives_geometric_angstrom_distances_not_coulomb_kernel():
    displacements = jnp.asarray([[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    distances = geometric_pair_distances(displacements, jnp.ones(2))
    np.testing.assert_allclose(distances, [2.0, 5.0])


def test_masked_zbl_distance_is_finite_and_inert():
    distances = geometric_pair_distances(
        jnp.zeros((1, 3)), jnp.zeros(1)
    )
    np.testing.assert_allclose(distances, [1.0])


def test_spooky_full_path_uses_geometric_distance_and_zbl_own_cutoff():
    model = SpookyPhysNet(
        features=4,
        max_degree=0,
        num_iterations=1,
        num_basis_functions=4,
        cutoff=6.0,
        max_atomic_number=2,
        max_padded_atoms=2,
        charges=False,
        zbl=True,
        trainable_zbl=False,
    )
    inputs = dict(
        atomic_numbers=jnp.asarray([1, 1]),
        charges=jnp.zeros((2, 1)),
        spins=jnp.ones((2, 1)),
        positions=jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        dst_idx=jnp.asarray([0, 1]),
        src_idx=jnp.asarray([1, 0]),
        batch_segments=jnp.zeros(2, dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.ones(2),
        atom_mask=jnp.ones(2),
        compute_forces=False,
    )
    variables = model.init(jax.random.PRNGKey(1), **inputs)
    output = model.apply(variables, **inputs)

    direct = ZBLRepulsion(cutoff=1.5, cuton=0.8)
    direct_inputs = _inputs()
    direct_inputs["distances"] = jnp.asarray([2.0, 2.0])
    direct_variables = direct.init(jax.random.PRNGKey(2), **direct_inputs)
    expected = direct.apply(direct_variables, **direct_inputs)
    np.testing.assert_allclose(output["repulsion"], expected, rtol=1e-6)


def test_default_model_zbl_is_exactly_zero_at_and_beyond_short_cutoff():
    module = ZBLRepulsion(cutoff=1.5, cuton=0.8)
    inputs = _inputs()
    inputs["distances"] = jnp.asarray([1.5, 2.0])
    variables = module.init(jax.random.PRNGKey(4), **inputs)
    energy = module.apply(variables, **inputs)
    np.testing.assert_allclose(energy, 0.0, atol=0.0)


def test_short_zbl_switch_has_continuous_zero_force_at_cutoff():
    module = ZBLRepulsion(cutoff=1.5, cuton=0.8)

    def pair_energy(r):
        inputs = _inputs()
        inputs["distances"] = jnp.asarray([r, r])
        variables = module.init(jax.random.PRNGKey(5), **inputs)
        return module.apply(variables, **inputs).sum()

    left_force = -jax.grad(pair_energy)(jnp.asarray(1.5 - 1.0e-6))
    cutoff_force = -jax.grad(pair_energy)(jnp.asarray(1.5))
    assert abs(float(left_force)) < 1.0e-8
    assert float(cutoff_force) == 0.0


@pytest.mark.parametrize(
    ("cuton", "cutoff"),
    [(-0.1, 1.5), (1.5, 1.5), (2.0, 1.5), (0.8, 0.0)],
)
def test_invalid_zbl_windows_fail_fast(cuton, cutoff):
    module = ZBLRepulsion(cutoff=cutoff, cuton=cuton)
    with pytest.raises(ValueError, match="ZBL"):
        module.init(jax.random.PRNGKey(6), **_inputs())


def test_spooky_fixed_zbl_force_matches_finite_difference():
    model = SpookyPhysNet(
        features=4,
        max_degree=0,
        num_iterations=1,
        num_basis_functions=4,
        cutoff=6.0,
        max_atomic_number=2,
        max_padded_atoms=2,
        charges=False,
        zbl=True,
        trainable_zbl=False,
    )
    inputs = dict(
        atomic_numbers=jnp.asarray([1, 1]),
        charges=jnp.zeros((2, 1)),
        spins=jnp.ones((2, 1)),
        positions=jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        dst_idx=jnp.asarray([0, 1]),
        src_idx=jnp.asarray([1, 0]),
        batch_segments=jnp.zeros(2, dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.ones(2),
        atom_mask=jnp.ones(2),
    )
    variables = model.init(
        jax.random.PRNGKey(3), **inputs, compute_forces=False
    )
    force = model.apply(variables, **inputs, compute_forces=True)["forces"][1, 0]
    h = 1.0e-3

    def energy_at(x):
        positions = inputs["positions"].at[1, 0].set(x)
        return model.apply(
            variables, **{**inputs, "positions": positions}, compute_forces=False
        )["energy"].sum()

    finite_difference_force = -(energy_at(2.0 + h) - energy_at(2.0 - h)) / (2 * h)
    np.testing.assert_allclose(force, finite_difference_force, rtol=3e-3, atol=3e-4)
