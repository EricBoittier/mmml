from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mmml.models.physnetjax.physnetjax.models.zbl import ZBLRepulsion
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
    assert parser.parse_args([]).trainable_zbl is False
    assert parser.parse_args(["--trainable-zbl"]).trainable_zbl is True
