"""Phase 2 PhysNet-family facade / mixin tests."""

from __future__ import annotations

import warnings

import numpy as np

from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.models.physnetjax.physnetjax.models.model_charge_spin import PhysNetChargeSpin
from mmml.models.physnetjax.physnetjax.models.physnet_family import (
    PhysNetFamilyConfig,
    PhysNetFamilyMixin,
    resolve_physnet_class,
)
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet


def test_resolve_physnet_class_flags():
    assert resolve_physnet_class(PhysNetFamilyConfig()) is PhysNet
    assert (
        resolve_physnet_class(PhysNetFamilyConfig(condition_on_charge_spin=True))
        is SpookyPhysNet
    )
    # Electrostatics implies charge prediction in the config helper
    cfg = PhysNetFamilyConfig(include_electrostatics=True, predict_charges=False)
    assert cfg.predict_charges is True


def test_physnet_and_spooky_use_family_mixin():
    assert issubclass(PhysNet, PhysNetFamilyMixin)
    assert issubclass(SpookyPhysNet, PhysNetFamilyMixin)
    assert issubclass(PhysNetChargeSpin, PhysNetFamilyMixin)
    # Module path used by helper_mlp spooky detection must stay stable
    assert "spooky_model" in SpookyPhysNet.__module__
    assert PhysNet.__module__.endswith(".model")


def test_charge_spin_emits_deprecation_on_return_attributes():
    model = PhysNetChargeSpin(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=4,
        cutoff=5.0,
        max_padded_atoms=3,
        zbl=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.return_attributes()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_mixin_dipole_matches_direct_kernel():
    from mmml.models.physnetjax.physnetjax.models.mpnn_kernels import (
        molecular_dipole_from_charges,
    )
    import jax.numpy as jnp

    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
    )
    z = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    q = jnp.asarray([0.0, 0.5, -0.5], dtype=jnp.float32)
    segments = jnp.zeros(3, dtype=jnp.int32)
    model = PhysNet(features=8, max_degree=1, num_iterations=1, max_padded_atoms=3, zbl=False)
    got = model._calculate_dipole(positions, z, q, segments, 1)
    ref = molecular_dipole_from_charges(positions, z, q, segments, 1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0.0, atol=0.0)
