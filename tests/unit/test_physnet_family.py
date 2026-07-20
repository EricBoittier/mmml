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


def test_return_attributes_covers_every_dataclass_field():
    """Every checkpoint-persisted field must round-trip through return_attributes().

    A field with no entry here silently drops out of saved checkpoints and
    falls back to the class default on reload -- exactly the failure mode
    that produced the hardcoded electrostatics-switch bug (see "No magic
    numbers" in docs/scientific-code.md). This test is self-maintaining: it
    catches any future field added to one of these dataclasses without a
    matching return_attributes() entry, without needing to name the field.
    """
    for cls, kwargs in (
        (PhysNet, dict(zbl=False)),
        (SpookyPhysNet, dict(zbl=False)),
        (PhysNetChargeSpin, dict(zbl=False)),
    ):
        model = cls(
            features=8,
            max_degree=1,
            num_iterations=1,
            num_basis_functions=4,
            cutoff=5.0,
            max_padded_atoms=3,
            **kwargs,
        )
        # "name"/"parent" are Flax-injected nn.Module bookkeeping fields, not
        # model hyperparameters, and are never checkpoint-persisted.
        declared_fields = set(model.__dataclass_fields__) - {"name", "parent"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            attrs = model.return_attributes()
        missing = declared_fields - set(attrs)
        assert not missing, (
            f"{cls.__name__} declares fields not surfaced by return_attributes(): "
            f"{sorted(missing)}"
        )


def test_efa_and_vdw_hyperparameters_have_backward_compatible_defaults():
    """Defaults must reproduce the values every pre-existing checkpoint was
    implicitly hardcoded to; changing any of these defaults changes model
    behavior for checkpoints trained before the fields existed."""
    for cls, kwargs in (
        (PhysNet, dict(zbl=False)),
        (SpookyPhysNet, dict(zbl=False)),
        (PhysNetChargeSpin, dict(zbl=False)),
    ):
        model = cls(max_padded_atoms=3, **kwargs)
        assert model.efa_lebedev_num == 194
        assert model.efa_max_length == 20.0
        assert model.efa_ti_degree_scaling_base == 0.5

    spooky = SpookyPhysNet(max_padded_atoms=3, zbl=False)
    assert spooky.vdw_soft_core_fraction == 0.8
    assert spooky.vdw_scale_center == 1.0
    assert spooky.vdw_scale_range == 0.5
    assert spooky.cgenff_fallback_sigma == 3.5
    assert spooky.cgenff_fallback_epsilon == 0.05


def test_efa_hyperparameters_are_threaded_into_efa_module():
    """A non-default efa_lebedev_num/efa_max_length must actually reach the
    constructed EFA submodule, not just sit unused on the dataclass."""
    model = SpookyPhysNet(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=4,
        cutoff=5.0,
        max_padded_atoms=3,
        zbl=False,
        efa=True,
        efa_lebedev_num=50,
        efa_max_length=12.0,
    )
    bound = model.bind({})
    assert bound.efa_final.lebedev_num == 50
    assert bound.efa_final.epe_max_length == 12.0


def test_vdw_soft_core_fraction_gates_lj_clamp():
    """A smaller vdw_soft_core_fraction must relax the short-range LJ clamp,
    changing the CGenFF vdW energy for an atom pair placed inside the clamp
    radius (proving the field is load-bearing, not dead)."""
    import jax.numpy as jnp

    def cgenff_vdw_energy(soft_core_fraction: float) -> float:
        model = SpookyPhysNet(
            features=8,
            max_degree=1,
            num_iterations=1,
            num_basis_functions=4,
            cutoff=5.0,
            max_padded_atoms=2,
            zbl=False,
            learn_cgenff_vdw_scale=False,
            vdw_soft_core_fraction=soft_core_fraction,
        )
        displacements = jnp.asarray([[[0.5, 0.0, 0.0]], [[-0.5, 0.0, 0.0]]], dtype=jnp.float32)
        off_dist = jnp.ones((2, 1, 1, 1), dtype=jnp.float32)
        cgenff_type_idx = jnp.asarray([0, 0], dtype=jnp.int32)
        sigmas = jnp.asarray([3.5], dtype=jnp.float32)
        epsilons = jnp.asarray([0.1], dtype=jnp.float32)
        atomic_numbers = jnp.asarray([6, 6], dtype=jnp.int32)
        dst_idx = jnp.asarray([0, 1], dtype=jnp.int32)
        src_idx = jnp.asarray([1, 0], dtype=jnp.int32)
        batch_mask = jnp.ones((2, 1, 1, 1), dtype=jnp.float32)
        batch_segments = jnp.asarray([0, 0], dtype=jnp.int32)

        def call(m):
            return m._calculate_cgenff_vdw(
                displacements,
                off_dist,
                cgenff_type_idx,
                sigmas,
                epsilons,
                atomic_numbers,
                None,
                dst_idx,
                src_idx,
                None,
                batch_mask,
                batch_segments,
                1,
            )

        _, batch_vdw = model.apply({}, method=call)
        return float(jnp.sum(jnp.asarray(batch_vdw)))

    loose = cgenff_vdw_energy(0.8)
    tight = cgenff_vdw_energy(0.1)
    assert loose != tight


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
