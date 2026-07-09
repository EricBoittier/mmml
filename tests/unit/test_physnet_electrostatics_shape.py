import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet

@pytest.mark.parametrize("model_cls", [PhysNet, SpookyPhysNet])
def test_physnet_electrostatics_damping_default_and_opt_out(model_cls):
    damped = model_cls()
    undamped = model_cls(electrostatics_damping_sigma=0.0)

    assert damped.return_attributes()["electrostatics_damping_sigma"] == 4.0
    assert undamped.return_attributes()["electrostatics_damping_sigma"] == 0.0

    displacements = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    batch_mask = jnp.array([1.0], dtype=jnp.float32)
    r_damped, _, _ = damped._calc_switches(displacements, batch_mask)
    r_undamped, _, _ = undamped._calc_switches(displacements, batch_mask)

    assert float(r_damped[0]) < float(r_undamped[0])


@pytest.mark.parametrize("model_cls", [PhysNet, SpookyPhysNet])
def test_physnet_electrostatics_shape_mismatch(model_cls):
    """Verify that a model trained/initialized with a smaller max_padded_atoms
    can evaluate a larger molecule without shape/broadcasting errors in electrostatics.
    """
    kwargs = {
        "features": 8,
        "max_degree": 1,
        "num_iterations": 1,
        "num_basis_functions": 16,
        "cutoff": 5.0,
        "max_atomic_number": 10,
        "charges": True,
        "max_padded_atoms": 10,
    }
    if model_cls is PhysNet:
        kwargs["include_electrostatics"] = True
        
    model = model_cls(**kwargs)

    key = jax.random.PRNGKey(0)
    
    # Initialize parameters with size 10 (matching max_padded_atoms)
    Z_init = jnp.ones(10, dtype=jnp.int32)
    R_init = jnp.zeros((10, 3), dtype=jnp.float32)
    dst_init, src_init = jnp.array([0, 1]), jnp.array([1, 0])
    
    # Prepare SpookyPhysNet specific inputs
    init_args = {}
    apply_args = {}
    if model_cls is SpookyPhysNet:
        # SpookyPhysNet requires charges and spins arrays of shape (n_atoms,)
        init_args["charges"] = jnp.zeros(10)
        init_args["spins"] = jnp.zeros(10)
        
        apply_args["charges"] = jnp.zeros(12)
        apply_args["spins"] = jnp.zeros(12)
        
    params = model.init(
        key,
        atomic_numbers=Z_init,
        positions=R_init,
        dst_idx=dst_init,
        src_idx=src_init,
        **init_args
    )
    
    # Run evaluation with size 12 (larger than max_padded_atoms)
    Z_large = jnp.ones(12, dtype=jnp.int32)
    R_large = jnp.zeros((12, 3), dtype=jnp.float32)
    dst_large = jnp.array([0, 1, 10, 11])
    src_large = jnp.array([1, 0, 11, 10])
    
    # This should complete successfully and not raise ValueError
    res = model.apply(
        params,
        atomic_numbers=Z_large,
        positions=R_large,
        dst_idx=dst_large,
        src_idx=src_large,
        **apply_args
    )
    
    assert "energy" in res
    assert res["energy"] is not None
    
    # Extract the scalar energy (since it has shape (1, 1))
    energy_val = float(res["energy"].squeeze())
    assert np.isfinite(energy_val)
