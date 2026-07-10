import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space
from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.interfaces.jaxmdInterface.hybrid_energy import make_peptide_water_ml_energy_fn


class SizeCheckingModel:
    def apply(
        self,
        params,
        *,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        compute_forces=True,
    ):
        del params, compute_forces
        n_atoms = positions.shape[0]
        assert atomic_numbers.shape[0] == n_atoms
        assert dst_idx.shape == src_idx.shape
        assert dst_idx.shape[0] == n_atoms * (n_atoms - 1)
        return {"energy": jnp.sum(positions * positions)}


def test_peptide_water_ml_energy_fn_jit():
    """Verify that make_peptide_water_ml_energy_fn compiles under JIT and calculates finite energies and gradients."""
    # 1. Setup a minimal PhysNet model
    kwargs = {
        "features": 4,
        "max_degree": 1,
        "num_iterations": 1,
        "num_basis_functions": 8,
        "cutoff": 5.0,
        "max_atomic_number": 10,
        "charges": True,
        "max_padded_atoms": 10,
        "include_electrostatics": True,
    }
    model = PhysNet(**kwargs)
    
    key = jax.random.PRNGKey(0)
    
    # Initialize model parameters using a dummy dimer of size 8 (5 peptide + 3 water)
    dimer_size = 8
    Z_init = jnp.ones(dimer_size, dtype=jnp.int32)
    R_init = jnp.zeros((dimer_size, 3), dtype=jnp.float32)
    dst_init, src_init = np.where(~np.eye(dimer_size, dtype=bool))
    dst_init = jnp.array(dst_init, dtype=jnp.int32)
    src_init = jnp.array(src_init, dtype=jnp.int32)
    
    params = model.init(
        key,
        atomic_numbers=Z_init,
        positions=R_init,
        dst_idx=dst_init,
        src_idx=src_init,
    )
    
    # 2. Setup indices for peptide (5 atoms) and 2 water molecules (3 atoms each)
    peptide_idx = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    water_indices = [
        jnp.array([5, 6, 7], dtype=jnp.int32),
        jnp.array([8, 9, 10], dtype=jnp.int32),
    ]
    
    n_atoms = 11
    jax_z = jnp.array([6, 1, 1, 6, 8, 8, 1, 1, 8, 1, 1], dtype=jnp.int32)  # dummy Z
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.5, 0.5, 0.0],
        [3.5, -0.5, 0.0],
        [0.0, 3.0, 0.0],
        [0.5, 3.5, 0.0],
        [-0.5, 3.5, 0.0],
    ], dtype=jnp.float64)
    
    displacement_fn, shift_fn = space.periodic(10.0)
    
    # 3. Create the energy function
    energy_fn = make_peptide_water_ml_energy_fn(
        model, params, jax_z, peptide_idx, water_indices, displacement_fn
    )
    
    # 4. Check JIT compilation and evaluate energy
    jitted_energy = jax.jit(energy_fn)
    energy = jitted_energy(positions)
    
    assert jnp.isfinite(energy)
    
    # 5. Check gradient compilation and evaluate forces
    jitted_forces = jax.jit(jax.grad(energy_fn))
    forces = jitted_forces(positions)
    
    assert forces.shape == (n_atoms, 3)
    assert jnp.isfinite(forces).all()


def test_peptide_water_ml_energy_fn_uses_dynamic_dimer_size():
    peptide_idx = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    water_indices = [
        jnp.array([5, 6, 7], dtype=jnp.int32),
        jnp.array([8, 9, 10], dtype=jnp.int32),
    ]
    jax_z = jnp.array([6, 1, 1, 6, 8, 8, 1, 1, 8, 1, 1], dtype=jnp.int32)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.5, 0.0],
            [3.5, -0.5, 0.0],
            [0.0, 3.0, 0.0],
            [0.5, 3.5, 0.0],
            [-0.5, 3.5, 0.0],
        ],
        dtype=jnp.float64,
    )

    displacement_fn, _ = space.periodic(10.0)
    energy_fn = make_peptide_water_ml_energy_fn(
        SizeCheckingModel(),
        {},
        jax_z,
        peptide_idx,
        water_indices,
        displacement_fn,
    )

    energy = jax.jit(energy_fn)(positions)
    forces = jax.jit(jax.grad(energy_fn))(positions)

    assert jnp.isfinite(energy)
    assert forces.shape == positions.shape
    assert jnp.isfinite(forces).all()
