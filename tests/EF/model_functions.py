"""
jax functions to calculate different partial derivatives of the energy with respect to the positions and electric field
"""

import jax
import jax.numpy as jnp

def energy_and_forces(model_apply, params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
    """Compute energy, forces (negative gradient of energy w.r.t. positions), and dipole."""
    def energy_fn(pos):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=pos,
            Ef=Ef,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        return -jnp.sum(energy), (energy, dipole)
    (_, (energy, dipole)), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return energy, forces, dipole


def energy_and_dipole_from_field_derivative(model_apply, params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
    """
    Compute energy and dipole via dE/dEf (Hellmann-Feynman theorem).
    
    The dipole moment is related to the energy derivative:
        μ = -dE/dEf
    
    This function computes dE/dEf using automatic differentiation.
    
    Parameters
    ----------
    model_apply : callable
        Model apply function
    params : dict
        Model parameters
    atomic_numbers : jnp.ndarray
        Atomic numbers, shape (B, N) or (B*N,)
    positions : jnp.ndarray
        Atomic positions, shape (B, N, 3) or (B*N, 3)
    Ef : jnp.ndarray
        Electric field, shape (B, 3)
    dst_idx_flat : jnp.ndarray
        Flattened destination indices
    src_idx_flat : jnp.ndarray
        Flattened source indices
    batch_segments : jnp.ndarray
        Batch segment indices
    batch_size : int
        Batch size
    dst_idx : jnp.ndarray, optional
        Original destination indices (for single-batch case)
    src_idx : jnp.ndarray, optional
        Original source indices (for single-batch case)
    
    Returns
    -------
    energy : jnp.ndarray
        Energy per molecule, shape (B,)
    dipole : jnp.ndarray
        Dipole moment per molecule, shape (B, 3)
        Note: This is -dE/dEf, so it represents the dipole moment
    """
    def energy_fn(ef):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=positions,
            Ef=ef,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        return jnp.sum(energy), (energy, dipole)
    
    # Compute dE/dEf
    (energy, (_, dipole_predicted)), dE_dEf = jax.value_and_grad(energy_fn, has_aux=True)(Ef)
    
    # Dipole moment is -dE/dEf (Hellmann-Feynman theorem)
    # Note: The sign depends on convention. In our case, E = E0 - μ·F, so μ = -dE/dF
    dipole = -dE_dEf
    
    return energy, dipole


def polarizability_from_field_derivative(model_apply, params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
    """
    Compute polarizability tensor via d²E/dEf² (second derivative of energy w.r.t. electric field).
    
    The polarizability tensor is:
        α_ij = -d²E/(dEf_i dEf_j)
    
    Parameters
    ----------
    model_apply : callable
        Model apply function
    params : dict
        Model parameters
    atomic_numbers : jnp.ndarray
        Atomic numbers, shape (B, N) or (B*N,)
    positions : jnp.ndarray
        Atomic positions, shape (B, N, 3) or (B*N, 3)
    Ef : jnp.ndarray
        Electric field, shape (B, 3)
    dst_idx_flat : jnp.ndarray
        Flattened destination indices
    src_idx_flat : jnp.ndarray
        Flattened source indices
    batch_segments : jnp.ndarray
        Batch segment indices
    batch_size : int
        Batch size
    dst_idx : jnp.ndarray, optional
        Original destination indices (for single-batch case)
    src_idx : jnp.ndarray, optional
        Original source indices (for single-batch case)
    
    Returns
    -------
    polarizability : jnp.ndarray
        Polarizability tensor per molecule, shape (B, 3, 3)
    """
    def energy_fn(ef):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=positions,
            Ef=ef,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        return jnp.sum(energy)
    
    # Compute Hessian: d²E/dEf²
    hessian = jax.hessian(energy_fn)(Ef)
    
    # Polarizability is -Hessian
    # α_ij = -d²E/(dEf_i dEf_j)
    polarizability = -hessian
    
    return polarizability
